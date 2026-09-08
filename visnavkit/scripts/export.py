"""Script to export the model to ONNX format."""

import copy
from pathlib import Path

import hydra
import numpy as np
import onnx
import onnxruntime as ort
import onnxslim
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from onnxruntime.transformers import float16
from torch.nn.utils.fusion import fuse_linear_bn_eval

from visnavkit.models.heads.diffusion_plan_head import DiffusionPlanHead
from visnavkit.models.lit_model import LitModel
from visnavkit.utils.logger import get_logger

logger = get_logger(__name__)


def enforce_output_order(model_onnx: onnx.ModelProto, output_names: list[str]) -> onnx.ModelProto:
    output_map = {out.name: out for out in model_onnx.graph.output}
    if not all(name in output_map for name in output_names):
        return model_onnx
    del model_onnx.graph.output[:]
    for name in output_names:
        model_onnx.graph.output.append(output_map[name])
    return model_onnx


def fuse_linear_bn_pairs(module: nn.Module) -> None:
    """Fold linear -> BN pairs"""
    children = list(module.named_children())
    for (name, child), (next_name, next_child) in zip(children, children[1:]):
        if isinstance(child, nn.Linear) and isinstance(next_child, nn.BatchNorm1d):
            setattr(module, name, fuse_linear_bn_eval(child, next_child))
            setattr(module, next_name, nn.Identity())

    for _, child in module.named_children():
        fuse_linear_bn_pairs(child)


def reparameterize_model(model: torch.nn.Module) -> torch.nn.Module:
    model = copy.deepcopy(model)
    for module in model.modules():
        if hasattr(module, "reparameterize"):
            module.reparameterize()
    fuse_linear_bn_pairs(model)
    return model


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    exponent = np.exp(x - np.max(x, axis=0))
    return exponent / np.sum(exponent, axis=0)


def parse_plan_output(output, M, num_pts, pose_width):
    output = output.reshape(1, M * (num_pts * 2 * pose_width + 1))
    pred_plans_reshaped = output.reshape(-1, M, 2 * pose_width * num_pts + 1)
    path_conf = softmax(pred_plans_reshaped[0, :, -1])
    paths = pred_plans_reshaped[:, :, :-1].reshape(-1, 2, num_pts, pose_width)[:, 0, :, :2]
    best_path = paths[path_conf.argmax()]
    return dict(
        pred_logits=pred_plans_reshaped[0, :, -1],
        pred_confs=path_conf,
        pred_plans=paths,
        best_plan=best_path,
    )


def print_sanity_check(model_path, plan_head, x, fb, *, reference=None, half=False):
    logger.info("Doing forward pass with model to get sanity check values")
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(model_path, sess_options=sess_options, providers=["CPUExecutionProvider"])
    output_map = {o.name: i for i, o in enumerate(sess.get_outputs())}
    feeds = {"input": x.cpu().numpy(), "feature_buffer": fb.cpu().numpy()}
    # A model without temporal mixing may allow the exporter to prune the buffer.
    output = sess.run(None, {node.name: feeds[node.name] for node in sess.get_inputs()})
    if reference is not None:
        for node, expected, observed in zip(sess.get_outputs(), reference, output):
            np.testing.assert_allclose(
                observed,
                expected.detach().cpu().numpy(),
                rtol=1e-2 if half else 2e-3,
                atol=2e-3 if half else 2e-4,
                err_msg=node.name,
            )
        logger.info("Nonzero-input PyTorch/ONNX numerical parity passed")
    parsed_output = parse_plan_output(
        output[output_map["plan"]],
        M=plan_head.num_modes,
        num_pts=plan_head.num_pts,
        pose_width=plan_head.pose_size,
    )

    print("=" * 40 + " SANITY CHECK " + "=" * 40)
    print(f"speed: {float(np.asarray(output[output_map['pose']]).reshape(-1)[0]):.4f}")
    print(f"logits: {np.round(parsed_output['pred_logits'], 2)}")
    print(f"best_plan p0: {np.round(parsed_output['best_plan'][0], 2)}")
    print(f"best_plan pN: {np.round(parsed_output['best_plan'][-1], 2)}")


def prepare_export_config(cfg):
    """Restore checkpoint architecture/preprocessing while retaining export options."""
    cfg = copy.deepcopy(cfg)
    if cfg.checkpoint:
        checkpoint = torch.load(cfg.checkpoint, map_location="cpu", weights_only=False)
        saved = checkpoint.get("hyper_parameters", {}).get("cfg")
        if saved is not None:
            saved = OmegaConf.create(saved)
            options = {key: cfg[key] for key in ("checkpoint", "output", "onnx_opset_version", "half", "export_heads")}
            cfg = OmegaConf.merge(OmegaConf.to_container(saved, resolve=True), options)
    return cfg


def _export_model(cfg):
    cfg = prepare_export_config(cfg)
    # None means we make a prediction for each token coming out of the temporal_encoder. This is a training only hack
    # For deployment, we just want a prediction for the last token
    temporal_encoder = cfg.model.modules.action_decoder.get("temporal_encoder")
    if temporal_encoder is not None and temporal_encoder.reduction == "none":
        temporal_encoder.reduction = "last"

    output_filepath = cfg.output
    Path(output_filepath).parent.mkdir(parents=True, exist_ok=True)
    if cfg.checkpoint is None:  # checkpoint=null: export with untrained weights (pipeline check)
        lmodel = LitModel(cfg)
    else:
        lmodel = LitModel.load_from_checkpoint(cfg.checkpoint, cfg=cfg)
    lmodel.eval()
    model = lmodel.model
    infer_model = reparameterize_model(model)
    # drop configured heads the model doesn't have
    export_heads = [name for name in cfg.export_heads if name in infer_model.vision_encoder.heads]
    infer_model.export_heads = export_heads

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    infer_model = infer_model.to(device)

    # pull out parameters from config
    input_channels = cfg.model.modules.vision_encoder.in_chans
    img_w = int(cfg.common.crop_wh[0] // cfg.common.downscale_factor)
    img_h = int(cfg.common.crop_wh[1] // cfg.common.downscale_factor)

    x = torch.rand(1, input_channels, img_h, img_w).to(device)
    export_cfg = cfg.model.export_cfg
    history_size = export_cfg.seq_step * export_cfg.seq_len - export_cfg.seq_step
    fb = torch.randn(1, history_size, cfg.model.feat_size).to(device) * 0.1
    logger.info(f"input shape: {x.shape} feature buffer shape: {fb.shape}")

    export_args = (x, fb)
    input_names = ["input", "feature_buffer"]

    with torch.no_grad():
        output = infer_model(*export_args)
    if not isinstance(output, (tuple, list)):
        raise TypeError(f"Expected export model output tuple/list, got {type(output)}")
    output_names = infer_model.get_export_output_names()
    if len(output) != len(output_names):
        raise ValueError(f"Expected {len(output_names)} outputs ({output_names}), got {len(output)}")
    if not all(torch.is_tensor(tensor) for tensor in output):
        raise TypeError(f"Export outputs must be tensors: {output_names}.")

    dynamic_axes = {name: {0: "batch_size"} for name in input_names}
    dynamic_axes.update({name: {0: "batch_size"} for name in output_names})

    logger.info("Exporting from torch model...")
    torch.onnx.export(
        infer_model,
        export_args,
        output_filepath,
        output_names=output_names,
        input_names=input_names,
        dynamic_axes=dynamic_axes,
        opset_version=cfg.onnx_opset_version,
        do_constant_folding=True,
        verbose=False,
        dynamo=False,
    )

    # slim
    logger.info("Slimming...")
    model_onnx = onnx.load(output_filepath)
    model_onnx = onnxslim.slim(model_onnx)

    # half
    if cfg.half:
        logger.info("Converting to fp16...")
        model_onnx = float16.convert_float_to_float16(model_onnx, keep_io_types=True)

    # Keep output order deterministic for index-based runtimes.
    model_onnx = enforce_output_order(model_onnx, output_names)

    logger.info(f"Saving onnx file to {output_filepath}")
    onnx.save(model_onnx, output_filepath)

    # load model and output values for comparison testing
    head = infer_model.action_decoder.plan_head
    print_sanity_check(
        output_filepath,
        head,
        x,
        fb,
        reference=None if isinstance(head, DiffusionPlanHead) else output,
        half=cfg.half,
    )
    if isinstance(head, DiffusionPlanHead):
        logger.info("Legacy diffusion export is stochastic; use benchmark export for explicit-noise numerical parity")


@hydra.main(version_base=None, config_path="../configs", config_name="export")
def main(cfg: DictConfig):
    fastpath = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    try:
        return _export_model(cfg)
    finally:
        torch.backends.mha.set_fastpath_enabled(fastpath)


if __name__ == "__main__":
    main()
