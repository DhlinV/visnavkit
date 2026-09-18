"""FlowPilot-STS: masked pair encoding, per-frame kv tokens, the anchored flow head and its two-mode output."""

import torch
from hydra import compose, initialize_config_module
from hydra.utils import instantiate

from visnavkit.models.lit_model import build_targets, disable_pretrained_downloads

SMALL = [
    "model=flowpilot_sts",
    "model.pretrained=false",
    "model.backbone_name=fastvit_t8",
    "model.dim=32",
    "model.temporal.num_layers=1",
    "model.temporal.num_heads=2",
    "model.head.num_layers=1",
    "model.head.num_heads=2",
    "model.head.num_anchors=4",
    "model.head.wp_dim=8",
    "plan_len_points=8",
    "plan_len_seconds=0.4",
    "common.seq_length=4",
    "common.uniform_t_anchors=true",
]
FRAME_MASK = torch.tensor([[True, True, True, True], [True, False, False, True]])


def make_model():
    with initialize_config_module(version_base=None, config_module="visnavkit.configs"):
        cfg = compose(config_name="train", overrides=SMALL)
    disable_pretrained_downloads(cfg.model)
    return instantiate(cfg.model)


def make_batch(b=2, t=4, hw=(32, 64)):
    """Straight 1 m/s motion; window 1 has no frames in its middle slots and no route."""
    times = torch.arange(1, 9) * 0.05
    future = torch.zeros(b, t, 8, 5)
    future[..., 0], future[..., 3] = times, 1.0
    bounds = torch.tensor([[-0.1, -0.1, -0.1, -0.1, -1.0], [0.3, 0.1, 0.1, 3.0, 1.0]])
    return dict(
        vision=torch.rand(b, t, 3, *hw),
        frame_mask=FRAME_MASK.clone(),
        route_patch=torch.randint(0, 3, (b, t, 80, 80)).float(),
        route_mask=torch.tensor([[True] * t, [False] * t]),
        goal=torch.tensor([10.0, 1.0, 0.0]).expand(b, t, 3).clone(),
        ego=torch.ones(b, t, 2),
        embodiment_id=torch.tensor([0, 3]),
        action_bounds=bounds.expand(b, 2, 5).clone(),
        future_poses=future,
        frame_speeds=torch.ones(b, t, 1),
        target_times_s=times.expand(b, 8).clone(),
    )


def inputs(batch):
    keys = ("frame_mask", "route_patch", "route_mask", "ego", "embodiment_id", "action_bounds")
    return batch["vision"], batch["goal"], {k: batch[k] for k in keys}


def test_training_supervises_the_frames_that_hold_a_frame_and_the_whole_pairs():
    model, batch = make_model().train(), make_batch()
    vision, goal, mods = inputs(batch)
    out = model(vision, goal, **mods)
    assert torch.equal(out.plan.idx, FRAME_MASK.reshape(-1).nonzero().squeeze(1))
    assert out.plan.tokens.shape == (6, 1 + 2 + 3 + 1, 32)  # global, 1 x 2 patches, route, goal, temporal, embodiment
    assert torch.all(out.speed[~FRAME_MASK.reshape(-1)] == 0)
    pair = out.pair_mask.reshape(2, 4)
    assert not pair[:, 0].any() and not pair[1, 3]  # no previous slot at all, or an empty one
    assert (out.plan.plans == 0).all()  # generative: no sampling while training

    losses, _ = model.get_losses(out, build_targets(batch))
    assert torch.isfinite(losses["loss"]) and {"action_reg", "action_cls", "vision_speed"} <= losses.keys()
    losses["loss"].backward()
    assert model.pair_encoder.backbone.stem_0.conv_kxk[0].conv.weight.grad is not None
    assert model.action_decoder.score[-1].weight.grad is not None
    assert not any(p.requires_grad for p in model.route_encoder.parameters())
    assert not model.route_encoder.training


def test_speed_head_is_supervised_on_whole_pairs_only():
    """An empty or dropped previous slot is black to the network and takes the slot's speed target with it."""
    model, batch = make_model().train(), make_batch()
    for part in (model.pair_encoder.backbone, model.pair_encoder.speed_head):
        part.eval()  # running BatchNorm statistics: the rows stay independent
    vision, goal, mods = inputs(batch)
    model.pair_encoder.p_drop_prev = 0.0
    out = model(vision, goal, **mods)
    pair = out.pair_mask.reshape(2, 4)
    assert pair.tolist() == [[False, True, True, True], [False] * 4]  # window 1: what a corpus below 20 fps gives
    speed = model.get_losses(out, build_targets(batch))[0]["vision_speed"]
    elsewhere = dict(batch, frame_speeds=torch.where(pair[..., None], batch["frame_speeds"], torch.tensor(50.0)))
    assert speed > 0 and model.get_losses(out, build_targets(elsewhere))[0]["vision_speed"] == speed

    other_past = vision.clone()
    other_past[0, 2] = torch.rand(3, 32, 64)  # slot 3's previous frame
    assert not torch.equal(model(other_past, goal, **mods).speed[3], out.speed[3])
    model.pair_encoder.p_drop_prev = 1.0  # every previous frame dropped: black, and no whole pair is left
    dropped = model(vision, goal, **mods)
    assert not dropped.pair_mask.any()
    torch.testing.assert_close(model(other_past, goal, **mods).speed[3], dropped.speed[3])

    sparse = {k: v[1:] for k, v in batch.items()}  # the window with gaps alone: plans to learn, no speed target
    losses, _ = model.get_losses(model(*inputs(sparse)[:2], **inputs(sparse)[2]), build_targets(sparse))
    assert losses["vision_speed"] == 0 and torch.isfinite(losses["loss"])
    losses["loss"].backward()
    grad = model.pair_encoder.speed_head.head[0].weight.grad
    assert grad is not None and not grad.any()  # still in the graph (DDP wants every parameter), without a signal


def test_eval_decodes_two_modes_and_leaves_rows_without_frames_empty():
    model, batch = make_model().eval(), make_batch()
    vision, goal, mods = inputs(batch)
    with torch.no_grad():
        out = model(vision, goal, **mods)
    assert out.plan.plans.shape == (8, 2 * (2 * 8 * 5 + 1))
    parsed = model.action_decoder.parse_output(out.plan.plans)
    assert parsed["plans"].shape == (8, 2, 8, 5) and parsed["confs"].shape == (8, 2)
    assert torch.all(out.plan.plans[~out.plan.valid] == 0) and torch.isfinite(parsed["plans"]).all()
    assert torch.equal(out.plan.valid, FRAME_MASK.reshape(-1))

    # the noise-0 mode is deterministic; black frames in the empty slots never reach the network
    garbage = dict(batch, vision=batch["vision"].clone())
    garbage["vision"][1, 1:3] = torch.rand(2, 3, 32, 64)
    with torch.no_grad():
        again = model(*inputs(garbage)[:2], **inputs(garbage)[2])
    torch.testing.assert_close(
        parsed["plans"][:, 0], model.action_decoder.parse_output(again.plan.plans)["plans"][:, 0]
    )
    with torch.no_grad():
        no_goal = model(vision, None, **mods)
    assert torch.isfinite(no_goal.plan.plans).all()


def test_example_batch_runs_the_smoke_path():
    model = make_model().eval()
    vision, goal, mods = model.example_batch(2, 4, (32, 64))
    with torch.no_grad():
        out = model(vision, goal, **mods)
    assert out.plan.plans.shape == (8, model.action_decoder.flat_size)
