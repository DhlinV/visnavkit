import subprocess
import sys
from pathlib import Path

from visnavkit.scripts.sanity_check import main

SMALL = [
    "model.feat_size=8",
    "model.vision_encoder.img_embed_size=16",
    "model.vision_encoder.neck_cfg.n_res_blocks=0",
    "model.temporal_encoder.ff_dim=16",
    "model.temporal_encoder.num_heads=2",
]


def test_sanity_check_cli_prints_real_pipeline_and_checks_training(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "visnavkit.scripts.sanity_check",
            "--output-dir",
            str(tmp_path),
            "model/vision_encoder=resnet18",
            "model/temporal_encoder=bidirectional",
            "model/goal_encoder=point",
            "model/action_decoder=regression",
            "common.step=2",
            *SMALL,
            "model.action_decoder.hidden=16",
        ],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )
    for text in (
        "NavigationPolicy (",
        "params)",
        "+-- vision_encoder: TimmCNNEncoder (resnet18, global,",
        "+-- temporal_encoder: BidirectionalTemporalEncoder (",
        "reduction=last -> [2, 1, 1, 8]",
        "+-- modalities: (none)",
        "+-- goal_encoder: ['PointGoalEncoder'] [[2, 3, 3]] -> tokens [2, 1, 8] (",
        "`-- action_decoder: RegressionDecoder (waypoint, 1 modes,",
        "trajectories -> [2, 1, 10, 3]",
        "[PASS] Training loss=",
        "[PASS] Feature-buffer parity (history=4, reduction=last)",
    ):
        assert text in result.stdout, result.stdout
    assert (tmp_path / "pipeline.txt").read_text() in result.stdout
    assert not (tmp_path / "model.onnx").exists()


def test_generative_sanity_check_with_onnx_export(tmp_path, capsys):
    main(
        [
            "--output-dir",
            str(tmp_path),
            "--onnx",
            "model/vision_encoder=resnet18",
            "model/action_decoder=flow_dit",
            "model/goal_encoder=instruction",
            "common.seq_length=2",
            *SMALL,
            "model.action_decoder.sample_steps=2",
            "model.action_decoder.denoiser.hidden=16",
            "model.action_decoder.denoiser.depth=1",
            "model.action_decoder.denoiser.num_heads=2",
        ]
    )
    output = capsys.readouterr().out
    assert "denoiser: DiTDenoiser x 2 steps (FlowMatchingScheduler)" in output
    assert "[PASS] Training loss=" in output
    assert "[PASS] Feature-buffer parity (history=1, reduction=last)" in output
    assert "[PASS] ONNX Runtime parity" in output
    assert (tmp_path / "model.onnx").exists()
