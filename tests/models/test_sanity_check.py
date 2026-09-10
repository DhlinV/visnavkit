import subprocess
import sys
from pathlib import Path

from visnavkit.scripts.sanity_check import main


def test_sanity_check_cli_prints_real_pipeline_and_checks_training(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "visnavkit.scripts.sanity_check",
            "--output-dir",
            str(tmp_path),
            "vision_encoder=cnn_resnet",
            "temporal_encoder=bidirectional",
            "action_decoder=waypoint",
            "common.step=2",
            "model.feat_size=8",
            "model.modules.vision_encoder.img_embed_size=16",
            "model.modules.vision_encoder.neck_cfg.n_res_blocks=0",
            "model.modules.action_decoder.temporal_encoder.ff_dim=16",
            "+model.modules.vision_encoder.weights=/missing/initial-weights.pt",
        ],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    for text in (
        "E2EModel",
        "+-- vision_encoder: ResNetEncoder",
        "`-- action_decoder: ActionDecoder",
        "temporal_encoder: BidirectionalTemporalEncoder",
        "reduction=last -> [2, 8]",
        "plan_head: WaypointHead",
        "trajectories -> [2, 1, 10, 3]",
        "[PASS] Training loss=",
        "backward gradients finite",
        "[PASS] Feature-buffer parity (history=4, reduction=last)",
    ):
        assert text in result.stdout
    diagram = (tmp_path / "pipeline.txt").read_text()
    assert diagram in result.stdout and "RGB frame pairs [2, 3, 6, 64, 64]" in diagram
    assert not (tmp_path / "model.onnx").exists()


def test_diffusion_sanity_check_matches_buffered_sampling(tmp_path, capsys):
    main(
        [
            "--output-dir",
            str(tmp_path),
            "vision_encoder=cnn_resnet",
            "action_decoder=diffusion",
            "common.seq_length=1",
            "model.feat_size=8",
            "model.modules.vision_encoder.img_embed_size=16",
            "model.modules.vision_encoder.neck_cfg.n_res_blocks=0",
            "model.modules.action_decoder.temporal_encoder.ff_dim=16",
            "model.modules.action_decoder.plan_head.hidden=16",
            "model.modules.action_decoder.plan_head.time_embed_dim=8",
            "model.modules.action_decoder.plan_head.train_timesteps=5",
            "model.modules.action_decoder.plan_head.sample_steps=2",
        ]
    )
    output = capsys.readouterr().out
    assert "plan_head: DiffusionPlanHead" in output
    assert "[PASS] Training loss=" in output
    assert "[PASS] Feature-buffer parity (history=0, reduction=last)" in output
