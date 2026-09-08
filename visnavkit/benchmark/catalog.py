"""Verified model provenance and pinned ONNX artifact downloads.

An available artifact is not a validated policy adapter. See ``docs/models.md``
for input-contract issues that must be resolved before publishing quality scores.
This module uses only the Python standard library and never imports remote code.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from urllib.parse import quote
from urllib.request import Request, urlopen

HF_REPOSITORY = "UCLA-VAIL/Navigation-Model-Zoo-Public"
HF_REVISION = "9c1f523ef8dddbccb6dee1d3d588112f754bc8f2"
VERIFIED_DATE = "2026-09-07"
MANIFEST_URL = f"https://huggingface.co/api/models/{HF_REPOSITORY}/tree/{HF_REVISION}?recursive=true&expand=false"
_CHUNK_BYTES = 1024 * 1024


@dataclass(frozen=True)
class Artifact:
    """A file in the pinned release; SHA-256 covers its complete contents."""

    path: str
    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        path = PurePosixPath(self.path)
        if path.is_absolute() or ".." in path.parts or "\\" in self.path or not path.name:
            raise ValueError(f"Artifact path must be a relative POSIX path: {self.path!r}")
        if self.size_bytes <= 0:
            raise ValueError("Artifact size must be positive")
        if len(self.sha256) != 64 or any(c not in "0123456789abcdef" for c in self.sha256):
            raise ValueError("Artifact SHA-256 must contain 64 lowercase hexadecimal characters")

    @property
    def url(self) -> str:
        return f"https://huggingface.co/{HF_REPOSITORY}/resolve/{HF_REVISION}/{quote(self.path, safe='/')}"


@dataclass(frozen=True)
class ModelSpec:
    """Research identity and release provenance, independent of the local recipes."""

    id: str
    name: str
    implementation_kind: str
    status: str
    modalities: tuple[str, ...]
    conditioning: tuple[str, ...]
    context_frames: int | None
    input_resolution: tuple[int, int] | None  # height, width
    artifacts: tuple[Artifact, ...]
    source_url: str
    paper_url: str
    license: str
    caveats: tuple[str, ...]
    local_recipe: str | None = None
    output_names: tuple[str, ...] = ()  # Source-reported; verify against the actual ONNX graph.
    output_timestamps_seconds: tuple[float, ...] | None = None
    output_scale_to_meters: float | None = None
    output_contract_verified: bool = False

    @property
    def download_bytes(self) -> int:
        return sum(artifact.size_bytes for artifact in self.artifacts)


class ModelUnavailableError(ValueError):
    """The catalog identifies a model, but contains no downloadable ONNX bundle."""


class ArtifactIntegrityError(ValueError):
    """Downloaded bytes do not match the pinned artifact manifest."""


# Sizes and LFS SHA-256 IDs were read from MANIFEST_URL. The small NoMaD distance
# external-data file is an ordinary Git blob; its SHA-256 was computed directly.
_MODELS = (
    ModelSpec(
        id="gnm",
        name="GNM (published image-goal export)",
        implementation_kind="published_export_variant",
        status="artifacts_available",
        modalities=("rgb",),
        conditioning=("image_goal",),
        context_frames=6,
        input_resolution=(64, 85),
        artifacts=(
            Artifact(
                "GNM_GL_Official/gnm_imagegoal.onnx",
                619733,
                "8914d698a58f0f49ca8da5a2f61d646d759a2178ad6f8a89f0e863b625110c77",
            ),
            Artifact(
                "GNM_GL_Official/gnm_imagegoal.onnx.data",
                34603008,
                "2764830aaf3d0a6189f1f2a0d5c9d2f4c725aa9cc9638a8beeae9e6c9dd59b8a",
            ),
        ),
        source_url="https://github.com/robodhruv/visualnav-transformer",
        paper_url="https://arxiv.org/abs/2210.03370",
        license="MIT upstream; Apache-2.0 zoo wrapper, subject to original model terms",
        caveats=(
            "The zoo's goal-free wrapper supplies random goal images to an image-goal architecture.",
            "Random-goal evaluation is a separate ablation; verify camera preprocessing and metric spacing.",
            "The local gnm recipe omits the original goal encoder and is an architecture adaptation.",
        ),
        local_recipe="gnm",
        output_names=("dist_pred", "action_pred"),
    ),
    ModelSpec(
        id="vint",
        name="ViNT (published image-goal export)",
        implementation_kind="published_export_variant",
        status="artifacts_available",
        modalities=("rgb",),
        conditioning=("image_goal",),
        context_frames=6,
        input_resolution=(64, 85),
        artifacts=(
            Artifact(
                "Vint_GL_Official/vint_imagegoal.onnx",
                1415900,
                "d480bf0fc28f6c1d84f568bf28733f6b418920218e3affd6f9d59fff1506ecec",
            ),
            Artifact(
                "Vint_GL_Official/vint_imagegoal.onnx.data",
                95748096,
                "399b8d50e7fcc812a69734fcc51f40dd313d964fec79d6fea3429252bc994791",
            ),
        ),
        source_url="https://github.com/robodhruv/visualnav-transformer",
        paper_url="https://arxiv.org/abs/2306.14846",
        license="MIT upstream; Apache-2.0 zoo wrapper, subject to original model terms",
        caveats=(
            "The zoo's goal-free wrapper supplies random goal images to an image-goal architecture.",
            "Verify observation cadence, camera preprocessing and embodiment-specific waypoint scaling.",
            "The local vint recipe omits the goal encoder and is an architecture adaptation.",
        ),
        local_recipe="vint",
        output_names=("dist_pred", "action_pred"),
    ),
    ModelSpec(
        id="nomad",
        name="NoMaD (published component exports)",
        implementation_kind="published_export_variant",
        status="artifacts_available",
        modalities=("rgb",),
        conditioning=("image_goal", "goal_free"),
        context_frames=4,
        input_resolution=(96, 96),
        artifacts=(
            Artifact(
                "NoMaD_GL_Official/nomad_vision_encoder.onnx",
                47986697,
                "a3477772d7ff671dbba9711c58b2df89deb9b1b6a93cae07c530524ab6d47377",
            ),
            Artifact(
                "NoMaD_GL_Official/nomad_vision_encoder.onnx.data",
                47448064,
                "57ceee7e8f0777b6e5cb444cddb1e17549e8188c78ba414463696b130031b320",
            ),
            Artifact(
                "NoMaD_GL_Official/nomad_noise_pred.onnx",
                15554469,
                "f3588e1ff53ba748240f160d3cd3fec0c9c987f05e59ec64a631c3536e036e34",
            ),
            Artifact(
                "NoMaD_GL_Official/nomad_dist_pred.onnx",
                71682,
                "adab58bef61d7b8d0f2bf04636853f0056229e06634e2536a5e7821d6b9def4f",
            ),
            Artifact(
                "NoMaD_GL_Official/nomad_dist_pred.onnx.data",
                69632,
                "f5c54bf7d70159a741188ad5d84348e11a673e424064f6493ffbed1a0ba4b76b",
            ),
        ),
        source_url="https://github.com/robodhruv/visualnav-transformer",
        paper_url="https://arxiv.org/abs/2310.07896",
        license="MIT upstream; Apache-2.0 zoo wrapper, subject to original model terms",
        caveats=(
            "These are components, not a complete inference graph: include the entire diffusion loop in timing.",
            "Published wrapper uses 10 DDPM steps and 8 samples; record scheduler, seed and sample count.",
            "The local nomad recipe uses a different denoiser and omits learned goal masking.",
        ),
        local_recipe="nomad",
    ),
    ModelSpec(
        id="citywalker",
        name="CityWalker (published point-goal export)",
        implementation_kind="published_export_variant",
        status="artifacts_available",
        modalities=("rgb", "past_odometry"),
        conditioning=("point_goal",),
        context_frames=5,
        input_resolution=(350, 630),
        artifacts=(
            Artifact(
                "CityWalker_PG_Official/citywalker.onnx",
                806082192,
                "9e8fb9ff081a883d80f0502d5b7e9046ed4b0dfafd46fe96ff5ad16194f0949a",
            ),
        ),
        source_url="https://github.com/ai4ce/CityWalker",
        paper_url="https://arxiv.org/abs/2411.17820",
        license="Apache-2.0 upstream; original model terms apply",
        caveats=(
            "Published wrapper ignores goal_xy and inserts a random final coordinate row; do not reuse it unchanged.",
            "Verify past-odometry normalization, goal row, coordinate conversion and image preprocessing.",
            "The local citywalker recipe omits coordinate conditioning and changes the decoder.",
        ),
        local_recipe="citywalker",
        output_names=("wp_pred", "arrive_pred"),
    ),
    ModelSpec(
        id="s2e",
        name="S2E BC Web100 (published point-goal export)",
        implementation_kind="published_export_variant",
        status="artifacts_available",
        modalities=("rgb",),
        conditioning=("point_goal",),
        context_frames=11,
        input_resolution=(256, 256),
        artifacts=(
            Artifact(
                "S2E/s2e.onnx",
                381993971,
                "ee1410ab55a54946cf25323e82b3be2a8cb106e2ccb2ad878e7638839686108a",
            ),
        ),
        source_url="https://github.com/VAIL-UCLA/S2E",
        paper_url="https://arxiv.org/abs/2507.22028",
        license="Apache-2.0 model zoo; original model terms apply",
        caveats=(
            "Released weights are the BC web-pretrained variant, not the paper's RL-finetuned policy.",
            "Wrapper comments conflict with executed goal encoding and output scaling; validate the actual contract.",
            "The local s2e recipe is a DINOv3/MHP adaptation, not the original anchor-guided architecture.",
        ),
        local_recipe="s2e",
        output_names=("wp_pred", "wp_pred_score"),
    ),
    ModelSpec(
        id="mimic",
        name="MIMIC (Honglin He, VAIL-UCLA; published goal-free export)",
        implementation_kind="published_export_variant",
        status="artifacts_available",
        modalities=("rgb",),
        conditioning=("goal_free",),
        context_frames=16,
        input_resolution=(288, 512),
        artifacts=(
            Artifact(
                "MIMIC/mimic.onnx",
                318192969,
                "7557512c791b824a1d693368d590ee6741af8a9aeef7974b763eb912df69d765",
            ),
        ),
        source_url="https://vail-ucla.github.io/MIMIC/",
        paper_url="https://arxiv.org/abs/2603.22527",
        license="Apache-2.0 model zoo; original model terms apply",
        caveats=(
            "Released export is goal-free, fixed batch 1; it is distinct from the paper's full point-goal model.",
            "Outputs have nonuniform timestamps; the published wrapper truncates 15 waypoints to 13.",
            "Wrapper has stale model documentation and an unnecessary urbansim import; verify scaling directly.",
            "The local mimic recipe lacks the paper's horizon-anchor decoder and hierarchical supervision.",
        ),
        local_recipe="mimic",
        output_timestamps_seconds=tuple(t / 5 for t in (1, 2, 4, 6, 7, 8, 10, 12, 14, 15, 17, 19, 21, 23, 25)),
        output_names=("output",),
    ),
    ModelSpec(
        id="navdp",
        name="NavDP",
        implementation_kind="external_integration_pending",
        status="unavailable",
        modalities=("rgb", "depth"),
        conditioning=("point_goal", "image_goal", "goal_free"),
        context_frames=8,
        input_resolution=(224, 224),
        artifacts=(),
        source_url="https://github.com/InternRobotics/NavDP",
        paper_url="https://arxiv.org/abs/2505.08712",
        license="CC-BY-NC-SA-4.0 upstream code; checkpoint terms require verification",
        caveats=(
            "Official checkpoint access requires the author's linked form; no public ONNX bundle is verified.",
            "RGB-D input, diffusion sampling and critic selection are required; an RGB-only substitute is not NavDP.",
            "Input dimensions describe upstream defaults, not a validated exported checkpoint.",
        ),
    ),
    ModelSpec(
        id="mbra",
        name="MBRA (published point-goal export)",
        implementation_kind="published_export_variant",
        status="artifacts_available",
        modalities=("rgb",),
        conditioning=("point_goal",),
        context_frames=6,
        input_resolution=(96, 96),
        artifacts=(
            Artifact(
                "MBRA_PG_Official/mbra.onnx",
                253629409,
                "fc42919737964d9c4db9428d8a836ad9d95007f1a41fcf5cf95526e1042f17a5",
            ),
        ),
        source_url="https://model-base-reannotation.github.io/",
        paper_url="https://arxiv.org/abs/2505.05592",
        license="Apache-2.0 model zoo; upstream model terms require verification",
        caveats=(
            "Verify point-goal pose encoding, observation cadence and embodiment-specific waypoint scaling.",
            "Artifact availability does not establish reference-export parity or open-loop metric validity.",
        ),
    ),
)
_BY_ID = {model.id: model for model in _MODELS}
MODELS: Mapping[str, ModelSpec] = MappingProxyType(_BY_ID)


def register_model(model: ModelSpec) -> None:
    """Register case-insensitive metadata without replacing an existing identity."""
    key = model.id.casefold()
    if not key or key != model.id.strip().casefold():
        raise ValueError("Model ID must be nonempty and have no surrounding whitespace")
    if key in _BY_ID:
        raise ValueError(f"Model {model.id!r} is already registered")
    _BY_ID[key] = model


def list_models() -> tuple[ModelSpec, ...]:
    """Return all catalog entries without network access or optional dependencies."""
    return tuple(_BY_ID.values())


def get_model(model_id: str) -> ModelSpec:
    """Look up an ID case-insensitively, e.g. ``get_model('NoMaD')``."""
    try:
        return _BY_ID[model_id.casefold()]
    except KeyError:
        raise KeyError(f"Unknown model {model_id!r}; choose from {', '.join(_BY_ID)}") from None


def _matches_artifact(path: Path, artifact: Artifact) -> bool:
    if not path.is_file() or path.stat().st_size != artifact.size_bytes:
        return False
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest() == artifact.sha256


def _download_artifact(artifact: Artifact, destination: Path, *, timeout: float) -> Path:
    root = destination.resolve()
    target = root.joinpath(*PurePosixPath(artifact.path).parts)
    if not target.resolve().is_relative_to(root):
        raise ValueError(f"Artifact destination escapes the download directory: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if _matches_artifact(target, artifact):
        return target

    request = Request(artifact.url, headers={"User-Agent": "visnavkit-benchmark/0.1"})
    temporary: Path | None = None
    try:
        with urlopen(request, timeout=timeout) as response:
            with tempfile.NamedTemporaryFile(dir=target.parent, prefix=f".{target.name}.", delete=False) as stream:
                temporary = Path(stream.name)
                digest = hashlib.sha256()
                size = 0
                while chunk := response.read(_CHUNK_BYTES):
                    size += len(chunk)
                    if size > artifact.size_bytes:
                        raise ArtifactIntegrityError(f"{artifact.path}: response exceeds expected size")
                    digest.update(chunk)
                    stream.write(chunk)
                if size != artifact.size_bytes or digest.hexdigest() != artifact.sha256:
                    raise ArtifactIntegrityError(f"{artifact.path}: size or SHA-256 does not match the pinned manifest")
                stream.flush()
                os.fsync(stream.fileno())
        os.replace(temporary, target)
        temporary = None
        return target
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def download_model(model_id: str, destination: str | Path, *, timeout: float = 60.0) -> tuple[Path, ...]:
    """Fetch verified ONNX files, preserving external-data sibling paths.

    Every existing file is checked before reuse. Each new file replaces its target
    atomically only after its size and SHA-256 match. If a bundle is interrupted,
    verified complete files remain reusable; the function returns only when all
    files are present. It never downloads or executes the release's Python code.
    ``timeout`` is the timeout for each blocking network operation, in seconds.
    """
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    model = get_model(model_id)
    if not model.artifacts:
        raise ModelUnavailableError(f"{model.name} has no verified public ONNX bundle. {model.caveats[0]}")
    return tuple(_download_artifact(artifact, Path(destination), timeout=timeout) for artifact in model.artifacts)
