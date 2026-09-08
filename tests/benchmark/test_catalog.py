"""Provenance and download integrity without network access or model weights."""

import hashlib
import io
from dataclasses import replace

import pytest

from visnavkit.benchmark import catalog


def _test_bundle(monkeypatch, payload=b"small ONNX fixture"):
    artifact = catalog.Artifact("fixture/model.onnx", len(payload), hashlib.sha256(payload).hexdigest())
    model = replace(catalog.get_model("gnm"), id="fixture", artifacts=(artifact,))
    monkeypatch.setitem(catalog._BY_ID, model.id, model)
    return artifact


def test_catalog_has_requested_models_and_verified_artifacts():
    assert {model.id for model in catalog.list_models()} == {
        "gnm",
        "vint",
        "nomad",
        "citywalker",
        "s2e",
        "mimic",
        "navdp",
        "mbra",
    }
    assert catalog.get_model("NoMaD").id == "nomad"
    assert catalog.get_model("navdp").artifacts == ()
    artifacts = [artifact for model in catalog.list_models() for artifact in model.artifacts]
    assert len(artifacts) == 13
    assert len({artifact.path for artifact in artifacts}) == len(artifacts)
    for artifact in artifacts:
        assert catalog.HF_REVISION in artifact.url
        assert artifact.size_bytes > 0
        assert len(artifact.sha256) == 64
        assert artifact.path.endswith((".onnx", ".onnx.data"))
    for model in catalog.list_models():
        assert model.download_bytes == sum(a.size_bytes for a in model.artifacts)
        assert not model.output_contract_verified


def test_mimic_identity_and_nonuniform_times():
    model = catalog.get_model("mimic")
    assert "Honglin He" in model.name
    assert model.conditioning == ("goal_free",)
    assert model.input_resolution == (288, 512)
    assert model.output_timestamps_seconds == (
        0.2,
        0.4,
        0.8,
        1.2,
        1.4,
        1.6,
        2.0,
        2.4,
        2.8,
        3.0,
        3.4,
        3.8,
        4.2,
        4.6,
        5.0,
    )
    assert model.output_scale_to_meters is None


def test_register_model_protects_existing_identities(monkeypatch):
    with pytest.raises(ValueError, match="already registered"):
        catalog.register_model(replace(catalog.get_model("gnm"), id="GNM"))
    model = replace(catalog.get_model("navdp"), id="custom_rgbd")
    with monkeypatch.context() as patch:
        patch.setattr(catalog, "_BY_ID", dict(catalog._BY_ID))
        catalog.register_model(model)
        assert catalog.get_model("CUSTOM_RGBD") == model
        assert model in catalog.list_models()


def test_download_uses_verified_cache_without_network(tmp_path, monkeypatch):
    payload = b"small ONNX fixture"
    artifact = _test_bundle(monkeypatch, payload)
    target = tmp_path / artifact.path
    target.parent.mkdir()
    target.write_bytes(payload)

    def unexpected_request(*args, **kwargs):
        pytest.fail("A verified cached artifact must not need the network")

    monkeypatch.setattr(catalog, "urlopen", unexpected_request)
    assert catalog.download_model("fixture", tmp_path) == (target,)


def test_download_checks_pinned_url_and_replaces_corrupt_cache(tmp_path, monkeypatch):
    payload = b"small ONNX fixture"
    artifact = _test_bundle(monkeypatch, payload)
    target = tmp_path / artifact.path
    target.parent.mkdir()
    target.write_bytes(b"broken cache")
    calls = []

    def request(request, timeout):
        calls.append((request.full_url, timeout))
        return io.BytesIO(payload)

    monkeypatch.setattr(catalog, "urlopen", request)
    assert catalog.download_model("fixture", tmp_path, timeout=5) == (target,)
    assert calls == [(artifact.url, 5)]
    assert target.read_bytes() == payload
    assert list(target.parent.iterdir()) == [target]


@pytest.mark.parametrize("payload", [b"truncated", b"x" * 18, b"oversized response content"])
def test_integrity_failure_preserves_existing_file(tmp_path, monkeypatch, payload):
    artifact = _test_bundle(monkeypatch)
    target = tmp_path / artifact.path
    target.parent.mkdir()
    target.write_bytes(b"previous file")
    monkeypatch.setattr(catalog, "urlopen", lambda *args, **kwargs: io.BytesIO(payload))
    with pytest.raises(catalog.ArtifactIntegrityError):
        catalog.download_model("fixture", tmp_path)
    assert target.read_bytes() == b"previous file"
    assert list(target.parent.iterdir()) == [target]


def test_navdp_fails_before_creating_destination(tmp_path):
    destination = tmp_path / "models"
    with pytest.raises(catalog.ModelUnavailableError, match="no verified public ONNX bundle"):
        catalog.download_model("navdp", destination)
    assert not destination.exists()


def test_download_rejects_symlink_escape(tmp_path, monkeypatch):
    _test_bundle(monkeypatch)
    destination = tmp_path / "models"
    destination.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (destination / "fixture").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="escapes"):
        catalog.download_model("fixture", destination)
    assert list(outside.iterdir()) == []


@pytest.mark.parametrize("path", ["../model.onnx", "/model.onnx", "folder/../../model.onnx", "folder\\model.onnx"])
def test_artifact_rejects_unsafe_paths(path):
    with pytest.raises(ValueError, match="relative POSIX"):
        catalog.Artifact(path, 1, "0" * 64)
