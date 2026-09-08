"""Scheduler and complete sampler tests; no remote artifacts required."""

from types import SimpleNamespace

import numpy as np
import pytest

from visnavkit.benchmark import published, runtime


class FakeSession:
    def __init__(self, kind):
        self.kind = kind
        self.calls = []
        names = {
            "vision": ["obs_img", "goal_img", "goal_mask"],
            "denoiser": ["sample", "timestep", "global_cond"],
            "distance": ["condition"],
        }[kind]
        self.inputs = [SimpleNamespace(name=name, shape=[1], type="tensor(float)") for name in names]

    def get_inputs(self):
        return self.inputs

    def get_session_options(self):
        return SimpleNamespace(intra_op_num_threads=1, execution_mode="sequential", graph_optimization_level="all")

    def get_providers(self):
        return ["CPUExecutionProvider"]

    def get_provider_options(self):
        return {}

    def run(self, names, feeds):
        self.calls.append(feeds)
        if self.kind == "vision":
            batch = feeds["obs_img"].shape[0]
            return [np.arange(batch * 3, dtype=np.float32).reshape(batch, 3)]
        if self.kind == "denoiser":
            return [np.zeros_like(feeds["sample"])]
        return [np.ones((feeds["condition"].shape[0], 1), dtype=np.float32)]


def _feeds(batch=2):
    return {
        "obs_img": np.ones((batch, 12, 2, 2), dtype=np.float32),
        "goal_img": np.ones((batch, 3, 2, 2), dtype=np.float32),
        "goal_mask": np.ones(batch, dtype=np.int64),
    }


def test_cosine_schedule_and_clean_sample_recovery():
    betas = published.cosine_betas()
    assert len(betas) == 10
    assert betas[0] == pytest.approx(0.02790726288603096)
    assert betas[-1] == 0.999
    assert np.all((betas > 0) & (betas < 1))
    clean = np.array([[[0.2, -0.8], [2.0, -2.0]]], dtype=np.float32)
    epsilon = np.full_like(clean, 0.3)
    noisy = np.sqrt(1 - betas[0]) * clean + np.sqrt(betas[0]) * epsilon
    recovered = published.ddpm_step(noisy, epsilon, 0, betas, np.ones_like(clean) * 99)
    np.testing.assert_allclose(recovered, np.clip(clean, -1, 1), atol=1e-6)


def test_ddpm_posterior_noise_variance():
    betas = np.array([0.1, 0.2], dtype=np.float64)
    sample = np.zeros((1, 1, 2), dtype=np.float32)
    zero = published.ddpm_step(sample, sample, 1, betas, sample)
    noisy = published.ddpm_step(sample, sample, 1, betas, np.ones_like(sample))
    # q(x_0 | x_1, clean) has variance beta_1 * (1-alpha_0) / (1-alpha_0*alpha_1).
    np.testing.assert_allclose(noisy - zero, np.sqrt(0.2 * 0.1 / 0.28), rtol=1e-6)


def test_full_sampler_counts_repeats_and_determinism():
    vision, denoiser = FakeSession("vision"), FakeSession("denoiser")
    sampler = published.NoMaDSampler(vision, denoiser, batch_size=2, num_samples=3, seed=7)
    first = sampler.run(None, _feeds())[0]
    second = sampler.run(None, _feeds())[0]
    np.testing.assert_array_equal(first, second)
    assert first.shape == (2, 3, 8, 2)
    assert len(vision.calls) == 2
    assert len(denoiser.calls) == 20
    assert [int(call["timestep"]) for call in denoiser.calls[:10]] == list(reversed(range(10)))
    np.testing.assert_array_equal(denoiser.calls[0]["global_cond"], np.repeat(np.arange(6).reshape(2, 3), 3, axis=0))
    assert np.isfinite(first).all()
    assert np.abs(first).max() <= 1.0


def test_runtime_times_all_sampler_steps_and_optional_distance(monkeypatch):
    monkeypatch.setattr(runtime, "_environment", lambda: {})
    vision, denoiser, distance = FakeSession("vision"), FakeSession("denoiser"), FakeSession("distance")
    sampler = published.NoMaDSampler(vision, denoiser, batch_size=2, num_samples=3, distance=distance)
    report = runtime.benchmark_session(sampler, _feeds(), warmup=1, iterations=2)
    assert len(vision.calls) == 3
    assert len(denoiser.calls) == 30
    assert len(distance.calls) == 3
    assert report["batch_size"] == 2
    assert len(report["latency_samples_ms"]) == 2


def test_sampler_rejects_mismatched_component_contract():
    vision, denoiser = FakeSession("vision"), FakeSession("denoiser")
    denoiser.inputs[0].name = "different_sample"
    with pytest.raises(ValueError, match="input contract differs"):
        published.NoMaDSampler(vision, denoiser)


def test_scheduler_matches_diffusers_when_available(monkeypatch):
    diffusers = pytest.importorskip("diffusers")
    import torch
    from diffusers.schedulers import scheduling_ddpm

    scheduler = diffusers.DDPMScheduler(
        num_train_timesteps=10, beta_schedule="squaredcos_cap_v2", prediction_type="epsilon", clip_sample=True
    )
    scheduler.set_timesteps(10)
    rng = np.random.default_rng(41)
    sample = rng.normal(size=(2, 8, 2)).astype(np.float32)
    epsilon = rng.normal(size=sample.shape).astype(np.float32)
    variance = rng.normal(size=sample.shape).astype(np.float32)
    monkeypatch.setattr(scheduling_ddpm, "randn_tensor", lambda *args, **kwargs: torch.from_numpy(variance))
    for timestep in reversed(range(10)):
        expected = scheduler.step(torch.from_numpy(epsilon), timestep, torch.from_numpy(sample)).prev_sample.numpy()
        actual = published.ddpm_step(sample, epsilon, timestep, published.cosine_betas(), variance)
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)
        sample = actual
