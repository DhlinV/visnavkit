import json

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from visnavkit.benchmark.runtime import benchmark_session, create_session, inspect_graph, make_inputs


def _linear_model(path, dynamic=False, activation=False, external=False):
    batch = "batch" if dynamic else 2
    nodes = [helper.make_node("MatMul", ["input", "weight"], ["linear" if activation else "output"])]
    if activation:
        nodes.append(helper.make_node("Relu", ["linear"], ["output"]))
    graph = helper.make_graph(
        nodes,
        "linear",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [batch, 3])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [batch, 4])],
        [numpy_helper.from_array(np.arange(12, dtype=np.float32).reshape(3, 4), name="weight")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=10)
    onnx.save_model(
        model,
        path,
        save_as_external_data=external,
        all_tensors_to_one_file=True,
        location="weights.data",
        size_threshold=0,
    )
    return path


def test_fixed_feeds_runtime_and_exact_matmul_count(tmp_path):
    path = _linear_model(tmp_path / "linear.onnx")
    session = create_session(path)
    feeds = make_inputs(session, seed=7)
    np.testing.assert_array_equal(feeds["input"], make_inputs(session, seed=7)["input"])
    assert np.all(feeds["input"] != 0)
    summary = benchmark_session(session, feeds, warmup=1, iterations=3)
    assert summary["batch_size"] == 2
    assert summary["latency_ms"]["min"] <= summary["latency_ms"]["p50"] <= summary["latency_ms"]["max"]
    assert summary["samples_per_second"] > 0
    assert len(summary["latency_samples_ms"]) == 3
    json.dumps(summary, allow_nan=False)
    inspection = inspect_graph(path, feeds)
    assert inspection["initializer_elements"] == 12
    assert inspection["initializer_bytes"] == 48
    assert inspection["flops"]["total_flops"] == 2 * 2 * 3 * 4
    assert inspection["flops"]["complete"]


def test_dynamic_shapes_require_explicit_override(tmp_path):
    path = _linear_model(tmp_path / "dynamic.onnx", dynamic=True)
    session = create_session(path)
    with pytest.raises(ValueError, match="unresolved shape"):
        make_inputs(session)
    feeds = make_inputs(session, shapes={"input": [5, 3]})
    assert feeds["input"].shape == (5, 3)
    assert inspect_graph(path)["flops"]["total_flops"] is None
    assert inspect_graph(path, feeds)["flops"]["total_flops"] == 120
    with pytest.raises(ValueError, match="static dimension"):
        make_inputs(session, shapes={"input": [5, 4]})


def test_partial_flops_are_not_reported_as_total(tmp_path):
    path = _linear_model(tmp_path / "relu.onnx", activation=True)
    result = inspect_graph(path)["flops"]
    assert result["counted_flops"] == 48
    assert result["total_flops"] is None
    assert result["unsupported_compute_ops"] == {"Relu": 1}


def test_external_weights_are_in_artifact_size_and_hash(tmp_path):
    path = _linear_model(tmp_path / "external.onnx", external=True)
    result = inspect_graph(path)
    assert len(result["artifact_files"]) == 2
    assert result["artifact_bytes"] == path.stat().st_size + (tmp_path / "weights.data").stat().st_size
    assert all(len(item["sha256"]) == 64 for item in result["artifact_files"])
    assert result["initializer_bytes"] == 48


def test_missing_provider_fails_instead_of_cpu_fallback(tmp_path):
    path = _linear_model(tmp_path / "linear.onnx")
    with pytest.raises(ValueError, match="available providers"):
        create_session(path, provider="MissingExecutionProvider")


def test_invalid_benchmark_iterations_and_feeds(tmp_path):
    session = create_session(_linear_model(tmp_path / "linear.onnx"))
    feeds = make_inputs(session)
    with pytest.raises(ValueError, match="positive integer"):
        benchmark_session(session, feeds, iterations=0)
    with pytest.raises(ValueError, match="Expected inputs"):
        benchmark_session(session, {})
    feeds["input"][0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        benchmark_session(session, feeds)


@pytest.mark.parametrize(
    "op,input_shape,weight_shape,output_shape,attributes,bias,expected,complete",
    [
        ("Conv", [1, 4, 8, 8], [6, 2, 3, 3], [1, 6, 6, 6], {"group": 2}, None, 7776, True),
        ("Conv", [1, 4, 8, 8], [6, 2, 3, 3], [1, 6, 6, 6], {"group": 2}, [6], 7776, False),
        ("Gemm", [3, 2], [3, 4], [2, 4], {"transA": 1}, None, 48, True),
        ("Gemm", [3, 2], [3, 4], [2, 4], {"transA": 1}, [4], 48, False),
        ("Gemm", [3, 2], [3, 4], [2, 4], {"transA": 1, "alpha": 0.5}, None, 48, False),
        ("MatMul", [5, 2, 3], [3, 4], [5, 2, 4], {}, None, 240, True),
    ],
)
def test_flop_conventions(tmp_path, op, input_shape, weight_shape, output_shape, attributes, bias, expected, complete):
    names = ["input", "weight"]
    initializers = [numpy_helper.from_array(np.ones(weight_shape, dtype=np.float32), name="weight")]
    if bias is not None:
        names.append("bias")
        initializers.append(numpy_helper.from_array(np.ones(bias, dtype=np.float32), name="bias"))
    graph = helper.make_graph(
        [helper.make_node(op, names, ["output"], **attributes)],
        "counting",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape)],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, output_shape)],
        initializers,
    )
    path = tmp_path / "counting.onnx"
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=10), path)
    result = inspect_graph(path)["flops"]
    assert result["counted_flops"] == expected
    assert result["complete"] is complete
    assert result["total_flops"] == (expected if complete else None)
    assert bool(result["unaccounted_arithmetic_ops"]) is (not complete)
