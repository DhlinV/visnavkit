"""ONNX Runtime measurements with explicit inputs, execution scope, and FLOP coverage."""

import hashlib
import importlib.metadata
import math
import os
import platform
import subprocess
import time
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort


def create_session(path, provider="CPUExecutionProvider", threads=1):
    """Load one graph, rejecting unavailable providers and CPU fallback for GPU runs."""
    if provider not in ort.get_available_providers():
        raise ValueError(f"Requested {provider}; available providers: {ort.get_available_providers()}")
    if not isinstance(threads, int) or isinstance(threads, bool) or threads < 1:
        raise ValueError("threads must be a positive integer")
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if provider != "CPUExecutionProvider":
        options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    session = ort.InferenceSession(str(path), sess_options=options, providers=[provider])
    session.disable_fallback()
    if session.get_providers()[0] != provider:
        raise RuntimeError(f"Requested {provider}, but session uses {session.get_providers()}")
    return session


_INPUT_DTYPES = {
    "float": np.float32,
    "float16": np.float16,
    "double": np.float64,
    "bool": np.bool_,
    **{name: np.dtype(name) for name in ("int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64")},
}


def make_inputs(session, seed=0, shapes=None):
    """Create repeatable nonzero feeds; every dynamic dimension needs an explicit shape.

    ``shapes`` maps input names to complete shapes. Integer inputs contain ones;
    callers must supply real inputs when a model imposes additional value constraints.
    """
    shapes = {} if shapes is None else shapes
    unknown = set(shapes) - {item.name for item in session.get_inputs()}
    if unknown:
        raise ValueError(f"Shape overrides name unknown inputs: {sorted(unknown)}")
    rng = np.random.default_rng(seed)
    feeds = {}
    for item in session.get_inputs():
        shape = shapes.get(item.name, item.shape)
        if len(shape) != len(item.shape):
            raise ValueError(f"Input {item.name} expects rank {len(item.shape)}, got {shape}")
        for actual, declared in zip(shape, item.shape):
            if not isinstance(actual, int) or isinstance(actual, bool) or actual < 0:
                raise ValueError(f"Input {item.name} has unresolved shape {shape}; provide an explicit shape override")
            if isinstance(declared, int) and actual != declared:
                raise ValueError(f"Input {item.name} dimension {actual} does not match static dimension {declared}")
        dtype = _INPUT_DTYPES.get(item.type.removeprefix("tensor(").removesuffix(")"))
        if dtype is None:
            raise ValueError(f"Cannot generate input {item.name} with ONNX type {item.type}")
        dtype = np.dtype(dtype)
        if np.issubdtype(dtype, np.floating):
            value = rng.uniform(0.05, 0.95, size=shape).astype(dtype)
        else:
            value = np.ones(shape, dtype=dtype)
        feeds[item.name] = value
    return feeds


def _environment():
    cpu_model = platform.processor()
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu_model = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        gpus = result.stdout.strip().splitlines()
    except (OSError, subprocess.SubprocessError):
        gpus = []
    versions = {}
    for name in ("numpy", "onnx", "onnxruntime", "onnxruntime-gpu", "torch"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return {
        "os": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "cpu_model": cpu_model,
        "logical_cpus": os.cpu_count(),
        "gpu_inventory_name_uuid_driver_memory_mib": gpus,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "versions": versions,
    }


def benchmark_session(session, feeds, warmup=10, iterations=100):
    """Time synchronous ``session.run`` including host/device transfers and output allocation.

    Session construction, feed generation, warmup, preprocessing, and video decode
    are excluded. The same feeds are reused for every measured call.
    """
    if not isinstance(warmup, int) or isinstance(warmup, bool) or warmup < 0:
        raise ValueError("warmup must be a nonnegative integer")
    if not isinstance(iterations, int) or isinstance(iterations, bool) or iterations < 1:
        raise ValueError("iterations must be a positive integer")
    expected = {item.name for item in session.get_inputs()}
    if set(feeds) != expected:
        raise ValueError(f"Expected inputs {sorted(expected)}, got {sorted(feeds)}")
    if any(not isinstance(value, np.ndarray) for value in feeds.values()):
        raise ValueError("Feeds must be NumPy arrays")
    if any(not np.isfinite(value).all() for value in feeds.values()):
        raise ValueError("Feeds must contain only finite values")
    feeds = {name: np.ascontiguousarray(value) if value.ndim else value for name, value in feeds.items()}
    primary = feeds[session.get_inputs()[0].name] if feeds else None
    batch_size = int(primary.shape[0]) if primary is not None and primary.ndim else 1
    if batch_size < 1:
        raise ValueError("Cannot benchmark an empty batch")
    for _ in range(warmup):
        session.run(None, feeds)
    durations = []
    for _ in range(iterations):
        start = time.perf_counter_ns()
        session.run(None, feeds)
        durations.append((time.perf_counter_ns() - start) / 1e6)
    measured = np.asarray(durations)
    options = session.get_session_options()
    return {
        "provider": session.get_providers()[0],
        "session_providers": session.get_providers(),
        "provider_options": session.get_provider_options(),
        "threads": options.intra_op_num_threads,
        "execution_mode": str(options.execution_mode),
        "graph_optimization": str(options.graph_optimization_level),
        "warmup": warmup,
        "iterations": iterations,
        "batch_size": batch_size,
        "batch_size_source": "first graph input leading dimension, or 1 for scalar/no input",
        "inputs": {name: {"shape": list(value.shape), "dtype": str(value.dtype)} for name, value in feeds.items()},
        "latency_ms": {
            "mean": float(measured.mean()),
            "std": float(measured.std()),
            "min": float(measured.min()),
            "max": float(measured.max()),
            **{f"p{p}": float(np.percentile(measured, p)) for p in (50, 90, 95, 99)},
        },
        "samples_per_second": float(batch_size * 1000 / measured.mean()),
        "latency_samples_ms": durations,
        "timing_scope": "synchronous session.run; host inputs and outputs; transfers and output allocation included",
        "excluded_from_timing": ["session construction", "input generation", "warmup", "preprocessing", "video decode"],
        "environment": _environment(),
    }


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _graph_tensors(graph):
    yield from graph.initializer
    for node in graph.node:
        for attribute in node.attribute:
            if attribute.type == onnx.AttributeProto.TENSOR:
                yield attribute.t
            elif attribute.type == onnx.AttributeProto.TENSORS:
                yield from attribute.tensors
            elif attribute.type == onnx.AttributeProto.GRAPH:
                yield from _graph_tensors(attribute.g)
            elif attribute.type == onnx.AttributeProto.GRAPHS:
                for child in attribute.graphs:
                    yield from _graph_tensors(child)


# These operations rearrange/select values or metadata; this estimate ignores their
# memory traffic. Arithmetic, activations, normalization, and control flow remain
# explicitly unsupported instead of silently contributing zero FLOPs.
_MOVEMENT_OPS = {
    "Constant",
    "Identity",
    "Reshape",
    "Flatten",
    "Transpose",
    "Squeeze",
    "Unsqueeze",
    "Concat",
    "Split",
    "Slice",
    "Gather",
    "GatherElements",
    "GatherND",
    "Shape",
    "Size",
    "Expand",
    "Tile",
    "ConstantOfShape",
    "Cast",
    "Pad",
    "Dropout",
}


def _flops(model, feeds):
    if feeds is not None:
        input_map = {value.name: value for value in model.graph.input}
        if set(feeds) - set(input_map):
            raise ValueError(f"Unknown graph inputs: {sorted(set(feeds) - set(input_map))}")
        for name, value in feeds.items():
            dims = input_map[name].type.tensor_type.shape.dim
            if len(dims) != value.ndim:
                raise ValueError(f"Input {name} rank does not match graph")
            for dim, size in zip(dims, value.shape):
                if dim.HasField("dim_value") and dim.dim_value != size:
                    raise ValueError(f"Input {name} shape does not match graph")
                dim.ClearField("dim_param")
                dim.dim_value = size
    inference_error = None
    try:
        model = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    except (onnx.shape_inference.InferenceError, ValueError) as error:
        inference_error = str(error)
    shapes = {tensor.name: tuple(tensor.dims) for tensor in model.graph.initializer}
    for value in (*model.graph.input, *model.graph.value_info, *model.graph.output):
        dims = value.type.tensor_type.shape.dim
        if value.type.tensor_type.HasField("shape") and all(dim.HasField("dim_value") for dim in dims):
            shapes[value.name] = tuple(dim.dim_value for dim in dims)
    unsupported = Counter()
    unknown_shapes = []
    unaccounted_arithmetic = Counter()
    counted = 0
    counted_nodes = Counter()
    for index, node in enumerate(model.graph.node):
        op = node.op_type
        if node.domain not in ("", "ai.onnx"):
            unsupported[f"{node.domain}::{op}"] += 1
            continue
        if op in _MOVEMENT_OPS:
            continue
        if op not in {"Conv", "Gemm", "MatMul"}:
            unsupported[op] += 1
            continue
        left = shapes.get(node.input[0])
        right = shapes.get(node.input[1])
        output = shapes.get(node.output[0])
        if left is None or right is None or output is None:
            unknown_shapes.append(node.name or f"{op}_{index}")
            continue
        attributes = {item.name: onnx.helper.get_attribute_value(item) for item in node.attribute}
        if op in {"Conv", "Gemm"} and len(node.input) > 2 and node.input[2]:
            unaccounted_arithmetic[f"{op}.bias"] += 1
        if op == "Gemm" and attributes.get("alpha", 1.0) != 1.0:
            unaccounted_arithmetic["Gemm.alpha_scaling"] += 1
        if op == "Gemm" and len(node.input) > 2 and node.input[2] and attributes.get("beta", 1.0) != 1.0:
            unaccounted_arithmetic["Gemm.beta_scaling"] += 1
        if op == "Conv":
            operations = 2 * math.prod(output) * math.prod(right[1:])
        elif op == "Gemm":
            inner = left[0] if attributes.get("transA", 0) else left[1]
            operations = 2 * math.prod(output) * inner
        else:
            operations = 2 * math.prod(output) * left[-1]
        counted += operations
        counted_nodes[op] += 1
    complete = not unsupported and not unknown_shapes and not unaccounted_arithmetic and inference_error is None
    return {
        "counted_flops": counted,
        "total_flops": counted if complete else None,
        "complete": complete,
        "counted_nodes": dict(counted_nodes),
        "unsupported_compute_ops": dict(sorted(unsupported.items())),
        "unknown_shape_ops": unknown_shapes,
        "unaccounted_arithmetic_ops": dict(unaccounted_arithmetic),
        "shape_inference_error": inference_error,
        "convention": "2 FLOPs per multiply-accumulate in Conv/Gemm/MatMul; bias and Gemm scaling excluded",
        "scope": "original ONNX graph at supplied shapes; no runtime fusion, memory traffic, or preprocessing cost",
    }


def inspect_graph(path, feeds=None):
    """Inspect artifact bytes, stored initializers, and explicitly partial static FLOPs.

    Initializers include constants and folded values: their elements are NOT a
    trainable or total model parameter count. External tensor files are included
    in artifact bytes and hashes, once per file.
    """
    path = Path(path).resolve()
    model = onnx.load(str(path), load_external_data=False)
    external = set()
    for tensor in _graph_tensors(model.graph):
        if tensor.data_location == onnx.TensorProto.EXTERNAL:
            info = {item.key: item.value for item in tensor.external_data}
            location = info.get("location")
            if not location:
                raise ValueError(f"External tensor {tensor.name} has no location")
            external_path = (path.parent / location).resolve()
            if not external_path.is_relative_to(path.parent):
                raise ValueError(f"External tensor file must be inside the artifact directory: {location}")
            external.add(external_path)
    files = [path, *sorted(external - {path})]
    artifacts = [{"path": str(item), "bytes": item.stat().st_size, "sha256": _sha256(item)} for item in files]
    initializers = []
    for tensor in model.graph.initializer:
        elements = math.prod(tensor.dims)
        dtype_name = onnx.TensorProto.DataType.Name(tensor.data_type)
        if dtype_name in {"UINT4", "INT4", "FLOAT4E2M1"}:
            size = (elements + 1) // 2
        elif dtype_name == "STRING":
            size = None
        else:
            size = elements * np.dtype(onnx.helper.tensor_dtype_to_np_dtype(tensor.data_type)).itemsize
        initializers.append({"name": tensor.name, "elements": elements, "dtype": dtype_name, "bytes": size})
    return {
        "artifact_bytes": sum(item["bytes"] for item in artifacts),
        "artifact_files": artifacts,
        "initializer_count": len(initializers),
        "initializer_elements": sum(item["elements"] for item in initializers),
        "initializer_bytes": sum(item["bytes"] for item in initializers)
        if all(item["bytes"] is not None for item in initializers)
        else None,
        "initializer_scope": "main graph stored initializers only; includes constants/folding; NOT model parameter count",
        "opsets": {item.domain or "ai.onnx": item.version for item in model.opset_import},
        "node_counts": dict(Counter(node.op_type for node in model.graph.node)),
        "flops": _flops(model, feeds),
    }
