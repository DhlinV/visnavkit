"""Hooks receive benchmark events; failures propagate instead of losing results."""

import csv
import json
from pathlib import Path

from visnavkit.benchmark.registry import HOOKS


def _flatten(value, prefix=""):
    output = {}
    for key, item in value.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(item, dict):
            output.update(_flatten(item, name))
        elif isinstance(item, (tuple, list)):
            output[name] = json.dumps(item)
        else:
            output[name] = item
    return output


@HOOKS.register("json_csv")
class ReportHook:
    def on_result(self, result, output_dir):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        text = json.dumps(result, indent=2, allow_nan=False) + "\n"
        temp = output_dir / "result.json.tmp"
        temp.write_text(text)
        temp.replace(output_dir / "result.json")
        rows = result.get("results", [result])
        flat = [_flatten(row) for row in rows]
        fields = sorted({key for row in flat for key in row})
        with (output_dir / "results.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(flat)


def emit(hooks, event, **kwargs):
    for hook in hooks:
        method = getattr(hook, event, None)
        if method is not None:
            method(**kwargs)
