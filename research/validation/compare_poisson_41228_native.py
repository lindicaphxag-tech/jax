"""Compare exact-commit JAX CPU benchmark JSON outputs; descriptive, not a CI gate.

The two runs must use the same host/runtime, seeds and matrix. Timing and
statistical differences are reported without applying an arbitrary pass cutoff.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


BASE = "d3f12669d7db55596d413eba9cfcd214ae5a3ea4"
CANDIDATE = "9428e21c09e6a19cadbf9ed02d6ec78b165f6246"


def load(path, expected_sha):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data["git_sha"] != expected_sha:
        raise ValueError(f"{path}: wrong commit {data['git_sha']}")
    if data["backend"] != "cpu":
        raise ValueError(f"{path}: expected CPU, got {data['backend']}")
    return data


def keyed(payload):
    records = {}
    for r in payload["records"]:
        key = (r["dtype"], r["n_rates"], r["requested_rate"])
        if key in records:
            raise ValueError(f"duplicate key {key}; use per-seed grouping")
        records[key] = r
    return records


def main():
    p = argparse.ArgumentParser()
    p.add_argument("baseline")
    p.add_argument("candidate")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    before = load(a.baseline, BASE)
    after = load(a.candidate, CANDIDATE)
    for key in ("platform", "backend", "jaxlib_version"):
        if before.get(key) != after.get(key):
            raise ValueError(f"{key} environment differs between runs")
    left, right = keyed(before), keyed(after)
    if left.keys() != right.keys():
        raise ValueError("input matrix differs")
    items = []
    for key in sorted(left):
        old, new = left[key], right[key]
        if old["effective_rate"] != new["effective_rate"]:
            raise ValueError(f"rate encoding differs: {key}")
        items.append({
            "dtype": key[0], "n_rates": key[1], "rate": key[2],
            "sample_median_ratio": (
                new["sampler"]["median_ms"] / old["sampler"]["median_ms"]
            ),
            "sample_p90_ratio": (
                new["sampler"]["p90_ms"] / old["sampler"]["p90_ms"]
            ),
            "logpmf_median_ratio": (
                new["logpmf"]["median_ms"] / old["logpmf"]["median_ms"]
            ),
            "baseline_logpmf_error": old["logpmf_max_abs_scipy"],
            "candidate_logpmf_error": new["logpmf_max_abs_scipy"],
            "baseline_sample_z": old["sample_mean_z"],
            "candidate_sample_z": new["sample_mean_z"],
            "baseline_variance_ratio": old["sample_variance_ratio"],
            "candidate_variance_ratio": new["sample_variance_ratio"],
        })
    report = {
        "baseline": BASE, "candidate": CANDIDATE,
        "warning": (
            "Descriptive CPU timing only. Separate checkouts can have unequal "
            "host load, JIT backends or run ordering. No statistical speedup "
            "or public PR merge-readiness is established by this report."
        ),
        "results": items,
    }
    Path(a.out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
