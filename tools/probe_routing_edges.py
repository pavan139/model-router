"""Small explicit source-fault probes in isolated copies; this is not TestProof.

Only selected reviewed behavior is probed. No production source is overwritten.
A green report is not a whole-project mutation score or a safety guarantee.
"""

from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

PROBES = [
    (
        "canonical-structure-leakage",
        "src/model_router/data.py",
        "structure = expression_structure(row.expression)",
        'structure = re.sub(r"\\d+", "#", re.sub(r"\\s+", "", row.expression))',
    ),
    ("exact-budget-equality", "src/model_router/core.py", "if spent + cost > budget:", "if spent + cost >= budget:"),
    (
        "dispatch-budget-equality",
        "src/model_router/core.py",
        "if quoted_costs[preferred] <= remaining:",
        "if quoted_costs[preferred] < remaining:",
    ),
    (
        "test-provenance",
        "src/model_router/core.py",
        '"model_evaluation": test_provenance == ["RECORDED"]',
        '"model_evaluation": any(response.provenance == "RECORDED" for row in records for response in row.responses)',
    ),
    (
        "bounded-import-read",
        "src/model_router/data.py",
        "line = source.readline(MAX_JSONL_LINE_CHARS + 1)",
        "line = source.readline()",
    ),
]


def run_tests(root: Path, source: Path) -> subprocess.CompletedProcess:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(source)
    return subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", str(root / "test"), "-v"],
        env=environment,
        capture_output=True,
        text=True,
        timeout=20,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    baseline = run_tests(root, root / "src")
    if baseline.returncode:
        raise SystemExit("Baseline failed; no source faults may be interpreted.")
    outcomes = []
    for name, relative, original, replacement in PROBES:
        with tempfile.TemporaryDirectory(prefix="review-source-probe-") as directory:
            copy = Path(directory)
            shutil.copytree(root / "src", copy / "src", ignore=shutil.ignore_patterns("__pycache__", "*.egg-info"))
            path = copy / relative
            source = path.read_text()
            if source.count(original) != 1:
                raise SystemExit(f"Probe {name} no longer identifies exactly one reviewed behavior")
            changed = source.replace(original, replacement, 1)
            compile(changed, str(path), "exec")
            path.write_text(changed)
            try:
                result = run_tests(root, copy / "src")
                match = re.search(r"Ran (\d+) tests?", result.stderr)
                status = (
                    "killed"
                    if result.returncode and match and "FAILED" in result.stderr
                    else "survived"
                    if not result.returncode
                    else "error"
                )
                outcomes.append(
                    {
                        "id": name,
                        "file": relative,
                        "line": source[: source.index(original)].count("\n") + 1,
                        "status": status,
                        "tests_run": int(match.group(1)) if match else None,
                        "returncode": result.returncode,
                    }
                )
            except subprocess.TimeoutExpired:
                outcomes.append({"id": name, "file": relative, "status": "timeout"})
    report = {
        "method": "explicit_selected_source_fault_probes",
        "baseline_passed": True,
        "scope": "Selected behaviors edited in this review; not whole-project mutation coverage or a TestProof score.",
        "comparison_base": "explicit reviewed source behaviors; no automatic git-diff enumeration",
        "total": len(outcomes),
        "killed": sum(row["status"] == "killed" for row in outcomes),
        "outcomes": outcomes,
    }
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")
    if report["killed"] != report["total"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
