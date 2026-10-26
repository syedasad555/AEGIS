#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AEGIS Phase 23 — Automated Regression Test Runner
=================================================
Discovers and executes all regression test suites, computes detailed pass/fail
statistics, timings, and outputs machine-readable JSON results.
"""

import json
import sys
import time
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def run_regression_suite(output_json: Path = PROJECT_ROOT / "outputs" / "reports" / "phase23_regression_results.json") -> bool:
    """Executes all tests and exports structured JSON summary."""
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir=str(PROJECT_ROOT / "tests"), pattern="test_*.py")

    print("=" * 75)
    print(" AEGIS Phase 23 — Comprehensive Regression Test Suite")
    print("=" * 75)

    start_time = time.time()
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    elapsed = time.time() - start_time

    total_tests = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    skipped = len(result.skipped)
    passed = total_tests - (failures + errors + skipped)
    pass_rate_pct = round((passed / max(1, total_tests)) * 100.0, 2)

    failure_details = []
    for test, trace in result.failures:
        failure_details.append({
            "test": str(test),
            "type": "FAILURE",
            "traceback": trace
        })
    for test, trace in result.errors:
        failure_details.append({
            "test": str(test),
            "type": "ERROR",
            "traceback": trace
        })

    summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "suite": "AEGIS Phase 23 Regression Tests",
        "total_tests": total_tests,
        "passed": passed,
        "failed": failures,
        "errors": errors,
        "skipped": skipped,
        "pass_rate_pct": pass_rate_pct,
        "execution_time_sec": round(elapsed, 3),
        "failures": failure_details,
        "environment": {
            "python_version": sys.version.split()[0],
            "platform": sys.platform,
            "torch_device": "cpu"
        }
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    phase25_json = PROJECT_ROOT / "outputs" / "reports" / "phase25_regression_results.json"
    summary_25 = dict(summary)
    summary_25["suite"] = "AEGIS Phase 25 Trajectory & System Regression Tests"
    with open(phase25_json, "w", encoding="utf-8") as f:
        json.dump(summary_25, f, indent=2)

    print("\n" + "=" * 75)
    print(f" REGRESSION TEST RESULTS: {passed}/{total_tests} Passed ({pass_rate_pct}%) in {elapsed:.2f}s")
    print(f" Machine-readable report saved: {output_json} & {phase25_json}")
    print("=" * 75)

    return (failures == 0 and errors == 0)


if __name__ == "__main__":
    success = run_regression_suite()
    sys.exit(0 if success else 1)
