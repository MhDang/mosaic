"""BFCL results: a run's per-rank generation files aggregated into accuracy metrics.

    python benchmarks/bfcl/summary.py outputs/bfcl_json/simple/*
    python benchmarks/bfcl/summary.py --table outputs/bfcl_paper    # run_paper.sh's runs, one row per setting

``ans`` (answer success) is BFCL's AST accuracy: the fraction of generations
the official checker accepts, averaged over examples. ``parse`` is the
fraction that parse as tool calls, ``constraint`` the fraction the constraint
automaton accepts (constrained runs only).
"""

import contextlib
import glob
import io
import json
import os
import sys


#: BFCL's AST splits: its non-live score is the unweighted mean over the first four, its live score
#: the mean over the live examples (each split weighted by its size)
NON_LIVE = ["simple", "multiple", "parallel", "parallel_multiple"]
LIVE = ["live_simple", "live_multiple", "live_parallel", "live_parallel_multiple"]


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    if sys.argv[1] == "--table":
        return table(sys.argv[2])
    for run_dir in sys.argv[1:]:
        files = sorted(glob.glob(os.path.join(run_dir, "generations_*.json")))
        if files:
            summarize(files, label=os.path.relpath(run_dir))
        else:
            print(f"(no generations in {run_dir})")


def summarize(files, file=None, label=None):
    """Print (and append to ``file``) the metrics for these generation files; returns them."""
    records = []
    for path in files:
        with open(path) as f:
            records.extend(json.load(f))

    merged = {}
    for r in records:
        m = merged.setdefault(r["id"], {
            "generated": 0, "ans_success": 0, "parse_success": 0,
            "constraint_success": 0, "ans_constraint_success": 0, "has_constraint": False, "wall_time": 0.0,
        })
        assert r["ans_failure_count"] + r["success_count"] + r["parsing_failure_count"] == r["generated_count"]
        m["generated"] += r["generated_count"]
        m["ans_success"] += r["success_count"]
        m["parse_success"] += r["generated_count"] - r["parsing_failure_count"]
        m["wall_time"] += r.get("wall_time", 0.0)
        if r.get("satisfies_constraint") is not None:
            m["has_constraint"] = True
            m["constraint_success"] += sum(r["satisfies_constraint"])
            m["ans_constraint_success"] += sum(
                bool(a) and b == 1 for a, b in zip(r["satisfies_constraint"], r["res_list_raw"])
            )

    n = len(merged)
    if n == 0:
        print("no records")
        return {}
    has_constraint = all(m["has_constraint"] for m in merged.values())

    def avg(key):
        return sum(m[key] / m["generated"] for m in merged.values()) / n

    total_generated = sum(m["generated"] for m in merged.values())
    out = {
        "num_examples": n,
        "avg_generated_count": total_generated / n,
        "avg_parse_success_rate": avg("parse_success"),
        "avg_ans_success_rate": avg("ans_success"),
        "avg_constraint_success_rate": avg("constraint_success") if has_constraint else None,
        "avg_ans_constraint_success_rate": avg("ans_constraint_success") if has_constraint else None,
        "wall_time_per_sample": sum(m["wall_time"] for m in merged.values()) / total_generated,
    }

    def fmt(v):
        return "n/a" if v is None else f"{v:.4f}"

    run_dir = os.path.dirname(os.path.abspath(files[0])) if files else ""
    lines = [
        f"=== {label or 'results'} ({len(files)} files, {n} examples) {run_dir} ===",
        f"  Avg parse success rate:          {fmt(out['avg_parse_success_rate'])}",
        f"  Avg constraint success rate:     {fmt(out['avg_constraint_success_rate'])}",
        f"  Avg ans success rate:            {fmt(out['avg_ans_success_rate'])}",
        f"  Avg ans constraint success rate: {fmt(out['avg_ans_constraint_success_rate'])}",
        f"  Wall time per sample (s):        {out['wall_time_per_sample']:.3f}",
    ]
    text = "\n".join(lines)
    print(text)
    if file is not None:
        with open(file, "a") as f:
            f.write(text + "\n")
    return out


def table(root):
    """AST accuracy (%) per split, and the non-live and live scores, for each finished run under
    ``root/<template>/<setting>/<split>`` (the layout of ``run_paper.sh``)."""
    print("| setting | " + " | ".join(NON_LIVE) + " | non-live | " + " | ".join(LIVE) + " | live |")
    print("|---" * (len(NON_LIVE) + len(LIVE) + 3) + "|")
    for setting in sorted(glob.glob(os.path.join(root, "*", "*"))):
        acc, n = {}, {}
        for split in NON_LIVE + LIVE:
            run = os.path.join(setting, split)
            if os.path.exists(os.path.join(run, "summary.txt")):
                with contextlib.redirect_stdout(io.StringIO()):
                    m = summarize(sorted(glob.glob(os.path.join(run, "generations_*.json"))))
                acc[split], n[split] = 100 * m["avg_ans_success_rate"], m["num_examples"]
        non_live = sum(acc[s] for s in NON_LIVE) / len(NON_LIVE) if all(s in acc for s in NON_LIVE) else None
        live = (sum(acc[s] * n[s] for s in LIVE) / sum(n[s] for s in LIVE)) if all(s in acc for s in LIVE) else None
        cells = [acc.get(s) for s in NON_LIVE] + [non_live] + [acc.get(s) for s in LIVE] + [live]
        print(f"| {os.path.relpath(setting, root)} | " + " | ".join("-" if c is None else f"{c:.2f}" for c in cells) + " |")


if __name__ == "__main__":
    main()
