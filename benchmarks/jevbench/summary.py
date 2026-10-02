"""JevBench results: each run's official metrics, and each variant's mean ± sd over its runs (seeds).

    python benchmarks/jevbench/summary.py --jevbench /path/to/jevbench RUN_DIR [RUN_DIR ...]

A RUN_DIR holds the records.jsonl that run_vllm.py (next to this file) wrote; a run with
several variants gives a row per variant. JevBench's rules: a format failure
counts as wrong in Intelligence and is left out of Calibration, which covers the parsed hard
items; Capability is their mean. :func:`summarize` is the summary.json run_vllm.py writes
next to each run's records.
"""

import argparse
import json
import os
import statistics as st
import sys

#: JevBench's tier weights in Intelligence.
TIER_WEIGHTS = {"easy": 0.14, "standard": 0.28, "judge": 0.28, "hard": 0.30}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jevbench", required=True, help="a checkout of fstandhartinger/jevbench")
    ap.add_argument("runs", nargs="+", help="run directories")
    args = ap.parse_args()
    sys.path.insert(0, args.jevbench)
    from jevbench.metrics import ece_top_label
    from jevbench.scoring import score_task
    from jevbench.tasks import load_jsonl

    tasks, tier_of = {}, {}
    for f, tier in (("easy", "easy"), ("original", "standard"), ("hard", "hard")):
        for t in load_jsonl(os.path.join(args.jevbench, "datasets/public", f + ".jsonl")):
            tasks[t.id], tier_of[t.id] = t, tier

    def official(recs):
        scored = {i: score_task(r["probs"], tasks[i]) for i, r in recs.items()}
        ok = {i: bool(s["correct"]) for i, s in scored.items()}
        intel = intelligence(ok, tasks, tier_of)
        rows = [(recs[i], s["probs"], ok[i]) for i, s in scored.items() if tier_of[i] == "hard" and s["valid"]]
        cal = calibration(rows, tasks, ece_top_label)
        hard = [ok[i] for i in ok if tier_of[i] == "hard"]
        return {"failures": sum(not s["valid"] for s in scored.values()), "hard_acc": st.mean(hard), "I": intel,
                "C": cal["calibration"], "ece": cal["ece"], "cal_n": len(rows), "cap": (intel + cal["calibration"]) / 2}

    by_variant = {}
    print("| run | variant | failures | hard acc | Intelligence | Calibration | Capability |")
    print("|---|---|---|---|---|---|---|")
    for run in args.runs:
        with open(os.path.join(run, "records.jsonl")) as f:
            records = [json.loads(line) for line in f]
        for variant in sorted({r["variant"] for r in records}):
            o = official({r["id"]: r for r in records if r["variant"] == variant})
            by_variant.setdefault(variant, []).append(o)
            print(f"| {os.path.basename(run.rstrip('/'))} | {variant} | {o['failures']} | {o['hard_acc']:.3f} | "
                  f"{o['I']:.1f} | {o['C']:.1f} (ECE {o['ece']:.3f}, n={o['cal_n']}) | **{o['cap']:.1f}** |")
    for variant, runs in by_variant.items():
        if len(runs) < 2:
            continue
        m = {f: st.mean(x[f] for x in runs) for f in ("failures", "hard_acc", "I", "C", "cap")}
        sd = {f: st.stdev(x[f] for x in runs) for f in ("hard_acc", "I", "C", "cap")}
        print(f"| mean±sd ({len(runs)}) | {variant} | {m['failures']:.1f} | {m['hard_acc']:.3f}±{sd['hard_acc']:.3f} | "
              f"{m['I']:.1f}±{sd['I']:.1f} | {m['C']:.1f}±{sd['C']:.1f} | **{m['cap']:.1f}±{sd['cap']:.1f}** |")


def calibration(scored, tasks, ece_fn):
    """JevBench v1.2 Calibration from (row, probs, correct) of valid hard items."""
    pairs = [(max(p.values()), c) for _, p, c in scored]
    tvds = [0.5 * sum(abs(p.get(k, 0.0) - v) for k, v in tasks[r["id"]].provenance["gold_probs"].items())
            for r, p, _ in scored if tasks[r["id"]].provenance.get("gold_probs")]
    ece = ece_fn(pairs)["ece"]
    tvd = sum(tvds) / len(tvds) if tvds else None
    a = max(0.0, 100 * (1 - ece / 0.5))
    return {"ece": ece, "tvd": tvd, "calibration": a if tvd is None else (a + 100 * (1 - tvd)) / 2}


def intelligence(correct, tasks, tier_of):
    """JevBench's Intelligence: per tier 100 (acc - chance) / (1 - chance), chance = mean 1/options; tier-weighted."""
    acc, chance = {}, {}
    for i, ok in correct.items():
        t = tier_of[i]
        acc.setdefault(t, []).append(ok)
        chance.setdefault(t, []).append(1 / len(tasks[i].labels))
    per = {}
    for t in acc:
        a, c = sum(acc[t]) / len(acc[t]), sum(chance[t]) / len(chance[t])
        per[t] = max(0.0, (a - c) / (1 - c))
    return 100 * sum(TIER_WEIGHTS[t] * v for t, v in per.items()) / sum(TIER_WEIGHTS[t] for t in per)


def summarize(records, tasks, jevbench_dir):
    sys.path.insert(0, jevbench_dir)
    from jevbench.metrics import ece_top_label
    from jevbench.scoring import score_task

    by_id = {t.id: (t, tier) for t, tier in tasks}
    per_task = json.load(open(os.path.join(jevbench_dir, "results/v1.2/jevbench-v1.2-per-task.json")))
    djev = per_task["systems"]["djev"]["public_tasks"]
    djev_thinking = per_task["systems"]["djev-thinking"]["public_tasks"]
    out = {}
    for variant in sorted({r["variant"] for r in records}):
        rows = [r for r in records if r["variant"] == variant]
        tier_acc, djev_acc, thinking_acc, hard_pairs, tvds, lat = {}, {}, {}, [], [], []
        failed = 0  # no valid distribution (JevBench's failures): wrong, and left out of Calibration
        for r in rows:
            task, tier = by_id[r["id"]]
            s = score_task(r["probs"], task)
            failed += not s["valid"]
            if s["correct"] is None:  # no ground truth: excluded, as in JevBench's headline
                continue
            tier_acc.setdefault(tier, []).append(bool(s["correct"]))
            if r["id"] in djev:
                djev_acc.setdefault(tier, []).append(djev[r["id"]][0] == "c")
            if r["id"] in djev_thinking:
                thinking_acc.setdefault(tier, []).append(djev_thinking[r["id"]][0] == "c")
            if tier == "hard" and s["valid"]:
                hard_pairs.append((max(s["probs"].values()), bool(s["correct"])))
                gold = task.provenance.get("gold_probs")
                if gold:
                    tvds.append(0.5 * sum(abs(s["probs"].get(k, 0.0) - v) for k, v in gold.items()))
            lat.append(r["latency_s"])
        lat.sort()
        ece = ece_top_label(hard_pairs)["ece"] if hard_pairs else None
        out[variant] = {
            "n": len(rows),
            "accuracy": {t: sum(v) / len(v) for t, v in tier_acc.items()},
            "failed": failed,
            "djev_accuracy_same_items": {t: sum(v) / len(v) for t, v in djev_acc.items()},
            "djev_thinking_accuracy_same_items": {t: sum(v) / len(v) for t, v in thinking_acc.items()},
            "hard_ece": ece,
            "mean_tvd_gold": sum(tvds) / len(tvds) if tvds else None,
            "calibration_v12": (100 * max(0, 1 - ece / 0.5) + 100 * (1 - sum(tvds) / len(tvds))) / 2
            if ece is not None and tvds else None,
            "latency_p50_s": lat[len(lat) // 2], "latency_p95_s": lat[min(len(lat) - 1, int(0.95 * len(lat)))],
        }
    return out


if __name__ == "__main__":
    main()
