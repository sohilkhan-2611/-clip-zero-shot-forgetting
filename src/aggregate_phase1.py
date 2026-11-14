"""Phase 1 table: mean and standard deviation over seeds of every revert / retain condition.

Run:  python -m src.aggregate_phase1 --inputs results/phase1/components_flowers_*.json \
                                     --out results/phase1/summary_components_flowers.json

Only complete per-seed files are used. Prints a Markdown table in accuracy points.
"""

from __future__ import annotations

import argparse
import glob
import json
import statistics
from pathlib import Path

from src.utils import run_metadata, save_json

FIELDS = ("target_acc", "gain", "mean_forgetting", "necessity", "recovery", "forgetting_share")


def mean_std(values: list[float]) -> dict:
    values = [v for v in values if v is not None]
    if not values:
        return {"mean": None, "std": None, "n": 0}
    return {"mean": statistics.fmean(values), "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "n": len(values)}


def aggregate(summaries: list[dict]) -> dict:
    """summaries: one summarize_conditions() output per seed. Returns condition -> field -> mean/std/n."""
    conditions = list(summaries[0])
    out = {}
    for cond in conditions:
        rows = [s[cond] for s in summaries if cond in s]
        out[cond] = {f: mean_std([r.get(f) for r in rows]) for f in FIELDS if any(f in r for r in rows)}
        held = rows[0]["forgetting"].keys()
        out[cond]["forgetting"] = {k: mean_std([r["forgetting"][k] for r in rows]) for k in held}
    return out


def _pts(stat: dict | None, scale: float = 100.0) -> str:
    if not stat or stat["mean"] is None:
        return ""
    return f"{scale * stat['mean']:.1f} ± {scale * stat['std']:.1f}"


def markdown_table(agg: dict) -> str:
    held = list(next(iter(agg.values()))["forgetting"])
    head = ["condition", "target acc", "gain", "necessity", "recovery (%)", "mean forgetting"] + held
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for cond, a in agg.items():
        cells = [cond, _pts(a.get("target_acc")), _pts(a.get("gain")), _pts(a.get("necessity")),
                 _pts(a.get("recovery")), _pts(a.get("mean_forgetting"))]
        cells += [_pts(a["forgetting"][k]) for k in held]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n\nValues are accuracy points (mean ± std over seeds); forgetting = pre minus condition."


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True, help="per-seed JSON files or glob patterns")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    files = sorted({f for pat in args.inputs for f in glob.glob(pat)})
    data = [json.loads(Path(f).read_text()) for f in files]
    used = [(f, d) for f, d in zip(files, data) if d.get("complete")]
    skipped = [f for f, d in zip(files, data) if not d.get("complete")]
    if not used:
        raise SystemExit("No complete per-seed files found")
    agg = aggregate([d["summary"] for _, d in used])
    table = markdown_table(agg)
    save_json({"meta": run_metadata(), "inputs": [f for f, _ in used], "skipped_incomplete": skipped,
               "n_seeds": len(used), "aggregate": agg, "table_markdown": table}, args.out)
    print(table)
    if skipped:
        print(f"skipped incomplete: {skipped}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
