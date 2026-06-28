from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import pandas as pd
from scipy import stats


PROJECT = Path(__file__).resolve().parents[1]
OUT_DEFAULT = PROJECT / "results" / "ipm_event_evidence_benchmark"
MVEC_FINAL = PROJECT / "results" / "ipm_multi_view_evidence_full_validation_final" / "ipm_multi_view_evidence_full_detail.csv"
QCPM_FINAL = PROJECT / "results" / "ipm_score_preserving_full_validation_final" / "full_detail.csv"
TRANSFER_FINAL = PROJECT / "results" / "transfer_score_audit_baselines" / "transfer_score_audit_baselines_detail.csv"
SMOKE_SPLITS = {
    "2025_01_to_2025_02",
    "2025_04_to_2025_05",
    "2025_07_to_2025_08",
    "2025_10_to_2025_11",
}

GATES = {
    "rolling": {"evidence": 0.0060, "support": 0.30, "precision": -0.015},
    "airport": {"evidence": 0.0030, "support": 0.00, "precision": -0.020},
    "airport_holdout": {"evidence": 0.0030, "support": 0.00, "precision": -0.020},
    "carrier": {"evidence": 0.0040, "support": 0.00, "precision": -0.020},
    "carrier_holdout": {"evidence": 0.0040, "support": 0.00, "precision": -0.020},
    "cross_year": {"evidence": 0.0050, "support": 0.00, "precision": -0.020},
}

METHOD_MAP = {
    "Teacher score reference": "teacher",
    "MVEC transfer-guarded consensus": "mvec_tg",
    "MVEC selected calibration rule": "calibrated_vote",
    "MVEC any-view retrieval": "mvec_any",
    "MVEC two-view consensus": "mvec_two",
    "MVEC three-view consensus": "mvec_three",
    "QCPM Fisher contrastive evidence": "qcpm_fisher",
    "QCPM hybrid contrastive evidence": "qcpm_hybrid",
    "PE-KB queue": "pe_kb_queue",
}


def parse_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def normalize_protocol(value: str) -> str:
    return {"airport": "airport_holdout", "carrier": "carrier_holdout"}.get(value, value)


def _to_float(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0.0)


def _queued_from_rows(rows: pd.Series) -> pd.Series:
    return _to_float(rows).map(lambda value: int(math.ceil(0.10 * value)))


def standardize_mvec(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    out = pd.DataFrame(
        {
            "protocol": df["protocol"].map(normalize_protocol),
            "split": df["split"].astype(str),
            "queue_source": "LightGBM",
            "evidence_method": df["method"].map(METHOD_MAP).fillna(df["method"].astype(str)),
            "rows": _to_float(df["rows"]).astype(int),
            "positives": _to_float(df["positives"]).astype(int),
            "reviewed": _to_float(df["queued"]).astype(int),
            "captured": _to_float(df["captured"]).astype(int),
            "supported": _to_float(df["supported"]).astype(int),
            "support_total": (_to_float(df["supported"]) + _to_float(df.get("normal_supported", 0))).astype(int),
            "pr_auc": _to_float(df["pr_auc"]),
            "top10_capture": _to_float(df["top10_capture"]),
            "support_precision": _to_float(df["support_precision"]),
        }
    )
    return out


def standardize_qcpm(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[~df["method"].eq("Teacher score reference")].copy()
    out = pd.DataFrame(
        {
            "protocol": df["protocol"].map(normalize_protocol),
            "split": df["split"].astype(str),
            "queue_source": "LightGBM",
            "evidence_method": df["method"].map(METHOD_MAP).fillna(df["method"].astype(str)),
            "rows": _to_float(df["rows"]).astype(int),
            "positives": _to_float(df["positives"]).astype(int),
            "reviewed": _queued_from_rows(df["rows"]).astype(int),
            "captured": _to_float(df["captured"]).astype(int),
            "supported": _to_float(df["supported"]).astype(int),
            "support_total": (_to_float(df["supported"]) + _to_float(df.get("normal_supported", 0))).astype(int),
            "pr_auc": _to_float(df["pr_auc"]),
            "top10_capture": _to_float(df["top10_capture"]),
        }
    )
    out["support_precision"] = out["supported"] / out["support_total"].clip(lower=1)
    return out


def standardize_transfer(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    queue = df["queue"].astype(str)
    queue_source = queue.str.extract(r"^(LightGBM|XGBoost|CatBoost)", expand=False).fillna("PE-KB")
    out = pd.DataFrame(
        {
            "protocol": df["protocol"].map(normalize_protocol),
            "split": df["fold"].astype(str),
            "queue_source": queue_source,
            "evidence_method": "unified_proto",
            "rows": _to_float(df["rows"]).astype(int),
            "positives": _to_float(df["positives"]).astype(int),
            "reviewed": _to_float(df["reviewed_paths"]).astype(int),
            "captured": _to_float(df["captured_severe_cases"]).astype(int),
            "supported": _to_float(df["evidence_supported_severe_cases"]).astype(int),
            "support_total": 0,
            "pr_auc": _to_float(df["pr_auc"]),
            "top10_capture": _to_float(df["top10_capture"]),
        }
    )
    precision = df.get("support_precision")
    if precision is None:
        out["support_precision"] = pd.NA
    else:
        out["support_precision"] = _to_float(precision)
    return out


def load_available_detail(include_transfer: bool = True) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    if MVEC_FINAL.exists():
        parts.append(standardize_mvec(MVEC_FINAL))
    if QCPM_FINAL.exists():
        parts.append(standardize_qcpm(QCPM_FINAL))
    if include_transfer and TRANSFER_FINAL.exists():
        parts.append(standardize_transfer(TRANSFER_FINAL))
    if not parts:
        return pd.DataFrame()
    detail = pd.concat(parts, ignore_index=True)
    detail["evidence_hit"] = detail["supported"] / detail["captured"].clip(lower=1)
    detail["supported_per_100"] = 100.0 * detail["supported"] / detail["reviewed"].clip(lower=1)
    needs_precision = detail["support_precision"].isna()
    detail.loc[needs_precision, "support_precision"] = detail.loc[needs_precision, "supported"] / detail.loc[
        needs_precision, "support_total"
    ].clip(lower=1)
    return detail


def filter_detail(detail: pd.DataFrame, mode: str, protocols: list[str], queue_sources: list[str], methods: list[str]) -> pd.DataFrame:
    if detail.empty:
        return detail
    protocols_norm = {normalize_protocol(protocol) for protocol in protocols}
    queue_set = set(queue_sources)
    method_set = set(methods)
    out = detail[detail["protocol"].isin(protocols_norm)].copy()
    out = out[out["queue_source"].isin(queue_set)]
    out = out[out["evidence_method"].isin(method_set)]
    if mode == "smoke":
        split_text = out["split"].astype(str)
        smoke_mask = split_text.isin(SMOKE_SPLITS)
        for split in SMOKE_SPLITS:
            smoke_mask = smoke_mask | split_text.str.startswith(f"{split}_")
        out = out[smoke_mask | out["protocol"].eq("cross_year")]
    return out.reset_index(drop=True)


def summarize_benchmark_detail(detail: pd.DataFrame) -> pd.DataFrame:
    if detail.empty:
        return pd.DataFrame()
    work = detail.copy()
    for col in ["rows", "positives", "reviewed", "captured", "supported"]:
        work[col] = pd.to_numeric(work[col], errors="coerce").fillna(0).astype(int)
    grouped = work.groupby(["protocol", "queue_source", "evidence_method"], as_index=False)
    summary = grouped.agg(
        splits=("split", "nunique"),
        rows=("rows", "sum"),
        positives=("positives", "sum"),
        reviewed=("reviewed", "sum"),
        captured=("captured", "sum"),
        supported=("supported", "sum"),
        mean_pr_auc=("pr_auc", "mean"),
        mean_top10_capture=("top10_capture", "mean"),
        mean_support_precision=("support_precision", "mean"),
    )
    summary["evidence_hit"] = summary["supported"] / summary["captured"].clip(lower=1)
    summary["supported_per_100"] = 100.0 * summary["supported"] / summary["reviewed"].clip(lower=1)
    summary["support_precision"] = summary["mean_support_precision"]
    return summary.sort_values(["protocol", "queue_source", "supported_per_100"], ascending=[True, True, False])


def compare_to_teacher(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for (protocol, queue_source), group in summary.groupby(["protocol", "queue_source"], sort=False):
        teacher_rows = group[group["evidence_method"].eq("teacher")]
        if teacher_rows.empty:
            continue
        teacher = teacher_rows.iloc[0]
        gate = GATES.get(protocol, GATES["airport_holdout"])
        for _, row in group.iterrows():
            evidence_delta = float(row["evidence_hit"] - teacher["evidence_hit"])
            support_delta = float(row["supported_per_100"] - teacher["supported_per_100"])
            precision_delta = float(row["support_precision"] - teacher["support_precision"])
            pr_delta = float(row["mean_pr_auc"] - teacher["mean_pr_auc"])
            top_delta = float(row["mean_top10_capture"] - teacher["mean_top10_capture"])
            decision = (
                "pass"
                if row["evidence_method"] == "teacher"
                or (
                    pr_delta >= -0.0005
                    and top_delta >= -0.0005
                    and evidence_delta >= gate["evidence"]
                    and support_delta >= gate["support"]
                    and precision_delta >= gate["precision"]
                )
                else "fail"
            )
            data = row.to_dict()
            data.update(
                {
                    "pr_auc_delta": pr_delta,
                    "top10_capture_delta": top_delta,
                    "evidence_hit_delta": evidence_delta,
                    "supported_per_100_delta": support_delta,
                    "support_precision_delta": precision_delta,
                    "absolute_supported_gain": int(row["supported"] - teacher["supported"]),
                    "decision": decision,
                }
            )
            rows.append(data)
    judged = pd.DataFrame(rows)
    return pareto_frontier_flags(judged) if not judged.empty else judged


def paired_method_tests(detail: pd.DataFrame, method: str = "mvec_tg", reference: str = "teacher") -> pd.DataFrame:
    if detail.empty:
        return pd.DataFrame()
    metrics = ["evidence_hit", "supported_per_100", "support_precision"]
    rows: list[dict[str, Any]] = []
    keys = ["protocol", "queue_source"]
    work = detail.copy()
    for metric in metrics:
        work[metric] = pd.to_numeric(work[metric], errors="coerce")
    for (protocol, queue_source), group in work.groupby(keys, sort=False):
        pivot = group.pivot_table(index="split", columns="evidence_method", values=metrics, aggfunc="mean")
        if method not in pivot.columns.get_level_values(1) or reference not in pivot.columns.get_level_values(1):
            continue
        for metric in metrics:
            paired = pivot[(metric, method)].to_frame("method").join(pivot[(metric, reference)].to_frame("reference"), how="inner").dropna()
            if paired.empty:
                continue
            delta = paired["method"] - paired["reference"]
            if len(delta) >= 2 and delta.std(ddof=1) > 0:
                t_p = float(stats.ttest_rel(paired["method"], paired["reference"]).pvalue)
            else:
                t_p = 1.0 if float(delta.abs().sum()) == 0.0 else 0.0
            if len(delta) >= 1 and float(delta.abs().sum()) > 0.0:
                try:
                    w_p = float(stats.wilcoxon(delta, zero_method="wilcox").pvalue)
                except ValueError:
                    w_p = 1.0
            else:
                w_p = 1.0
            rows.append(
                {
                    "protocol": protocol,
                    "queue_source": queue_source,
                    "method": method,
                    "reference": reference,
                    "metric": metric,
                    "paired_splits": int(len(delta)),
                    "mean_delta": float(delta.mean()),
                    "median_delta": float(delta.median()),
                    "min_delta": float(delta.min()),
                    "max_delta": float(delta.max()),
                    "paired_t_p": t_p,
                    "wilcoxon_p": w_p,
                }
            )
    return pd.DataFrame(rows)


def pareto_frontier_flags(table: pd.DataFrame) -> pd.DataFrame:
    if table.empty:
        return table.copy()
    out = table.copy()
    flags: list[bool] = []
    metrics = ["evidence_hit", "supported_per_100", "support_precision"]
    for idx, row in out.iterrows():
        peers = out[(out["protocol"].eq(row["protocol"])) & (out["queue_source"].eq(row["queue_source"])) & (out.index != idx)]
        dominated = False
        for _, peer in peers.iterrows():
            at_least = all(float(peer[metric]) >= float(row[metric]) for metric in metrics)
            strictly = any(float(peer[metric]) > float(row[metric]) for metric in metrics)
            if at_least and strictly:
                dominated = True
                break
        flags.append(not dominated)
    out["pareto_frontier"] = flags
    return out


def write_report(out: Path, detail: pd.DataFrame, summary: pd.DataFrame, judgement: pd.DataFrame, paired_tests: pd.DataFrame | None = None) -> None:
    lines = [
        "# IPM event evidence benchmark",
        "",
        "This benchmark normalizes available event evidence results into a queue-source by evidence-method comparison table.",
        "",
        "## Judgement",
        "",
        judgement.to_markdown(index=False, floatfmt=".4f") if not judgement.empty else "(no judgement rows)",
        "",
        "## Summary",
        "",
        summary.to_markdown(index=False, floatfmt=".4f") if not summary.empty else "(no summary rows)",
        "",
    ]
    if paired_tests is not None and not paired_tests.empty:
        lines.extend(
            [
                "## Paired split-level tests",
                "",
                paired_tests.to_markdown(index=False, floatfmt=".4g"),
                "",
            ]
        )
    (out / "ipm_event_evidence_benchmark_report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT_DEFAULT)
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--protocols", default="rolling,airport,carrier,cross_year")
    parser.add_argument("--queue-sources", default="LightGBM,XGBoost,CatBoost")
    parser.add_argument(
        "--evidence-methods",
        default="teacher,unified_proto,qcpm_fisher,qcpm_hybrid,mvec_any,mvec_two,mvec_tg,calibrated_vote",
    )
    parser.add_argument("--include-transfer", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    detail = load_available_detail(include_transfer=args.include_transfer)
    detail = filter_detail(
        detail,
        args.mode,
        parse_list(args.protocols),
        parse_list(args.queue_sources),
        parse_list(args.evidence_methods),
    )
    summary = summarize_benchmark_detail(detail)
    judgement = compare_to_teacher(summary)
    paired_tests = paired_method_tests(detail)
    detail.to_csv(args.out / "ipm_event_evidence_benchmark_detail.csv", index=False)
    summary.to_csv(args.out / "ipm_event_evidence_benchmark_summary.csv", index=False)
    judgement.to_csv(args.out / "ipm_event_evidence_benchmark_judgement.csv", index=False)
    paired_tests.to_csv(args.out / "ipm_event_evidence_paired_tests.csv", index=False)
    write_report(args.out, detail, summary, judgement, paired_tests)
    print(judgement.to_string(index=False) if not judgement.empty else "no judgement")


if __name__ == "__main__":
    main()
