"""Command-line interface for the student-performance pipeline."""

from __future__ import annotations

import argparse
import json
import sys
import time

import pandas as pd

from . import config, data, eda, model, report
from .clean import clean, save_processed
from .merge import compare_subjects, pair_students, pairing_report


def _banner(step: str) -> None:
    print(f"\n{'=' * 72}\n {step}\n{'=' * 72}")


def _prepare(verbose: bool = True) -> tuple[pd.DataFrame, dict, dict]:
    frames = data.load_all(verbose=verbose)
    combined = data.combine(frames)
    validation = data.validate(combined)
    cleaned, cleaning_log = clean(combined)
    if verbose:
        print(f"[data] validation passed ({len(validation['checks'])} checks, "
              f"{validation['rows']:,} rows)")
        print(f"[clean] pass rate {cleaning_log['pass_rate']:.1%} | "
              f"mean G3 {cleaning_log['mean_g3']:.2f} | "
              f"{cleaning_log['duplicates_kept']} identical rows kept")
    return cleaned, cleaning_log, validation


def cmd_download(args: argparse.Namespace) -> int:
    _banner("STEP 1/4 - download & validate")
    for subject, path in data.download(force=args.force).items():
        print(f"  [{subject}] {path.name}")
    frames = data.load_all()
    validation = data.validate(data.combine(frames))
    for check in validation["checks"]:
        status = "PASS" if check["passed"] else ("FAIL" if check["critical"] else "warn")
        print(f"  [{status:>4}] {check['check']:<26} {check['detail']}")
    return 0


def cmd_merge(args: argparse.Namespace) -> int:
    _banner("STEP 2/4 - paired maths vs Portuguese analysis")
    frames = data.load_all(verbose=False)
    pairing = pairing_report(frames["mat"], frames["por"])
    pairs = pair_students(frames["mat"], frames["por"])
    comparison = compare_subjects(pairs)

    print(f"[merge] documented keys={len(config.MERGE_KEYS)} -> "
          f"{pairing['pairs']} paired students")
    print(f"[merge] naive merge on {pairing['all_shared_columns_tried']} shared "
          f"columns -> {pairing['naive_merge_rows']} rows")
    print(f"[merge] maths mean={comparison['mean_mat']:.2f} | "
          f"Portuguese mean={comparison['mean_por']:.2f} | "
          f"delta={comparison['mean_delta']:+.2f}")
    print(f"[merge] improved={comparison['improved']} "
          f"worsened={comparison['worsened']} tied={comparison['tied']} "
          f"| sign test p={comparison['sign_test_p']:.2g}")

    payload = {"pairing": pairing, "comparison": comparison}
    path = config.REPORT_DIR / "pairing.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[merge] report -> {path.relative_to(config.ROOT)}")
    return 0


def cmd_eda(args: argparse.Namespace) -> int:
    _banner("STEP 3/4 - exploratory data analysis")
    frames = data.load_all()
    cleaned, _, _ = _prepare(verbose=False)
    save_processed(cleaned)
    cleaned_frames = {subject: clean(frame)[0] for subject, frame in frames.items()}
    eda.run_eda(cleaned_frames)
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    _banner("STEP 4/4 - modelling under three leakage scenarios")
    frames = data.load_all(verbose=False)
    cleaned, cleaning_log, validation = _prepare(verbose=False)
    save_processed(cleaned)
    cleaned_frames = {subject: clean(frame)[0] for subject, frame in frames.items()}
    summary = eda.compute_summary(cleaned_frames)
    pairs = pair_students(cleaned_frames["mat"], cleaned_frames["por"])
    summary["paired_comparison"] = compare_subjects(pairs)
    metrics = model.train(cleaned, cleaned_frames)
    report.write_reports(summary, metrics, cleaning_log, validation)
    return 0


def cmd_all(args: argparse.Namespace) -> int:
    started = time.perf_counter()
    frames = data.load_all()
    cleaned, cleaning_log, validation = _prepare()
    save_processed(cleaned)
    cleaned_frames = {subject: clean(frame)[0] for subject, frame in frames.items()}

    _banner("STEP 2/4 - paired maths vs Portuguese analysis")
    pairing = pairing_report(cleaned_frames["mat"], cleaned_frames["por"])
    pairs = pair_students(cleaned_frames["mat"], cleaned_frames["por"])
    comparison = compare_subjects(pairs)
    (config.REPORT_DIR / "pairing.json").write_text(
        json.dumps({"pairing": pairing, "comparison": comparison}, indent=2),
        encoding="utf-8",
    )
    print(f"[merge] {pairing['pairs']} pairs | delta={comparison['mean_delta']:+.2f} "
          f"| sign test p={comparison['sign_test_p']:.2g}")

    _banner("STEP 3/4 - exploratory data analysis")
    summary = eda.run_eda(cleaned_frames)
    summary["paired_comparison"] = comparison

    _banner("STEP 4/4 - modelling under three leakage scenarios")
    metrics = model.train(cleaned, cleaned_frames)
    report.write_reports(summary, metrics, cleaning_log, validation)

    print(f"\n[done] end-to-end pipeline finished in {time.perf_counter() - started:.1f}s")
    return 0


def cmd_score(args: argparse.Namespace) -> int:
    bundle = model.load_model()
    frame = pd.read_csv(args.input, sep=args.sep)
    predictions = model.predict(frame, bundle)
    output = pd.DataFrame({
        "row_id": range(len(frame)),
        "predicted_g3": predictions.round(2),
        "predicted_pass": (predictions >= config.PASS_MARK).astype(int),
    })
    output.to_csv(args.output, index=False)
    print(f"[score] {len(output):,} rows -> {args.output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="student", description="Student performance analysis pipeline"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    download = subparsers.add_parser("download", help="download and validate the data")
    download.add_argument("--force", action="store_true", help="re-download")
    download.set_defaults(func=cmd_download)

    subparsers.add_parser("merge", help="paired subject analysis").set_defaults(
        func=cmd_merge)
    subparsers.add_parser("eda", help="run exploratory analysis").set_defaults(
        func=cmd_eda)
    subparsers.add_parser("train", help="train and evaluate").set_defaults(func=cmd_train)
    subparsers.add_parser("all", help="run the full pipeline").set_defaults(func=cmd_all)

    score = subparsers.add_parser("score", help="score a CSV with the saved model")
    score.add_argument("--input", required=True)
    score.add_argument("--output", default="predictions.csv")
    score.add_argument("--sep", default=";")
    score.set_defaults(func=cmd_score)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except Exception as exc:  # noqa: BLE001 - clean CLI failure
        print(f"\n[error] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
