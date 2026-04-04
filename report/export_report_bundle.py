# report_outputs/에 holdout 표·md·json·RUN_INFO 생성.
# python report/export_report_bundle.py [--csv path]
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS = _ROOT / "scripts"
for p in (_ROOT, _SCRIPTS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from fds_pipeline import CSV_DEFAULT
from model_comparison import run_benchmark


def _df_to_markdown(df) -> str:
    # DataFrame → 마크다운 표.
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                if v != v:  # NaN
                    cells.append("")
                else:
                    cells.append(f"{v:.6f}" if abs(v) < 1e6 else str(v))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="report_outputs/ 생성")
    parser.add_argument("--csv", type=Path, default=CSV_DEFAULT)
    args = parser.parse_args()

    out_dir = _ROOT / "report_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    df, meta = run_benchmark(args.csv)

    csv_path = out_dir / "holdout_benchmark.csv"
    md_path = out_dir / "holdout_benchmark.md"
    meta_path = out_dir / "benchmark_meta.json"
    info_path = out_dir / "RUN_INFO.txt"

    df.to_csv(csv_path, index=False)

    md_header = (
        "# Hold-out (임계값 0.5)\n\n"
        "- Train: SMOTETomek, fds_pipeline 분할·seed 동일.\n"
        "- Test: 미증식.\n"
        "- 튜닝 행 CV: F2, RandomizedSearchCV.\n\n"
    )
    md_path.write_text(md_header + _df_to_markdown(df) + "\n", encoding="utf-8")

    meta["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    meta_path.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    info_path.write_text(
        f"생성 시각 (로컬): {datetime.now().isoformat(timespec='seconds')}\n"
        f"데이터: {meta['csv_path']}\n",
        encoding="utf-8",
    )

    print(f"\nOK: report_outputs/\n  - {csv_path.name}\n  - {md_path.name}\n  - {meta_path.name}\n  - {info_path.name}\n")


if __name__ == "__main__":
    main()
