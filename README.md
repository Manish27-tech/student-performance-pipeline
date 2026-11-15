# student-performance-pipeline

End-to-end pipeline for the UCI Student Performance dataset. Predicts the final
grade (G3) and pass/fail (G3 >= 10) for secondary-school students, and quantifies
how much of the apparent predictability is just grade leakage.

## Headline results

Three leakage scenarios (same 38 features base, differing only in grade access):

| scenario | regression R2 | RMSE | AUC | F1 |
|---|---|---|---|---|
| no_grades (early, usable) | 0.224 | 3.39 | 0.766 | 0.861 |
| with_g1 (after period 1) | 0.843 | 1.53 | 0.943 | 0.948 |
| with_g1_g2 (after period 2) | 0.876 | 1.36 | 0.959 | 0.927 |

- Cross-subject transfer (trained on one subject, tested on the other) is weak:
  maths -> Portuguese R2 = -0.304, Portuguese -> maths R2 = 0.100. The
  "no_grades" model does not travel across subjects.

## Pairing maths vs Portuguese

- 13 documented merge keys -> 382 matched pairs (96.7% of maths, 58.9% of Portuguese).
- Naive merge on all 30 shared columns yields only 39 rows (the classic trap).
- Paired mean delta (Portuguese - maths) = +2.13 (p = 4.7e-17); 249 improved,
  95 worsened, 38 tied. Pass rate 66.8% (maths) vs 91.6% (Portuguese).

## Dataset

- student-mat.csv (395) + student-por.csv (649) = 1,044 records, 33 attributes.
- Pass rate 78.0%, mean G3 = 11.34.

## Leakage / honesty notes

- grade_trend (G2 - G1) is excluded from the no_grades scenario (it is a grade).
- Scenarios are fitted and reported separately; do not quote with_g1_g2 as an
  "early prediction" result.

## Reproduce

    python -m venv .venv
    .venv\Scripts\activate
    pip install -r requirements.txt
    python run.py all          # download -> merge -> EDA -> train
    python -m pytest          # 39 tests
    python -m ruff check src tests

## Layout

    src/student/   config, data, clean, features, merge, eda, model, report, cli
    tests/         39 tests
    reports/       figures (8), eda_summary.json, metrics.json, pairing.json, REPORT.md

License: MIT.
