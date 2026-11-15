# Model Card — Student Grade Early-Warning Model

## Overview
| Field | Value |
| --- | --- |
| Version | 0.1.0 |
| Task | Regression (G3, 0–20) and pass/fail classification (G3 ≥ 10) |
| Algorithm | `random_forest` |
| Trained | 2026-11-15T11:52:24+00:00 |
| Data | UCI Student Performance, 1,044 student-subject records |
| Features | No prior grades (early-warning setting) |

## Intended use
Teaching and portfolio analysis of regression and classification on small tabular data,
plus a worked example of grade leakage. **Not** for admissions, streaming, discipline or
any decision about a student. Predictions describe group patterns from two Portuguese
schools in 2005–2006.

## Performance
### Final grade (regression)
| Metric | Value |
| --- | --- |
| R² | 0.224 |
| RMSE | 3.39 |
| MAE | 2.44 |

### Pass/fail (classification)
| Metric | Value |
| --- | --- |
| ROC AUC | 0.766 |
| F1 | 0.861 |
| Precision | 0.846 |
| Recall | 0.877 |

## Leakage warning
| Scenario | R² | AUC |
| --- | --- | --- |
| no_grades | 0.224 | 0.766 |
| with_g1 | 0.843 | 0.943 |
| with_g1_g2 | 0.876 | 0.959 |

The high-R² variant requires grades that only exist late in the course. Deploying it
would leak the outcome and produce a model that cannot be used when help is still
possible.

## Limitations
- Two schools, two subjects, one academic year; no external validation.
- Grade distributions differ by subject; a pooled model needs the subject indicator.
- 0–20 grades are a coarse ordinal scale treated here as interval.
- No information on teaching quality, class size, or prior attainment before G1.

## Ethical considerations
- Risk scores must never be shown to students as forecasts of their ability.
- Any intervention should be supportive, optional and evaluated for differential impact
  across the family-background variables that drive the model.
