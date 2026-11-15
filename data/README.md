# Data provenance

The raw files are **not** committed. `python run.py download` fetches the archive from
UCI and caches it in `data/raw/` (git-ignored). This file records exactly what was used.

## Source

- Dataset: **Student Performance**, UCI Machine Learning Repository, ID 320.
- Original paper: P. Cortez and A. Silva (2008), *Using Data Mining to Predict Secondary
  School Student Performance*, AEPIA.
- Two Portuguese secondary schools, 2005–2006 academic year.
- URL: <https://archive.ics.uci.edu/ml/machine-learning-databases/00320/>

## Files

| File | Rows | Bytes | SHA-256 |
| --- | ---: | ---: | --- |
| `student.zip` | — | 20,478 | `4f671ae4598c20bb4e64de0f65931c98a609a52e383604e2601bd8f7e3822427` |
| `student-mat.csv` | 395 | 56,993 | `e47f9ee225e1ee6e69b7564e6dac7123e80b8486677fe111f351964cef5dec80` |
| `student-por.csv` | 649 | 93,220 | `a7594a11d7771c0efe1a740824e0e833da9c4cad07c39a9766a874575563fb3f` |

Combined: **1,044** student-subject records. The documented pairing yields **382**
students who appear in both files.

## Quirks the loader handles

1. **Semicolon-separated, quoted strings.** `pd.read_csv(sep=";")` removes the quotes;
   column names are stripped.
2. **Two files, one schema.** `mat` and `por` share all 33 columns; the `subject`
   column is added by the loader so a pooled model can learn the level shift.
3. **G3 is 0–20 and 0 is real.** 53 records end at exactly 0 — students who dropped out
   or missed the assessment. They are kept, counted and reported, never silently
   replaced.
4. **The pairing keys are documented, not arbitrary.** `student-merge.R` inside the
   archive pairs on 13 attributes and reports 382 students. Joining on all 30 shared
   non-grade columns looks safer but returns only **39** rows, because questionnaire
   items legitimately change between school years.
5. **Duplicate rows are genuine.** 16 rows repeat every attribute and 16 repeat the
   pairing keys; two students can share a demographic sheet, so they are kept.
6. **Grades leak.** G1 and G2 are strong predictors of G3 but only exist late in the
   course; the pipeline fits three explicit scenarios instead of quietly choosing the
   best-looking one.

## Licence / terms

The dataset is distributed by UCI for research use; see the repository page for the
current terms. The code in this repository is MIT licensed.
