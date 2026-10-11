# -Depression_Severity_Paper
# Depression Severity Classification from Text

**Benchmarking Classical Machine Learning for Four-Level Depression Severity Classification from Text: A Leakage-Controlled Comparison of Bag-of-Words and TF-IDF Represen
tations**

Raghav Vashisht, Aanandit Kanwar, Apurav Gautam
Department of Computer Science and Engineering, Chitkara Institute of Engineering (CUIET)

📄 Paper: [`Depression_Severity_Paper.pdf`](Depression_Severity_Paper.pdf) · LaTeX source: [`paper/`](paper/)

---

## Overview

This project benchmarks three classical machine-learning classifiers — **Naive Bayes**, **Logistic Regression** and **Random Forest** — with two text representations — **
Bag-of-Words** (Experiment A) and **TF-IDF** (Experiment B) — for classifying text into four depression severity levels: **Minimal, Mild, Moderate, Severe**.

Before modelling, the dataset is audited for duplicate posts and posts with contradictory labels, which would otherwise leak between training and test sets.

> ⚠️ This is a research study. The models are **not** diagnostic tools. Dataset labels are text annotations, not clinical diagnoses.

## Key results

Test set: 5,866 posts (20% stratified hold-out). Hyper-parameters tuned with exhaustive grid search and stratified 5-fold cross-validation on the training set only.

| Features | Model | Accuracy | Macro-F1 | Severe recall | Train time |
|---|---|---:|---:|---:|---:|
| TF-IDF | Logistic Regression | **0.691** | **0.661** | 0.677 | 107 s |
| BoW | Naive Bayes | 0.680 | 0.652 | **0.703** | 1.7 s |
| BoW | Random Forest | 0.656 | 0.609 | 0.623 | 50 s |
| BoW | Logistic Regression | 0.640 | 0.603 | 0.602 | 232 s |
| TF-IDF | Naive Bayes | 0.663 | 0.602 | 0.690 | 1.8 s |
| TF-IDF | Random Forest | 0.655 | 0.598 | 0.624 | 77 s |

**Findings**

- **Logistic Regression + TF-IDF** has the best overall performance; **Naive Bayes + BoW** detects the most Severe posts and trains ~60× faster.
- **Data audit:** 29.9% of the raw dataset rows were duplicates or carried contradictory labels. After cleaning, the Moderate class shrank by 81.6% — the dataset's appare
nt class balance came from duplicated rows.
- **Markers of Severe predictions:** suicidal-ideation terms (*suicide, suicidal, die, attempt, kill myself*) plus the pronoun *i* and negation *cannot* — words that stan
dard stop-word lists remove.
- All models plateau near 0.60–0.66 macro-F1, pointing to label consists the main limit.

## Dataset

[Depression Severity Levels Dataset](https://github.com/KUAS-ubicomp-laels_Dataset) (KUAS Ubiquitous Computing Lab) — 41,873 posts, four BDI-alig
ned severity labels. Downloaded automatically on first run.

| Audit step | Rows removed | Rows remaining |
|---|---:|---:|
| Raw corpus | — | 41,873 |
| Empty text | 14 | 41,859 |
| Conflicting labels (3,053 texts, all copies) | 6,305 | 35,554 |
| Exact duplicates | 6,194 | 29,360 |
| Empty after preprocessing | 32 | **29,328** |

## Method

1. **Audit** — drop empty texts, every copy of a text with conflicting labels, and exact duplicates.
2. **Preprocess** — lowercase, remove URLs and mentions, expand negativcan not*), strip non-letters, remove stop words (keeping pronouns, negatio
ns and words like *alone, empty, cry*), WordNet lemmatisation.
3. **Features** — unigrams + bigrams, min df 5, max df 0.90, top 20,000IDF. The vectoriser is fitted inside each CV fold (no leakage).
4. **Models and grids**

   | Model | Search space | Configs |
   |---|---|---:|
   | Naive Bayes | Multinomial / Complement × α ∈ {0.1, 0.5, 1.0} | 6 |
   | Logistic Regression | C ∈ {0.1, 1, 10} × L1 ratio ∈ {1.0, 0.5}; SA|
   | Random Forest | trees ∈ {100, 300, 500} × depth ∈ {20, 50, 100} × min split ∈ {2, 5, 10}; balanced weights | 27 |

5. **Evaluate** — 80/20 stratified split (seed 42); best config by mean CV macro-F1, refitted and tested once. Metrics: accuracy, macro precision / recall / F1, Severe-cl
ass recall, train and inference time.

## Project structure

```
├── Depression_Severity_Paper.pdf   final paper
├── run_all.sh                      runs the full pipeline end to end
├── requirements.txt
├── src/
│   ├── config.py        paths, seed, class names, settings
│   ├── data.py          download, label mapping, audit and de-duplicat
│   ├── preprocess.py    text cleaning, custom stop list, lemmatisation
│   ├── models.py        vectorisers, classifiers, hyper-parameter grid
│   ├── evaluate.py      metrics, confusion matrices, top features
│   ├── plots.py         publication figures
│   ├── funnel.py        main runner (one or more models per run)
│   └── report.py        merges runs into tables, figures and LaTeX sni
├── tests/               30 unit tests (pytest)
├── paper/               LaTeX source (main.tex, refs.bib, figures/)
├── data/                raw and cleaned dataset (created on first run)
├── results/             metrics, confusion matrices, CV results, figur
└── results_with_duplicates/   sensitivity run on the uncleaned corpus
```

## Setup

Requires Python 3.9+.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m nltk.downloader wordnet omw-1.4
```

## Usage

Run everything (NB first, then RF and LR in parallel, then the sensitiv

```bash
./run_all.sh
```

Or run pieces separately:

```bash
.venv/bin/python -m src.funnel --models nb                 # Naive Bayes, both feature sets (~1 min)
.venv/bin/python -m src.funnel --models rf --n-jobs 4      # Random For
.venv/bin/python -m src.funnel --models lr --n-jobs 4      # Logistic Regression (~80 min)
.venv/bin/python -m src.funnel --models nb --keep-duplicates --out resu
.venv/bin/python -m src.report                             # merge results → tables, figures, LaTeX
```

Useful options for `src.funnel`:

| Option | Meaning |
|---|---|
| `--models nb lr rf` | which models to run |
| `--experiments bow tfidf` | which feature sets to run |
| `--search random --n-iter 10` | randomized instead of exhaustive sear
| `--sample 3000` | quick smoke run on a stratified subsample |
| `--keep-duplicates` | use the corpus as distributed (no audit) |

Run times above are from an 8-core, 8 GB laptop and will vary.

Run the tests:

```bash
.venv/bin/python -m pytest -q
```

## Outputs

| File | Contents |
|---|---|
| `results/metrics.csv` | one row per model × feature set |
| `results/confusion_<model>_<features>.csv` | row-normalised confusion matrices |
| `results/cv_results_<model>_<features>.csv` | every grid configuratio
| `results/classification_reports.txt` | per-class precision / recall / F1 |
| `results/lr_top_coefficients_tfidf.csv` | strongest Severe-class LR c
| `results/rf_top_importances_tfidf.csv` | top Random Forest Gini importances |
| `results/figures/` | all figures (PNG + PDF) |

## Building the paper

```bash
cd paper
tectonic -X compile main.tex     # or upload paper/ to Overleaf
```

## Limitations

- Labels are text annotations aligned to BDI bands, not clinical diagnoses.
- Only negative contractions are expanded, so *I'm* / *I've* leave fragvocabulary.
- Logistic Regression is early-stopped at 100 SAGA epochs; the Random Forest optimum sits at the edge of its grid.
- Single train/test split, no significance tests; the duplicate sensities only.
- Experiment C (TF-IDF + DSAS clinical lexicon) is future work.

## Acknowledgment

Thanks to the KUAS Ubiquitous Computing Lab for releasing the Depression Severity Levels Dataset.
