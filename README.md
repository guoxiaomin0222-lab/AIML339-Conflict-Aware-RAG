# AIML339 — BERT-Based Conflict Classification for RAG

Code and results for the AIML339 final project, **Evaluating BERT-Based Conflict Type Classification for Improving RAG Answer Quality**, at Victoria University of Wellington.

## Project overview

Retrieved documents can agree, complement one another, or contain conflicting information. This project investigates whether a supervised BERT conflict classifier can guide retrieval-augmented generation (RAG) answers.

The pipeline is:

1. **Conflict classification:** Compare four BERT document-aggregation architectures with TF–IDF logistic regression and a prompted GPT-4o classifier.
2. **Deterministic routing:** Convert a predicted conflict label into a fixed, type-specific instruction.
3. **RAG generation:** Use the same GPT-4o generator and retrieved evidence under four conditions: Standard, LLM-guided, BERT-guided, and Oracle (gold label).
4. **Evaluation:** Compare Expected Behavior, Answer Recall, and Factual Grounding.


## Dataset

The project uses the [CONFLICTS benchmark](https://github.com/google-research-datasets/rag_conflicts), with five categories: No Conflict, Complementary Information, Conflicting Opinions and Research Outcomes, Outdated Information, and Misinformation.

- **Original data:** 458 expert-labelled examples.
- **Synthetic training additions:** 110 Misinformation + 53 Outdated examples. GPT-based tools assisted generation; Claude reviewed quality and labels, with partial manual spot-checking.
- **Combined dataset:** 621 examples.
- **Fixed split (seed 42):** 481 training examples (318 original + 163 synthetic), 68 validation examples (original only), and 72 test examples (original only).

Files in `data/` contain the two synthetic datasets and the fixed split manifest. The original 458-example CONFLICTS JSONL file is **not included**; obtain it from the source repository and retain its original record order. `00_dataset_preparation.ipynb` builds `conflicts_621.jsonl`. The prepared dataset and split manifest are checked by the later notebooks.

The classifiers use questions, document titles, and short texts, without a separate source-date field. For answer generation, available original source dates are restored as evidence metadata.

## Repository structure

```text
AIML339-Conflict-Aware-RAG/
├── data/
│   ├── conflicts_original_eval_split.json
│   ├── misinformation_110.jsonl
│   └── outdated_53.jsonl
├── notebooks/
│   ├── 00_dataset_preparation.ipynb
│   ├── 01a-bert-full.ipynb
│   ├── 01a1-bert-mean-only.ipynb
│   ├── 01a2-bert-interaction-only.ipynb
│   ├── 01a3-bert-spread-only.ipynb
│   ├── 01B_tfidf_logistic_regression.ipynb
│   ├── 01C_prompted_llm.ipynb
│   ├── 02_rag_generation.ipynb
│   └── 03_rag_answer_evaluation.ipynb
├── results/
│   ├── 01a-results/            # Full BERT
│   ├── 01a1-results/           # Mean-only
│   ├── 01a2-results/           # Interaction-only
│   ├── 01a3-results/           # Spread-only
│   └── *.csv                   # Classification and RAG results
└── README.md
```

The BERT variants differ in whether they use an additional document-interaction Transformer and/or a spread feature (per-dimension max minus min). The selected model is **spread-only BERT**, seed **43**, based on validation Macro-F1 across the four adequately represented classes.

## Requirements and setup

Python, Jupyter, `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `torch`, `transformers`, and `openai` are used. BERT is based on `bert-base-uncased`. GPU acceleration is recommended for training. GPT-4o API stages require an API key and may incur charges.

```bash
python -m pip install jupyterlab numpy pandas matplotlib scikit-learn torch transformers openai
```

**Important:** The notebooks currently look for input files in their **current working directory**. The GitHub folder layout is for organization; to run the notebooks unchanged, first prepare a local working folder. From the repository root:

```bash
mkdir -p workspace
cp notebooks/*.ipynb workspace/
cp data/*.jsonl data/*.json workspace/
# Also copy the original CONFLICTS JSONL into workspace/
# and name it conflicts_458.jsonl.
cd workspace
jupyter lab
```

Run notebooks in this order:

1. `00_dataset_preparation.ipynb` — Merge and validate data, create `conflicts_621.jsonl`, and produce the fixed split.
2. `01a-bert-full.ipynb`, `01a1-bert-mean-only.ipynb`, `01a2-bert-interaction-only.ipynb`, `01a3-bert-spread-only.ipynb` — Train and evaluate BERT variants. Training uses seeds **42, 43, 44**, learning rate **1e-5**, up to **8 epochs**, batch size **1**, gradient accumulation **4**, and early-stopping patience **3**.
3. `01B_tfidf_logistic_regression.ipynb` — Run the lexical classification baseline.
4. `01C_prompted_llm.ipynb` — Run the prompted GPT-4o classification baseline.
5. `02_rag_generation.ipynb` — Generate four RAG conditions. The notebook requires the original CONFLICTS file and the validation-selected spread-only BERT checkpoint (`best_interaction_0_spread_1_original_eval_seed43.pt`). Check its seed-selection and checkpoint-path settings before running. If the checkpoint is not supplied, retrain it using the spread-only notebook.
6. `03_rag_answer_evaluation.ipynb` — Judge generated answers and export per-condition, per-class, classifier, and paired-comparison results. It requires Stage 02 output and the original CONFLICTS file.

For Stages 01C, 02, and 03, check API settings before executing. Do not put an API key in the repository. The Stage 02/03 notebooks are configured for the final-test workflow; validation settings should be used for development rather than tuning on the test set.

## Main Results

The following results were obtained on the **72 original test examples**.

| Conflict classifier | Accuracy | Notebook |
| --- | ---: | --- |
| TF–IDF + logistic regression | 43.06% | `01B_tfidf_logistic_regression.ipynb` |
| Prompted GPT-4o | 52.78% | `01C_prompted_llm.ipynb` |
| Spread-only BERT (seed 43) | **63.89%** | `01a3-bert-spread-only.ipynb` |

| RAG condition | Expected Behavior |
| --- | ---: |
| Standard | 61.11% |
| LLM-guided | 73.61% |
| BERT-guided | 75.00% |
| Oracle | 79.17% |

The four RAG conditions are implemented in `02_rag_generation.ipynb` and evaluated in `03_rag_answer_evaluation.ipynb`.

The corresponding result files include `results/rag_test_condition_summary.csv`, `results/rag_test_per_class_summary.csv`, `results/rag_test_classifier_predictions.csv`, and `results/rag_test_paired_differences.csv`. Results from the four BERT aggregation architectures are stored in the respective `results/01a*-results/` folders.

For detailed experimental analysis and limitations, refer to the final report.

## Reproducibility Notes

- The original CONFLICTS dataset is not included in this repository and must be obtained from the [official source](https://github.com/google-research-datasets/rag_conflicts).
- Reproducing BERT-guided RAG requires training the spread-only model with seed 43 and loading the resulting checkpoint.
- Some notebooks use relative file paths and require the local working directory to be configured correctly (see **Requirements and setup**).
- GPT-4o classification, generation, and evaluation require an API key and may incur API costs. Results may vary across repeated API calls.
