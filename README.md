# IncidentInsight 🔍

**ML-Based Application Log Analysis and Incident Intelligence**

IncidentInsight is a Python-based application log analysis project that helps developers identify recurring errors, group similar log messages, prioritize incidents, and understand potential causes through machine learning and optional LLM-assisted explanations.

Instead of manually inspecting thousands of log lines, IncidentInsight processes logs through a structured pipeline and presents useful findings through an interactive Streamlit dashboard and a command-line workflow.

## Overview

Application logs contain valuable information about failures, timeouts, authentication problems, database errors, and other operational issues. However, large log files can be difficult to analyze manually, particularly when similar errors appear repeatedly.

IncidentInsight combines log processing, text-based clustering, incident ranking, and explanation utilities to make log analysis more structured and accessible.

### Key Features

- **Log parsing:** Extracts structured information from supported application log formats.
- **Log normalization:** Standardizes variable message content to help identify recurring patterns.
- **Machine learning:** Uses TF-IDF text features and agglomerative clustering to group similar log messages.
- **Incident ranking:** Ranks clusters using frequency and severity-based scoring.
- **Cluster evaluation:** Provides tools for comparing clustering results with labeled ground-truth data.
- **Incident explanations:** Generates structured explanations and suggested troubleshooting steps.
- **Interactive dashboard:** Explores incident summaries, severity distributions, cluster details, and log entries.
- **Incident-aware chatbot:** Allows users to ask questions about analyzed incident information.
- **Optional LLM integration:** Supports hosted-LLM assistance when configured, with offline or deterministic fallback behavior where implemented.
- **Automated tests:** Includes unit and integration tests for core processing, evaluation, ranking, and workflow behavior.

## Architecture

```text
PRODUCTION PIPELINE — src/pipeline.py

  Application Log File
          |
          v
     Log Parsing
          |
          v
   Log Normalization
          |
          v
   TF-IDF Feature Extraction   (performed inside the clustering stage)
          |
          v
   Agglomerative Clustering
          |
          v
     Incident Ranking
          |
          +----------------------------+
          |                            |
          v                            v
   Command line                 Workflow — src/workflow.py
   run.py                              |
   prints the ranked                   v
   incident clusters            Incident Explanations
                                (deterministic, optional LLM)
                                       |
                                       v
                                Streamlit Dashboard
                                and Incident Chatbot — app.py


SEPARATE EVALUATION — NOT part of the production pipeline

   Clustered output   +   ground_truth.csv   (aligned by line_id)
          |
          v
   src/evaluation.py and src/ranking_evaluation.py
   ARI, AMI, cluster composition, cluster purity, ranking coverage
```

Evaluation is deliberately kept out of the production pipeline: `src/pipeline.py` states that ground-truth comparison "is deliberately NOT part of the production pipeline" and that the pipeline never requires ground-truth labels to run. The evaluation utilities are invoked separately, and only when labeled data is available.

The command-line workflow (`run.py`) runs the analysis pipeline and prints ranked clusters. It does not generate explanations, call the LLM, or use the dashboard.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core implementation |
| Pandas | Structured log data processing |
| NumPy | Numerical operations |
| SciPy | Scientific computing and clustering-related support |
| Scikit-learn | TF-IDF features, clustering, and evaluation metrics |
| Streamlit | Interactive dashboard |
| Pytest | Automated testing |
| Hosted LLM API | Optional AI-assisted explanations and chatbot responses |

See [`requirements.txt`](requirements.txt) for the declared dependencies.

## Project Structure

```text
IncidentInsight/
├── app.py
├── run.py
├── data_generator.py
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
├── data/
│   ├── raw/
│   ├── processed/
│   └── baseline/
│       └── 7126c46/
│           ├── synthetic_logs.log
│           ├── ground_truth.csv
│           ├── generation_meta.json
│           └── MANIFEST.md
├── src/
│   ├── config.py
│   ├── log_parser.py
│   ├── log_normalizer.py
│   ├── feature_extractor.py
│   ├── clusterer.py
│   ├── ranker.py
│   ├── evaluation.py
│   ├── ranking_evaluation.py
│   ├── incident_explainer.py
│   ├── chatbot.py
│   ├── llm_explainer.py
│   ├── workflow.py
│   └── ...
└── tests/
```

*The tree highlights the main project components; consult the repository for the complete file list.*

## Getting Started

### Prerequisites

- Python 3.13 or a compatible Python version supported by the declared dependencies.
- Git.
- A Windows terminal such as PowerShell or Command Prompt.

The commands below assume that you have cloned the repository and opened a terminal in its root directory.

### 1. Create a virtual environment

```powershell
python -m venv venv
```

### 2. Activate the environment

**PowerShell:**

```powershell
.\venv\Scripts\Activate.ps1
```

**Command Prompt:**

```bat
venv\Scripts\activate.bat
```

If PowerShell blocks activation because of its execution policy, use Command Prompt or follow your organization's approved Python environment guidance.

### 3. Install dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Configure optional LLM access

Review [`.env.example`](.env.example) for the environment variables supported by the application.

The application reads configuration from environment variables. Creating a `.env` file alone does not load variables automatically; no `python-dotenv` dependency is assumed.

LLM credentials are optional. Use the application's available offline or fallback functionality when a hosted service is not configured or available.

**Security:** Never commit real API keys, tokens, passwords, or populated secret files to GitHub.

## Dataset and Data Setup

### Historical baseline

The repository preserves a historical dataset in:

```text
data/baseline/7126c46/
```

It contains:

- `synthetic_logs.log` — 1,000 historical log entries.
- `ground_truth.csv` — corresponding labeled records.
- `generation_meta.json` — historical generation metadata.
- `MANIFEST.md` — provenance, integrity hashes, and verification details.

The historical log file contains 1,000 lines, and the ground-truth file contains 1,000 rows aligned by `line_id`, covering eight incident categories.

**Important:** The preserved historical dataset and the current `data_generator.py` output are not equivalent. The current generator produces a different dataset and does not reproduce the historical logs or labels byte-for-byte.

The metadata also contains category and severity counts that conflict with the actual historical files. These values have been preserved rather than silently rewritten. Consult the baseline manifest for the verified file hashes, observed counts, and provenance details.

### Line endings and hash verification

The repository's `.gitattributes` sets `* text=auto eol=lf`. The preserved baseline files currently have Windows `CRLF` line endings on disk, so Git normalizes them to `LF` when it stores them, and a fresh clone checks out the `LF` form.

Two sets of SHA-256 values are therefore both correct, for different forms of the same content:

- **Byte-level (CRLF)** — matches the files exactly as preserved on disk, and the maintainer's external backup.
- **LF-normalized** — matches what a clone checks out after Git's normalization.

Git **blob IDs** are a different identifier again: they are SHA-1 hashes of blob content, not SHA-256 hashes of a file. `git hash-object --no-filters` reports the blob ID of the on-disk CRLF bytes, while `git hash-object` with filters applied reports the blob ID of the LF form Git would store. The manifest records both pairs.

A hash that differs after cloning is expected, not corruption. Log content is unaffected: the parser reads lines with `splitlines()`, which treats CRLF and LF identically.

For the recorded values and read-only verification commands, see sections 2 and 3 of [`data/baseline/7126c46/MANIFEST.md`](data/baseline/7126c46/MANIFEST.md), which remains the provenance reference.

### Running the default workflow

The standard workflow expects the raw log at:

```text
data/raw/synthetic_logs.log
```

The evaluation tests that use labeled data also expect:

```text
data/processed/ground_truth.csv
```

These canonical data files are ignored by Git and are not automatically populated by cloning the repository. Consequently, a fresh clone cannot run the default analysis workflow until the required input data is supplied.

#### Using the preserved baseline as input

To run the default workflow against the historical baseline, copy the two required files to the canonical paths. Run these PowerShell commands from the repository root. They copy **only when the destination does not already exist**, so they cannot overwrite data you already have:

```powershell
if (-not (Test-Path data\raw\synthetic_logs.log)) {
    Copy-Item data\baseline\7126c46\synthetic_logs.log data\raw\synthetic_logs.log
}
if (-not (Test-Path data\processed\ground_truth.csv)) {
    Copy-Item data\baseline\7126c46\ground_truth.csv data\processed\ground_truth.csv
}
```

If a file already exists, its copy is skipped. Inspect the existing file before deciding to replace it deliberately.

Both destinations are ignored by Git, so these copies never enter a commit. The baseline directory is not read automatically by the workflow or by the data-dependent test gates — copying the files to the canonical paths is what makes the baseline usable.

The preserved `generation_meta.json` is provenance metadata only. No pipeline stage or test reads it, so it does not need to be copied.

**Do not run `data_generator.py` to populate these paths.** It generates a different dataset and can overwrite existing files at the canonical locations.

### Generating a fresh dataset

The repository includes a data generator:

```powershell
python data_generator.py
```

**Warning:** Review the generator's implementation and output paths before running this command. It can write to the canonical data locations and overwrite existing files. It generates a different dataset from the preserved historical baseline.

Do not run it over historical data you want to retain. Back up important files and verify the generator's configured output paths first.

The historical baseline under `data/baseline/7126c46/` is preserved separately and is not automatically used by the default workflow or the data-dependent test gates.

## Running IncidentInsight

Run these commands from the repository root with the virtual environment activated.

### Streamlit dashboard

```powershell
python -m streamlit run app.py
```

Open the local URL printed by Streamlit in your browser.

Provide supported log input through the interface. If the default input file is absent, the application reports an analysis failure rather than producing an incident analysis.

### Command-line workflow

```powershell
python run.py
```

The supported arguments, as defined by `build_parser()` in `run.py`:

| Argument | Type | Default | Purpose |
|---|---|---|---|
| `log_path` (positional, optional) | path | `data/raw/synthetic_logs.log` | Log file to analyze |
| `--threshold` | float | `0.5` | Clustering distance threshold |
| `--top` | int | `10` | Number of top-ranked clusters to display |
| `-h`, `--help` | flag | — | Show usage help (provided by `argparse`) |

Examples:

```powershell
python run.py
python run.py path\to\logs.log
python run.py --threshold 0.4 --top 15
python run.py --help
```

Exit codes: `0` on success, `1` when the log file is missing or the input is unusable, and `2` when `--top` is less than 1.

The CLI runs the analysis pipeline only. Incident explanations, optional LLM enhancement, and the chatbot are part of the Streamlit workflow, not `run.py`.

## Machine Learning Pipeline

### 1. Parsing

The parser extracts supported fields from log messages, such as timestamps, severity levels, service or request information when available, and the message text.

### 2. Normalization

Normalization reduces irrelevant differences between log messages, such as changing identifiers or other variable values, so related messages can be compared more effectively.

### 3. TF-IDF feature extraction

Term Frequency–Inverse Document Frequency (TF-IDF) transforms normalized text into numerical features. These features allow text similarity to be used in clustering.

### 4. Agglomerative clustering

Agglomerative clustering groups similar log messages into clusters. The resulting groups help reveal recurring patterns that may represent related incidents.

Cluster assignments depend on the data and clustering configuration. Cluster IDs should not be treated as permanent identifiers across independent runs.

### 5. Incident ranking

The ranking module combines frequency and severity-related information to prioritize clusters for inspection. Ranking helps developers focus on recurring or higher-severity patterns first.

### 6. Evaluation

The evaluation modules compare clustering results with labeled data and inspect cluster composition and ranking behavior.

## Evaluation Metrics

The project includes evaluation utilities for the following:

| Metric or utility | Purpose |
|---|---|
| Adjusted Rand Index (ARI) | Measures agreement between predicted clusters and reference labels, adjusted for chance |
| Adjusted Mutual Information (AMI) | Measures shared information between predicted clusters and reference categories, adjusted for chance |
| Cluster composition | Shows how reference categories are distributed within predicted clusters |
| Cluster purity | Measures the proportion of records belonging to the dominant reference category in each cluster |
| Ranking evaluation | Examines category coverage and composition among highly ranked clusters |
| Threshold comparison | Compares clustering behavior under different configuration thresholds |

These metrics measure different aspects of clustering quality. ARI and AMI are not accuracy percentages, and results can vary with clustering parameters and dataset characteristics.

The preserved baseline includes ground-truth labels for evaluation. Its results should not be interpreted as evidence of performance on every real-world logging environment.

## Running Tests

From the repository root, activate the virtual environment and run:

```powershell
python -m pytest
```

For more detailed output:

```powershell
python -m pytest -v
```

### Fresh-clone test limitations

Fourteen existing tests depend on generated data files that are ignored by Git. On a fresh clone, these tests are expected to skip when the required raw log is absent.

Seven of these tests also require the ground-truth CSV. Their existing skip condition checks only the raw log, so supplying the raw log without the CSV causes those seven tests to fail. Measured in the clean copy with only the raw log present: **7 failed, 675 passed** (exit code 1). All seven failures are caused by the missing `data/processed/ground_truth.csv`. Provision both files, as described above, to avoid this state.

Measured result for this release preparation, run from the repository root with the canonical data files present:

```powershell
python -m pytest -q
```

**682 passed, 0 failed, 0 skipped, in 18.27 seconds** (exit code 0). With the data present, the 14 data-dependent tests ran and passed instead of skipping.

An earlier report of "682 passed and 14 skipped" is internally inconsistent and was not reproduced: the suite collects 682 cases in total, so 14 skips would leave 668 passing.

Fresh-clone behavior has since been measured in an isolated clean copy of the release content, using a fresh virtual environment on Python 3.13.7 with `requirements.txt` installed unchanged:

- **Pristine copy, no input data present:** 668 passed, 14 skipped, 0 failed, in 18.49 seconds (exit code 0). The 14 data-dependent tests skipped, so a fresh clone passes without any dataset.
- **Both baseline input files provisioned as described above:** 682 passed, 0 skipped, 0 failed, in 15.68 seconds (exit code 0).

These are automated test results, not a manual browser test. The Streamlit application was confirmed to start successfully in the clean copy, and its interface behavior is exercised by the automated `AppTest` suite. No manual browser walkthrough was performed, and no screenshots were captured.

Run the tests in your own environment and record the actual output when validating a release.

## Current Limitations

- The default workflow requires input data at the configured canonical path.
- The current generator does not reproduce the preserved historical dataset.
- Historical generation metadata contains counts inconsistent with the accompanying data.
- Fourteen data-dependent tests require local data files and skip on a fresh clone when the raw log is missing.
- Clustering quality depends on text normalization, the dataset, and clustering parameters.
- Cluster IDs are not guaranteed to remain stable across independent runs.
- LLM-assisted behavior depends on external service configuration and availability; fallback behavior is available where implemented.
- Database persistence is implemented in parts of the codebase but is not wired into the standard CLI and Streamlit entry points.
- Redaction rules are not a comprehensive personally identifiable information (PII) detection system.
- The project does not currently provide continuous live monitoring, proactive production alerting, or guaranteed time-aware conversational analysis.

## Future Improvements

Potential future work includes:

- Reliable ingestion of continuously generated application logs.
- Proactive incident notifications.
- More robust incident identity across repeated analysis runs.
- Broader evaluation against independently sourced log datasets.
- Improved test-data provisioning for clean clones.
- Automated continuous integration and release validation.
- More comprehensive sensitive-data detection and redaction.

These are future directions, not claims about current functionality.

## Project Goal

IncidentInsight demonstrates how Python, natural language processing techniques, unsupervised machine learning, evaluation metrics, and an interactive application can be combined to make application log analysis more structured and useful for developers.

The long-term goal is to reduce the effort required to identify recurring failures and investigate potentially related incidents.

---

**Status:** Portfolio project under active development.

For dataset provenance and integrity details, see [`data/baseline/7126c46/MANIFEST.md`](data/baseline/7126c46/MANIFEST.md).