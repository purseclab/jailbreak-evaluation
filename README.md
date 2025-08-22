<p align="center">
  <img src="logo.png">
</p>

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](./LICENSE)
![GitHub stars](https://img.shields.io/github/stars/purseclab/jailbreak-evaluation.svg)
![GitHub forks](https://img.shields.io/github/forks/purseclab/jailbreak-evaluation.svg)


# jailbreak-evaluation

The jailbreak-evaluation is an easy-to-use Python package for language model jailbreak evaluation.
The jailbreak-evaluation is designed for comprehensive and accurate evaluation of language model jailbreak attempts.
Currently, jailbreak-evaluation support evaluating a language model jailbreak attempt on multiple metrics.

This is the official research repository for "[Rethinking How to Evaluate Language Model Jailbreak](https://arxiv.org/abs/2404.06407)", by Hongyu Cai, Arjun Arunasalam, Leo Y. Lin, Antonio Bianchi, and Z. Berkay Celik.

## Quick start
- Python 3.10+ recommended.
- Install base deps:
    - Optional: create a virtual env.
    - Install requirements with pip ( Windows PowerShell example ):
        - Optional commands to run:
            ```powershell
            pip install -r requirements.txt
            ```

Some evaluators require extra packages and a GPU. Install them only if you plan to use those evaluators:
- Optional packages (GPU/LLM evaluators):
    - transformers, torch, fschat, matplotlib, tqdm
    - Optional commands to run:
        ```powershell
        pip install transformers torch fschat matplotlib tqdm
        ```

## Environment variables
Create a `.env` file in the repo root (or set system env vars). Required for DB and some judges.

- MongoDB (required for all flows)
    - `MONGODB_USERNAME`
    - `MONGODB_PASSWORD`
    - `MONGODB_ENDPOINT` (Atlas cluster host, no protocol)

- OpenAI (required for Chao/StrongReject judges and `core/apply.py` default judge)
    - `OPENAI_API_KEY`

- Google Cloud Translate (optional; used by manual labeling to show translated responses)
    - `GOOGLE_APPLICATION_CREDENTIALS` pointing to a service account JSON with Translate API enabled

Tip: NLTK is used for sentence tokenization. If needed, download punkt once in Python: `import nltk; nltk.download('punkt')`.

## Repository layout (key parts)
- `src/core/`
    - `utils.py`: data model, MongoDB helpers.
    - `upload.py`: ingest CSVs (Princeton, Penn) and write to MongoDB.
    - `label.py`: manual/automatic labeling loop over DB items.
    - `download.py`: export labeled documents to CSV/JSON.
    - `apply.py`: multifaceted evaluation with an LM judge (document/paragraph/sentence levels) and report generation.
    - `count.py`, `count_apply.py`, `patch.py`, `converge.py`: maintenance and analysis helpers.
- `src/hypnotist/`
    - `evaluation.py`: judge implementations: Zou, Huang, Chao (OpenAI), HarmBench (HF), LlamaGuard (HF), StrongReject (OpenAI), DAN (ChatGLM-6B), Manual.
    - `language_models.py`: simple OpenAI cache helper.

## Data model: `DataPoint` (in `src/core/utils.py`)
Fields stored per record:
- intent, prompt, response
- model_id, publication_id, dataset_id, dataset_version (0: original, 1: add period)
- publication_report_jailbreak_label: bool
- publication_report_jailbreak_label_confidence: Optional[float]

Validation conventions (update these lists if you extend the schema):
- models: `["Llama-2-7b-chat-hf", "gpt-3.5-turbo"]`
- publications/datasets (ids share the same index space):
    - 0: Universal and Transferable Adversarial Attacks on Aligned Language Models
    - 1: Catastrophic Jailbreak of Open-source LLMs via Exploiting Generation
    - 2: Jailbreaking Black Box Large Language Models in Twenty Queries

If you add models or publications, modify `DataPoint` and its `__post_init__` accordingly.

## Typical workflows

### 1) Ingest datasets into MongoDB
Princeton CSV (expects columns: `Input`, `Output`, `Jailbreak`, `Score`):
- Optional commands to run:
    ```powershell
    python .\src\core\upload.py princeton <input_csv> <model_id> <dataset_id> <dataset_version>
    ```

Penn CSV (expects columns: `original_goal`, `prompt`, `target_response`, `is_jailbroken`, `judge_scores`):
- Optional commands to run:
    ```powershell
    python .\src\core\upload.py penn <input_csv> <model_id> <dataset_id> <dataset_version>
    ```

### 2) Label examples (manual or automatic)
CLI entry: `src/core/label.py`
- Arguments:
    - `--mode`: `manual` or `automatic`
    - `--name`:
        - manual: `hongyu`
        - automatic: `zou`, `huang`, `chao`, `harmbench`, `llamaguard`, `strongreject`, `dan`
    - `--topic`: one of `safeguard_violation`, `informativeness`, `relative_truthfulness`, `none` (for automatic judges that produce a single decision)
    - `--publication_id`, `--dataset_id`, `--dataset_version`, `--model_id`: integers per the schema

Notes:
- Labels are written back to MongoDB under fields like `manual_hongyu_<topic>_label` or `automatic_<name>_<topic|none>_label`.
- Some evaluators require GPUs and model downloads:
    - `llamaguard`: meta-llama/Llama-Guard-3-8B (CUDA required)
    - `harmbench`: cais/HarmBench-Llama-2-13b-cls (CUDA required)
    - `dan`: THUDM/chatglm-6b (CUDA required)
    - `chao` and `strongreject`: OpenAI models (set `OPENAI_API_KEY`)

### 3) Export labeled data for analysis
Run `src/core/download.py` to write `all_data.csv` and `all_data.json` with core fields and labels present in DB.
- Optional commands to run:
    ```powershell
    python .\src\core\download.py
    ```

### 4) Multifaceted evaluation and figures
`src/core/apply.py` runs a hierarchical, multifaceted judge (document/paragraph/sentence) using `gpt-4o-mini` by default with caching.
- Requirements: `OPENAI_API_KEY`, NLTK punkt, and optional GPU if switching to local HF models.
- Outputs: cached completions (`cache_*.pkl`), a CSV/PKL with automatic labels, and printed metrics; figures saved under `figure/<model>/`.
- Optional commands to run:
    ```powershell
    python .\src\core\apply.py
    ```

Interpreting outputs and levels:
- DL/PL/SL: document/paragraph/sentence-level labels per metric (SV, I, RT).
- JL: joint OR across levels; CL: ensemble with SV from JL and I/RT from PL.
- Aggregated results (accuracy/F1/precision/recall) are printed.

### 5) Reproduce table metrics
`src/experiment/table_2.py` prints aggregated metrics for selected runs using labels stored in MongoDB.
- Optional commands to run:
    ```powershell
    python .\src\experiment\table_2.py
    ```

## Requirements
Base requirements are listed in `requirements.txt`:
- pymongo, python-dotenv, pandas, numpy, openai, typer, rich, google-cloud-translate, scikit-learn, seaborn, nltk, tenacity, pyarrow

Optional (for certain evaluators/plots):
- transformers, torch (GPU strongly recommended), fschat, matplotlib, tqdm

## Troubleshooting
- Mongo connection: verify `.env` has valid Atlas credentials and IP allowlist; the client pings admin on startup.
- OpenAI limits/errors: set `OPENAI_API_KEY`; the code retries with backoff in some paths; costs may apply.
- NLTK: if you see tokenizer errors, run `import nltk; nltk.download('punkt')` once.
- GPU models: ensure CUDA is available for LlamaGuard, HarmBench, and ChatGLM.
- Schema assertions: if you extend models/publications, update `DataPoint.__post_init__` lists.

## Notes
- We use shared id spaces for `publication_id` and `dataset_id`. Keep them consistent across ingestion and labeling.
- When adding new models or publications, update the lists and validations in `src/core/utils.py`.


## Contributing

Please let us know if you encounter a bug or have any suggestions by [filing an issue](https://github.com/purseclab/jailbreak-evaluation/issues).
We welcome all contributions from bug fixes to new features and extensions.
We expect all contributions discussed in the issue tracker and going through PRs.

## Cite

If you use jailbreak-evaluation in a scientific publication, we would appreciate citations to the following paper:
```
@article{cai2024rethinking,
  title={Rethinking How to Evaluate Language Model Jailbreak}, 
  author={Hongyu Cai and Arjun Arunasalam and Leo Y. Lin and Antonio Bianchi and Z. Berkay Celik},
  year={2025},
  journal={Proceedings of the 2025 Workshop on Artificial Intelligence and Security (AISec 2025)}
}
```

## The Team

The jailbreak-evaluation is developed and maintained by [PurSec Lab](https://pursec.cs.purdue.edu/).
