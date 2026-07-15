# Text-Summarizer

An end-to-end NLP project that builds a powerful abstractive text summarizer by fine-tuning a Transformer-based model (e.g., T5, BART, or PEGASUS). The system takes long documents, articles, or web content as input and generates concise, coherent summaries while preserving key information and meaning.

## Features

- End-to-end MLOps style project structure
- Modular and configurable pipeline
- Support for custom configuration via YAML files
- GPU-accelerated training using PyTorch

## Installation

### Prerequisites

- Python 3.11 or higher
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

### Clone the Repository

```bash
git clone https://github.com/your-username/Text-Summarizer.git
cd Text-Summarizer
```

### Install Dependencies + Editable Package

We strongly recommend using **uv** (faster and more reliable):

```bash
uv sync
```

This will:
- Create a virtual environment (`.venv`)
- Install all dependencies
- Install the project in **editable mode** (`-e .`)

> **Why do we install in editable mode (`-e .`)?**  
> It allows you to import the package (`from textSummarizer.xxx import ...`) while still being able to edit the source code directly. This is essential during development. Without it, you will get `ModuleNotFoundError` when trying to import modules from `textSummarizer`.

### GPU PyTorch Installation (Important for CUDA)

If you have an NVIDIA GPU and want to use CUDA acceleration, install the correct PyTorch version **after** running `uv sync`:

```bash
# For CUDA 13.2 (used in this project)
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu132

# Alternative: For CUDA 12.1 (more commonly used)
# uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

> **Note:** Choose the CUDA version according to your GPU and driver. You can check available versions at the [official PyTorch website](https://pytorch.org/get-started/locally/).

### Alternative: Using pip (without uv)

```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/Mac:
source .venv/bin/activate

pip install -r requirements.txt
pip install -e .
```

Then install GPU PyTorch using the command shown above.

## Project Structure

```
Text-Summarizer/
├── src/
│   └── textSummarizer/          # Main package (importable)
│       ├── components/
│       ├── config/
│       ├── constants/
│       ├── pipeline/
│       ├── utils/
│       └── __init__.py
├── research/                    # Jupyter notebooks for experimentation
├── params.yaml
├── config.yaml
├── setup.py
├── pyproject.toml
├── uv.lock
└── README.md
```

## Workflow

1. Update `config.yaml`
2. Update `params.yaml`
3. Update `entity`
4. Update the configuration manager in `src/config`
5. Update the components
6. Update the pipeline
7. Update `main.py`
8. Update `app.py` (if building API)

## Usage

After installation, you can run the notebooks inside the `research/` folder or execute the pipeline using:

```bash
uv run python main.py
```

## Development

If you are modifying or developing the code locally:

```bash
uv sync
```

This ensures the package remains importable (`from textSummarizer import ...`) while you make changes to the source code.

## Troubleshooting: Model Training Errors (Pegasus + Samsum Project)

This section documents the three main errors faced while setting up and running the ModelTrainer for fine-tuning `google/pegasus-cnn_dailymail` on the Samsum dataset, along with the exact fixes applied.

### Error 1: evaluation_strategy argument error

**Error Message:**
TypeError: TrainingArguments.__init__() got an unexpected keyword argument 'evaluation_strategy'

**Root Cause:**
In newer versions of the transformers library (≥ 4.46), the argument `evaluation_strategy` was deprecated and removed. It was replaced by `eval_strategy`.

**Fix Applied:**
Changed the argument name in TrainingArguments from `evaluation_strategy` to `eval_strategy`.

Example fix in code:
trainer_args = TrainingArguments(
    ...
    eval_strategy=self.config.eval_strategy,   # Changed from evaluation_strategy
    ...
)

### Error 2: Missing or outdated accelerate library

**Error Message:**
ImportError: Using the `Trainer` with `PyTorch` requires `accelerate>=1.1.0`

**Root Cause:**
Newer versions of transformers made `accelerate` a hard dependency for the Trainer class. The installed version was either missing or older than 1.1.0.

**Fix Applied:**
Run these two commands and restart the Jupyter kernel:

pip install --upgrade accelerate
pip install --upgrade "transformers[torch]"

### Error 3: tokenizer argument removed from Trainer

**Error Message:**
TypeError: Trainer.__init__() got an unexpected keyword argument 'tokenizer'

**Root Cause:**
In recent versions of transformers, the Trainer class no longer accepts the `tokenizer=` parameter. It was replaced by `processing_class=`.

**Fix Applied:**
Updated the Trainer initialization:

From:
trainer = Trainer(
    model=model_pegasus,
    args=trainer_args,
    tokenizer=tokenizer,
    data_collator=seq2seq_data_collator,
    train_dataset=dataset_samsum_pt["test"],
    eval_dataset=dataset_samsum_pt["validation"]
)

To:
trainer = Trainer(
    model=model_pegasus,
    args=trainer_args,
    processing_class=tokenizer,
    data_collator=seq2seq_data_collator,
    train_dataset=dataset_samsum_pt["test"],
    eval_dataset=dataset_samsum_pt["validation"]
)

### Additional Notes

- Model was sometimes loading from a PR branch (refs/pr/12) instead of main. This was fixed by adding revision="main" in from_pretrained() calls and clearing the local Hugging Face cache.
- The "Missing embed_positions.weight" warnings are harmless. The checkpoint did not contain these weights, so they were randomly initialized.
- The tokenizer alignment warning (PAD/BOS/EOS tokens) is also harmless. The library automatically fixed it.
- All three errors were caused by version incompatibility between older tutorial code and newer versions of transformers + accelerate.

After applying the above fixes, training started successfully and showed step-wise progress with training and validation loss.

## License
This project is licensed under the MIT License.