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

## License
This project is licensed under the MIT License.