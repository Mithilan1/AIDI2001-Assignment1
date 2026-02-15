# AIDI2001-Assignment1

Assignment 1 for AIDI2001 (Deep Learning for NLP).

## Prerequisites

- Python 3.10+
- `pip`

## Required Libraries

- `torch`: PyTorch deep learning framework
- `transformers`: Hugging Face transformers library (BERT, GPT models)
- `tokenizers`: Fast tokenizer implementations
- `pandas`: Data manipulation and analysis
- `seaborn`: Statistical data visualization
- `matplotlib`: Plotting and visualization
- `accelerate`: Distributed training and inference utilities
- `safetensors`: Safe serialization format for tensors
- `sentencepiece`: Text tokenization and detokenization library
- `datasets`: Hugging Face datasets library
- `notebook`: Jupyter notebook support

## Install

```bash
pip install torch transformers tokenizers pandas seaborn matplotlib accelerate safetensors sentencepiece datasets notebook
```

Or with uv:

```bash
uv pip install torch transformers tokenizers pandas seaborn matplotlib accelerate safetensors sentencepiece datasets notebook
```

## Project Files

- `Assignment1.ipynb`: main submission notebook
- `bert.py`: Part 1 tokenizer experiments
- `gpt.py`: Part 2 generation experiments
- `BERTvsGPT.py`: Parts 3, 3.5, and 4 experiments
- `outputs/`: saved CSVs and figures

## Run

```bash
python bert.py
python gpt.py
python BERTvsGPT.py
```

Or open and run the notebook:

```bash
jupyter notebook Assignment1.ipynb
```
