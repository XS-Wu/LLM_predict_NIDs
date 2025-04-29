# LLM-NID Forecasting

This repository contains two core scripts for infectious disease time-series forecasting:

- **Python**: `llm_nid_core.py` — LoRA-tuned Qwen-2.5-3B regression on a single disease series.
- **R**: `forecast_comparison_core.R` — Comparison of ARIMA, GARCH, ETS, XGBoost, LSTM, and LLM.

## Prerequisites
- Python ≥ 3.10
- R ≥ 4.0
- NVIDIA GPU with CUDA 11.3 or higher (≥8 GB VRAM) for efficient LoRA fine-tuning and inference (**optional but strongly recommended**)
- CPU-only mode is supported but will be significantly slower

## Installation

### Python
```bash
pip install -r requirements.txt
