# LLM-NID Forecasting

This repository contains two core scripts for infectious disease time-series forecasting:

- **Python**: `llm_nid_core.py` — LoRA-tuned Qwen-2.5-3B regression on a single disease series.
- **R**: `forecast_comparison_core.R` — Comparison of ARIMA, GARCH, ETS, XGBoost, LSTM, and LLM.

## Repository Structure

```
.
├── llm_nid_core.py               # Python script: LoRA-tuned Qwen-2.5-3B
├── forecast_comparison_core.R    # R script: multi-model benchmarking
├── requirements.txt              # Python dependencies
├── environment.yml               # Conda environment (optional)
├── CONTRIBUTING.md               # Contribution guidelines
├── LICENSE                       # License file
└── .github/
    ├── ISSUE_TEMPLATE.md         # Issue template
    └── workflows/
        └── ci.yml                # GitHub Actions workflow
```

## Prerequisites
- Python ≥ 3.10
- R ≥ 4.0
- **GPU**: NVIDIA GPU with >=8 GB VRAM and CUDA Toolkit >=11.7 (optional; CPU-only is supported but slower)

## Installation

1. Clone the repository  
   
2. Set up Python environment  
   ```bash
   conda env create -f environment.yml
   conda activate infectious-forecast
   # or: pip install -r requirements.txt
   ```
3. Set up R environment  
   ```r
   renv::restore()
   # or: install.packages(c("keras","forecast","rugarch","xgboost","tibble","dplyr","readxl","stringr","reticulate","openxlsx"))
   ```

## Data

- Online available at: https://www.chinacdc.cn/jksj/jksj01/index.html

## Usage

- **Python script**:  
  ```bash
  python llm_nid_core.py
  ```
- **R script**:  
  ```bash
  Rscript forecast_comparison_core.R
  ```

## Reproducibility

- Random seed fixed at `3407`.  
- 8 GB GPU recommended for LoRA fine-tuning; CPU-only supported.  

## Citation

If you use this code, please cite:  
> Xinsheng Wu, Jinyuan Wu, Zhongwen Wang, Ye Yao, Huachun Zou. "Large language models as versatile predictive engines for notifiable infectious diseases in China." 2025. (unpublished)

## License

MIT License

