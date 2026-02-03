# LLM-NID Forecasting

This repository contains two core scripts for infectious disease time-series forecasting:

- **Python**: `llm_nid_core.py` — LoRA-tuned Qwen-2.5-3B regression on a single disease series.
- **R**: `forecast_comparison_core.R` — Comparison of ARIMA, GARCH, ETS, XGBoost, LSTM, and LLM.

## Repository Structure

```
.
├── LLM_core.ipynb
├── forecast_comparison_core.R
├── zero_shot_core.ipynb
├── interpretability_core.ipynb
└── README.txt

```

## Prerequisites
- Python ≥ 3.10
- R ≥ 4.0
- **GPU**: NVIDIA GPU with >=50 GB VRAM and CUDA Toolkit >=11.7 (optional; CPU-only is supported but slower)

## Data

- Online available at: https://www.chinacdc.cn/jksj/jksj01/index.html and https://wonder.cdc.gov/nndss-annual-summary.html


## Citation

If you use this code, please cite:  
> Xinsheng Wu, Jinyuan Wu, Zhongwen Wang, Ye Yao, Jason J. Ong, and Huachun Zou. "Large language models as versatile predictive engines for notifiable infectious diseases." 2026. (unpublished)

## License

MIT License

