# Re-Imagining Price Trends

A CNN-based stock trend prediction system that trains convolutional neural networks on OHLC candlestick chart images to predict short-term equity price movements. Based on the methodology from [Jiang, Kelly, and Xiu (2023)](https://doi.org/10.1111/jofi.13268) — *"Re-Imagining Price Trends"*, The Journal of Finance.

## Overview

This project converts raw stock price data into candlestick chart images and trains CNN models to classify future price direction. Predicted signals are used to construct long-short decile portfolios with systematic rebalancing.

**Key capabilities:**
- Multi-horizon prediction (5-day, 20-day, 60-day windows)
- 2D CNN (image-based) and 1D CNN (time-series) architectures
- US (CRSP) and China (CSMAR) market support with cross-market transfer learning
- Ensemble inference across multiple seeds and dropout configurations
- Automated portfolio construction, backtesting, and performance analysis
- Live signal generation pipeline for Chinese equities

## Architecture

```
Raw OHLC Data
    |
    v
Chart Image Generation ──> OHLC candlestick images (5D / 20D / 60D windows)
    |
    v
CNN Training ──> 2D Conv or 1D Conv models with early stopping
    |
    v
Ensemble Prediction ──> Averaged probability scores across models
    |
    v
Portfolio Construction ──> Long-short decile portfolios (equal/value-weighted)
    |
    v
Backtesting & Analysis ──> Sharpe ratios, regression tables, factor analysis
```

## Project Structure

```
src/
├── DataHandler/        # Data loading and preprocessing (US/China market adapters)
├── Renderer/           # OHLC chart image generation and PyTorch datasets
├── CNNModeler/         # 2D and 1D CNN model definitions
├── Experimenter/       # Training orchestration, validation, ensemble logic
├── PortfolioManager/   # Portfolio construction and weighting schemes
├── BackTester/         # Out-of-sample backtesting
├── Analyzer/           # Performance metrics, regression tables, factor analysis
├── StrategyExecutor/   # Live trading signal generation (Chinese market)
├── Misc/               # Configuration, caching, utility functions
├── train_eval.py       # Entry point: model training and evaluation
├── render_all.py       # Entry point: batch chart image generation
├── precompute_all.py   # Entry point: precompute all data artifacts
└── current.py          # Entry point: live trading signal pipeline
```

## Setup

```bash
conda env create -f environment.yml
conda activate cnn310
```

## Usage

**1. Generate chart images**
```python
from src.Renderer.generate_chart import GenerateStockData

dgp = GenerateStockData(
    country="USA", year=2019, window_size=20,
    freq="week", ma_lags=[20], volume_bar=True, chart_type="bar"
)
dgp.save_annual_data()      # 2D CNN images
dgp.save_annual_ts_data()   # 1D CNN time-series
```

**2. Train models**
```bash
python src/train_eval.py
```

**3. Generate live signals (China market)**
```bash
python src/current.py
```

## Configuration

Key parameters in `src/Misc/config.py`:
- `BATCH_SIZE`: Training batch size (default: 64)
- `SPLIT_YEARS`: In-sample / out-of-sample split year
- `BENCHMARK_MODEL_LAYERNUM_DICT`: CNN depth per window size

Image dimensions in `src/DataHandler/dgp_config.py`:
| Window | Width | Height |
|--------|-------|--------|
| 5-day  | 15    | 32     |
| 20-day | 60    | 64     |
| 60-day | 180   | 96     |

## Tech Stack

- **Framework:** PyTorch
- **Data:** pandas, NumPy, pyarrow, akshare
- **Visualization:** matplotlib, Pillow
- **Environment:** Python 3.10, Conda
