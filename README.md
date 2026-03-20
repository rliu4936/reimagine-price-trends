# Re-Imagining Price Trends

An end-to-end deep learning system that predicts short-term stock price movements by training convolutional neural networks on candlestick chart images. Implements and extends the methodology from [Jiang, Kelly, and Xiu (2023)](https://doi.org/10.1111/jofi.13268), published in *The Journal of Finance*.

The system processes raw OHLC market data across 27 international markets, generates chart images at multiple time horizons, trains ensemble CNN models, and constructs systematic long-short equity portfolios — including a live trading pipeline that produces daily buy/sell orders for the Chinese A-share market.

## How It Works

**1. Data Ingestion** — Loads and preprocesses daily stock data from CRSP (US) and CSMAR (China), computing forward-looking returns at weekly, monthly, and quarterly frequencies.

**2. Image Generation** — Converts OHLC price history into grayscale candlestick chart images at three lookback windows (5, 20, and 60 days), with optional moving average overlays and volume bars. Each image encodes the visual pattern a technical trader would see.

**3. CNN Training** — Trains 2D convolutional neural networks (and a 1D time-series variant) to classify whether a stock's price will rise or fall. Uses ensemble averaging over 5 models with different random seeds and dropout masks to improve prediction stability.

**4. Portfolio Construction** — Ranks all stocks by predicted up-probability, forms decile portfolios, and computes long-short (top decile minus bottom decile) returns with both equal-weight and value-weight schemes.

**5. Live Signal Generation** — Scrapes real-time Chinese market data, generates chart images for the current date, runs ensemble inference, merges CNN signals with fundamental metrics (P/E, P/B, debt-to-asset), and outputs algorithmic trade orders.

## Architecture

The 2D CNN follows a standard image classification pipeline scaled by lookback window:

| Window | Image Size | Conv Layers | Architecture |
|--------|-----------|-------------|--------------|
| 5-day  | 32 x 15   | 2           | Conv2D → BatchNorm → LeakyReLU → MaxPool (x2) → Dropout → FC |
| 20-day | 64 x 60   | 3           | Conv2D → BatchNorm → LeakyReLU → MaxPool (x3) → Dropout → FC |
| 60-day | 96 x 180  | 4           | Conv2D → BatchNorm → LeakyReLU → MaxPool (x4) → Dropout → FC |

- Xavier initialization, 50% dropout, Adam optimizer with early stopping
- Configurable filter sizes, strides, dilation, and channel depths for architecture search
- Transfer learning support across time scales and international markets

## Project Structure

```
src/
├── DataHandler/        # Market data loading, preprocessing, US/China adapters
├── Renderer/           # OHLC chart image generation, PyTorch Dataset classes
├── CNNModeler/         # 2D CNN and 1D CNN model definitions
├── Experimenter/       # Training loop, ensemble inference, checkpointing
├── PortfolioManager/   # Decile portfolio construction, weighting schemes
├── BackTester/         # Out-of-sample performance evaluation
├── Analyzer/           # Sharpe ratios, factor regressions, cross-market analysis
├── StrategyExecutor/   # Live data scraping, signal generation, order execution
└── Misc/               # Configuration, caching, utilities
```

**Entry points:**
- `train_eval.py` — Train and evaluate CNN models
- `render_all.py` — Batch generate chart images across years
- `current.py` — Run the live trading signal pipeline

## Setup

```bash
conda env create -f environment.yml
conda activate cnn310
```

## Tech Stack

**Deep Learning:** PyTorch (CNN architectures, custom Dataset/DataLoader pipelines, GPU training)

**Data Processing:** pandas, NumPy, pyarrow, SciPy

**Market Data:** akshare, efinance (Chinese market real-time data)

**Visualization:** matplotlib, Pillow (programmatic chart image generation)
