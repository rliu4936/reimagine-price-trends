# generate_images.py
import os
import os.path as op
from datetime import datetime

import pandas as pd
from tqdm import tqdm

from Renderer.draw_ohlc import DrawOHLC, DrawChartError
from DataHandler import dgp_config as dcf  # must define STOCKS_SAVEPATH

def generate_images_for_date(save_date: str | None = None,
                             window: int = 20,
                             has_volume_bar: bool = True,
                             chart_type: str = "bar",
                             ma_lags: tuple[int, ...] = (20,),
                             verbose: bool = False) -> str:
    """
    Render OHLC (or pixel) images from previously downloaded CSVs.

    Parameters
    ----------
    save_date : str (YYYYMMDD). Folder with CSVs. Default: today.
    window : int in {5, 20, 60}. Number of last rows to draw.
    has_volume_bar : bool
    chart_type : 'bar' | 'pixel' | 'centered_pixel'
    ma_lags : tuple of MA lags expected (columns must exist, e.g. ma20).
    verbose : bool

    Returns
    -------
    out_dir : directory where images are written.
    """
    if save_date is None:
        save_date = datetime.today().strftime("%Y%m%d")
    assert window in (5, 20, 60), "window must be 5, 20, or 60."

    data_dir = op.join(dcf.WORK_DIR, "current", save_date, "price_data")
    if not op.isdir(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")


    out_dir = op.join(dcf.WORK_DIR, "current", save_date, "images")
    if not op.isdir(out_dir):
         os.makedirs(out_dir, exist_ok=True)


    csv_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
    if not csv_files:
        if verbose:
            print(f"No CSV files in {data_dir}")
        return out_dir

    for file in tqdm(csv_files, desc="Rendering charts"):
        code = file[:-4]

        # ---------- CACHE CHECK ----------
        img_path = op.join(out_dir, f"{code}.png")
        if op.isfile(img_path):
            if verbose:
                print(f"Skip {code}: image already exists")
            continue
        # ---------------------------------
        try:
            df = pd.read_csv(op.join(data_dir, file), parse_dates=["Date"], index_col="Date")
            if len(df) < window:
                continue
            df = df.tail(window).copy()

            # Required OHLC columns
            for col in ["Open", "High", "Low", "Close"]:
                if col not in df.columns:
                    raise ValueError(f"Missing {col}")

            # Volume if needed
            if has_volume_bar:
                if "Volume" not in df.columns:
                    raise ValueError("Missing Volume")
                df["Vol"] = df["Volume"].astype(float)
            else:
                df["Vol"] = 0.0

            # Check MA columns
            for lag in ma_lags:
                ma_col = f"ma{lag}"
                if ma_col not in df.columns:
                    raise ValueError(f"Missing {ma_col}")

            # Normalize price & MA columns so first Close == 1.000
            first_close = float(df.iloc[0]["Close"])
            price_cols = ["Open", "High", "Low", "Close"] + [f"ma{lag}" for lag in ma_lags]
            df[price_cols] = df[price_cols] / first_close

            # Build drawing dataframe
            draw_cols = ["Open", "High", "Low", "Close", "Vol"] + [f"ma{lag}" for lag in ma_lags]
            draw_df = df[draw_cols].copy()

            chart = DrawOHLC(draw_df,
                             has_volume_bar=has_volume_bar,
                             ma_lags=list(ma_lags),
                             chart_type=chart_type)
            image = chart.draw_image()
            if image is not None:
                image.save(op.join(out_dir, f"{code}.png"))
            else:
                if verbose:
                    print(f"Skip {code}: image returned None")

        except (DrawChartError, ValueError) as e:
            if verbose:
                print(f"Skip {code}: {e}")
        except Exception as e:
            if verbose:
                print(f"Unexpected {code}: {e}")

    if verbose:
        print(f"Finished generating images in {out_dir}")
    return out_dir


if __name__ == "__main__":
    generate_images_for_date()
