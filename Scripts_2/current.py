import os
import warnings
from datetime import datetime
import pandas as pd

from StrategyExecutor import (
    scrape_current_data,
    generate_current_images,
    cats_csv_generator,
    fundamental_strat
)
from Experimenter import cnn_experiment as ce
from PortfolioManager import portfolio as pf
from Misc import config as cf
from DataHandler import dgp_config as dcf  # must define STOCKS_SAVEPATH

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def main(save_date: str = None):
    # If no date is provided, use the current date
    if save_date is None:
        save_date = datetime.today().strftime("%Y%m%d")
    
    # Step 1: Download raw CN stock data
    scrape_current_data.download_cn_stocks(save_date=save_date)

    # Step 2: Generate chart images for the current date
    generate_current_images.generate_images_for_date(save_date=save_date)

    # Step 3: Load CNN experiment object and generate ensemble predictions
    exp_obj = ce.get_exp_obj_by_spec(20, 5, lr=1e-4, drop_prob=0.75)
    ensem_res = exp_obj.generate_current_ensem_res("week", save_date=save_date)
    ensem_res["Date"] = pd.to_datetime(ensem_res["ending_date"])
    ensem_res.set_index("Date", inplace=True)

    # Step 4: Save technical and fundamental rankings
    tech_path = os.path.join(cf.WORK_DIR, "current", save_date, "tech_rank.csv")
    tech = pf.save_current_tech_rank(ensem_res, output_path=tech_path)

    fund_path = os.path.join(cf.WORK_DIR, "current", save_date, "fund_rank.csv")
    fund = fundamental_strat.get_fundamental_weights(fund_path)

    # Step 5: Merge rankings
    tech["StockID"] = tech["StockID"].astype(str).str.zfill(6)
    fund["StockID"] = fund["StockID"].astype(str).str.zfill(6)

    merged = pd.merge(tech, fund, how="inner", on="StockID")
    merged["tech_rank"] = pd.to_numeric(merged["tech_rank"])
    merged["fund_rank"] = pd.to_numeric(merged["fund_rank"])
    merged["merge_rank"] = merged["tech_rank"] + merged["fund_rank"]

    merged = merged.sort_values(by="merge_rank", ascending=True)
    merged = merged[[
        "StockID", "up_prob", "Close", "pi", "pe", "pb",
        "merge_rank", "tech_rank", "fund_rank", "debt-to-asset"
    ]]
    merged["StockID"] = merged["StockID"].astype(str).str.zfill(6)

    merged_path = os.path.join(cf.WORK_DIR, "current", save_date, "merged_rank.csv")
    merged.to_csv(merged_path, index=False)

    # Step 6: Generate order file from top-decile weights
    ratio = 0.97
    orders_path = os.path.join(cf.WORK_DIR, "current", save_date, "orders.csv")
    cats_csv_generator.export_algo_orders(
        account_name="12341234",
        total_amount=round(2_000_000 * ratio),
        begin_time="093000",
        end_time="100000",
        input_csv=merged_path,
        output_csv=orders_path
    )

    # TODO: Add current portfolio (not implemented yet)

# Example of running the main function
if __name__ == "__main__":
    main("20250727")  # If no date is passed, current date will be used
