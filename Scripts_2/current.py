import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


from datetime import datetime
from StrategyExecutor import (
    scrape_current_data,
    generate_current_images,
    cats_csv_generator
)

from Experimenter import cnn_experiment as ce
from PortfolioManager import portfolio as pf
from Misc import config as cf

import pandas as pd
import os

from DataHandler import dgp_config as dcf  # must define STOCKS_SAVEPATH

# Download raw CN stock data

scrape_current_data.download_cn_stocks(verbose=True)

# Generate chart images for the current date
generate_current_images.generate_images_for_date(verbose=True)

# Load CNN experiment object
exp_obj = ce.get_exp_obj_by_spec(20, 5, lr=1e-4, drop_prob=0.75)

# Generate ensemble prediction CSV
ensem_res = exp_obj.generate_current_ensem_res("week", True)
ensem_res["Date"] = pd.to_datetime(ensem_res["ending_date"])
ensem_res.set_index("Date", inplace=True)

save_date = datetime.today().strftime("%Y%m%d")
out_dir = os.path.join(dcf.WORK_DIR, "current", save_date, "images")


# Assign deciles and save current weights
weights_path = os.path.join(cf.WORK_DIR, "current", save_date, "current_weights.csv")
pf.save_current_decile_weights(ensem_res, top=150, output_path=weights_path)

# Generate order file from top-decile weights
ratio = 0.97
orders_path = os.path.join(cf.WORK_DIR, "current", save_date, "orders.csv")
cats_csv_generator.export_algo_orders(
    account_name="12341234", 
    total_amount=round(3_000_000 * ratio), 
    begin_time="093000", 
    end_time="100000",
    input_csv=weights_path, 
    output_csv=orders_path
)
