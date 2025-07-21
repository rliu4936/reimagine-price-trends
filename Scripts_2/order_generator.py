from StrategyExecutor import cats_csv_generator
from Misc import config as cf
import os
import pandas as pd
# Define output path
intput_path = os.path.join(cf.WORK_DIR, "weights", "current_weights.csv")
output_path = os.path.join(cf.WORK_DIR, "weights", "orders.csv")
cats_csv_generator.export_algo_orders(input_csv=intput_path, output_csv=output_path)