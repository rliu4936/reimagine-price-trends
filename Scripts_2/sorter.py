import os
import pandas as pd
from PortfolioManager import portfolio as pf
from Misc import config as cf

# Path to ensemble results CSV
ensem_res_path = os.path.join(
    "..", "WORK_SPACE", "new_model_res",
    "D20L3F53S31D21MP21F53S11D11MP21F53S11D11MP21C64",
    "20d5p-lr1E-05-dp0.50-maTrue-vbTrue-weeklyTrained-noDelayedReturn-CN",
    "ensem_res", "ensem5_res_current.csv"
)

# Load results
ensem_res = pd.read_csv(ensem_res_path)

ensem_res["Date"] = pd.to_datetime(ensem_res["ending_date"])
ensem_res.set_index("Date", inplace=True)


print(ensem_res.index)
print(ensem_res.index.dtype)

# Define output path
output_path = os.path.join(cf.WORK_DIR, "weights", "current_weights.csv")


# Save decile weights
pf.save_current_decile_weights(ensem_res, cut=10, output_path=output_path)
