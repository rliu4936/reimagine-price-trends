# Core training function
from Experimenter.cnn_experiment import train_ensem_model

# Optional: if these are used in the script or configs
import torch
import numpy as np
import pandas as pd
import os

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()  # Recommended for Windows
    train_ensem_model(
        ws_list=[20],
        pw_list=[5],
        country="CN",
        total_worker=1,
        calculate_portfolio=True,
        ts1d_model=False,
        ts_scale="image_scale",
        regression_label=None,
        pf_delay_list=[0],
        lr=1e-4,
    )
