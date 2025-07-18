# Core training function
from Experimenter.cnn_experiment import run_arch_comparison

# Optional: if these are used in the script or configs
import torch
import numpy as np
import pandas as pd
import os

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()  # Recommended for Windows
    run_arch_comparison(
        ws=20,                   # window size
        pw=5,                   # prediction window
        ensem=5,                 # ensemble size
        load_saved_data=True,   # whether to load saved dataset
        dn=None,                # domain number (e.g., 0 for CN, 1 for US)
        ohlc_len=None,
        tl=None,
        pretrained=True,
        chart_type="bar",
        total_worker=1,
        worker_idx=0,
    )
