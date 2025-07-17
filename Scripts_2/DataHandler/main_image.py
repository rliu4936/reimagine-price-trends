import os #
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" #
from Experimenter.cnn_experiment import train_us_model
from Scripts_2.Renderer.generate_chart import GenerateStockData
from Data.generate_year import GenerateYearStockData
import torch #


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(6)


if __name__ == "__main__":
    # Generate Image Data
    year_list = list(range(2019, 2020)) #
    chart_type = "pixel"
    ws = 5
    freq = "week"
    ma_lags = [ws]
    vb = True
    for year in year_list:
        print(f"{ws}D {freq} {chart_type} {year}")
        dgp_obj = GenerateStockData(
            "USA",
            year,
            ws,
            freq,
            chart_freq=1,  # for time-scale I20/R20 to R5/R5, set ws=20 and chart_freq=4
            ma_lags=ma_lags,
            volume_bar=vb,
            need_adjust_price=True,
            allow_tqdm=True,
            chart_type=chart_type,
        )
        # generate CNN2D Data
        dgp_obj.save_annual_data()
      # generate CNN1D Data
        dgp_obj.save_annual_ts_data()
