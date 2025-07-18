from DataHandler.equity_data import  get_index_ret, get_stock_ret_freqs
index_df = get_index_ret(freq="month", country="CN")
print("Index Return Data:", index_df.shape)
index_df.head()
period_end_df = get_stock_ret_freqs(freq="month", country="CN")
print("Period End Data:", period_end_df.shape)
period_end_df.head()