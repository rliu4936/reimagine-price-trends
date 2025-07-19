# run_download.p

from StrategyExecutor import scrape_current_data, generate_current_images
# Run with defaults (40 days, today's date)
scrape_current_data.download_cn_stocks(verbose=True)

generate_current_images.generate_images_for_date()

# Example: Save under specific date string
# download_cn_stocks(save_date="20251010", verbose=True)
