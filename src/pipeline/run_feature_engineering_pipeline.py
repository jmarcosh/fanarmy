import pandas as pd

from config.config import (
    PROCESSED_DATA_PATH, MOVIEMETER_DATA_PATH, IMDB_DATA_PATH,
    IMDB_ENCODING, PERIODS_IN_YEAR, MAX_LAG, VALID_LAGS_THRESHOLD ,
    MOVIEMETER_TREND_WINDOWS, IMPORTANT_LICENSES, MIN_PRODUCED_MOVIES,
    MERGE_KEY
)

from src.feature_engineering.imdb import engineer_imdb_features
from src.feature_engineering.integration import merge_sales_and_imdb_data
from src.feature_engineering.moviemeter import engineer_moviemeter_features
from src.feature_engineering.sales import engineer_sales_features


def run_feature_engineering_pipeline():
    processed_sales = pd.read_csv(PROCESSED_DATA_PATH / 'processed_sales.csv')
    sales = engineer_sales_features(processed_sales, PERIODS_IN_YEAR, MAX_LAG, VALID_LAGS_THRESHOLD)

    moviemeter_raw = pd.read_csv(MOVIEMETER_DATA_PATH)
    moviemeter = engineer_moviemeter_features(moviemeter_raw, MOVIEMETER_TREND_WINDOWS)

    imdb_raw = pd.read_csv(IMDB_DATA_PATH, encoding=IMDB_ENCODING)
    imdb = engineer_imdb_features(imdb_raw, MIN_PRODUCED_MOVIES)

    sales_imdb = merge_sales_and_imdb_data([sales, moviemeter, imdb], MERGE_KEY, IMPORTANT_LICENSES)
    sales_imdb.to_csv(PROCESSED_DATA_PATH / 'data_features.csv', index=False)

if __name__ == "__main__":
    run_feature_engineering_pipeline()