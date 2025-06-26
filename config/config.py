from pathlib import Path


from src.utils.varnames import ColNames as C


# ==== Data paths ====
SALES_DATA_PATH = "/home/jmarcosh/Projects/fanarmy/data/raw/Fan Army (Abril).xlsx"
PROCESSED_DATA_PATH = Path("/home/jmarcosh/Projects/fanarmy/data/processed")
MOVIEMETER_DATA_PATH = "/home/jmarcosh/Projects/fanarmy/data/raw/moviemeter_monthly.csv"
IMDB_DATA_PATH = "/home/jmarcosh/Projects/fanarmy/data/raw/imdb.csv"

# ==== Sales filtering ====
SUPPLIER_EXCLUDE = ['PROVEEDOR DE PLAYERA']
PLATFORM_INCLUDE = ['Amazon', 'Mercado Libre']
MINIMUM_SALES_PER_SKU = 15

# === Clustering and pricing parameters ===
PRICE_DUMMY_CONTIGUOUS_CELLS_NUM = 2
PRICE_DUMMY_WEIGHTS_DISTRIBUTION = [1, 0.5, 0.25]
NUMBER_OF_CLUSTERS = 200

# ==== Stockout labeling ====
DEV_SALE_ZERO = 1
SECOND_POINT = 10
DEV_SECOND_POINT = 2
SURGE_THRESHOLD = 1.25
FILTER_OUT_STOCKOUT = True

# ==== Time features ====
PERIODS_IN_YEAR = 12

# ==== Lag features ====
MAX_LAG = 15
VALID_LAGS_THRESHOLD = 0.5
# CORRELATION_THRESHOLD = 0.3  # Placeholder if you decide to filter lags

# ==== Moviemeter features ====
MOVIEMETER_TREND_WINDOWS = [3, 6]

# ==== IMDb features ====
IMPORTANT_LICENSES = ['MINECRAFT', 'SUPER MARIO']
MIN_PRODUCED_MOVIES = 2

# ==== Merge keys ====
MERGE_KEY = [C.LICENSE]

# ==== Encoding ====
IMDB_ENCODING = 'latin1'
