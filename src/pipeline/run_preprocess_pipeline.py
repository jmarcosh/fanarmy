from config.config import (
    SALES_DATA_PATH, PLATFORM_INCLUDE,
    SUPPLIER_EXCLUDE, MINIMUM_SALES_PER_SKU,
    DEV_SALE_ZERO, SECOND_POINT, DEV_SECOND_POINT, SURGE_THRESHOLD, PRICE_DUMMY_CONTIGUOUS_CELLS_NUM,
    PRICE_DUMMY_WEIGHTS_DISTRIBUTION, NUMBER_OF_CLUSTERS, PROCESSED_DATA_PATH, FILTER_OUT_STOCKOUT,
    BOTTOM_VAR, AGGREGATION_LEVELS, VALUE_COLS, CATEGORICAL_COLS, AVERAGE_COLS
)
from src.preprocess.aggregate_data import aggregate_data
from src.preprocess.categorize_data import categorize_data
from src.preprocess.load_sales_data import load_sales_data, filter_out_skus_with_non_significant_sales
from src.preprocess.stockout_labeling import stockout_labeling


def run_preprocess_pipeline():
    sales = load_sales_data(data_path=SALES_DATA_PATH, platform_include=PLATFORM_INCLUDE, supplier_exclude=SUPPLIER_EXCLUDE)

    # fix in source file
    sales.loc[(sales['Categoría'] == 'DISNEY') & (
        sales['Descripción'].str.contains('stich', case=False, na=False)), 'Categoría'] = "STITCH"
    sales.loc[(sales['Categoría'] == 'DISNEY') & (
        sales['Descripción'].str.contains('stitch', case=False, na=False)), 'Categoría'] = "STITCH"
    sales.loc[(sales['Categoría'] == 'DISNEY') & (sales['Descripción'].str.contains('mandalorian', case=False, na=False)), 'Categoría'] = "THE MANDALORIAN"
    sales.loc[(sales['Categoría'] == 'DISNEY') & (sales['Descripción'].str.contains('star wars', case=False, na=False)), 'Categoría'] = "STAR WARS"

    sales = categorize_data(sales, PRICE_DUMMY_CONTIGUOUS_CELLS_NUM, PRICE_DUMMY_WEIGHTS_DISTRIBUTION, NUMBER_OF_CLUSTERS)
    sales = filter_out_skus_with_non_significant_sales(sales, MINIMUM_SALES_PER_SKU)
    sales = stockout_labeling(sales, DEV_SALE_ZERO, SECOND_POINT, DEV_SECOND_POINT, SURGE_THRESHOLD, FILTER_OUT_STOCKOUT)
    sales = aggregate_data(sales, BOTTOM_VAR, AGGREGATION_LEVELS, VALUE_COLS, CATEGORICAL_COLS, AVERAGE_COLS)
    sales.to_csv(PROCESSED_DATA_PATH / 'processed_sales.csv', index=False)

if __name__ == "__main__":
    run_preprocess_pipeline()
