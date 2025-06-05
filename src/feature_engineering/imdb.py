import pandas as pd

from src.utils.varnames import ColNames as c

def engineer_imdb_features(df, min_produced_movies) -> pd.DataFrame:
    df = df.rename({'license': c.LICENSE}, axis=1)
    df['run_time'] = df['run_time'].str.replace(' min', '').astype(float)
    df['imdb_rating'] = df['imdb_rating'].str.replace('/10', '').astype(float)
    df['next_release_dummy'] = df['next release'].notna().astype(int)
    df = create_production_company_dummies(df, min_produced_movies)
    return df


def create_production_company_dummies(df: pd.DataFrame, min_produced_movies: int) -> pd.DataFrame:
    company_counts = df[[c.LICENSE, 'production_company']].drop_duplicates()['production_company'].value_counts()
    selected_companies = company_counts[company_counts >= min_produced_movies].index.tolist()
    dummies = pd.get_dummies(df['production_company'])[selected_companies].astype(int)
    dummies.columns = [f'{production_dummy.replace(" ", "_").replace(".", "").lower()}' for production_dummy in
                       selected_companies]
    for col in dummies.columns:
        df[col] = dummies[col]
    return df