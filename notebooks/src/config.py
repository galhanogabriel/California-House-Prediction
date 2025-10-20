from pathlib import Path


PROJECT_FOLDER = Path(__file__).resolve().parents[2]

DATA_FOLDER = PROJECT_FOLDER / "dados" #data folder

# Place the path to your project's data files below.
ORIGINAL_DATA = DATA_FOLDER / "housing.csv"
CLEANED_DATA = DATA_FOLDER / "housing_clean.parquet"
ORIGINAL_GEO_DATA = DATA_FOLDER / "california_counties.geojson"
GEO_MEDIAN_DATA = DATA_FOLDER / "geo_median.parquet"

# Place the path to your project's model files below.
MODELS_FOLDER = PROJECT_FOLDER / "models"
FINAL_MODEL = MODELS_FOLDER / "ridge_polyfeat_target_quantile.joblib"
