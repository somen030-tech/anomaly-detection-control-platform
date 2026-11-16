
from pathlib import Path
import json
import logging

import pandas as pd


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_FILE = PROJECT_ROOT / "data" / "raw" / "Online Retail.xlsx"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
LOG_DIR = PROJECT_ROOT / "outputs" / "logs"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. LOGGING CONFIGURATION
# --------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(
            LOG_DIR / "data_ingestion.log",
            encoding="utf-8"
        ),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


# --------------------------------------------------
# 3. REQUIRED DATASET COLUMNS
# --------------------------------------------------

REQUIRED_COLUMNS = [
    "InvoiceNo",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "UnitPrice",
    "CustomerID",
    "Country"
]


# --------------------------------------------------
# 4. DATA INGESTION FUNCTION
# --------------------------------------------------

def load_raw_data():

    logger.info("Starting Module 02: Data Ingestion")

    if not RAW_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {RAW_FILE}"
        )

    logger.info("Loading original Excel dataset...")

    df = pd.read_excel(
        RAW_FILE,
        sheet_name=0,
        engine="openpyxl"
    )

    logger.info("Dataset loaded successfully")

    return df


# --------------------------------------------------
# 5. SCHEMA VALIDATION
# --------------------------------------------------

def validate_schema(df):

    logger.info("Validating dataset schema...")

    missing_columns = (
        set(REQUIRED_COLUMNS) - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    logger.info("Schema validation PASSED")

    return True


# --------------------------------------------------
# 6. BASIC DATA PROFILING
# --------------------------------------------------

def profile_data(df):

    logger.info("Starting initial data profiling...")

    # Convert dates for inspection
    df["InvoiceDate"] = pd.to_datetime(
        df["InvoiceDate"],
        errors="coerce"
    )

    # Calculate basic statistics
    profile = {

        "dataset_name": "UCI Online Retail",

        "total_rows": int(len(df)),

        "total_columns": int(len(df.columns)),

        "column_names": list(df.columns),

        "missing_values": {
            col: int(count)
            for col, count in df.isnull().sum().items()
        },

        "duplicate_rows": int(
            df.duplicated().sum()
        ),

        "unique_invoices": int(
            df["InvoiceNo"].nunique()
        ),

        "unique_products": int(
            df["StockCode"].nunique()
        ),

        "unique_customers": int(
            df["CustomerID"].nunique()
        ),

        "date_min": (
            str(df["InvoiceDate"].min())
            if df["InvoiceDate"].notna().any()
            else None
        ),

        "date_max": (
            str(df["InvoiceDate"].max())
            if df["InvoiceDate"].notna().any()
            else None
        ),

        "cancellation_invoices": int(
            df["InvoiceNo"]
            .astype(str)
            .str.startswith("C")
            .sum()
        ),

        "negative_quantity_records": int(
            (df["Quantity"] < 0).sum()
        ),

        "non_positive_price_records": int(
            (df["UnitPrice"] <= 0).sum()
        ),

        "invalid_dates": int(
            df["InvoiceDate"].isna().sum()
        )
    }

    return profile


# --------------------------------------------------
# 7. SAVE INGESTED DATA
# --------------------------------------------------

def save_ingested_data(df, profile):

    logger.info("Saving ingested dataset...")

    output_file = (
        PROCESSED_DIR / "online_retail_ingested.csv.gz"
    )

    df.to_csv(
        output_file,
        index=False,
        compression="gzip"
    )

    logger.info(
        f"Processed dataset saved: {output_file}"
    )

    profile_file = (
        PROCESSED_DIR / "data_profile.json"
    )

    with open(
        profile_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            profile,
            file,
            indent=4
        )

    logger.info(
        f"Data profile saved: {profile_file}"
    )


# --------------------------------------------------
# 8. MAIN EXECUTION
# --------------------------------------------------

def main():

    try:

        df = load_raw_data()

        validate_schema(df)

        profile = profile_data(df)

        save_ingested_data(df, profile)

        logger.info("=" * 50)

        logger.info("DATA INGESTION COMPLETED")

        logger.info(
            f"Total records: {profile['total_rows']}"
        )

        logger.info(
            f"Total columns: {profile['total_columns']}"
        )

        logger.info(
            f"Duplicate records: {profile['duplicate_rows']}"
        )

        logger.info(
            f"Unique customers: {profile['unique_customers']}"
        )

        logger.info("=" * 50)

        print("\nDATASET PROFILE")
        print(json.dumps(profile, indent=4))

    except Exception:

        logger.exception(
            "DATA INGESTION FAILED"
        )

        raise


if __name__ == "__main__":
    main()
