
from pathlib import Path
import json
import logging

import pandas as pd


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "online_retail_ingested.csv.gz"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"
LOG_DIR = PROJECT_ROOT / "outputs" / "logs"
ANOMALY_DIR = PROJECT_ROOT / "outputs" / "anomalies"

for directory in [
    OUTPUT_DIR,
    REPORT_DIR,
    LOG_DIR,
    ANOMALY_DIR
]:
    directory.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. LOGGING CONFIGURATION
# --------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(
            LOG_DIR / "data_cleaning.log",
            encoding="utf-8"
        ),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


# --------------------------------------------------
# 3. LOAD DATA
# --------------------------------------------------

def load_data():

    logger.info("Loading ingested transaction dataset...")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input dataset not found: {INPUT_FILE}"
        )

    df = pd.read_csv(
        INPUT_FILE,
        compression="gzip",
        dtype={
            "InvoiceNo": "string",
            "StockCode": "string",
            "CustomerID": "string"
        },
        parse_dates=["InvoiceDate"],
        low_memory=False
    )

    logger.info(
        f"Original records loaded: {len(df):,}"
    )

    return df


# --------------------------------------------------
# 4. VALIDATE SCHEMA AND DATA TYPES
# --------------------------------------------------

def validate_and_prepare(df):

    logger.info("Validating schema and data types...")

    required_columns = [
        "InvoiceNo",
        "StockCode",
        "Description",
        "Quantity",
        "InvoiceDate",
        "UnitPrice",
        "CustomerID",
        "Country"
    ]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    # Create a unique source-row identifier.
    # This is a row identifier, not a business transaction ID.
    df = df.copy()

    df.insert(
        0,
        "source_record_id",
        range(1, len(df) + 1)
    )

    # Ensure consistent types.
    df["InvoiceNo"] = df["InvoiceNo"].astype("string")
    df["StockCode"] = df["StockCode"].astype("string")
    df["CustomerID"] = df["CustomerID"].astype("string")

    df["InvoiceDate"] = pd.to_datetime(
        df["InvoiceDate"],
        errors="coerce"
    )

    df["Quantity"] = pd.to_numeric(
        df["Quantity"],
        errors="coerce"
    )

    df["UnitPrice"] = pd.to_numeric(
        df["UnitPrice"],
        errors="coerce"
    )

    # Trim whitespace in textual identifiers.
    for col in [
        "InvoiceNo",
        "StockCode",
        "CustomerID"
    ]:
        df[col] = df[col].str.strip()

    return df


# --------------------------------------------------
# 5. CREATE RECORD-LEVEL DECISIONS
# --------------------------------------------------

def classify_records(df):

    logger.info("Classifying transaction records...")

    df = df.copy()

    # Detect exact duplicate copies.
    duplicate_mask = df.duplicated(
        subset=[
            col for col in df.columns
            if col != "source_record_id"
        ],
        keep="first"
    )

    # Detect invalid mandatory fields.
    missing_invoice = (
        df["InvoiceNo"].isna()
        | df["InvoiceNo"].eq("")
    )

    missing_stock = (
        df["StockCode"].isna()
        | df["StockCode"].eq("")
    )

    missing_date = df["InvoiceDate"].isna()

    invalid_quantity = df["Quantity"].isna()

    invalid_price = df["UnitPrice"].isna()

    # Invalid records are quarantined before
    # cancellation and return classification.
    invalid_price_value = (
        df["UnitPrice"].notna()
        & (df["UnitPrice"] <= 0)
    )

    zero_quantity = (
        df["Quantity"].notna()
        & (df["Quantity"] == 0)
    )

    cancellation_mask = (
        df["InvoiceNo"]
        .str.startswith("C", na=False)
    )

    return_mask = (
        df["Quantity"].notna()
        & (df["Quantity"] < 0)
    )

    positive_sales_mask = (
        df["Quantity"].notna()
        & (df["Quantity"] > 0)
    )

    # Initialize decision columns.
    df["record_status"] = "PENDING"

    df["decision_reason"] = "Not classified"

    # Priority 1: Duplicate records.
    df.loc[
        duplicate_mask,
        "record_status"
    ] = "QUARANTINED"

    df.loc[
        duplicate_mask,
        "decision_reason"
    ] = "EXACT_DUPLICATE"

    # Priority 2: Missing mandatory identifiers.
    invalid_mandatory = (
        missing_invoice
        | missing_stock
        | missing_date
        | invalid_quantity
        | invalid_price
    )

    mask = (
        df["record_status"].eq("PENDING")
        & invalid_mandatory
    )

    df.loc[
        mask,
        "record_status"
    ] = "QUARANTINED"

    df.loc[
        mask,
        "decision_reason"
    ] = "MISSING_OR_INVALID_MANDATORY_FIELD"

    # Priority 3: Non-positive unit price.
    mask = (
        df["record_status"].eq("PENDING")
        & invalid_price_value
    )

    df.loc[
        mask,
        "record_status"
    ] = "QUARANTINED"

    df.loc[
        mask,
        "decision_reason"
    ] = "NON_POSITIVE_UNIT_PRICE"

    # Priority 4: Zero quantity.
    mask = (
        df["record_status"].eq("PENDING")
        & zero_quantity
    )

    df.loc[
        mask,
        "record_status"
    ] = "QUARANTINED"

    df.loc[
        mask,
        "decision_reason"
    ] = "ZERO_QUANTITY"

    # Priority 5: Cancellation invoices.
    mask = (
        df["record_status"].eq("PENDING")
        & cancellation_mask
    )

    df.loc[
        mask,
        "record_status"
    ] = "CANCELLATION"

    df.loc[
        mask,
        "decision_reason"
    ] = "CANCELLATION_INVOICE"

    # Priority 6: Returns.
    mask = (
        df["record_status"].eq("PENDING")
        & return_mask
    )

    df.loc[
        mask,
        "record_status"
    ] = "RETURN"

    df.loc[
        mask,
        "decision_reason"
    ] = "NEGATIVE_QUANTITY_RETURN"

    # Priority 7: Valid positive sales.
    mask = (
        df["record_status"].eq("PENDING")
        & positive_sales_mask
    )

    df.loc[
        mask,
        "record_status"
    ] = "ACCEPTED"

    df.loc[
        mask,
        "decision_reason"
    ] = "VALID_POSITIVE_SALE"

    if df["record_status"].eq("PENDING").any():
        raise ValueError(
            "Some records could not be classified."
        )

    logger.info("Record classification completed.")

    return df


# --------------------------------------------------
# 6. SEPARATE DATASETS
# --------------------------------------------------

def separate_datasets(df):

    logger.info("Separating classified records...")

    sales = df.loc[
        df["record_status"] == "ACCEPTED"
    ].copy()

    returns = df.loc[
        df["record_status"] == "RETURN"
    ].copy()

    cancellations = df.loc[
        df["record_status"] == "CANCELLATION"
    ].copy()

    quarantine = df.loc[
        df["record_status"] == "QUARANTINED"
    ].copy()

    return sales, returns, cancellations, quarantine


# --------------------------------------------------
# 7. SAVE CLEANED DATASETS
# --------------------------------------------------

def save_datasets(
    sales,
    returns,
    cancellations,
    quarantine,
    classified
):

    logger.info("Saving classified datasets...")

    datasets = {

        "sales_transactions.csv.gz": sales,

        "returns_transactions.csv.gz": returns,

        "cancellation_transactions.csv.gz": cancellations,

        "quarantined_transactions.csv.gz": quarantine,

        "record_decisions.csv.gz": classified[
            [
                "source_record_id",
                "InvoiceNo",
                "InvoiceDate",
                "record_status",
                "decision_reason"
            ]
        ]
    }

    for filename, dataset in datasets.items():

        output_path = OUTPUT_DIR / filename

        dataset.to_csv(
            output_path,
            index=False,
            compression="gzip"
        )

        logger.info(
            f"Saved {filename}: {len(dataset):,} records"
        )


# --------------------------------------------------
# 8. RECONCILIATION AND CONTROL CHECKS
# --------------------------------------------------

def generate_reconciliation(
    original_count,
    sales,
    returns,
    cancellations,
    quarantine
):

    logger.info("Running reconciliation controls...")

    counts = {

        "original_records": original_count,

        "accepted_sales": len(sales),

        "returns": len(returns),

        "cancellations": len(cancellations),

        "quarantined": len(quarantine)
    }

    classified_total = (
        counts["accepted_sales"]
        + counts["returns"]
        + counts["cancellations"]
        + counts["quarantined"]
    )

    counts["classified_total"] = classified_total

    counts["reconciliation_difference"] = (
        original_count - classified_total
    )

    counts["reconciliation_status"] = (
        "PASS"
        if original_count == classified_total
        else "FAIL"
    )

    counts["missing_customer_ids_in_sales"] = int(
        sales["CustomerID"].isna().sum()
    )

    counts["sales_quantity_positive"] = bool(
        (sales["Quantity"] > 0).all()
    )

    counts["sales_price_positive"] = bool(
        (sales["UnitPrice"] > 0).all()
    )

    counts["unique_source_ids"] = int(
        (
            sales["source_record_id"].nunique()
            + returns["source_record_id"].nunique()
            + cancellations["source_record_id"].nunique()
            + quarantine["source_record_id"].nunique()
        )
    )

    if counts["reconciliation_status"] != "PASS":
        raise ValueError(
            "Reconciliation failed. Check record classification."
        )

    if counts["unique_source_ids"] != original_count:
        raise ValueError(
            "Source record IDs are not uniquely reconciled."
        )

    if not counts["sales_quantity_positive"]:
        raise ValueError(
            "Accepted sales contain non-positive quantities."
        )

    if not counts["sales_price_positive"]:
        raise ValueError(
            "Accepted sales contain non-positive prices."
        )

    output_file = (
        REPORT_DIR / "cleaning_reconciliation.json"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            counts,
            file,
            indent=4
        )

    logger.info("Reconciliation completed successfully.")

    return counts


# --------------------------------------------------
# 9. MAIN EXECUTION
# --------------------------------------------------

def main():

    logger.info("=" * 55)

    logger.info("STARTING MODULE 04")

    logger.info("=" * 55)

    df = load_data()

    original_count = len(df)

    df = validate_and_prepare(df)

    classified = classify_records(df)

    sales, returns, cancellations, quarantine = (
        separate_datasets(classified)
    )

    save_datasets(
        sales,
        returns,
        cancellations,
        quarantine,
        classified
    )

    reconciliation = generate_reconciliation(
        original_count,
        sales,
        returns,
        cancellations,
        quarantine
    )

    logger.info("=" * 55)

    logger.info("DATA CLEANING COMPLETED")

    logger.info(
        f"Original records: {original_count:,}"
    )

    logger.info(
        f"Accepted sales: {len(sales):,}"
    )

    logger.info(
        f"Returns: {len(returns):,}"
    )

    logger.info(
        f"Cancellations: {len(cancellations):,}"
    )

    logger.info(
        f"Quarantined: {len(quarantine):,}"
    )

    logger.info(
        f"Reconciliation: "
        f"{reconciliation['reconciliation_status']}"
    )

    logger.info("=" * 55)

    print("\nRECONCILIATION REPORT")

    print(
        json.dumps(
            reconciliation,
            indent=4
        )
    )


if __name__ == "__main__":
    main()
