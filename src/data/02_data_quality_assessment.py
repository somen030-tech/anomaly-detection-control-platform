
from pathlib import Path
import json
import logging

import numpy as np
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

OUTPUT_DIR = PROJECT_ROOT / "outputs"
REPORT_DIR = OUTPUT_DIR / "reports"
ANOMALY_DIR = OUTPUT_DIR / "anomalies"
LOG_DIR = OUTPUT_DIR / "logs"

for directory in [REPORT_DIR, ANOMALY_DIR, LOG_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. LOGGING
# --------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(
            LOG_DIR / "data_quality.log",
            encoding="utf-8"
        ),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


# --------------------------------------------------
# 3. REQUIRED COLUMNS
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
# 4. LOAD DATA
# --------------------------------------------------

def load_data():

    logger.info("Loading ingested transaction dataset...")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    df = pd.read_csv(
        INPUT_FILE,
        compression="gzip",
        parse_dates=["InvoiceDate"],
        low_memory=False
    )

    logger.info(
        f"Loaded {len(df):,} transaction records"
    )

    return df


# --------------------------------------------------
# 5. SCHEMA VALIDATION
# --------------------------------------------------

def validate_schema(df):

    logger.info("Checking required columns...")

    missing_columns = [
        col for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    return True


# --------------------------------------------------
# 6. STRUCTURAL QUALITY CHECKS
# --------------------------------------------------

def structural_checks(df):

    logger.info("Running structural quality checks...")

    results = []

    total_records = len(df)

    # Missing values
    for col in REQUIRED_COLUMNS:

        missing_count = int(df[col].isna().sum())

        missing_pct = (
            missing_count / total_records * 100
            if total_records else 0
        )

        status = (
            "WARNING" if missing_count > 0
            else "PASS"
        )

        results.append({
            "control": f"MISSING_{col.upper()}",
            "category": "STRUCTURAL",
            "affected_records": missing_count,
            "affected_percentage": round(missing_pct, 4),
            "severity": "MEDIUM",
            "status": status,
            "description": f"Missing values in {col}"
        })

    # Exact duplicate rows
    duplicate_count = int(df.duplicated().sum())

    results.append({
        "control": "EXACT_DUPLICATE_ROWS",
        "category": "STRUCTURAL",
        "affected_records": duplicate_count,
        "affected_percentage": round(
            duplicate_count / total_records * 100, 4
        ) if total_records else 0,
        "severity": "MEDIUM",
        "status": (
            "WARNING" if duplicate_count > 0
            else "PASS"
        ),
        "description": "Exact duplicate transaction rows"
    })

    # Invalid dates
    invalid_dates = int(
        df["InvoiceDate"].isna().sum()
    )

    results.append({
        "control": "INVALID_INVOICE_DATES",
        "category": "STRUCTURAL",
        "affected_records": invalid_dates,
        "affected_percentage": round(
            invalid_dates / total_records * 100, 4
        ) if total_records else 0,
        "severity": "HIGH",
        "status": (
            "FAIL" if invalid_dates > 0
            else "PASS"
        ),
        "description": "Missing or unparseable invoice dates"
    })

    return results


# --------------------------------------------------
# 7. FINANCIAL QUALITY CHECKS
# --------------------------------------------------

def financial_checks(df):

    logger.info("Running financial quality checks...")

    results = []

    total_records = len(df)

    # Negative quantities
    negative_quantity = int(
        (df["Quantity"] < 0).sum()
    )

    results.append({
        "control": "NEGATIVE_QUANTITY",
        "category": "FINANCIAL",
        "affected_records": negative_quantity,
        "affected_percentage": round(
            negative_quantity / total_records * 100, 4
        ) if total_records else 0,
        "severity": "LOW",
        "status": (
            "WARNING" if negative_quantity > 0
            else "PASS"
        ),
        "description": "Negative quantities, potentially representing returns"
    })

    # Zero quantities
    zero_quantity = int(
        (df["Quantity"] == 0).sum()
    )

    results.append({
        "control": "ZERO_QUANTITY",
        "category": "FINANCIAL",
        "affected_records": zero_quantity,
        "affected_percentage": round(
            zero_quantity / total_records * 100, 4
        ) if total_records else 0,
        "severity": "MEDIUM",
        "status": (
            "WARNING" if zero_quantity > 0
            else "PASS"
        ),
        "description": "Transactions with zero quantity"
    })

    # Non-positive unit prices
    invalid_price = int(
        (df["UnitPrice"] <= 0).sum()
    )

    results.append({
        "control": "NON_POSITIVE_UNIT_PRICE",
        "category": "FINANCIAL",
        "affected_records": invalid_price,
        "affected_percentage": round(
            invalid_price / total_records * 100, 4
        ) if total_records else 0,
        "severity": "HIGH",
        "status": (
            "WARNING" if invalid_price > 0
            else "PASS"
        ),
        "description": "Zero or negative unit prices"
    })

    # Cancellation invoices
    cancellation_count = int(
        df["InvoiceNo"]
        .astype("string")
        .str.startswith("C", na=False)
        .sum()
    )

    results.append({
        "control": "CANCELLATION_INVOICES",
        "category": "FINANCIAL",
        "affected_records": cancellation_count,
        "affected_percentage": round(
            cancellation_count / total_records * 100, 4
        ) if total_records else 0,
        "severity": "LOW",
        "status": (
            "WARNING" if cancellation_count > 0
            else "PASS"
        ),
        "description": "Invoices identified as cancellations"
    })

    return results


# --------------------------------------------------
# 8. STATISTICAL OUTLIER CHECKS
# --------------------------------------------------

def statistical_checks(df):

    logger.info("Running statistical outlier checks...")

    results = []

    # IQR-based outlier detection
    # Apply only to positive quantities and prices

    variables = {
        "Quantity": df.loc[
            df["Quantity"] > 0, "Quantity"
        ],
        "UnitPrice": df.loc[
            df["UnitPrice"] > 0, "UnitPrice"
        ]
    }

    for column, series in variables.items():

        if len(series) == 0:
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)

        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outlier_count = int(
            (
                (series < lower_bound)
                | (series > upper_bound)
            ).sum()
        )

        results.append({
            "control": f"IQR_OUTLIER_{column.upper()}",
            "category": "STATISTICAL",
            "affected_records": outlier_count,
            "affected_percentage": round(
                outlier_count / len(series) * 100, 4
            ),
            "severity": "LOW",
            "status": (
                "WARNING" if outlier_count > 0
                else "PASS"
            ),
            "description": (
                f"Potential extreme values in {column}; "
                "requires investigation, not automatic rejection"
            )
        })

    return results


# --------------------------------------------------
# 9. CREATE RECORD-LEVEL FLAGS
# --------------------------------------------------

def create_quality_flags(df):

    logger.info("Creating record-level quality flags...")

    flagged = df.copy()

    flagged["is_duplicate"] = (
        flagged.duplicated(keep=False)
    )

    flagged["is_missing_customer"] = (
        flagged["CustomerID"].isna()
    )

    flagged["is_negative_quantity"] = (
        flagged["Quantity"] < 0
    )

    flagged["is_zero_quantity"] = (
        flagged["Quantity"] == 0
    )

    flagged["is_invalid_price"] = (
        flagged["UnitPrice"] <= 0
    )

    flagged["is_cancellation"] = (
        flagged["InvoiceNo"]
        .astype("string")
        .str.startswith("C", na=False)
    )

    flagged["is_invalid_date"] = (
        flagged["InvoiceDate"].isna()
    )

    # Positive-value IQR flags
    for column in ["Quantity", "UnitPrice"]:

        valid_values = flagged.loc[
            flagged[column] > 0, column
        ]

        if len(valid_values) == 0:
            flagged[f"is_{column.lower()}_outlier"] = False
            continue

        q1 = valid_values.quantile(0.25)
        q3 = valid_values.quantile(0.75)

        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        flagged[f"is_{column.lower()}_outlier"] = (
            (flagged[column] > 0)
            & (
                (flagged[column] < lower)
                | (flagged[column] > upper)
            )
        )

    flag_columns = [
        col for col in flagged.columns
        if col.startswith("is_")
    ]

    flagged["quality_issue_count"] = (
        flagged[flag_columns].sum(axis=1)
    )

    flagged_records = flagged.loc[
        flagged["quality_issue_count"] > 0
    ].copy()

    logger.info(
        f"Flagged records: {len(flagged_records):,}"
    )

    return flagged_records


# --------------------------------------------------
# 10. GENERATE QUALITY REPORT
# --------------------------------------------------

def generate_report(df, results):

    logger.info("Generating quality control report...")

    report = pd.DataFrame(results)

    report_file = (
        REPORT_DIR / "quality_control_report.csv"
    )

    report.to_csv(
        report_file,
        index=False
    )

    total_records = len(df)

    summary = {

        "project": "Anomaly Detection & Control Documentation Platform",

        "module": "03 - Data Quality Assessment",

        "total_records": total_records,

        "total_controls": int(len(report)),

        "passed_controls": int(
            (report["status"] == "PASS").sum()
        ),

        "warning_controls": int(
            (report["status"] == "WARNING").sum()
        ),

        "failed_controls": int(
            (report["status"] == "FAIL").sum()
        ),

        "overall_status": (
            "FAIL"
            if (report["status"] == "FAIL").any()
            else (
                "WARNING"
                if (report["status"] == "WARNING").any()
                else "PASS"
            )
        ),

        "quality_control_results": results
    }

    summary_file = (
        REPORT_DIR / "quality_summary.json"
    )

    with open(
        summary_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            summary,
            file,
            indent=4
        )

    logger.info("Quality reports saved successfully")

    return report, summary


# --------------------------------------------------
# 11. MAIN EXECUTION
# --------------------------------------------------

def main():

    logger.info("=" * 55)

    logger.info("STARTING MODULE 03")

    logger.info("=" * 55)

    df = load_data()

    validate_schema(df)

    results = []

    results.extend(structural_checks(df))

    results.extend(financial_checks(df))

    results.extend(statistical_checks(df))

    flagged_records = create_quality_flags(df)

    report, summary = generate_report(
        df,
        results
    )

    flagged_file = (
        ANOMALY_DIR / "quality_flagged_records.csv.gz"
    )

    flagged_records.to_csv(
        flagged_file,
        index=False,
        compression="gzip"
    )

    logger.info("=" * 55)

    logger.info("DATA QUALITY ASSESSMENT COMPLETED")

    logger.info(
        f"Total records: {summary['total_records']:,}"
    )

    logger.info(
        f"Passed controls: {summary['passed_controls']}"
    )

    logger.info(
        f"Warning controls: {summary['warning_controls']}"
    )

    logger.info(
        f"Failed controls: {summary['failed_controls']}"
    )

    logger.info(
        f"Overall status: {summary['overall_status']}"
    )

    logger.info("=" * 55)

    print("\nQUALITY CONTROL REPORT")

    print(
        report[
            [
                "control",
                "affected_records",
                "status",
                "severity"
            ]
        ].to_string(index=False)
    )

    print("\nQUALITY SUMMARY")

    print(
        json.dumps(summary, indent=4)
    )


if __name__ == "__main__":
    main()
