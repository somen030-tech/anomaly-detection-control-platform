
"""
MODULE 05: FINANCIAL FEATURE ENGINEERING & CASH-FLOW ANALYTICS

Project: Anomaly Detection & Control Documentation Platform

Inputs:
    data/processed/sales_transactions.csv.gz
    data/processed/returns_transactions.csv.gz
    data/processed/cancellation_transactions.csv.gz

Outputs:
    data/processed/daily_financial_features.csv
    outputs/reports/financial_analytics_summary.json
    outputs/logs/financial_features.log

Important:
    Online Retail net sales are a transaction-based proxy.
    They are not actual bank cash-flow or settlement records.
"""

from pathlib import Path
import json
import logging
import sys

import numpy as np
import pandas as pd


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "processed"
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"
LOG_DIR = PROJECT_ROOT / "outputs" / "logs"

SALES_FILE = DATA_DIR / "sales_transactions.csv.gz"
RETURNS_FILE = DATA_DIR / "returns_transactions.csv.gz"
CANCELLATIONS_FILE = DATA_DIR / "cancellation_transactions.csv.gz"

OUTPUT_FILE = DATA_DIR / "daily_financial_features.csv"

SUMMARY_FILE = (
    REPORT_DIR / "financial_analytics_summary.json"
)

LOG_FILE = LOG_DIR / "financial_features.log"

SHORT_WINDOW = 7
LONG_WINDOW = 30
MIN_ROLLING_OBSERVATIONS = 7


# ============================================================
# 2. LOGGING AND DIRECTORY SETUP
# ============================================================

def create_directories():

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def setup_logging():

    create_directories()

    logger = logging.getLogger(
        "financial_feature_engineering"
    )

    logger.setLevel(logging.INFO)

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )

    file_handler = logging.FileHandler(
        LOG_FILE,
        encoding="utf-8"
    )

    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


logger = setup_logging()


# ============================================================
# 3. LOAD INPUT DATASETS
# ============================================================

def load_transaction_file(file_path, dataset_name):

    logger.info("Loading %s dataset...", dataset_name)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {file_path}"
        )

    df = pd.read_csv(
        file_path,
        compression="gzip",
        low_memory=False
    )

    logger.info(
        "%s records loaded: %s",
        dataset_name,
        f"{len(df):,}"
    )

    return df


def load_all_datasets():

    sales = load_transaction_file(
        SALES_FILE,
        "Sales"
    )

    returns = load_transaction_file(
        RETURNS_FILE,
        "Returns"
    )

    cancellations = load_transaction_file(
        CANCELLATIONS_FILE,
        "Cancellations"
    )

    return sales, returns, cancellations


# ============================================================
# 4. PREPARE TRANSACTION DATA
# ============================================================

def prepare_transaction_data(df, dataset_name):

    required_columns = [
        "InvoiceDate",
        "Quantity",
        "UnitPrice",
        "InvoiceNo"
    ]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{dataset_name} dataset is missing columns: "
            f"{missing_columns}"
        )

    df = df.copy()

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

    df["InvoiceNo"] = df["InvoiceNo"].astype("string")

    if "CustomerID" in df.columns:
        df["CustomerID"] = df["CustomerID"].astype("string")
    else:
        df["CustomerID"] = pd.Series(
            pd.NA,
            index=df.index,
            dtype="string"
        )

    original_count = len(df)

    df = df.dropna(
        subset=[
            "InvoiceDate",
            "Quantity",
            "UnitPrice"
        ]
    ).copy()

    removed_count = original_count - len(df)

    if removed_count > 0:
        logger.warning(
            "%s invalid records removed: %s",
            dataset_name,
            removed_count
        )

    df["transaction_value"] = (
        df["Quantity"] * df["UnitPrice"]
    )

    df["date"] = df["InvoiceDate"].dt.normalize()

    return df


# ============================================================
# 5. EMPTY AGGREGATION SCHEMAS
# ============================================================

def empty_returns_dataframe():

    return pd.DataFrame(
        columns=[
            "date",
            "return_value",
            "return_transaction_count",
            "return_invoice_count",
            "returned_units"
        ]
    )


def empty_cancellations_dataframe():

    return pd.DataFrame(
        columns=[
            "date",
            "cancellation_value",
            "cancellation_transaction_count",
            "cancellation_invoice_count",
            "cancelled_units"
        ]
    )


# ============================================================
# 6. AGGREGATE DAILY SALES
# ============================================================

def aggregate_sales(sales):

    logger.info("Aggregating daily sales metrics...")

    if sales.empty:
        raise ValueError(
            "Sales dataset is empty. Cannot build "
            "the daily financial dataset."
        )

    sales = prepare_transaction_data(
        sales,
        "Sales"
    )

    if sales.empty:
        raise ValueError(
            "No valid sales transactions remain after preparation."
        )

    daily_sales = (
        sales.groupby("date")
        .agg(
            gross_sales=(
                "transaction_value",
                "sum"
            ),
            sales_transaction_count=(
                "InvoiceNo",
                "count"
            ),
            sales_invoice_count=(
                "InvoiceNo",
                "nunique"
            ),
            unique_customers=(
                "CustomerID",
                "nunique"
            ),
            units_sold=(
                "Quantity",
                "sum"
            ),
            average_unit_price=(
                "UnitPrice",
                "mean"
            )
        )
        .reset_index()
    )

    logger.info(
        "Daily sales aggregation completed: %s dates",
        len(daily_sales)
    )

    return daily_sales


# ============================================================
# 7. AGGREGATE DAILY RETURNS
# ============================================================

def aggregate_returns(returns):

    logger.info("Aggregating daily returns metrics...")

    # FIX: Return the expected columns even when empty.
    if returns.empty:

        logger.warning(
            "Returns dataset contains zero records. "
            "Using an empty aggregation."
        )

        return empty_returns_dataframe()

    returns = prepare_transaction_data(
        returns,
        "Returns"
    )

    if returns.empty:

        logger.warning(
            "No valid return transactions remain after preparation."
        )

        return empty_returns_dataframe()

    returns["return_value_line"] = (
        returns["transaction_value"].abs()
    )

    returns["returned_units_line"] = (
        returns["Quantity"].abs()
    )

    daily_returns = (
        returns.groupby("date")
        .agg(
            return_value=(
                "return_value_line",
                "sum"
            ),
            return_transaction_count=(
                "InvoiceNo",
                "count"
            ),
            return_invoice_count=(
                "InvoiceNo",
                "nunique"
            ),
            returned_units=(
                "returned_units_line",
                "sum"
            )
        )
        .reset_index()
    )

    logger.info(
        "Daily returns aggregation completed: %s dates",
        len(daily_returns)
    )

    return daily_returns


# ============================================================
# 8. AGGREGATE DAILY CANCELLATIONS
# ============================================================

def aggregate_cancellations(cancellations):

    logger.info("Aggregating daily cancellation metrics...")

    # FIX: Return the expected columns even when empty.
    if cancellations.empty:

        logger.warning(
            "Cancellations dataset contains zero records. "
            "Using an empty aggregation."
        )

        return empty_cancellations_dataframe()

    cancellations = prepare_transaction_data(
        cancellations,
        "Cancellations"
    )

    if cancellations.empty:

        logger.warning(
            "No valid cancellation transactions remain "
            "after preparation."
        )

        return empty_cancellations_dataframe()

    cancellations["cancellation_value_line"] = (
        cancellations["transaction_value"].abs()
    )

    cancellations["cancelled_units_line"] = (
        cancellations["Quantity"].abs()
    )

    daily_cancellations = (
        cancellations.groupby("date")
        .agg(
            cancellation_value=(
                "cancellation_value_line",
                "sum"
            ),
            cancellation_transaction_count=(
                "InvoiceNo",
                "count"
            ),
            cancellation_invoice_count=(
                "InvoiceNo",
                "nunique"
            ),
            cancelled_units=(
                "cancelled_units_line",
                "sum"
            )
        )
        .reset_index()
    )

    logger.info(
        "Daily cancellations aggregation completed: %s dates",
        len(daily_cancellations)
    )

    return daily_cancellations


# ============================================================
# 9. CREATE DAILY FINANCIAL DATASET
# ============================================================

def create_daily_financial_dataset(
    sales,
    returns,
    cancellations
):

    logger.info("Creating daily financial dataset...")

    daily_sales = aggregate_sales(sales)

    daily_returns = aggregate_returns(returns)

    daily_cancellations = aggregate_cancellations(
        cancellations
    )

    # Collect all available dates.
    available_dates = []

    for df in [
        daily_sales,
        daily_returns,
        daily_cancellations
    ]:

        if not df.empty and "date" in df.columns:

            available_dates.extend(
                df["date"].dropna().tolist()
            )

    if not available_dates:
        raise ValueError(
            "No valid transaction dates found."
        )

    start_date = min(available_dates)
    end_date = max(available_dates)

    # Build a continuous daily calendar.
    calendar = pd.DataFrame({
        "date": pd.date_range(
            start=start_date,
            end=end_date,
            freq="D"
        )
    })

    daily = calendar.merge(
        daily_sales,
        on="date",
        how="left"
    )

    # These merges now work even if returns or
    # cancellations contain zero records.
    daily = daily.merge(
        daily_returns,
        on="date",
        how="left"
    )

    daily = daily.merge(
        daily_cancellations,
        on="date",
        how="left"
    )

    numeric_columns = [
        "gross_sales",
        "sales_transaction_count",
        "sales_invoice_count",
        "unique_customers",
        "units_sold",
        "average_unit_price",
        "return_value",
        "return_transaction_count",
        "return_invoice_count",
        "returned_units",
        "cancellation_value",
        "cancellation_transaction_count",
        "cancellation_invoice_count",
        "cancelled_units"
    ]

    for col in numeric_columns:

        if col not in daily.columns:
            daily[col] = 0

        daily[col] = daily[col].fillna(0)

    # Calculate net sales proxy.
    daily["net_sales"] = (
        daily["gross_sales"]
        - daily["return_value"]
        - daily["cancellation_value"]
    )

    # Average transaction value.
    daily["average_transaction_value"] = np.where(
        daily["sales_invoice_count"] > 0,
        daily["gross_sales"]
        / daily["sales_invoice_count"].replace(0, np.nan),
        0
    )

    daily["average_transaction_value"] = (
        daily["average_transaction_value"].fillna(0)
    )

    daily = daily.sort_values(
        "date"
    ).reset_index(drop=True)

    logger.info(
        "Daily financial dataset created: %s observations",
        len(daily)
    )

    return daily


# ============================================================
# 10. SAFE PERCENTAGE CHANGE
# ============================================================


def calculate_safe_percentage_change(series):
    """
    Calculate percentage change safely using NumPy.

    Handles:
    - Previous-day value equal to zero
    - Missing values
    - Pandas nullable numeric types
    - Division-by-zero errors

    Undefined percentage changes are stored as NaN.
    """

    # Convert Pandas Series into a standard NumPy float array.
    values = pd.to_numeric(
        series,
        errors="coerce"
    ).to_numpy(
        dtype=np.float64,
        na_value=np.nan
    )

    # Handle empty input safely.
    if len(values) == 0:
        return pd.Series(
            percentage_changes,
            index=series.index,
            name=series.name,
            dtype="float64"
        )

    # Create an array containing previous-day values.
    previous_values = np.empty_like(values)

    previous_values[0] = np.nan
    previous_values[1:] = values[:-1]

    # Initialize all percentage changes as NaN.
    percentage_changes = np.full(
        len(values),
        np.nan,
        dtype=np.float64
    )

    # Only calculate when both values are valid
    # and the previous value is not zero.
    valid_mask = (
        np.isfinite(values)
        & np.isfinite(previous_values)
        & (previous_values != 0)
    )

    # Calculate percentage changes using NumPy.
    percentage_changes[valid_mask] = (
        (
            values[valid_mask]
            - previous_values[valid_mask]
        )
        / previous_values[valid_mask]
    ) * 100

    # Return a regular float64 Pandas Series.
    return pd.Series(
        percentage_changes,
        index=series.index,
        name=series.name,
        dtype="float64"
    )

# ============================================================
# 11. ENGINEER TIME-SERIES FEATURES
# ============================================================

def engineer_time_series_features(daily):

    logger.info("Engineering time-series features...")

    daily = daily.copy()

    daily["date"] = pd.to_datetime(
        daily["date"],
        errors="coerce"
    )

    daily = daily.sort_values(
        "date"
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # A. PERCENTAGE CHANGE
    # --------------------------------------------------------

    daily["gross_sales_pct_change"] = (
        calculate_safe_percentage_change(
            daily["gross_sales"]
        )
    )

    daily["net_sales_pct_change"] = (
        calculate_safe_percentage_change(
            daily["net_sales"]
        )
    )

    # --------------------------------------------------------
    # B. PRIOR-DAY ROLLING AVERAGES
    # --------------------------------------------------------

    daily["gross_sales_7d_avg"] = (
        daily["gross_sales"]
        .rolling(
            window=SHORT_WINDOW,
            min_periods=MIN_ROLLING_OBSERVATIONS
        )
        .mean()
        .shift(1)
    )

    daily["gross_sales_30d_avg"] = (
        daily["gross_sales"]
        .rolling(
            window=LONG_WINDOW,
            min_periods=MIN_ROLLING_OBSERVATIONS
        )
        .mean()
        .shift(1)
    )

    daily["net_sales_7d_avg"] = (
        daily["net_sales"]
        .rolling(
            window=SHORT_WINDOW,
            min_periods=MIN_ROLLING_OBSERVATIONS
        )
        .mean()
        .shift(1)
    )

    daily["net_sales_30d_avg"] = (
        daily["net_sales"]
        .rolling(
            window=LONG_WINDOW,
            min_periods=MIN_ROLLING_OBSERVATIONS
        )
        .mean()
        .shift(1)
    )

    # --------------------------------------------------------
    # C. PRIOR-DAY ROLLING STANDARD DEVIATION
    # --------------------------------------------------------

    daily["net_sales_7d_std"] = (
        daily["net_sales"]
        .rolling(
            window=SHORT_WINDOW,
            min_periods=MIN_ROLLING_OBSERVATIONS
        )
        .std()
        .shift(1)
    )

    daily["net_sales_30d_std"] = (
        daily["net_sales"]
        .rolling(
            window=LONG_WINDOW,
            min_periods=MIN_ROLLING_OBSERVATIONS
        )
        .std()
        .shift(1)
    )

    # --------------------------------------------------------
    # D. CASH-FLOW DROP PERCENTAGE
    # --------------------------------------------------------

    valid_baseline = (
        daily["net_sales_30d_avg"].notna()
        & (daily["net_sales_30d_avg"] > 0)
    )

    # Convert inputs to NumPy float64 arrays before calculation.
    # Assign complete arrays back to the DataFrame to avoid
    # Pandas dtype/index coercion issues during masked .loc assignment.
    net_sales_values = pd.to_numeric(
        daily["net_sales"],
        errors="coerce"
    ).to_numpy(dtype=np.float64, na_value=np.nan)

    baseline_values = pd.to_numeric(
        daily["net_sales_30d_avg"],
        errors="coerce"
    ).to_numpy(dtype=np.float64, na_value=np.nan)

    valid_baseline_array = (
        np.isfinite(baseline_values)
        & (baseline_values > 0)
        & np.isfinite(net_sales_values)
    )

    cash_flow_drop_values = np.full(
        len(daily),
        np.nan,
        dtype=np.float64
    )

    cash_flow_drop_values[valid_baseline_array] = (
        (
            baseline_values[valid_baseline_array]
            - net_sales_values[valid_baseline_array]
        )
        / baseline_values[valid_baseline_array]
    ) * 100.0

    daily["cash_flow_drop_pct"] = cash_flow_drop_values

    # --------------------------------------------------------
    # E. NET SALES 30-DAY Z-SCORE
    # --------------------------------------------------------

    std_values = pd.to_numeric(
        daily["net_sales_30d_std"],
        errors="coerce"
    ).to_numpy(dtype=np.float64, na_value=np.nan)

    valid_std_array = (
        np.isfinite(std_values)
        & (std_values > 0)
        & np.isfinite(baseline_values)
        & np.isfinite(net_sales_values)
    )

    zscore_values = np.full(
        len(daily),
        np.nan,
        dtype=np.float64
    )

    zscore_values[valid_std_array] = (
        net_sales_values[valid_std_array]
        - baseline_values[valid_std_array]
    ) / std_values[valid_std_array]

    daily["net_sales_30d_zscore"] = zscore_values

    # --------------------------------------------------------
    # F. ACTIVITY AND CONTROL FLAGS
    # --------------------------------------------------------

    daily["sales_activity_flag"] = (
        daily["sales_transaction_count"] > 0
    ).astype(int)

    daily["negative_net_sales_flag"] = (
        daily["net_sales"] < 0
    ).astype(int)

    daily["below_30d_baseline_flag"] = (
        daily["net_sales_30d_avg"].notna()
        & (
            daily["net_sales"]
            < daily["net_sales_30d_avg"]
        )
    ).astype(int)

    # --------------------------------------------------------
    # G. REMOVE INFINITY VALUES
    # --------------------------------------------------------

    daily.replace(
        [np.inf, -np.inf],
        np.nan,
        inplace=True
    )

    logger.info(
        "Time-series feature engineering completed."
    )

    return daily


# ============================================================
# 12. VALIDATE DAILY FEATURES
# ============================================================

def validate_daily_features(daily):

    logger.info("Validating daily financial features...")

    if daily.empty:
        raise ValueError(
            "Daily financial dataset is empty."
        )

    if daily["date"].isna().any():
        raise ValueError(
            "Invalid dates found in daily financial dataset."
        )

    if daily["date"].duplicated().any():
        raise ValueError(
            "Duplicate dates found in daily financial dataset."
        )

    if not daily["date"].is_monotonic_increasing:
        raise ValueError(
            "Daily dates are not in chronological order."
        )

    required_columns = [
        "gross_sales",
        "net_sales",
        "return_value",
        "cancellation_value",
        "net_sales_30d_avg",
        "net_sales_30d_zscore",
        "cash_flow_drop_pct"
    ]

    missing_columns = [
        col for col in required_columns
        if col not in daily.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required features: {missing_columns}"
        )

    if np.isinf(
        daily.select_dtypes(include=[np.number]).to_numpy()
    ).any():
        raise ValueError(
            "Infinite numeric values found in final dataset."
        )

    logger.info(
        "Validation passed: %s daily observations",
        len(daily)
    )

    return True


# ============================================================
# 13. CREATE FINANCIAL ANALYTICS SUMMARY
# ============================================================

def create_financial_summary(daily):

    logger.info("Creating financial analytics summary...")

    summary = {

        "project": (
            "Anomaly Detection & Control Documentation Platform"
        ),

        "module": "Financial Feature Engineering",

        "dataset": "UCI Online Retail",

        "note": (
            "Net sales is a transaction-based proxy, "
            "not actual bank cash flow."
        ),

        "period": {
            "start_date": daily["date"].min().strftime("%Y-%m-%d"),
            "end_date": daily["date"].max().strftime("%Y-%m-%d")
        },

        "record_count": int(len(daily)),

        "financial_metrics": {

            "total_gross_sales": float(
                daily["gross_sales"].sum()
            ),

            "total_returns": float(
                daily["return_value"].sum()
            ),

            "total_cancellations": float(
                daily["cancellation_value"].sum()
            ),

            "total_net_sales": float(
                daily["net_sales"].sum()
            ),

            "average_daily_gross_sales": float(
                daily["gross_sales"].mean()
            ),

            "average_daily_net_sales": float(
                daily["net_sales"].mean()
            ),

            "maximum_daily_net_sales": float(
                daily["net_sales"].max()
            ),

            "minimum_daily_net_sales": float(
                daily["net_sales"].min()
            )
        },

        "activity_metrics": {

            "total_sales_transactions": int(
                daily["sales_transaction_count"].sum()
            ),

            "total_sales_invoices": int(
                daily["sales_invoice_count"].sum()
            ),

            "total_return_transactions": int(
                daily["return_transaction_count"].sum()
            ),

            "total_cancellation_transactions": int(
                daily["cancellation_transaction_count"].sum()
            ),

            "days_with_sales_activity": int(
                daily["sales_activity_flag"].sum()
            ),

            "days_with_negative_net_sales": int(
                daily["negative_net_sales_flag"].sum()
            ),

            "days_below_30d_baseline": int(
                daily["below_30d_baseline_flag"].sum()
            )
        },

        "feature_metrics": {

            "rolling_short_window": SHORT_WINDOW,

            "rolling_long_window": LONG_WINDOW,

            "days_with_30d_baseline": int(
                daily["net_sales_30d_avg"].notna().sum()
            ),

            "days_with_valid_30d_zscore": int(
                daily["net_sales_30d_zscore"].notna().sum()
            ),

            "days_with_valid_cash_flow_drop": int(
                daily["cash_flow_drop_pct"].notna().sum()
            ),

            "undefined_net_sales_pct_changes": int(
                daily["net_sales_pct_change"].isna().sum()
            )
        }
    }

    with open(
        SUMMARY_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            summary,
            file,
            indent=4,
            allow_nan=False
        )

    logger.info(
        "Financial analytics summary saved: %s",
        SUMMARY_FILE
    )

    return summary


# ============================================================
# 14. SAVE DAILY FINANCIAL FEATURES
# ============================================================

def save_daily_features(daily):

    logger.info("Saving daily financial features...")

    daily.to_csv(
        OUTPUT_FILE,
        index=False,
        date_format="%Y-%m-%d"
    )

    logger.info(
        "Daily financial features saved: %s",
        OUTPUT_FILE
    )

    logger.info(
        "Output dimensions: %s rows, %s columns",
        daily.shape[0],
        daily.shape[1]
    )


# ============================================================
# 15. MAIN EXECUTION
# ============================================================

def main():

    logger.info("=" * 65)

    logger.info(
        "STARTING MODULE 05: FINANCIAL FEATURE ENGINEERING"
    )

    logger.info("=" * 65)

    try:

        # Step 1: Load datasets.
        sales, returns, cancellations = load_all_datasets()

        # Step 2: Create daily financial metrics.
        daily = create_daily_financial_dataset(
            sales,
            returns,
            cancellations
        )

        # Step 3: Engineer time-series features.
        daily = engineer_time_series_features(daily)

        # Step 4: Validate final dataset.
        validate_daily_features(daily)

        # Step 5: Save daily feature dataset.
        save_daily_features(daily)

        # Step 6: Create JSON summary.
        summary = create_financial_summary(daily)

        logger.info("=" * 65)

        logger.info(
            "MODULE 05 COMPLETED SUCCESSFULLY"
        )

        logger.info("=" * 65)

        logger.info(
            "Total daily observations: %s",
            summary["record_count"]
        )

        logger.info(
            "Total net sales: %.2f",
            summary["financial_metrics"]["total_net_sales"]
        )

        logger.info(
            "Output file: %s",
            OUTPUT_FILE
        )

        logger.info(
            "Summary file: %s",
            SUMMARY_FILE
        )

    except Exception:

        logger.exception(
            "Module 05 failed during execution."
        )

        raise


if __name__ == "__main__":
    main()
