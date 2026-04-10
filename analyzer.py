import pandas as pd
import matplotlib.pyplot as plt


def load_data(file_path: str) -> pd.DataFrame:
    """Load financial data from a CSV file."""
    df = pd.read_csv(file_path)
    return df


def calculate_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate core financial metrics."""
    df = df.copy()

    df["profit_margin"] = df["profit"] / df["revenue"]
    df["expense_ratio"] = df["expenses"] / df["revenue"]
    df["revenue_growth"] = df["revenue"].pct_change()
    df["profit_growth"] = df["profit"].pct_change()

    return df


def detect_anomalies(df: pd.DataFrame) -> list[str]:
    """Detect unusual financial patterns."""
    warnings = []

    for i in range(1, len(df)):
        current_year = int(df.loc[i, "year"])

        prev_revenue = df.loc[i - 1, "revenue"]
        current_revenue = df.loc[i, "revenue"]

        prev_profit = df.loc[i - 1, "profit"]
        current_profit = df.loc[i, "profit"]

        prev_expenses = df.loc[i - 1, "expenses"]
        current_expenses = df.loc[i, "expenses"]

        current_profit_margin = df.loc[i, "profit_margin"]

        if current_profit < prev_profit * 0.5:
            warnings.append(
                f"Warning: Profit dropped significantly in {current_year}."
            )

        if current_expenses > prev_expenses * 1.3:
            warnings.append(
                f"Warning: Expense spike detected in {current_year}."
            )

        if current_revenue < prev_revenue * 0.7:
            warnings.append(
                f"Warning: Revenue dropped significantly in {current_year}."
            )

        if current_profit_margin < 0.10:
            warnings.append(
                f"Warning: Low profit margin in {current_year}."
            )

    return warnings


def print_report(df: pd.DataFrame, warnings: list[str]) -> None:
    """Print a simple audit-style report."""
    print("\nFINANCIAL STATEMENT ANALYSIS REPORT")
    print("-" * 60)

    for _, row in df.iterrows():
        print(
            f"Year: {int(row['year'])} | Revenue: {row['revenue']} | "
            f"Expenses: {row['expenses']} | Profit: {row['profit']} | "
            f"Profit Margin: {row['profit_margin']:.2%}"
        )

    print("\nANOMALY CHECK")
    print("-" * 60)

    if warnings:
        for warning in warnings:
            print(warning)
    else:
        print("No major anomalies detected.")


def plot_data(df: pd.DataFrame) -> None:
    """Plot revenue, expenses, and profit over time."""
    plt.figure()
    plt.plot(df["year"], df["revenue"], label="Revenue")
    plt.plot(df["year"], df["expenses"], label="Expenses")
    plt.plot(df["year"], df["profit"], label="Profit")
    plt.title("Financial Performance Over Time")
    plt.xlabel("Year")
    plt.ylabel("Amount")
    plt.legend()
    plt.tight_layout()
    plt.show()


def main() -> None:
    file_path = "data/financials.csv"

    df = load_data(file_path)
    df = calculate_metrics(df)
    warnings = detect_anomalies(df)

    print_report(df, warnings)
    plot_data(df)


if __name__ == "__main__":
    main()