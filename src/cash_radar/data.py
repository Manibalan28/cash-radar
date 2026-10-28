"""Load a company's data from a folder of CSV files (see data/README.md for the schema)."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .engine import Settings, to_date

SAMPLE_DIR = Path(__file__).resolve().parents[2] / "data" / "sample"

REQUIRED = {
    "customers": ["customer_id", "customer_name", "segment", "payment_terms_days"],
    "invoices": ["invoice_id", "customer_id", "issue_date", "due_date", "amount", "paid_date", "status"],
    "orders": ["order_id", "customer_id", "due_date", "amount", "probability"],
    "planned_payments": ["payment_id", "date", "category", "counterparty", "amount"],
    "bank_transactions": ["date", "amount", "category", "counterparty"],
}
DEFAULTS = {
    "invoices": {"description": ""},
    "orders": {"description": ""},
    "planned_payments": {"certainty": "fixed", "deferrable": False, "uncertainty_pct": 0.0, "note": ""},
}


@dataclass
class Dataset:
    settings: Settings
    customers: pd.DataFrame
    invoices: pd.DataFrame
    orders: pd.DataFrame
    planned_payments: pd.DataFrame
    bank_transactions: pd.DataFrame
    profit_for_period: float | None = None


def clean(name: str, df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED[name] if c not in df.columns]
    if missing:
        raise ValueError(f"{name}.csv is missing column(s): {', '.join(missing)}")
    df = df.copy()
    for col, val in DEFAULTS.get(name, {}).items():
        if col not in df.columns:
            df[col] = val
    for c in ("issue_date", "due_date", "date"):
        if c in df.columns:
            df[c] = df[c].map(to_date)
    if "paid_date" in df.columns:
        df["paid_date"] = df["paid_date"].map(lambda x: None if pd.isna(x) or x == "" else to_date(x))
    if "deferrable" in df.columns:
        df["deferrable"] = df["deferrable"].astype(str).str.lower().isin(["true", "1", "yes"])
    if "status" in df.columns:
        df["status"] = df["status"].fillna("open").str.lower()
    for c in ("amount", "probability", "uncertainty_pct", "payment_terms_days"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="raise")
    for c in ("customer_id", "invoice_id", "order_id", "payment_id"):
        if c in df.columns:
            df[c] = df[c].astype(str)
    return df


def settings_from_json(d: dict) -> Settings:
    return Settings(as_of=to_date(d["as_of"]), forecast_start=to_date(d["forecast_start"]),
                    opening_balance=float(d["opening_balance"]), buffer=float(d.get("buffer", 100_000)),
                    overdraft_limit=float(d.get("overdraft_limit", 0)), weeks=int(d.get("weeks", 13)),
                    lookback_days=int(d.get("lookback_days", 365)), company=d.get("company", ""))


def load(folder: str | Path = SAMPLE_DIR) -> Dataset:
    folder = Path(folder)
    meta = json.loads((folder / "settings.json").read_text())
    tables = {n: clean(n, pd.read_csv(folder / f"{n}.csv")) for n in REQUIRED}
    return Dataset(settings=settings_from_json(meta), profit_for_period=meta.get("profit_for_period"), **tables)
