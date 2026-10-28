"""Rebuild data/sample from the simulation.

    python scripts/make_sample_data.py

Runs the simulation of the fictional company Atelier Verde SAS (fixed random
seed, so the output is always the same) and writes the CSV files the app reads.
"""
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SIM = ROOT / "scripts" / "simulation"
OUT = ROOT / "data" / "sample"
OUT.mkdir(parents=True, exist_ok=True)

subprocess.run([sys.executable, "export.py"], cwd=SIM, check=True, stdout=subprocess.DEVNULL)
T = lambda n: pd.read_pickle(SIM / "tables" / f"{n}.pkl")

cu = T("customers")
cu[["customer_id", "customer_name", "segment", "country", "payment_terms_days", "payment_method"]].to_csv(OUT / "customers.csv", index=False)

ar = T("ar_invoices")
status = ar.status.map({"Paid": "paid", "Open": "open", "Open – disputed": "disputed"})
pd.DataFrame({"invoice_id": ar.invoice_id, "customer_id": ar.customer_id, "issue_date": ar.issue_date,
              "due_date": ar.due_date, "amount": ar.amount_incl_vat, "paid_date": ar.paid_date,
              "status": status, "description": ar.description}).to_csv(OUT / "invoices.csv", index=False)

ob = T("order_book")
pd.DataFrame({"order_id": ob.order_id, "customer_id": ob.customer_id, "description": ob.description,
              "expected_invoice_date": ob.planned_invoice_date, "due_date": ob.planned_due_date,
              "amount": ob.amount_incl_vat, "probability": ob.probability}).to_csv(OUT / "orders.csv", index=False)

ap, po, sp = T("ap_bills"), T("committed_purchases"), T("scheduled_payments")
ap = ap[ap.status != "Paid"]
rows = [pd.DataFrame({"payment_id": ap.bill_id, "date": ap.scheduled_payment_date, "category": "Suppliers",
                      "counterparty": ap.supplier_name, "amount": ap.amount_incl_vat, "certainty": "bill received",
                      "deferrable": False, "uncertainty_pct": 0.0, "note": ap.description}),
        pd.DataFrame({"payment_id": po.po_id, "date": po.planned_payment_date, "category": "Suppliers",
                      "counterparty": po.supplier_name, "amount": po.amount_incl_vat, "certainty": "purchase order",
                      "deferrable": False, "uncertainty_pct": 0.0, "note": po.description})]
cat = sp.category.str.replace(r" \(.*\)$", "", regex=True)
cat = cat.where(~sp.category.str.startswith("Payroll"), "Salaries").where(~sp.category.str.startswith("Capex"), "Capex")
cat = cat.where(~sp.category.str.startswith("VAT"), "VAT")
unc = sp.apply(lambda r: 15.0 if "Card" in r.category else (10.0 if r.category.startswith("VAT") and "Estimate" in r.certainty else 0.0), axis=1)
rows.append(pd.DataFrame({"payment_id": sp.payment_id, "date": sp.date, "category": cat, "counterparty": sp.counterparty,
                          "amount": sp.amount_eur, "certainty": sp.certainty, "deferrable": sp.category.str.contains("discretionary"),
                          "uncertainty_pct": unc, "note": sp.note}))
pd.concat(rows).sort_values("date").to_csv(OUT / "planned_payments.csv", index=False)

tx = T("bank_transactions")
pd.DataFrame({"date": tx.date, "amount": tx.amount_eur, "category": tx.category, "counterparty": tx.counterparty_name,
              "reference": tx.reference}).to_csv(OUT / "bank_transactions.csv", index=False)

pnl = T("pnl_monthly")
pnl.to_csv(OUT / "profit_and_loss_monthly.csv", index=False)
ba = T("bank_accounts")
settings = {"company": "Atelier Verde SAS (fictional)", "as_of": "2026-10-09", "forecast_start": "2026-10-12",
            "opening_balance": round(float(ba.balance_eur.sum()), 2), "buffer": 100000,
            "overdraft_limit": float(ba.overdraft_limit_eur.sum()), "weeks": 13, "lookback_days": 365,
            "profit_for_period": float(pnl[pnl.basis == "Forecast"].net_profit.sum()),
            "profit_period_label": "Q4 2026 net profit (forecast, accrual basis)"}
(OUT / "settings.json").write_text(json.dumps(settings, indent=2))
print("sample data written to", OUT)
