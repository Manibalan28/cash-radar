"""Simulation of the fictional company Atelier Verde SAS (part 2: publish tables). Run via scripts/make_sample_data.py."""
"""Turn simulated data into the published dataset tables (CSV + pickles for the workbook)."""
import pandas as pd, numpy as np, datetime as dt, os, json
from generate import CUTOFF, FC_START, FC_END, PUBLISH_START, customers, suppliers, inv, bills, oth, vat, led, rng

OUT = os.path.dirname(os.path.abspath(__file__)) + "/tables"
SECRET = os.path.dirname(os.path.abspath(__file__)) + "/answer_key"
os.makedirs(OUT, exist_ok=True); os.makedirs(SECRET, exist_ok=True)
D = lambda s: pd.to_datetime(s)

# ---------------- Bank accounts at cutoff
total = round(led[led.date <= CUTOFF].balance.iloc[-1], 2)
main = round(total - 37411.25, 2)
bank_accounts = pd.DataFrame([
    dict(account_id="BA01", bank="Banque Régionale (fictional)", account_type="Main current account",
         iban_masked="FR76 •••• •••• •••• 4821", currency="EUR", balance_eur=main, as_at="2026-10-09 18:00",
         overdraft_limit_eur=150000, overdraft_rate="EURIBOR 3M + 2.5% (≈4.6% p.a.)", source_channel="EBICS"),
    dict(account_id="BA02", bank="Banque Mutuelle (fictional)", account_type="Secondary current account (card spending)",
         iban_masked="FR76 •••• •••• •••• 0937", currency="EUR", balance_eur=37411.25, as_at="2026-10-09 18:00",
         overdraft_limit_eur=0, overdraft_rate="n/a", source_channel="PSD2 API"),
])

# ---------------- Bank transactions (published history)
names = dict(customers.set_index("customer_id").customer_name) | dict(suppliers.set_index("supplier_id").supplier_name)
tx = led[(led.date >= PUBLISH_START) & (led.date <= CUTOFF)].copy()
tx["counterparty_name"] = tx.counterparty.map(lambda c: names.get(c, c))
tx["direction"] = np.where(tx.amount > 0, "Inflow", "Outflow")
tx = tx.rename(columns={"amount": "amount_eur", "balance": "balance_after_eur", "ref": "reference"})
tx = tx[["date","direction","category","counterparty_name","reference","amount_eur","balance_after_eur"]]
tx.insert(0, "transaction_id", [f"TX{i+1:05d}" for i in range(len(tx))])
tx["amount_eur"] = tx.amount_eur.round(2); tx["balance_after_eur"] = tx.balance_after_eur.round(2)

# ---------------- Weekly bank history (Mon-Sun weeks)
l2 = led[(led.date >= PUBLISH_START) & (led.date <= CUTOFF)].copy()
l2["week_start"] = l2.date.map(lambda d: d - dt.timedelta(days=d.weekday()))
open_bal = led[led.date < PUBLISH_START].balance.iloc[-1]
wk = l2.groupby("week_start").agg(receipts_eur=("amount", lambda s: s[s > 0].sum()),
                                  payments_eur=("amount", lambda s: -s[s < 0].sum())).reset_index()
allw = pd.DataFrame({"week_start": [PUBLISH_START - dt.timedelta(days=PUBLISH_START.weekday()) + dt.timedelta(weeks=i)
                                    for i in range(0, 200)]})
allw = allw[allw.week_start <= CUTOFF]
wk = allw.merge(wk, how="left").fillna(0)
wk["opening_balance_eur"] = 0.0; wk["closing_balance_eur"] = 0.0
b = open_bal
for i in wk.index:
    wk.at[i, "opening_balance_eur"] = b; b = b + wk.at[i, "receipts_eur"] - wk.at[i, "payments_eur"]; wk.at[i, "closing_balance_eur"] = b
wk["week_end"] = wk.week_start + dt.timedelta(days=6)
wk["iso_week"] = wk.week_start.map(lambda d: f"{d.isocalendar()[0]}-W{d.isocalendar()[1]:02d}")
wk = wk[["iso_week","week_start","week_end","opening_balance_eur","receipts_eur","payments_eur","closing_balance_eur"]].round(2)
assert abs(wk.closing_balance_eur.iloc[-1] - total) < 0.01

# ---------------- AR invoices (issued up to cutoff)
cust_name = customers.set_index("customer_id").customer_name
ar = inv[inv.issue_date <= CUTOFF].copy()
ar["customer_name"] = ar.customer_id.map(cust_name)
paid = ar.true_pay_date <= CUTOFF
ar["status"] = np.where(paid, "Paid", "Open")
ar["paid_date"] = np.where(paid, ar.true_pay_date, pd.NaT)
ar["days_late"] = [ (p - d).days if s == "Paid" else None for p, d, s in zip(ar.true_pay_date, ar.due_date, ar.status)]
ar["days_overdue_at_cutoff"] = [ max(0, (CUTOFF - d).days) if s == "Open" else None for d, s in zip(ar.due_date, ar.status)]
# two open invoices flagged as disputed by the customer
op = ar[(ar.status == "Open") & (ar.customer_id.isin(["C04", "C09"])) & (ar.due_date < CUTOFF)]
for cid in ["C04", "C09"]:
    cand = op[op.customer_id == cid]
    if len(cand): ar.loc[cand.index[0], "status"] = "Open – disputed"
ar["dispute_note"] = ""
ar.loc[(ar.status == "Open – disputed") & (ar.customer_id == "C04"), "dispute_note"] = "Customer claims 3 desks damaged in transport; credit note under review"
ar.loc[(ar.status == "Open – disputed") & (ar.customer_id == "C09"), "dispute_note"] = "Invoice rejected in Chorus Pro: missing purchase-order number; to be re-submitted"
ar_pub = ar[["invoice_id","customer_id","customer_name","issue_date","due_date","amount_ex_vat","vat","amount_incl_vat",
             "description","project","status","paid_date","days_late","days_overdue_at_cutoff","dispute_note"]]
# full invoice history (from Dec 2024) so payment habits can be learned from the start

# ---------------- Order book: invoices planned in the forecast window
ob = inv[(inv.issue_date > CUTOFF) & (inv.issue_date <= FC_END)].copy()
ob["customer_name"] = ob.customer_id.map(cust_name)
ob["order_status"] = np.where(ob.project.notna(), "Contract signed – milestone",
                     np.where(ob.issue_date <= FC_START + dt.timedelta(weeks=6), "Confirmed order", "Pipeline (expected, not yet ordered)"))
ob["probability"] = np.where(ob.order_status == "Pipeline (expected, not yet ordered)", 0.8, 1.0)
ob["planned_invoice_date"] = ob.issue_date
ob["planned_due_date"] = ob.due_date
ob = ob.reset_index(drop=True)
ob.insert(0, "order_id", [f"ORD-{2600+i+1}" for i in range(len(ob))])
ob_pub = ob[["order_id","customer_id","customer_name","description","project","order_status","probability",
             "planned_invoice_date","planned_due_date","amount_ex_vat","vat","amount_incl_vat"]]

# ---------------- AP bills
sup_name = suppliers.set_index("supplier_id").supplier_name
ap = bills[bills.bill_date <= CUTOFF].copy()
ap["supplier_name"] = ap.supplier_id.map(sup_name)
p2 = ap.pay_date_sim <= CUTOFF
ap["status"] = np.where(p2, "Paid", "Open – scheduled")
ap["paid_date"] = np.where(p2, ap.pay_date_sim, pd.NaT)
ap["scheduled_payment_date"] = np.where(~p2, ap.pay_date_sim, pd.NaT)
ap_pub = ap[["bill_id","supplier_id","supplier_name","bill_date","due_date","amount_ex_vat","vat","amount_incl_vat",
             "description","status","paid_date","scheduled_payment_date"]]
ap_pub = ap_pub[ap_pub.bill_date >= dt.date(2025,1,1)]

# committed purchase orders: bills expected in the window
po = bills[(bills.bill_date > CUTOFF) & (bills.pay_date_sim <= FC_END)].copy()
po["supplier_name"] = po.supplier_id.map(sup_name)
po = po.reset_index(drop=True)
po.insert(0, "po_id", [f"PO-{2600+i+1}" for i in range(len(po))])
po["po_status"] = np.where(po.description.str.startswith("Winter"), "PO confirmed 28 Sep 2026 – delivery 21 Oct",
                   np.where(po.bill_date <= FC_START + dt.timedelta(weeks=6), "PO confirmed", "Expected (based on order book)"))
po_pub = po[["po_id","supplier_id","supplier_name","description","po_status","bill_date","due_date","pay_date_sim",
             "amount_ex_vat","vat","amount_incl_vat"]].rename(columns={"bill_date":"expected_bill_date","pay_date_sim":"planned_payment_date"})

# ---------------- Scheduled & recurring payments in the window
sp = oth[(oth.date > CUTOFF) & (oth.date <= FC_END)].copy()
sp["amount_eur"] = -sp.amount
is_card = sp.category.str.contains("Card")
sp.loc[is_card, "amount_eur"] = [1800.0 if (d.month == 12 and d.day > 23) else 4700.0 for d in sp[is_card].date]
sp.loc[is_card, "note"] = "Run-rate estimate (avg of last 26 weeks, rounded)"
sp["certainty"] = np.where(sp.category.str.contains("Card"), "Estimate (run-rate)",
                  np.where(sp.category.str.contains("discretionary"), "Planned – can be deferred", "Contractual / statutory"))
v = vat[(vat.payment_date > CUTOFF) & (vat.payment_date <= FC_END)].copy()
vrows = pd.DataFrame(dict(date=v.payment_date, category="VAT payment (monthly CA3 return)", counterparty="DGFiP",
                          amount_eur=v.net_vat_payable.round(-1), note=["VAT for " + p for p in v.period],
                          certainty=["Known (return prepared)" if p == "2026-09" else "Estimate (from invoices & bills)" for p in v.period]))
sp = pd.concat([sp[["date","category","counterparty","amount_eur","note","certainty"]], vrows]).sort_values("date").reset_index(drop=True)
sp.insert(0, "payment_id", [f"PAY-{i+1:03d}" for i in range(len(sp))])
sp["amount_eur"] = sp.amount_eur.round(2)

# ---------------- VAT history
vat_pub = vat[(vat.payment_date >= PUBLISH_START) & (vat.payment_date <= CUTOFF)].round(2)

# ---------------- Monthly P&L (accrual basis) Apr 2025 - Dec 2026 (Oct-Dec 2026 = forecast)
months = pd.period_range("2025-04", "2026-12", freq="M")
pnl = []
proj_rev_tot = inv[inv.project.notna()].amount_ex_vat.sum()
proj_mat_tot = 58000+24000+71000+29000+14000+22000+155000*0.55+32000   # part of the big order is winter stock
for p in months:
    r_reg = inv[(inv.project.isna()) & (inv.issue_date.map(lambda d: d.year) == p.year) & (inv.issue_date.map(lambda d: d.month) == p.month)].amount_ex_vat.sum()
    r_prj = inv[(inv.project.notna()) & (inv.issue_date.map(lambda d: d.year) == p.year) & (inv.issue_date.map(lambda d: d.month) == p.month)].amount_ex_vat.sum()
    rev = r_reg + r_prj
    mat = r_reg * 0.439 + r_prj * 0.42
    transport = rev * 0.03
    y26 = p.year == 2026
    staff = (106000 + 86000 if y26 else 101500 + 82000) + ((72000 + 52000) / 12 if y26 else (68000 + 52000) / 12)  # 13th-month bonus accrued monthly
    rent = 6500; energy = 11800 if p.month in (11,12,1,2) else 8200
    other = 2400 + 3200 + 2600 + 4700*52/12/1.2 + 9000   # IT, lease, insurance, sundry, professional fees
    ebitda = rev - mat - transport - staff - rent - energy - other
    da = 16500; interest = 1600 if y26 else 1900
    ebt = ebitda - da - interest
    tax = max(0, ebt) * 0.25
    pnl.append(dict(month=str(p), basis="Forecast" if p >= pd.Period("2026-10") else "Actual",
                    revenue=rev, materials=-mat, transport=-transport, staff_costs=-staff, rent=-rent, energy=-energy,
                    other_overheads=-other, ebitda=ebitda, depreciation=-da, interest=-interest, profit_before_tax=ebt,
                    corporate_tax=-tax, net_profit=ebt - tax))
pnl = pd.DataFrame(pnl).round(0)

# ---------------- answer key (NOT published in the workbook): true pay dates for open AR & order book
key = pd.concat([ar[ar.status != "Paid"][["invoice_id","customer_id","due_date","amount_incl_vat","true_pay_date"]],
                 ob.assign(invoice_id=ob.order_id)[["invoice_id","customer_id","due_date","amount_incl_vat","true_pay_date"]]])
key.to_csv(f"{SECRET}/true_receipt_dates.csv", index=False)
led[(led.date > CUTOFF) & (led.date <= FC_END)].to_csv(f"{SECRET}/true_window_ledger.csv", index=False)

tables = dict(bank_accounts=bank_accounts, bank_weekly_history=wk, bank_transactions=tx,
              customers=customers.drop(columns=["late_mean","late_sd","revenue_share","avg_invoice_ex_vat"]),
              ar_invoices=ar_pub, order_book=ob_pub, suppliers=suppliers.drop(columns=["pct_of_revenue"]),
              ap_bills=ap_pub, committed_purchases=po_pub, scheduled_payments=sp, vat_returns=vat_pub, pnl_monthly=pnl)
for k, df in tables.items():
    df.to_csv(f"{OUT}/{k}.csv", index=False); df.to_pickle(f"{OUT}/{k}.pkl")
    print(f"{k:22s} {len(df):5d} rows")
print("cash at cutoff", total)
