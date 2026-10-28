"""
Fictional dataset generator for 'Atelier Verde SAS' (Financial Navigator case study).
ALL DATA IS FICTIONAL. Simulates 18 months of history (Apr 2025 - 9 Oct 2026) plus
the confirmed commitments for the 13-week forecast window (12 Oct 2026 - 10 Jan 2027).
"""
import numpy as np, pandas as pd, datetime as dt, json, os

rng = np.random.default_rng(2026)
OUT = os.path.dirname(os.path.abspath(__file__)) + "/out"
os.makedirs(OUT, exist_ok=True)

START = dt.date(2024, 12, 1)       # simulation warm-up start
PUBLISH_START = dt.date(2025, 4, 1) # history shown in the workbook starts here
CUTOFF = dt.date(2026, 10, 9)          # data "as at" Friday 9 Oct 2026, close of business
FC_START = dt.date(2026, 10, 12)       # forecast week 1 = Mon 12 Oct 2026
FC_END = FC_START + dt.timedelta(weeks=13) - dt.timedelta(days=1)  # Sun 10 Jan 2027
SIM_END = dt.date(2027, 3, 31)
VAT = 0.20

HOLIDAYS = {dt.date(2024,12,25), dt.date(2025,1,1), dt.date(2025,4,21), dt.date(2025,5,1), dt.date(2025,5,8), dt.date(2025,5,29),
            dt.date(2025,6,9), dt.date(2025,7,14), dt.date(2025,8,15), dt.date(2025,11,1),
            dt.date(2025,11,11), dt.date(2025,12,25), dt.date(2026,1,1), dt.date(2026,4,6),
            dt.date(2026,5,1), dt.date(2026,5,8), dt.date(2026,5,14), dt.date(2026,5,25),
            dt.date(2026,7,14), dt.date(2026,8,15), dt.date(2026,11,1), dt.date(2026,11,11),
            dt.date(2026,12,25), dt.date(2027,1,1)}

def bday_fwd(d):
    while d.weekday() >= 5 or d in HOLIDAYS:
        d += dt.timedelta(days=1)
    return d
def bday_back(d):
    while d.weekday() >= 5 or d in HOLIDAYS:
        d -= dt.timedelta(days=1)
    return d
def last_bday(y, m):
    nxt = dt.date(y + (m == 12), m % 12 + 1, 1)
    return bday_back(nxt - dt.timedelta(days=1))
def months(a, b):
    y, m = a.year, a.month
    while (y, m) <= (b.year, b.month):
        yield y, m
        y, m = (y + (m == 12), m % 12 + 1)
def thursday_on_or_before(d):
    # weekly supplier payment run every Thursday
    d = d - dt.timedelta(days=(d.weekday() - 3) % 7)
    return bday_back(d)

# ------------------------------------------------------------------ customers
customers = pd.DataFrame([
 # id, name, segment, country, terms_days, method, share, avg_invoice_HT, late_mean, late_sd, credit_limit
 ("C01","Lumen Hospitality Group","Hospitality (projects)","FR",60,"Bank transfer",0.00,0,14,6,300000),
 ("C02","Bâtir Bureaux SAS","Office fit-out contractor","FR",60,"Bank transfer",0.17,18000,22,10,120000),
 ("C03","Espace & Co Aménagement","Office fit-out contractor","FR",60,"Bank transfer",0.11,12000,12,8,80000),
 ("C04","Nova Interiors SARL","Office fit-out contractor","FR",45,"Bank transfer",0.08,9000,20,10,50000),
 ("C05","Officea Distribution","Furniture distributor","FR",45,"LCR (bill of exchange)",0.12,7000,0,0.6,90000),
 ("C06","Mobilier Pro Rhône","Furniture distributor","FR",45,"Bank transfer",0.09,5500,6,5,60000),
 ("C07","Studio Kaya Design","Office fit-out contractor","FR",30,"Bank transfer",0.04,3500,2,3,20000),
 ("C08","Université Lyon-Est","Public sector","FR",30,"Bank transfer (Chorus Pro)",0.05,14000,8,8,80000),
 ("C09","CHU Vallée du Rhône","Public sector (hospital)","FR",50,"Bank transfer (Chorus Pro)",0.04,11000,21,12,60000),
 ("C10","Azur Collection Hôtels","Hospitality (projects)","FR",45,"Bank transfer",0.00,0,8,5,250000),
 ("C11","Hub & Desk Coworking SAS","Direct corporate client","FR",30,"Bank transfer",0.04,6000,9,7,30000),
 ("C12","Arkade Interiors SRL","Export distributor","BE",60,"SEPA transfer",0.06,10000,5,4,60000),
], columns=["customer_id","customer_name","segment","country","payment_terms_days","payment_method",
            "revenue_share","avg_invoice_ex_vat","late_mean","late_sd","credit_limit_eur"])

SEAS = {1:.85,2:.90,3:1.0,4:1.0,5:1.0,6:1.05,7:.90,8:.55,9:1.15,10:1.20,11:1.20,12:1.0}
BASE_MONTHLY = 590000  # non-project revenue ex VAT, 2025 level

def nova_late(issue):  # Nova Interiors is slowing down during 2026 (anomaly to detect)
    t = (issue - dt.date(2025,4,1)).days / 365
    return 18 + max(0, t - 0.75) * 30   # ~18 days late until early 2026, ~40 by Oct 2026

inv_rows = []
seq = {"n": 0}
def add_invoice(cid, issue, amount_ht, label, late_override=None, project=None):
    c = customers.set_index("customer_id").loc[cid]
    vat = 0 if c.country != "FR" else round(amount_ht * VAT, 2)
    due = issue + dt.timedelta(days=int(c.payment_terms_days))
    if late_override is not None:
        late = late_override
    elif cid == "C05":
        late = 0
    else:
        mu = nova_late(issue) if cid == "C04" else c.late_mean
        late = rng.normal(mu, c.late_sd)
        if rng.random() < 0.06:          # occasional long delay (admin issue, missing PO number...)
            late += rng.integers(15, 40)
        late = max(late, -7)
    pay = bday_fwd(due + dt.timedelta(days=int(round(late))))
    seq["n"] += 1
    inv_rows.append(dict(invoice_id=f"INV-{issue.year}-{seq['n']:04d}", customer_id=cid, issue_date=issue,
                         due_date=due, amount_ex_vat=round(amount_ht,2), vat=vat,
                         amount_incl_vat=round(amount_ht+vat,2), description=label, project=project,
                         true_pay_date=pay))

# regular (non-project) invoicing, Apr 2025 -> Mar 2027
for y, m in months(START, SIM_END):
    growth = 1.0 if y == 2025 else 1.08
    if y == 2024: growth = 0.96
    if y == 2027: growth = 1.12
    month_total = BASE_MONTHLY * SEAS[m] * growth
    for _, c in customers[customers.revenue_share > 0].iterrows():
        target = month_total * c.revenue_share
        n = max(1, int(round(target / c.avg_invoice_ex_vat * rng.uniform(0.8, 1.2))))
        w = rng.lognormal(0, 0.45, n); w = w / w.sum()
        last = (dt.date(y + (m == 12), m % 12 + 1, 1) - dt.timedelta(days=1)).day
        for k in range(n):
            if m == 12 and y == 2026:   # workshop closed 24 Dec - 3 Jan
                day = int(rng.integers(1, 23))
            elif m == 8:
                day = int(rng.integers(1, 15)) if rng.random() < .8 else int(rng.integers(24, last+1))
            else:
                day = int(rng.integers(1, last + 1))
            issue = bday_back(dt.date(y, m, day)) if dt.date(y, m, day) >= dt.date(y, m, 1) else dt.date(y, m, 1)
            if issue.month != m: issue = bday_fwd(dt.date(y, m, day))
            amt = round(target * w[k] / 10) * 10
            add_invoice(c.customer_id, issue, amt, "Furniture order")

# hospitality projects (milestone billing 30% order / 40% delivery / 30% installation)
# 2025: Lumen - Hôtel Lumen Chambéry refurbishment, EUR 390k
add_invoice("C01", dt.date(2025,5,12), 117000, "Hôtel Lumen Chambéry – 30% order deposit", project="LUM-CHB")
add_invoice("C01", dt.date(2025,9,8), 156000, "Hôtel Lumen Chambéry – 40% on delivery", project="LUM-CHB")
add_invoice("C01", dt.date(2025,10,20), 117000, "Hôtel Lumen Chambéry – 30% on installation", project="LUM-CHB")
# 2026: Lumen - Hôtel Lumen Annecy, EUR 480k
add_invoice("C01", dt.date(2026,6,3), 144000, "Hôtel Lumen Annecy – 30% order deposit", project="LUM-ANN")
add_invoice("C01", dt.date(2026,9,14), 192000, "Hôtel Lumen Annecy – 40% on delivery", project="LUM-ANN", late_override=13)
add_invoice("C01", dt.date(2026,10,30), 144000, "Hôtel Lumen Annecy – 30% on installation", project="LUM-ANN", late_override=14)
# 2025: Azur - small pilot project, then the big 2026-27 project (order signed 2 Nov 2026)
add_invoice("C10", dt.date(2025,11,17), 36000, "Azur Nice Port – 30% order deposit", project="AZR-NIC")
add_invoice("C10", dt.date(2026,1,26), 48000, "Azur Nice Port – 40% on delivery", project="AZR-NIC")
add_invoice("C10", dt.date(2026,3,9), 36000, "Azur Nice Port – 30% on installation", project="AZR-NIC")
add_invoice("C10", dt.date(2026,11,2), 108000, "Azur Cannes & Antibes – 30% order deposit", project="AZR-CAN", late_override=6)
add_invoice("C10", dt.date(2027,2,1), 144000, "Azur Cannes & Antibes – 40% on delivery", project="AZR-CAN")

inv = pd.DataFrame(inv_rows).sort_values("issue_date").reset_index(drop=True)
inv["invoice_id"] = [f"INV-{d.year}-{i+1:04d}" for i, d in enumerate(inv.issue_date)]

# a couple of disputed invoices at cutoff (realistic noise)
# ------------------------------------------------------------------ suppliers & bills
suppliers = pd.DataFrame([
 ("S01","Holzwerk Bayern GmbH","Timber & plywood","DE",30,"Monthly","Reverse-charge (intra-EU, no VAT paid)",0.22),
 ("S02","Aciers Rhône-Métal","Steel frames & legs","FR",45,"Twice monthly","20% VAT",0.11),
 ("S03","Feutre & Textiles Lyonnais","Acoustic felt & fabrics","FR",60,"Monthly","20% VAT",0.055),
 ("S04","Quincaillerie Pro Est","Hardware & fittings","FR",30,"Weekly","20% VAT",0.034),
 ("S05","Vernis Écolo SAS","Water-based finishes","FR",30,"Monthly","20% VAT",0.02),
 ("S06","TransAlp Logistique","Delivery & installation transport","FR",30,"Monthly","20% VAT",0.03),
], columns=["supplier_id","supplier_name","category","country","payment_terms_days","billing_frequency","vat_treatment","pct_of_revenue"])

rev_by_month = inv.assign(ym=inv.issue_date.map(lambda d: (d.year, d.month))).groupby("ym").amount_ex_vat.sum()
bill_rows = []
def add_bill(sid, issue, amt_ht, desc, paid_early=False, true_pay=None):
    s = suppliers.set_index("supplier_id").loc[sid]
    vat = 0 if s.country != "FR" else round(amt_ht * VAT, 2)
    due = issue + dt.timedelta(days=int(s.payment_terms_days))
    pay = true_pay or thursday_on_or_before(due)
    bill_rows.append(dict(supplier_id=sid, bill_date=issue, due_date=due, amount_ex_vat=round(amt_ht,2),
                          vat=vat, amount_incl_vat=round(amt_ht+vat,2), description=desc, pay_date_sim=pay))

for y, m in months(START, dt.date(2027,1,31)):
    nxt = (y + (m == 12), m % 12 + 1)
    # materials bought ~1 month ahead of invoicing (projects' materials handled separately below)
    reg = inv[(inv.project.isna())]
    rv_next = reg[reg.issue_date.map(lambda d: (d.year, d.month)) == nxt].amount_ex_vat.sum()
    rv_this = reg[reg.issue_date.map(lambda d: (d.year, d.month)) == (y, m)].amount_ex_vat.sum()
    noise = lambda: rng.uniform(0.9, 1.1)
    add_bill("S01", bday_fwd(dt.date(y,m,5)), rv_next*0.22*noise(), f"Timber & plywood – {dt.date(y,m,1):%b %Y} delivery")
    add_bill("S02", bday_fwd(dt.date(y,m,3)), rv_next*0.055*noise(), "Steel frames – 1st half")
    add_bill("S02", bday_fwd(dt.date(y,m,17)), rv_next*0.055*noise(), "Steel frames – 2nd half")
    add_bill("S03", bday_fwd(dt.date(y,m,10)), rv_next*0.055*noise(), "Acoustic felt & fabrics")
    for wk in (2, 9, 16, 23):
        add_bill("S04", bday_fwd(dt.date(y,m,wk)), rv_next*0.0085*noise(), "Hardware & fittings")
    add_bill("S05", bday_fwd(dt.date(y,m,12)), rv_next*0.02*noise(), "Finishes")
    add_bill("S06", last_bday(y, m), rv_this*0.03*noise(), f"Transport {dt.date(y,m,1):%b %Y}")

# project materials
add_bill("S01", dt.date(2025,6,16), 58000, "Oak & plywood – Hôtel Lumen Chambéry")
add_bill("S02", dt.date(2025,7,1), 24000, "Steel – Hôtel Lumen Chambéry")
add_bill("S01", dt.date(2026,6,22), 71000, "Oak & plywood – Hôtel Lumen Annecy")
add_bill("S02", dt.date(2026,7,6), 29000, "Steel – Hôtel Lumen Annecy")
add_bill("S03", dt.date(2026,7,6), 14000, "Felt – Hôtel Lumen Annecy")
add_bill("S01", dt.date(2025,12,8), 22000, "Oak – Azur Nice Port")
# THE BIG ONE: winter pre-order + materials for the Azur Cannes & Antibes project (PO placed 28 Sep 2026)
BIG_BILL = dict(supplier_id="S01", bill_date=dt.date(2026,10,21), amount_ex_vat=155000.0,
                description="Winter timber pre-order (3% volume discount) + Azur Cannes & Antibes materials")
add_bill("S01", BIG_BILL["bill_date"], BIG_BILL["amount_ex_vat"], BIG_BILL["description"])
add_bill("S02", dt.date(2026,11,9), 32000, "Steel – Azur Cannes & Antibes")

bills = pd.DataFrame(bill_rows).sort_values("bill_date").reset_index(drop=True)
bills["bill_id"] = [f"BILL-{d.year}-{i+1:04d}" for i, d in enumerate(bills.bill_date)]
bills = bills[bills.amount_ex_vat > 0]

# ------------------------------------------------------------------ other cash flows (payroll, tax, rent...)
other = []  # (date, category, counterparty, amount (negative = out), ex_vat_for_input_vat)
def out(d, cat, cp, amt, ht=None, note=""):
    other.append(dict(date=d, category=cat, counterparty=cp, amount=-round(amt,2), input_vat=round((amt-ht),2) if ht else 0.0, note=note))

for y, m in months(START, dt.date(2027,1,31)):
    yr26 = (y, m) >= (2026, 1)
    net_pay = 106000 if yr26 else 101500
    charges = 86000 if yr26 else 82000
    out(last_bday(y, m), "Payroll (net salaries)", "Employees (38 FTE)", net_pay)
    if m == 12:
        out(last_bday(y, m), "Payroll (year-end bonus)", "Employees (38 FTE)", 72000 if y == 2026 else 68000, note="13th-month bonus, paid with December salaries")
    pm = (y - (m == 1), 12 if m == 1 else m - 1)
    prev_charges = (86000 if pm >= (2026,1) else 82000) + (52000 if pm[1] == 12 else 0)
    out(bday_fwd(dt.date(y, m, 15)), "Social charges & income tax withheld", "URSSAF / AGIRC-ARRCO / DGFiP (PAS)", prev_charges, note="for previous month's payroll")
    out(bday_fwd(dt.date(y, m, 5)), "Bank loan repayment", "Banque Régionale (CNC machine loan)", 8900, note="principal + interest")
    out(bday_fwd(dt.date(y, m, 10)), "Equipment lease", "LeasEquip France", 3200*1.2, 3200)
    out(bday_fwd(dt.date(y, m, 8)), "Energy", "Electricity & gas supplier (direct debit)", (11800 if m in (11,12,1,2) else 8200)*1.2, (11800 if m in (11,12,1,2) else 8200))
    out(bday_fwd(dt.date(y, m, 3)), "IT, telecom & software", "Various (direct debits)", 2400*1.2, 2400)
    if m in (1, 4, 7, 10):
        out(bday_fwd(dt.date(y, m, 1)), "Rent (quarterly in advance)", "SCI Les Ateliers du Rhône (landlord)", 19500*1.2, 19500)
        out(bday_fwd(dt.date(y, m, 1)), "Insurance (quarterly)", "Insurer", 7800)
    if m in (3, 6, 9, 12):
        out(bday_fwd(dt.date(y, m, 15)), "Corporate income tax instalment", "DGFiP", 21000 if y == 2025 else 24500, note="acompte d'IS")
    if (y, m) in ((2025, 12), (2026, 12)):
        out(dt.date(y, 12, 15), "Business property tax (CFE)", "DGFiP", 6200)
    if (y, m) == (2025, 5): out(dt.date(2025,5,15), "Corporate income tax balance", "DGFiP", 9800)
    if (y, m) == (2026, 5): out(dt.date(2026,5,15), "Corporate income tax balance", "DGFiP", 12400)
    if (y, m) == (2025, 6): out(dt.date(2025,6,27), "Dividend", "Shareholders", 90000)
    if (y, m) == (2026, 6): out(dt.date(2026,6,26), "Dividend", "Shareholders", 120000)
    if (y, m) == (2026, 3): out(dt.date(2026,3,20), "Capex", "Sanding line upgrade", 28800, 24000)

# weekly card / petty spending (fuel, travel, consumables)
d = START
while d <= dt.date(2027,1,31):
    if d.weekday() == 2:
        amt = rng.uniform(3800, 5600) if not (d.month == 8 or (d.month == 12 and d.day > 23)) else rng.uniform(1200, 2200)
        out(d, "Card & sundry spending", "Fuel, travel, consumables", amt, amt/1.2)
    d += dt.timedelta(days=1)

# planned capex: edge-banding machine, 30% deposit at order (discretionary - can be deferred)
out(dt.date(2026,11,25), "Capex (planned, discretionary)", "Biesse-type edge-bander supplier", 96000*0.3*1.2, 96000*0.3, note="30% deposit on EUR 96k machine; delivery Feb 2027")

oth = pd.DataFrame(other)

# ------------------------------------------------------------------ VAT (monthly CA3, paid ~24th of following month)
inv["ym"] = inv.issue_date.map(lambda d: (d.year, d.month))
bills["ym"] = bills.bill_date.map(lambda d: (d.year, d.month))
oth["ym"] = oth.date.map(lambda d: (d.year, d.month))
vat_rows = []
for y, m in months(START, dt.date(2026,12,31)):
    out_vat = inv[inv.ym == (y, m)].vat.sum()
    in_vat = bills[bills.ym == (y, m)].vat.sum() + oth[oth.ym == (y, m)].input_vat.sum()
    net = round(out_vat - in_vat, 2)
    ny, nm = (y + (m == 12), m % 12 + 1)
    pay = bday_fwd(dt.date(ny, nm, 24))
    vat_rows.append(dict(period=f"{y}-{m:02d}", output_vat=round(out_vat,2), deductible_vat=round(in_vat,2), net_vat_payable=net, payment_date=pay))
vat = pd.DataFrame(vat_rows)

# ------------------------------------------------------------------ build full cash ledger & anchor balance
ledger = []
for r in inv.itertuples():
    ledger.append((r.true_pay_date, "Customer receipt", r.customer_id, r.amount_incl_vat, r.invoice_id))
for r in bills.itertuples():
    ledger.append((r.pay_date_sim, "Supplier payment", r.supplier_id, -r.amount_incl_vat, r.bill_id))
for r in oth.itertuples():
    ledger.append((r.date, r.category, r.counterparty, r.amount, ""))
for r in vat.itertuples():
    if r.net_vat_payable > 0:
        ledger.append((r.payment_date, "VAT payment", "DGFiP", -r.net_vat_payable, r.period))
led = pd.DataFrame(ledger, columns=["date","category","counterparty","amount","ref"]).sort_values("date").reset_index(drop=True)

TARGET_CUTOFF_BAL = float(os.environ.get('TARGET', '381240.65'))
hist_flows = led[led.date <= CUTOFF].amount.sum()
OPENING_APR25 = round(TARGET_CUTOFF_BAL - hist_flows, 2)
led["balance"] = OPENING_APR25 + led.amount.cumsum()

json.dump(dict(opening_apr25=OPENING_APR25), open(f"{OUT}/_meta.json", "w"))
for name, df in dict(customers=customers, invoices=inv, suppliers=suppliers, bills=bills, other=oth, vat=vat, ledger=led).items():
    df.to_pickle(f"{OUT}/{name}.pkl")
print("opening balance 1 Apr 2025:", OPENING_APR25)
print("invoices", len(inv), "bills", len(bills), "other", len(oth))
