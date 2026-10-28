"""Generates notebooks/cash_radar_analysis.ipynb (run once; the notebook is then executed with nbconvert)."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: C.append(nbf.v4.new_code_cell(s.strip()))

md("""
# Cash Radar — spotting a cash shortage before it happens

**Question:** a fictional French furniture maker, *Atelier Verde SAS*, is profitable on paper.
Will it have enough cash for the next 13 weeks — and what happens if its biggest customer pays late?

This notebook walks through the full analysis behind the Cash Radar feature:

1. The data and the profit-vs-cash problem
2. Customer payment habits
3. The 13-week forecast
4. Uncertainty (a likely range, not one number)
5. A delayed-payment scenario and the warning it produces
6. Actions the CFO could take
7. How accurate the method is (backtest)

> All company data is **fictional** and simulated (see `scripts/`). French payment rules and Banque de France
> statistics were used to make it realistic.
""")
code("""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path.cwd().parent / "src"))
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import cash_radar as cr

FIG = pathlib.Path.cwd().parent / "reports" / "figures"; FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.alpha": .25, "font.size": 10})
GREEN, RED, ORANGE, GREY, BLUE = "#1f6e54", "#c0392b", "#e08a1e", "#8b918d", "#2f6db5"
eur = lambda v: f"{'−' if v < 0 else ''}€{abs(v):,.0f}"
kfmt = plt.FuncFormatter(lambda v, _: f"{'−' if v < 0 else ''}€{abs(v)/1000:,.0f}k")

ds = cr.load()
s = ds.settings
print(s)
""")
md("## 1. The data")
code("""
pd.DataFrame({name: [len(getattr(ds, name))] for name in
              ["customers", "invoices", "orders", "planned_payments", "bank_transactions"]}, index=["rows"]).T
""")
code("""
ds.customers
""")
md("""
### Profit is not cash

Over a year the company is profitable (about €190k net profit expected in 2026), even if quiet months like August
lose money. But the bank balance swings by hundreds of thousands of euros, because customers pay 45–60 days (or more)
after the invoice, while salaries, VAT and timber are paid on fixed dates. In summer 2025 the company went into its
overdraft without seeing it coming, and in November 2025 cash fell to almost zero.
""")
code("""
pnl = pd.read_csv(cr.data.SAMPLE_DIR / "profit_and_loss_monthly.csv")
pnl = pnl[pnl.basis == "Actual"]
bal = cr.engine.balance_history(ds.bank_transactions, s)
weekly_bal = bal.resample("W-SUN").last()

fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=False)
ax[0].bar(pnl.month, pnl.net_profit, color=[GREEN if v >= 0 else RED for v in pnl.net_profit])
ax[0].set_title(f"Monthly net profit, actual (accrual basis) — total {eur(pnl.net_profit.sum())} over {len(pnl)} months"); ax[0].yaxis.set_major_formatter(kfmt)
ax[0].tick_params(axis="x", rotation=90, labelsize=8)
ax[1].plot(weekly_bal.index, weekly_bal.values, color=BLUE)
ax[1].axhline(s.buffer, color=GREY, ls="--", lw=1); ax[1].axhline(0, color="black", lw=.8)
ax[1].set_title("Cash in the bank, end of each week"); ax[1].yaxis.set_major_formatter(kfmt)
plt.tight_layout(); plt.savefig(FIG / "profit_vs_cash.png", bbox_inches="tight"); plt.show()
print("Lowest weekly balance in history:", eur(weekly_bal.min()), "on", weekly_bal.idxmin().date())
""")
md("""
## 2. Customer payment habits

Customers rarely pay on the due date. For each customer we take the **median days late** over the last 12 months.
A customer whose last 90 days are 7+ days slower than usual is flagged.
""")
code("""
habits = cr.habit_table(ds.invoices, ds.customers, s.as_of)
habits.sort_values("usual_days_late", ascending=False)
""")
code("""
h = habits.sort_values("usual_days_late")
fig, ax = plt.subplots(figsize=(9, 4.2))
ax.barh(h.customer_name, h.usual_days_late, color=[ORANGE if x else GREY for x in h.slowing])
for y, (v, r) in enumerate(zip(h.usual_days_late, h.recent_days_late_90d)):
    ax.text(v + .5, y, f"{v:.0f} d" + (f"  (recently {r:.0f} d)" if not np.isnan(r) and r > v + 7 else ""), va="center", fontsize=8)
ax.set_xlabel("median days late after the due date (last 12 months)")
ax.set_title("Payment habits by customer (orange = paying slower recently)")
plt.tight_layout(); plt.savefig(FIG / "customer_habits.png", bbox_inches="tight"); plt.show()
""")
md("""
## 3. The 13-week forecast

Every open invoice and confirmed order becomes an expected receipt on *due date + usual delay*; every bill, salary,
tax and loan payment is placed on its own date. Disputed invoices are left out until they are settled.
""")
code("""
lines = cr.build_lines(ds.invoices, ds.orders, ds.planned_payments, ds.customers, s)
print(len(lines), "forecast lines;", (~lines.included).sum(), "excluded (disputed)")
lines.drop(columns="samples").head(8)
""")
code("""
base = cr.weekly(lines, s)
base.style.format({c: "€{:,.0f}" for c in ["opening", "receipts", "payments", "net", "closing"]})
""")
md("""
## 4. Uncertainty

The forecast is repeated 4,000 times, each time drawing every invoice's delay from that customer's past delays.
This gives a likely range (P10–P90: 8 futures out of 10) and the chance of falling below the €100k minimum.
""")
code("""
sims = cr.simulate(lines, s, n=4000)
b0 = cr.bands(sims, s)
print(f"Chance of at least one week below the minimum: {(sims.min(axis=1) < s.buffer).mean():.0%}")
b0.style.format({"p10": "€{:,.0f}", "p50": "€{:,.0f}", "p90": "€{:,.0f}", "p_below_buffer": "{:.0%}", "p_overdraft": "{:.0%}"})
""")
md("""
## 5. What if the biggest customer pays late?

Scenario: **Lumen Hospitality pays its €230k invoice 45 days later than usual.**
""")
code("""
sc = cr.Scenario("Lumen pays 45 days late", customer_delay={"C01": 45})
lines_sc = cr.apply_scenario(lines, s, sc)
scen = cr.weekly(lines_sc, s)
sims_sc = cr.simulate(lines, s, sc, n=4000)
b1 = cr.bands(sims_sc, s)
wn = cr.warning(base, scen, lines, lines_sc, sims_sc, s, sc.name)
print(cr.warning_text(wn, ds.profit_for_period))
""")
code("""
x = np.arange(1, s.weeks + 1)
fig, ax = plt.subplots(figsize=(10, 4.6))
ax.fill_between(x, b1.p10, b1.p90, color=RED, alpha=.09, lw=0, label="likely range if Lumen is late (8 in 10)")
ax.fill_between(x, scen.closing, base.closing, color=ORANGE, alpha=.28, lw=0, label="cash missing because of the delay")
ax.plot(x, base.closing, "-o", color=GREEN, ms=4, label="normal case")
ax.plot(x, scen.closing, "-o", color=RED, ms=4, label="Lumen pays 45 days late")
ax.axhline(s.buffer, color=GREY, ls="--", lw=1); ax.text(0.8, s.buffer + 12000, "€100k minimum", fontsize=8, color=GREY)
ax.axhline(0, color="black", lw=.8)
ax.annotate(f"{eur(wn['lowest_balance'])} in week {wn['lowest_week']}", (wn["lowest_week"], wn["lowest_balance"]),
            xytext=(8, -14), textcoords="offset points", color=RED, fontweight="bold", fontsize=9)
ax.set_xticks(x, [f"W{w}\\n{d:%d %b}" for w, d in zip(x, s.week_starts)], fontsize=8)
ax.yaxis.set_major_formatter(kfmt); ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=4, frameon=False)
ax.set_title("One late payment turns a tight quarter into an overdraft")
plt.tight_layout(); plt.savefig(FIG / "forecast_scenario.png", bbox_inches="tight"); plt.show()
""")
md("""
**Profit vs cash in one line:** the quarter still shows a profit of about €182k, yet cash falls by more than €300k over
the 13 weeks in this scenario. Nothing is wrong with the business — the money is simply late.
""")
md("""
## 6. Actions the CFO could take

Cash Radar re-runs the forecast for each action. These are **suggestions only**: a person approves or rejects them.
""")
code("""
big_bill = lines_sc[lines_sc.flow == "payment"].sort_values("amount").source_id.iloc[0]
lumen_inv = lines_sc[(lines_sc.customer_id == "C01") & (lines_sc.flow == "receipt")].sort_values("amount").source_id.iloc[-1]
low0 = scen.closing.min()
actions = {
    "Chase overdue invoices (50% in week 2)": cr.act_collect_overdue(lines_sc, s),
    "Delay flexible payments (machine deposit)": cr.act_defer_flexible(lines_sc, s),
    f"Pay the largest bill in two parts ({big_bill})": cr.act_split_payment(lines_sc, s, big_bill),
    "Invoice financing: 80% of Lumen's invoice in week 4": cr.act_finance_invoice(lines_sc, s, lumen_inv),
}
rows = []
for name, l in actions.items():
    w = cr.weekly(l, s)
    rows.append({"action": name, "lowest cash": w.closing.min(), "improvement": w.closing.min() - low0,
                 "weeks below minimum": int(w.below_buffer.sum())})
combo = cr.act_finance_invoice(cr.act_split_payment(cr.act_defer_flexible(cr.act_collect_overdue(lines_sc, s), s), s, big_bill), s, lumen_inv)
wc = cr.weekly(combo, s)
rows.append({"action": "All four together", "lowest cash": wc.closing.min(), "improvement": wc.closing.min() - low0,
             "weeks below minimum": int(wc.below_buffer.sum())})
pd.DataFrame(rows).style.format({"lowest cash": "€{:,.0f}", "improvement": "+€{:,.0f}"})
""")
md("""
## 7. How accurate is the method?

**Backtest:** every Friday from April 2025, re-run the forecast using only what was known that day and compare it
with the real bank balance over the next 13 weeks. Compare with a naive forecast where everyone pays on the due date.
Payments are taken as known so the test isolates the hard part: *when customers pay*.
""")
code("""
bt = cr.backtest(ds.invoices, ds.bank_transactions, s)
sm = cr.backtest_summary(bt, s.buffer)
print(f"{sm['forecasts']} past forecasts")
print(f"Average error — Cash Radar: {eur(sm['mae'])}   naive: {eur(sm['mae_naive'])}   reduction: {sm['error_reduction']:.0%}")
print(f"Bias: on average {eur(sm['bias'])} too optimistic.  False alarms: {sm['false_alarm_share']:.0%}")
cr.lead_times(bt, s.buffer)
""")
code("""
byh = bt.groupby("weeks_ahead")[["error", "error_naive"]].agg(lambda e: e.abs().mean())
fig, ax = plt.subplots(figsize=(8, 3.6))
ax.plot(byh.index, byh.error_naive, "-o", color=GREY, ms=3, label="everyone pays on the due date")
ax.plot(byh.index, byh.error, "-o", color=GREEN, ms=3, label="Cash Radar (customer habits)")
ax.set_xlabel("weeks ahead"); ax.yaxis.set_major_formatter(kfmt); ax.set_ylim(0)
ax.set_title("Average forecast error by horizon (66 past forecasts)"); ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig(FIG / "backtest_error.png", bbox_inches="tight"); plt.show()
""")
md("""
## 8. Conclusions and limitations

- **Result:** using each customer's own payment habit cuts the average weekly forecast error by about 75% compared with
  assuming on-time payment, and it would have flagged the October 2025 dip 12 weeks ahead (the naive forecast never did).
- **Warning:** if Lumen pays 45 days late, cash breaks the €100k minimum in week 8 with **7 weeks of warning**,
  while the quarter is still profitable.
- **Limitations:** the method is still about €54k too optimistic on average, so the pessimistic end of the range should
  always be shown; it missed the summer 2025 overdraft because only four months of history existed; payments are taken
  as known in the backtest; all data is simulated.
- **Next steps:** spread each invoice over several possible weeks (probability-weighted timing, ~14% more accurate in
  tests), learn seasonality, and test on real client data in a pilot.
""")
nb["cells"] = C
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, "cash_radar_analysis.ipynb")
print("notebook written")
