"""Cash Radar — interactive 13-week cash forecast.

Run locally:  streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import io
import sys
import zipfile
from datetime import timedelta
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import cash_radar as cr  # noqa: E402
from cash_radar.data import SAMPLE_DIR, Dataset, clean  # noqa: E402

st.set_page_config(page_title="Cash Radar", page_icon="📡", layout="wide")

GREEN, RED, ORANGE, GREY, BLUE = "#1f6e54", "#c0392b", "#e08a1e", "#8b918d", "#2f6db5"


def eur(v: float) -> str:
    return f"{'−' if v < 0 else ''}€{abs(v):,.0f}"


# ------------------------------------------------------------------ data
@st.cache_data(show_spinner=False)
def load_sample() -> Dataset:
    return cr.load(SAMPLE_DIR)


@st.cache_data(show_spinner=False)
def sample_zip() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for f in sorted(SAMPLE_DIR.glob("*")):
            z.write(f, f.name)
    return buf.getvalue()


@st.cache_data(show_spinner="Testing the method on past weeks…")
def run_backtest(_ds: Dataset, key: str):
    bt = cr.backtest(_ds.invoices, _ds.bank_transactions, _ds.settings)
    return bt, cr.backtest_summary(bt, _ds.settings.buffer), cr.lead_times(bt, _ds.settings.buffer)


st.sidebar.title("📡 Cash Radar")
source = st.sidebar.radio("Data", ["Sample company (fictional)", "Upload my own CSV files"])

ds: Dataset | None = None
if source.startswith("Sample"):
    ds = load_sample()
else:
    st.sidebar.download_button("Download sample files as templates", sample_zip(), "cash_radar_sample_data.zip")
    up = {n: st.sidebar.file_uploader(f"{n}.csv", type="csv", key=n)
          for n in ["customers", "invoices", "orders", "planned_payments", "bank_transactions"]}
    with st.sidebar.expander("Company settings", expanded=True):
        as_of = st.date_input("Data date (last bank balance)", value=pd.Timestamp.today().date())
        opening = st.number_input("Cash in the bank on that date (€)", value=100_000.0, step=1_000.0)
        profit = st.number_input("Profit expected for the period (€, optional)", value=0.0, step=1_000.0)
    needed = ["customers", "invoices", "planned_payments"]
    if all(up[n] for n in needed):
        try:
            tables = {n: clean(n, pd.read_csv(up[n])) for n in needed}
            tables["orders"] = clean("orders", pd.read_csv(up["orders"])) if up["orders"] else \
                pd.DataFrame(columns=["order_id", "customer_id", "due_date", "amount", "probability", "description"])
            tables["bank_transactions"] = clean("bank_transactions", pd.read_csv(up["bank_transactions"])) if up["bank_transactions"] else \
                pd.DataFrame(columns=["date", "amount", "category", "counterparty"])
            start = as_of + timedelta(days=(7 - as_of.weekday()) % 7 or 7)
            ds = Dataset(settings=cr.Settings(as_of=as_of, forecast_start=start, opening_balance=opening, company="Your company"),
                         profit_for_period=profit or None, **tables)
        except Exception as e:  # show the user what is wrong with their file
            st.sidebar.error(f"Could not read the files: {e}")
    else:
        st.info("Upload at least **customers.csv**, **invoices.csv** and **planned_payments.csv** "
                "(same columns as the sample files), or switch back to the sample company.")
        st.stop()
if ds is None:
    st.stop()

s0 = ds.settings
st.sidebar.header("Your limits")
buffer = st.sidebar.number_input("Minimum cash to keep (€)", value=float(s0.buffer), step=10_000.0, format="%.0f",
                                 help="The 'buffer'. A warning is raised for any week below it.")
overdraft = st.sidebar.number_input("Bank overdraft limit (€)", value=float(s0.overdraft_limit), step=10_000.0, format="%.0f")
s = cr.Settings(**{**s0.__dict__, "buffer": buffer, "overdraft_limit": overdraft})

lines = cr.build_lines(ds.invoices, ds.orders, ds.planned_payments, ds.customers, s)
cust = ds.customers.set_index("customer_id").customer_name.to_dict()
receipts_by_cust = lines[lines.flow == "receipt"].groupby("customer_id").amount.sum().sort_values(ascending=False)

st.sidebar.header("What if a customer pays late?")
default_c = [c for c in ["C01"] if c in receipts_by_cust.index] if source.startswith("Sample") else []
late_customers = st.sidebar.multiselect("Customers paying late", list(receipts_by_cust.index), default=default_c,
                                        format_func=lambda c: f"{cust.get(c, c)} ({eur(receipts_by_cust[c])} expected)")
late_days = st.sidebar.slider("Extra delay for these customers (days)", 0, 90, 45 if default_c else 0, step=5)
segments = sorted(ds.customers.segment.unique())
seg = st.sidebar.selectbox("A whole customer group pays late", ["(none)"] + segments)
seg_days = st.sidebar.slider("Extra delay for that group (days)", 0, 60, 15, step=5) if seg != "(none)" else 0

scenario = cr.Scenario(name="Scenario", customer_delay={c: late_days for c in late_customers},
                       segment_delay={seg: seg_days} if seg != "(none)" else {})
has_scenario = bool(late_days and late_customers) or bool(seg_days)

st.sidebar.header("Test CFO actions")
st.sidebar.caption("What-if only. Nothing is sent to a bank, customer or supplier.")
a_collect = st.sidebar.checkbox("Chase overdue invoices")
collect_share = st.sidebar.slider("Share of overdue amount collected in week 2", 10, 100, 50, step=10) if a_collect else 50
a_defer = st.sidebar.checkbox("Delay flexible payments (e.g. equipment)", disabled=not lines.deferrable.any())
big_pay = lines[(lines.flow == "payment") & (lines.week != cr.AFTER)].sort_values("amount").head(8)
a_split = st.sidebar.checkbox("Pay a large bill in two parts")
split_id = st.sidebar.selectbox("Bill to split", big_pay.source_id,
                                format_func=lambda i: f"{big_pay.set_index('source_id').counterparty[i]} · {eur(-big_pay.set_index('source_id').amount[i])} · week {big_pay.set_index('source_id').week[i]}") if a_split else None
big_rec = lines[(lines.flow == "receipt") & lines.included].sort_values("amount", ascending=False).head(8)
a_fin = st.sidebar.checkbox("Get 80% of an invoice early (invoice financing)")
fin_id = st.sidebar.selectbox("Invoice to finance", big_rec.source_id,
                              format_func=lambda i: f"{big_rec.set_index('source_id').counterparty[i]} · {eur(big_rec.set_index('source_id').amount[i])}") if a_fin else None
n_sims = st.sidebar.select_slider("Simulations for the likely range", [500, 1000, 2000, 4000], value=2000)

# ------------------------------------------------------------------ compute
base = cr.weekly(lines, s)
lines_sc = cr.apply_scenario(lines, s, scenario)
scen = cr.weekly(lines_sc, s)
lines_act = lines_sc
if a_collect:
    lines_act = cr.act_collect_overdue(lines_act, s, share=collect_share / 100)
if a_defer:
    lines_act = cr.act_defer_flexible(lines_act, s)
if a_split and split_id:
    lines_act = cr.act_split_payment(lines_act, s, split_id)
if a_fin and fin_id:
    lines_act = cr.act_finance_invoice(lines_act, s, fin_id)
any_action = a_collect or a_defer or a_split or a_fin
acted = cr.weekly(lines_act, s) if any_action else None
sims = cr.simulate(lines, s, scenario, n=n_sims)
bd = cr.bands(sims, s)
wn = cr.warning(base, scen, lines, lines_sc, sims, s, "scenario")

# ------------------------------------------------------------------ header
st.title(f"Will {s.company.replace(' (fictional)', '') or 'the company'} have enough cash for the next {s.weeks} weeks?")
st.caption(f"Data as at {s.as_of:%d %b %Y} · forecast {s.forecast_start:%d %b %Y} – {s.week_starts[-1] + timedelta(days=6):%d %b %Y}"
           + (" · fictional company, simulated data" if source.startswith("Sample") else ""))

shown = scen
k = st.columns(5)
k[0].metric("Cash today", eur(s.opening_balance))
k[1].metric(f"Lowest cash, normal (wk {int(base.loc[base.closing.idxmin(), 'week'])})", eur(base.closing.min()))
diff = shown.closing.min() - base.closing.min()
k[2].metric(f"Lowest cash, scenario (wk {int(shown.loc[shown.closing.idxmin(), 'week'])})" if has_scenario else "Lowest cash, no scenario set",
            eur(shown.closing.min()), (f"-€{abs(diff):,.0f} vs normal" if diff < 0 else f"+€{diff:,.0f} vs normal") if has_scenario and diff else None)
k[3].metric(f"Warning time · breach wk {wn['first_week']}" if wn else "Warning time", f"{wn['lead_time_weeks']} weeks" if wn else "no warning")
if ds.profit_for_period:
    k[4].metric("Profit (≠ cash)", eur(ds.profit_for_period))
else:
    k[4].metric("Chance of a breach", f"{(sims.min(axis=1) < s.buffer).mean():.0%}")

text = cr.warning_text(wn, ds.profit_for_period)
if wn:
    st.error("**⚠ Cash shortfall warning.** " + text)
else:
    st.success("**✓ No shortfall expected.** " + text)
if acted is not None:
    low = acted.closing.min()
    msg = f"With the selected actions, the lowest cash becomes **{eur(low)}** (week {int(acted.loc[acted.closing.idxmin(), 'week'])})."
    (st.success if low >= s.buffer else st.warning)(msg + " These are suggestions: the CFO approves or rejects each one.")

# ------------------------------------------------------------------ chart
x = [f"W{w}<br>{d:%d %b}" for w, d in zip(base.week, base.week_start)]
fig = go.Figure()
col = RED if has_scenario else GREEN
fig.add_trace(go.Scatter(x=x + x[::-1], y=list(bd.p90) + list(bd.p10[::-1]), fill="toself", fillcolor="rgba(120,120,120,0.15)",
                         line=dict(width=0), hoverinfo="skip", name="Likely range (8 in 10)"))
if has_scenario:
    fig.add_trace(go.Scatter(x=x, y=base.closing, name="Normal case", line=dict(color=GREY, dash="dash", width=2)))
fig.add_trace(go.Scatter(x=x, y=scen.closing, name="Your scenario" if has_scenario else "Normal case",
                         line=dict(color=col, width=3), mode="lines+markers"))
if acted is not None:
    fig.add_trace(go.Scatter(x=x, y=acted.closing, name="With actions", line=dict(color=BLUE, width=2.5)))
fig.add_hline(y=s.buffer, line=dict(color=GREY, dash="dot"), annotation_text=f"minimum {eur(s.buffer)}", annotation_position="top left")
fig.add_hline(y=0, line=dict(color="#555", width=1))
if s.overdraft_limit and min(scen.closing.min(), bd.p10.min()) < -0.5 * s.overdraft_limit:
    fig.add_hline(y=-s.overdraft_limit, line=dict(color=RED, dash="dot"), annotation_text=f"overdraft limit {eur(-s.overdraft_limit)}",
                  annotation_position="bottom left")
fig.update_layout(height=460, margin=dict(l=10, r=10, t=30, b=10), hovermode="x unified", yaxis_tickprefix="€",
                  yaxis_tickformat=",.0f", legend=dict(orientation="h", yanchor="top", y=-0.18, x=0), title="Cash in the bank at the end of each week")
st.plotly_chart(fig, width="stretch")
st.caption("The line is the most likely balance. The grey band shows the range in 8 out of 10 simulations of when customers pay.")

# ------------------------------------------------------------------ tabs
t1, t2, t3, t4, t5 = st.tabs(["Week by week", "Customer habits", "Every cash line", "How accurate is it?", "How it works"])

with t1:
    tab = scen[["week", "week_start", "receipts", "payments", "closing", "status"]].copy()
    tab.insert(5, "likely low (P10)", bd.p10.values)
    tab.insert(6, "chance below minimum", bd.p_below_buffer.values)
    if has_scenario:
        tab.insert(5, "normal case", base.closing.values)
    if acted is not None:
        tab["with actions"] = acted.closing.values
    money = [c for c in tab.columns if c not in ("week", "week_start", "status", "chance below minimum")]
    st.dataframe(tab.style.format({c: eur for c in money} | {"chance below minimum": "{:.0%}", "week_start": "{:%d %b %Y}"}),
                 hide_index=True, width="stretch")
    st.download_button("Download this table (CSV)", tab.to_csv(index=False), "cash_forecast_13_weeks.csv")

with t2:
    h = cr.habit_table(ds.invoices, ds.customers, s.as_of, s.lookback_days)
    openamt = lines[lines.flow == "receipt"].groupby("customer_id").amount.sum()
    h["expected_in_forecast"] = h.customer_id.map(openamt).fillna(0)
    h["flag"] = h.slowing.map({True: "Paying slower recently", False: ""})
    st.write("Each receipt is expected on its due date **plus the customer's usual delay** (median over 12 months). "
             "A customer paying 7+ days slower over the last 90 days is flagged — a reason to talk, never a penalty.")
    st.dataframe(h.drop(columns=["slowing"]).style.format({"usual_days_late": "{:.0f}", "recent_days_late_90d": "{:.1f}",
                                                           "expected_in_forecast": "€{:,.0f}"}, na_rep="—"),
                 hide_index=True, width="stretch")

with t3:
    show = lines_sc.drop(columns=["samples"]).copy()
    show["week"] = show.week.astype(str)
    st.write(f"{len(show)} lines. Every weekly total can be traced to these invoices, orders and payments. "
             "Disputed invoices are left out until they are settled.")
    st.dataframe(show[["source", "source_id", "flow", "category", "counterparty", "amount", "probability", "due_date",
                       "assumed_days_late", "expected_date", "week", "status", "included"]],
                 hide_index=True, width="stretch", height=420)

with t4:
    if ds.bank_transactions.empty:
        st.info("Upload bank_transactions.csv to test the method on your own past weeks.")
    else:
        bt, sm, lt = run_backtest(ds, f"{len(ds.invoices)}-{len(ds.bank_transactions)}-{s.buffer}")
        c = st.columns(4)
        c[0].metric("Past forecasts tested", sm["forecasts"])
        c[1].metric("Average error, Cash Radar", eur(sm["mae"]))
        c[2].metric("Average error, 'everyone pays on time'", eur(sm["mae_naive"]))
        c[3].metric("Error reduction", f"{sm['error_reduction']:.0%}")
        byh = bt.groupby("weeks_ahead")[["error", "error_naive"]].agg(lambda e: e.abs().mean()).reset_index()
        f2 = go.Figure([go.Scatter(x=byh.weeks_ahead, y=byh.error_naive, name="Everyone pays on time", line=dict(color=GREY, width=2)),
                        go.Scatter(x=byh.weeks_ahead, y=byh.error, name="Cash Radar", line=dict(color=GREEN, width=3))])
        f2.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10), title="Average error by weeks ahead",
                         xaxis_title="weeks ahead", yaxis_tickprefix="€", yaxis_tickformat=",.0f", legend=dict(orientation="h"))
        st.plotly_chart(f2, width="stretch")
        st.write(f"On average the forecast is still **{eur(sm['bias'])} too optimistic**, so keep an eye on the likely low (P10). "
                 f"{sm['false_alarm_share']:.0%} of 'below minimum' predictions did not happen (false alarms).")
        if not lt.empty:
            st.write("**Real dips below the minimum, and how early they were flagged:**")
            st.dataframe(lt.style.format({"lowest": "€{:,.0f}"}), hide_index=True)

with t5:
    st.markdown("""
**Cash Radar combines three things, every week:** money in the bank, money customers are expected to pay,
and bills the company must pay.

1. **Learn habits.** For each customer, how many days late they usually pay (median of the last 12 months).
2. **Predict receipts.** Each open invoice or confirmed order is expected on its due date plus that delay.
   An invoice already overdue only uses past delays longer than the current one.
3. **Add payments** on their due dates (bills, salaries, taxes, rent, loans).
4. **Show uncertainty.** The forecast is repeated thousands of times with different past delays → a likely range
   and the chance of falling below the minimum.
5. **Warn and explain** the first week below the minimum: when, how much, why, and how much time is left.
6. **Test actions** (chase invoices, delay flexible spending, split a bill, invoice financing). The tool only
   suggests; **a person always decides**.

**Why this method?** It is simple to explain and check. On the sample company it cut the average weekly error by
75% compared with a spreadsheet that assumes customers pay on time.

*Sample data is fictional (Atelier Verde SAS). Built by Manibalan Amaranathan for the Best Case Study Competition 2026
(Financial Navigator by A352).*
""")
