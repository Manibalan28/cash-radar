"""Cash Radar forecasting engine.

A direct (receipts and payments) weekly cash forecast:

1. Learn each customer's payment habit: days late after the due date, from
   invoices paid in the last 12 months.
2. Date every expected receipt at ``due date + the customer's median delay``.
   An invoice that is already overdue only uses past delays longer than the
   delay it already has.
3. Place planned payments on their own dates.
4. Sum by week from an opening bank balance.
5. Repeat the forecast many times, resampling past delays, to get a likely
   range (P10-P90) and the chance of falling below the minimum cash buffer.

Everything works on plain pandas DataFrames, so the same code runs on the
sample data, in the notebook, in the Streamlit app and in the tests.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
import pandas as pd

AFTER = "after horizon"


# --------------------------------------------------------------------------- helpers
def to_date(x) -> date:
    return pd.Timestamp(x).date()


def next_business_day(d: date) -> date:
    while d.weekday() >= 5:
        d += timedelta(days=1)
    return d


def week_number(d: date, start: date) -> int:
    """Week 1 is the 7 days starting on ``start``."""
    return (d - start).days // 7 + 1


# --------------------------------------------------------------------------- settings
@dataclass
class Settings:
    as_of: date                 # date of the data (last bank balance)
    forecast_start: date        # Monday of week 1
    opening_balance: float      # cash in the bank on ``as_of``
    buffer: float = 100_000     # minimum cash the CFO wants to keep
    overdraft_limit: float = 0  # authorised overdraft (positive number)
    weeks: int = 13
    lookback_days: int = 365
    company: str = ""

    @property
    def first_possible_receipt(self) -> date:
        return next_business_day(self.as_of + timedelta(days=3))

    @property
    def week_starts(self) -> list[date]:
        return [self.forecast_start + timedelta(weeks=i) for i in range(self.weeks)]


# --------------------------------------------------------------------------- 1. payment habits
def learn_delays(invoices: pd.DataFrame, as_of: date, lookback_days: int = 365) -> dict[str, np.ndarray]:
    """Days late (paid_date - due_date) per customer for invoices paid in the look-back window."""
    paid = invoices.dropna(subset=["paid_date"]).copy()
    paid["paid_date"] = paid["paid_date"].map(to_date)
    paid["due_date"] = paid["due_date"].map(to_date)
    window = paid[(paid.paid_date <= as_of) & (paid.paid_date > as_of - timedelta(days=lookback_days))]
    late = (pd.to_datetime(window.paid_date) - pd.to_datetime(window.due_date)).dt.days.astype(float)
    out = {cid: late[g.index].to_numpy() for cid, g in window.groupby("customer_id")}
    out["__all__"] = late.to_numpy()
    return out


def remaining_delays(samples: np.ndarray, already_late: int) -> np.ndarray:
    """Past delays still possible for an invoice that is already ``already_late`` days overdue."""
    if already_late <= 0:
        return samples
    longer = samples[samples > already_late]
    if len(longer) >= 3:
        return longer
    return np.array([already_late + 7.0, already_late + 14.0, already_late + 21.0])


def habit_table(invoices: pd.DataFrame, customers: pd.DataFrame, as_of: date, lookback_days: int = 365) -> pd.DataFrame:
    """One row per customer: usual delay, recent delay and a 'slowing down' flag."""
    full = learn_delays(invoices, as_of, lookback_days)
    recent = learn_delays(invoices, as_of, 91)
    rows = []
    for c in customers.itertuples():
        f, r = full.get(c.customer_id), recent.get(c.customer_id)
        med = float(np.median(f)) if f is not None and len(f) else np.nan
        rec = float(np.mean(r)) if r is not None and len(r) else np.nan
        rows.append(dict(customer_id=c.customer_id, customer_name=c.customer_name, segment=c.segment,
                         payment_terms_days=c.payment_terms_days, invoices_paid_12m=0 if f is None else len(f),
                         usual_days_late=med, recent_days_late_90d=rec,
                         slowing=bool(not np.isnan(med) and not np.isnan(rec) and rec > med + 7)))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- 2-3. forecast lines
def build_lines(invoices: pd.DataFrame, orders: pd.DataFrame, payments: pd.DataFrame,
                customers: pd.DataFrame, s: Settings) -> pd.DataFrame:
    """One row per expected cash movement in the horizon (receipts positive, payments negative)."""
    delays = learn_delays(invoices, s.as_of, s.lookback_days)
    seg = customers.set_index("customer_id").segment.to_dict()
    name = customers.set_index("customer_id").customer_name.to_dict()
    rows = []

    def receipt(src, rid, cid, amount, due, prob, status, desc):
        due = to_date(due)
        late_now = max(0, (s.as_of - due).days)
        samples = remaining_delays(delays.get(cid, delays["__all__"]), late_now)
        typical = float(np.median(samples))
        expected = max(next_business_day(due + timedelta(days=int(round(typical)))), s.first_possible_receipt)
        rows.append(dict(source=src, source_id=rid, flow="receipt", category=f"Customers – {seg.get(cid, 'other')}",
                         counterparty=name.get(cid, cid), customer_id=cid, amount=float(amount), probability=float(prob),
                         due_date=due, days_late_now=late_now, assumed_days_late=round(typical, 1),
                         expected_date=expected, status=status, included=status != "disputed",
                         deferrable=False, uncertainty_pct=0.0, description=desc, samples=samples))

    open_inv = invoices[invoices.paid_date.isna()]
    for r in open_inv.itertuples():
        receipt("invoice", r.invoice_id, r.customer_id, r.amount, r.due_date, 1.0, r.status, r.description)
    for r in orders.itertuples():
        receipt("order", r.order_id, r.customer_id, r.amount, r.due_date, r.probability, "expected", r.description)
    for r in payments.itertuples():
        rows.append(dict(source="payment", source_id=r.payment_id, flow="payment", category=r.category,
                         counterparty=r.counterparty, customer_id="", amount=-abs(float(r.amount)), probability=1.0,
                         due_date=to_date(r.date), days_late_now=0, assumed_days_late=0.0, expected_date=to_date(r.date),
                         status=r.certainty, included=True, deferrable=bool(r.deferrable),
                         uncertainty_pct=float(r.uncertainty_pct), description=r.note if isinstance(r.note, str) else "",
                         samples=None))
    lines = pd.DataFrame(rows)
    lines["expected_value"] = lines.amount * lines.probability
    lines["week"] = lines.expected_date.map(lambda d: _wk(d, s))
    return lines


def _wk(d: date, s: Settings):
    w = week_number(d, s.forecast_start)
    return w if w <= s.weeks else AFTER


# --------------------------------------------------------------------------- scenarios
@dataclass
class Scenario:
    """Extra payment delay (days) for named customers and/or whole segments."""
    name: str = "Base case"
    customer_delay: dict[str, int] = field(default_factory=dict)
    segment_delay: dict[str, int] = field(default_factory=dict)

    def extra_days(self, customer_id: str, category: str) -> int:
        seg = category.replace("Customers – ", "")
        return int(self.customer_delay.get(customer_id, 0)) + int(self.segment_delay.get(seg, 0))


def apply_scenario(lines: pd.DataFrame, s: Settings, sc: Scenario | None) -> pd.DataFrame:
    out = lines.copy()
    if sc is None:
        return out
    for i, r in out[out.flow == "receipt"].iterrows():
        extra = sc.extra_days(r.customer_id, r.category)
        if extra:
            d = next_business_day(r.expected_date + timedelta(days=extra))
            out.at[i, "expected_date"] = d
            out.at[i, "week"] = _wk(d, s)
    return out


# --------------------------------------------------------------------------- 4. weekly forecast
def weekly(lines: pd.DataFrame, s: Settings) -> pd.DataFrame:
    """Receipts, payments and closing balance per week (expected values)."""
    m = lines[lines.included & (lines.week != AFTER)].copy()
    m["week"] = m.week.astype(int)
    idx = range(1, s.weeks + 1)
    rec = m[m.flow == "receipt"].groupby("week").expected_value.sum().reindex(idx, fill_value=0.0)
    pay = m[m.flow == "payment"].groupby("week").expected_value.sum().reindex(idx, fill_value=0.0)
    out = pd.DataFrame({"week": list(idx), "week_start": s.week_starts, "receipts": rec.values, "payments": pay.values})
    out["net"] = out.receipts + out.payments
    out["closing"] = s.opening_balance + out.net.cumsum()
    out["opening"] = out.closing - out.net
    out["below_buffer"] = out.closing < s.buffer
    out["status"] = np.select([out.closing < -s.overdraft_limit, out.closing < 0, out.closing < s.buffer],
                              ["beyond overdraft", "overdraft", "below buffer"], "ok")
    return out[["week", "week_start", "opening", "receipts", "payments", "net", "closing", "below_buffer", "status"]]


# --------------------------------------------------------------------------- 5. uncertainty
def simulate(lines: pd.DataFrame, s: Settings, sc: Scenario | None = None, n: int = 4000, seed: int = 11) -> np.ndarray:
    """Closing balance for ``n`` simulated futures, shape (n, weeks)."""
    g = np.random.default_rng(seed)
    flows = np.zeros((n, s.weeks))
    floor = (s.first_possible_receipt - s.forecast_start).days
    for r in lines[lines.included].itertuples():
        if r.flow == "receipt":
            extra = sc.extra_days(r.customer_id, r.category) if sc else 0
            delays = g.choice(r.samples, size=n) + extra
            day = np.maximum((r.due_date - s.forecast_start).days + np.round(delays).astype(int), floor)
            w = day // 7
            happens = g.random(n) < r.probability
            ok = (w >= 0) & (w < s.weeks) & happens
            np.add.at(flows, (np.where(ok)[0], w[ok]), r.amount)
        else:
            if r.week == AFTER:
                continue
            amt = np.full(n, r.amount)
            if r.uncertainty_pct:
                amt = amt * g.normal(1, r.uncertainty_pct / 100, n)
            flows[:, int(r.week) - 1] += amt
    return s.opening_balance + flows.cumsum(axis=1)


def bands(sims: np.ndarray, s: Settings) -> pd.DataFrame:
    return pd.DataFrame({"week": range(1, s.weeks + 1),
                         "p10": np.percentile(sims, 10, axis=0), "p50": np.percentile(sims, 50, axis=0),
                         "p90": np.percentile(sims, 90, axis=0), "p_below_buffer": (sims < s.buffer).mean(axis=0),
                         "p_overdraft": (sims < 0).mean(axis=0)})


# --------------------------------------------------------------------------- warning
def warning(base: pd.DataFrame, scen: pd.DataFrame, lines_base: pd.DataFrame, lines_scen: pd.DataFrame,
            sims: np.ndarray, s: Settings, scenario_name: str = "") -> dict | None:
    """Plain-language warning for the first week below the buffer, or None if cash stays above it."""
    below = scen[scen.below_buffer]
    if below.empty:
        return None
    w = int(below.week.iloc[0])
    low = scen.loc[scen.closing.idxmin()]
    # what moved: receipts whose week changed between base and scenario
    moved = lines_scen[(lines_scen.flow == "receipt") & (lines_scen.week != lines_base.week)]
    moved = moved.assign(from_week=lines_base.loc[moved.index, "week"]).sort_values("amount", ascending=False)
    week_out = (lines_scen[(lines_scen.flow == "payment") & (lines_scen.week == w)]
                .groupby("counterparty").amount.sum().sort_values().head(3))
    return dict(
        scenario=scenario_name, first_week=w, first_week_start=s.week_starts[w - 1], lead_time_weeks=w - 1,
        lowest_balance=float(low.closing), lowest_week=int(low.week), gap_to_buffer=float(s.buffer - low.closing),
        into_overdraft=bool(low.closing < 0), beyond_overdraft=bool(low.closing < -s.overdraft_limit),
        chance_of_breach=float((sims.min(axis=1) < s.buffer).mean()),
        delayed_receipts=moved[["counterparty", "amount", "from_week", "week"]].head(3).to_dict("records"),
        big_payments_that_week=week_out.to_dict(),
    )


def warning_text(wn: dict | None, profit: float | None = None) -> str:
    if wn is None:
        return "Cash stays above the minimum buffer in every week of the forecast."
    eur = lambda v: f"{'−' if v < 0 else ''}€{abs(v):,.0f}"
    parts = [f"Cash falls below the buffer in week {wn['first_week']} (week of {wn['first_week_start']:%d %b %Y}) "
             f"and reaches {eur(wn['lowest_balance'])} in week {wn['lowest_week']}, {eur(wn['gap_to_buffer'])} below the buffer"
             + (", into the overdraft." if wn["into_overdraft"] else ".")]
    if wn["delayed_receipts"]:
        d = wn["delayed_receipts"][0]
        to = "after the horizon" if d["week"] == AFTER else f"week {d['week']}"
        parts.append(f"Main cause: {d['counterparty']}'s {eur(d['amount'])} payment moves from week {d['from_week']} to {to}.")
    if wn["big_payments_that_week"]:
        parts.append("Large payments that week: " + ", ".join(f"{k} {eur(-v)}" for k, v in wn["big_payments_that_week"].items()) + ".")
    parts.append(f"You have {wn['lead_time_weeks']} weeks to act. A shortfall happens in {wn['chance_of_breach']:.0%} of simulations.")
    if profit is not None:
        parts.append(f"Profit for the period is still {eur(profit)}: this is a timing problem, not a profit problem.")
    return " ".join(parts)


# --------------------------------------------------------------------------- CFO actions (what-if)
def act_collect_overdue(lines: pd.DataFrame, s: Settings, share: float = 0.5, week: int = 2) -> pd.DataFrame:
    """Collect the largest overdue invoices (up to ``share`` of the overdue total) in ``week``."""
    out = lines.copy()
    m = (out.source == "invoice") & (out.days_late_now > 0) & out.included
    idx = out[m].sort_values("amount", ascending=False).index
    target, done = share * out.loc[idx, "amount"].sum(), 0.0
    for i in idx:
        if done >= target:
            break
        out.at[i, "week"] = week
        done += out.at[i, "amount"]
    return out


def act_defer_flexible(lines: pd.DataFrame, s: Settings) -> pd.DataFrame:
    """Move payments marked as deferrable (e.g. capex) after the horizon."""
    out = lines.copy()
    out.loc[out.deferrable, "week"] = AFTER
    return out


def act_split_payment(lines: pd.DataFrame, s: Settings, source_id: str, later_weeks: int = 4) -> pd.DataFrame:
    """Pay half of one payment on time and half ``later_weeks`` weeks later."""
    out = lines.copy()
    i = out.index[out.source_id == source_id][0]
    half = out.loc[i].copy()
    out.at[i, "amount"] = out.at[i, "expected_value"] = out.at[i, "amount"] / 2
    half["amount"] = half["expected_value"] = out.at[i, "amount"]
    half["source_id"] = f"{source_id}-b"
    w = out.at[i, "week"]
    half["week"] = AFTER if w == AFTER or int(w) + later_weeks > s.weeks else int(w) + later_weeks
    return pd.concat([out, half.to_frame().T], ignore_index=True).astype({"included": bool, "deferrable": bool})


def act_finance_invoice(lines: pd.DataFrame, s: Settings, source_id: str, advance: float = 0.8,
                        week: int = 4, fee_pct: float = 0.015) -> pd.DataFrame:
    """Receive ``advance`` of an invoice early (invoice financing); the rest arrives when the customer pays."""
    out = lines.copy()
    i = out.index[out.source_id == source_id][0]
    full = out.at[i, "amount"]
    early = out.loc[i].copy()
    early["amount"] = early["expected_value"] = full * advance - full * fee_pct
    early["source_id"] = f"{source_id}-advance"
    early["week"] = week
    out.at[i, "amount"] = out.at[i, "expected_value"] = full * (1 - advance)
    return pd.concat([out, early.to_frame().T], ignore_index=True).astype({"included": bool, "deferrable": bool})


# --------------------------------------------------------------------------- backtest
def balance_history(transactions: pd.DataFrame, s: Settings) -> pd.Series:
    """Daily end-of-day bank balance, rebuilt backwards from the balance on ``as_of``."""
    tx = transactions.copy()
    tx["date"] = pd.to_datetime(tx.date)
    daily = tx.groupby("date").amount.sum()
    days = pd.date_range(daily.index.min() - pd.Timedelta(days=1), pd.Timestamp(s.as_of))
    daily = daily.reindex(days, fill_value=0.0)
    after = daily[::-1].cumsum()[::-1].shift(-1, fill_value=0.0)  # flows after each day, up to as_of
    return s.opening_balance - after


def backtest(invoices: pd.DataFrame, transactions: pd.DataFrame, s: Settings, step_weeks: int = 1,
             warmup_days: int = 0) -> pd.DataFrame:
    """Re-run the forecast every Friday in the past and compare with the real balance.

    Isolates the hard part, *when customers pay*: payments are taken as known,
    and invoices issued during the window are assumed known in advance.
    Compares this method with a naive forecast where everyone pays on the due date.
    """
    bal = balance_history(transactions, s)
    inv = invoices.copy()
    for c in ("issue_date", "due_date", "paid_date"):
        inv[c] = pd.to_datetime(inv[c])
    out_tx = transactions[transactions.amount < 0].copy()
    out_tx["date"] = pd.to_datetime(out_tx.date)
    first = bal.index.min() + pd.Timedelta(days=warmup_days)  # optionally wait for more history
    first += pd.Timedelta(days=(4 - first.weekday()) % 7)    # a Friday
    rows = []
    t = first
    while (t + pd.Timedelta(days=3 + 7 * s.weeks)).date() <= s.as_of:
        w0 = t + pd.Timedelta(days=3)
        end = w0 + pd.Timedelta(weeks=s.weeks)
        delays = learn_delays(invoices.assign(paid_date=invoices.paid_date.where(pd.to_datetime(invoices.paid_date) <= t)),
                              t.date(), s.lookback_days)
        cand = inv[((inv.issue_date <= t) & ((inv.paid_date > t) | inv.paid_date.isna()))
                   | ((inv.issue_date > t) & (inv.issue_date <= end))]
        model, naive = np.zeros(s.weeks), np.zeros(s.weeks)
        floor = t + pd.Timedelta(days=3)
        for r in cand.itertuples():
            late_now = max(0, (t - r.due_date).days)
            samp = remaining_delays(delays.get(r.customer_id, delays["__all__"]), late_now)
            dm = max(r.due_date + pd.Timedelta(days=int(round(np.median(samp)))), floor)
            dn = max(r.due_date, floor)
            for d, arr in ((dm, model), (dn, naive)):
                k = (d - w0).days // 7
                if 0 <= k < s.weeks:
                    arr[k] += r.amount
        pays = out_tx[(out_tx.date > t) & (out_tx.date < end)]
        p = np.zeros(s.weeks)
        for r in pays.itertuples():
            k = (r.date - w0).days // 7
            if 0 <= k < s.weeks:
                p[k] += r.amount
        start_bal = bal.loc[t]
        for k in range(s.weeks):
            week_end = w0 + pd.Timedelta(days=7 * k + 6)
            rows.append(dict(forecast_date=t.date(), weeks_ahead=k + 1, week_start=(w0 + pd.Timedelta(weeks=k)).date(),
                             actual=float(bal.loc[week_end]),
                             forecast=float(start_bal + model[:k + 1].sum() + p[:k + 1].sum()),
                             naive=float(start_bal + naive[:k + 1].sum() + p[:k + 1].sum())))
        t += pd.Timedelta(weeks=step_weeks)
    bt = pd.DataFrame(rows)
    bt["error"] = bt.forecast - bt.actual
    bt["error_naive"] = bt.naive - bt.actual
    return bt


def backtest_summary(bt: pd.DataFrame, buffer: float) -> dict:
    mae, mae_n = bt.error.abs().mean(), bt.error_naive.abs().mean()
    flagged = bt[bt.forecast < buffer]
    return dict(forecasts=int(bt.forecast_date.nunique()), mae=float(mae), mae_naive=float(mae_n),
                error_reduction=float(1 - mae / mae_n), bias=float(bt.error.mean()),
                false_alarm_share=float((flagged.actual >= buffer).mean()) if len(flagged) else 0.0)


def lead_times(bt: pd.DataFrame, buffer: float) -> pd.DataFrame:
    """For each real dip below the buffer: how many weeks ahead each method first flagged it."""
    actual = bt.drop_duplicates("week_start").set_index("week_start").actual.sort_index()
    weeks = list(actual[actual < buffer].index)
    episodes = []
    for w in weeks:
        if episodes and (w - episodes[-1][-1]).days <= 7:
            episodes[-1].append(w)
        else:
            episodes.append([w])
    rows = []
    for ep in episodes:
        r = bt[bt.week_start == ep[0]]
        lt = lambda col: max(0, int(r[r[col] < buffer].weeks_ahead.max()) - 1) if (r[col] < buffer).any() else 0
        rows.append(dict(first_week=ep[0], weeks_below=len(ep), lowest=float(actual[ep].min()),
                         lead_time_weeks=lt("forecast"), lead_time_naive=lt("naive")))
    return pd.DataFrame(rows)
