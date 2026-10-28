"""Tests for the Cash Radar engine, run with:  pytest"""
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

import cash_radar as cr
from cash_radar.engine import learn_delays, next_business_day, remaining_delays, week_number


@pytest.fixture(scope="module")
def sample():
    ds = cr.load()
    lines = cr.build_lines(ds.invoices, ds.orders, ds.planned_payments, ds.customers, ds.settings)
    return ds, lines


# ---------------------------------------------------------------- small helpers
def test_next_business_day_skips_weekend():
    assert next_business_day(date(2026, 10, 10)) == date(2026, 10, 12)  # Saturday -> Monday
    assert next_business_day(date(2026, 10, 12)) == date(2026, 10, 12)


def test_week_number():
    start = date(2026, 10, 12)
    assert week_number(start, start) == 1
    assert week_number(start + timedelta(days=6), start) == 1
    assert week_number(start + timedelta(days=7), start) == 2


def test_remaining_delays_uses_only_longer_delays():
    s = np.array([0, 5, 10, 20, 30, 40.0])
    assert set(remaining_delays(s, 12)) == {20, 30, 40}
    assert len(remaining_delays(s, 100)) == 3  # fallback when history has no longer delay


def test_learn_delays_tiny_example():
    inv = pd.DataFrame({"customer_id": ["A", "A", "B"],
                        "due_date": [date(2026, 1, 1)] * 3,
                        "paid_date": [date(2026, 1, 11), date(2026, 1, 21), date(2026, 1, 1)]})
    d = learn_delays(inv, date(2026, 6, 1))
    assert sorted(d["A"]) == [10, 20] and list(d["B"]) == [0]


# ---------------------------------------------------------------- forecast on the sample company
def test_cash_balances_add_up(sample):
    ds, lines = sample
    w = cr.weekly(lines, ds.settings)
    assert w.closing.iloc[-1] == pytest.approx(ds.settings.opening_balance + w.net.sum())
    assert (w.opening.iloc[1:].values == pytest.approx(w.closing.iloc[:-1].values))


def test_base_case_stays_just_above_buffer(sample):
    ds, lines = sample
    w = cr.weekly(lines, ds.settings)
    assert w.closing.min() == pytest.approx(100_857, abs=1)
    assert int(w.loc[w.closing.idxmin(), "week"]) == 3
    assert not w.below_buffer.any()


def test_lumen_delay_triggers_warning_seven_weeks_ahead(sample):
    ds, lines = sample
    s = ds.settings
    sc = cr.Scenario("Lumen +45", {"C01": 45})
    l1 = cr.apply_scenario(lines, s, sc)
    base, scen = cr.weekly(lines, s), cr.weekly(l1, s)
    wn = cr.warning(base, scen, lines, l1, cr.simulate(lines, s, sc, n=500), s)
    assert wn["first_week"] == 8 and wn["lead_time_weeks"] == 7
    assert wn["lowest_balance"] == pytest.approx(-66_075, abs=1)
    assert wn["delayed_receipts"][0]["counterparty"] == "Lumen Hospitality Group"
    assert "timing problem" in cr.warning_text(wn, ds.profit_for_period)


def test_disputed_invoices_are_excluded(sample):
    _, lines = sample
    disputed = lines[lines.status == "disputed"]
    assert len(disputed) == 2 and not disputed.included.any()


def test_simulation_range_contains_expected_path(sample):
    ds, lines = sample
    b = cr.bands(cr.simulate(lines, ds.settings, n=1000), ds.settings)
    assert (b.p10 <= b.p90).all()
    assert ((b.p50 - cr.weekly(lines, ds.settings).closing).abs() < 120_000).all()


def test_actions_improve_lowest_cash(sample):
    ds, lines = sample
    s = ds.settings
    l1 = cr.apply_scenario(lines, s, cr.Scenario(customer_delay={"C01": 45}))
    low = cr.weekly(l1, s).closing.min()
    assert cr.weekly(cr.act_defer_flexible(l1, s), s).closing.min() > low
    collected = cr.weekly(cr.act_collect_overdue(l1, s), s)
    assert collected.closing.iloc[1] > cr.weekly(l1, s).closing.iloc[1]  # cash arrives earlier
    assert collected.closing.min() >= low
    big = l1[l1.flow == "payment"].sort_values("amount").source_id.iloc[0]
    split = cr.act_split_payment(l1, s, big)
    assert split.amount.sum() == pytest.approx(l1.amount.sum())  # money is moved, not created


def test_backtest_beats_naive(sample):
    ds, _ = sample
    bt = cr.backtest(ds.invoices, ds.bank_transactions, ds.settings, step_weeks=4)
    sm = cr.backtest_summary(bt, ds.settings.buffer)
    assert sm["mae"] < 0.5 * sm["mae_naive"]
