# Data

`sample/` holds the **fictional** company Atelier Verde SAS (a 38-person furniture maker near Lyon).
Everything is simulated by `scripts/make_sample_data.py` with a fixed random seed. French payment rules and
Banque de France statistics were used to make it realistic; no real company or person is represented.

To use your own data in the app, provide CSV files with the same columns (extra columns are ignored).
Dates are `YYYY-MM-DD`; amounts are in euros **including VAT** (the cash that actually moves).

| File | One row per | Required columns | Optional |
|---|---|---|---|
| `customers.csv` | customer | `customer_id`, `customer_name`, `segment`, `payment_terms_days` | `country`, `payment_method` |
| `invoices.csv` | customer invoice, paid or not (≥ 12 months of history) | `invoice_id`, `customer_id`, `issue_date`, `due_date`, `amount`, `paid_date` (blank if unpaid), `status` (`paid` / `open` / `disputed`) | `description` |
| `orders.csv` | invoice expected but not yet issued | `order_id`, `customer_id`, `due_date`, `amount`, `probability` (0–1) | `description`, `expected_invoice_date` |
| `planned_payments.csv` | payment due in the next 13 weeks | `payment_id`, `date`, `category`, `counterparty`, `amount` (positive) | `certainty`, `deferrable` (True/False), `uncertainty_pct`, `note` |
| `bank_transactions.csv` | past bank movement (for the accuracy test) | `date`, `amount` (+ in / − out), `category`, `counterparty` | `reference` |
| `settings.json` | — | `as_of`, `forecast_start`, `opening_balance` | `buffer`, `overdraft_limit`, `weeks`, `profit_for_period`, `company` |

Also included: `profit_and_loss_monthly.csv` (accrual-basis P&L, to compare profit with cash).

**Sample company facts (9 Oct 2026):** cash €381,241 · minimum buffer €100,000 · overdraft €150,000 ·
1,210 invoices since Dec 2024 · 12 customers · 374 cash lines in the 13-week window · 2 disputed invoices.
