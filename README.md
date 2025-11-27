# IdleX ERP - Comprehensive Test Suite

## Overview

This test suite validates all financial calculations in the IdleX ERP system to ensure CFO-grade accuracy for board presentations.

## Quick Start

```bash
# Install dependencies
pip install pytest pytest-html pytest-xdist pandas sqlalchemy

# Navigate to test directory
cd D:\Dropbox\IdleX\IdleX_ERP\tests

# Run all tests
python run_tests.py

# Run with verbose output
python run_tests.py -v

# Run specific module
python run_tests.py --module cash

# Generate HTML report
python run_tests.py --html report.html

# Quick sanity check
python run_tests.py --quick

# Generate CFO summary
python run_tests.py --summary
```

## Test Modules

| Module | File | Tests | Description |
|--------|------|-------|-------------|
| Schema & Seeding | `test_seed_schema.py` | S01-S08 | Database structure and initial data |
| Financial Engine | `test_financial_engine.py` | F01-F07 | Revenue, COGS, payroll calculations |
| Cash Waterfall | `test_cash_waterfall.py` | C01-C07 | Cash flow and LOC logic |
| Production & Channel | `test_production.py` | P01-P04 | Channel mix synchronization |
| Regression | `test_regression.py` | R01-R03 | Edge cases and cross-DB parity |

## Test Categories

### S: Schema & Seeding Tests
- **S01**: Core tables exist with expected columns
- **S02**: Global config values (start_cash, loc_limit, msrp)
- **S03**: BOM cost = sum(cost × qty_per_unit)
- **S04**: Production unit counts match plan
- **S05**: Total units per year (2026: 2,187 | 2027: 11,322 | 2028: 31,351)
- **S06**: Staffing rows and scaling
- **S07**: Pricing config per year
- **S08**: Channel mix config progression

### F: Financial Engine Tests
- **F01**: Unit material cost matches seeder
- **F02**: Direct vs dealer revenue per unit
- **F03**: Aggregate P&L revenue vs manual calc
- **F04**: P&L COGS = material_cost × units
- **F05**: Payroll from staffing plan
- **F06**: General expenses in P&L
- **F07**: Ledger completeness (1 entry per unit)

### C: Cash Waterfall Tests
- **C01**: Pure inflow (no LOC)
- **C02**: Material outflow with sufficient cash
- **C03**: Material outflow draws LOC
- **C04**: LOC limit enforcement
- **C05**: OpEx cannot use LOC (strict mode)
- **C06**: OpEx can use LOC (relaxed mode)
- **C07**: Cash + LOC identity

### P: Production & Channel Tests
- **P01**: Only PLANNED units deleted on regeneration
- **P02**: Schedule on workdays only
- **P03**: Channel mix enforcement per quarter
- **P04**: Manifest edits persist

### R: Regression Tests
- **R01**: Empty tables fail gracefully
- **R02**: Missing config uses fallbacks
- **R03**: SQLite vs Postgres parity

## Critical Bug Fixed

### Channel Mix Inconsistency

**Problem**: Production Manifest showed 75% dealer while Channel Mix Config showed different percentages (e.g., 1% or 15%).

**Root Cause**: `seed_db.py` used hardcoded 25% direct split instead of reading from `channel_mix_config`.

**Impact**:
- Revenue forecasts using channel config were wrong
- Cash timing predictions incorrect (dealers have 30-day payment lag)
- Margin calculations per channel inaccurate

**Fix Applied**: Modified `seed_db.py` to use quarterly channel mix percentages when creating production units.

**Test Coverage**: `test_production.py::TestChannelMixConsistency`

## Running Against PostgreSQL

```bash
# Set environment variable
export TEST_DATABASE_URL=postgresql://user:pass@host/dbname

# Run with Postgres
python run_tests.py --postgres
```

## Expected Metrics

After running `python run_tests.py --summary`:

```
📊 PRODUCTION SUMMARY
  2026:    2,187 units
  2027:   11,322 units
  2028:   31,351 units
  Total:  44,860 units

📈 CHANNEL MIX (Actual)
  DIRECT: ~30%
  DEALER: ~70%

💰 REVENUE BY YEAR
  2026: $15,000,000
  2027: $85,000,000
  2028: $216,000,000

🔧 UNIT ECONOMICS
  Material Cost/Unit:  $3,800
  Direct Gross Margin:    55%
  Dealer Gross Margin:    40%
```

## Files in Test Suite

```
tests/
├── conftest.py              # Shared fixtures
├── run_tests.py             # Main test runner
├── test_seed_schema.py      # Schema & seeding tests
├── test_financial_engine.py # Financial calculations
├── test_cash_waterfall.py   # Cash flow logic
├── test_production.py       # Channel mix & production
└── test_regression.py       # Edge cases & compatibility
```

## What's Still Missing (Future Work)

### Not Yet Implemented

1. **UI Interaction Tests** - Streamlit button clicks, form submissions
2. **Concurrent Session Tests** - Multiple users editing simultaneously
3. **Data Migration Tests** - Schema upgrades without data loss
4. **Audit Trail Verification** - Every financial change logged
5. **Export/Import Tests** - Excel, PDF generation accuracy
6. **Scenario Planning Tests** - What-if model calculations
7. **API Endpoint Tests** - If REST API is added later
8. **Load/Stress Tests** - 100K+ units, many concurrent users

### Recommended Additional Validations

1. **Covenant Compliance Tests** - Debt covenants from covenant_config
2. **Warranty Reserve Tests** - Accrual calculations
3. **Service Revenue Tests** - Subscription billing logic
4. **Fleet Assignment Tests** - Unit tracking accuracy
5. **Inventory Reorder Tests** - MRP logic validation

## CI/CD Integration

Add to GitHub Actions or similar:

```yaml
name: Financial Integrity Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - run: pip install pytest pytest-html pandas sqlalchemy
      - run: python tests/run_tests.py --html report.html
      - uses: actions/upload-artifact@v3
        with:
          name: test-report
          path: report.html
```

## Sign-Off Checklist

Before board presentation:

- [ ] All automated tests pass (`python run_tests.py`)
- [ ] CFO summary shows expected ranges (`python run_tests.py --summary`)
- [ ] Channel mix matches across modules
- [ ] Revenue calculation verified manually
- [ ] Cash waterfall shows no unexpected crunches
- [ ] Gross margins in expected range (40-55%)
- [ ] Database rebuilt with latest seed (`Manual Database Rebuild` button)
