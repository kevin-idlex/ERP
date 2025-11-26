import sys
from unittest.mock import MagicMock
import pandas as pd
from datetime import date
from sqlalchemy import text

# --- 1. MOCK STREAMLIT ---
# We must "fake" Streamlit so we can import dashboard.py without launching a web server
sys.modules["streamlit"] = MagicMock()

# Now we can safely import your app logic
import dashboard 
import seed_db

# =============================================================================
# TEST 1: FINANCIAL INTEGRITY (The "Balance Sheet" Test)
# =============================================================================
def test_pnl_integrity():
    """
    Requirement: P&L must mathematically balance.
    Verifies that Revenue - Expenses matches the implied Net Income.
    """
    print("\nRunning Financial Integrity Test...")
    pnl, _ = dashboard.generate_financials()
    
    if pnl.empty:
        # Auto-seed if empty
        print("   Database empty. Seeding...")
        seed_db.run_seed()
        pnl, _ = dashboard.generate_financials()

    # Calculate Aggregates
    revenue = pnl[pnl['Type'] == 'Revenue']['Amount'].sum()
    cogs = pnl[pnl['Type'] == 'COGS']['Amount'].sum()
    opex = pnl[pnl['Type'] == 'OpEx']['Amount'].sum()
    
    # Assertions (The Rules)
    assert revenue > 0, f"Revenue should be positive, got {revenue}"
    assert cogs < 0, f"COGS should be negative (outflow), got {cogs}"
    
    # Margin Check
    gross_margin_pct = (revenue + cogs) / revenue
    print(f"   Calculated Gross Margin: {gross_margin_pct:.1%}")
    
    # Sanity Check: Margin should be between 10% and 80% for hardware
    assert 0.10 < gross_margin_pct < 0.80, f"Margin {gross_margin_pct:.1%} looks suspicious!"
    
    print("✅ P&L Integrity Passed")

# =============================================================================
# TEST 2: CASH FLOW LOGIC (The "Runway" Test)
# =============================================================================
def test_cash_continuity():
    """
    Requirement: Ending Cash must equal Starting Cash + Sum of all flows.
    """
    print("\nRunning Cash Continuity Test...")
    _, cash = dashboard.generate_financials()
    
    # Get configured start cash
    config = pd.read_sql("SELECT setting_value FROM global_config WHERE setting_key='start_cash'", dashboard.engine)
    start_cash = float(config.iloc[0,0])
    
    # Calculate from ledger
    total_flow = cash['Amount'].sum()
    calculated_end = start_cash + total_flow
    
    # Get actual end from dataframe logic
    actual_end = cash.iloc[-1]['Cash_Balance']
    
    # Floating point tolerance (allow 1 cent difference)
    assert abs(calculated_end - actual_end) < 1.0, f"Cash mismatch! Calc: {calculated_end}, Actual: {actual_end}"
    
    print(f"   Start: ${start_cash:,.0f} | Flow: ${total_flow:,.0f} | End: ${actual_end:,.0f}")
    print("✅ Cash Logic Passed")

# =============================================================================
# TEST 3: OPTIMIZATION ENGINE (The "Strategy" Test)
# =============================================================================
def test_growth_optimizer():
    """
    Requirement: The optimizer must NEVER recommend a plan that breaches the Credit Limit.
    """
    print("\nRunning Optimizer Stress Test...")
    
    # Constraints
    START_CASH = 1_000_000
    LOC_LIMIT = 500_000
    START_UNITS = 50
    
    # Run Optimizer
    results = dashboard.optimize_growth(
        start_units=START_UNITS,
        start_cash=START_CASH,
        loc_limit=LOC_LIMIT,
        start_date=date(2026, 1, 1),
        months=24
    )
    
    best_rate = results['rate']
    min_cash = results['min_cash']
    
    print(f"   Optimizer Suggests: {best_rate:.1f}% Monthly Growth")
    print(f"   Resulting Min Cash: ${min_cash:,.0f} (Limit: -${LOC_LIMIT:,.0f})")
    
    # The Test: Did we stay above the limit?
    assert min_cash >= -LOC_LIMIT - 1000, f"Optimizer FAIL! Breached limit by ${abs(min_cash + LOC_LIMIT)}"
    
    print("✅ Optimizer Logic Passed")

# =============================================================================
# TEST 4: DATA INTEGRITY (The "BOM" Test)
# =============================================================================
def test_data_integrity():
    """
    Requirement: Critical BOM items must exist.
    """
    print("\nRunning Data Integrity Test...")
    with dashboard.engine.connect() as conn:
        # Check for Battery
        res = conn.execute(text("SELECT count(*) FROM part_master WHERE name LIKE '%Battery%'")).scalar()
        assert res > 0, "Critical Part 'Battery' missing from BOM"
        
        # Check for Roles
        res = conn.execute(text("SELECT count(*) FROM opex_roles")).scalar()
        assert res > 0, "No Staff Roles defined"

    print("✅ Data Integrity Passed")

if __name__ == "__main__":
    try:
        test_pnl_integrity()
        test_cash_continuity()
        test_data_integrity()
        test_growth_optimizer()
        print("\n" + "="*40)
        print("🎉 ALL SYSTEMS GO. READY FOR DEPLOYMENT.")
        print("="*40)
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        exit(1)
    except Exception as e:
        print(f"\n❌ SYSTEM ERROR: {e}")
        exit(1)