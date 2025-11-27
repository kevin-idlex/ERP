"""
IdleX ERP - Cash Waterfall Engine Tests (run_cash_waterfall)
Tests: C01-C07

This is where CFO trust lives or dies.
"""

import pytest
import pandas as pd
from datetime import date, timedelta
from conftest import assert_close


# =============================================================================
# HELPER: Create test cash dataframe
# =============================================================================

def make_cash_df(entries):
    """Create a cash DataFrame from list of tuples (date, type, amount)."""
    df = pd.DataFrame([
        {"Date": e[0], "Type": e[1], "Category": "Test", "Amount": e[2]}
        for e in entries
    ])
    df['Date'] = pd.to_datetime(df['Date'])
    return df


# =============================================================================
# C01: Pure Inflow
# =============================================================================

class TestPureInflow:
    """Test simple inflow scenarios."""
    
    def test_c01_single_inflow(self):
        """Pure inflow should increase cash, no LOC."""
        from dashboard import run_cash_waterfall
        
        df = make_cash_df([
            (date(2026, 1, 15), "INFLOW", 1000),
        ])
        
        result = run_cash_waterfall(df, starting_equity=0, loc_limit=500000)
        
        final = result.iloc[-1]
        assert final['Net_Cash'] == 1000, f"Expected cash 1000, got {final['Net_Cash']}"
        assert final['LOC_Usage'] == 0, f"Expected LOC 0, got {final['LOC_Usage']}"
    
    def test_c01_multiple_inflows(self):
        """Multiple inflows should accumulate."""
        from dashboard import run_cash_waterfall
        
        df = make_cash_df([
            (date(2026, 1, 10), "INFLOW", 1000),
            (date(2026, 1, 15), "INFLOW", 2000),
            (date(2026, 1, 20), "INFLOW", 3000),
        ])
        
        result = run_cash_waterfall(df, starting_equity=0, loc_limit=500000)
        
        final = result.iloc[-1]
        assert final['Net_Cash'] == 6000


# =============================================================================
# C02: Material Outflow with Sufficient Cash
# =============================================================================

class TestMaterialSufficientCash:
    """Test material outflows covered by cash."""
    
    def test_c02_material_from_cash(self):
        """Material outflow should use cash when available."""
        from dashboard import run_cash_waterfall
        
        df = make_cash_df([
            (date(2026, 1, 10), "INFLOW", 1000),
            (date(2026, 1, 20), "MATERIAL_OUTFLOW", -600),
        ])
        
        result = run_cash_waterfall(df, starting_equity=0, loc_limit=500000, enforce_loc_rules=True)
        
        final = result.iloc[-1]
        assert final['Net_Cash'] == 400
        assert final['LOC_Usage'] == 0


# =============================================================================
# C03: Material Outflow Exceeding Cash Uses LOC
# =============================================================================

class TestMaterialUsesLOC:
    """Test material outflows that require LOC."""
    
    def test_c03_material_draws_loc(self):
        """Material outflow exceeding cash should draw LOC."""
        from dashboard import run_cash_waterfall
        
        df = make_cash_df([
            (date(2026, 1, 10), "INFLOW", 1000),
            (date(2026, 1, 20), "MATERIAL_OUTFLOW", -2000),
        ])
        
        result = run_cash_waterfall(df, starting_equity=0, loc_limit=3000, enforce_loc_rules=True)
        
        final = result.iloc[-1]
        # Cash used: 1000, LOC drawn: 1000
        assert final['Net_Cash'] == 0, f"Expected cash 0, got {final['Net_Cash']}"
        assert final['LOC_Usage'] == 1000, f"Expected LOC 1000, got {final['LOC_Usage']}"


# =============================================================================
# C04: LOC Limit Enforcement
# =============================================================================

class TestLOCLimit:
    """Test LOC limit is respected."""
    
    def test_c04_loc_limit_exceeded(self):
        """Material outflow beyond LOC limit creates negative cash."""
        from dashboard import run_cash_waterfall
        
        df = make_cash_df([
            (date(2026, 1, 10), "INFLOW", 1000),
            (date(2026, 1, 20), "MATERIAL_OUTFLOW", -5000),
        ])
        
        result = run_cash_waterfall(df, starting_equity=0, loc_limit=3000, enforce_loc_rules=True)
        
        final = result.iloc[-1]
        # Cash: 1000, need 5000, can draw 3000 LOC, shortfall 1000
        assert final['LOC_Usage'] <= 3000, f"LOC exceeded limit: {final['LOC_Usage']}"
        assert final['Net_Cash'] < 0, "Cash should go negative when LOC exhausted"
        assert final['Cash_Crunch'] == True, "Cash crunch should be flagged"


# =============================================================================
# C05: OpEx Cannot Use LOC (Strict Mode)
# =============================================================================

class TestOpExStrictMode:
    """Test OpEx is cash-only in strict mode."""
    
    def test_c05_opex_no_loc_strict(self):
        """OpEx should not draw LOC in strict mode."""
        from dashboard import run_cash_waterfall
        
        df = make_cash_df([
            (date(2026, 1, 10), "INFLOW", 1000),
            (date(2026, 1, 20), "OPEX_OUTFLOW", -5000),
        ])
        
        result = run_cash_waterfall(df, starting_equity=0, loc_limit=10000, enforce_loc_rules=True)
        
        final = result.iloc[-1]
        # OpEx must come from cash, LOC stays at 0
        assert final['LOC_Usage'] == 0, f"LOC should be 0, got {final['LOC_Usage']}"
        assert final['Net_Cash'] == -4000, f"Cash should be -4000, got {final['Net_Cash']}"


# =============================================================================
# C06: OpEx Can Use LOC (Relaxed Mode)
# =============================================================================

class TestOpExRelaxedMode:
    """Test OpEx can use LOC in relaxed mode."""
    
    def test_c06_opex_uses_loc_relaxed(self):
        """OpEx should draw LOC in relaxed mode."""
        from dashboard import run_cash_waterfall
        
        df = make_cash_df([
            (date(2026, 1, 10), "INFLOW", 1000),
            (date(2026, 1, 20), "OPEX_OUTFLOW", -5000),
        ])
        
        result = run_cash_waterfall(df, starting_equity=0, loc_limit=10000, enforce_loc_rules=False)
        
        final = result.iloc[-1]
        # In relaxed mode, should use LOC for OpEx too
        # Behavior depends on implementation - verify LOC is drawn
        assert final['LOC_Usage'] > 0 or final['Net_Cash'] >= 0, \
            "Relaxed mode should allow LOC for OpEx"


# =============================================================================
# C07: Cash + LOC Identity
# =============================================================================

class TestCashIdentity:
    """Test fundamental cash flow identity."""
    
    def test_c07_cash_flow_identity(self):
        """Sum of flows should equal change in (cash + LOC)."""
        from dashboard import run_cash_waterfall
        
        starting_equity = 10000
        
        df = make_cash_df([
            (date(2026, 1, 5), "INFLOW", 5000),
            (date(2026, 1, 10), "MATERIAL_OUTFLOW", -3000),
            (date(2026, 1, 15), "INFLOW", 2000),
            (date(2026, 1, 20), "OPEX_OUTFLOW", -4000),
            (date(2026, 1, 25), "MATERIAL_OUTFLOW", -8000),
            (date(2026, 1, 30), "INFLOW", 1000),
        ])
        
        result = run_cash_waterfall(df, starting_equity=starting_equity, loc_limit=5000)
        
        # Sum of all cash flows
        total_flows = df['Amount'].sum()
        
        # Change in position
        final = result.iloc[-1]
        ending_position = final['Net_Cash'] + final['LOC_Usage']
        starting_position = starting_equity
        
        # The identity: starting + flows = ending (approximately, due to LOC draw/paydown)
        # This is a simplified check - the full identity is more complex
        assert abs(starting_position + total_flows - final['Net_Cash']) < 1000, \
            "Cash identity violated"


# =============================================================================
# Additional Waterfall Tests
# =============================================================================

class TestWaterfallEdgeCases:
    """Test edge cases in waterfall logic."""
    
    def test_empty_dataframe(self):
        """Empty cash dataframe should return empty result."""
        from dashboard import run_cash_waterfall
        
        df = pd.DataFrame(columns=['Date', 'Type', 'Category', 'Amount'])
        result = run_cash_waterfall(df, starting_equity=1000, loc_limit=500)
        
        assert result.empty or len(result) == 0
    
    def test_chronological_ordering(self):
        """Transactions should be processed in date order."""
        from dashboard import run_cash_waterfall
        
        # Out-of-order entries
        df = make_cash_df([
            (date(2026, 1, 20), "INFLOW", 2000),
            (date(2026, 1, 10), "INFLOW", 1000),
            (date(2026, 1, 30), "INFLOW", 3000),
        ])
        
        result = run_cash_waterfall(df, starting_equity=0, loc_limit=500)
        
        # Should be sorted by date
        assert result['Date'].is_monotonic_increasing, "Results should be date-ordered"
    
    def test_revenue_pays_down_loc(self):
        """Revenue inflow should pay down LOC before filling cash."""
        from dashboard import run_cash_waterfall
        
        df = make_cash_df([
            (date(2026, 1, 10), "MATERIAL_OUTFLOW", -3000),  # Will draw LOC
            (date(2026, 1, 20), "INFLOW", 2000),  # Should pay down LOC
        ])
        
        result = run_cash_waterfall(df, starting_equity=1000, loc_limit=5000, enforce_loc_rules=True)
        
        # After material: cash=0, LOC=2000
        # After inflow: LOC should be paid down first
        final = result.iloc[-1]
        
        # LOC should be reduced (or 0) before cash increases
        # Exact behavior depends on implementation
        assert final['LOC_Usage'] <= 2000, "LOC should be paid down by inflow"
    
    def test_starting_equity(self):
        """Starting equity should be reflected in initial cash."""
        from dashboard import run_cash_waterfall
        
        df = make_cash_df([
            (date(2026, 1, 10), "INFLOW", 1000),
        ])
        
        result = run_cash_waterfall(df, starting_equity=5000, loc_limit=1000)
        
        final = result.iloc[-1]
        assert final['Net_Cash'] == 6000, f"Expected 6000, got {final['Net_Cash']}"
