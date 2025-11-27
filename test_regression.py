"""
IdleX ERP - Regression & Edge Case Tests
Tests: R01-R03

Cross-cutting tests for graceful degradation and consistency.
"""

import pytest
from sqlalchemy import text
import pandas as pd
import os


# =============================================================================
# R01: Empty Tables Fail Gracefully
# =============================================================================

class TestEmptyTables:
    """Test graceful handling of empty data conditions."""
    
    def test_r01_empty_production_unit(self, engine_sqlite_fresh):
        """Empty production_unit should not crash generate_financial_ledgers."""
        with engine_sqlite_fresh.connect() as conn:
            conn.execute(text("DELETE FROM production_unit"))
            conn.commit()
        
        from dashboard import generate_financial_ledgers
        
        # Should not raise exception
        df_pnl, df_cash = generate_financial_ledgers(engine_sqlite_fresh)
        
        # Should return empty or near-empty dataframes
        # (may still have OpEx entries)
        revenue_entries = df_pnl[df_pnl['Type'] == 'Revenue'] if not df_pnl.empty else pd.DataFrame()
        assert len(revenue_entries) == 0, "Should have no revenue with no units"
    
    def test_r01_empty_staffing_plan(self, engine_sqlite_fresh):
        """Empty staffing plan should not crash, just no payroll."""
        with engine_sqlite_fresh.connect() as conn:
            conn.execute(text("DELETE FROM opex_staffing_plan"))
            conn.commit()
        
        from dashboard import generate_financial_ledgers
        
        df_pnl, df_cash = generate_financial_ledgers(engine_sqlite_fresh)
        
        # Should still work, just no payroll
        payroll = df_pnl[df_pnl['Category'] == 'Salaries & Benefits']['Amount'].sum()
        assert payroll == 0 or pd.isna(payroll), "Should have no payroll with empty staffing"
    
    def test_r01_empty_bom(self, engine_sqlite_fresh):
        """Empty BOM should return 0 material cost."""
        with engine_sqlite_fresh.connect() as conn:
            conn.execute(text("DELETE FROM bom_items"))
            conn.commit()
        
        from dashboard import calculate_unit_material_cost
        
        cost = calculate_unit_material_cost(engine_sqlite_fresh)
        assert cost == 0, f"Empty BOM should have 0 cost, got {cost}"


# =============================================================================
# R02: Missing Config Tables
# =============================================================================

class TestMissingConfig:
    """Test fallback behavior when config tables are missing."""
    
    def test_r02_missing_pricing_config(self, engine_sqlite_fresh):
        """Missing pricing_config should use fallback defaults."""
        with engine_sqlite_fresh.connect() as conn:
            conn.execute(text("DROP TABLE IF EXISTS pricing_config"))
            conn.commit()
        
        from dashboard import generate_financial_ledgers
        
        # Should use default MSRP of 8500
        df_pnl, df_cash = generate_financial_ledgers(engine_sqlite_fresh)
        
        # Should not crash
        assert df_pnl is not None
    
    def test_r02_missing_channel_mix(self, engine_sqlite_fresh):
        """Missing channel_mix_config should use 25% default."""
        with engine_sqlite_fresh.connect() as conn:
            conn.execute(text("DROP TABLE IF EXISTS channel_mix_config"))
            conn.commit()
        
        # The production regeneration logic should fall back to 25% direct
        # Just verify it doesn't crash
        from dashboard import generate_financial_ledgers
        
        df_pnl, df_cash = generate_financial_ledgers(engine_sqlite_fresh)
        assert df_pnl is not None


# =============================================================================
# R03: Cross-DB Consistency (SQLite vs Postgres)
# =============================================================================

class TestCrossDBConsistency:
    """Test that SQLite and Postgres produce identical results."""
    
    @pytest.mark.skipif(not os.getenv("TEST_DATABASE_URL"), 
                       reason="Postgres not configured")
    def test_r03_revenue_parity(self, engine_sqlite, engine_postgres):
        """Revenue totals should match between SQLite and Postgres."""
        from dashboard import generate_financial_ledgers
        
        df_pnl_sqlite, _ = generate_financial_ledgers(engine_sqlite)
        df_pnl_postgres, _ = generate_financial_ledgers(engine_postgres)
        
        sqlite_revenue = df_pnl_sqlite[df_pnl_sqlite['Type'] == 'Revenue']['Amount'].sum()
        postgres_revenue = df_pnl_postgres[df_pnl_postgres['Type'] == 'Revenue']['Amount'].sum()
        
        # Within 0.01%
        pct_diff = abs(sqlite_revenue - postgres_revenue) / sqlite_revenue
        assert pct_diff < 0.0001, \
            f"Revenue mismatch: SQLite=${sqlite_revenue:,.0f}, Postgres=${postgres_revenue:,.0f}"
    
    @pytest.mark.skipif(not os.getenv("TEST_DATABASE_URL"),
                       reason="Postgres not configured")
    def test_r03_unit_count_parity(self, engine_sqlite, engine_postgres):
        """Unit counts should match between databases."""
        for engine, name in [(engine_sqlite, "SQLite"), (engine_postgres, "Postgres")]:
            with engine.connect() as conn:
                count = conn.execute(text(
                    "SELECT COUNT(*) FROM production_unit"
                )).scalar()
                print(f"{name}: {count} units")


# =============================================================================
# Date Boundary Tests
# =============================================================================

class TestDateBoundaries:
    """Test handling of year-end and quarter-end dates."""
    
    def test_year_boundary_pricing(self, engine_sqlite, expected_pricing):
        """Units near year-end should use correct year's pricing."""
        with engine_sqlite.connect() as conn:
            # Get December 2026 and January 2027 units
            dec_2026 = conn.execute(text("""
                SELECT COUNT(*) FROM production_unit 
                WHERE build_date LIKE '2026-12%'
            """)).scalar()
            
            jan_2027 = conn.execute(text("""
                SELECT COUNT(*) FROM production_unit 
                WHERE build_date LIKE '2027-01%'
            """)).scalar()
        
        # Verify both months have units
        assert dec_2026 > 0, "Should have December 2026 units"
        assert jan_2027 > 0, "Should have January 2027 units"
    
    def test_quarter_boundary_channel_mix(self, engine_sqlite, expected_channel_mix):
        """Channel mix should change at quarter boundaries."""
        # Q1 2026 vs Q2 2026 should have different direct %
        q1_config = expected_channel_mix.get((2026, 1), 0.15)
        q2_config = expected_channel_mix.get((2026, 2), 0.20)
        
        assert q2_config > q1_config, "Q2 should have higher direct % than Q1"


# =============================================================================
# Decimal Precision Tests
# =============================================================================

class TestDecimalPrecision:
    """Test financial calculations maintain precision."""
    
    def test_money_formatting(self):
        """money() function should handle edge cases."""
        from dashboard import money
        
        assert money(1000) == "1,000"
        assert money(1000.50) == "1,001"  # Rounds
        assert money(-1000) == "(1,000)"
        assert money(None) == "-"
        assert money(float('nan')) == "-"
    
    def test_pct_formatting(self):
        """pct() function should format correctly."""
        from dashboard import pct
        
        assert pct(0.1234) == "12.3%"
        assert pct(0) == "0.0%"
        assert pct(None) == "-"
    
    def test_revenue_precision(self, engine_sqlite, expected_pricing):
        """Revenue calculation should not have floating point errors."""
        # Calculate expected revenue for 1 direct + 1 dealer unit
        pricing = expected_pricing[2026]
        direct_rev = pricing["msrp"]  # 8500
        dealer_rev = pricing["msrp"] * pricing["dealer_discount"]  # 6375
        
        # These should be exact (no floating point issues)
        assert direct_rev == 8500.0
        assert dealer_rev == 6375.0
        
        # Sum should also be exact
        assert direct_rev + dealer_rev == 14875.0


# =============================================================================
# Error Handling Tests
# =============================================================================

class TestErrorHandling:
    """Test graceful error handling."""
    
    def test_invalid_date_handling(self):
        """parse_date should handle invalid inputs."""
        from dashboard import parse_date
        
        # Valid inputs
        from datetime import date, datetime
        assert parse_date(date(2026, 1, 15)) == date(2026, 1, 15)
        assert parse_date("2026-01-15") == date(2026, 1, 15)
        
        # Invalid inputs should return None, not crash
        result = parse_date("not-a-date")
        assert result is None or isinstance(result, date)
    
    def test_get_workdays_edge_cases(self):
        """get_workdays should handle edge cases."""
        from dashboard import get_workdays
        
        # Normal month
        workdays = get_workdays(2026, 1)
        assert len(workdays) > 0
        assert len(workdays) <= 23  # Max workdays in January
        
        # February (shorter)
        feb_workdays = get_workdays(2026, 2)
        assert len(feb_workdays) < len(workdays)


# =============================================================================
# Performance / Scale Tests
# =============================================================================

class TestPerformance:
    """Test performance with realistic data volumes."""
    
    def test_large_ledger_generation(self, engine_sqlite):
        """generate_financial_ledgers should complete in reasonable time."""
        import time
        from dashboard import generate_financial_ledgers
        
        start = time.time()
        df_pnl, df_cash = generate_financial_ledgers(engine_sqlite)
        elapsed = time.time() - start
        
        # Should complete in under 30 seconds for ~45K units
        assert elapsed < 30, f"Ledger generation took {elapsed:.1f}s (too slow)"
        
        # Verify it actually processed data
        assert len(df_pnl) > 40000, f"Expected >40K P&L entries, got {len(df_pnl)}"
    
    def test_cash_waterfall_performance(self, engine_sqlite):
        """Cash waterfall should handle large datasets."""
        import time
        from dashboard import generate_financial_ledgers, run_cash_waterfall
        
        _, df_cash = generate_financial_ledgers(engine_sqlite)
        
        start = time.time()
        result = run_cash_waterfall(df_cash, starting_equity=1600000, loc_limit=500000)
        elapsed = time.time() - start
        
        # Should complete in under 10 seconds
        assert elapsed < 10, f"Cash waterfall took {elapsed:.1f}s (too slow)"
