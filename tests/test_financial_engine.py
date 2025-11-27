"""
IdleX ERP - Financial Engine Tests (generate_financial_ledgers)
Tests: F01-F07

Core financial calculations - treat like a mini accounting system.
"""

import pytest
from sqlalchemy import text
import pandas as pd
import numpy as np
from datetime import date
from conftest import assert_close


# =============================================================================
# F01: Unit Material Cost
# =============================================================================

class TestUnitMaterialCost:
    """Verify unit material cost calculations."""
    
    def test_f01_calculate_unit_material_cost(self, engine_sqlite):
        """calculate_unit_material_cost should match SQL calculation."""
        # Import the function
        import sys
        sys.path.insert(0, '..')
        from dashboard import calculate_unit_material_cost
        
        # Calculate via function
        func_cost = calculate_unit_material_cost(engine_sqlite)
        
        # Calculate via SQL
        with engine_sqlite.connect() as conn:
            sql_cost = conn.execute(text("""
                SELECT SUM(p.cost * b.qty_per_unit)
                FROM bom_items b
                JOIN part_master p ON b.part_id = p.id
            """)).scalar()
        
        assert_close(func_cost, sql_cost)
    
    def test_f01_material_cost_range(self, engine_sqlite):
        """Unit material cost should be in reasonable range."""
        from dashboard import calculate_unit_material_cost
        
        cost = calculate_unit_material_cost(engine_sqlite)
        
        # Expected: $3,500 - $4,500 based on BOM
        assert 3000 < cost < 5000, f"Material cost ${cost:,.2f} outside expected range"


# =============================================================================
# F02: Revenue Per Unit Logic
# =============================================================================

class TestRevenuePerUnit:
    """Verify revenue calculations per unit and channel."""
    
    def test_f02_direct_revenue(self, engine_sqlite, expected_pricing):
        """Direct sales should use full MSRP."""
        msrp = expected_pricing[2026]["msrp"]
        
        # Direct revenue = MSRP
        expected_direct_revenue = msrp
        assert expected_direct_revenue == 8500
    
    def test_f02_dealer_revenue(self, engine_sqlite, expected_pricing):
        """Dealer sales should use discounted price."""
        msrp = expected_pricing[2026]["msrp"]
        discount = expected_pricing[2026]["dealer_discount"]
        
        # Dealer revenue = MSRP × discount
        expected_dealer_revenue = msrp * discount
        assert expected_dealer_revenue == 6375  # 8500 × 0.75
    
    def test_f02_dealer_less_than_direct(self, engine_sqlite, expected_pricing):
        """Dealer price must always be less than direct."""
        for year, pricing in expected_pricing.items():
            direct = pricing["msrp"]
            dealer = pricing["msrp"] * pricing["dealer_discount"]
            
            assert dealer < direct, f"Year {year}: Dealer ${dealer} >= Direct ${direct}"


# =============================================================================
# F03: Monthly Revenue Totals
# =============================================================================

class TestMonthlyRevenue:
    """Verify aggregate revenue calculations."""
    
    def test_f03_generate_financial_ledgers_returns_data(self, engine_sqlite):
        """generate_financial_ledgers should return non-empty dataframes."""
        from dashboard import generate_financial_ledgers
        
        df_pnl, df_cash = generate_financial_ledgers(engine_sqlite)
        
        assert not df_pnl.empty, "P&L dataframe is empty"
        assert not df_cash.empty, "Cash dataframe is empty"
    
    def test_f03_revenue_matches_manual_calc(self, engine_sqlite, expected_pricing):
        """Total 2026 revenue should match manual calculation."""
        from dashboard import generate_financial_ledgers
        
        # Get unit counts by channel
        with engine_sqlite.connect() as conn:
            units = conn.execute(text("""
                SELECT sales_channel, COUNT(*) as cnt
                FROM production_unit
                WHERE substr(build_date, 1, 4) = '2026'
                GROUP BY sales_channel
            """)).fetchall()
        
        channel_counts = {row[0]: row[1] for row in units}
        direct_count = channel_counts.get('DIRECT', 0)
        dealer_count = channel_counts.get('DEALER', 0)
        
        # Calculate expected revenue
        pricing = expected_pricing[2026]
        expected_revenue = (direct_count * pricing["msrp"]) + \
                          (dealer_count * pricing["msrp"] * pricing["dealer_discount"])
        
        # Get actual from ledger
        df_pnl, _ = generate_financial_ledgers(engine_sqlite)
        
        # Filter 2026 revenue
        df_pnl['Date'] = pd.to_datetime(df_pnl['Date'])
        revenue_2026 = df_pnl[
            (df_pnl['Date'].dt.year == 2026) & 
            (df_pnl['Type'] == 'Revenue')
        ]['Amount'].sum()
        
        assert_close(revenue_2026, expected_revenue, tolerance_pct=0.02)


# =============================================================================
# F04: COGS Calculation
# =============================================================================

class TestCOGSCalculation:
    """Verify COGS matches unit cost × volume."""
    
    def test_f04_total_cogs_calculation(self, engine_sqlite):
        """Total COGS should equal material_cost × total_units."""
        from dashboard import generate_financial_ledgers, calculate_unit_material_cost
        
        # Get unit count
        with engine_sqlite.connect() as conn:
            total_units = conn.execute(text(
                "SELECT COUNT(*) FROM production_unit"
            )).scalar()
        
        # Get unit material cost
        unit_cost = calculate_unit_material_cost(engine_sqlite)
        
        # Expected COGS (negative)
        expected_cogs = -total_units * unit_cost
        
        # Get actual from ledger
        df_pnl, _ = generate_financial_ledgers(engine_sqlite)
        actual_cogs = df_pnl[
            (df_pnl['Type'] == 'COGS') & 
            (df_pnl['Category'] == 'Raw Materials')
        ]['Amount'].sum()
        
        assert_close(actual_cogs, expected_cogs, tolerance_pct=0.02)


# =============================================================================
# F05: Payroll from Staffing Plan
# =============================================================================

class TestPayrollCalculation:
    """Verify payroll matches staffing plan."""
    
    def test_f05_monthly_payroll_calculation(self, engine_sqlite):
        """Monthly payroll should match staffing × salaries / 12."""
        # Calculate expected from staffing plan
        with engine_sqlite.connect() as conn:
            expected_payroll = conn.execute(text("""
                SELECT SUM(r.annual_salary / 12.0 * s.headcount) as total
                FROM opex_staffing_plan s
                JOIN opex_roles r ON s.role_id = r.id
            """)).scalar()
        
        # Get actual from ledger
        from dashboard import generate_financial_ledgers
        df_pnl, df_cash = generate_financial_ledgers(engine_sqlite)
        
        actual_payroll_pnl = abs(df_pnl[
            df_pnl['Category'] == 'Salaries & Benefits'
        ]['Amount'].sum())
        
        assert_close(actual_payroll_pnl, expected_payroll, tolerance_pct=0.05)


# =============================================================================
# F06: General Expenses
# =============================================================================

class TestGeneralExpenses:
    """Verify general expenses flow correctly."""
    
    def test_f06_expenses_in_pnl(self, engine_sqlite):
        """Each general expense should appear in P&L."""
        from dashboard import generate_financial_ledgers
        
        # Get expense categories from DB
        with engine_sqlite.connect() as conn:
            expense_sum = conn.execute(text("""
                SELECT SUM(amount) FROM opex_general_expenses
            """)).scalar() or 0
        
        # Get from ledger
        df_pnl, _ = generate_financial_ledgers(engine_sqlite)
        
        # General expenses are OpEx (excluding Salaries & Benefits)
        pnl_expenses = abs(df_pnl[
            (df_pnl['Type'] == 'OpEx') & 
            (df_pnl['Category'] != 'Salaries & Benefits')
        ]['Amount'].sum())
        
        # Should be close (may have other OpEx categories)
        if expense_sum > 0:
            assert pnl_expenses >= expense_sum * 0.9, \
                f"P&L expenses ${pnl_expenses:,.0f} < DB expenses ${expense_sum:,.0f}"


# =============================================================================
# F07: Ledger Completeness
# =============================================================================

class TestLedgerCompleteness:
    """Verify all units generate proper entries."""
    
    def test_f07_revenue_entry_per_unit(self, engine_sqlite):
        """Each unit should generate one revenue entry."""
        from dashboard import generate_financial_ledgers
        
        with engine_sqlite.connect() as conn:
            unit_count = conn.execute(text(
                "SELECT COUNT(*) FROM production_unit"
            )).scalar()
        
        df_pnl, _ = generate_financial_ledgers(engine_sqlite)
        revenue_entries = len(df_pnl[df_pnl['Type'] == 'Revenue'])
        
        assert revenue_entries == unit_count, \
            f"Expected {unit_count} revenue entries, got {revenue_entries}"
    
    def test_f07_cogs_entry_per_unit(self, engine_sqlite):
        """Each unit should generate one COGS entry."""
        from dashboard import generate_financial_ledgers
        
        with engine_sqlite.connect() as conn:
            unit_count = conn.execute(text(
                "SELECT COUNT(*) FROM production_unit"
            )).scalar()
        
        df_pnl, _ = generate_financial_ledgers(engine_sqlite)
        cogs_entries = len(df_pnl[
            (df_pnl['Type'] == 'COGS') & 
            (df_pnl['Category'] == 'Raw Materials')
        ])
        
        assert cogs_entries == unit_count, \
            f"Expected {unit_count} COGS entries, got {cogs_entries}"
    
    def test_f07_cash_entry_per_unit(self, engine_sqlite):
        """Each unit should generate one cash collection entry."""
        from dashboard import generate_financial_ledgers
        
        with engine_sqlite.connect() as conn:
            unit_count = conn.execute(text(
                "SELECT COUNT(*) FROM production_unit"
            )).scalar()
        
        _, df_cash = generate_financial_ledgers(engine_sqlite)
        cash_entries = len(df_cash[df_cash['Category'] == 'Customer Collections'])
        
        assert cash_entries == unit_count, \
            f"Expected {unit_count} cash entries, got {cash_entries}"
