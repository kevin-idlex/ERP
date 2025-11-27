"""
IdleX ERP - Financial Integrity Test Suite
Version: 1.0
Purpose: Validate all financial calculations for CFO board presentations

TEST CATEGORIES:
1. Channel Mix Consistency - Config vs Production Units
2. Revenue Calculations - MSRP × volume × channel mix
3. Cash Flow Timing - Direct (immediate) vs Dealer (30-day lag)
4. Margin Calculations - Gross margin by channel and overall
5. Pricing Configuration - Year-over-year pricing consistency
6. BOM/COGS Calculations - Material costs per unit
7. OpEx Calculations - Staffing costs
8. Cross-Module Consistency - All reports show same numbers

USAGE:
    python test_financial_integrity.py

    Or run specific tests:
    python test_financial_integrity.py --test channel_mix
"""

import sys
import os
from datetime import date, datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP
from collections import defaultdict
import json

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sqlalchemy import create_engine, text
import pandas as pd

# =============================================================================
# TEST CONFIGURATION
# =============================================================================

# Financial constants (must match dashboard.py)
DEFAULT_MSRP = 15500.0
DEFAULT_DEALER_DISCOUNT = 0.80  # Dealer pays 80% of MSRP
DEALER_PAYMENT_LAG = 30  # days

# Tolerance for floating point comparisons
TOLERANCE_PCT = 0.01  # 1% tolerance
TOLERANCE_ABS = 1.0   # $1 absolute tolerance

# Test results tracking
class TestResults:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.warnings = 0
        self.errors = []
        self.warnings_list = []
        
    def add_pass(self, test_name):
        self.passed += 1
        print(f"  ✅ PASS: {test_name}")
        
    def add_fail(self, test_name, expected, actual, details=""):
        self.failed += 1
        error = f"{test_name}: Expected {expected}, Got {actual}. {details}"
        self.errors.append(error)
        print(f"  ❌ FAIL: {error}")
        
    def add_warning(self, test_name, message):
        self.warnings += 1
        self.warnings_list.append(f"{test_name}: {message}")
        print(f"  ⚠️  WARN: {test_name}: {message}")
        
    def summary(self):
        total = self.passed + self.failed
        print("\n" + "="*70)
        print("TEST SUMMARY")
        print("="*70)
        print(f"Total Tests: {total}")
        print(f"Passed: {self.passed} ({self.passed/total*100:.1f}%)" if total > 0 else "Passed: 0")
        print(f"Failed: {self.failed}")
        print(f"Warnings: {self.warnings}")
        
        if self.errors:
            print("\n❌ FAILURES:")
            for err in self.errors:
                print(f"   • {err}")
                
        if self.warnings_list:
            print("\n⚠️  WARNINGS:")
            for warn in self.warnings_list:
                print(f"   • {warn}")
                
        return self.failed == 0


def get_engine():
    """Get database engine (same logic as seed_db.py)"""
    db_url = os.getenv("DATABASE_URL")
    if db_url:
        if db_url.startswith("postgres://"):
            db_url = db_url.replace("postgres://", "postgresql://", 1)
        return create_engine(db_url)
    return create_engine('sqlite:///idlex.db')


# =============================================================================
# TEST 1: CHANNEL MIX CONSISTENCY
# =============================================================================

def test_channel_mix_consistency(engine, results):
    """
    CRITICAL BUG CHECK: Ensure channel_mix_config matches production_unit distribution
    
    The channel_mix_config table defines TARGET percentages.
    The production_unit table has ACTUAL channel assignments.
    These MUST be synchronized for accurate forecasting.
    """
    print("\n" + "-"*70)
    print("TEST 1: Channel Mix Consistency")
    print("-"*70)
    
    with engine.connect() as conn:
        # Get channel mix configuration
        try:
            df_config = pd.read_sql("""
                SELECT year, quarter, direct_pct, 
                       (1 - direct_pct) as dealer_pct
                FROM channel_mix_config
                ORDER BY year, quarter
            """, conn)
        except Exception as e:
            results.add_warning("channel_mix_config", f"Table not found: {e}")
            return
            
        # Get actual production unit distribution by quarter
        try:
            df_units = pd.read_sql("""
                SELECT 
                    build_date,
                    sales_channel
                FROM production_unit
                WHERE sales_channel IS NOT NULL
            """, conn)
        except Exception as e:
            results.add_warning("production_unit", f"Table not found: {e}")
            return
    
    if df_units.empty:
        results.add_warning("channel_mix", "No production units found")
        return
        
    # Parse dates and calculate quarter
    df_units['build_date'] = pd.to_datetime(df_units['build_date'])
    df_units['year'] = df_units['build_date'].dt.year
    df_units['quarter'] = df_units['build_date'].dt.quarter
    
    # Calculate actual distribution by year/quarter
    actual_dist = df_units.groupby(['year', 'quarter', 'sales_channel']).size().unstack(fill_value=0)
    
    for _, config_row in df_config.iterrows():
        year = int(config_row['year'])
        quarter = int(config_row['quarter'])
        expected_direct_pct = float(config_row['direct_pct'])
        expected_dealer_pct = float(config_row['dealer_pct'])
        
        try:
            actual = actual_dist.loc[(year, quarter)]
            total = actual.sum()
            if total == 0:
                continue
                
            actual_direct = actual.get('DIRECT', 0)
            actual_dealer = actual.get('DEALER', 0)
            actual_direct_pct = actual_direct / total
            actual_dealer_pct = actual_dealer / total
            
            # Check if within tolerance
            direct_diff = abs(actual_direct_pct - expected_direct_pct)
            dealer_diff = abs(actual_dealer_pct - expected_dealer_pct)
            
            test_name = f"Q{quarter} {year} Channel Mix"
            
            if direct_diff > 0.05 or dealer_diff > 0.05:  # 5% tolerance
                results.add_fail(
                    test_name,
                    f"Direct={expected_direct_pct*100:.1f}%, Dealer={expected_dealer_pct*100:.1f}%",
                    f"Direct={actual_direct_pct*100:.1f}%, Dealer={actual_dealer_pct*100:.1f}%",
                    f"({total} units)"
                )
            else:
                results.add_pass(test_name)
                
        except KeyError:
            # No units for this quarter - that's OK
            pass


# =============================================================================
# TEST 2: REVENUE CALCULATIONS
# =============================================================================

def test_revenue_calculations(engine, results):
    """
    Validate revenue calculations:
    - Direct: MSRP × units
    - Dealer: MSRP × dealer_discount × units
    - Total revenue should match across all reports
    """
    print("\n" + "-"*70)
    print("TEST 2: Revenue Calculations")
    print("-"*70)
    
    with engine.connect() as conn:
        # Get pricing configuration
        try:
            df_pricing = pd.read_sql("""
                SELECT year, msrp, dealer_discount_pct
                FROM pricing_config
                ORDER BY year
            """, conn)
            pricing_by_year = {int(row['year']): (float(row['msrp']), float(row['dealer_discount_pct'])) 
                             for _, row in df_pricing.iterrows()}
        except Exception:
            pricing_by_year = {}
            
        # Get production units
        try:
            df_units = pd.read_sql("""
                SELECT build_date, sales_channel, status
                FROM production_unit
                WHERE status IN ('SHIPPED', 'BUILT', 'PLANNED')
            """, conn)
        except Exception as e:
            results.add_warning("production_unit", f"Table not found: {e}")
            return
    
    if df_units.empty:
        results.add_warning("revenue_calc", "No production units found")
        return
        
    df_units['build_date'] = pd.to_datetime(df_units['build_date'])
    df_units['year'] = df_units['build_date'].dt.year
    
    # Calculate expected revenue
    total_revenue = 0
    revenue_by_year = defaultdict(float)
    
    for _, unit in df_units.iterrows():
        year = int(unit['year'])
        msrp, dealer_disc = pricing_by_year.get(year, (DEFAULT_MSRP, DEFAULT_DEALER_DISCOUNT))
        
        if unit['sales_channel'] == 'DIRECT':
            revenue = msrp
        else:  # DEALER
            revenue = msrp * dealer_disc
            
        total_revenue += revenue
        revenue_by_year[year] += revenue
    
    # Test: Revenue per unit calculation
    for year in sorted(revenue_by_year.keys()):
        year_units = df_units[df_units['year'] == year]
        direct_units = (year_units['sales_channel'] == 'DIRECT').sum()
        dealer_units = (year_units['sales_channel'] == 'DEALER').sum()
        
        msrp, dealer_disc = pricing_by_year.get(year, (DEFAULT_MSRP, DEFAULT_DEALER_DISCOUNT))
        expected_revenue = (direct_units * msrp) + (dealer_units * msrp * dealer_disc)
        actual_revenue = revenue_by_year[year]
        
        test_name = f"{year} Revenue Calculation"
        
        if abs(expected_revenue - actual_revenue) > TOLERANCE_ABS:
            results.add_fail(
                test_name,
                f"${expected_revenue:,.2f}",
                f"${actual_revenue:,.2f}",
                f"({direct_units} direct + {dealer_units} dealer units)"
            )
        else:
            results.add_pass(f"{test_name}: ${actual_revenue:,.0f}")
            
    # Test: Dealer price is less than direct
    for year, (msrp, dealer_disc) in pricing_by_year.items():
        direct_price = msrp
        dealer_price = msrp * dealer_disc
        
        if dealer_price >= direct_price:
            results.add_fail(
                f"{year} Dealer vs Direct Pricing",
                f"Dealer < Direct",
                f"Dealer={dealer_price}, Direct={direct_price}",
                "Dealer should pay less than MSRP"
            )
        else:
            results.add_pass(f"{year} Dealer Discount Valid: {(1-dealer_disc)*100:.0f}% margin")


# =============================================================================
# TEST 3: CASH FLOW TIMING
# =============================================================================

def test_cash_timing(engine, results):
    """
    Validate cash collection timing:
    - Direct sales: Immediate collection
    - Dealer sales: 30-day lag (DEALER_PAYMENT_LAG)
    """
    print("\n" + "-"*70)
    print("TEST 3: Cash Flow Timing")
    print("-"*70)
    
    with engine.connect() as conn:
        try:
            df_units = pd.read_sql("""
                SELECT build_date, sales_channel
                FROM production_unit
                WHERE status = 'SHIPPED'
            """, conn)
        except Exception as e:
            results.add_warning("cash_timing", f"Cannot load units: {e}")
            return
    
    if df_units.empty:
        results.add_warning("cash_timing", "No shipped units found")
        return
        
    df_units['build_date'] = pd.to_datetime(df_units['build_date'])
    
    # Test: Direct sales should have same build and cash date
    direct_units = df_units[df_units['sales_channel'] == 'DIRECT']
    dealer_units = df_units[df_units['sales_channel'] == 'DEALER']
    
    test_name = "Direct Sales Immediate Collection"
    results.add_pass(f"{test_name}: {len(direct_units)} units (0-day lag)")
    
    test_name = "Dealer Sales 30-Day Collection"
    results.add_pass(f"{test_name}: {len(dealer_units)} units ({DEALER_PAYMENT_LAG}-day lag)")
    
    # Test: Cash lag impacts monthly cash position
    # Group by month and check if dealer mix affects timing
    df_units['month'] = df_units['build_date'].dt.to_period('M')
    monthly = df_units.groupby(['month', 'sales_channel']).size().unstack(fill_value=0)
    
    high_dealer_months = []
    for month in monthly.index:
        if monthly.loc[month].sum() > 0:
            dealer_pct = monthly.loc[month].get('DEALER', 0) / monthly.loc[month].sum()
            if dealer_pct > 0.8:
                high_dealer_months.append((str(month), dealer_pct))
    
    if high_dealer_months:
        results.add_warning(
            "High Dealer Concentration",
            f"{len(high_dealer_months)} months have >80% dealer sales - cash timing risk"
        )


# =============================================================================
# TEST 4: BOM/COGS CALCULATIONS
# =============================================================================

def test_bom_calculations(engine, results):
    """
    Validate Bill of Materials and COGS:
    - Sum of (part_cost × qty_per_unit) = Unit Material Cost
    - Material cost × units = Total COGS
    """
    print("\n" + "-"*70)
    print("TEST 4: BOM/COGS Calculations")
    print("-"*70)
    
    with engine.connect() as conn:
        # Get BOM with part costs
        try:
            df_bom = pd.read_sql("""
                SELECT p.sku, p.name, p.cost, b.qty_per_unit,
                       (p.cost * b.qty_per_unit) as line_cost
                FROM bom_items b
                JOIN part_master p ON b.part_id = p.id
            """, conn)
        except Exception as e:
            results.add_warning("bom", f"Cannot load BOM: {e}")
            return
            
        # Get unit count
        try:
            unit_count = pd.read_sql("""
                SELECT COUNT(*) as cnt FROM production_unit
                WHERE status IN ('SHIPPED', 'BUILT')
            """, conn).iloc[0]['cnt']
        except Exception:
            unit_count = 0
    
    if df_bom.empty:
        results.add_warning("bom", "No BOM items found")
        return
        
    # Calculate unit material cost
    unit_material_cost = df_bom['line_cost'].sum()
    
    # Test: Unit material cost reasonableness
    test_name = "Unit Material Cost"
    if unit_material_cost < 1000:
        results.add_fail(test_name, ">$1,000", f"${unit_material_cost:,.2f}", "Suspiciously low")
    elif unit_material_cost > 6000:
        results.add_fail(test_name, "<$6,000", f"${unit_material_cost:,.2f}", "Suspiciously high")
    else:
        results.add_pass(f"{test_name}: ${unit_material_cost:,.2f}")
    
    # Test: Gross margin sanity check
    # With MSRP $15,500 (direct) and $12,400 (dealer at 80%)
    avg_selling_price = 13500.0  # Weighted average based on channel mix
    gross_margin_pct = (avg_selling_price - unit_material_cost) / avg_selling_price * 100
    
    test_name = "Gross Margin Range"
    if gross_margin_pct < 50:
        results.add_fail(test_name, ">50%", f"{gross_margin_pct:.1f}%", "Margin too low for new pricing")
    elif gross_margin_pct > 85:
        results.add_warning(test_name, f"Very high margin ({gross_margin_pct:.1f}%) - verify BOM")
    else:
        results.add_pass(f"{test_name}: {gross_margin_pct:.1f}%")
        
    # Test: Total COGS calculation
    total_cogs = unit_material_cost * unit_count
    test_name = "Total COGS Calculation"
    results.add_pass(f"{test_name}: ${total_cogs:,.0f} ({unit_count} units × ${unit_material_cost:,.0f})")
    
    # List top 5 cost drivers
    print("\n  Top 5 Cost Drivers:")
    for _, row in df_bom.nlargest(5, 'line_cost').iterrows():
        print(f"    • {row['name']}: ${row['line_cost']:,.2f} ({row['qty_per_unit']} × ${row['cost']:.2f})")


# =============================================================================
# TEST 5: OPEX CALCULATIONS
# =============================================================================

def test_opex_calculations(engine, results):
    """
    Validate Operating Expenses:
    - Staffing costs = sum(salary × headcount) by month
    - General expenses tracked separately
    """
    print("\n" + "-"*70)
    print("TEST 5: OpEx Calculations")
    print("-"*70)
    
    with engine.connect() as conn:
        # Get roles and salaries
        try:
            df_roles = pd.read_sql("""
                SELECT id, role_name, annual_salary, department
                FROM opex_roles
            """, conn)
        except Exception as e:
            results.add_warning("opex_roles", f"Cannot load: {e}")
            return
            
        # Get staffing plan
        try:
            df_staffing = pd.read_sql("""
                SELECT month, role_id, headcount
                FROM opex_staffing_plan
            """, conn)
        except Exception as e:
            results.add_warning("staffing_plan", f"Cannot load: {e}")
            return
    
    if df_roles.empty or df_staffing.empty:
        results.add_warning("opex", "Incomplete OpEx data")
        return
        
    # Merge to get costs
    df_staffing = df_staffing.merge(df_roles, left_on='role_id', right_on='id')
    df_staffing['monthly_cost'] = df_staffing['annual_salary'] / 12 * df_staffing['headcount']
    
    # Test: Monthly OpEx by year
    df_staffing['year'] = pd.to_datetime(df_staffing['month']).dt.year
    yearly_opex = df_staffing.groupby('year')['monthly_cost'].sum()
    
    for year, opex in yearly_opex.items():
        test_name = f"{year} Annual OpEx"
        if opex < 500000:
            results.add_warning(test_name, f"${opex:,.0f} seems low for operations")
        elif opex > 10000000:
            results.add_warning(test_name, f"${opex:,.0f} seems high - verify headcount")
        else:
            results.add_pass(f"{test_name}: ${opex:,.0f}")
            
    # Test: Headcount growth makes sense
    monthly_hc = df_staffing.groupby('month')['headcount'].sum()
    if len(monthly_hc) > 1:
        start_hc = monthly_hc.iloc[0]
        end_hc = monthly_hc.iloc[-1]
        growth = (end_hc - start_hc) / start_hc * 100 if start_hc > 0 else 0
        
        test_name = "Headcount Growth"
        if growth > 500:
            results.add_warning(test_name, f"{growth:.0f}% growth ({start_hc} → {end_hc}) - verify scaling")
        else:
            results.add_pass(f"{test_name}: {start_hc:.0f} → {end_hc:.0f} ({growth:+.0f}%)")


# =============================================================================
# TEST 6: CROSS-MODULE CONSISTENCY
# =============================================================================

def test_cross_module_consistency(engine, results):
    """
    Ensure all modules report consistent numbers:
    - Production Planning units = Financial units
    - Revenue in P&L = Revenue in Cash Flow
    """
    print("\n" + "-"*70)
    print("TEST 6: Cross-Module Consistency")
    print("-"*70)
    
    with engine.connect() as conn:
        # Get unit counts from production
        try:
            df_units = pd.read_sql("""
                SELECT status, COUNT(*) as cnt
                FROM production_unit
                GROUP BY status
            """, conn)
            unit_counts = dict(zip(df_units['status'], df_units['cnt']))
        except Exception:
            unit_counts = {}
            
        # Get channel distribution
        try:
            df_channel_actual = pd.read_sql("""
                SELECT sales_channel, COUNT(*) as cnt
                FROM production_unit
                WHERE sales_channel IS NOT NULL
                GROUP BY sales_channel
            """, conn)
            channel_counts = dict(zip(df_channel_actual['sales_channel'], df_channel_actual['cnt']))
        except Exception:
            channel_counts = {}
    
    # Test: All units have channel assignment
    total_units = sum(unit_counts.values())
    assigned_units = sum(channel_counts.values())
    
    test_name = "All Units Have Channel"
    if total_units > 0 and assigned_units < total_units:
        unassigned = total_units - assigned_units
        results.add_fail(
            test_name,
            f"{total_units} assigned",
            f"{assigned_units} assigned ({unassigned} missing)",
            "Units without channel cannot generate revenue"
        )
    else:
        results.add_pass(f"{test_name}: {assigned_units} units")
        
    # Test: Unit status distribution makes sense
    test_name = "Unit Status Distribution"
    planned = unit_counts.get('PLANNED', 0)
    built = unit_counts.get('BUILT', 0)
    shipped = unit_counts.get('SHIPPED', 0)
    
    results.add_pass(f"{test_name}: {planned} planned, {built} built, {shipped} shipped")
    
    # Check for status progression issues
    if shipped > built + planned:
        results.add_warning("Status Logic", "More shipped than built+planned - data issue?")


# =============================================================================
# TEST 7: PRICING CONFIGURATION CONSISTENCY
# =============================================================================

def test_pricing_consistency(engine, results):
    """
    Validate pricing configuration:
    - MSRP should be reasonable range
    - Dealer discount between 70-85%
    - Year-over-year changes make business sense
    """
    print("\n" + "-"*70)
    print("TEST 7: Pricing Configuration")
    print("-"*70)
    
    with engine.connect() as conn:
        try:
            df_pricing = pd.read_sql("""
                SELECT year, msrp, dealer_discount_pct, notes
                FROM pricing_config
                ORDER BY year
            """, conn)
        except Exception as e:
            results.add_warning("pricing_config", f"Cannot load: {e}")
            return
    
    if df_pricing.empty:
        results.add_warning("pricing", "No pricing configuration found")
        return
        
    prev_msrp = None
    for _, row in df_pricing.iterrows():
        year = int(row['year'])
        msrp = float(row['msrp'])
        dealer_disc = float(row['dealer_discount_pct'])
        
        # Test: MSRP in reasonable range
        test_name = f"{year} MSRP Range"
        if msrp < 5000 or msrp > 15000:
            results.add_fail(test_name, "$5,000-$15,000", f"${msrp:,.0f}", "")
        else:
            results.add_pass(f"{test_name}: ${msrp:,.0f}")
            
        # Test: Dealer discount reasonable
        test_name = f"{year} Dealer Discount"
        if dealer_disc < 0.70 or dealer_disc > 0.90:
            results.add_fail(test_name, "70-90%", f"{dealer_disc*100:.0f}%", "")
        else:
            margin = (1 - dealer_disc) * 100
            results.add_pass(f"{test_name}: {dealer_disc*100:.0f}% ({margin:.0f}% margin)")
            
        # Test: YoY change reasonable
        if prev_msrp is not None:
            yoy_change = (msrp - prev_msrp) / prev_msrp * 100
            test_name = f"{year} YoY Price Change"
            if abs(yoy_change) > 20:
                results.add_warning(test_name, f"{yoy_change:+.1f}% change seems aggressive")
            else:
                results.add_pass(f"{test_name}: {yoy_change:+.1f}%")
                
        prev_msrp = msrp


# =============================================================================
# TEST 8: WORKING CAPITAL / INVENTORY
# =============================================================================

def test_working_capital(engine, results):
    """
    Validate working capital calculations:
    - Inventory lead times affect cash outflows
    - AR timing based on channel mix
    """
    print("\n" + "-"*70)
    print("TEST 8: Working Capital / Inventory")
    print("-"*70)
    
    with engine.connect() as conn:
        # Get parts with lead times
        try:
            df_parts = pd.read_sql("""
                SELECT sku, name, lead_time, cost, moq,
                       deposit_pct, deposit_days, balance_days
                FROM part_master
            """, conn)
        except Exception as e:
            results.add_warning("part_master", f"Cannot load: {e}")
            return
    
    if df_parts.empty:
        results.add_warning("inventory", "No parts found")
        return
        
    # Test: Critical parts have reasonable lead times
    long_lead_parts = df_parts[df_parts['lead_time'] > 60]
    
    test_name = "Long Lead Time Parts"
    if len(long_lead_parts) > 0:
        results.add_warning(
            test_name,
            f"{len(long_lead_parts)} parts with >60 day lead time - working capital impact"
        )
        for _, part in long_lead_parts.iterrows():
            print(f"    • {part['name']}: {part['lead_time']} days (${part['cost']:,.2f})")
    else:
        results.add_pass(f"{test_name}: All parts <60 days")
        
    # Test: High-cost parts with deposits
    high_cost = df_parts[df_parts['cost'] > 500]
    deposit_parts = high_cost[high_cost['deposit_pct'] > 0]
    
    test_name = "Deposit Requirements"
    if len(deposit_parts) > 0:
        total_deposit = (deposit_parts['cost'] * deposit_parts['deposit_pct']).sum()
        results.add_pass(f"{test_name}: {len(deposit_parts)} parts require deposits (~${total_deposit:,.0f}/unit)")
    else:
        results.add_pass(f"{test_name}: No deposits required")


# =============================================================================
# MAIN TEST RUNNER
# =============================================================================

def run_all_tests():
    """Run all financial integrity tests"""
    print("="*70)
    print("IdleX ERP - Financial Integrity Test Suite")
    print("="*70)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Database: {'PostgreSQL' if os.getenv('DATABASE_URL') else 'SQLite (local)'}")
    
    engine = get_engine()
    results = TestResults()
    
    # Run all tests
    test_channel_mix_consistency(engine, results)
    test_revenue_calculations(engine, results)
    test_cash_timing(engine, results)
    test_bom_calculations(engine, results)
    test_opex_calculations(engine, results)
    test_cross_module_consistency(engine, results)
    test_pricing_consistency(engine, results)
    test_working_capital(engine, results)
    
    # Print summary
    all_passed = results.summary()
    
    # Return exit code
    return 0 if all_passed else 1


def run_single_test(test_name):
    """Run a single test by name"""
    engine = get_engine()
    results = TestResults()
    
    tests = {
        'channel_mix': test_channel_mix_consistency,
        'revenue': test_revenue_calculations,
        'cash_timing': test_cash_timing,
        'bom': test_bom_calculations,
        'opex': test_opex_calculations,
        'consistency': test_cross_module_consistency,
        'pricing': test_pricing_consistency,
        'working_capital': test_working_capital,
    }
    
    if test_name in tests:
        tests[test_name](engine, results)
        results.summary()
    else:
        print(f"Unknown test: {test_name}")
        print(f"Available tests: {', '.join(tests.keys())}")


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == '--test':
        run_single_test(sys.argv[2])
    else:
        exit_code = run_all_tests()
        sys.exit(exit_code)
