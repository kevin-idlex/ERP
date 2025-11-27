#!/usr/bin/env python
"""
IdleX ERP - Comprehensive Test Runner
=====================================

Run all financial integrity tests and generate CFO-ready summary.

USAGE:
    # Run all tests
    python run_tests.py
    
    # Run specific test module
    python run_tests.py --module seed_schema
    
    # Run with verbose output
    python run_tests.py -v
    
    # Run with Postgres (requires TEST_DATABASE_URL)
    TEST_DATABASE_URL=postgresql://... python run_tests.py --postgres
    
    # Generate HTML report
    python run_tests.py --html report.html

PREREQUISITES:
    pip install pytest pytest-html pytest-xdist
"""

import subprocess
import sys
import os
import argparse
from datetime import datetime
from pathlib import Path


def print_header(title):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def run_pytest(args, extra_args=None):
    """Run pytest with given arguments."""
    cmd = [sys.executable, "-m", "pytest"]
    cmd.extend(args)
    if extra_args:
        cmd.extend(extra_args)
    
    result = subprocess.run(cmd, cwd=Path(__file__).parent)
    return result.returncode


def run_quick_sanity_check():
    """Run quick sanity check without full pytest."""
    print_header("Quick Sanity Check")
    
    try:
        # Try to import key modules
        sys.path.insert(0, str(Path(__file__).parent.parent))
        
        from seed_db import get_db_engine, run_seed
        from dashboard import (
            money, pct, parse_date, get_workdays,
            calculate_unit_material_cost, generate_financial_ledgers
        )
        
        print("✅ All imports successful")
        
        # Quick DB check
        engine = get_db_engine()
        from sqlalchemy import text
        with engine.connect() as conn:
            unit_count = conn.execute(text(
                "SELECT COUNT(*) FROM production_unit"
            )).scalar()
            print(f"✅ Database accessible: {unit_count:,} production units")
        
        # Quick calculation check
        material_cost = calculate_unit_material_cost(engine)
        print(f"✅ Material cost calculation: ${material_cost:,.2f}/unit")
        
        return 0
    except Exception as e:
        print(f"❌ Sanity check failed: {e}")
        return 1


def generate_cfo_summary(engine):
    """Generate CFO-ready summary of key metrics."""
    print_header("CFO Dashboard Summary")
    
    from sqlalchemy import text
    import pandas as pd
    
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from dashboard import (
        calculate_unit_material_cost, 
        generate_financial_ledgers,
        run_cash_waterfall
    )
    
    with engine.connect() as conn:
        # Unit counts
        units_by_year = conn.execute(text("""
            SELECT substr(build_date, 1, 4) as year, COUNT(*) as cnt
            FROM production_unit
            GROUP BY year ORDER BY year
        """)).fetchall()
        
        # Channel distribution
        channel_dist = conn.execute(text("""
            SELECT sales_channel, COUNT(*) as cnt
            FROM production_unit
            GROUP BY sales_channel
        """)).fetchall()
        
    total_units = sum(r[1] for r in units_by_year)
    channel_pcts = {r[0]: r[1]/total_units*100 for r in channel_dist}
    
    print("\n📊 PRODUCTION SUMMARY")
    print("-" * 40)
    for year, count in units_by_year:
        print(f"  {year}: {count:>8,} units")
    print(f"  {'Total':8}: {total_units:>8,} units")
    
    print("\n📈 CHANNEL MIX (Actual)")
    print("-" * 40)
    for channel, pct_val in channel_pcts.items():
        print(f"  {channel}: {pct_val:>5.1f}%")
    
    # Generate financials
    df_pnl, df_cash = generate_financial_ledgers(engine)
    df_pnl['Date'] = pd.to_datetime(df_pnl['Date'])
    
    print("\n💰 REVENUE BY YEAR")
    print("-" * 40)
    revenue = df_pnl[df_pnl['Type'] == 'Revenue'].groupby(
        df_pnl['Date'].dt.year
    )['Amount'].sum()
    for year, amount in revenue.items():
        print(f"  {year}: ${amount:>15,.0f}")
    
    # Material cost
    material_cost = calculate_unit_material_cost(engine)
    print("\n🔧 UNIT ECONOMICS")
    print("-" * 40)
    print(f"  Material Cost/Unit:  ${material_cost:>10,.2f}")
    print(f"  Direct Price:        ${8500:>10,.2f}")
    print(f"  Dealer Price:        ${6375:>10,.2f}")
    print(f"  Direct Gross Margin: {(8500-material_cost)/8500*100:>9.1f}%")
    print(f"  Dealer Gross Margin: {(6375-material_cost)/6375*100:>9.1f}%")
    
    # Cash waterfall
    if not df_cash.empty:
        waterfall = run_cash_waterfall(df_cash, 1600000, 500000)
        if not waterfall.empty:
            min_cash = waterfall['Net_Cash'].min()
            max_loc = waterfall['LOC_Usage'].max()
            end_cash = waterfall.iloc[-1]['Net_Cash']
            
            print("\n💵 CASH POSITION")
            print("-" * 40)
            print(f"  Starting Equity:     ${1600000:>12,.0f}")
            print(f"  Minimum Cash:        ${min_cash:>12,.0f}")
            print(f"  Peak LOC Usage:      ${max_loc:>12,.0f}")
            print(f"  Ending Cash:         ${end_cash:>12,.0f}")
            
            if min_cash < 0:
                print(f"\n  ⚠️  CASH CRUNCH: Need additional ${abs(min_cash):,.0f}")
    
    print("\n" + "=" * 70)
    return 0


def main():
    parser = argparse.ArgumentParser(description="IdleX ERP Test Runner")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--module", type=str, help="Run specific test module")
    parser.add_argument("--html", type=str, help="Generate HTML report")
    parser.add_argument("--postgres", action="store_true", help="Run against Postgres")
    parser.add_argument("--quick", action="store_true", help="Quick sanity check only")
    parser.add_argument("--summary", action="store_true", help="Generate CFO summary")
    parser.add_argument("-x", "--exitfirst", action="store_true", help="Exit on first failure")
    
    args = parser.parse_args()
    
    print_header("IdleX ERP - Financial Integrity Test Suite")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Python: {sys.version.split()[0]}")
    
    # Quick sanity check
    if args.quick:
        return run_quick_sanity_check()
    
    # CFO summary
    if args.summary:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from seed_db import get_db_engine
        return generate_cfo_summary(get_db_engine())
    
    # Build pytest arguments
    pytest_args = []
    
    if args.verbose:
        pytest_args.append("-v")
    
    if args.exitfirst:
        pytest_args.append("-x")
    
    if args.html:
        pytest_args.extend(["--html", args.html, "--self-contained-html"])
    
    if args.module:
        module_map = {
            "seed": "test_seed_schema.py",
            "seed_schema": "test_seed_schema.py",
            "financial": "test_financial_engine.py",
            "engine": "test_financial_engine.py",
            "cash": "test_cash_waterfall.py",
            "waterfall": "test_cash_waterfall.py",
            "production": "test_production.py",
            "channel": "test_production.py",
            "regression": "test_regression.py",
        }
        module_file = module_map.get(args.module, f"test_{args.module}.py")
        pytest_args.append(module_file)
    
    # Run tests
    print_header("Running Tests")
    exit_code = run_pytest(pytest_args)
    
    # Print summary
    if exit_code == 0:
        print_header("✅ ALL TESTS PASSED")
        print("\nFinancial calculations verified. Ready for CFO presentation.")
    else:
        print_header("❌ TESTS FAILED")
        print("\nReview failures before presenting to board.")
    
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
