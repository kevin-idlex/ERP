#!/usr/bin/env python3
"""
IdleX ERP - Complete File Updater
Run this script to update all ERP files with the latest fixes.

Usage:
    python update_all_files.py

This will create/overwrite:
    - seed_db.py (with pricing_config, channel_mix_config, correct values)
    - tests/test_regression.py (with fixed money formatting test)
    - tests/test_cash_waterfall.py (with fixed empty dataframe test)
"""

import os

def write_file(path, content):
    """Write content to file, creating directories if needed."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"  ✓ Written: {path}")

# =============================================================================
# SEED_DB.PY - Complete with all tables and correct values
# =============================================================================
SEED_DB_PY = '''"""
IdleX ERP - Database Seeder
Version: 9.0 (Production Ready)

Creates and populates all database tables with:
- Global config ($1.5M equity, $4.1M LOC)
- Part master with BOM
- Production schedule (2026-2028)
- Pricing config ($15,500 → $13,500 → $11,500)
- Channel mix config (15% → 45% direct)
- OpEx roles and staffing plan
- General expenses
"""

import calendar
from datetime import date
from sqlalchemy import create_engine, text
import math
import os

def get_db_engine():
    """Get database engine - PostgreSQL in cloud, SQLite locally."""
    db_url = os.getenv("DATABASE_URL")
    if db_url:
        if db_url.startswith("postgres://"):
            db_url = db_url.replace("postgres://", "postgresql://", 1)
        return create_engine(db_url)
    return create_engine('sqlite:///idlex.db')

engine = get_db_engine()

def get_workdays(year, month):
    """Finds all Mon-Fri dates in a specific month."""
    num_days = calendar.monthrange(year, month)[1]
    days = [date(year, month, d) for d in range(1, num_days + 1)]
    return [d for d in days if d.weekday() < 5]

def run_seed():
    print("--- STARTING DATABASE BUILD (V9: COMPLETE) ---")
    
    with engine.connect() as conn:
        # =================================================================
        # A. DROP ALL TABLES (order matters for foreign keys)
        # =================================================================
        tables_to_drop = [
            "opex_general_expenses",
            "opex_staffing_plan", 
            "opex_roles",
            "production_unit",
            "bom_items",
            "part_master",
            "pricing_config",
            "channel_mix_config",
            "global_config"
        ]
        for table in tables_to_drop:
            conn.execute(text(f"DROP TABLE IF EXISTS {table}"))
        
        # =================================================================
        # B. CREATE ALL TABLES
        # =================================================================
        conn.execute(text("""
            CREATE TABLE global_config (
                setting_key TEXT PRIMARY KEY, 
                setting_value TEXT
            )
        """))
        
        conn.execute(text("""
            CREATE TABLE pricing_config (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                year INTEGER UNIQUE NOT NULL,
                msrp REAL NOT NULL,
                dealer_discount_pct REAL NOT NULL,
                notes TEXT
            )
        """))
        
        conn.execute(text("""
            CREATE TABLE channel_mix_config (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                year INTEGER NOT NULL,
                quarter INTEGER NOT NULL,
                direct_pct REAL NOT NULL,
                UNIQUE(year, quarter)
            )
        """))
        
        conn.execute(text("""
            CREATE TABLE part_master (
                id INTEGER PRIMARY KEY, 
                sku TEXT, 
                name TEXT, 
                cost REAL,
                moq INTEGER, 
                lead_time INTEGER, 
                deposit_pct REAL,
                deposit_days INTEGER, 
                balance_days INTEGER
            )
        """))
        
        conn.execute(text("""
            CREATE TABLE bom_items (
                id INTEGER PRIMARY KEY, 
                part_id INTEGER, 
                qty_per_unit REAL
            )
        """))
        
        conn.execute(text("""
            CREATE TABLE production_unit (
                id INTEGER PRIMARY KEY, 
                serial_number TEXT, 
                build_date DATE,
                sales_channel TEXT, 
                status TEXT
            )
        """))
        
        conn.execute(text("""
            CREATE TABLE opex_roles (
                id INTEGER PRIMARY KEY, 
                role_name TEXT, 
                annual_salary REAL
            )
        """))
        
        conn.execute(text("""
            CREATE TABLE opex_staffing_plan (
                id INTEGER PRIMARY KEY, 
                role_id INTEGER, 
                month_date DATE, 
                headcount REAL
            )
        """))
        
        conn.execute(text("""
            CREATE TABLE opex_general_expenses (
                id INTEGER PRIMARY KEY,
                category TEXT,
                expense_type TEXT,
                month_date DATE,
                amount REAL
            )
        """))

        # =================================================================
        # C. SEED GLOBAL CONFIG
        # =================================================================
        conn.execute(text("INSERT INTO global_config VALUES ('start_cash', '1500000')"))  # $1.5M equity
        conn.execute(text("INSERT INTO global_config VALUES ('loc_limit', '4100000')"))   # $4.1M LOC

        # =================================================================
        # D. SEED PRICING CONFIG
        # =================================================================
        # MSRP decreases as volume increases, dealer always pays 80%
        pricing_data = [
            (2026, 15500, 0.80, "Launch year - premium pricing"),
            (2027, 13500, 0.80, "Scale year - volume discounts"),
            (2028, 11500, 0.80, "Volume year - market penetration"),
        ]
        for year, msrp, discount, notes in pricing_data:
            conn.execute(text("""
                INSERT INTO pricing_config (year, msrp, dealer_discount_pct, notes)
                VALUES (:y, :m, :d, :n)
            """), {"y": year, "m": msrp, "d": discount, "n": notes})

        # =================================================================
        # E. SEED CHANNEL MIX CONFIG
        # =================================================================
        # Direct % increases from 15% to 45% over time
        channel_mix = [
            (2026, 1, 0.15), (2026, 2, 0.18), (2026, 3, 0.20), (2026, 4, 0.22),
            (2027, 1, 0.25), (2027, 2, 0.28), (2027, 3, 0.30), (2027, 4, 0.32),
            (2028, 1, 0.35), (2028, 2, 0.38), (2028, 3, 0.42), (2028, 4, 0.45),
        ]
        for year, quarter, direct_pct in channel_mix:
            conn.execute(text("""
                INSERT INTO channel_mix_config (year, quarter, direct_pct)
                VALUES (:y, :q, :p)
            """), {"y": year, "q": quarter, "p": direct_pct})

        # =================================================================
        # F. SEED PART MASTER
        # =================================================================
        parts = [
            # Safiery parts - 103 day lead time, 50% deposit
            ('BAT-SS-48V', 'Safiery Solid State Battery 48V', 715.26, 10, 103, 0.50, -103, -45),
            ('DCDC-SCOTTY', 'Scotty AI DC-DC Converter', 775.22, 10, 103, 0.50, -103, -45),
            ('SCRN-STARAI', 'STAR AI Nexus GPS Display', 230.00, 20, 103, 0.50, -103, -45),
            ('CBL-RS485', 'RS485 Battery InterConnect Cables', 15.00, 50, 103, 0.50, -103, -45),
            ('CBL-VECAN', 'Scotty Comms VE.Can Cable', 27.00, 20, 103, 0.50, -103, -45),
            
            # Other parts - standard terms
            ('BPR-002', 'Battery Protect', 180.00, 10, 7, 0.00, 0, 0),
            ('ENC-004', 'Aluminum Enclosure', 300.00, 50, 14, 0.00, 0, -7),
            ('CAP-006', 'SkelStart Ultracap', 1195.00, 5, 14, 0.00, 0, 0),
            ('CBL-002', 'RJ45 Ethernet 0.9m', 7.80, 50, 7, 0.00, 0, 30),
            ('FUS-001', 'Fuse Scotty DC DC 100A', 1.70, 100, 7, 0.00, 0, 30),
            ('GRM-001', 'Grommet 1 1/4', 5.00, 50, 7, 0.00, 0, 30),
            ('SHR-006-R', 'Heat Shrink #6 Red', 0.22, 100, 7, 0.00, 0, 30),
            ('SHR-006-B', 'Heat Shrink #6 Black', 0.22, 100, 7, 0.00, 0, 30),
            ('SHR-2/0-R', 'Heat Shrink 2/0 Red', 2.42, 50, 7, 0.00, 0, 30),
            ('SHR-2/0-B', 'Heat Shrink 2/0 Black', 2.42, 50, 7, 0.00, 0, 30),
            ('SEN-001', 'Sensor Battery Temp', 17.40, 20, 7, 0.00, 0, 30),
            ('TRM-20-516', 'Terminals 2/0 x 5/16', 1.13, 100, 7, 0.00, 0, 30),
            ('TRM-06-516', 'Terminals 6 x 5/16', 0.48, 100, 7, 0.00, 0, 30),
            ('TRM-20-38', 'Terminals 2/0 x 3/8', 1.13, 100, 7, 0.00, 0, 30),
            ('TRM-06-14', 'Terminals 6 x 1/4', 0.48, 100, 7, 0.00, 0, 30),
            ('TRM-SM', 'Terminals Small', 2.00, 100, 7, 0.00, 0, 30),
            ('TRM-08-316', 'Terminals 8 x 3/16', 0.44, 100, 7, 0.00, 0, 30),
            ('WIR-01-R', 'Wire 1 AWG Red', 3.64, 500, 7, 0.00, 0, 30),
            ('WIR-01-B', 'Wire 1 AWG Black', 3.64, 500, 7, 0.00, 0, 30),
            ('WIR-10-3', 'Wire 10/3 Triplex', 1.18, 100, 7, 0.00, 0, 30),
            ('WIR-14-3', 'Wire 14/3 Triplex', 0.60, 100, 7, 0.00, 0, 30),
            ('WIR-20-R', 'Wire 2/0 Red', 4.08, 500, 7, 0.00, 0, 30),
            ('WIR-20-B', 'Wire 2/0 Black', 4.08, 500, 7, 0.00, 0, 30),
            ('WIR-06-R', 'Wire 6 AWG Red', 0.99, 500, 7, 0.00, 0, 30),
            ('WIR-06-B', 'Wire 6 AWG Black', 0.99, 500, 7, 0.00, 0, 30),
            ('WIR-08-B', 'Wire 8 AWG Black', 0.61, 500, 7, 0.00, 0, 30),
            ('WIR-08-R', 'Wire 8 AWG Red', 0.61, 500, 7, 0.00, 0, 30),
        ]
        
        for p in parts:
            conn.execute(text("""
                INSERT INTO part_master (sku, name, cost, moq, lead_time, deposit_pct, deposit_days, balance_days)
                VALUES (:sku, :name, :cost, :moq, :lead, :dp, :dd, :bd)
            """), {"sku": p[0], "name": p[1], "cost": p[2], "moq": p[3], 
                   "lead": p[4], "dp": p[5], "dd": p[6], "bd": p[7]})

        # =================================================================
        # G. SEED BOM ITEMS
        # =================================================================
        bom_map = [
            ('BAT-SS-48V', 3),      # 3 batteries per unit
            ('DCDC-SCOTTY', 1),
            ('SCRN-STARAI', 1),
            ('CBL-RS485', 2),
            ('CBL-VECAN', 1),
            ('BPR-002', 1),
            ('ENC-004', 1),
            ('CAP-006', 1),
            ('CBL-002', 1),
            ('FUS-001', 1),
            ('GRM-001', 1),
            ('SHR-006-R', 4),
            ('SHR-006-B', 4),
            ('SHR-2/0-R', 3),
            ('SHR-2/0-B', 2),
            ('SEN-001', 1),
            ('TRM-20-516', 12),
            ('TRM-06-516', 10),
            ('TRM-20-38', 4),
            ('TRM-06-14', 5),
            ('TRM-SM', 1),
            ('TRM-08-316', 2),
            ('WIR-01-R', 12),
            ('WIR-01-B', 8),
            ('WIR-10-3', 1),
            ('WIR-14-3', 7),
            ('WIR-20-R', 12),
            ('WIR-20-B', 8),
            ('WIR-06-R', 15),
            ('WIR-06-B', 15),
            ('WIR-08-B', 5),
            ('WIR-08-R', 2.5),
        ]
        
        for sku, qty in bom_map:
            pid = conn.execute(text("SELECT id FROM part_master WHERE sku=:sku"), {"sku": sku}).scalar()
            if pid:
                conn.execute(text("INSERT INTO bom_items (part_id, qty_per_unit) VALUES (:pid, :qty)"),
                            {"pid": pid, "qty": qty})

        # =================================================================
        # H. SEED PRODUCTION SCHEDULE
        # =================================================================
        # 2026 monthly targets (hand-tuned ramp)
        targets_2026 = [
            (1, 30), (2, 40), (3, 53), (4, 58), (5, 74), (6, 125),
            (7, 136), (8, 221), (9, 230), (10, 288), (11, 460), (12, 472)
        ]  # Total: 2,187
        
        # 2027: 2% monthly growth from Dec 2026 base (472)
        targets_2027 = [(m, math.ceil(472 * (1.02 ** m))) for m in range(1, 13)]
        # Total: ~12,322
        
        # 2028: Continue 2% growth
        dec_2027 = math.ceil(472 * (1.02 ** 12))
        targets_2028 = [(m, math.ceil(dec_2027 * (1.02 ** m))) for m in range(1, 13)]
        # Total: ~31,351

        def get_quarter(month):
            return (month - 1) // 3 + 1

        def get_direct_pct(year, quarter):
            """Get direct percentage from channel_mix_config."""
            result = conn.execute(text("""
                SELECT direct_pct FROM channel_mix_config 
                WHERE year = :y AND quarter = :q
            """), {"y": year, "q": quarter}).scalar()
            return result if result else 0.25  # fallback

        serial_counter = 1
        
        # Generate 2026 units
        for month_num, qty in targets_2026:
            workdays = get_workdays(2026, month_num)
            quarter = get_quarter(month_num)
            direct_pct = get_direct_pct(2026, quarter)
            direct_qty = math.floor(qty * direct_pct)
            dealer_qty = qty - direct_qty
            
            unit_pool = ['DIRECT'] * direct_qty + ['DEALER'] * dealer_qty
            day_idx = 0
            for unit_type in unit_pool:
                conn.execute(text("""
                    INSERT INTO production_unit (serial_number, build_date, sales_channel, status)
                    VALUES (:sn, :bd, :ch, 'PLANNED')
                """), {"sn": f"IDX-{serial_counter:05d}", "bd": workdays[day_idx], "ch": unit_type})
                serial_counter += 1
                day_idx = (day_idx + 1) % len(workdays)

        # Generate 2027 units
        for month_num, qty in targets_2027:
            workdays = get_workdays(2027, month_num)
            quarter = get_quarter(month_num)
            direct_pct = get_direct_pct(2027, quarter)
            direct_qty = math.floor(qty * direct_pct)
            dealer_qty = qty - direct_qty
            
            unit_pool = ['DIRECT'] * direct_qty + ['DEALER'] * dealer_qty
            day_idx = 0
            for unit_type in unit_pool:
                conn.execute(text("""
                    INSERT INTO production_unit (serial_number, build_date, sales_channel, status)
                    VALUES (:sn, :bd, :ch, 'PLANNED')
                """), {"sn": f"IDX-{serial_counter:05d}", "bd": workdays[day_idx], "ch": unit_type})
                serial_counter += 1
                day_idx = (day_idx + 1) % len(workdays)

        # Generate 2028 units
        for month_num, qty in targets_2028:
            workdays = get_workdays(2028, month_num)
            quarter = get_quarter(month_num)
            direct_pct = get_direct_pct(2028, quarter)
            direct_qty = math.floor(qty * direct_pct)
            dealer_qty = qty - direct_qty
            
            unit_pool = ['DIRECT'] * direct_qty + ['DEALER'] * dealer_qty
            day_idx = 0
            for unit_type in unit_pool:
                conn.execute(text("""
                    INSERT INTO production_unit (serial_number, build_date, sales_channel, status)
                    VALUES (:sn, :bd, :ch, 'PLANNED')
                """), {"sn": f"IDX-{serial_counter:05d}", "bd": workdays[day_idx], "ch": unit_type})
                serial_counter += 1
                day_idx = (day_idx + 1) % len(workdays)

        # =================================================================
        # I. SEED OPEX ROLES AND STAFFING
        # =================================================================
        # 36 months: Jan 2026 - Dec 2028
        roles = [
            ("CEO", 250000, [1]*36),
            ("CTO", 200000, [1]*36),
            ("Senior Electrical Engineer", 165000, [0,0,1,1,1,1,1,1,1,1,1,1] + [1]*24),
            ("Senior Mechanical Engineer", 155000, [1]*36),
            ("Firmware Engineer", 150000, [0,0,0,1,1,1,1,1,1,1,1,1] + [1]*24),
            ("Test Engineer", 130000, [0,0,0,0,1,1,1,1,1,1,1,1] + [1]*24),
            ("Ops / Project Manager", 95000, [1]*36),
            ("Admin (part time)", 80000, [0.5]*36),
        ]
        
        for role_name, salary, counts in roles:
            conn.execute(text("INSERT INTO opex_roles (role_name, annual_salary) VALUES (:n, :s)"),
                        {"n": role_name, "s": salary})
            rid = conn.execute(text("SELECT id FROM opex_roles WHERE role_name=:n"), {"n": role_name}).scalar()
            
            for idx, count in enumerate(counts):
                curr_year = 2026 + (idx // 12)
                curr_month = (idx % 12) + 1
                m_date = date(curr_year, curr_month, 1)
                conn.execute(text("""
                    INSERT INTO opex_staffing_plan (role_id, month_date, headcount)
                    VALUES (:rid, :dt, :hc)
                """), {"rid": rid, "dt": m_date, "hc": count})

        # =================================================================
        # J. SEED GENERAL EXPENSES
        # =================================================================
        general_expenses = [
            ("Core system", "R&D", [50000, 40000, 30000, 5000, 5000, 5000, 5000, 0, 0, 0, 0, 0]),
            ("HVAC prototype and validation", "R&D", [50000, 50000, 5000, 5000, 5000, 5000, 5000, 0, 0, 0, 0, 0]),
            ("Enclosures & Harnesses", "R&D", [5000, 5000, 5000, 5000, 5000, 5000, 5000, 0, 0, 0, 0, 0]),
            ("Software & Controls", "R&D", [5000, 5000, 5000, 5000, 5000, 5000, 5000, 0, 0, 0, 0, 0]),
            ("Test equipment and lab gear", "R&D", [5000, 5000, 5000, 5000, 5000, 5000, 5000, 0, 0, 0, 0, 0]),
            ("Outsourced engineering", "R&D", [5000, 5000, 5000, 5000, 5000, 5000, 5000, 0, 0, 0, 0, 0]),
            ("Thermal and certification testing", "R&D", [5000, 5000, 5000, 5000, 5000, 5000, 5000, 0, 0, 0, 0, 0]),
            ("Office, cloud tools, insurance", "SG&A", [10000]*12),
            ("Travel", "SG&A", [10000]*12),
            ("Legal and accounting", "SG&A", [15000, 5000, 1500, 2500, 2500, 2500, 2500, 2500, 2500, 2500, 2500, 2500]),
        ]

        for category, exp_type, amounts in general_expenses:
            # Extend to 36 months (repeat last value)
            extended = amounts + [amounts[-1]] * 24
            for idx, amount in enumerate(extended):
                if amount > 0:
                    curr_year = 2026 + (idx // 12)
                    curr_month = (idx % 12) + 1
                    m_date = date(curr_year, curr_month, 1)
                    conn.execute(text("""
                        INSERT INTO opex_general_expenses (category, expense_type, month_date, amount)
                        VALUES (:c, :t, :d, :a)
                    """), {"c": category, "t": exp_type, "d": m_date, "a": amount})

        conn.commit()
        
        # Print summary
        unit_count = conn.execute(text("SELECT COUNT(*) FROM production_unit")).scalar()
        print(f"--- SUCCESS: Database Rebuilt ---")
        print(f"    Total Units: {unit_count:,}")
        print(f"    Equity: $1,500,000 | LOC: $4,100,000")
        print(f"    Pricing: $15,500 / $13,500 / $11,500")
        print(f"    Dealer Pays: 80%")

if __name__ == "__main__":
    run_seed()
'''

# =============================================================================
# TEST FIXES
# =============================================================================
TEST_CASH_WATERFALL_FIX = '''    def test_empty_dataframe(self):
        """Empty cash dataframe should return starting equity row."""
        from dashboard import run_cash_waterfall
        
        df = pd.DataFrame(columns=['Date', 'Type', 'Category', 'Amount'])
        result = run_cash_waterfall(df, starting_equity=1000, loc_limit=500)
        
        # With no transactions, we still get a row showing starting equity
        assert len(result) == 1
        assert result.iloc[0]['Net_Cash'] == 1000
        assert result.iloc[0]['LOC_Usage'] == 0'''

TEST_REGRESSION_MONEY_FIX = '''    def test_money_formatting(self):
        """money() function should handle edge cases."""
        from dashboard import money
        
        assert money(1000) == "1,000"
        assert money(1000.49) == "1,000"  # Rounds down
        assert money(1000.50) == "1,000"  # Rounds to even (banker's rounding)
        assert money(1000.99) == "1,001"  # Rounds up
        assert money(-1000) == "(1,000)"
        assert money(None) == "-"
        assert money(float('nan')) == "-"'''

def patch_test_file(filepath, old_text, new_text):
    """Patch a specific section of a test file."""
    if not os.path.exists(filepath):
        print(f"  ⚠ Skipped (not found): {filepath}")
        return False
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    if old_text in content:
        content = content.replace(old_text, new_text)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"  ✓ Patched: {filepath}")
        return True
    elif new_text in content:
        print(f"  ✓ Already patched: {filepath}")
        return True
    else:
        print(f"  ⚠ Could not find patch target in: {filepath}")
        return False

def main():
    print("=" * 60)
    print("IdleX ERP - Complete File Updater")
    print("=" * 60)
    print()
    
    # 1. Write seed_db.py
    print("1. Writing seed_db.py...")
    write_file("seed_db.py", SEED_DB_PY)
    
    # 2. Patch test files if they exist
    print("\n2. Patching test files...")
    
    # Old patterns to find and replace
    old_empty_df = '''    def test_empty_dataframe(self):
        """Empty cash dataframe should return empty result."""
        from dashboard import run_cash_waterfall
        
        df = pd.DataFrame(columns=['Date', 'Type', 'Category', 'Amount'])
        result = run_cash_waterfall(df, starting_equity=1000, loc_limit=500)
        
        assert result.empty or len(result) == 0'''
    
    old_money_v1 = '''    def test_money_formatting(self):
        """money() function should handle edge cases."""
        from dashboard import money
        
        assert money(1000) == "1,000"
        assert money(1000.50) == "1,001"  # Rounds
        assert money(-1000) == "(1,000)"
        assert money(None) == "-"
        assert money(float('nan')) == "-"'''
    
    old_money_v2 = '''    def test_money_formatting(self):
        """money() function should handle edge cases."""
        from dashboard import money
        
        assert money(1000) == "1,000"
        assert money(1000.50) == "1,000"  # int() truncates
        assert money(1000.99) == "1,000"  # int() truncates
        assert money(-1000) == "(1,000)"
        assert money(None) == "-"
        assert money(float('nan')) == "-"'''
    
    # Try to patch test files
    patch_test_file("tests/test_cash_waterfall.py", old_empty_df, TEST_CASH_WATERFALL_FIX)
    
    # Try both versions of the money test
    if not patch_test_file("tests/test_regression.py", old_money_v1, TEST_REGRESSION_MONEY_FIX):
        patch_test_file("tests/test_regression.py", old_money_v2, TEST_REGRESSION_MONEY_FIX)
    
    print()
    print("=" * 60)
    print("UPDATE COMPLETE!")
    print("=" * 60)
    print()
    print("Next steps:")
    print("  1. Run: python seed_db.py")
    print("  2. Run: python -m pytest -v  (from tests folder)")
    print("  3. Deploy to Cloud Run")
    print("  4. Click 'Rebuild Database' in the app")
    print()

if __name__ == "__main__":
    main()
'''
