"""
IdleX ERP - Database Seeder
Version: 9.0 (Production Ready)
"""

import calendar
from datetime import date
from sqlalchemy import create_engine, text
import math
import os

def get_db_engine():
    db_url = os.getenv("DATABASE_URL")
    if db_url:
        if db_url.startswith("postgres://"):
            db_url = db_url.replace("postgres://", "postgresql://", 1)
        return create_engine(db_url)
    return create_engine('sqlite:///idlex.db')

engine = get_db_engine()

def get_workdays(year, month):
    num_days = calendar.monthrange(year, month)[1]
    days = [date(year, month, d) for d in range(1, num_days + 1)]
    return [d for d in days if d.weekday() < 5]

def run_seed():
    print("--- STARTING DATABASE BUILD (V9) ---")
    
    with engine.connect() as conn:
        # DROP ALL TABLES
        for t in ["opex_general_expenses", "opex_staffing_plan", "opex_roles",
                  "production_unit", "bom_items", "part_master", 
                  "pricing_config", "channel_mix_config", "global_config"]:
            conn.execute(text(f"DROP TABLE IF EXISTS {t}"))
        
        # CREATE TABLES
        conn.execute(text("CREATE TABLE global_config (setting_key TEXT PRIMARY KEY, setting_value TEXT)"))
        
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
                id INTEGER PRIMARY KEY, sku TEXT, name TEXT, cost REAL,
                moq INTEGER, lead_time INTEGER, deposit_pct REAL,
                deposit_days INTEGER, balance_days INTEGER
            )
        """))
        
        conn.execute(text("CREATE TABLE bom_items (id INTEGER PRIMARY KEY, part_id INTEGER, qty_per_unit REAL)"))
        
        conn.execute(text("""
            CREATE TABLE production_unit (
                id INTEGER PRIMARY KEY, serial_number TEXT, build_date DATE,
                sales_channel TEXT, status TEXT
            )
        """))
        
        conn.execute(text("CREATE TABLE opex_roles (id INTEGER PRIMARY KEY, role_name TEXT, annual_salary REAL)"))
        conn.execute(text("CREATE TABLE opex_staffing_plan (id INTEGER PRIMARY KEY, role_id INTEGER, month_date DATE, headcount REAL)"))
        conn.execute(text("""
            CREATE TABLE opex_general_expenses (
                id INTEGER PRIMARY KEY, category TEXT, expense_type TEXT, month_date DATE, amount REAL
            )
        """))

        # GLOBAL CONFIG
        conn.execute(text("INSERT INTO global_config VALUES ('start_cash', '1500000')"))
        conn.execute(text("INSERT INTO global_config VALUES ('loc_limit', '4100000')"))

        # PRICING CONFIG
        for year, msrp, notes in [(2026, 15500, "Launch"), (2027, 13500, "Scale"), (2028, 11500, "Volume")]:
            conn.execute(text("INSERT INTO pricing_config (year, msrp, dealer_discount_pct, notes) VALUES (:y, :m, 0.80, :n)"),
                        {"y": year, "m": msrp, "n": notes})

        # CHANNEL MIX CONFIG
        channel_mix = [
            (2026, 1, 0.15), (2026, 2, 0.18), (2026, 3, 0.20), (2026, 4, 0.22),
            (2027, 1, 0.25), (2027, 2, 0.28), (2027, 3, 0.30), (2027, 4, 0.32),
            (2028, 1, 0.35), (2028, 2, 0.38), (2028, 3, 0.42), (2028, 4, 0.45),
        ]
        for y, q, p in channel_mix:
            conn.execute(text("INSERT INTO channel_mix_config (year, quarter, direct_pct) VALUES (:y, :q, :p)"),
                        {"y": y, "q": q, "p": p})

        # PARTS
        parts = [
            ('BAT-SS-48V', 'Safiery Solid State Battery 48V', 715.26, 10, 103, 0.50, -103, -45),
            ('DCDC-SCOTTY', 'Scotty AI DC-DC Converter', 775.22, 10, 103, 0.50, -103, -45),
            ('SCRN-STARAI', 'STAR AI Nexus GPS Display', 230.00, 20, 103, 0.50, -103, -45),
            ('CBL-RS485', 'RS485 Battery InterConnect Cables', 15.00, 50, 103, 0.50, -103, -45),
            ('CBL-VECAN', 'Scotty Comms VE.Can Cable', 27.00, 20, 103, 0.50, -103, -45),
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
            """), {"sku": p[0], "name": p[1], "cost": p[2], "moq": p[3], "lead": p[4], "dp": p[5], "dd": p[6], "bd": p[7]})

        # BOM
        bom_map = [
            ('BAT-SS-48V', 3), ('DCDC-SCOTTY', 1), ('SCRN-STARAI', 1), ('CBL-RS485', 2), ('CBL-VECAN', 1),
            ('BPR-002', 1), ('ENC-004', 1), ('CAP-006', 1), ('CBL-002', 1), ('FUS-001', 1), ('GRM-001', 1),
            ('SHR-006-R', 4), ('SHR-006-B', 4), ('SHR-2/0-R', 3), ('SHR-2/0-B', 2), ('SEN-001', 1),
            ('TRM-20-516', 12), ('TRM-06-516', 10), ('TRM-20-38', 4), ('TRM-06-14', 5), ('TRM-SM', 1), ('TRM-08-316', 2),
            ('WIR-01-R', 12), ('WIR-01-B', 8), ('WIR-10-3', 1), ('WIR-14-3', 7),
            ('WIR-20-R', 12), ('WIR-20-B', 8), ('WIR-06-R', 15), ('WIR-06-B', 15), ('WIR-08-B', 5), ('WIR-08-R', 2.5),
        ]
        for sku, qty in bom_map:
            pid = conn.execute(text("SELECT id FROM part_master WHERE sku=:sku"), {"sku": sku}).scalar()
            if pid:
                conn.execute(text("INSERT INTO bom_items (part_id, qty_per_unit) VALUES (:pid, :qty)"), {"pid": pid, "qty": qty})

        # PRODUCTION SCHEDULE
        targets_2026 = [(1,30),(2,40),(3,53),(4,58),(5,74),(6,125),(7,136),(8,221),(9,230),(10,288),(11,460),(12,472)]
        targets_2027 = [(m, math.ceil(472 * (1.02 ** m))) for m in range(1, 13)]
        dec_2027 = math.ceil(472 * (1.02 ** 12))
        targets_2028 = [(m, math.ceil(dec_2027 * (1.02 ** m))) for m in range(1, 13)]

        def get_quarter(month): return (month - 1) // 3 + 1
        
        def get_direct_pct(year, quarter):
            result = conn.execute(text("SELECT direct_pct FROM channel_mix_config WHERE year=:y AND quarter=:q"),
                                 {"y": year, "q": quarter}).scalar()
            return result if result else 0.25

        serial_counter = 1
        for year, targets in [(2026, targets_2026), (2027, targets_2027), (2028, targets_2028)]:
            for month_num, qty in targets:
                workdays = get_workdays(year, month_num)
                direct_pct = get_direct_pct(year, get_quarter(month_num))
                direct_qty = math.floor(qty * direct_pct)
                dealer_qty = qty - direct_qty
                unit_pool = ['DIRECT'] * direct_qty + ['DEALER'] * dealer_qty
                day_idx = 0
                for unit_type in unit_pool:
                    conn.execute(text("INSERT INTO production_unit (serial_number, build_date, sales_channel, status) VALUES (:sn, :bd, :ch, 'PLANNED')"),
                                {"sn": f"IDX-{serial_counter:05d}", "bd": workdays[day_idx], "ch": unit_type})
                    serial_counter += 1
                    day_idx = (day_idx + 1) % len(workdays)

        # STAFFING
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
            conn.execute(text("INSERT INTO opex_roles (role_name, annual_salary) VALUES (:n, :s)"), {"n": role_name, "s": salary})
            rid = conn.execute(text("SELECT id FROM opex_roles WHERE role_name=:n"), {"n": role_name}).scalar()
            for idx, count in enumerate(counts):
                m_date = date(2026 + (idx // 12), (idx % 12) + 1, 1)
                conn.execute(text("INSERT INTO opex_staffing_plan (role_id, month_date, headcount) VALUES (:rid, :dt, :hc)"),
                            {"rid": rid, "dt": m_date, "hc": count})

        # GENERAL EXPENSES
        expenses = [
            ("Core system", "R&D", [50000,40000,30000,5000,5000,5000,5000,0,0,0,0,0]),
            ("HVAC prototype", "R&D", [50000,50000,5000,5000,5000,5000,5000,0,0,0,0,0]),
            ("Enclosures", "R&D", [5000]*7 + [0]*5),
            ("Software", "R&D", [5000]*7 + [0]*5),
            ("Test equipment", "R&D", [5000]*7 + [0]*5),
            ("Outsourced eng", "R&D", [5000]*7 + [0]*5),
            ("Certification", "R&D", [5000]*7 + [0]*5),
            ("Office/cloud/insurance", "SG&A", [10000]*12),
            ("Travel", "SG&A", [10000]*12),
            ("Legal/accounting", "SG&A", [15000,5000,1500,2500,2500,2500,2500,2500,2500,2500,2500,2500]),
        ]
        for category, exp_type, amounts in expenses:
            extended = amounts + [amounts[-1]] * 24
            for idx, amount in enumerate(extended):
                if amount > 0:
                    m_date = date(2026 + (idx // 12), (idx % 12) + 1, 1)
                    conn.execute(text("INSERT INTO opex_general_expenses (category, expense_type, month_date, amount) VALUES (:c, :t, :d, :a)"),
                                {"c": category, "t": exp_type, "d": m_date, "a": amount})

        conn.commit()
        unit_count = conn.execute(text("SELECT COUNT(*) FROM production_unit")).scalar()
        print(f"--- SUCCESS ---")
        print(f"    Units: {unit_count:,}")
        print(f"    Equity: $1,500,000 | LOC: $4,100,000")
        print(f"    Pricing: $15,500 / $13,500 / $11,500 (80% dealer)")

if __name__ == "__main__":
    run_seed()
