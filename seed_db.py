import calendar
from datetime import date
from sqlalchemy import create_engine, text
import math
import os

# 1. SETUP THE DATABASE CONNECTION (CLOUD READY)
def get_db_engine():
    # Look for cloud password
    db_url = os.getenv("DATABASE_URL")
    if db_url:
        # Fix for some cloud providers using 'postgres://'
        if db_url.startswith("postgres://"):
            db_url = db_url.replace("postgres://", "postgresql://", 1)
        return create_engine(db_url)
    # Fallback to local laptop file
    return create_engine('sqlite:///idlex.db')

engine = get_db_engine()

def get_workdays(year, month):
    """Finds all Mon-Fri dates in a specific month."""
    num_days = calendar.monthrange(year, month)[1]
    days = [date(year, month, d) for d in range(1, num_days + 1)]
    return [d for d in days if d.weekday() < 5] 

def run_seed():
    print("--- STARTING DATABASE BUILD (CLOUD READY V1) ---")
    
    with engine.connect() as conn:
        # A. WIPE SLATE CLEAN
        conn.execute(text("DROP TABLE IF EXISTS opex_general_expenses")) 
        conn.execute(text("DROP TABLE IF EXISTS opex_staffing_plan"))
        conn.execute(text("DROP TABLE IF EXISTS opex_roles"))
        conn.execute(text("DROP TABLE IF EXISTS production_unit"))
        conn.execute(text("DROP TABLE IF EXISTS bom_items"))
        conn.execute(text("DROP TABLE IF EXISTS part_master"))
        conn.execute(text("DROP TABLE IF EXISTS global_config"))
        
        # B. CREATE TABLES
        conn.execute(text("CREATE TABLE global_config (setting_key TEXT PRIMARY KEY, setting_value TEXT)"))
        
        conn.execute(text("""
            CREATE TABLE part_master (
                id INTEGER PRIMARY KEY, sku TEXT, name TEXT, cost REAL, 
                moq INTEGER, lead_time INTEGER, deposit_pct REAL, 
                deposit_days INTEGER, balance_days INTEGER
            )
        """))
        
        conn.execute(text("""
            CREATE TABLE production_unit (
                id INTEGER PRIMARY KEY, serial_number TEXT, build_date DATE,
                sales_channel TEXT, status TEXT
            )
        """))
        
        conn.execute(text("CREATE TABLE bom_items (id INTEGER PRIMARY KEY, part_id INTEGER, qty_per_unit REAL)"))

        # --- OPEX TABLES ---
        conn.execute(text("CREATE TABLE opex_roles (id INTEGER PRIMARY KEY, role_name TEXT, annual_salary REAL)"))
        conn.execute(text("CREATE TABLE opex_staffing_plan (id INTEGER PRIMARY KEY, role_id INTEGER, month_date DATE, headcount REAL)"))
        
        conn.execute(text("""
            CREATE TABLE opex_general_expenses (
                id INTEGER PRIMARY KEY, 
                category TEXT, 
                expense_type TEXT, 
                month_date DATE, 
                amount REAL
            )
        """))

        # C. INSERT DATA
        conn.execute(text("INSERT INTO global_config VALUES ('start_cash', '1000000')"))
        conn.execute(text("INSERT INTO global_config VALUES ('loc_limit', '5000000')"))
        
        # 2. THE BOM
        parts = [
            ('BAT-001', 'Safiery Solid State Battery', 715.26, 10, 103, 0.50, -103, -45),
            ('BPR-002', 'Battery Protect', 180.00, 10, 7, 0.00, 0, 0),
            ('DCDC-003', 'Scotty AI DC-DC', 775.22, 10, 103, 0.50, -103, -45),
            ('ENC-004', 'Aluminum Enclosure', 300.00, 50, 14, 0.00, 0, -7),
            ('SCR-005', 'STAR AI Nexus Screen', 230.00, 20, 103, 0.50, -103, -45),
            ('CAP-006', 'SkelStart Ultracap', 1195.00, 5, 14, 0.00, 0, 0),
            ('CBL-001', 'RS485 InterConnect Cables', 15.00, 50, 75, 0.00, 0, -45),
            ('CBL-002', 'RJ45 Ethernet 0.9m', 7.80, 50, 7, 0.00, 0, 30),
            ('CBL-003', 'Scotty Comms VE.Can', 27.00, 20, 103, 0.00, 0, -45),
            ('FUS-001', 'Fuse Scotty DC DC 100A', 1.70, 100, 7, 0.00, 0, 30),
            ('GRM-001', 'Grommet 1 1/4', 5.00, 50, 7, 0.00, 0, 30),
            ('SHR-006-R', 'Heat Shrink #6 Red', 0.22, 100, 7, 0.00, 0, 30),
            ('SHR-006-B', 'Heat Shrink #6 Black', 0.22, 100, 7, 0.00, 0, 30),
            ('SHR-2/0-R', 'Heat Shrink 2/0 Red', 2.42, 50, 7, 0.00, 0, 30),
            ('SHR-2/0-B', 'Heat Shrink 2/0 Black', 2.42, 50, 7, 0.00, 0, 30),
            ('SEN-001', 'Sensor Battery Temp', 17.40, 20, 7, 0.00, 0, 30),
            ('TRM-20-516', 'Terminals 2/0 x 5/16', 1.13, 100, 7, 0.00, 0, 30),
            ('TRM-06-516', 'Terminals 6 x 5/16', 0.48, 100, 7, 0.00, 0, 30),
            ('TRM-20-38',  'Terminals 2/0 x 3/8', 1.13, 100, 7, 0.00, 0, 30),
            ('TRM-06-14',  'Terminals 6 x 1/4', 0.48, 100, 7, 0.00, 0, 30),
            ('TRM-SM',     'Terminals Small', 2.00, 100, 7, 0.00, 0, 30),
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
            ('WIR-08-R', 'Wire 8 AWG Red', 0.61, 500, 7, 0.00, 0, 30)
        ]
        
        for p in parts:
            conn.execute(text("""
                INSERT INTO part_master (sku, name, cost, moq, lead_time, deposit_pct, deposit_days, balance_days)
                VALUES (:sku, :name, :cost, :moq, :lead, :dp, :dd, :bd)
            """), {"sku":p[0], "name":p[1], "cost":p[2], "moq":p[3], "lead":p[4], "dp":p[5], "dd":p[6], "bd":p[7]})
            
        # Link BOM
        bom_map = [
            ('BAT-001', 3), ('BPR-002', 1), ('DCDC-003', 1), ('ENC-004', 1), 
            ('SCR-005', 1), ('CAP-006', 1), ('CBL-001', 2), ('CBL-002', 1), 
            ('CBL-003', 1), ('FUS-001', 1), ('GRM-001', 1), 
            ('SHR-006-R', 4), ('SHR-006-B', 4), ('SHR-2/0-R', 3), ('SHR-2/0-B', 2),
            ('SEN-001', 1), 
            ('TRM-20-516', 12), ('TRM-06-516', 10), ('TRM-20-38', 4), 
            ('TRM-06-14', 5), ('TRM-SM', 1), ('TRM-08-316', 2),
            ('WIR-01-R', 12), ('WIR-01-B', 8),
            ('WIR-10-3', 1), ('WIR-14-3', 7),
            ('WIR-20-R', 12), ('WIR-20-B', 8), ('WIR-06-R', 15), 
            ('WIR-06-B', 15), ('WIR-08-B', 5), ('WIR-08-R', 2.5)
        ]
        
        db_parts = conn.execute(text("SELECT id, sku FROM part_master")).fetchall()
        part_map = {row[1]: row[0] for row in db_parts}
        for sku, qty in bom_map:
            conn.execute(text("INSERT INTO bom_items (part_id, qty_per_unit) VALUES (:pid, :qty)"), {"pid": part_map[sku], "qty": qty})

        # 3. SCHEDULE
        targets_2026 = [(1, 30), (2, 40), (3, 53), (4, 58), (5, 74), (6, 125), (7, 136), (8, 221), (9, 230), (10, 288), (11, 460), (12, 468)]
        targets_2027 = [(m, math.ceil(468 * (1.02**m))) for m in range(1, 7)]

        serial_counter = 1
        for month_num, qty in targets_2026:
            workdays = get_workdays(2026, month_num)
            direct_qty = math.floor(qty * 0.25)
            dealer_qty = qty - direct_qty
            unit_pool = ['DIRECT'] * direct_qty + ['DEALER'] * dealer_qty
            day_idx = 0
            for unit_type in unit_pool:
                conn.execute(text("INSERT INTO production_unit (serial_number, build_date, sales_channel, status) VALUES (:sn, :bd, :ch, 'PLANNED')"), 
                             {"sn": f"IDX-{serial_counter:04d}", "bd": workdays[day_idx], "ch": unit_type})
                serial_counter += 1
                day_idx = (day_idx + 1) % len(workdays)

        for month_num, qty in targets_2027:
            workdays = get_workdays(2027, month_num)
            direct_qty = math.floor(qty * 0.25)
            dealer_qty = qty - direct_qty
            unit_pool = ['DIRECT'] * direct_qty + ['DEALER'] * dealer_qty
            day_idx = 0
            for unit_type in unit_pool:
                conn.execute(text("INSERT INTO production_unit (serial_number, build_date, sales_channel, status) VALUES (:sn, :bd, :ch, 'PLANNED')"), 
                             {"sn": f"IDX-27-{serial_counter:04d}", "bd": workdays[day_idx], "ch": unit_type})
                serial_counter += 1
                day_idx = (day_idx + 1) % len(workdays)

        # 4. STAFFING & OPEX
        roles = [
            ("CTO", 160000, [1]*18),
            ("Senior Electrical Engineer", 165000, [0,0,1,1,1,1,1,1,1,1,1,1] + [1]*6),
            ("Senior Mechanical Engineer", 155000, [1]*18),
            ("Firmware Engineer", 150000, [0,0,0,1,1,1,1,1,1,1,1,1] + [1]*6),
            ("Test Engineer", 130000, [0,0,0,0,1,1,1,1,1,1,1,1] + [1]*6),
            ("Ops / Project Manager", 95000, [1]*18),
            ("Admin (part time)", 80000, [0.5]*18),
        ]
        
        for r_name, salary, counts in roles:
            conn.execute(text("INSERT INTO opex_roles (role_name, annual_salary) VALUES (:n, :s)"), {"n": r_name, "s": salary})
            rid = conn.execute(text("SELECT last_insert_rowid()")).scalar()
            
            for idx, count in enumerate(counts):
                curr_year = 2026 + (idx // 12)
                curr_month = (idx % 12) + 1
                m_date = date(curr_year, curr_month, 1)
                conn.execute(text("INSERT INTO opex_staffing_plan (role_id, month_date, headcount) VALUES (:rid, :dt, :hc)"),
                             {"rid": rid, "dt": m_date, "hc": count})

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
            extended_amounts = amounts + [amounts[-1]] * 6
            
            for idx, amount in enumerate(extended_amounts):
                curr_year = 2026 + (idx // 12)
                curr_month = (idx % 12) + 1
                m_date = date(curr_year, curr_month, 1)
                
                if amount > 0:
                    conn.execute(text("INSERT INTO opex_general_expenses (category, expense_type, month_date, amount) VALUES (:c, :t, :d, :a)"),
                                 {"c": category, "t": exp_type, "d": m_date, "a": amount})

        conn.commit()
        print(f"--- SUCCESS: Database Rebuilt with Full OpEx Budget ---")

if __name__ == "__main__":
    run_seed()