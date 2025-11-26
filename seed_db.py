"""
IdleX CFO Console - Database Schema and Seeder
Version: 5.0 Enterprise Edition
Full feature set: Scenarios, Inventory/PO, Capacity, Covenants, Fleet ROI, 
                  Warranty, Service Revenue, Board Pack, Audit Log, External Integrations
"""

from sqlalchemy import create_engine, text
from datetime import date, timedelta
import os
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_db_engine():
    db_url = os.getenv("DATABASE_URL")
    if db_url:
        if db_url.startswith("postgres://"):
            db_url = db_url.replace("postgres://", "postgresql://", 1)
        return create_engine(db_url)
    return create_engine('sqlite:///idlex.db')


def get_db_type():
    db_url = os.getenv("DATABASE_URL")
    return "postgresql" if db_url and "postgres" in db_url else "sqlite"


def run_seed():
    engine = get_db_engine()
    db_type = get_db_type()
    
    # Type mappings
    if db_type == "postgresql":
        SERIAL_PK = "SERIAL PRIMARY KEY"
        JSON_TYPE = "JSONB"
        DATETIME_TYPE = "TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
        BOOL_TYPE = "BOOLEAN"
        BOOL_TRUE = "TRUE"
        BOOL_FALSE = "FALSE"
    else:
        SERIAL_PK = "INTEGER PRIMARY KEY AUTOINCREMENT"
        JSON_TYPE = "TEXT"
        DATETIME_TYPE = "DATETIME DEFAULT CURRENT_TIMESTAMP"
        BOOL_TYPE = "INTEGER"
        BOOL_TRUE = "1"
        BOOL_FALSE = "0"
    
    with engine.connect() as conn:
        # =====================================================================
        # DROP ALL TABLES
        # =====================================================================
        tables = [
            "audit_log", "external_data_import",
            "unit_service_subscription", "service_plan",
            "unit_warranty_event", "warranty_policy",
            "unit_fleet_assignment", "fleet",
            "covenant_config",
            "work_center_assignment", "routing_step", "work_center",
            "purchase_order_line", "purchase_order_header", "inventory_balance",
            "scenario_cash_timeseries", "scenario_growth_profile", "scenario_header",
            "opex_general_expenses", "opex_staffing_plan", "opex_roles",
            "bom_items", "production_unit", "part_master", "global_config"
        ]
        
        for t in tables:
            try:
                conn.execute(text(f"DROP TABLE IF EXISTS {t}"))
            except Exception as e:
                logger.warning(f"Drop {t}: {e}")
        conn.commit()
        
        # =====================================================================
        # CORE TABLES
        # =====================================================================
        
        conn.execute(text(f"""
            CREATE TABLE global_config (
                id {SERIAL_PK},
                setting_key TEXT UNIQUE NOT NULL,
                setting_value TEXT,
                description TEXT
            )
        """))
        
        conn.execute(text(f"""
            CREATE TABLE part_master (
                id {SERIAL_PK},
                sku TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                cost REAL NOT NULL,
                moq INTEGER DEFAULT 1,
                lead_time INTEGER DEFAULT 0,
                deposit_pct REAL DEFAULT 0,
                deposit_days INTEGER DEFAULT 0,
                balance_days INTEGER DEFAULT 0,
                reorder_point INTEGER DEFAULT 0,
                safety_stock INTEGER DEFAULT 0,
                supplier_name TEXT
            )
        """))
        
        conn.execute(text(f"""
            CREATE TABLE production_unit (
                id {SERIAL_PK},
                serial_number TEXT UNIQUE NOT NULL,
                build_date DATE NOT NULL,
                sales_channel TEXT DEFAULT 'DEALER',
                status TEXT DEFAULT 'PLANNED',
                warranty_policy_id INTEGER,
                notes TEXT
            )
        """))
        
        conn.execute(text(f"""
            CREATE TABLE bom_items (
                id {SERIAL_PK},
                part_id INTEGER NOT NULL,
                qty_per_unit REAL NOT NULL,
                FOREIGN KEY (part_id) REFERENCES part_master(id)
            )
        """))
        
        conn.execute(text(f"""
            CREATE TABLE opex_roles (
                id {SERIAL_PK},
                role_name TEXT UNIQUE NOT NULL,
                annual_salary REAL NOT NULL,
                department TEXT DEFAULT 'Operations'
            )
        """))
        
        conn.execute(text(f"""
            CREATE TABLE opex_staffing_plan (
                id {SERIAL_PK},
                role_id INTEGER NOT NULL,
                month_date DATE NOT NULL,
                headcount REAL NOT NULL,
                FOREIGN KEY (role_id) REFERENCES opex_roles(id),
                UNIQUE(role_id, month_date)
            )
        """))
        
        conn.execute(text(f"""
            CREATE TABLE opex_general_expenses (
                id {SERIAL_PK},
                category TEXT NOT NULL,
                expense_type TEXT NOT NULL,
                month_date DATE NOT NULL,
                amount REAL NOT NULL
            )
        """))
        
        # =====================================================================
        # SCENARIO TABLES (Feature 1)
        # =====================================================================
        
        conn.execute(text(f"""
            CREATE TABLE scenario_header (
                id {SERIAL_PK},
                name TEXT UNIQUE NOT NULL,
                created_at {DATETIME_TYPE},
                created_by TEXT DEFAULT 'system',
                description TEXT,
                is_plan_of_record {BOOL_TYPE} DEFAULT {BOOL_FALSE},
                base_start_cash REAL,
                base_loc_limit REAL,
                start_units INTEGER,
                growth_rate REAL,
                start_date DATE,
                forecast_months INTEGER,
                total_revenue REAL,
                total_units INTEGER,
                min_cash REAL,
                notes TEXT
            )
        """))
        
        conn.execute(text(f"""
            CREATE TABLE scenario_growth_profile (
                id {SERIAL_PK},
                scenario_id INTEGER NOT NULL,
                month_number INTEGER NOT NULL,
                month_date DATE,
                monthly_growth_pct REAL,
                planned_units INTEGER,
                FOREIGN KEY (scenario_id) REFERENCES scenario_header(id) ON DELETE CASCADE
            )
        """))
        
        conn.execute(text(f"""
            CREATE TABLE scenario_cash_timeseries (
                id {SERIAL_PK},
                scenario_id INTEGER NOT NULL,
                date DATE NOT NULL,
                cash_balance REAL,
                cumulative_revenue REAL,
                FOREIGN KEY (scenario_id) REFERENCES scenario_header(id) ON DELETE CASCADE
            )
        """))
        
        # =====================================================================
        # INVENTORY & PURCHASING TABLES (Feature 2)
        # =====================================================================
        
        conn.execute(text(f"""
            CREATE TABLE inventory_balance (
                id {SERIAL_PK},
                part_id INTEGER NOT NULL,
                as_of_date DATE NOT NULL,
                quantity_on_hand REAL NOT NULL,
                quantity_reserved REAL DEFAULT 0,
                FOREIGN KEY (part_id) REFERENCES part_master(id),
                UNIQUE(part_id, as_of_date)
            )
        """))
        
        conn.execute(text(f"""
            CREATE TABLE purchase_order_header (
                id {SERIAL_PK},
                po_number TEXT UNIQUE NOT NULL,
                supplier_name TEXT,
                order_date DATE NOT NULL,
                expected_delivery_date DATE,
                status TEXT DEFAULT 'PLANNED',
                total_value REAL DEFAULT 0,
                notes TEXT
            )
        """))
        
        conn.execute(text(f"""
            CREATE TABLE purchase_order_line (
                id {SERIAL_PK},
                po_id INTEGER NOT NULL,
                part_id INTEGER NOT NULL,
                quantity_ordered REAL NOT NULL,
                quantity_received REAL DEFAULT 0,
                unit_cost REAL NOT NULL,
                deposit_pct REAL,
                deposit_days INTEGER,
                balance_days INTEGER,
                FOREIGN KEY (po_id) REFERENCES purchase_order_header(id) ON DELETE CASCADE,
                FOREIGN KEY (part_id) REFERENCES part_master(id)
            )
        """))
        
        # =====================================================================
        # CAPACITY PLANNING TABLES (Feature 3)
        # =====================================================================
        
        conn.execute(text(f"""
            CREATE TABLE work_center (
                id {SERIAL_PK},
                name TEXT UNIQUE NOT NULL,
                description TEXT,
                hours_per_day REAL DEFAULT 8,
                days_per_week INTEGER DEFAULT 5,
                efficiency_pct REAL DEFAULT 0.85
            )
        """))
        
        conn.execute(text(f"""
            CREATE TABLE routing_step (
                id {SERIAL_PK},
                work_center_id INTEGER NOT NULL,
                step_name TEXT NOT NULL,
                step_order INTEGER DEFAULT 1,
                minutes_per_unit REAL NOT NULL,
                is_bottleneck {BOOL_TYPE} DEFAULT {BOOL_FALSE},
                FOREIGN KEY (work_center_id) REFERENCES work_center(id) ON DELETE CASCADE
            )
        """))
        
        conn.execute(text(f"""
            CREATE TABLE work_center_assignment (
                id {SERIAL_PK},
                role_id INTEGER NOT NULL,
                work_center_id INTEGER NOT NULL,
                fraction_of_time REAL DEFAULT 1.0,
                FOREIGN KEY (role_id) REFERENCES opex_roles(id),
                FOREIGN KEY (work_center_id) REFERENCES work_center(id)
            )
        """))
        
        # =====================================================================
        # COVENANT TABLES (Feature 4)
        # =====================================================================
        
        conn.execute(text(f"""
            CREATE TABLE covenant_config (
                id {SERIAL_PK},
                name TEXT UNIQUE NOT NULL,
                description TEXT,
                covenant_type TEXT NOT NULL,
                threshold_value REAL NOT NULL,
                comparison TEXT DEFAULT 'GREATER_EQUAL',
                lookback_months INTEGER,
                active {BOOL_TYPE} DEFAULT {BOOL_TRUE}
            )
        """))
        
        # =====================================================================
        # FLEET & ROI TABLES (Feature 5)
        # =====================================================================
        
        conn.execute(text(f"""
            CREATE TABLE fleet (
                id {SERIAL_PK},
                name TEXT UNIQUE NOT NULL,
                fleet_type TEXT DEFAULT 'MID_FLEET',
                contact_name TEXT,
                primary_lanes TEXT,
                truck_count INTEGER,
                nights_on_road_per_year REAL DEFAULT 250,
                idle_hours_per_night REAL DEFAULT 8,
                diesel_price_assumption REAL DEFAULT 4.50,
                gallons_per_idle_hour REAL DEFAULT 1.0,
                notes TEXT
            )
        """))
        
        conn.execute(text(f"""
            CREATE TABLE unit_fleet_assignment (
                id {SERIAL_PK},
                production_unit_id INTEGER NOT NULL,
                fleet_id INTEGER NOT NULL,
                in_service_date DATE,
                truck_identifier TEXT,
                purchase_price REAL,
                FOREIGN KEY (production_unit_id) REFERENCES production_unit(id),
                FOREIGN KEY (fleet_id) REFERENCES fleet(id)
            )
        """))
        
        # =====================================================================
        # WARRANTY TABLES (Feature 6)
        # =====================================================================
        
        conn.execute(text(f"""
            CREATE TABLE warranty_policy (
                id {SERIAL_PK},
                policy_name TEXT UNIQUE NOT NULL,
                duration_months INTEGER DEFAULT 12,
                coverage_notes TEXT
            )
        """))
        
        conn.execute(text(f"""
            CREATE TABLE unit_warranty_event (
                id {SERIAL_PK},
                production_unit_id INTEGER NOT NULL,
                event_date DATE NOT NULL,
                failure_mode TEXT,
                part_id INTEGER,
                cost_of_repair REAL DEFAULT 0,
                is_in_warranty {BOOL_TYPE} DEFAULT {BOOL_TRUE},
                is_replaced {BOOL_TYPE} DEFAULT {BOOL_FALSE},
                resolution_notes TEXT,
                FOREIGN KEY (production_unit_id) REFERENCES production_unit(id),
                FOREIGN KEY (part_id) REFERENCES part_master(id)
            )
        """))
        
        # =====================================================================
        # SERVICE REVENUE TABLES (Feature 7)
        # =====================================================================
        
        conn.execute(text(f"""
            CREATE TABLE service_plan (
                id {SERIAL_PK},
                name TEXT UNIQUE NOT NULL,
                annual_price REAL NOT NULL,
                term_months INTEGER DEFAULT 12,
                coverage_notes TEXT
            )
        """))
        
        conn.execute(text(f"""
            CREATE TABLE unit_service_subscription (
                id {SERIAL_PK},
                production_unit_id INTEGER NOT NULL,
                service_plan_id INTEGER NOT NULL,
                fleet_id INTEGER,
                start_date DATE NOT NULL,
                end_date DATE,
                status TEXT DEFAULT 'ACTIVE',
                payment_type TEXT DEFAULT 'ANNUAL',
                FOREIGN KEY (production_unit_id) REFERENCES production_unit(id),
                FOREIGN KEY (service_plan_id) REFERENCES service_plan(id),
                FOREIGN KEY (fleet_id) REFERENCES fleet(id)
            )
        """))
        
        # =====================================================================
        # AUDIT LOG (Feature 9)
        # =====================================================================
        
        conn.execute(text(f"""
            CREATE TABLE audit_log (
                id {SERIAL_PK},
                timestamp {DATETIME_TYPE},
                user_name TEXT DEFAULT 'system',
                action TEXT NOT NULL,
                object_type TEXT,
                object_id TEXT,
                data_before {JSON_TYPE},
                data_after {JSON_TYPE}
            )
        """))
        
        # =====================================================================
        # EXTERNAL INTEGRATION (Feature 10)
        # =====================================================================
        
        conn.execute(text(f"""
            CREATE TABLE external_data_import (
                id {SERIAL_PK},
                source_system TEXT NOT NULL,
                import_type TEXT NOT NULL,
                imported_at {DATETIME_TYPE},
                payload {JSON_TYPE},
                processed {BOOL_TYPE} DEFAULT {BOOL_FALSE},
                process_notes TEXT
            )
        """))
        
        conn.commit()
        
        # =====================================================================
        # SEED DATA
        # =====================================================================
        
        # Global Config
        configs = [
            ("start_cash", "1600000", "Starting cash balance"),
            ("loc_limit", "500000", "Line of credit limit"),
            ("msrp_price", "8500", "Unit MSRP"),
            ("dealer_discount", "0.75", "Dealer discount rate"),
            ("direct_sales_pct", "0.25", "Direct sales percentage"),
            ("company_name", "IdleX", "Company name for reports"),
        ]
        for k, v, d in configs:
            conn.execute(text("INSERT INTO global_config (setting_key, setting_value, description) VALUES (:k, :v, :d)"),
                        {"k": k, "v": v, "d": d})
        
        # Parts with suppliers
        parts = [
            ("BAT-LFP-48V", "48V LFP Battery Pack", 1850.00, 50, 112, 0.30, -84, 30, 25, 10, "CATL America"),
            ("INV-3KW", "3kW Inverter", 420.00, 25, 56, 0.25, -42, 14, 15, 5, "Victron Energy"),
            ("HVAC-12K", "12,000 BTU HVAC Unit", 780.00, 20, 84, 0.25, -63, 21, 10, 5, "Dometic"),
            ("CTRL-V2", "Control Module V2", 285.00, 50, 42, 0.00, 0, 30, 30, 10, "Custom Electronics"),
            ("WIRE-KIT", "Wiring Harness Kit", 145.00, 100, 28, 0.00, 0, 30, 50, 20, "Waytek"),
            ("ENCL-AL", "Aluminum Enclosure", 320.00, 25, 35, 0.20, -21, 14, 15, 5, "80/20 Inc"),
            ("DSPLY-7", "7-inch Display Panel", 165.00, 50, 42, 0.00, 0, 30, 25, 10, "Newhaven Display"),
            ("MISC-HW", "Misc Hardware Kit", 85.00, 100, 14, 0.00, 0, 14, 100, 25, "Fastenal"),
        ]
        
        for p in parts:
            conn.execute(text("""
                INSERT INTO part_master (sku, name, cost, moq, lead_time, deposit_pct, deposit_days, 
                                         balance_days, reorder_point, safety_stock, supplier_name)
                VALUES (:sku, :name, :cost, :moq, :lt, :dep, :dd, :bd, :rop, :ss, :sup)
            """), {"sku": p[0], "name": p[1], "cost": p[2], "moq": p[3], "lt": p[4], 
                   "dep": p[5], "dd": p[6], "bd": p[7], "rop": p[8], "ss": p[9], "sup": p[10]})
        
        # BOM
        for pid in range(1, 9):
            conn.execute(text("INSERT INTO bom_items (part_id, qty_per_unit) VALUES (:p, 1)"), {"p": pid})
        
        # Roles
        roles = [
            ("CEO", 250000, "Executive"),
            ("VP Engineering", 200000, "Engineering"),
            ("VP Operations", 180000, "Operations"),
            ("Mechanical Engineer", 120000, "Engineering"),
            ("Electrical Engineer", 125000, "Engineering"),
            ("Production Supervisor", 85000, "Operations"),
            ("Assembler - Senior", 65000, "Operations"),
            ("Assembler - Junior", 52000, "Operations"),
            ("Quality Technician", 60000, "Operations"),
            ("Supply Chain Manager", 95000, "Operations"),
            ("Finance Manager", 110000, "Finance"),
            ("Sales Manager", 100000, "Sales"),
            ("Field Service Tech", 70000, "Service"),
        ]
        
        for r in roles:
            conn.execute(text("INSERT INTO opex_roles (role_name, annual_salary, department) VALUES (:n, :s, :d)"),
                        {"n": r[0], "s": r[1], "d": r[2]})
        
        # Staffing (24 months with growth)
        base_hc = {1:1, 2:1, 3:1, 4:1, 5:1, 6:1, 7:2, 8:3, 9:1, 10:1, 11:1, 12:1, 13:0}
        for m in range(24):
            month = date(2026, 1, 1) + timedelta(days=30*m)
            month = month.replace(day=1)
            growth = 1 + m * 0.04
            for rid, hc in base_hc.items():
                actual = hc * growth if rid in [7, 8, 9] else hc
                if actual > 0:
                    conn.execute(text("INSERT INTO opex_staffing_plan (role_id, month_date, headcount) VALUES (:r, :m, :h)"),
                                {"r": rid, "m": month, "h": round(actual, 1)})
        
        # Production units
        for i in range(200):
            bd = date(2026, 1, 6) + timedelta(days=i // 4)
            while bd.weekday() >= 5:
                bd += timedelta(days=1)
            ch = 'DIRECT' if i % 4 == 0 else 'DEALER'
            conn.execute(text("INSERT INTO production_unit (serial_number, build_date, sales_channel, status, warranty_policy_id) VALUES (:sn, :bd, :ch, 'PLANNED', 1)"),
                        {"sn": f"IDX-{i+1:04d}", "bd": bd, "ch": ch})
        
        # Work Centers
        wcs = [
            ("Assembly Line 1", "Primary assembly", 8, 5, 0.85),
            ("Assembly Line 2", "Secondary/overflow", 8, 5, 0.80),
            ("QA Station", "Testing and QA", 8, 5, 0.90),
            ("Packaging", "Final prep and pack", 8, 5, 0.95),
        ]
        for wc in wcs:
            conn.execute(text("INSERT INTO work_center (name, description, hours_per_day, days_per_week, efficiency_pct) VALUES (:n, :d, :h, :dw, :e)"),
                        {"n": wc[0], "d": wc[1], "h": wc[2], "dw": wc[3], "e": wc[4]})
        
        # Routing
        routing = [
            (1, "Battery Install", 1, 45, True),
            (1, "Inverter Mount", 2, 30, False),
            (1, "HVAC Integration", 3, 55, True),
            (1, "Wiring Harness", 4, 35, False),
            (2, "Control Module", 1, 25, False),
            (2, "Display Install", 2, 15, False),
            (3, "Electrical Test", 1, 30, False),
            (3, "Thermal Cycle Test", 2, 45, True),
            (4, "Final Inspection", 1, 20, False),
            (4, "Packaging", 2, 15, False),
        ]
        for r in routing:
            conn.execute(text("INSERT INTO routing_step (work_center_id, step_name, step_order, minutes_per_unit, is_bottleneck) VALUES (:wc, :n, :o, :m, :b)"),
                        {"wc": r[0], "n": r[1], "o": r[2], "m": r[3], "b": 1 if r[4] else 0})
        
        # Work Center Assignments
        assigns = [(7, 1, 0.6), (7, 2, 0.4), (8, 1, 0.5), (8, 2, 0.5), (9, 3, 1.0), (6, 4, 0.3)]
        for a in assigns:
            conn.execute(text("INSERT INTO work_center_assignment (role_id, work_center_id, fraction_of_time) VALUES (:r, :w, :f)"),
                        {"r": a[0], "w": a[1], "f": a[2]})
        
        # Covenants
        covs = [
            ("Minimum Cash", "Maintain minimum cash balance at all times", "MIN_CASH", 250000, "GREATER_EQUAL", None),
            ("Runway Floor", "Maintain minimum months of runway", "MIN_RUNWAY", 6, "GREATER_EQUAL", None),
            ("Gross Margin", "Maintain minimum gross margin", "MIN_MARGIN", 0.25, "GREATER_EQUAL", 3),
            ("LOC Utilization", "Maximum LOC draw percentage", "MAX_LOC_UTIL", 0.80, "LESS_EQUAL", None),
        ]
        for c in covs:
            conn.execute(text("INSERT INTO covenant_config (name, description, covenant_type, threshold_value, comparison, lookback_months, active) VALUES (:n, :d, :t, :v, :c, :lb, 1)"),
                        {"n": c[0], "d": c[1], "t": c[2], "v": c[3], "c": c[4], "lb": c[5]})
        
        # Fleets
        fleets = [
            ("Werner Enterprises", "MEGA_FLEET", "I-80 Corridor", 8000, 280, 8.5, 4.25, 1.1),
            ("Heartland Express", "MEGA_FLEET", "Midwest Hub", 4500, 260, 8.0, 4.50, 1.0),
            ("Southeast Toyota Transport", "MID_FLEET", "Southeast", 850, 250, 7.5, 4.30, 0.9),
            ("Regional Carrier A", "SMALL_FLEET", "Texas Triangle", 120, 240, 7.0, 4.40, 1.0),
            ("Owner Operator Group", "SMALL_FLEET", "Various", 45, 280, 9.0, 4.50, 1.2),
        ]
        for f in fleets:
            conn.execute(text("""
                INSERT INTO fleet (name, fleet_type, primary_lanes, truck_count, nights_on_road_per_year, 
                                   idle_hours_per_night, diesel_price_assumption, gallons_per_idle_hour)
                VALUES (:n, :t, :l, :tc, :nights, :hrs, :diesel, :gal)
            """), {"n": f[0], "t": f[1], "l": f[2], "tc": f[3], "nights": f[4], "hrs": f[5], "diesel": f[6], "gal": f[7]})
        
        # Warranty Policies
        conn.execute(text("INSERT INTO warranty_policy (policy_name, duration_months, coverage_notes) VALUES ('Standard', 12, 'Parts and labor for manufacturing defects')"))
        conn.execute(text("INSERT INTO warranty_policy (policy_name, duration_months, coverage_notes) VALUES ('Extended', 24, 'Comprehensive coverage including wear items')"))
        conn.execute(text("INSERT INTO warranty_policy (policy_name, duration_months, coverage_notes) VALUES ('Fleet Premium', 36, 'Full coverage with expedited service')"))
        
        # Service Plans
        conn.execute(text("INSERT INTO service_plan (name, annual_price, term_months, coverage_notes) VALUES ('Basic Monitoring', 299, 12, 'Remote diagnostics and alerts')"))
        conn.execute(text("INSERT INTO service_plan (name, annual_price, term_months, coverage_notes) VALUES ('Pro Service', 599, 12, 'Monitoring plus preventive maintenance')"))
        conn.execute(text("INSERT INTO service_plan (name, annual_price, term_months, coverage_notes) VALUES ('Fleet Enterprise', 449, 12, 'Volume pricing with dedicated support')"))
        
        # Initial inventory
        for pid in range(1, 9):
            conn.execute(text("INSERT INTO inventory_balance (part_id, as_of_date, quantity_on_hand, quantity_reserved) VALUES (:p, :d, :q, 0)"),
                        {"p": pid, "d": date(2026, 1, 1), "q": 150})
        
        # Sample fleet assignments (first 50 units)
        for i in range(1, 51):
            fid = (i % 5) + 1
            conn.execute(text("""
                INSERT INTO unit_fleet_assignment (production_unit_id, fleet_id, in_service_date, purchase_price)
                VALUES (:u, :f, :d, :p)
            """), {"u": i, "f": fid, "d": date(2026, 1, 1) + timedelta(days=i*2), "p": 8500 if i % 4 == 0 else 6375})
        
        # Sample warranty events
        events = [
            (5, date(2026, 3, 15), "Inverter fault", 2, 420, True, False),
            (12, date(2026, 4, 1), "Display malfunction", 7, 165, True, True),
            (23, date(2026, 4, 20), "Wiring harness short", 5, 145, True, False),
        ]
        for e in events:
            conn.execute(text("""
                INSERT INTO unit_warranty_event (production_unit_id, event_date, failure_mode, part_id, cost_of_repair, is_in_warranty, is_replaced)
                VALUES (:u, :d, :m, :p, :c, :w, :r)
            """), {"u": e[0], "d": e[1], "m": e[2], "p": e[3], "c": e[4], "w": 1 if e[5] else 0, "r": 1 if e[6] else 0})
        
        # Sample service subscriptions
        for i in range(1, 31):
            plan = (i % 3) + 1
            conn.execute(text("""
                INSERT INTO unit_service_subscription (production_unit_id, service_plan_id, fleet_id, start_date, status)
                VALUES (:u, :p, :f, :d, 'ACTIVE')
            """), {"u": i, "p": plan, "f": (i % 5) + 1, "d": date(2026, 2, 1)})
        
        conn.commit()
        logger.info("Database seeded with v5.0 Enterprise schema")


if __name__ == "__main__":
    run_seed()
