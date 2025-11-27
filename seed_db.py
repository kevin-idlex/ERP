"""
IdleX ERP - Database Schema & Seeder
Version: 7.0 (Production Cloud)
Compatible: PostgreSQL (Cloud Run) / SQLite (Local Dev)

CRITICAL: All date columns use TEXT type for cross-DB compatibility.
         All dates stored as ISO format strings: 'YYYY-MM-DD'
"""

from sqlalchemy import create_engine, text
from datetime import date, timedelta
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_db_engine():
    """
    Dual-engine strategy:
    - If DATABASE_URL exists (Cloud), use PostgreSQL
    - Otherwise, use local SQLite file
    
    CRITICAL: SQLAlchemy requires 'postgresql://' not 'postgres://'
    """
    db_url = os.getenv("DATABASE_URL")
    if db_url:
        # Google Cloud / Heroku use 'postgres://' but SQLAlchemy needs 'postgresql://'
        if db_url.startswith("postgres://"):
            db_url = db_url.replace("postgres://", "postgresql://", 1)
        return create_engine(db_url)
    return create_engine('sqlite:///idlex.db')


def get_db_type():
    """Returns 'postgresql' or 'sqlite' for conditional SQL syntax."""
    db_url = os.getenv("DATABASE_URL")
    return "postgresql" if db_url and "postgres" in db_url else "sqlite"


def run_seed():
    """
    Master database initialization.
    
    Design Philosophy:
    - DROP all tables first (clean slate)
    - CREATE with proper types for each DB engine
    - INSERT realistic seed data for demonstration
    
    Called by:
    - Manual "Rebuild Database" button
    - Auto-healing on app startup if tables missing
    """
    engine = get_db_engine()
    db_type = get_db_type()
    
    # SQL type mappings for cross-DB compatibility
    if db_type == "postgresql":
        SERIAL_PK = "SERIAL PRIMARY KEY"
        JSON_TYPE = "JSONB"
        DATETIME_DEF = "TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
    else:
        SERIAL_PK = "INTEGER PRIMARY KEY AUTOINCREMENT"
        JSON_TYPE = "TEXT"
        DATETIME_DEF = "DATETIME DEFAULT CURRENT_TIMESTAMP"
    
    with engine.connect() as conn:
        # =================================================================
        # PHASE 1: DROP ALL TABLES (reverse dependency order)
        # =================================================================
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
                logger.warning(f"Could not drop {t}: {e}")
        conn.commit()
        
        # =================================================================
        # PHASE 2: CREATE TABLES
        # =================================================================
        
        # --- Configuration ---
        conn.execute(text(f"""
            CREATE TABLE global_config (
                id {SERIAL_PK},
                setting_key TEXT UNIQUE NOT NULL,
                setting_value TEXT,
                description TEXT
            )
        """))
        
        # --- Supply Chain ---
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
            CREATE TABLE bom_items (
                id {SERIAL_PK},
                part_id INTEGER NOT NULL,
                qty_per_unit REAL NOT NULL
            )
        """))
        
        # --- Production ---
        conn.execute(text(f"""
            CREATE TABLE production_unit (
                id {SERIAL_PK},
                serial_number TEXT UNIQUE NOT NULL,
                build_date TEXT NOT NULL,
                sales_channel TEXT DEFAULT 'DEALER',
                status TEXT DEFAULT 'PLANNED',
                warranty_policy_id INTEGER,
                notes TEXT
            )
        """))
        
        # --- OpEx ---
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
                month_date TEXT NOT NULL,
                headcount REAL NOT NULL,
                UNIQUE(role_id, month_date)
            )
        """))
        
        conn.execute(text(f"""
            CREATE TABLE opex_general_expenses (
                id {SERIAL_PK},
                category TEXT NOT NULL,
                expense_type TEXT NOT NULL,
                month_date TEXT NOT NULL,
                amount REAL NOT NULL
            )
        """))
        
        # --- Scenarios ---
        conn.execute(text(f"""
            CREATE TABLE scenario_header (
                id {SERIAL_PK},
                name TEXT UNIQUE NOT NULL,
                created_at {DATETIME_DEF},
                created_by TEXT DEFAULT 'system',
                description TEXT,
                is_plan_of_record INTEGER DEFAULT 0,
                base_start_cash REAL,
                base_loc_limit REAL,
                start_units INTEGER,
                growth_rate REAL,
                start_date TEXT,
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
                month_date TEXT,
                monthly_growth_pct REAL,
                planned_units INTEGER
            )
        """))
        
        conn.execute(text(f"""
            CREATE TABLE scenario_cash_timeseries (
                id {SERIAL_PK},
                scenario_id INTEGER NOT NULL,
                date TEXT NOT NULL,
                cash_balance REAL,
                cumulative_revenue REAL
            )
        """))
        
        # --- Inventory & Purchasing ---
        conn.execute(text(f"""
            CREATE TABLE inventory_balance (
                id {SERIAL_PK},
                part_id INTEGER NOT NULL,
                as_of_date TEXT NOT NULL,
                quantity_on_hand REAL NOT NULL,
                quantity_reserved REAL DEFAULT 0,
                UNIQUE(part_id, as_of_date)
            )
        """))
        
        conn.execute(text(f"""
            CREATE TABLE purchase_order_header (
                id {SERIAL_PK},
                po_number TEXT UNIQUE NOT NULL,
                supplier_name TEXT,
                order_date TEXT NOT NULL,
                expected_delivery_date TEXT,
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
                unit_cost REAL NOT NULL
            )
        """))
        
        # --- Capacity ---
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
                is_bottleneck INTEGER DEFAULT 0
            )
        """))
        
        conn.execute(text(f"""
            CREATE TABLE work_center_assignment (
                id {SERIAL_PK},
                role_id INTEGER NOT NULL,
                work_center_id INTEGER NOT NULL,
                fraction_of_time REAL DEFAULT 1.0
            )
        """))
        
        # --- Covenants ---
        conn.execute(text(f"""
            CREATE TABLE covenant_config (
                id {SERIAL_PK},
                name TEXT UNIQUE NOT NULL,
                description TEXT,
                covenant_type TEXT NOT NULL,
                threshold_value REAL NOT NULL,
                comparison TEXT DEFAULT 'GREATER_EQUAL',
                lookback_months INTEGER,
                active INTEGER DEFAULT 1
            )
        """))
        
        # --- Fleet ---
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
                in_service_date TEXT,
                truck_identifier TEXT,
                purchase_price REAL
            )
        """))
        
        # --- Warranty ---
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
                event_date TEXT NOT NULL,
                failure_mode TEXT,
                part_id INTEGER,
                cost_of_repair REAL DEFAULT 0,
                is_in_warranty INTEGER DEFAULT 1,
                is_replaced INTEGER DEFAULT 0,
                resolution_notes TEXT
            )
        """))
        
        # --- Service ---
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
                start_date TEXT NOT NULL,
                end_date TEXT,
                status TEXT DEFAULT 'ACTIVE',
                payment_type TEXT DEFAULT 'ANNUAL'
            )
        """))
        
        # --- Audit ---
        conn.execute(text(f"""
            CREATE TABLE audit_log (
                id {SERIAL_PK},
                timestamp {DATETIME_DEF},
                user_name TEXT DEFAULT 'system',
                action TEXT NOT NULL,
                object_type TEXT,
                object_id TEXT,
                data_before {JSON_TYPE},
                data_after {JSON_TYPE}
            )
        """))
        
        # --- External Integration ---
        conn.execute(text(f"""
            CREATE TABLE external_data_import (
                id {SERIAL_PK},
                source_system TEXT NOT NULL,
                import_type TEXT NOT NULL,
                imported_at {DATETIME_DEF},
                payload {JSON_TYPE},
                processed INTEGER DEFAULT 0,
                process_notes TEXT
            )
        """))
        
        conn.commit()
        
        # =================================================================
        # PHASE 3: SEED DATA
        # =================================================================
        
        # --- Global Config ---
        configs = [
            ("start_cash", "1600000", "Default starting cash balance"),
            ("loc_limit", "500000", "Default line of credit limit"),
            ("msrp_price", "8500", "Unit MSRP price"),
            ("dealer_discount", "0.75", "Dealer price as % of MSRP"),
            ("direct_sales_pct", "0.25", "Target % of direct sales"),
            ("company_name", "IdleX", "Company name for reports"),
        ]
        for k, v, d in configs:
            conn.execute(text(
                "INSERT INTO global_config (setting_key, setting_value, description) VALUES (:k, :v, :d)"
            ), {"k": k, "v": v, "d": d})
        
        # --- Parts (BOM) - Actual IdleX eAPU Bill of Materials ---
        # Format: (sku, name, unit_cost, moq, lead_time_days, deposit_pct, deposit_days, balance_days, reorder_point, safety_stock, supplier, qty_per_unit)
        # 
        # Payment Terms Logic:
        #   deposit_days: Days relative to BUILD when deposit is due (negative = before build)
        #   balance_days: Days relative to BUILD when balance is due (negative = before, positive = after)
        #   For items with deposit: deposit due at order time = -(lead_time)
        #
        # CRITICAL: qty_per_unit varies (e.g., 3 batteries per eAPU)
        
        bom_parts = [
            # === MAJOR COMPONENTS (Safiery / Long Lead) ===
            # SKU, Name, Unit Cost, MOQ, Lead Time, Deposit%, Deposit Days, Balance Days, ROP, SS, Supplier, Qty/Unit
            ("BAT-SS-48V", "Safiery Solid State 2C Lithium 48V", 715.26, 50, 103, 0.50, -103, -45, 150, 50, "Safiery", 3),
            ("DCDC-SCOTTY", "Scotty AI DC-DC (Slim Case)", 775.22, 10, 103, 0.50, -103, -45, 10, 5, "Safiery", 1),
            ("SCRN-STARAI", "STAR AI Nexus GPS + eSIM Display", 230.00, 25, 103, 0.00, 0, -45, 25, 10, "Safiery", 1),
            ("CBL-RS485", "RS485 Battery Interconnect Cables", 15.00, 100, 75, 0.00, 0, -45, 200, 50, "Safiery", 2),
            ("CBL-VECAN", "VE.Can to NMEA2000 Micro-C Male", 27.00, 50, 103, 0.00, 0, -45, 50, 20, "Safiery", 1),
            
            # === POWER COMPONENTS ===
            ("BPR-220400", "Battery Protect BPR000220400", 180.00, 25, 7, 0.00, 0, 0, 25, 10, "Victron", 1),
            ("UCAP-SKEL", "SkelStart 2100 CCA Ultracapacitor", 975.00, 10, 14, 0.00, 0, 0, 10, 5, "Skeleton", 1),
            ("FUSE-MEGA100", "MEGA Fuse 100A (Scotty DC-DC)", 1.70, 100, 7, 0.00, 0, 30, 100, 25, "Victron", 1),
            ("SENS-BTEMP", "Battery Temperature Sensor", 17.40, 50, 7, 0.00, 0, 30, 50, 20, "Victron", 1),
            
            # === ENCLOSURE ===
            ("ENCL-AL", "Aluminum Enclosure Case", 300.00, 10, 14, 0.00, 0, -7, 10, 5, "Custom Fab", 1),
            ("GROM-125", "1-1/4 Grommet (9600K58)", 5.00, 100, 7, 0.00, 0, 30, 100, 25, "McMaster-Carr", 1),
            
            # === CABLES & CONNECTORS ===
            ("CBL-ETH09", "Ethernet RJ45 Cable 0.9m", 7.80, 50, 7, 0.00, 0, 30, 50, 20, "Amazon", 1),
            
            # === WIRE (Per Foot Costs, Qty = Feet per Unit) ===
            ("WIRE-1AWG-A", "Wire 1 AWG (Battery Bus)", 3.64, 100, 7, 0.00, 0, 30, 200, 50, "Battery Cable USA", 8),
            ("WIRE-1AWG-B", "Wire 1 AWG (Power Distribution)", 3.64, 100, 7, 0.00, 0, 30, 200, 50, "Battery Cable USA", 12),
            ("WIRE-20-RED", "Wire 2/0 AWG Red", 4.08, 100, 7, 0.00, 0, 30, 200, 50, "Fisheries Supply", 12),
            ("WIRE-20-BLK", "Wire 2/0 AWG Black", 4.08, 100, 7, 0.00, 0, 30, 200, 50, "Fisheries Supply", 8),
            ("WIRE-6-RED", "Wire 6 AWG Red", 0.99, 500, 7, 0.00, 0, 30, 500, 100, "Fisheries Supply", 15),
            ("WIRE-6-BLK", "Wire 6 AWG Black", 0.99, 500, 7, 0.00, 0, 30, 500, 100, "Fisheries Supply", 15),
            ("WIRE-8-RED", "Wire 8 AWG Red", 0.61, 500, 7, 0.00, 0, 30, 500, 100, "Fisheries Supply", 2.5),
            ("WIRE-8-BLK", "Wire 8 AWG Black", 0.61, 500, 7, 0.00, 0, 30, 500, 100, "Fisheries Supply", 5),
            ("WIRE-103", "Ancor 10/3 Triplex", 1.18, 250, 7, 0.00, 0, 30, 250, 50, "Fisheries Supply", 1),
            ("WIRE-143", "Ancor 14/3 Triplex", 0.60, 500, 7, 0.00, 0, 30, 500, 100, "Fisheries Supply", 7),
            
            # === TERMINALS ===
            ("TERM-20-516", "Terminal 2/0 x 5/16", 1.13, 500, 7, 0.00, 0, 30, 500, 100, "Battery Cable USA", 12),
            ("TERM-6-516", "Terminal 6 x 5/16", 0.48, 500, 7, 0.00, 0, 30, 500, 100, "Battery Cable USA", 10),
            ("TERM-20-38", "Terminal 2/0 x 3/8", 1.13, 500, 7, 0.00, 0, 30, 500, 100, "Battery Cable USA", 4),
            ("TERM-6-14", "Terminal 6 x 1/4", 0.48, 500, 7, 0.00, 0, 30, 500, 100, "Battery Cable USA", 5),
            ("TERM-8-316", "Terminal 8 x 3/16", 0.44, 500, 7, 0.00, 0, 30, 500, 100, "Battery Cable USA", 2),
            ("TERM-SMALL", "Terminals Small Misc", 2.00, 100, 7, 0.00, 0, 30, 100, 25, "Battery Cable USA", 1),
            
            # === HEAT SHRINK ===
            ("HS-6-RED", "Heat Shrink #6 Red", 0.22, 1000, 7, 0.00, 0, 30, 1000, 200, "Battery Cable USA", 4),
            ("HS-6-BLK", "Heat Shrink #6 Black", 0.22, 1000, 7, 0.00, 0, 30, 1000, 200, "Battery Cable USA", 4),
            ("HS-20-RED", "Heat Shrink 2/0 Red 1in", 2.42, 500, 7, 0.00, 0, 30, 500, 100, "Battery Cable USA", 3),
            ("HS-20-BLK", "Heat Shrink 2/0 Black 1in", 2.42, 500, 7, 0.00, 0, 30, 500, 100, "Battery Cable USA", 2),
        ]
        
        # Insert parts and BOM in one pass
        for idx, p in enumerate(bom_parts, start=1):
            sku, name, cost, moq, lead_time, dep_pct, dep_days, bal_days, rop, ss, supplier, qty = p
            
            # Insert part
            conn.execute(text("""
                INSERT INTO part_master (sku, name, cost, moq, lead_time, deposit_pct, deposit_days, balance_days, reorder_point, safety_stock, supplier_name)
                VALUES (:sku, :name, :cost, :moq, :lt, :dep_pct, :dep_days, :bal_days, :rop, :ss, :supplier)
            """), {
                "sku": sku, "name": name, "cost": cost, "moq": moq, "lt": lead_time,
                "dep_pct": dep_pct, "dep_days": dep_days, "bal_days": bal_days,
                "rop": rop, "ss": ss, "supplier": supplier
            })
            
            # Insert BOM entry with actual quantity per unit
            conn.execute(text(
                "INSERT INTO bom_items (part_id, qty_per_unit) VALUES (:pid, :qty)"
            ), {"pid": idx, "qty": qty})
        
        # --- Roles ---
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
            conn.execute(text(
                "INSERT INTO opex_roles (role_name, annual_salary, department) VALUES (:name, :salary, :dept)"
            ), {"name": r[0], "salary": r[1], "dept": r[2]})
        
        # --- Staffing Plan (24 months with growth for assemblers) ---
        base_headcount = {
            1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1,  # Exec & Engineering
            7: 2, 8: 3, 9: 1,  # Assembly & QA (these grow)
            10: 1, 11: 1, 12: 1, 13: 0  # Support roles
        }
        for month_offset in range(24):
            year = 2026 + month_offset // 12
            month = (month_offset % 12) + 1
            month_str = f"{year}-{month:02d}-01"
            growth_factor = 1 + month_offset * 0.04  # 4% growth per month for assemblers
            
            for role_id, base_hc in base_headcount.items():
                # Only assemblers and QA grow with production
                actual_hc = base_hc * growth_factor if role_id in [7, 8, 9] else base_hc
                if actual_hc > 0:
                    conn.execute(text(
                        "INSERT INTO opex_staffing_plan (role_id, month_date, headcount) VALUES (:rid, :dt, :hc)"
                    ), {"rid": role_id, "dt": month_str, "hc": round(actual_hc, 1)})
        
        # --- Production Units (200 units, spread across workdays) ---
        for i in range(200):
            build_date = date(2026, 1, 6) + timedelta(days=i // 4)
            # Skip weekends
            while build_date.weekday() >= 5:
                build_date += timedelta(days=1)
            # 25% direct, 75% dealer
            channel = 'DIRECT' if i % 4 == 0 else 'DEALER'
            conn.execute(text(
                "INSERT INTO production_unit (serial_number, build_date, sales_channel, status) VALUES (:sn, :bd, :ch, 'PLANNED')"
            ), {"sn": f"IDX-{i+1:04d}", "bd": str(build_date), "ch": channel})
        
        # --- Work Centers ---
        work_centers = [
            ("Assembly Line 1", "Primary assembly line", 8, 5, 0.85),
            ("Assembly Line 2", "Secondary/overflow line", 8, 5, 0.80),
            ("QA Station", "Testing and quality assurance", 8, 5, 0.90),
            ("Packaging", "Final packaging and prep", 8, 5, 0.95),
        ]
        for wc in work_centers:
            conn.execute(text(
                "INSERT INTO work_center (name, description, hours_per_day, days_per_week, efficiency_pct) VALUES (:n, :d, :h, :dw, :e)"
            ), {"n": wc[0], "d": wc[1], "h": wc[2], "dw": wc[3], "e": wc[4]})
        
        # --- Routing Steps ---
        routing = [
            (1, "Battery Install", 1, 45, 1),
            (1, "Inverter Mount", 2, 30, 0),
            (1, "HVAC Integration", 3, 55, 1),
            (1, "Wiring Harness", 4, 35, 0),
            (2, "Control Module", 1, 25, 0),
            (2, "Display Install", 2, 15, 0),
            (3, "Electrical Test", 1, 30, 0),
            (3, "Thermal Cycle Test", 2, 45, 1),
            (4, "Final Inspection", 1, 20, 0),
            (4, "Packaging", 2, 15, 0),
        ]
        for r in routing:
            conn.execute(text(
                "INSERT INTO routing_step (work_center_id, step_name, step_order, minutes_per_unit, is_bottleneck) VALUES (:wc, :name, :ord, :mins, :bn)"
            ), {"wc": r[0], "name": r[1], "ord": r[2], "mins": r[3], "bn": r[4]})
        
        # --- Work Center Assignments ---
        assignments = [(7, 1, 0.6), (7, 2, 0.4), (8, 1, 0.5), (8, 2, 0.5), (9, 3, 1.0), (6, 4, 0.3)]
        for a in assignments:
            conn.execute(text(
                "INSERT INTO work_center_assignment (role_id, work_center_id, fraction_of_time) VALUES (:r, :w, :f)"
            ), {"r": a[0], "w": a[1], "f": a[2]})
        
        # --- Covenants ---
        covenants = [
            ("Minimum Cash", "Maintain minimum cash balance at all times", "MIN_CASH", 250000, "GREATER_EQUAL", None),
            ("Runway Floor", "Maintain minimum months of cash runway", "MIN_RUNWAY", 6, "GREATER_EQUAL", None),
            ("Gross Margin", "Maintain minimum gross margin percentage", "MIN_MARGIN", 0.25, "GREATER_EQUAL", 3),
        ]
        for c in covenants:
            conn.execute(text(
                "INSERT INTO covenant_config (name, description, covenant_type, threshold_value, comparison, lookback_months, active) VALUES (:n, :d, :t, :v, :cmp, :lb, 1)"
            ), {"n": c[0], "d": c[1], "t": c[2], "v": c[3], "cmp": c[4], "lb": c[5]})
        
        # --- Fleets ---
        fleets = [
            ("Werner Enterprises", "MEGA_FLEET", "I-80 Corridor", 8000, 280, 8.5, 4.25, 1.1),
            ("Heartland Express", "MEGA_FLEET", "Midwest Hub", 4500, 260, 8.0, 4.50, 1.0),
            ("Southeast Toyota Transport", "MID_FLEET", "Southeast", 850, 250, 7.5, 4.30, 0.9),
            ("Regional Carrier A", "SMALL_FLEET", "Texas Triangle", 120, 240, 7.0, 4.40, 1.0),
            ("Owner Operator Group", "SMALL_FLEET", "Various", 45, 280, 9.0, 4.50, 1.2),
        ]
        for f in fleets:
            conn.execute(text("""
                INSERT INTO fleet (name, fleet_type, primary_lanes, truck_count, nights_on_road_per_year, idle_hours_per_night, diesel_price_assumption, gallons_per_idle_hour)
                VALUES (:n, :t, :lanes, :trucks, :nights, :hrs, :diesel, :gal)
            """), {"n": f[0], "t": f[1], "lanes": f[2], "trucks": f[3], "nights": f[4], "hrs": f[5], "diesel": f[6], "gal": f[7]})
        
        # --- Warranty Policies ---
        for w in [("Standard", 12, "Parts and labor for manufacturing defects"), 
                  ("Extended", 24, "Comprehensive coverage including wear items"),
                  ("Fleet Premium", 36, "Full coverage with expedited service")]:
            conn.execute(text(
                "INSERT INTO warranty_policy (policy_name, duration_months, coverage_notes) VALUES (:n, :m, :c)"
            ), {"n": w[0], "m": w[1], "c": w[2]})
        
        # --- Service Plans ---
        for s in [("Basic Monitoring", 299, 12, "Remote diagnostics and alerts"),
                  ("Pro Service", 599, 12, "Monitoring plus preventive maintenance"),
                  ("Fleet Enterprise", 449, 12, "Volume pricing with dedicated support")]:
            conn.execute(text(
                "INSERT INTO service_plan (name, annual_price, term_months, coverage_notes) VALUES (:n, :p, :m, :c)"
            ), {"n": s[0], "p": s[1], "m": s[2], "c": s[3]})
        
        # --- Initial Inventory (scaled by reorder point from BOM) ---
        # We'll use a query to get the part count since it's dynamic now
        part_count_result = conn.execute(text("SELECT COUNT(*) FROM part_master"))
        part_count = part_count_result.scalar()
        
        for part_id in range(1, part_count + 1):
            # Get the reorder point for this part to set initial inventory
            rop_result = conn.execute(text("SELECT reorder_point FROM part_master WHERE id = :pid"), {"pid": part_id})
            rop = rop_result.scalar() or 50
            initial_qty = rop * 2  # Start with 2x reorder point
            
            conn.execute(text(
                "INSERT INTO inventory_balance (part_id, as_of_date, quantity_on_hand, quantity_reserved) VALUES (:p, '2026-01-01', :qty, 0)"
            ), {"p": part_id, "qty": initial_qty})
        
        # --- Fleet Assignments (first 50 units) ---
        for i in range(1, 51):
            fleet_id = (i % 5) + 1
            in_service = str(date(2026, 1, 1) + timedelta(days=i * 2))
            price = 8500 if i % 4 == 0 else 6375
            conn.execute(text(
                "INSERT INTO unit_fleet_assignment (production_unit_id, fleet_id, in_service_date, purchase_price) VALUES (:u, :f, :d, :p)"
            ), {"u": i, "f": fleet_id, "d": in_service, "p": price})
        
        # --- Sample Warranty Events ---
        # Reference parts: 1=Battery, 2=DC-DC, 3=Screen, 7=Ultracap
        events = [
            (5, "2026-03-15", "Battery cell imbalance", 1, 715.26, 1, 0),
            (12, "2026-04-01", "Display connection fault", 3, 230.00, 1, 1),
            (23, "2026-04-20", "DC-DC thermal shutdown", 2, 775.22, 1, 0),
        ]
        for e in events:
            conn.execute(text("""
                INSERT INTO unit_warranty_event (production_unit_id, event_date, failure_mode, part_id, cost_of_repair, is_in_warranty, is_replaced)
                VALUES (:u, :d, :mode, :part, :cost, :in_warr, :replaced)
            """), {"u": e[0], "d": e[1], "mode": e[2], "part": e[3], "cost": e[4], "in_warr": e[5], "replaced": e[6]})
        
        # --- Service Subscriptions (first 30 units) ---
        for i in range(1, 31):
            plan_id = (i % 3) + 1
            fleet_id = (i % 5) + 1
            conn.execute(text(
                "INSERT INTO unit_service_subscription (production_unit_id, service_plan_id, fleet_id, start_date, status) VALUES (:u, :p, :f, '2026-02-01', 'ACTIVE')"
            ), {"u": i, "p": plan_id, "f": fleet_id})
        
        conn.commit()
        logger.info("✅ IdleX ERP database initialized successfully")
        return True


if __name__ == "__main__":
    run_seed()
