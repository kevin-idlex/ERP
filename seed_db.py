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
import random

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
        
        # --- Pricing Configuration (by Year) ---
        conn.execute(text(f"""
            CREATE TABLE pricing_config (
                id {SERIAL_PK},
                year INTEGER UNIQUE NOT NULL,
                msrp REAL NOT NULL,
                dealer_discount_pct REAL NOT NULL,
                notes TEXT
            )
        """))
        
        # --- Channel Mix Configuration (by Quarter) ---
        conn.execute(text(f"""
            CREATE TABLE channel_mix_config (
                id {SERIAL_PK},
                year INTEGER NOT NULL,
                quarter INTEGER NOT NULL,
                direct_pct REAL NOT NULL,
                notes TEXT,
                UNIQUE(year, quarter)
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
        
        # --- Staffing Plan (36 months - Outsourced Manufacturing Model) ---
        # IdleX uses 100% outsourced manufacturing and sells through dealers
        # Internal team is lean: Product Development, Supplier Management, Sales
        #
        # Staffing scales with REVENUE, not production volume
        # Revenue milestones: $15M (2026), $85M (2027), $216M (2028)
        
        # Monthly revenue for scaling calculations
        monthly_revenue = {
            "2026-01": 207000, "2026-02": 276000, "2026-03": 380000, "2026-04": 428000,
            "2026-05": 552000, "2026-06": 877000, "2026-07": 822000, "2026-08": 1394000,
            "2026-09": 1753000, "2026-10": 2016000, "2026-11": 3127000, "2026-12": 3265000,
            "2027-01": 3672000, "2027-02": 4059000, "2027-03": 5502000, "2027-04": 4915000,
            "2027-05": 6813000, "2027-06": 6040000, "2027-07": 6599000, "2027-08": 9091000,
            "2027-09": 7953000, "2027-10": 8567000, "2027-11": 11632000, "2027-12": 10218000,
            "2028-01": 14472000, "2028-02": 12289000, "2028-03": 12966000, "2028-04": 13863000,
            "2028-05": 18055000, "2028-06": 15738000, "2028-07": 21046000, "2028-08": 18099000,
            "2028-09": 19410000, "2028-10": 25907000, "2028-11": 22207000, "2028-12": 22366000,
        }
        
        for month_offset in range(36):
            year = 2026 + month_offset // 12
            month = (month_offset % 12) + 1
            month_key = f"{year}-{month:02d}"
            month_str = f"{year}-{month:02d}-01"
            
            revenue = monthly_revenue.get(month_key, 1000000)
            annual_run_rate = revenue * 12
            
            # Cumulative revenue for installed base calculations
            cumulative_rev = sum(v for k, v in monthly_revenue.items() if k <= month_key)
            units_in_field = int(cumulative_rev / 6900)  # Approximate units deployed
            
            # Lean staffing model - scales with revenue milestones
            # Base team through $20M ARR, then add incrementally
            
            headcount_by_role = {
                1: 1,     # CEO - always 1
                2: 1,     # VP Engineering - always 1
                3: 1 if annual_run_rate < 50_000_000 else 2,  # VP Operations
                4: 1 + (annual_run_rate // 50_000_000),       # Mechanical Engineer
                5: 1 + (annual_run_rate // 50_000_000),       # Electrical Engineer
                6: 0,     # Production Supervisor - OUTSOURCED
                7: 0,     # Assembler Senior - OUTSOURCED
                8: 0,     # Assembler Junior - OUTSOURCED
                9: max(0, units_in_field // 2000),            # Quality Technician (field issues)
                10: 1 + (annual_run_rate // 100_000_000),     # Supply Chain Manager
                11: 1,    # Finance Manager - always 1
                12: 1 + (annual_run_rate // 30_000_000),      # Sales Manager (dealer relations)
                13: max(0, units_in_field // 1500),           # Field Service Tech
            }
            
            for role_id, headcount in headcount_by_role.items():
                if headcount > 0:
                    conn.execute(text(
                        "INSERT INTO opex_staffing_plan (role_id, month_date, headcount) VALUES (:rid, :dt, :hc)"
                    ), {"rid": role_id, "dt": month_str, "hc": round(headcount, 1)})
        
        # --- Production Units - Actual IdleX 3-Year Production Plan ---
        # Monthly targets derived from weekly production schedule
        # Units distributed across workdays (Mon-Fri) with 25% direct / 75% dealer split
        
        monthly_production_plan = [
            # 2026 - Ramp Year
            ("2026-01-01", 30),    # Jan 2026 - Production start
            ("2026-02-01", 40),    # Feb 2026
            ("2026-03-01", 55),    # Mar 2026
            ("2026-04-01", 62),    # Apr 2026
            ("2026-05-01", 80),    # May 2026
            ("2026-06-01", 127),   # Jun 2026
            ("2026-07-01", 119),   # Jul 2026
            ("2026-08-01", 202),   # Aug 2026
            ("2026-09-01", 254),   # Sep 2026
            ("2026-10-01", 292),   # Oct 2026
            ("2026-11-01", 453),   # Nov 2026
            ("2026-12-01", 473),   # Dec 2026 (Year 1 Total: 2,187)
            
            # 2027 - Growth Year
            ("2027-01-01", 532),   # Jan 2027
            ("2027-02-01", 588),   # Feb 2027
            ("2027-03-01", 797),   # Mar 2027
            ("2027-04-01", 712),   # Apr 2027
            ("2027-05-01", 987),   # May 2027
            ("2027-06-01", 875),   # Jun 2027
            ("2027-07-01", 956),   # Jul 2027
            ("2027-08-01", 1317),  # Aug 2027
            ("2027-09-01", 1152),  # Sep 2027
            ("2027-10-01", 1241),  # Oct 2027
            ("2027-11-01", 1685),  # Nov 2027
            ("2027-12-01", 1480),  # Dec 2027 (Year 2 Total: 11,322)
            
            # 2028 - Scale Year
            ("2028-01-01", 2096),  # Jan 2028
            ("2028-02-01", 1780),  # Feb 2028
            ("2028-03-01", 1878),  # Mar 2028
            ("2028-04-01", 2008),  # Apr 2028
            ("2028-05-01", 2616),  # May 2028
            ("2028-06-01", 2280),  # Jun 2028
            ("2028-07-01", 3049),  # Jul 2028
            ("2028-08-01", 2622),  # Aug 2028
            ("2028-09-01", 2812),  # Sep 2028
            ("2028-10-01", 3753),  # Oct 2028
            ("2028-11-01", 3217),  # Nov 2028
            ("2028-12-01", 3240),  # Dec 2028 (Year 3 Total: 31,351)
        ]
        # Grand Total: 44,860 units over 36 months
        
        import calendar
        
        def get_workdays_for_month(year, month):
            """Get list of workdays (Mon-Fri) in a month."""
            num_days = calendar.monthrange(year, month)[1]
            workdays = []
            for day in range(1, num_days + 1):
                dt = date(year, month, day)
                if dt.weekday() < 5:  # Monday = 0, Friday = 4
                    workdays.append(dt)
            return workdays
        
        serial_num = 1
        
        for month_str, target_units in monthly_production_plan:
            if target_units == 0:
                continue
            
            # Parse month
            year = int(month_str[:4])
            month = int(month_str[5:7])
            
            # Get workdays for this month
            workdays = get_workdays_for_month(year, month)
            if not workdays:
                continue
            
            # Calculate units per day (distribute evenly)
            units_per_day = target_units / len(workdays)
            
            # Track remaining units for rounding
            remaining = target_units
            day_idx = 0
            
            # Group workdays by quarter for channel mix lookup
            # Channel mix is defined per quarter in channel_mix_config
            channel_mix_defaults = {
                # 2026 - Heavy dealer reliance during launch
                (2026, 1): 0.15, (2026, 2): 0.20, (2026, 3): 0.25, (2026, 4): 0.25,
                # 2027 - Gradual shift to direct
                (2027, 1): 0.28, (2027, 2): 0.30, (2027, 3): 0.32, (2027, 4): 0.35,
                # 2028 - Strong direct presence
                (2028, 1): 0.38, (2028, 2): 0.40, (2028, 3): 0.42, (2028, 4): 0.45,
            }
            
            # Distribute units across workdays with CORRECT channel mix per quarter
            random.seed(42)  # Reproducible randomness
            
            for i in range(target_units):
                build_date = workdays[i % len(workdays)]
                
                # Determine quarter for this build date
                build_quarter = (build_date.month - 1) // 3 + 1
                
                # Get direct percentage for this quarter (default 25% if not found)
                direct_pct = channel_mix_defaults.get((year, build_quarter), 0.25)
                
                # Randomly assign channel based on configured percentage
                channel = 'DIRECT' if random.random() < direct_pct else 'DEALER'
                
                conn.execute(text("""
                    INSERT INTO production_unit (serial_number, build_date, sales_channel, status)
                    VALUES (:sn, :bd, :ch, 'PLANNED')
                """), {
                    "sn": f"IDX-{serial_num:05d}",
                    "bd": str(build_date),
                    "ch": channel
                })
                serial_num += 1
        
        logger.info(f"Created {serial_num - 1:,} production units")
        
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
        
        # --- Fleet Assignments (first 500 units as sample deployments) ---
        # In reality, this would grow as units ship
        for i in range(1, 501):
            fleet_id = (i % 5) + 1
            # Stagger in-service dates across first few months
            days_offset = (i - 1) * 2
            in_service = str(date(2026, 1, 15) + timedelta(days=days_offset))
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
        
        # --- Service Subscriptions (first 300 units as sample) ---
        for i in range(1, 301):
            plan_id = (i % 3) + 1
            fleet_id = (i % 5) + 1
            conn.execute(text(
                "INSERT INTO unit_service_subscription (production_unit_id, service_plan_id, fleet_id, start_date, status) VALUES (:u, :p, :f, '2026-02-01', 'ACTIVE')"
            ), {"u": i, "p": plan_id, "f": fleet_id})
        
        # --- General Expenses (R&D and SG&A) - Actual IdleX Budget ---
        # These expenses are based on the 2026 operating budget
        # R&D front-loaded for product development, SG&A scales with revenue
        
        general_expenses = []
        
        # === R&D MATERIALS AND OUTSOURCING (2026) ===
        # Core system development
        general_expenses.extend([
            ("R&D - Core System", "R&D", "2026-01-01", 50000),
            ("R&D - Core System", "R&D", "2026-02-01", 40000),
            ("R&D - Core System", "R&D", "2026-03-01", 30000),
            ("R&D - Core System", "R&D", "2026-04-01", 5000),
            ("R&D - Core System", "R&D", "2026-05-01", 5000),
            ("R&D - Core System", "R&D", "2026-06-01", 5000),
            ("R&D - Core System", "R&D", "2026-07-01", 5000),
        ])
        
        # HVAC prototype and validation
        general_expenses.extend([
            ("R&D - HVAC Prototype", "R&D", "2026-01-01", 50000),
            ("R&D - HVAC Prototype", "R&D", "2026-02-01", 25000),
            ("R&D - HVAC Prototype", "R&D", "2026-03-01", 10000),
            ("R&D - HVAC Prototype", "R&D", "2026-04-01", 5000),
            ("R&D - HVAC Prototype", "R&D", "2026-05-01", 5000),
            ("R&D - HVAC Prototype", "R&D", "2026-06-01", 5000),
            ("R&D - HVAC Prototype", "R&D", "2026-07-01", 5000),
        ])
        
        # Enclosures & Harnesses (Feb-Jul)
        for m in range(2, 8):
            general_expenses.append(("R&D - Enclosures & Harnesses", "R&D", f"2026-{m:02d}-01", 5000))
        
        # Software & Controls (Feb-Jul)
        for m in range(2, 8):
            general_expenses.append(("R&D - Software & Controls", "R&D", f"2026-{m:02d}-01", 5000))
        
        # Test equipment and lab gear (Feb-Jul)
        for m in range(2, 8):
            general_expenses.append(("R&D - Test Equipment", "R&D", f"2026-{m:02d}-01", 5000))
        
        # Outsourced engineering (Feb-Jul)
        for m in range(2, 8):
            general_expenses.append(("R&D - Outsourced Engineering", "R&D", f"2026-{m:02d}-01", 5000))
        
        # Thermal and certification testing (Apr-Aug)
        for m in range(4, 9):
            general_expenses.append(("R&D - Certification Testing", "R&D", f"2026-{m:02d}-01", 5000))
        
        # === SG&A (All 36 months, scales with revenue) ===
        # Revenue scaling factors by year
        sga_scale = {2026: 1.0, 2027: 1.5, 2028: 2.5}
        
        for year in [2026, 2027, 2028]:
            scale = sga_scale[year]
            
            for month in range(1, 13):
                month_str = f"{year}-{month:02d}-01"
                
                # Office, cloud tools, insurance - $10K/mo base, scales with company
                general_expenses.append(("Office & Cloud Tools", "SG&A", month_str, int(10000 * scale)))
                
                # Marketing - $15K/mo base, scales with revenue
                general_expenses.append(("Marketing", "SG&A", month_str, int(15000 * scale)))
                
                # Travel - $10K/mo base, scales with sales activity
                general_expenses.append(("Travel", "SG&A", month_str, int(10000 * scale)))
                
                # Legal and accounting - front-loaded in Jan/Feb, then steady
                if year == 2026:
                    if month == 1:
                        legal_amt = 10000
                    elif month == 2:
                        legal_amt = 5000
                    else:
                        legal_amt = 1500
                else:
                    legal_amt = int(2500 * scale)  # Scales in later years
                
                general_expenses.append(("Legal & Accounting", "SG&A", month_str, legal_amt))
        
        # Insert all general expenses
        for category, exp_type, month_date, amount in general_expenses:
            conn.execute(text("""
                INSERT INTO opex_general_expenses (category, expense_type, month_date, amount)
                VALUES (:cat, :type, :dt, :amt)
            """), {"cat": category, "type": exp_type, "dt": month_date, "amt": amount})
        
        # --- Pricing Configuration (by Year) ---
        # MSRP and dealer discount can change over time as product matures
        pricing_by_year = [
            (2026, 8500.00, 0.75, "Launch pricing - 25% dealer margin"),
            (2027, 8500.00, 0.75, "Maintain pricing through growth"),
            (2028, 8750.00, 0.77, "Price increase with premium features"),
        ]
        
        for year, msrp, dealer_disc, notes in pricing_by_year:
            conn.execute(text("""
                INSERT INTO pricing_config (year, msrp, dealer_discount_pct, notes)
                VALUES (:y, :m, :d, :n)
            """), {"y": year, "m": msrp, "d": dealer_disc, "n": notes})
        
        # --- Channel Mix Configuration (by Quarter) ---
        # Direct vs Dealer split - shifts toward direct as brand builds
        channel_mix = [
            # 2026 - Heavy dealer reliance during launch
            (2026, 1, 0.15, "Q1 2026 - Launch via dealer network"),
            (2026, 2, 0.20, "Q2 2026 - Building direct presence"),
            (2026, 3, 0.25, "Q3 2026 - Direct sales growing"),
            (2026, 4, 0.25, "Q4 2026 - Balanced mix"),
            # 2027 - Gradual shift to direct
            (2027, 1, 0.28, "Q1 2027"),
            (2027, 2, 0.30, "Q2 2027"),
            (2027, 3, 0.32, "Q3 2027"),
            (2027, 4, 0.35, "Q4 2027"),
            # 2028 - Strong direct presence
            (2028, 1, 0.38, "Q1 2028"),
            (2028, 2, 0.40, "Q2 2028"),
            (2028, 3, 0.42, "Q3 2028"),
            (2028, 4, 0.45, "Q4 2028 - Target mix achieved"),
        ]
        
        for year, quarter, direct_pct, notes in channel_mix:
            conn.execute(text("""
                INSERT INTO channel_mix_config (year, quarter, direct_pct, notes)
                VALUES (:y, :q, :d, :n)
            """), {"y": year, "q": quarter, "d": direct_pct, "n": notes})
        
        conn.commit()
        logger.info("✅ IdleX ERP database initialized successfully")
        return True


if __name__ == "__main__":
    run_seed()
