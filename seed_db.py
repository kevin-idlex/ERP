from sqlalchemy import create_engine, text
from datetime import date, timedelta
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_db_engine():
    db_url = os.getenv("DATABASE_URL")
    if db_url:
        if db_url.startswith("postgres://"):
            db_url = db_url.replace("postgres://", "postgresql://", 1)
        return create_engine(db_url)
    return create_engine('sqlite:///idlex.db')


def run_seed():
    engine = get_db_engine()
    db_url = os.getenv("DATABASE_URL")
    is_pg = db_url and "postgres" in db_url
    
    SERIAL_PK = "SERIAL PRIMARY KEY" if is_pg else "INTEGER PRIMARY KEY AUTOINCREMENT"
    JSON_TYPE = "JSONB" if is_pg else "TEXT"
    DATETIME_DEF = "TIMESTAMP DEFAULT CURRENT_TIMESTAMP" if is_pg else "DATETIME DEFAULT CURRENT_TIMESTAMP"
    
    with engine.connect() as conn:
        tables = [
            "audit_log", "external_data_import", "unit_service_subscription", "service_plan",
            "unit_warranty_event", "warranty_policy", "unit_fleet_assignment", "fleet",
            "covenant_config", "work_center_assignment", "routing_step", "work_center",
            "purchase_order_line", "purchase_order_header", "inventory_balance",
            "scenario_cash_timeseries", "scenario_growth_profile", "scenario_header",
            "opex_general_expenses", "opex_staffing_plan", "opex_roles",
            "bom_items", "production_unit", "part_master", "global_config"
        ]
        for t in tables:
            try:
                conn.execute(text(f"DROP TABLE IF EXISTS {t}"))
            except:
                pass
        conn.commit()
        
        conn.execute(text(f"CREATE TABLE global_config (id {SERIAL_PK}, setting_key TEXT UNIQUE NOT NULL, setting_value TEXT, description TEXT)"))
        conn.execute(text(f"CREATE TABLE part_master (id {SERIAL_PK}, sku TEXT UNIQUE NOT NULL, name TEXT NOT NULL, cost REAL NOT NULL, moq INTEGER DEFAULT 1, lead_time INTEGER DEFAULT 0, deposit_pct REAL DEFAULT 0, deposit_days INTEGER DEFAULT 0, balance_days INTEGER DEFAULT 0, reorder_point INTEGER DEFAULT 0, safety_stock INTEGER DEFAULT 0, supplier_name TEXT)"))
        conn.execute(text(f"CREATE TABLE production_unit (id {SERIAL_PK}, serial_number TEXT UNIQUE NOT NULL, build_date TEXT NOT NULL, sales_channel TEXT DEFAULT 'DEALER', status TEXT DEFAULT 'PLANNED', warranty_policy_id INTEGER, notes TEXT)"))
        conn.execute(text(f"CREATE TABLE bom_items (id {SERIAL_PK}, part_id INTEGER NOT NULL, qty_per_unit REAL NOT NULL)"))
        conn.execute(text(f"CREATE TABLE opex_roles (id {SERIAL_PK}, role_name TEXT UNIQUE NOT NULL, annual_salary REAL NOT NULL, department TEXT DEFAULT 'Operations')"))
        conn.execute(text(f"CREATE TABLE opex_staffing_plan (id {SERIAL_PK}, role_id INTEGER NOT NULL, month_date TEXT NOT NULL, headcount REAL NOT NULL, UNIQUE(role_id, month_date))"))
        conn.execute(text(f"CREATE TABLE opex_general_expenses (id {SERIAL_PK}, category TEXT NOT NULL, expense_type TEXT NOT NULL, month_date TEXT NOT NULL, amount REAL NOT NULL)"))
        conn.execute(text(f"CREATE TABLE scenario_header (id {SERIAL_PK}, name TEXT UNIQUE NOT NULL, created_at {DATETIME_DEF}, created_by TEXT DEFAULT 'system', description TEXT, is_plan_of_record INTEGER DEFAULT 0, base_start_cash REAL, base_loc_limit REAL, start_units INTEGER, growth_rate REAL, start_date TEXT, forecast_months INTEGER, total_revenue REAL, total_units INTEGER, min_cash REAL, notes TEXT)"))
        conn.execute(text(f"CREATE TABLE scenario_growth_profile (id {SERIAL_PK}, scenario_id INTEGER NOT NULL, month_number INTEGER NOT NULL, month_date TEXT, monthly_growth_pct REAL, planned_units INTEGER)"))
        conn.execute(text(f"CREATE TABLE scenario_cash_timeseries (id {SERIAL_PK}, scenario_id INTEGER NOT NULL, date TEXT NOT NULL, cash_balance REAL, cumulative_revenue REAL)"))
        conn.execute(text(f"CREATE TABLE inventory_balance (id {SERIAL_PK}, part_id INTEGER NOT NULL, as_of_date TEXT NOT NULL, quantity_on_hand REAL NOT NULL, quantity_reserved REAL DEFAULT 0, UNIQUE(part_id, as_of_date))"))
        conn.execute(text(f"CREATE TABLE purchase_order_header (id {SERIAL_PK}, po_number TEXT UNIQUE NOT NULL, supplier_name TEXT, order_date TEXT NOT NULL, expected_delivery_date TEXT, status TEXT DEFAULT 'PLANNED', total_value REAL DEFAULT 0, notes TEXT)"))
        conn.execute(text(f"CREATE TABLE purchase_order_line (id {SERIAL_PK}, po_id INTEGER NOT NULL, part_id INTEGER NOT NULL, quantity_ordered REAL NOT NULL, quantity_received REAL DEFAULT 0, unit_cost REAL NOT NULL)"))
        conn.execute(text(f"CREATE TABLE work_center (id {SERIAL_PK}, name TEXT UNIQUE NOT NULL, description TEXT, hours_per_day REAL DEFAULT 8, days_per_week INTEGER DEFAULT 5, efficiency_pct REAL DEFAULT 0.85)"))
        conn.execute(text(f"CREATE TABLE routing_step (id {SERIAL_PK}, work_center_id INTEGER NOT NULL, step_name TEXT NOT NULL, step_order INTEGER DEFAULT 1, minutes_per_unit REAL NOT NULL, is_bottleneck INTEGER DEFAULT 0)"))
        conn.execute(text(f"CREATE TABLE work_center_assignment (id {SERIAL_PK}, role_id INTEGER NOT NULL, work_center_id INTEGER NOT NULL, fraction_of_time REAL DEFAULT 1.0)"))
        conn.execute(text(f"CREATE TABLE covenant_config (id {SERIAL_PK}, name TEXT UNIQUE NOT NULL, description TEXT, covenant_type TEXT NOT NULL, threshold_value REAL NOT NULL, comparison TEXT DEFAULT 'GREATER_EQUAL', lookback_months INTEGER, active INTEGER DEFAULT 1)"))
        conn.execute(text(f"CREATE TABLE fleet (id {SERIAL_PK}, name TEXT UNIQUE NOT NULL, fleet_type TEXT DEFAULT 'MID_FLEET', contact_name TEXT, primary_lanes TEXT, truck_count INTEGER, nights_on_road_per_year REAL DEFAULT 250, idle_hours_per_night REAL DEFAULT 8, diesel_price_assumption REAL DEFAULT 4.50, gallons_per_idle_hour REAL DEFAULT 1.0, notes TEXT)"))
        conn.execute(text(f"CREATE TABLE unit_fleet_assignment (id {SERIAL_PK}, production_unit_id INTEGER NOT NULL, fleet_id INTEGER NOT NULL, in_service_date TEXT, truck_identifier TEXT, purchase_price REAL)"))
        conn.execute(text(f"CREATE TABLE warranty_policy (id {SERIAL_PK}, policy_name TEXT UNIQUE NOT NULL, duration_months INTEGER DEFAULT 12, coverage_notes TEXT)"))
        conn.execute(text(f"CREATE TABLE unit_warranty_event (id {SERIAL_PK}, production_unit_id INTEGER NOT NULL, event_date TEXT NOT NULL, failure_mode TEXT, part_id INTEGER, cost_of_repair REAL DEFAULT 0, is_in_warranty INTEGER DEFAULT 1, is_replaced INTEGER DEFAULT 0, resolution_notes TEXT)"))
        conn.execute(text(f"CREATE TABLE service_plan (id {SERIAL_PK}, name TEXT UNIQUE NOT NULL, annual_price REAL NOT NULL, term_months INTEGER DEFAULT 12, coverage_notes TEXT)"))
        conn.execute(text(f"CREATE TABLE unit_service_subscription (id {SERIAL_PK}, production_unit_id INTEGER NOT NULL, service_plan_id INTEGER NOT NULL, fleet_id INTEGER, start_date TEXT NOT NULL, end_date TEXT, status TEXT DEFAULT 'ACTIVE', payment_type TEXT DEFAULT 'ANNUAL')"))
        conn.execute(text(f"CREATE TABLE audit_log (id {SERIAL_PK}, timestamp {DATETIME_DEF}, user_name TEXT DEFAULT 'system', action TEXT NOT NULL, object_type TEXT, object_id TEXT, data_before {JSON_TYPE}, data_after {JSON_TYPE})"))
        conn.execute(text(f"CREATE TABLE external_data_import (id {SERIAL_PK}, source_system TEXT NOT NULL, import_type TEXT NOT NULL, imported_at {DATETIME_DEF}, payload {JSON_TYPE}, processed INTEGER DEFAULT 0, process_notes TEXT)"))
        conn.commit()
        
        # Seed data
        for k, v, d in [("start_cash", "1600000", "Starting cash"), ("loc_limit", "500000", "Credit limit"), ("msrp_price", "8500", "MSRP"), ("dealer_discount", "0.75", "Dealer discount"), ("direct_sales_pct", "0.25", "Direct %"), ("company_name", "IdleX", "Company")]:
            conn.execute(text("INSERT INTO global_config (setting_key, setting_value, description) VALUES (:k, :v, :d)"), {"k": k, "v": v, "d": d})
        
        parts = [("BAT-LFP-48V", "48V LFP Battery Pack", 1850, 50, 112, 0.30, -84, 30, 25, 10, "CATL"), ("INV-3KW", "3kW Inverter", 420, 25, 56, 0.25, -42, 14, 15, 5, "Victron"), ("HVAC-12K", "12K BTU HVAC", 780, 20, 84, 0.25, -63, 21, 10, 5, "Dometic"), ("CTRL-V2", "Control Module", 285, 50, 42, 0, 0, 30, 30, 10, "Custom"), ("WIRE-KIT", "Wiring Kit", 145, 100, 28, 0, 0, 30, 50, 20, "Waytek"), ("ENCL-AL", "Enclosure", 320, 25, 35, 0.20, -21, 14, 15, 5, "80/20"), ("DSPLY-7", "Display", 165, 50, 42, 0, 0, 30, 25, 10, "Newhaven"), ("MISC-HW", "Hardware Kit", 85, 100, 14, 0, 0, 14, 100, 25, "Fastenal")]
        for p in parts:
            conn.execute(text("INSERT INTO part_master (sku, name, cost, moq, lead_time, deposit_pct, deposit_days, balance_days, reorder_point, safety_stock, supplier_name) VALUES (:a,:b,:c,:d,:e,:f,:g,:h,:i,:j,:k)"), {"a":p[0],"b":p[1],"c":p[2],"d":p[3],"e":p[4],"f":p[5],"g":p[6],"h":p[7],"i":p[8],"j":p[9],"k":p[10]})
        
        for pid in range(1, 9):
            conn.execute(text("INSERT INTO bom_items (part_id, qty_per_unit) VALUES (:p, 1)"), {"p": pid})
        
        roles = [("CEO", 250000, "Executive"), ("VP Engineering", 200000, "Engineering"), ("VP Operations", 180000, "Operations"), ("Mechanical Engineer", 120000, "Engineering"), ("Electrical Engineer", 125000, "Engineering"), ("Production Supervisor", 85000, "Operations"), ("Assembler - Senior", 65000, "Operations"), ("Assembler - Junior", 52000, "Operations"), ("Quality Technician", 60000, "Operations"), ("Supply Chain Manager", 95000, "Operations"), ("Finance Manager", 110000, "Finance"), ("Sales Manager", 100000, "Sales"), ("Field Service Tech", 70000, "Service")]
        for r in roles:
            conn.execute(text("INSERT INTO opex_roles (role_name, annual_salary, department) VALUES (:n, :s, :d)"), {"n": r[0], "s": r[1], "d": r[2]})
        
        base_hc = {1:1, 2:1, 3:1, 4:1, 5:1, 6:1, 7:2, 8:3, 9:1, 10:1, 11:1, 12:1, 13:0}
        for m in range(24):
            yr, mo = 2026 + m // 12, (m % 12) + 1
            month_str = f"{yr}-{mo:02d}-01"
            growth = 1 + m * 0.04
            for rid, hc in base_hc.items():
                actual = hc * growth if rid in [7, 8, 9] else hc
                if actual > 0:
                    conn.execute(text("INSERT INTO opex_staffing_plan (role_id, month_date, headcount) VALUES (:r, :m, :h)"), {"r": rid, "m": month_str, "h": round(actual, 1)})
        
        for i in range(200):
            bd = date(2026, 1, 6) + timedelta(days=i // 4)
            while bd.weekday() >= 5:
                bd += timedelta(days=1)
            conn.execute(text("INSERT INTO production_unit (serial_number, build_date, sales_channel, status) VALUES (:sn, :bd, :ch, 'PLANNED')"), {"sn": f"IDX-{i+1:04d}", "bd": str(bd), "ch": 'DIRECT' if i % 4 == 0 else 'DEALER'})
        
        for wc in [("Assembly Line 1", "Primary", 8, 5, 0.85), ("Assembly Line 2", "Secondary", 8, 5, 0.80), ("QA Station", "Testing", 8, 5, 0.90), ("Packaging", "Final", 8, 5, 0.95)]:
            conn.execute(text("INSERT INTO work_center (name, description, hours_per_day, days_per_week, efficiency_pct) VALUES (:a,:b,:c,:d,:e)"), {"a":wc[0],"b":wc[1],"c":wc[2],"d":wc[3],"e":wc[4]})
        
        for r in [(1,"Battery Install",1,45,1),(1,"Inverter Mount",2,30,0),(1,"HVAC Integration",3,55,1),(1,"Wiring",4,35,0),(2,"Control Module",1,25,0),(2,"Display",2,15,0),(3,"Electrical Test",1,30,0),(3,"Thermal Test",2,45,1),(4,"Final Inspection",1,20,0),(4,"Packaging",2,15,0)]:
            conn.execute(text("INSERT INTO routing_step (work_center_id, step_name, step_order, minutes_per_unit, is_bottleneck) VALUES (:a,:b,:c,:d,:e)"), {"a":r[0],"b":r[1],"c":r[2],"d":r[3],"e":r[4]})
        
        for a in [(7,1,0.6),(7,2,0.4),(8,1,0.5),(8,2,0.5),(9,3,1.0),(6,4,0.3)]:
            conn.execute(text("INSERT INTO work_center_assignment (role_id, work_center_id, fraction_of_time) VALUES (:a,:b,:c)"), {"a":a[0],"b":a[1],"c":a[2]})
        
        for c in [("Minimum Cash","Min cash","MIN_CASH",250000,"GREATER_EQUAL",None),("Runway","Min runway","MIN_RUNWAY",6,"GREATER_EQUAL",None),("Gross Margin","Min margin","MIN_MARGIN",0.25,"GREATER_EQUAL",3)]:
            conn.execute(text("INSERT INTO covenant_config (name, description, covenant_type, threshold_value, comparison, lookback_months, active) VALUES (:a,:b,:c,:d,:e,:f,1)"), {"a":c[0],"b":c[1],"c":c[2],"d":c[3],"e":c[4],"f":c[5]})
        
        for f in [("Werner Enterprises","MEGA_FLEET","I-80",8000,280,8.5,4.25,1.1),("Heartland Express","MEGA_FLEET","Midwest",4500,260,8.0,4.50,1.0),("Southeast Toyota Transport","MID_FLEET","Southeast",850,250,7.5,4.30,0.9),("Regional Carrier A","SMALL_FLEET","Texas",120,240,7.0,4.40,1.0),("Owner Operator Group","SMALL_FLEET","Various",45,280,9.0,4.50,1.2)]:
            conn.execute(text("INSERT INTO fleet (name, fleet_type, primary_lanes, truck_count, nights_on_road_per_year, idle_hours_per_night, diesel_price_assumption, gallons_per_idle_hour) VALUES (:a,:b,:c,:d,:e,:f,:g,:h)"), {"a":f[0],"b":f[1],"c":f[2],"d":f[3],"e":f[4],"f":f[5],"g":f[6],"h":f[7]})
        
        for w in [("Standard",12,"Parts & labor"),("Extended",24,"Comprehensive"),("Fleet Premium",36,"Full coverage")]:
            conn.execute(text("INSERT INTO warranty_policy (policy_name, duration_months, coverage_notes) VALUES (:a,:b,:c)"), {"a":w[0],"b":w[1],"c":w[2]})
        
        for s in [("Basic Monitoring",299,12,"Remote diagnostics"),("Pro Service",599,12,"Full maintenance"),("Fleet Enterprise",449,12,"Volume pricing")]:
            conn.execute(text("INSERT INTO service_plan (name, annual_price, term_months, coverage_notes) VALUES (:a,:b,:c,:d)"), {"a":s[0],"b":s[1],"c":s[2],"d":s[3]})
        
        for pid in range(1, 9):
            conn.execute(text("INSERT INTO inventory_balance (part_id, as_of_date, quantity_on_hand, quantity_reserved) VALUES (:p, '2026-01-01', 150, 0)"), {"p": pid})
        
        for i in range(1, 51):
            conn.execute(text("INSERT INTO unit_fleet_assignment (production_unit_id, fleet_id, in_service_date, purchase_price) VALUES (:u, :f, :d, :p)"), {"u": i, "f": (i%5)+1, "d": str(date(2026,1,1)+timedelta(days=i*2)), "p": 8500 if i%4==0 else 6375})
        
        for e in [(5,"2026-03-15","Inverter fault",2,420,1,0),(12,"2026-04-01","Display issue",7,165,1,1),(23,"2026-04-20","Wiring short",5,145,1,0)]:
            conn.execute(text("INSERT INTO unit_warranty_event (production_unit_id, event_date, failure_mode, part_id, cost_of_repair, is_in_warranty, is_replaced) VALUES (:a,:b,:c,:d,:e,:f,:g)"), {"a":e[0],"b":e[1],"c":e[2],"d":e[3],"e":e[4],"f":e[5],"g":e[6]})
        
        for i in range(1, 31):
            conn.execute(text("INSERT INTO unit_service_subscription (production_unit_id, service_plan_id, fleet_id, start_date, status) VALUES (:u, :p, :f, '2026-02-01', 'ACTIVE')"), {"u": i, "p": (i%3)+1, "f": (i%5)+1})
        
        conn.commit()
        logger.info("IdleX ERP database initialized")


if __name__ == "__main__":
    run_seed()
