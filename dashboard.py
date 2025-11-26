"""
IdleX CFO Console - Enterprise Edition
Version: 5.0
Complete Feature Set:
  1. Scenario Library & Versioning
  2. Inventory & Purchase Planner
  3. Capacity & Throughput Planner
  4. Covenant & Runway Monitor
  5. Fleet ROI & Unit Economics
  6. Warranty & RMA Tracking
  7. Service & Recurring Revenue
  8. Board Pack Generator
  9. Audit Log
  10. External Integration Hooks
"""

import streamlit as st
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from datetime import datetime, timedelta, date
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import calendar
import os
import logging
import json
import io

# Import seeder
import seed_db

# =============================================================================
# CONFIGURATION
# =============================================================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MSRP_PRICE = 8500.00
DEALER_DISCOUNT_RATE = 0.75
DIRECT_SALES_PCT = 0.25
DEALER_PAYMENT_LAG = 30
OPTIMIZER_ITERATIONS = 15

# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(page_title="IdleX CFO Console", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    .block-container { padding: 0.5rem 1rem; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { padding: 8px 16px; }
    
    .metric-card { background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); 
                   border-radius: 10px; padding: 15px; margin: 5px 0; }
    .status-ok { color: #10B981; font-weight: bold; }
    .status-warn { color: #F59E0B; font-weight: bold; }
    .status-danger { color: #EF4444; font-weight: bold; }
    
    .financial-table { font-family: Georgia, serif; font-size: 13px; width: 100%; 
                       border-collapse: collapse; background: white; color: black; }
    .financial-table th { text-align: right; border-bottom: 2px solid black; padding: 6px; }
    .financial-table td { padding: 5px 6px; }
    .financial-table .row-header { text-align: left; min-width: 140px; }
    .financial-table .section-header { font-weight: bold; text-decoration: underline; }
    .financial-table .total-row { font-weight: bold; border-top: 1px solid black; }
    .financial-table .grand-total { font-weight: bold; border-top: 1px solid black; border-bottom: 3px double black; }
    .financial-table .indent { padding-left: 15px; }
    
    @media (max-width: 768px) {
        .block-container { padding: 0.5rem; }
        h1 { font-size: 1.4rem !important; }
        .stMetric label { font-size: 0.7rem !important; }
        .stMetric [data-testid="stMetricValue"] { font-size: 1.1rem !important; }
        .stButton button { min-height: 44px; font-size: 0.9rem; }
        .stNumberInput input { font-size: 16px !important; }
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# DATABASE
# =============================================================================
@st.cache_resource
def get_engine():
    url = os.getenv("DATABASE_URL")
    if url:
        if url.startswith("postgres://"):
            url = url.replace("postgres://", "postgresql://", 1)
        return create_engine(url)
    return create_engine('sqlite:///idlex.db')

def get_db_type():
    url = os.getenv("DATABASE_URL")
    return "postgresql" if url and "postgres" in url else "sqlite"

engine = get_engine()
DB_TYPE = get_db_type()

# =============================================================================
# AUDIT LOGGING
# =============================================================================
def audit_log(action, obj_type=None, obj_id=None, before=None, after=None):
    try:
        with engine.connect() as conn:
            conn.execute(text("""
                INSERT INTO audit_log (user_name, action, object_type, object_id, data_before, data_after)
                VALUES (:u, :a, :t, :i, :b, :af)
            """), {"u": "system", "a": action, "t": obj_type, "i": str(obj_id) if obj_id else None,
                   "b": json.dumps(before) if before else None, "af": json.dumps(after) if after else None})
            conn.commit()
    except Exception as e:
        logger.warning(f"Audit failed: {e}")

# =============================================================================
# UTILITIES
# =============================================================================
def get_workdays(year, month, threshold=None):
    days = [date(year, month, d) for d in range(1, calendar.monthrange(year, month)[1] + 1)]
    valid = [d for d in days if d.weekday() < 5]
    return [d for d in valid if d >= threshold] if threshold else valid

def fmt_currency(v, compact=False):
    if pd.isna(v) or v is None: return ""
    if compact:
        if abs(v) >= 1e6: return f"${v/1e6:.1f}M"
        if abs(v) >= 1e3: return f"${v/1e3:.0f}K"
    return f"(${abs(v):,.0f})" if v < 0 else f"${v:,.0f}"

def fmt_pct(v):
    return f"{v*100:.1f}%" if v else "N/A"

def fmt_banker(v):
    if pd.isna(v) or v is None or v == "": return ""
    if isinstance(v, str): return v
    return f"({abs(v):,.0f})" if v < 0 else f"{v:,.0f}"

def load_config():
    try:
        cfg = pd.read_sql("SELECT setting_key, setting_value FROM global_config", engine)
        return dict(zip(cfg['setting_key'], cfg['setting_value']))
    except:
        return {}

# =============================================================================
# FINANCIAL ENGINE
# =============================================================================
def generate_financials(units_df=None, start_cash_override=None, include_service=True):
    """Generate P&L and Cash Flow with optional service revenue."""
    try:
        parts = pd.read_sql("SELECT * FROM part_master", engine)
        bom = pd.read_sql("SELECT * FROM bom_items", engine)
        staffing = pd.read_sql("SELECT * FROM opex_staffing_plan", engine)
        roles = pd.read_sql("SELECT * FROM opex_roles", engine)
        
        try:
            expenses = pd.read_sql("SELECT * FROM opex_general_expenses", engine)
            expenses['month_date'] = pd.to_datetime(expenses['month_date'])
        except:
            expenses = pd.DataFrame()
        
        if units_df is not None:
            units = units_df.copy()
        else:
            units = pd.read_sql("SELECT * FROM production_unit", engine)
        
        cfg = load_config()
        start_cash = float(start_cash_override) if start_cash_override else float(cfg.get('start_cash', 1600000))
        
    except Exception as e:
        logger.error(f"Load error: {e}")
        return pd.DataFrame(), pd.DataFrame()
    
    if units.empty:
        return pd.DataFrame(), pd.DataFrame()
    
    units['build_date'] = pd.to_datetime(units['build_date'])
    staffing['month_date'] = pd.to_datetime(staffing['month_date'])
    
    ledger = []
    
    # Unit material cost
    mat_cost = 0
    if not bom.empty and not parts.empty:
        merged = pd.merge(bom, parts, left_on='part_id', right_on='id')
        mat_cost = (merged['qty_per_unit'] * merged['cost']).sum()
    
    # Revenue & COGS
    for _, u in units.iterrows():
        direct = u['sales_channel'] == 'DIRECT'
        rev = MSRP_PRICE if direct else MSRP_PRICE * DEALER_DISCOUNT_RATE
        dt = u['build_date']
        lag = 0 if direct else DEALER_PAYMENT_LAG
        
        ledger.append({"Date": dt, "Category": "Product Sales", "Type": "Revenue", "Amount": rev, "Report": "PnL"})
        ledger.append({"Date": dt + timedelta(days=lag), "Category": "Customer Collections", "Type": "Ops", "Amount": rev, "Report": "Cash"})
        ledger.append({"Date": dt, "Category": "Materials", "Type": "COGS", "Amount": -mat_cost, "Report": "PnL"})
    
    # Supply chain cash
    monthly = units.groupby(pd.Grouper(key='build_date', freq='MS')).size()
    for mo, cnt in monthly.items():
        if cnt == 0: continue
        for _, p in parts.iterrows():
            b = bom[bom['part_id'] == p['id']]
            if b.empty: continue
            cost = b.iloc[0]['qty_per_unit'] * cnt * p['cost']
            if p['deposit_pct'] > 0:
                ledger.append({"Date": mo + timedelta(days=int(p['deposit_days'])), "Category": "Supplier Deposits", "Type": "Ops", "Amount": -cost * p['deposit_pct'], "Report": "Cash"})
            if p['deposit_pct'] < 1:
                ledger.append({"Date": mo + timedelta(days=int(p['balance_days'])), "Category": "Supplier Payments", "Type": "Ops", "Amount": -cost * (1 - p['deposit_pct']), "Report": "Cash"})
    
    # Payroll
    if not staffing.empty and not roles.empty:
        merged = pd.merge(staffing, roles, left_on='role_id', right_on='id')
        for _, r in merged.iterrows():
            cost = (r['annual_salary'] / 12) * r['headcount']
            if cost > 0:
                labor = "Assembler" in r['role_name']
                cat = "Direct Labor" if labor else "Salaries & Wages"
                typ = "COGS" if labor else "OpEx"
                ledger.append({"Date": r['month_date'], "Category": cat, "Type": typ, "Amount": -cost, "Report": "PnL"})
                ledger.append({"Date": r['month_date'], "Category": "Payroll", "Type": "Ops", "Amount": -cost, "Report": "Cash"})
    
    # Expenses
    if not expenses.empty:
        for _, e in expenses.iterrows():
            if e['amount'] > 0:
                ledger.append({"Date": e['month_date'], "Category": e['category'], "Type": "OpEx", "Amount": -e['amount'], "Report": "PnL"})
                ledger.append({"Date": e['month_date'], "Category": "OpEx Payments", "Type": "Ops", "Amount": -e['amount'], "Report": "Cash"})
    
    # Service Revenue
    if include_service:
        try:
            subs = pd.read_sql("""
                SELECT s.*, p.annual_price, p.term_months 
                FROM unit_service_subscription s 
                JOIN service_plan p ON s.service_plan_id = p.id 
                WHERE s.status = 'ACTIVE'
            """, engine)
            if not subs.empty:
                subs['start_date'] = pd.to_datetime(subs['start_date'])
                for _, sub in subs.iterrows():
                    monthly_rev = sub['annual_price'] / 12
                    start = sub['start_date']
                    for m in range(sub['term_months']):
                        rev_date = start + timedelta(days=30*m)
                        ledger.append({"Date": rev_date, "Category": "Service Revenue", "Type": "Revenue", "Amount": monthly_rev, "Report": "PnL"})
                        ledger.append({"Date": rev_date, "Category": "Service Collections", "Type": "Ops", "Amount": monthly_rev, "Report": "Cash"})
        except:
            pass
    
    # Warranty costs
    try:
        warranty = pd.read_sql("SELECT * FROM unit_warranty_event", engine)
        if not warranty.empty:
            warranty['event_date'] = pd.to_datetime(warranty['event_date'])
            for _, w in warranty.iterrows():
                if w['cost_of_repair'] > 0:
                    ledger.append({"Date": w['event_date'], "Category": "Warranty Expense", "Type": "OpEx", "Amount": -w['cost_of_repair'], "Report": "PnL"})
                    ledger.append({"Date": w['event_date'], "Category": "Warranty Payments", "Type": "Ops", "Amount": -w['cost_of_repair'], "Report": "Cash"})
    except:
        pass
    
    if not ledger:
        return pd.DataFrame(), pd.DataFrame()
    
    df = pd.DataFrame(ledger)
    pnl = df[df['Report'] == 'PnL'].sort_values('Date')
    cash = df[df['Report'] == 'Cash'].sort_values('Date')
    cash['Cash_Balance'] = cash['Amount'].cumsum() + start_cash
    
    return pnl, cash

# =============================================================================
# SIMULATION ENGINE
# =============================================================================
def simulate_scenario(start_units, growth_pct, start_date, months):
    """Generate simulated production units."""
    units = []
    current = start_units
    sn = 1
    dt = start_date.replace(day=1)
    
    for _ in range(months):
        target = int(current)
        days = get_workdays(dt.year, dt.month)
        if target > 0 and days:
            direct = int(target * DIRECT_SALES_PCT)
            pool = ['DIRECT'] * direct + ['DEALER'] * (target - direct)
            for i, ch in enumerate(pool):
                units.append({
                    "serial_number": f"SIM-{sn:05d}",
                    "build_date": days[i % len(days)],
                    "sales_channel": ch,
                    "status": "PLANNED"
                })
                sn += 1
        current *= (1 + growth_pct / 100)
        dt = date(dt.year + (1 if dt.month == 12 else 0), 1 if dt.month == 12 else dt.month + 1, 1)
    
    return pd.DataFrame(units)

def optimize_growth(start_units, start_cash, loc_limit, start_date, months):
    """Binary search for max sustainable growth."""
    best = {'rate': 0, 'cash_df': pd.DataFrame(), 'units_df': pd.DataFrame()}
    low, high = 0.0, 100.0
    
    for _ in range(OPTIMIZER_ITERATIONS):
        mid = (low + high) / 2
        sim = simulate_scenario(start_units, mid, start_date, months)
        _, cash = generate_financials(units_df=sim, start_cash_override=start_cash, include_service=False)
        
        min_cash = cash['Cash_Balance'].min() if not cash.empty else 0
        if min_cash >= -loc_limit:
            best = {'rate': mid, 'cash_df': cash, 'units_df': sim}
            low = mid
        else:
            high = mid
    
    # Add summary
    if not best['units_df'].empty:
        best['units_df']['build_date'] = pd.to_datetime(best['units_df']['build_date'])
        monthly = best['units_df'].groupby(best['units_df']['build_date'].dt.to_period('M')).size().reset_index()
        monthly.columns = ['Month', 'Units']
        monthly['Month'] = monthly['Month'].astype(str)
        best['monthly'] = monthly
    
    pnl, _ = generate_financials(units_df=best['units_df'], start_cash_override=start_cash, include_service=False) if not best['units_df'].empty else (pd.DataFrame(), None)
    best['total_revenue'] = pnl[pnl['Type'] == 'Revenue']['Amount'].sum() if not pnl.empty else 0
    best['total_units'] = len(best['units_df'])
    best['min_cash'] = best['cash_df']['Cash_Balance'].min() if not best['cash_df'].empty else 0
    
    return best

# =============================================================================
# COVENANT ENGINE
# =============================================================================
def evaluate_covenants(pnl, cash):
    """Evaluate all active covenants."""
    try:
        covs = pd.read_sql("SELECT * FROM covenant_config WHERE active = 1", engine)
    except:
        return []
    
    if covs.empty or cash.empty:
        return []
    
    cfg = load_config()
    loc_limit = float(cfg.get('loc_limit', 500000))
    
    results = []
    for _, c in covs.iterrows():
        val, status = None, "OK"
        
        if c['covenant_type'] == 'MIN_CASH':
            val = cash['Cash_Balance'].min()
            status = "BREACH" if val < c['threshold_value'] else ("WARNING" if val < c['threshold_value'] * 1.25 else "OK")
        
        elif c['covenant_type'] == 'MIN_RUNWAY':
            burn = cash['Amount'].mean()
            end = cash.iloc[-1]['Cash_Balance']
            val = abs(end / burn) if burn < 0 else 99
            status = "BREACH" if val < c['threshold_value'] else ("WARNING" if val < c['threshold_value'] * 1.5 else "OK")
        
        elif c['covenant_type'] == 'MIN_MARGIN':
            if not pnl.empty:
                rev = pnl[pnl['Type'] == 'Revenue']['Amount'].sum()
                cogs = abs(pnl[pnl['Type'] == 'COGS']['Amount'].sum())
                val = (rev - cogs) / rev if rev > 0 else 0
                status = "BREACH" if val < c['threshold_value'] else ("WARNING" if val < c['threshold_value'] * 1.1 else "OK")
        
        elif c['covenant_type'] == 'MAX_LOC_UTIL':
            min_cash = cash['Cash_Balance'].min()
            util = max(0, -min_cash) / loc_limit if loc_limit > 0 else 0
            val = util
            status = "BREACH" if util > c['threshold_value'] else ("WARNING" if util > c['threshold_value'] * 0.9 else "OK")
        
        results.append({'name': c['name'], 'type': c['covenant_type'], 'threshold': c['threshold_value'],
                       'comparison': c['comparison'], 'current': val, 'status': status})
    
    return results

# =============================================================================
# CAPACITY ENGINE
# =============================================================================
def calculate_capacity(start_date, months):
    """Calculate monthly capacity based on headcount and routing."""
    try:
        wcs = pd.read_sql("SELECT * FROM work_center", engine)
        routing = pd.read_sql("SELECT * FROM routing_step", engine)
        assigns = pd.read_sql("SELECT * FROM work_center_assignment", engine)
        staffing = pd.read_sql("SELECT * FROM opex_staffing_plan", engine)
    except:
        return pd.DataFrame()
    
    if wcs.empty or routing.empty:
        return pd.DataFrame()
    
    bottlenecks = routing[routing['is_bottleneck'] == 1] if 'is_bottleneck' in routing.columns else routing
    results = []
    dt = start_date.replace(day=1)
    
    for _ in range(months):
        workdays = len(get_workdays(dt.year, dt.month))
        month_staff = staffing[pd.to_datetime(staffing['month_date']).dt.to_period('M') == pd.Period(dt, 'M')]
        
        min_cap = float('inf')
        limiting = None
        
        for _, step in bottlenecks.iterrows():
            wc = wcs[wcs['id'] == step['work_center_id']]
            if wc.empty: continue
            wc = wc.iloc[0]
            
            # Get labor hours
            wc_assigns = assigns[assigns['work_center_id'] == step['work_center_id']]
            labor_hrs = 0
            for _, a in wc_assigns.iterrows():
                staff = month_staff[month_staff['role_id'] == a['role_id']]
                if not staff.empty:
                    hc = staff['headcount'].sum()
                    labor_hrs += hc * a['fraction_of_time'] * wc['hours_per_day'] * workdays
            
            if labor_hrs == 0:
                labor_hrs = wc['hours_per_day'] * workdays
            
            avail_mins = labor_hrs * 60 * wc['efficiency_pct']
            cap = avail_mins / step['minutes_per_unit'] if step['minutes_per_unit'] > 0 else float('inf')
            
            if cap < min_cap:
                min_cap = cap
                limiting = step['step_name']
        
        results.append({
            'Month': dt.strftime('%Y-%m'),
            'Capacity': int(min_cap) if min_cap != float('inf') else 0,
            'Bottleneck': limiting or 'N/A',
            'Workdays': workdays
        })
        
        dt = date(dt.year + (1 if dt.month == 12 else 0), 1 if dt.month == 12 else dt.month + 1, 1)
    
    return pd.DataFrame(results)

# =============================================================================
# INVENTORY & PO ENGINE
# =============================================================================
def calculate_suggested_pos(months_ahead=6):
    """Calculate suggested POs based on production and inventory."""
    try:
        units = pd.read_sql("SELECT * FROM production_unit WHERE status = 'PLANNED'", engine)
        parts = pd.read_sql("SELECT * FROM part_master", engine)
        bom = pd.read_sql("SELECT * FROM bom_items", engine)
        inv = pd.read_sql("SELECT * FROM inventory_balance", engine)
    except:
        return pd.DataFrame()
    
    if units.empty or parts.empty:
        return pd.DataFrame()
    
    units['build_date'] = pd.to_datetime(units['build_date'])
    
    # Current inventory
    current_inv = {}
    if not inv.empty:
        latest = inv.sort_values('as_of_date').groupby('part_id').last()
        current_inv = latest['quantity_on_hand'].to_dict()
    
    monthly = units.groupby(units['build_date'].dt.to_period('M')).size()
    suggestions = []
    
    for _, p in parts.iterrows():
        b = bom[bom['part_id'] == p['id']]
        if b.empty: continue
        
        qty_per = b.iloc[0]['qty_per_unit']
        on_hand = current_inv.get(p['id'], 0)
        running = on_hand
        
        for period, cnt in monthly.head(months_ahead).items():
            required = qty_per * cnt
            running -= required
            
            if running < p.get('reorder_point', 0):
                order_qty = max(p['moq'], -running + p.get('safety_stock', 0))
                order_qty = ((order_qty // p['moq']) + 1) * p['moq']
                order_date = period.to_timestamp().date() - timedelta(days=p['lead_time'])
                
                suggestions.append({
                    'Part': p['name'], 'SKU': p['sku'], 'Supplier': p.get('supplier_name', ''),
                    'Month_Needed': str(period), 'Required': int(required),
                    'On_Hand': int(max(0, running + required)), 'Order_Qty': int(order_qty),
                    'Order_By': order_date, 'Lead_Days': p['lead_time'],
                    'Unit_Cost': p['cost'], 'PO_Value': order_qty * p['cost']
                })
                running += order_qty
    
    return pd.DataFrame(suggestions)

# =============================================================================
# FLEET ROI ENGINE
# =============================================================================
def calculate_fleet_roi():
    """Calculate ROI metrics by fleet."""
    try:
        fleets = pd.read_sql("SELECT * FROM fleet", engine)
        assigns = pd.read_sql("""
            SELECT a.*, u.serial_number, u.build_date 
            FROM unit_fleet_assignment a 
            JOIN production_unit u ON a.production_unit_id = u.id
        """, engine)
    except:
        return pd.DataFrame(), pd.DataFrame()
    
    if fleets.empty:
        return fleets, pd.DataFrame()
    
    results = []
    for _, f in fleets.iterrows():
        fleet_units = assigns[assigns['fleet_id'] == f['id']]
        unit_count = len(fleet_units)
        
        # Annual savings per unit
        annual_idle_cost = f['nights_on_road_per_year'] * f['idle_hours_per_night'] * f['gallons_per_idle_hour'] * f['diesel_price_assumption']
        
        # Average purchase price
        avg_price = fleet_units['purchase_price'].mean() if not fleet_units.empty and 'purchase_price' in fleet_units.columns else MSRP_PRICE * DEALER_DISCOUNT_RATE
        
        # Payback
        payback_months = (avg_price / annual_idle_cost * 12) if annual_idle_cost > 0 else 999
        
        # 5-year ROI
        five_yr_savings = annual_idle_cost * 5
        roi_5yr = (five_yr_savings - avg_price) / avg_price if avg_price > 0 else 0
        
        results.append({
            'Fleet': f['name'], 'Type': f['fleet_type'], 'Trucks': f['truck_count'],
            'Units_Deployed': unit_count, 'Annual_Savings': annual_idle_cost,
            'Avg_Price': avg_price, 'Payback_Months': payback_months,
            'ROI_5yr': roi_5yr, 'Fleet_ID': f['id']
        })
    
    return fleets, pd.DataFrame(results)

# =============================================================================
# SCENARIO LIBRARY
# =============================================================================
def save_scenario(name, desc, results, inputs):
    """Save scenario to library."""
    with engine.connect() as conn:
        try:
            start_dt = inputs['start_date']
            if hasattr(start_dt, 'isoformat'):
                start_dt = start_dt.isoformat()
            
            conn.execute(text("""
                INSERT INTO scenario_header (name, description, base_start_cash, base_loc_limit,
                    start_units, growth_rate, start_date, forecast_months, total_revenue, total_units, min_cash)
                VALUES (:n, :d, :c, :l, :u, :r, :s, :m, :rev, :tu, :mc)
            """), {
                "n": name, "d": desc, "c": inputs['start_cash'], "l": inputs['loc_limit'],
                "u": inputs['start_units'], "r": results['rate'], "s": start_dt,
                "m": inputs['months'], "rev": results['total_revenue'], "tu": results['total_units'],
                "mc": results['min_cash']
            })
            
            # Get ID
            if DB_TYPE == "postgresql":
                sid = conn.execute(text("SELECT id FROM scenario_header WHERE name = :n"), {"n": name}).scalar()
            else:
                sid = conn.execute(text("SELECT last_insert_rowid()")).scalar()
            
            # Save monthly
            if 'monthly' in results:
                for i, row in results['monthly'].iterrows():
                    conn.execute(text("""
                        INSERT INTO scenario_growth_profile (scenario_id, month_number, monthly_growth_pct, planned_units)
                        VALUES (:s, :m, :g, :u)
                    """), {"s": sid, "m": i+1, "g": results['rate'], "u": row['Units']})
            
            # Save cash
            if not results['cash_df'].empty:
                for _, row in results['cash_df'].iterrows():
                    dt = row['Date']
                    if hasattr(dt, 'date'):
                        dt = dt.date()
                    if hasattr(dt, 'isoformat'):
                        dt = dt.isoformat()
                    conn.execute(text("""
                        INSERT INTO scenario_cash_timeseries (scenario_id, date, cash_balance)
                        VALUES (:s, :d, :b)
                    """), {"s": sid, "d": dt, "b": row['Cash_Balance']})
            
            conn.commit()
            audit_log("SCENARIO_SAVED", "scenario_header", sid, None, {"name": name})
            return True, "Saved"
        except Exception as e:
            conn.rollback()
            return False, str(e)

def push_to_production(units_df):
    """Push scenario to production plan."""
    if units_df.empty:
        return False, "No units", 0
    
    with engine.connect() as conn:
        try:
            result = conn.execute(text("DELETE FROM production_unit WHERE status = 'PLANNED'"))
            deleted = result.rowcount
            
            last = conn.execute(text("SELECT serial_number FROM production_unit ORDER BY id DESC LIMIT 1")).scalar()
            next_sn = int(''.join(filter(str.isdigit, last or '0'))) + 1
            
            for _, row in units_df.iterrows():
                bd = row['build_date']
                if isinstance(bd, pd.Timestamp): bd = bd.date()
                if hasattr(bd, 'isoformat'): bd = bd.isoformat()
                conn.execute(text("""
                    INSERT INTO production_unit (serial_number, build_date, sales_channel, status)
                    VALUES (:s, :d, :c, 'PLANNED')
                """), {"s": f"IDX-{next_sn:04d}", "d": bd, "c": row['sales_channel']})
                next_sn += 1
            
            conn.commit()
            audit_log("SCENARIO_PUSHED", "production_unit", None, {"deleted": deleted}, {"inserted": len(units_df)})
            return True, f"Replaced {deleted} with {len(units_df)} units", len(units_df)
        except Exception as e:
            conn.rollback()
            return False, str(e), 0

# =============================================================================
# RENDER HELPERS
# =============================================================================
def render_financial_table(df, title):
    html = f"<h4>{title}</h4><div style='overflow-x:auto;'><table class='financial-table'>"
    html += "<thead><tr><th class='row-header'>Account</th>"
    for col in df.columns:
        html += f"<th>{col}</th>"
    html += "</tr></thead><tbody>"
    
    sections = ['Revenue', 'Cost of Goods Sold', 'Operating Expenses', 'Operating Activities']
    totals = ['Gross Profit', 'Net Cash Flow', 'Total OpEx']
    grands = ['Net Income', 'Ending Cash']
    
    for idx, row in df.iterrows():
        s = str(idx).strip()
        is_sec = s in sections
        cls = "section-header" if is_sec else ("total-row" if s in totals else ("grand-total" if s in grands else "indent"))
        
        html += f"<tr class='{cls}'><td class='row-header'>{s}</td>"
        if is_sec:
            html += "<td></td>" * len(df.columns)
        else:
            for col in df.columns:
                html += f"<td style='text-align:right;'>{fmt_banker(row[col])}</td>"
        html += "</tr>"
    
    html += "</tbody></table></div>"
    st.markdown(html, unsafe_allow_html=True)

def render_covenant_card(cov):
    icon = "✅" if cov['status'] == "OK" else ("⚠️" if cov['status'] == "WARNING" else "❌")
    color = "#10B981" if cov['status'] == "OK" else ("#F59E0B" if cov['status'] == "WARNING" else "#EF4444")
    
    if cov['type'] in ['MIN_MARGIN', 'MAX_LOC_UTIL']:
        curr = fmt_pct(cov['current']) if cov['current'] else "N/A"
        thresh = fmt_pct(cov['threshold'])
    elif cov['type'] == 'MIN_RUNWAY':
        curr = f"{cov['current']:.0f} mo" if cov['current'] else "N/A"
        thresh = f"{cov['threshold']:.0f} mo"
    else:
        curr = fmt_currency(cov['current'], True) if cov['current'] else "N/A"
        thresh = fmt_currency(cov['threshold'], True)
    
    st.markdown(f"**{cov['name']}**")
    st.markdown(f"Current: **{curr}** | Target: {thresh}")
    st.markdown(f"<span style='color:{color};font-weight:bold;'>{icon} {cov['status']}</span>", unsafe_allow_html=True)

# =============================================================================
# MAIN APPLICATION
# =============================================================================
def main():
    # Sidebar
    st.sidebar.title("IdleX CFO Console")
    
    if st.sidebar.button("⚠️ Reset Database"):
        with st.spinner("Initializing..."):
            seed_db.run_seed()
        st.success("Database initialized!")
        st.rerun()
    
    st.sidebar.divider()
    
    views = [
        "📊 Dashboard",
        "🚀 Scenario Planner",
        "📚 Scenario Library",
        "📦 Inventory & POs",
        "🏭 Capacity",
        "🚛 Fleet ROI",
        "🛡️ Warranty",
        "💳 Service Revenue",
        "📈 Financials",
        "📋 Board Pack",
        "📝 Audit Log",
        "🔌 Integrations"
    ]
    
    view = st.sidebar.radio("Navigation", views)
    
    # Generate base financials
    pnl, cash = pd.DataFrame(), pd.DataFrame()
    if view not in ["🚀 Scenario Planner", "📚 Scenario Library"]:
        pnl, cash = generate_financials()
    
    cfg = load_config()
    
    # =========================================================================
    # DASHBOARD
    # =========================================================================
    if view == "📊 Dashboard":
        st.title("Executive Dashboard")
        
        if not pnl.empty:
            # Top metrics
            rev = pnl[pnl['Type'] == 'Revenue']['Amount'].sum()
            cogs = abs(pnl[pnl['Type'] == 'COGS']['Amount'].sum())
            margin = rev - cogs
            margin_pct = margin / rev if rev > 0 else 0
            min_cash = cash['Cash_Balance'].min() if not cash.empty else 0
            end_cash = cash.iloc[-1]['Cash_Balance'] if not cash.empty else 0
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Revenue", fmt_currency(rev, True))
            c2.metric("Gross Margin", f"{fmt_currency(margin, True)} ({margin_pct:.0%})")
            c3.metric("Min Cash", fmt_currency(min_cash, True))
            c4.metric("Ending Cash", fmt_currency(end_cash, True))
            
            # Cash chart
            fig = px.area(cash, x='Date', y='Cash_Balance', title="Cash Forecast", color_discrete_sequence=['#10B981'])
            loc = float(cfg.get('loc_limit', 500000))
            fig.add_hline(y=-loc, line_dash="dash", line_color="red", annotation_text="Credit Limit")
            fig.add_hline(y=0, line_dash="dot", line_color="gray")
            fig.update_layout(height=280, margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig, use_container_width=True)
            
            # Covenants
            st.subheader("🛡️ Covenant Monitor")
            covs = evaluate_covenants(pnl, cash)
            if covs:
                cols = st.columns(len(covs))
                for i, c in enumerate(covs):
                    with cols[i]:
                        render_covenant_card(c)
            
            # Runway
            burn = cash['Amount'].mean() if not cash.empty else 0
            runway = abs(end_cash / burn) if burn < 0 else 99
            st.metric("📅 Runway", f"{runway:.0f} months")
            
            # Service Revenue summary
            try:
                subs = pd.read_sql("SELECT COUNT(*) as cnt FROM unit_service_subscription WHERE status = 'ACTIVE'", engine)
                svc_rev = pnl[pnl['Category'] == 'Service Revenue']['Amount'].sum()
                c1, c2 = st.columns(2)
                c1.metric("Active Subscriptions", int(subs.iloc[0]['cnt']))
                c2.metric("Service Revenue", fmt_currency(svc_rev, True))
            except:
                pass
        else:
            st.info("No data. Click 'Reset Database' in sidebar.")
    
    # =========================================================================
    # SCENARIO PLANNER
    # =========================================================================
    elif view == "🚀 Scenario Planner":
        st.title("Growth Scenario Planner")
        
        def_cash = float(cfg.get('start_cash', 1600000))
        def_loc = float(cfg.get('loc_limit', 500000))
        
        if 'scenario_results' not in st.session_state:
            st.session_state.scenario_results = None
        
        st.subheader("① Constraints")
        c1, c2 = st.columns(2)
        with c1:
            inv_cash = st.number_input("Investor Equity ($)", value=def_cash, step=100000.0, format="%.0f")
            start_vol = st.number_input("Starting Units/Month", value=50, min_value=1)
        with c2:
            loc_limit = st.number_input("Credit Limit ($)", value=def_loc, step=100000.0, format="%.0f")
            months = st.slider("Forecast Months", 12, 60, 36)
        
        start_dt = st.date_input("Start Date", value=date(2026, 1, 1))
        
        if st.button("🔍 Optimize Growth Rate", type="primary", use_container_width=True):
            with st.spinner("Optimizing..."):
                results = optimize_growth(start_vol, inv_cash, loc_limit, start_dt, months)
                st.session_state.scenario_results = results
                st.session_state.scenario_inputs = {
                    'start_units': start_vol, 'start_cash': inv_cash, 'loc_limit': loc_limit,
                    'start_date': start_dt, 'months': months
                }
            st.rerun()
        
        if st.session_state.scenario_results:
            res = st.session_state.scenario_results
            inp = st.session_state.scenario_inputs
            
            st.divider()
            st.subheader("② Results")
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Max Growth", f"{res['rate']:.1f}%/mo")
            c2.metric("Total Units", f"{res['total_units']:,}")
            c3.metric("Total Revenue", fmt_currency(res['total_revenue'], True))
            c4.metric("Min Cash", fmt_currency(res['min_cash'], True))
            
            if not res['cash_df'].empty:
                fig = px.area(res['cash_df'], x='Date', y='Cash_Balance', 
                             title=f"Cash Flow @ {res['rate']:.1f}%/mo Growth", color_discrete_sequence=['#10B981'])
                fig.add_hline(y=-inp['loc_limit'], line_dash="dash", line_color="red")
                fig.update_layout(height=280)
                st.plotly_chart(fig, use_container_width=True)
                
                with st.expander("📅 Monthly Plan"):
                    if 'monthly' in res:
                        st.dataframe(res['monthly'], hide_index=True)
            
            st.divider()
            st.subheader("③ Save & Deploy")
            
            c1, c2 = st.columns([2, 1])
            with c1:
                name = st.text_input("Scenario Name", f"Scenario_{datetime.now().strftime('%Y%m%d_%H%M')}")
                desc = st.text_area("Description", height=60)
            with c2:
                if st.button("💾 Save to Library", use_container_width=True):
                    ok, msg = save_scenario(name, desc, res, inp)
                    if ok: st.success("Saved!")
                    else: st.error(msg)
                
                if st.button("🚀 Push to Production", type="primary", use_container_width=True):
                    ok, msg, cnt = push_to_production(res['units_df'])
                    if ok:
                        st.success(msg)
                        st.balloons()
                    else:
                        st.error(msg)
    
    # =========================================================================
    # SCENARIO LIBRARY
    # =========================================================================
    elif view == "📚 Scenario Library":
        st.title("Scenario Library")
        
        try:
            scenarios = pd.read_sql("SELECT * FROM scenario_header ORDER BY created_at DESC", engine)
        except:
            scenarios = pd.DataFrame()
        
        if scenarios.empty:
            st.info("No scenarios saved yet.")
        else:
            for _, sc in scenarios.iterrows():
                star = "⭐ " if sc['is_plan_of_record'] else ""
                with st.expander(f"{star}{sc['name']} - {fmt_currency(sc['total_revenue'], True)}", expanded=False):
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Growth", f"{sc['growth_rate']:.1f}%/mo")
                    c2.metric("Units", f"{sc['total_units']:,}")
                    c3.metric("Revenue", fmt_currency(sc['total_revenue'], True))
                    c4.metric("Min Cash", fmt_currency(sc['min_cash'], True))
                    
                    st.caption(sc['description'] or "No description")
                    
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        if not sc['is_plan_of_record']:
                            if st.button("⭐ Set as Plan of Record", key=f"por_{sc['id']}"):
                                with engine.connect() as conn:
                                    conn.execute(text("UPDATE scenario_header SET is_plan_of_record = 0"))
                                    conn.execute(text("UPDATE scenario_header SET is_plan_of_record = 1 WHERE id = :id"), {"id": sc['id']})
                                    conn.commit()
                                st.rerun()
                    with c3:
                        if st.button("🗑️ Delete", key=f"del_{sc['id']}"):
                            with engine.connect() as conn:
                                conn.execute(text("DELETE FROM scenario_header WHERE id = :id"), {"id": sc['id']})
                                conn.commit()
                            st.rerun()
            
            # Compare
            if len(scenarios) >= 2:
                st.divider()
                st.subheader("Compare Scenarios")
                c1, c2 = st.columns(2)
                with c1:
                    sc1 = st.selectbox("Scenario A", scenarios['name'].tolist())
                with c2:
                    sc2 = st.selectbox("Scenario B", scenarios['name'].tolist(), index=min(1, len(scenarios)-1))
                
                if st.button("📊 Compare"):
                    id1 = scenarios[scenarios['name'] == sc1]['id'].values[0]
                    id2 = scenarios[scenarios['name'] == sc2]['id'].values[0]
                    
                    c1 = pd.read_sql(text("SELECT * FROM scenario_cash_timeseries WHERE scenario_id = :id"), engine, params={"id": id1})
                    c2_df = pd.read_sql(text("SELECT * FROM scenario_cash_timeseries WHERE scenario_id = :id"), engine, params={"id": id2})
                    
                    fig = go.Figure()
                    if not c1.empty:
                        fig.add_trace(go.Scatter(x=c1['date'], y=c1['cash_balance'], name=sc1, mode='lines'))
                    if not c2_df.empty:
                        fig.add_trace(go.Scatter(x=c2_df['date'], y=c2_df['cash_balance'], name=sc2, mode='lines'))
                    fig.update_layout(title="Cash Comparison", height=350)
                    st.plotly_chart(fig, use_container_width=True)
    
    # =========================================================================
    # INVENTORY & POS
    # =========================================================================
    elif view == "📦 Inventory & POs":
        st.title("Inventory & Purchase Orders")
        
        tab1, tab2, tab3 = st.tabs(["Inventory", "Suggested POs", "Open POs"])
        
        with tab1:
            try:
                inv = pd.read_sql("""
                    SELECT p.sku, p.name, p.supplier_name, p.cost, p.moq, p.lead_time,
                           COALESCE(i.quantity_on_hand, 0) as on_hand, p.reorder_point, p.safety_stock
                    FROM part_master p
                    LEFT JOIN (SELECT part_id, quantity_on_hand FROM inventory_balance 
                               WHERE as_of_date = (SELECT MAX(as_of_date) FROM inventory_balance)) i
                    ON p.id = i.part_id
                """, engine)
                
                inv['Status'] = inv.apply(lambda r: '🔴 LOW' if r['on_hand'] < r['reorder_point'] 
                                          else ('🟡 OK' if r['on_hand'] < r['safety_stock']*2 else '🟢 Good'), axis=1)
                st.dataframe(inv, hide_index=True, use_container_width=True)
                
                # Total inventory value
                inv['Value'] = inv['on_hand'] * inv['cost']
                st.metric("Total Inventory Value", fmt_currency(inv['Value'].sum(), True))
            except Exception as e:
                st.error(f"Error: {e}")
        
        with tab2:
            months = st.slider("Months Ahead", 3, 12, 6)
            pos = calculate_suggested_pos(months)
            
            if not pos.empty:
                st.dataframe(pos, hide_index=True, use_container_width=True)
                
                total = pos['PO_Value'].sum()
                st.metric("Total PO Value", fmt_currency(total))
                
                by_month = pos.groupby('Month_Needed')['PO_Value'].sum().reset_index()
                fig = px.bar(by_month, x='Month_Needed', y='PO_Value', title="PO Value by Month")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No POs needed.")
        
        with tab3:
            try:
                open_pos = pd.read_sql("""
                    SELECT po_number, supplier_name, order_date, expected_delivery_date, status, total_value
                    FROM purchase_order_header WHERE status NOT IN ('RECEIVED_FULL', 'CANCELLED')
                """, engine)
                if not open_pos.empty:
                    st.dataframe(open_pos, hide_index=True, use_container_width=True)
                else:
                    st.info("No open POs.")
            except:
                st.info("No PO data.")
    
    # =========================================================================
    # CAPACITY
    # =========================================================================
    elif view == "🏭 Capacity":
        st.title("Capacity Planner")
        
        tab1, tab2 = st.tabs(["Capacity vs Plan", "Configuration"])
        
        with tab1:
            try:
                units = pd.read_sql("SELECT * FROM production_unit WHERE status = 'PLANNED'", engine)
                units['build_date'] = pd.to_datetime(units['build_date'])
                planned = units.groupby(units['build_date'].dt.to_period('M')).size().reset_index()
                planned.columns = ['Month', 'Planned']
                planned['Month'] = planned['Month'].astype(str)
            except:
                planned = pd.DataFrame(columns=['Month', 'Planned'])
            
            capacity = calculate_capacity(date(2026, 1, 1), 24)
            
            if not capacity.empty and not planned.empty:
                merged = pd.merge(capacity, planned, on='Month', how='outer').fillna(0)
                merged['Buffer'] = merged['Capacity'] - merged['Planned']
                merged['Status'] = merged['Buffer'].apply(lambda x: '🟢' if x >= 0 else '🔴')
                
                st.dataframe(merged[['Month', 'Planned', 'Capacity', 'Buffer', 'Status', 'Bottleneck']], 
                            hide_index=True, use_container_width=True)
                
                fig = go.Figure()
                fig.add_trace(go.Bar(x=merged['Month'], y=merged['Planned'], name='Planned'))
                fig.add_trace(go.Scatter(x=merged['Month'], y=merged['Capacity'], name='Capacity', mode='lines+markers'))
                fig.update_layout(title="Planned vs Capacity", height=350)
                st.plotly_chart(fig, use_container_width=True)
                
                shortfall = merged[merged['Buffer'] < 0]
                if not shortfall.empty:
                    st.error(f"⚠️ Capacity shortfall in {len(shortfall)} months!")
                else:
                    st.success("✅ Capacity meets plan")
            else:
                st.info("Configure work centers to enable.")
        
        with tab2:
            st.subheader("Work Centers")
            try:
                wcs = pd.read_sql("SELECT * FROM work_center", engine)
                st.dataframe(wcs, hide_index=True, use_container_width=True)
            except:
                pass
            
            st.subheader("Routing")
            try:
                routing = pd.read_sql("""
                    SELECT r.*, w.name as work_center FROM routing_step r 
                    JOIN work_center w ON r.work_center_id = w.id ORDER BY r.work_center_id, r.step_order
                """, engine)
                st.dataframe(routing, hide_index=True, use_container_width=True)
            except:
                pass
    
    # =========================================================================
    # FLEET ROI
    # =========================================================================
    elif view == "🚛 Fleet ROI":
        st.title("Fleet Unit Economics")
        
        tab1, tab2, tab3 = st.tabs(["ROI Summary", "Fleet Config", "Unit Assignments"])
        
        with tab1:
            fleets, roi = calculate_fleet_roi()
            
            if not roi.empty:
                st.dataframe(roi[['Fleet', 'Type', 'Units_Deployed', 'Annual_Savings', 'Payback_Months', 'ROI_5yr']].style.format({
                    'Annual_Savings': '${:,.0f}',
                    'Payback_Months': '{:.1f}',
                    'ROI_5yr': '{:.1%}'
                }), hide_index=True, use_container_width=True)
                
                # Charts
                c1, c2 = st.columns(2)
                with c1:
                    fig = px.bar(roi, x='Fleet', y='Payback_Months', title="Payback Period (Months)", color='Type')
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
                with c2:
                    fig = px.bar(roi, x='Fleet', y='ROI_5yr', title="5-Year ROI", color='Type')
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Summary
                total_units = roi['Units_Deployed'].sum()
                avg_payback = roi['Payback_Months'].mean()
                avg_roi = roi['ROI_5yr'].mean()
                
                c1, c2, c3 = st.columns(3)
                c1.metric("Total Deployed", int(total_units))
                c2.metric("Avg Payback", f"{avg_payback:.1f} mo")
                c3.metric("Avg 5yr ROI", f"{avg_roi:.0%}")
        
        with tab2:
            try:
                fleets_df = pd.read_sql("SELECT * FROM fleet", engine)
                edited = st.data_editor(fleets_df, hide_index=True, use_container_width=True, disabled=['id'])
                
                if st.button("💾 Save Fleet Config"):
                    with engine.connect() as conn:
                        for _, r in edited.iterrows():
                            conn.execute(text("""
                                UPDATE fleet SET name=:n, fleet_type=:t, nights_on_road_per_year=:nights,
                                idle_hours_per_night=:hrs, diesel_price_assumption=:diesel, gallons_per_idle_hour=:gal
                                WHERE id=:id
                            """), {"n": r['name'], "t": r['fleet_type'], "nights": r['nights_on_road_per_year'],
                                   "hrs": r['idle_hours_per_night'], "diesel": r['diesel_price_assumption'],
                                   "gal": r['gallons_per_idle_hour'], "id": r['id']})
                        conn.commit()
                    st.success("Saved!")
                    st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")
        
        with tab3:
            try:
                assigns = pd.read_sql("""
                    SELECT a.*, u.serial_number, f.name as fleet_name 
                    FROM unit_fleet_assignment a
                    JOIN production_unit u ON a.production_unit_id = u.id
                    JOIN fleet f ON a.fleet_id = f.id
                """, engine)
                st.dataframe(assigns[['serial_number', 'fleet_name', 'in_service_date', 'purchase_price']], 
                            hide_index=True, use_container_width=True)
            except:
                st.info("No assignments.")
    
    # =========================================================================
    # WARRANTY
    # =========================================================================
    elif view == "🛡️ Warranty":
        st.title("Warranty & Quality")
        
        tab1, tab2, tab3 = st.tabs(["Events", "Analytics", "Policies"])
        
        with tab1:
            try:
                events = pd.read_sql("""
                    SELECT e.*, u.serial_number, p.name as part_name
                    FROM unit_warranty_event e
                    JOIN production_unit u ON e.production_unit_id = u.id
                    LEFT JOIN part_master p ON e.part_id = p.id
                    ORDER BY e.event_date DESC
                """, engine)
                
                if not events.empty:
                    st.dataframe(events[['serial_number', 'event_date', 'failure_mode', 'part_name', 'cost_of_repair', 'is_in_warranty']],
                                hide_index=True, use_container_width=True)
                    
                    total_cost = events['cost_of_repair'].sum()
                    in_warranty = events[events['is_in_warranty'] == 1]['cost_of_repair'].sum()
                    
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Total Events", len(events))
                    c2.metric("Total Cost", fmt_currency(total_cost))
                    c3.metric("Warranty Cost", fmt_currency(in_warranty))
                else:
                    st.info("No warranty events recorded.")
            except Exception as e:
                st.error(f"Error: {e}")
        
        with tab2:
            try:
                events = pd.read_sql("SELECT * FROM unit_warranty_event", engine)
                units = pd.read_sql("SELECT COUNT(*) as cnt FROM production_unit", engine)
                
                if not events.empty:
                    events['event_date'] = pd.to_datetime(events['event_date'])
                    by_month = events.groupby(events['event_date'].dt.to_period('M')).agg({
                        'id': 'count', 'cost_of_repair': 'sum'
                    }).reset_index()
                    by_month.columns = ['Month', 'Events', 'Cost']
                    by_month['Month'] = by_month['Month'].astype(str)
                    
                    fig = make_subplots(specs=[[{"secondary_y": True}]])
                    fig.add_trace(go.Bar(x=by_month['Month'], y=by_month['Events'], name='Events'))
                    fig.add_trace(go.Scatter(x=by_month['Month'], y=by_month['Cost'], name='Cost', mode='lines'), secondary_y=True)
                    fig.update_layout(title="Warranty Trends", height=300)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Failure rate
                    total_units = units.iloc[0]['cnt']
                    failure_rate = len(events) / total_units * 100 if total_units > 0 else 0
                    st.metric("Failure Rate", f"{failure_rate:.1f}%")
            except:
                st.info("No data for analytics.")
        
        with tab3:
            try:
                policies = pd.read_sql("SELECT * FROM warranty_policy", engine)
                st.dataframe(policies, hide_index=True, use_container_width=True)
            except:
                st.info("No policies configured.")
    
    # =========================================================================
    # SERVICE REVENUE
    # =========================================================================
    elif view == "💳 Service Revenue":
        st.title("Service & Recurring Revenue")
        
        tab1, tab2, tab3 = st.tabs(["Subscriptions", "Revenue Forecast", "Plans"])
        
        with tab1:
            try:
                subs = pd.read_sql("""
                    SELECT s.*, u.serial_number, p.name as plan_name, p.annual_price, f.name as fleet_name
                    FROM unit_service_subscription s
                    JOIN production_unit u ON s.production_unit_id = u.id
                    JOIN service_plan p ON s.service_plan_id = p.id
                    LEFT JOIN fleet f ON s.fleet_id = f.id
                    ORDER BY s.start_date DESC
                """, engine)
                
                if not subs.empty:
                    active = subs[subs['status'] == 'ACTIVE']
                    arr = active['annual_price'].sum()
                    
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Active Subscriptions", len(active))
                    c2.metric("ARR", fmt_currency(arr))
                    c3.metric("Avg Contract Value", fmt_currency(active['annual_price'].mean()))
                    
                    st.dataframe(subs[['serial_number', 'fleet_name', 'plan_name', 'annual_price', 'start_date', 'status']],
                                hide_index=True, use_container_width=True)
                else:
                    st.info("No subscriptions.")
            except Exception as e:
                st.error(f"Error: {e}")
        
        with tab2:
            try:
                subs = pd.read_sql("""
                    SELECT s.start_date, p.annual_price, p.term_months
                    FROM unit_service_subscription s
                    JOIN service_plan p ON s.service_plan_id = p.id
                    WHERE s.status = 'ACTIVE'
                """, engine)
                
                if not subs.empty:
                    subs['start_date'] = pd.to_datetime(subs['start_date'])
                    
                    # Generate monthly revenue forecast
                    forecast = []
                    for _, s in subs.iterrows():
                        monthly = s['annual_price'] / 12
                        for m in range(int(s['term_months'])):
                            dt = s['start_date'] + timedelta(days=30*m)
                            forecast.append({'Month': dt.strftime('%Y-%m'), 'Revenue': monthly})
                    
                    if forecast:
                        fc_df = pd.DataFrame(forecast)
                        by_month = fc_df.groupby('Month')['Revenue'].sum().reset_index()
                        
                        fig = px.bar(by_month, x='Month', y='Revenue', title="Monthly Recurring Revenue Forecast")
                        fig.update_layout(height=350)
                        st.plotly_chart(fig, use_container_width=True)
            except:
                st.info("No data for forecast.")
        
        with tab3:
            try:
                plans = pd.read_sql("SELECT * FROM service_plan", engine)
                st.dataframe(plans, hide_index=True, use_container_width=True)
            except:
                st.info("No plans configured.")
    
    # =========================================================================
    # FINANCIALS
    # =========================================================================
    elif view == "📈 Financials":
        st.title("Financial Statements")
        
        if not pnl.empty:
            freq = st.radio("Period", ["Monthly", "Quarterly", "Yearly"], horizontal=True, index=1)
            fmap = {"Monthly": "ME", "Quarterly": "QE", "Yearly": "YE"}
            
            # P&L
            st.header("Income Statement")
            agg = pnl.groupby([pd.Grouper(key='Date', freq=fmap[freq]), 'Type', 'Category']).sum()['Amount'].unstack(level=[1, 2]).fillna(0)
            
            if freq == "Monthly": agg.index = agg.index.strftime('%Y-%b')
            elif freq == "Quarterly": agg.index = agg.index.to_period("Q").astype(str)
            else: agg.index = agg.index.strftime('%Y')
            
            stmt = pd.DataFrame(columns=agg.index)
            
            def ssum(keys):
                t = pd.Series(0.0, index=agg.index)
                for k in keys:
                    if k in agg.columns: t += agg[k]
                return t
            
            stmt.loc['Revenue'] = None
            stmt.loc['Product Sales'] = ssum([('Revenue', 'Product Sales')])
            stmt.loc['Service Revenue'] = ssum([('Revenue', 'Service Revenue')])
            stmt.loc['Cost of Goods Sold'] = None
            stmt.loc['Materials'] = ssum([('COGS', 'Materials')])
            stmt.loc['Direct Labor'] = ssum([('COGS', 'Direct Labor')])
            stmt.loc['Gross Profit'] = stmt.loc['Product Sales'] + stmt.loc['Service Revenue'] + stmt.loc['Materials'] + stmt.loc['Direct Labor']
            stmt.loc['Operating Expenses'] = None
            stmt.loc['Salaries & Wages'] = ssum([('OpEx', 'Salaries & Wages')])
            stmt.loc['Warranty Expense'] = ssum([('OpEx', 'Warranty Expense')])
            
            other_opex = [c for c in agg.columns if c[0] == 'OpEx' and c[1] not in ['Salaries & Wages', 'Warranty Expense']]
            for c in other_opex:
                stmt.loc[c[1]] = ssum([c])
            
            stmt.loc['Total OpEx'] = ssum([('OpEx', c[1]) for c in other_opex]) + stmt.loc['Salaries & Wages'] + stmt.loc['Warranty Expense']
            stmt.loc['Net Income'] = stmt.loc['Gross Profit'] + stmt.loc['Total OpEx']
            
            render_financial_table(stmt, "")
            
            # Cash Flow
            st.header("Cash Flow Statement")
            cash_agg = cash.groupby([pd.Grouper(key='Date', freq=fmap[freq]), 'Category']).sum()['Amount'].unstack().fillna(0)
            
            if freq == "Monthly": cash_agg.index = cash_agg.index.strftime('%Y-%b')
            elif freq == "Quarterly": cash_agg.index = cash_agg.index.to_period("Q").astype(str)
            else: cash_agg.index = cash_agg.index.strftime('%Y')
            
            cf = pd.DataFrame(columns=cash_agg.index)
            cf.loc['Operating Activities'] = None
            cf.loc['Customer Collections'] = cash_agg.get('Customer Collections', 0) + cash_agg.get('Service Collections', 0)
            cf.loc['Supplier Payments'] = cash_agg.get('Supplier Deposits', 0) + cash_agg.get('Supplier Payments', 0)
            cf.loc['Payroll'] = cash_agg.get('Payroll', 0)
            cf.loc['OpEx Payments'] = cash_agg.get('OpEx Payments', 0) + cash_agg.get('Warranty Payments', 0)
            cf.loc['Net Cash Flow'] = cf.loc['Customer Collections'] + cf.loc['Supplier Payments'] + cf.loc['Payroll'] + cf.loc['OpEx Payments']
            
            end_bals = cash.set_index('Date').resample(fmap[freq])['Cash_Balance'].last()
            if len(end_bals) == len(cf.columns):
                end_bals.index = cf.columns
                cf.loc['Ending Cash'] = end_bals
            
            render_financial_table(cf, "")
        else:
            st.info("No data.")
    
    # =========================================================================
    # BOARD PACK
    # =========================================================================
    elif view == "📋 Board Pack":
        st.title("Board Pack Generator")
        
        st.info("Generate a comprehensive summary for board meetings and investor updates.")
        
        c1, c2 = st.columns(2)
        with c1:
            period = st.selectbox("Period", ["Q1 2026", "Q2 2026", "Q3 2026", "Q4 2026", "Full Year 2026"])
        with c2:
            include_fleet = st.checkbox("Include Fleet ROI", value=True)
            include_warranty = st.checkbox("Include Warranty Stats", value=True)
        
        if st.button("📊 Generate Board Pack", type="primary", use_container_width=True):
            st.divider()
            
            # Executive Summary
            st.header("Executive Summary")
            if not pnl.empty:
                rev = pnl[pnl['Type'] == 'Revenue']['Amount'].sum()
                cogs = abs(pnl[pnl['Type'] == 'COGS']['Amount'].sum())
                opex = abs(pnl[pnl['Type'] == 'OpEx']['Amount'].sum())
                net_income = rev - cogs - opex
                
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Total Revenue", fmt_currency(rev, True))
                c2.metric("Gross Margin", fmt_pct((rev - cogs) / rev if rev > 0 else 0))
                c3.metric("Net Income", fmt_currency(net_income, True))
                c4.metric("Cash Position", fmt_currency(cash.iloc[-1]['Cash_Balance'] if not cash.empty else 0, True))
            
            # Covenant Status
            st.header("Covenant Compliance")
            covs = evaluate_covenants(pnl, cash)
            if covs:
                cov_df = pd.DataFrame(covs)
                st.dataframe(cov_df[['name', 'status']], hide_index=True)
            
            # Fleet ROI
            if include_fleet:
                st.header("Fleet Performance")
                _, roi = calculate_fleet_roi()
                if not roi.empty:
                    st.dataframe(roi[['Fleet', 'Units_Deployed', 'Payback_Months', 'ROI_5yr']].head(5), hide_index=True)
            
            # Warranty
            if include_warranty:
                st.header("Quality Metrics")
                try:
                    events = pd.read_sql("SELECT COUNT(*) as cnt, SUM(cost_of_repair) as cost FROM unit_warranty_event", engine)
                    units = pd.read_sql("SELECT COUNT(*) as cnt FROM production_unit", engine)
                    
                    c1, c2 = st.columns(2)
                    c1.metric("Total Warranty Events", int(events.iloc[0]['cnt']))
                    c2.metric("Warranty Cost", fmt_currency(events.iloc[0]['cost'] or 0))
                except:
                    pass
            
            # Download
            st.divider()
            st.download_button(
                "📥 Download as CSV",
                data=pnl.to_csv(index=False) if not pnl.empty else "",
                file_name=f"board_pack_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
    
    # =========================================================================
    # AUDIT LOG
    # =========================================================================
    elif view == "📝 Audit Log":
        st.title("Audit Log")
        
        try:
            logs = pd.read_sql("SELECT * FROM audit_log ORDER BY timestamp DESC LIMIT 200", engine)
            
            if not logs.empty:
                # Filters
                c1, c2 = st.columns(2)
                with c1:
                    actions = ['All'] + logs['action'].unique().tolist()
                    action_filter = st.selectbox("Action", actions)
                with c2:
                    types = ['All'] + logs['object_type'].dropna().unique().tolist()
                    type_filter = st.selectbox("Object Type", types)
                
                filtered = logs
                if action_filter != 'All':
                    filtered = filtered[filtered['action'] == action_filter]
                if type_filter != 'All':
                    filtered = filtered[filtered['object_type'] == type_filter]
                
                st.dataframe(filtered[['timestamp', 'user_name', 'action', 'object_type', 'object_id']], 
                            hide_index=True, use_container_width=True)
            else:
                st.info("No audit entries.")
        except Exception as e:
            st.error(f"Error: {e}")
    
    # =========================================================================
    # INTEGRATIONS
    # =========================================================================
    elif view == "🔌 Integrations":
        st.title("External Integrations")
        
        st.info("Placeholder for external data imports (Samsara telemetry, Salesforce pipeline, etc.)")
        
        tab1, tab2 = st.tabs(["Import Data", "Import History"])
        
        with tab1:
            source = st.selectbox("Source System", ["SAMSARA", "SALESFORCE", "QUICKBOOKS", "CUSTOM"])
            import_type = st.selectbox("Import Type", ["TELEMETRY", "PIPELINE", "GL_EXPORT", "CUSTOM"])
            
            uploaded = st.file_uploader("Upload JSON payload", type=['json'])
            
            if uploaded:
                try:
                    payload = json.load(uploaded)
                    st.json(payload)
                    
                    if st.button("💾 Save Import"):
                        with engine.connect() as conn:
                            conn.execute(text("""
                                INSERT INTO external_data_import (source_system, import_type, payload, processed)
                                VALUES (:s, :t, :p, 0)
                            """), {"s": source, "t": import_type, "p": json.dumps(payload)})
                            conn.commit()
                        st.success("Import saved!")
                except Exception as e:
                    st.error(f"Invalid JSON: {e}")
        
        with tab2:
            try:
                imports = pd.read_sql("SELECT * FROM external_data_import ORDER BY imported_at DESC LIMIT 50", engine)
                if not imports.empty:
                    st.dataframe(imports[['source_system', 'import_type', 'imported_at', 'processed']], hide_index=True)
                else:
                    st.info("No imports.")
            except:
                st.info("No import history.")


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"System Error: {e}")
        logger.exception("Unhandled exception")
