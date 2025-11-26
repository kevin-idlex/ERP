import streamlit as st
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from datetime import datetime, timedelta, date
import plotly.express as px
import plotly.graph_objects as go
import calendar
import os
import logging
import json

import seed_db

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# BRAND COLORS (from IdleX Brandbook 2025)
# =============================================================================
NAVY = "#1E3466"      # Dark Navy Blue - Primary
X_BLUE = "#3A77D8"    # X Blue - Accent
YELLOW = "#FFB400"    # Electric Bolt Yellow
SLATE = "#A5ABB5"     # Slate Gray
LIGHT = "#E6E8EC"     # Light Gray
WHITE = "#FFFFFF"

# =============================================================================
# CONSTANTS
# =============================================================================
MSRP_PRICE = 8500.00
DEALER_DISCOUNT = 0.75
DIRECT_PCT = 0.25
DEALER_LAG = 30
OPT_ITERATIONS = 15

# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="IdleX ERP",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# IdleX Brand Styling
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@600;700;800&family=Open+Sans:wght@400;500;600&display=swap');
    
    /* Global */
    .stApp {{
        background: linear-gradient(180deg, {WHITE} 0%, {LIGHT} 100%);
    }}
    
    .block-container {{
        padding: 1rem 1.5rem;
        max-width: 1400px;
    }}
    
    /* Typography */
    h1, h2, h3, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {{
        font-family: 'Montserrat', sans-serif !important;
        color: {NAVY} !important;
        font-weight: 700 !important;
    }}
    
    p, span, label, .stMarkdown p {{
        font-family: 'Open Sans', sans-serif !important;
    }}
    
    /* Sidebar */
    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, {NAVY} 0%, #101C37 100%) !important;
    }}
    
    [data-testid="stSidebar"] * {{
        color: {WHITE} !important;
    }}
    
    [data-testid="stSidebar"] .stRadio label {{
        color: {WHITE} !important;
        font-family: 'Open Sans', sans-serif !important;
    }}
    
    [data-testid="stSidebar"] .stRadio label:hover {{
        color: {X_BLUE} !important;
    }}
    
    /* Buttons */
    .stButton > button {{
        background: linear-gradient(135deg, {NAVY} 0%, {X_BLUE} 100%) !important;
        color: {WHITE} !important;
        border: none !important;
        border-radius: 8px !important;
        font-family: 'Montserrat', sans-serif !important;
        font-weight: 600 !important;
        padding: 0.6rem 1.2rem !important;
        transition: all 0.3s ease !important;
    }}
    
    .stButton > button:hover {{
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 12px rgba(30, 52, 102, 0.3) !important;
    }}
    
    .stButton > button[kind="primary"] {{
        background: linear-gradient(135deg, {X_BLUE} 0%, {NAVY} 100%) !important;
    }}
    
    /* Metrics */
    [data-testid="stMetricValue"] {{
        font-family: 'Montserrat', sans-serif !important;
        color: {NAVY} !important;
        font-weight: 700 !important;
    }}
    
    [data-testid="stMetricLabel"] {{
        font-family: 'Open Sans', sans-serif !important;
        color: {SLATE} !important;
    }}
    
    /* Cards */
    .metric-card {{
        background: {WHITE};
        border-radius: 12px;
        padding: 1.25rem;
        box-shadow: 0 2px 8px rgba(30, 52, 102, 0.08);
        border-left: 4px solid {X_BLUE};
    }}
    
    .metric-card-accent {{
        border-left-color: {YELLOW};
    }}
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
        background: {LIGHT};
        padding: 4px;
        border-radius: 8px;
    }}
    
    .stTabs [data-baseweb="tab"] {{
        background: transparent !important;
        border-radius: 6px !important;
        color: {NAVY} !important;
        font-family: 'Montserrat', sans-serif !important;
        font-weight: 600 !important;
    }}
    
    .stTabs [aria-selected="true"] {{
        background: {WHITE} !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
    }}
    
    /* Tables */
    .dataframe {{
        font-family: 'Open Sans', sans-serif !important;
    }}
    
    .dataframe th {{
        background: {NAVY} !important;
        color: {WHITE} !important;
        font-weight: 600 !important;
    }}
    
    /* Status indicators */
    .status-ok {{ color: #10B981; font-weight: 600; }}
    .status-warn {{ color: {YELLOW}; font-weight: 600; }}
    .status-danger {{ color: #EF4444; font-weight: 600; }}
    
    /* Financial tables */
    .fin-table {{
        font-family: 'Open Sans', sans-serif;
        width: 100%;
        border-collapse: collapse;
        background: {WHITE};
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }}
    
    .fin-table th {{
        background: {NAVY};
        color: {WHITE};
        padding: 12px 8px;
        text-align: right;
        font-weight: 600;
    }}
    
    .fin-table th:first-child {{
        text-align: left;
    }}
    
    .fin-table td {{
        padding: 10px 8px;
        border-bottom: 1px solid {LIGHT};
        text-align: right;
    }}
    
    .fin-table td:first-child {{
        text-align: left;
        color: {NAVY};
    }}
    
    .fin-table .section {{
        font-weight: 700;
        color: {NAVY};
        background: {LIGHT};
    }}
    
    .fin-table .total {{
        font-weight: 700;
        border-top: 2px solid {NAVY};
    }}
    
    /* Expander */
    .streamlit-expanderHeader {{
        font-family: 'Montserrat', sans-serif !important;
        font-weight: 600 !important;
        color: {NAVY} !important;
    }}
    
    /* Divider */
    hr {{
        border-color: {LIGHT} !important;
    }}
    
    /* Mobile */
    @media (max-width: 768px) {{
        .block-container {{ padding: 0.5rem; }}
        h1 {{ font-size: 1.5rem !important; }}
        [data-testid="stMetricValue"] {{ font-size: 1.2rem !important; }}
        .stButton button {{ min-height: 48px; width: 100%; }}
    }}
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

engine = get_engine()

# =============================================================================
# UTILITIES
# =============================================================================
def fmt_currency(v, compact=False):
    if pd.isna(v) or v is None: return ""
    if compact:
        if abs(v) >= 1e6: return f"${v/1e6:.1f}M"
        if abs(v) >= 1e3: return f"${v/1e3:.0f}K"
    return f"(${abs(v):,.0f})" if v < 0 else f"${v:,.0f}"

def fmt_pct(v):
    return f"{v*100:.1f}%" if v else "N/A"

def get_workdays(year, month, threshold=None):
    days = [date(year, month, d) for d in range(1, calendar.monthrange(year, month)[1] + 1)]
    valid = [d for d in days if d.weekday() < 5]
    return [d for d in valid if d >= threshold] if threshold else valid

def load_config():
    try:
        cfg = pd.read_sql("SELECT setting_key, setting_value FROM global_config", engine)
        return dict(zip(cfg['setting_key'], cfg['setting_value']))
    except:
        return {}

def audit_log(action, obj_type=None, obj_id=None, before=None, after=None):
    try:
        with engine.connect() as conn:
            conn.execute(text("INSERT INTO audit_log (user_name, action, object_type, object_id, data_before, data_after) VALUES (:u, :a, :t, :i, :b, :af)"),
                {"u": "system", "a": action, "t": obj_type, "i": str(obj_id) if obj_id else None,
                 "b": json.dumps(before) if before else None, "af": json.dumps(after) if after else None})
            conn.commit()
    except:
        pass

# =============================================================================
# FINANCIAL ENGINE
# =============================================================================
def generate_financials(units_df=None, start_cash_override=None):
    try:
        parts = pd.read_sql("SELECT * FROM part_master", engine)
        bom = pd.read_sql("SELECT * FROM bom_items", engine)
        staffing = pd.read_sql("SELECT * FROM opex_staffing_plan", engine)
        roles = pd.read_sql("SELECT * FROM opex_roles", engine)
        units = units_df if units_df is not None else pd.read_sql("SELECT * FROM production_unit", engine)
        cfg = load_config()
        start_cash = float(start_cash_override or cfg.get('start_cash', 1600000))
    except Exception as e:
        return pd.DataFrame(), pd.DataFrame()
    
    if units.empty:
        return pd.DataFrame(), pd.DataFrame()
    
    units['build_date'] = pd.to_datetime(units['build_date'])
    staffing['month_date'] = pd.to_datetime(staffing['month_date'])
    
    # Material cost
    mat_cost = 0
    if not bom.empty and not parts.empty:
        merged = pd.merge(bom, parts, left_on='part_id', right_on='id')
        mat_cost = (merged['qty_per_unit'] * merged['cost']).sum()
    
    ledger = []
    
    # Revenue & COGS
    for _, u in units.iterrows():
        direct = u['sales_channel'] == 'DIRECT'
        rev = MSRP_PRICE if direct else MSRP_PRICE * DEALER_DISCOUNT
        dt = u['build_date']
        lag = 0 if direct else DEALER_LAG
        
        ledger.append({"Date": dt, "Category": "Product Sales", "Type": "Revenue", "Amount": rev, "Report": "PnL"})
        ledger.append({"Date": dt + timedelta(days=lag), "Category": "Collections", "Type": "Ops", "Amount": rev, "Report": "Cash"})
        ledger.append({"Date": dt, "Category": "Materials", "Type": "COGS", "Amount": -mat_cost, "Report": "PnL"})
    
    # Supply chain
    monthly = units.groupby(pd.Grouper(key='build_date', freq='MS')).size()
    for mo, cnt in monthly.items():
        if cnt == 0: continue
        for _, p in parts.iterrows():
            b = bom[bom['part_id'] == p['id']]
            if b.empty: continue
            cost = b.iloc[0]['qty_per_unit'] * cnt * p['cost']
            if p['deposit_pct'] > 0:
                ledger.append({"Date": mo + timedelta(days=int(p['deposit_days'])), "Category": "Deposits", "Type": "Ops", "Amount": -cost * p['deposit_pct'], "Report": "Cash"})
            if p['deposit_pct'] < 1:
                ledger.append({"Date": mo + timedelta(days=int(p['balance_days'])), "Category": "Payments", "Type": "Ops", "Amount": -cost * (1 - p['deposit_pct']), "Report": "Cash"})
    
    # Payroll
    if not staffing.empty and not roles.empty:
        merged = pd.merge(staffing, roles, left_on='role_id', right_on='id')
        for _, r in merged.iterrows():
            cost = (r['annual_salary'] / 12) * r['headcount']
            if cost > 0:
                labor = "Assembler" in r['role_name']
                ledger.append({"Date": r['month_date'], "Category": "Direct Labor" if labor else "Salaries", "Type": "COGS" if labor else "OpEx", "Amount": -cost, "Report": "PnL"})
                ledger.append({"Date": r['month_date'], "Category": "Payroll", "Type": "Ops", "Amount": -cost, "Report": "Cash"})
    
    # Service revenue
    try:
        subs = pd.read_sql("SELECT s.*, p.annual_price, p.term_months FROM unit_service_subscription s JOIN service_plan p ON s.service_plan_id = p.id WHERE s.status = 'ACTIVE'", engine)
        if not subs.empty:
            subs['start_date'] = pd.to_datetime(subs['start_date'])
            for _, sub in subs.iterrows():
                monthly_rev = sub['annual_price'] / 12
                for m in range(int(sub['term_months'])):
                    rev_date = sub['start_date'] + timedelta(days=30*m)
                    ledger.append({"Date": rev_date, "Category": "Service Revenue", "Type": "Revenue", "Amount": monthly_rev, "Report": "PnL"})
                    ledger.append({"Date": rev_date, "Category": "Service", "Type": "Ops", "Amount": monthly_rev, "Report": "Cash"})
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
# SIMULATION
# =============================================================================
def simulate_scenario(start_units, growth_pct, start_date, months):
    units = []
    current = start_units
    sn = 1
    dt = start_date.replace(day=1)
    
    for _ in range(months):
        target = int(current)
        days = get_workdays(dt.year, dt.month)
        if target > 0 and days:
            direct = int(target * DIRECT_PCT)
            pool = ['DIRECT'] * direct + ['DEALER'] * (target - direct)
            for i, ch in enumerate(pool):
                units.append({"serial_number": f"SIM-{sn:05d}", "build_date": days[i % len(days)], "sales_channel": ch, "status": "PLANNED"})
                sn += 1
        current *= (1 + growth_pct / 100)
        dt = date(dt.year + (1 if dt.month == 12 else 0), 1 if dt.month == 12 else dt.month + 1, 1)
    
    return pd.DataFrame(units)

def optimize_growth(start_units, start_cash, loc_limit, start_date, months):
    best = {'rate': 0, 'cash_df': pd.DataFrame(), 'units_df': pd.DataFrame()}
    low, high = 0.0, 100.0
    
    for _ in range(OPT_ITERATIONS):
        mid = (low + high) / 2
        sim = simulate_scenario(start_units, mid, start_date, months)
        _, cash = generate_financials(units_df=sim, start_cash_override=start_cash)
        
        min_cash = cash['Cash_Balance'].min() if not cash.empty else 0
        if min_cash >= -loc_limit:
            best = {'rate': mid, 'cash_df': cash, 'units_df': sim}
            low = mid
        else:
            high = mid
    
    if not best['units_df'].empty:
        best['units_df']['build_date'] = pd.to_datetime(best['units_df']['build_date'])
        monthly = best['units_df'].groupby(best['units_df']['build_date'].dt.to_period('M')).size().reset_index()
        monthly.columns = ['Month', 'Units']
        monthly['Month'] = monthly['Month'].astype(str)
        best['monthly'] = monthly
    
    pnl, _ = generate_financials(units_df=best['units_df'], start_cash_override=start_cash) if not best['units_df'].empty else (pd.DataFrame(), None)
    best['total_revenue'] = pnl[pnl['Type'] == 'Revenue']['Amount'].sum() if not pnl.empty else 0
    best['total_units'] = len(best['units_df'])
    best['min_cash'] = best['cash_df']['Cash_Balance'].min() if not best['cash_df'].empty else 0
    
    return best

# =============================================================================
# COVENANTS
# =============================================================================
def evaluate_covenants(pnl, cash):
    try:
        covs = pd.read_sql("SELECT * FROM covenant_config WHERE active = 1", engine)
    except:
        return []
    
    if covs.empty or cash.empty:
        return []
    
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
        
        results.append({'name': c['name'], 'type': c['covenant_type'], 'threshold': c['threshold_value'], 'current': val, 'status': status})
    
    return results

# =============================================================================
# CAPACITY
# =============================================================================
def calculate_capacity(start_date, months):
    try:
        wcs = pd.read_sql("SELECT * FROM work_center", engine)
        routing = pd.read_sql("SELECT * FROM routing_step WHERE is_bottleneck = 1", engine)
        assigns = pd.read_sql("SELECT * FROM work_center_assignment", engine)
        staffing = pd.read_sql("SELECT * FROM opex_staffing_plan", engine)
    except:
        return pd.DataFrame()
    
    if wcs.empty or routing.empty:
        return pd.DataFrame()
    
    results = []
    dt = start_date.replace(day=1)
    
    for _ in range(months):
        workdays = len(get_workdays(dt.year, dt.month))
        staffing['month_date'] = pd.to_datetime(staffing['month_date'])
        month_staff = staffing[staffing['month_date'].dt.to_period('M') == pd.Period(dt, 'M')]
        
        min_cap = float('inf')
        limiting = None
        
        for _, step in routing.iterrows():
            wc = wcs[wcs['id'] == step['work_center_id']]
            if wc.empty: continue
            wc = wc.iloc[0]
            
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
        
        results.append({'Month': dt.strftime('%Y-%m'), 'Capacity': int(min_cap) if min_cap != float('inf') else 0, 'Bottleneck': limiting or 'N/A', 'Workdays': workdays})
        dt = date(dt.year + (1 if dt.month == 12 else 0), 1 if dt.month == 12 else dt.month + 1, 1)
    
    return pd.DataFrame(results)

# =============================================================================
# FLEET ROI
# =============================================================================
def calculate_fleet_roi():
    try:
        fleets = pd.read_sql("SELECT * FROM fleet", engine)
        assigns = pd.read_sql("SELECT a.*, u.serial_number FROM unit_fleet_assignment a JOIN production_unit u ON a.production_unit_id = u.id", engine)
    except:
        return pd.DataFrame(), pd.DataFrame()
    
    if fleets.empty:
        return fleets, pd.DataFrame()
    
    results = []
    for _, f in fleets.iterrows():
        fleet_units = assigns[assigns['fleet_id'] == f['id']]
        unit_count = len(fleet_units)
        annual_savings = f['nights_on_road_per_year'] * f['idle_hours_per_night'] * f['gallons_per_idle_hour'] * f['diesel_price_assumption']
        avg_price = fleet_units['purchase_price'].mean() if not fleet_units.empty else MSRP_PRICE * DEALER_DISCOUNT
        payback = (avg_price / annual_savings * 12) if annual_savings > 0 else 999
        roi_5yr = ((annual_savings * 5) - avg_price) / avg_price if avg_price > 0 else 0
        
        results.append({'Fleet': f['name'], 'Type': f['fleet_type'], 'Trucks': f['truck_count'], 'Units': unit_count, 'Annual_Savings': annual_savings, 'Payback_Mo': payback, 'ROI_5yr': roi_5yr})
    
    return fleets, pd.DataFrame(results)

# =============================================================================
# SCENARIO LIBRARY
# =============================================================================
def save_scenario(name, desc, results, inputs):
    with engine.connect() as conn:
        try:
            start_dt = str(inputs['start_date'])
            conn.execute(text("INSERT INTO scenario_header (name, description, base_start_cash, base_loc_limit, start_units, growth_rate, start_date, forecast_months, total_revenue, total_units, min_cash) VALUES (:n, :d, :c, :l, :u, :r, :s, :m, :rev, :tu, :mc)"),
                {"n": name, "d": desc, "c": inputs['start_cash'], "l": inputs['loc_limit'], "u": inputs['start_units'], "r": results['rate'], "s": start_dt, "m": inputs['months'], "rev": results['total_revenue'], "tu": results['total_units'], "mc": results['min_cash']})
            conn.commit()
            return True, "Saved"
        except Exception as e:
            conn.rollback()
            return False, str(e)

def push_to_production(units_df):
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
                if hasattr(bd, 'strftime'): bd = bd.strftime('%Y-%m-%d')
                elif hasattr(bd, 'isoformat'): bd = bd.isoformat()
                conn.execute(text("INSERT INTO production_unit (serial_number, build_date, sales_channel, status) VALUES (:s, :d, :c, 'PLANNED')"),
                    {"s": f"IDX-{next_sn:04d}", "d": str(bd), "c": row['sales_channel']})
                next_sn += 1
            
            conn.commit()
            return True, f"Replaced {deleted} with {len(units_df)} units", len(units_df)
        except Exception as e:
            conn.rollback()
            return False, str(e), 0

# =============================================================================
# RENDER HELPERS
# =============================================================================
def render_covenant_card(cov):
    icon = "✅" if cov['status'] == "OK" else ("⚠️" if cov['status'] == "WARNING" else "❌")
    color = "#10B981" if cov['status'] == "OK" else (YELLOW if cov['status'] == "WARNING" else "#EF4444")
    
    if cov['type'] == 'MIN_MARGIN':
        curr, thresh = fmt_pct(cov['current']), fmt_pct(cov['threshold'])
    elif cov['type'] == 'MIN_RUNWAY':
        curr, thresh = f"{cov['current']:.0f} mo" if cov['current'] else "N/A", f"{cov['threshold']:.0f} mo"
    else:
        curr, thresh = fmt_currency(cov['current'], True), fmt_currency(cov['threshold'], True)
    
    st.markdown(f"""
    <div class="metric-card">
        <div style="font-weight:700;color:{NAVY};font-family:Montserrat;">{cov['name']}</div>
        <div style="margin:8px 0;font-size:1.5rem;font-weight:700;color:{NAVY};">{curr}</div>
        <div style="color:{SLATE};font-size:0.85rem;">Target: {thresh}</div>
        <div style="margin-top:8px;color:{color};font-weight:600;">{icon} {cov['status']}</div>
    </div>
    """, unsafe_allow_html=True)

# =============================================================================
# MAIN APPLICATION
# =============================================================================
def main():
    # Header
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:12px;margin-bottom:1rem;">
        <div style="font-size:2.5rem;font-weight:800;font-family:Montserrat;color:{NAVY};">idle<span style="color:{X_BLUE};">X</span></div>
        <div style="font-size:1rem;color:{SLATE};font-family:Open Sans;border-left:2px solid {LIGHT};padding-left:12px;">ERP</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown(f"""
        <div style="text-align:center;padding:1rem 0;">
            <div style="font-size:1.8rem;font-weight:800;font-family:Montserrat;">idle<span style="color:{X_BLUE};">X</span></div>
            <div style="font-size:0.8rem;color:{SLATE};margin-top:4px;">Enterprise Resource Planning</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.divider()
        
        if st.button("⚡ Initialize Database", use_container_width=True):
            with st.spinner("Initializing..."):
                seed_db.run_seed()
            st.success("Ready!")
            st.rerun()
        
        st.divider()
        
        views = ["📊 Dashboard", "🚀 Scenario Planner", "📚 Scenarios", "📦 Inventory", "🏭 Capacity", "🚛 Fleet ROI", "🛡️ Warranty", "💳 Service", "📈 Financials", "📋 Board Pack", "📝 Audit Log"]
        view = st.radio("Navigation", views, label_visibility="collapsed")
    
    # Generate financials
    pnl, cash = pd.DataFrame(), pd.DataFrame()
    if view not in ["🚀 Scenario Planner", "📚 Scenarios"]:
        pnl, cash = generate_financials()
    
    cfg = load_config()
    
    # =========================================================================
    # DASHBOARD
    # =========================================================================
    if view == "📊 Dashboard":
        if not pnl.empty:
            rev = pnl[pnl['Type'] == 'Revenue']['Amount'].sum()
            cogs = abs(pnl[pnl['Type'] == 'COGS']['Amount'].sum())
            margin = rev - cogs
            margin_pct = margin / rev if rev > 0 else 0
            min_cash = cash['Cash_Balance'].min() if not cash.empty else 0
            end_cash = cash.iloc[-1]['Cash_Balance'] if not cash.empty else 0
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Revenue", fmt_currency(rev, True))
            c2.metric("Gross Margin", f"{margin_pct:.0%}")
            c3.metric("Min Cash", fmt_currency(min_cash, True))
            c4.metric("Ending Cash", fmt_currency(end_cash, True))
            
            # Cash chart with brand colors
            fig = px.area(cash, x='Date', y='Cash_Balance', color_discrete_sequence=[X_BLUE])
            fig.update_layout(
                title=dict(text="Cash Forecast", font=dict(family="Montserrat", size=18, color=NAVY)),
                height=300, margin=dict(l=0, r=0, t=40, b=0),
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor=LIGHT)
            )
            loc = float(cfg.get('loc_limit', 500000))
            fig.add_hline(y=-loc, line_dash="dash", line_color="#EF4444", annotation_text="Credit Limit")
            fig.add_hline(y=0, line_dash="dot", line_color=SLATE)
            st.plotly_chart(fig, use_container_width=True)
            
            # Covenants
            st.markdown(f"<h3 style='color:{NAVY};font-family:Montserrat;'>Covenant Monitor</h3>", unsafe_allow_html=True)
            covs = evaluate_covenants(pnl, cash)
            if covs:
                cols = st.columns(len(covs))
                for i, c in enumerate(covs):
                    with cols[i]:
                        render_covenant_card(c)
            
            # Runway
            burn = cash['Amount'].mean() if not cash.empty else 0
            runway = abs(end_cash / burn) if burn < 0 else 99
            
            st.markdown(f"""
            <div class="metric-card metric-card-accent" style="margin-top:1rem;">
                <div style="font-weight:700;color:{NAVY};font-family:Montserrat;">Cash Runway</div>
                <div style="font-size:2rem;font-weight:700;color:{NAVY};margin:8px 0;">{runway:.0f} months</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Click 'Initialize Database' in sidebar to get started.")
    
    # =========================================================================
    # SCENARIO PLANNER
    # =========================================================================
    elif view == "🚀 Scenario Planner":
        st.markdown(f"<h2>Growth Scenario Planner</h2>", unsafe_allow_html=True)
        
        def_cash = float(cfg.get('start_cash', 1600000))
        def_loc = float(cfg.get('loc_limit', 500000))
        
        if 'scenario_results' not in st.session_state:
            st.session_state.scenario_results = None
        
        c1, c2 = st.columns(2)
        with c1:
            inv_cash = st.number_input("Investor Equity ($)", value=def_cash, step=100000.0, format="%.0f")
            start_vol = st.number_input("Starting Units/Month", value=50, min_value=1)
        with c2:
            loc_limit = st.number_input("Credit Limit ($)", value=def_loc, step=100000.0, format="%.0f")
            months = st.slider("Forecast Months", 12, 60, 36)
        
        start_dt = st.date_input("Start Date", value=date(2026, 1, 1))
        
        if st.button("🔍 Find Maximum Growth Rate", type="primary", use_container_width=True):
            with st.spinner("Optimizing..."):
                results = optimize_growth(start_vol, inv_cash, loc_limit, start_dt, months)
                st.session_state.scenario_results = results
                st.session_state.scenario_inputs = {'start_units': start_vol, 'start_cash': inv_cash, 'loc_limit': loc_limit, 'start_date': start_dt, 'months': months}
            st.rerun()
        
        if st.session_state.scenario_results:
            res = st.session_state.scenario_results
            inp = st.session_state.scenario_inputs
            
            st.divider()
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Max Growth", f"{res['rate']:.1f}%/mo")
            c2.metric("Total Units", f"{res['total_units']:,}")
            c3.metric("Revenue", fmt_currency(res['total_revenue'], True))
            c4.metric("Min Cash", fmt_currency(res['min_cash'], True))
            
            if not res['cash_df'].empty:
                fig = px.area(res['cash_df'], x='Date', y='Cash_Balance', color_discrete_sequence=[X_BLUE])
                fig.update_layout(title=f"Cash Flow @ {res['rate']:.1f}%/mo Growth", height=280, plot_bgcolor='rgba(0,0,0,0)')
                fig.add_hline(y=-inp['loc_limit'], line_dash="dash", line_color="#EF4444")
                st.plotly_chart(fig, use_container_width=True)
            
            st.divider()
            c1, c2 = st.columns([2, 1])
            with c1:
                name = st.text_input("Scenario Name", f"Scenario_{datetime.now().strftime('%Y%m%d_%H%M')}")
                desc = st.text_area("Description", height=60)
            with c2:
                if st.button("💾 Save Scenario", use_container_width=True):
                    ok, msg = save_scenario(name, desc, res, inp)
                    st.success("Saved!") if ok else st.error(msg)
                
                if st.button("🚀 Push to Production", type="primary", use_container_width=True):
                    ok, msg, _ = push_to_production(res['units_df'])
                    if ok:
                        st.success(msg)
                        st.balloons()
                    else:
                        st.error(msg)
    
    # =========================================================================
    # SCENARIOS
    # =========================================================================
    elif view == "📚 Scenarios":
        st.markdown("<h2>Scenario Library</h2>", unsafe_allow_html=True)
        try:
            scenarios = pd.read_sql("SELECT * FROM scenario_header ORDER BY created_at DESC", engine)
            if scenarios.empty:
                st.info("No scenarios saved yet.")
            else:
                for _, sc in scenarios.iterrows():
                    star = "⭐ " if sc['is_plan_of_record'] else ""
                    with st.expander(f"{star}{sc['name']} — {fmt_currency(sc['total_revenue'], True)}"):
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("Growth", f"{sc['growth_rate']:.1f}%/mo")
                        c2.metric("Units", f"{sc['total_units']:,}")
                        c3.metric("Revenue", fmt_currency(sc['total_revenue'], True))
                        c4.metric("Min Cash", fmt_currency(sc['min_cash'], True))
                        st.caption(sc['description'] or "No description")
                        
                        c1, c2 = st.columns(2)
                        with c1:
                            if not sc['is_plan_of_record']:
                                if st.button("⭐ Set as Plan of Record", key=f"por_{sc['id']}"):
                                    with engine.connect() as conn:
                                        conn.execute(text("UPDATE scenario_header SET is_plan_of_record = 0"))
                                        conn.execute(text("UPDATE scenario_header SET is_plan_of_record = 1 WHERE id = :id"), {"id": sc['id']})
                                        conn.commit()
                                    st.rerun()
                        with c2:
                            if st.button("🗑️ Delete", key=f"del_{sc['id']}"):
                                with engine.connect() as conn:
                                    conn.execute(text("DELETE FROM scenario_header WHERE id = :id"), {"id": sc['id']})
                                    conn.commit()
                                st.rerun()
        except:
            st.info("No scenarios.")
    
    # =========================================================================
    # INVENTORY
    # =========================================================================
    elif view == "📦 Inventory":
        st.markdown("<h2>Inventory & Purchasing</h2>", unsafe_allow_html=True)
        tab1, tab2 = st.tabs(["Current Inventory", "Open POs"])
        
        with tab1:
            try:
                inv = pd.read_sql("""
                    SELECT p.sku, p.name, p.supplier_name, p.cost, p.moq, p.lead_time,
                           COALESCE(i.quantity_on_hand, 0) as on_hand, p.reorder_point, p.safety_stock
                    FROM part_master p
                    LEFT JOIN (SELECT part_id, quantity_on_hand FROM inventory_balance WHERE as_of_date = (SELECT MAX(as_of_date) FROM inventory_balance)) i ON p.id = i.part_id
                """, engine)
                inv['Status'] = inv.apply(lambda r: '🔴 LOW' if r['on_hand'] < r['reorder_point'] else ('🟡 OK' if r['on_hand'] < r['safety_stock']*2 else '🟢 Good'), axis=1)
                inv['Value'] = inv['on_hand'] * inv['cost']
                
                st.metric("Total Inventory Value", fmt_currency(inv['Value'].sum()))
                st.dataframe(inv[['sku', 'name', 'on_hand', 'reorder_point', 'Status', 'Value']], hide_index=True, use_container_width=True)
            except Exception as e:
                st.error(f"Error: {e}")
        
        with tab2:
            try:
                pos = pd.read_sql("SELECT * FROM purchase_order_header WHERE status NOT IN ('RECEIVED_FULL', 'CANCELLED')", engine)
                if not pos.empty:
                    st.dataframe(pos, hide_index=True, use_container_width=True)
                else:
                    st.info("No open POs.")
            except:
                st.info("No PO data.")
    
    # =========================================================================
    # CAPACITY
    # =========================================================================
    elif view == "🏭 Capacity":
        st.markdown("<h2>Capacity Planning</h2>", unsafe_allow_html=True)
        
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
            
            st.dataframe(merged[['Month', 'Planned', 'Capacity', 'Buffer', 'Status', 'Bottleneck']], hide_index=True, use_container_width=True)
            
            fig = go.Figure()
            fig.add_trace(go.Bar(x=merged['Month'], y=merged['Planned'], name='Planned', marker_color=NAVY))
            fig.add_trace(go.Scatter(x=merged['Month'], y=merged['Capacity'], name='Capacity', mode='lines+markers', line=dict(color=X_BLUE, width=3)))
            fig.update_layout(title="Planned vs Capacity", height=350, plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Configure work centers to enable.")
    
    # =========================================================================
    # FLEET ROI
    # =========================================================================
    elif view == "🚛 Fleet ROI":
        st.markdown("<h2>Fleet Unit Economics</h2>", unsafe_allow_html=True)
        
        _, roi = calculate_fleet_roi()
        
        if not roi.empty:
            st.dataframe(roi.style.format({'Annual_Savings': '${:,.0f}', 'Payback_Mo': '{:.1f}', 'ROI_5yr': '{:.0%}'}), hide_index=True, use_container_width=True)
            
            c1, c2 = st.columns(2)
            with c1:
                fig = px.bar(roi, x='Fleet', y='Payback_Mo', title="Payback (Months)", color_discrete_sequence=[X_BLUE])
                fig.update_layout(height=300, plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                fig = px.bar(roi, x='Fleet', y='ROI_5yr', title="5-Year ROI", color_discrete_sequence=[NAVY])
                fig.update_layout(height=300, plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, use_container_width=True)
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Deployed", int(roi['Units'].sum()))
            c2.metric("Avg Payback", f"{roi['Payback_Mo'].mean():.1f} mo")
            c3.metric("Avg 5yr ROI", f"{roi['ROI_5yr'].mean():.0%}")
        else:
            st.info("No fleet data.")
    
    # =========================================================================
    # WARRANTY
    # =========================================================================
    elif view == "🛡️ Warranty":
        st.markdown("<h2>Warranty & Quality</h2>", unsafe_allow_html=True)
        
        try:
            events = pd.read_sql("SELECT e.*, u.serial_number, p.name as part_name FROM unit_warranty_event e JOIN production_unit u ON e.production_unit_id = u.id LEFT JOIN part_master p ON e.part_id = p.id ORDER BY e.event_date DESC", engine)
            
            if not events.empty:
                total_cost = events['cost_of_repair'].sum()
                c1, c2, c3 = st.columns(3)
                c1.metric("Total Events", len(events))
                c2.metric("Total Cost", fmt_currency(total_cost))
                c3.metric("Avg Cost", fmt_currency(total_cost / len(events)))
                
                st.dataframe(events[['serial_number', 'event_date', 'failure_mode', 'part_name', 'cost_of_repair']], hide_index=True, use_container_width=True)
            else:
                st.info("No warranty events.")
        except Exception as e:
            st.error(f"Error: {e}")
    
    # =========================================================================
    # SERVICE
    # =========================================================================
    elif view == "💳 Service":
        st.markdown("<h2>Service & Recurring Revenue</h2>", unsafe_allow_html=True)
        
        try:
            subs = pd.read_sql("SELECT s.*, u.serial_number, p.name as plan_name, p.annual_price, f.name as fleet_name FROM unit_service_subscription s JOIN production_unit u ON s.production_unit_id = u.id JOIN service_plan p ON s.service_plan_id = p.id LEFT JOIN fleet f ON s.fleet_id = f.id ORDER BY s.start_date DESC", engine)
            
            if not subs.empty:
                active = subs[subs['status'] == 'ACTIVE']
                arr = active['annual_price'].sum()
                
                c1, c2, c3 = st.columns(3)
                c1.metric("Active Subscriptions", len(active))
                c2.metric("ARR", fmt_currency(arr))
                c3.metric("Avg Contract", fmt_currency(active['annual_price'].mean()))
                
                st.dataframe(subs[['serial_number', 'fleet_name', 'plan_name', 'annual_price', 'status']], hide_index=True, use_container_width=True)
            else:
                st.info("No subscriptions.")
        except:
            st.info("No data.")
    
    # =========================================================================
    # FINANCIALS
    # =========================================================================
    elif view == "📈 Financials":
        st.markdown("<h2>Financial Statements</h2>", unsafe_allow_html=True)
        
        if not pnl.empty:
            freq = st.radio("Period", ["Monthly", "Quarterly"], horizontal=True, index=1)
            fmap = {"Monthly": "ME", "Quarterly": "QE"}
            
            rev = pnl[pnl['Type'] == 'Revenue']['Amount'].sum()
            cogs = abs(pnl[pnl['Type'] == 'COGS']['Amount'].sum())
            opex = abs(pnl[pnl['Type'] == 'OpEx']['Amount'].sum())
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Revenue", fmt_currency(rev, True))
            c2.metric("COGS", fmt_currency(cogs, True))
            c3.metric("Gross Profit", fmt_currency(rev - cogs, True))
            c4.metric("Net Income", fmt_currency(rev - cogs - opex, True))
            
            # Simple P&L table
            agg = pnl.groupby([pd.Grouper(key='Date', freq=fmap[freq]), 'Type']).sum()['Amount'].unstack().fillna(0)
            if freq == "Monthly":
                agg.index = agg.index.strftime('%Y-%b')
            else:
                agg.index = agg.index.to_period("Q").astype(str)
            
            st.dataframe(agg.style.format("${:,.0f}"), use_container_width=True)
        else:
            st.info("No data.")
    
    # =========================================================================
    # BOARD PACK
    # =========================================================================
    elif view == "📋 Board Pack":
        st.markdown("<h2>Board Pack Generator</h2>", unsafe_allow_html=True)
        
        if st.button("📊 Generate Summary", type="primary", use_container_width=True):
            if not pnl.empty:
                rev = pnl[pnl['Type'] == 'Revenue']['Amount'].sum()
                cogs = abs(pnl[pnl['Type'] == 'COGS']['Amount'].sum())
                opex = abs(pnl[pnl['Type'] == 'OpEx']['Amount'].sum())
                
                st.markdown(f"### Executive Summary")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Revenue", fmt_currency(rev, True))
                c2.metric("Gross Margin", fmt_pct((rev - cogs) / rev if rev > 0 else 0))
                c3.metric("Net Income", fmt_currency(rev - cogs - opex, True))
                c4.metric("Cash", fmt_currency(cash.iloc[-1]['Cash_Balance'] if not cash.empty else 0, True))
                
                st.markdown("### Covenants")
                covs = evaluate_covenants(pnl, cash)
                if covs:
                    cov_df = pd.DataFrame(covs)[['name', 'status']]
                    st.dataframe(cov_df, hide_index=True)
                
                st.download_button("📥 Download CSV", data=pnl.to_csv(index=False), file_name=f"board_pack_{datetime.now().strftime('%Y%m%d')}.csv", mime="text/csv")
            else:
                st.info("No data.")
    
    # =========================================================================
    # AUDIT LOG
    # =========================================================================
    elif view == "📝 Audit Log":
        st.markdown("<h2>Audit Log</h2>", unsafe_allow_html=True)
        
        try:
            logs = pd.read_sql("SELECT * FROM audit_log ORDER BY timestamp DESC LIMIT 100", engine)
            if not logs.empty:
                st.dataframe(logs[['timestamp', 'user_name', 'action', 'object_type', 'object_id']], hide_index=True, use_container_width=True)
            else:
                st.info("No audit entries.")
        except:
            st.info("No data.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Error: {e}")
        logger.exception("Unhandled exception")
