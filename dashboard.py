"""
IdleX CFO Console - Dashboard Application
Version: 2.0 (Production Release)
"""

import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text
from datetime import timedelta, date
import plotly.express as px
import plotly.graph_objects as go
import calendar
import os
import logging
import math
import seed_db

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================
MSRP_PRICE = 8500.00
DEALER_DISCOUNT_RATE = 0.75
DEALER_PAYMENT_LAG_DAYS = 30

st.set_page_config(page_title="IdleX CFO Console", layout="wide")

# Custom CSS for "Banker Style" Financials
st.markdown("""
<style>
    .financial-table {
        font-family: 'Georgia', serif;
        font-size: 16px;
        border-collapse: collapse;
        width: 100%;
        color: #000000 !important;
        background-color: #ffffff;
        margin-bottom: 20px;
    }
    .financial-table th {
        text-align: right;
        border-bottom: 2px solid #000;
        padding: 10px;
        font-weight: bold;
        color: #000000 !important;
    }
    .financial-table td {
        padding: 8px 10px;
        border: none;
        color: #000000 !important;
    }
    .financial-table tr:hover {
        background-color: #f5f5f5;
    }
    .financial-table .row-header {
        text-align: left;
        width: 40%;
    }
    .financial-table .section-header {
        font-weight: bold;
        text-decoration: underline;
        padding-top: 20px;
    }
    .financial-table .total-row {
        font-weight: bold;
        border-top: 1px solid #000;
    }
    .financial-table .grand-total {
        font-weight: bold;
        border-top: 1px solid #000;
        border-bottom: 3px double #000;
    }
    .financial-table .indent {
        padding-left: 25px;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# DATABASE HELPERS
# =============================================================================
@st.cache_resource
def get_db_engine():
    """Handles connection string differences between Local and Cloud."""
    db_url = os.getenv("DATABASE_URL")
    if db_url:
        if db_url.startswith("postgres://"):
            db_url = db_url.replace("postgres://", "postgresql://", 1)
        return create_engine(db_url)
    return create_engine('sqlite:///idlex.db')

def get_db_type():
    db_url = os.getenv("DATABASE_URL")
    if db_url and ("postgresql" in db_url or "postgres" in db_url):
        return "postgresql"
    return "sqlite"

engine = get_db_engine()
DB_TYPE = get_db_type()

def get_upsert_sql(db_type):
    """Returns correct SQL syntax for Upsert (Insert or Update)"""
    if db_type == "postgresql":
        return """
            INSERT INTO opex_staffing_plan (role_id, month_date, headcount) 
            VALUES (:rid, :dt, :hc)
            ON CONFLICT (role_id, month_date) 
            DO UPDATE SET headcount = EXCLUDED.headcount
        """
    else:
        return """
            INSERT OR REPLACE INTO opex_staffing_plan (id, role_id, month_date, headcount) 
            VALUES ((SELECT id FROM opex_staffing_plan WHERE role_id=:rid AND month_date=:dt), :rid, :dt, :hc)
        """

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
def get_workdays(year, month, start_threshold=None):
    num_days = calendar.monthrange(year, month)[1]
    days = [date(year, month, d) for d in range(1, num_days + 1)]
    valid_days = [d for d in days if d.weekday() < 5] 
    if start_threshold:
        valid_days = [d for d in valid_days if d >= start_threshold]
    return valid_days

def format_banker(val):
    if pd.isna(val) or val == "": return ""
    if isinstance(val, str): return val
    if val < 0: return f"({abs(val):,.0f})"
    return f"{val:,.0f}"

def render_financial_statement(df, title):
    html = f"<h3>{title}</h3><div style='border:1px solid #ddd; overflow-x:auto;'><table class='financial-table'>"
    html += "<thead><tr><th class='row-header'>Account</th>"
    for col in df.columns: html += f"<th>{col}</th>"
    html += "</tr></thead><tbody>"
    
    section_headers = ['Revenue', 'Cost of Goods Sold', 'Operating Expenses', 'Operating Activities']
    total_rows = ['Gross Profit', 'Net Cash Flow', 'Total OpEx']
    grand_total = ['Net Income', 'Ending Cash Balance']

    for index, row in df.iterrows():
        clean_index = str(index).strip()
        row_class = "indent"
        if clean_index in section_headers: row_class = "section-header"
        elif clean_index in total_rows: row_class = "total-row"
        elif clean_index in grand_total: row_class = "grand-total"
        
        html += f"<tr class='{row_class}'><td class='row-header'>{clean_index}</td>"
        for col in df.columns:
            html += f"<td style='text-align: right;'>{format_banker(row[col])}</td>"
        html += "</tr>"
    html += "</tbody></table></div>"
    st.markdown(html, unsafe_allow_html=True)

def parse_serial_number(serial):
    if not serial: return 0
    try:
        digits = ''.join(filter(str.isdigit, serial))
        return int(digits) if digits else 0
    except: return 0

# =============================================================================
# CORE LOGIC ENGINES
# =============================================================================
def generate_financials():
    try:
        df_units = pd.read_sql("SELECT * FROM production_unit", engine)
        df_parts = pd.read_sql("SELECT * FROM part_master", engine)
        df_bom = pd.read_sql("SELECT * FROM bom_items", engine)
        df_opex = pd.read_sql("SELECT * FROM opex_staffing_plan", engine)
        df_roles = pd.read_sql("SELECT * FROM opex_roles", engine)
        df_gen_exp = pd.read_sql("SELECT * FROM opex_general_expenses", engine)
        config = pd.read_sql("SELECT * FROM global_config", engine)
    except Exception as e:
        logger.error(f"DB Read Error: {e}")
        return pd.DataFrame(), pd.DataFrame()

    # Conversions
    df_units['build_date'] = pd.to_datetime(df_units['build_date'])
    df_opex['month_date'] = pd.to_datetime(df_opex['month_date'])
    df_gen_exp['month_date'] = pd.to_datetime(df_gen_exp['month_date'])
    
    start_cash_row = config[config['setting_key'] == 'start_cash']['setting_value']
    start_cash = float(start_cash_row.values[0]) if not start_cash_row.empty else 0.0

    ledger = []

    # 1. Revenue & COGS (Unit Level)
    # Pre-calc unit cost for performance
    bom_with_parts = pd.merge(df_bom, df_parts, left_on='part_id', right_on='id')
    unit_mat_cost = (bom_with_parts['qty_per_unit'] * bom_with_parts['cost']).sum()

    for _, unit in df_units.iterrows():
        is_direct = unit['sales_channel'] == 'DIRECT'
        rev_amt = MSRP_PRICE if is_direct else MSRP_PRICE * DEALER_DISCOUNT_RATE
        pnl_date = unit['build_date']
        cash_lag = 0 if is_direct else DEALER_PAYMENT_LAG_DAYS

        ledger.append({"Date": pnl_date, "Category": "Sales of Goods", "Type": "Revenue", "Amount": rev_amt, "Report": "PnL"})
        ledger.append({"Date": pnl_date + timedelta(days=cash_lag), "Category": "Cash from Customers", "Type": "Operations", "Amount": rev_amt, "Report": "Cash"})
        ledger.append({"Date": pnl_date, "Category": "Raw Materials", "Type": "COGS", "Amount": -unit_mat_cost, "Report": "PnL"})

    # 2. Supply Chain Cash Flow (Batch Level)
    monthly_builds = df_units.groupby(pd.Grouper(key='build_date', freq='MS')).size()
    for month_start, count in monthly_builds.items():
        if count == 0: continue
        delivery = month_start
        
        for _, part in df_parts.iterrows():
            bom_row = df_bom[df_bom['part_id'] == part['id']]
            if bom_row.empty: continue
            
            total_po_cost = bom_row.iloc[0]['qty_per_unit'] * count * part['cost']
            
            if part['deposit_pct'] > 0:
                ledger.append({"Date": delivery + timedelta(days=int(part['deposit_days'])), "Category": "Supplier Deposits", "Type": "Operations", "Amount": -(total_po_cost * part['deposit_pct']), "Report": "Cash"})
            if part['deposit_pct'] < 1.0:
                ledger.append({"Date": delivery + timedelta(days=int(part['balance_days'])), "Category": "Supplier Settlements", "Type": "Operations", "Amount": -(total_po_cost * (1 - part['deposit_pct'])), "Report": "Cash"})

    # 3. Payroll
    opex_merged = pd.merge(df_opex, df_roles, left_on='role_id', right_on='id')
    for _, row in opex_merged.iterrows():
        monthly_cost = (row['annual_salary'] / 12) * row['headcount']
        if monthly_cost > 0:
            is_direct = "Assembler" in row['role_name']
            cat = "Direct Labor" if is_direct else "Salaries & Wages"
            pnl_type = "COGS" if is_direct else "OpEx"
            
            ledger.append({"Date": row['month_date'], "Category": cat, "Type": pnl_type, "Amount": -monthly_cost, "Report": "PnL"})
            ledger.append({"Date": row['month_date'], "Category": "Payroll Paid", "Type": "Operations", "Amount": -monthly_cost, "Report": "Cash"})

    # 4. General Expenses
    for _, row in df_gen_exp.iterrows():
        if row['amount'] > 0:
            ledger.append({"Date": row['month_date'], "Category": row['category'], "Type": "OpEx", "Amount": -row['amount'], "Report": "PnL"})
            ledger.append({"Date": row['month_date'], "Category": "OpEx Paid", "Type": "Operations", "Amount": -row['amount'], "Report": "Cash"})

    if not ledger: return pd.DataFrame(), pd.DataFrame()
    
    df_master = pd.DataFrame(ledger)
    df_pnl = df_master[df_master['Report'] == "PnL"].sort_values('Date')
    df_cash = df_master[df_master['Report'] == "Cash"].sort_values('Date')
    df_cash['Cash_Balance'] = df_cash['Amount'].cumsum() + start_cash
    
    return df_pnl, df_cash

def regenerate_production_schedule(edit_plan, start_date):
    """Safely regenerates schedule without deleting WIP."""
    with engine.connect() as conn:
        try:
            # Delete PLANNED only
            conn.execute(text("DELETE FROM production_unit WHERE status = 'PLANNED'"))
            
            # Get next serial
            last_sn = conn.execute(text("SELECT serial_number FROM production_unit ORDER BY id DESC LIMIT 1")).scalar()
            next_sn = parse_serial_number(last_sn) + 1
            
            for _, row in edit_plan.iterrows():
                target = int(row['Target'])
                if target == 0: continue
                
                month_date = row['Month']
                if isinstance(month_date, pd.Timestamp): month_date = month_date.date()
                
                # Calculate remainder needed
                month_str = month_date.strftime('%Y-%m')
                year_month_sql = "TO_CHAR(build_date, 'YYYY-MM')" if DB_TYPE == 'postgresql' else "strftime('%Y-%m', build_date)"
                
                locked_count = conn.execute(text(f"SELECT COUNT(*) FROM production_unit WHERE {year_month_sql} = :ms AND status != 'PLANNED'"), {"ms": month_str}).scalar() or 0
                
                to_build = target - locked_count
                if to_build <= 0: continue
                
                # Date math
                threshold = start_date if (month_date.year == start_date.year and month_date.month == start_date.month) else None
                last_day = date(month_date.year, month_date.month, calendar.monthrange(month_date.year, month_date.month)[1])
                if last_day < start_date: continue
                
                workdays = get_workdays(month_date.year, month_date.month, threshold)
                if not workdays: continue
                
                # Distribution
                direct_count = int(to_build * 0.25)
                pool = ['DIRECT']*direct_count + ['DEALER']*(to_build - direct_count)
                
                for idx, channel in enumerate(pool):
                    build_day = workdays[idx % len(workdays)]
                    conn.execute(text("INSERT INTO production_unit (serial_number, build_date, sales_channel, status) VALUES (:sn, :bd, :ch, 'PLANNED')"),
                                 {"sn": f"IDX-{next_sn:04d}", "bd": build_day, "ch": channel})
                    next_sn += 1
            
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e

# =================================================================