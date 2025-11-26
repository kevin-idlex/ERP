import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text
from datetime import timedelta, date
import plotly.express as px
import plotly.graph_objects as go
import calendar
import math
import os
import seed_db

# 1. PAGE SETUP
st.set_page_config(page_title="IdleX CFO Console", layout="wide")

# Custom CSS for "Banker Style" Financials
st.markdown("""
<style>
    .financial-table {
        font-family: 'Georgia', serif;
        font-size: 16px;
        border-collapse: collapse;
        width: 100%;
        color: #000000 !important; /* Force Black Text */
        background-color: #ffffff; /* Force White Background (Paper Look) */
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

# Cloud-Ready DB Connection
@st.cache_resource
def get_db_engine():
    db_url = os.getenv("DATABASE_URL")
    if db_url:
        if db_url.startswith("postgres://"):
            db_url = db_url.replace("postgres://", "postgresql://", 1)
        return create_engine(db_url)
    return create_engine('sqlite:///idlex.db')

engine = get_db_engine()

# --- HELPER FUNCTIONS ---
def get_workdays(year, month, start_threshold=None):
    num_days = calendar.monthrange(year, month)[1]
    days = [date(year, month, d) for d in range(1, num_days + 1)]
    valid_days = [d for d in days if d.weekday() < 5] 
    if start_threshold:
        valid_days = [d for d in valid_days if d >= start_threshold]
    return valid_days

def format_banker(val):
    # FIX: Check if value is a number before doing math
    if pd.isna(val) or val == "": return ""
    if isinstance(val, str): return val
    if val < 0: return f"({abs(val):,.0f})"
    return f"{val:,.0f}"

def render_financial_statement(df, title):
    html = f"<h3>{title}</h3><div style='border:1px solid #ddd; overflow-x:auto;'><table class='financial-table'>"
    html += "<thead><tr><th class='row-header'>Account</th>"
    for col in df.columns: html += f"<th>{col}</th>"
    html += "</tr></thead><tbody>"
    for index, row in df.iterrows():
        clean_index = str(index).strip()
        row_class = "section-header" if clean_index in ['Revenue', 'Cost of Goods Sold', 'Operating Expenses', 'Operating Activities'] else \
                    "total-row" if clean_index in ['Gross Profit', 'Net Cash Flow', 'Total OpEx'] else \
                    "grand-total" if clean_index in ['Net Income', 'Ending Cash Balance'] else "indent"
        html += f"<tr class='{row_class}'><td class='row-header'>{clean_index}</td>"
        for col in df.columns: 
            html += f"<td style='text-align: right;'>{format_banker(row[col])}</td>"
        html += "</tr>"
    html += "</tbody></table></div>"
    st.markdown(html, unsafe_allow_html=True)

# 2. FINANCIAL ENGINE
def generate_financials():
    try:
        df_units = pd.read_sql("SELECT * FROM production_unit", engine)
        df_parts = pd.read_sql("SELECT * FROM part_master", engine)
        df_bom = pd.read_sql("SELECT * FROM bom_items", engine)
        df_opex = pd.read_sql("SELECT * FROM opex_staffing_plan", engine)
        df_roles = pd.read_sql("SELECT * FROM opex_roles", engine)
        df_gen_exp = pd.read_sql("SELECT * FROM opex_general_expenses", engine)
        config = pd.read_sql("SELECT * FROM global_config", engine)
    except Exception:
        return pd.DataFrame(), pd.DataFrame()

    df_units['build_date'] = pd.to_datetime(df_units['build_date'])
    df_opex['month_date'] = pd.to_datetime(df_opex['month_date'])
    df_gen_exp['month_date'] = pd.to_datetime(df_gen_exp['month_date'])
    start_cash = float(config[config['setting_key']=='start_cash']['setting_value'].values[0])
    
    ledger = []
    
    # Revenue & COGS
    unit_mat_cost = 0
    for _, part in df_parts.iterrows():
        bom_row = df_bom[df_bom['part_id'] == part['id']]
        if not bom_row.empty:
            unit_mat_cost += bom_row.iloc[0]['qty_per_unit'] * part['cost']

    msrp = 8500.00
    for _, unit in df_units.iterrows():
        rev_amt = msrp if unit['sales_channel'] == 'DIRECT' else msrp * 0.75
        pnl_date = unit['build_date']
        cash_lag = 0 if unit['sales_channel'] == 'DIRECT' else 30
        ledger.append({"Date": pnl_date, "Category": "Sales of Goods", "Type": "Revenue", "Amount": rev_amt, "Report": "PnL"})
        ledger.append({"Date": pnl_date + timedelta(days=cash_lag), "Category": "Cash from Customers", "Type": "Operations", "Amount": rev_amt, "Report": "Cash"})
        ledger.append({"Date": pnl_date, "Category": "Raw Materials", "Type": "COGS", "Amount": -unit_mat_cost, "Report": "PnL"})
    
    # Supply Chain Cash
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

    # Payroll
    opex_merged = pd.merge(df_opex, df_roles, left_on='role_id', right_on='id')
    for _, row in opex_merged.iterrows():
        cost = (row['annual_salary']/12) * row['headcount']
        if cost > 0:
            cat_pnl = "Direct Labor" if "Assembler" in row['role_name'] else "Salaries & Wages"
            type_pnl = "COGS" if "Assembler" in row['role_name'] else "OpEx"
            ledger.append({"Date": row['month_date'], "Category": cat_pnl, "Type": type_pnl, "Amount": -cost, "Report": "PnL"})
            ledger.append({"Date": row['month_date'], "Category": "Payroll Paid", "Type": "Operations", "Amount": -cost, "Report": "Cash"})

    # General Expenses
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

# 3. EXECUTE
try:
    df_pnl, df_cash = generate_financials()
except Exception as e:
    df_pnl, df_cash = pd.DataFrame(), pd.DataFrame()

# 4. VISUAL LAYOUT
st.sidebar.title("IdleX CFO Console")

if st.sidebar.button("⚠️ Rebuild Database"):
    with st.spinner("Resetting Database to V7 Defaults..."):
        seed_db.run_seed()
    st.sidebar.success("Done! Refresh page.")
    st.rerun()

view = st.sidebar.radio("Navigation", ["Executive Dashboard", "Financial Statements", "Production & Sales", "OpEx Planning", "BOM & Supply Chain"])

if not df_pnl.empty:
    years = sorted(df_pnl['Date'].dt.year.unique().tolist())
    st.sidebar.divider()
    selected_period = st.sidebar.selectbox("Fiscal Year:", ["All Time"] + years)
    
    if selected_period == "All Time":
        pnl_view, cash_view = df_pnl, df_cash
    else:
        pnl_view = df_pnl[df_pnl['Date'].dt.year == selected_period]
        cash_view = df_cash[df_cash['Date'].dt.year == selected_period]
else:
    pnl_view, cash_view = pd.DataFrame(), pd.DataFrame()

if view == "Executive Dashboard":
    st.title(f"Executive Dashboard")
    if not pnl_view.empty:
        rev = pnl_view[pnl_view['Category']=='Sales of Goods']['Amount'].sum()
        cogs = abs(pnl_view[pnl_view['Type']=='COGS']['Amount'].sum())
        margin = rev - cogs
        min_c = cash_view['Cash_Balance'].min()
        end_c = cash_view.iloc[-1]['Cash_Balance'] if not cash_view.empty else 0
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Revenue", f"${rev:,.0f}")
        c2.metric("Gross Margin", f"${margin:,.0f}")
        c3.metric("Min Net Cash", f"${min_c:,.0f}", delta_color="inverse")
        c4.metric("Ending Cash", f"${end_c:,.0f}")
        
        fig = px.area(cash_view, x='Date', y='Cash_Balance', title="Liquidity Forecast", color_discrete_sequence=['#10B981'])
        fig.add_hline(y=0, line_dash="dash", line_color="red")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Database is empty. Please click 'Rebuild Database' in the sidebar.")

elif view == "Financial Statements":
    st.title("Financial Statements")
    if not pnl_view.empty:
        col1, col2 = st.columns(2)
        with col1: freq = st.radio("Period Aggregation:", ["Monthly", "Quarterly", "Yearly"], horizontal=True, index=1)
        freq_map = {"Monthly": "ME", "Quarterly": "QE", "Yearly": "YE"}
        
        st.header("Consolidated Statement of Operations")
        pnl_agg = pnl_view.groupby([pd.Grouper(key='Date', freq=freq_map[freq]), 'Type', 'Category']).sum()['Amount'].unstack(level=[1,2]).fillna(0)
        
        if freq == "Monthly": pnl_agg.index = pnl_agg.index.strftime('%Y-%b')
        elif freq == "Quarterly": pnl_agg.index = pnl_agg.index.to_period("Q").astype(str)
        else: pnl_agg.index = pnl_agg.index.strftime('%Y')
        
        stmt = pd.DataFrame(columns=pnl_agg.index)
        def safe_sum(keys):
            total = pd.Series(0, index=pnl_agg.index)
            for k in keys: 
                if k in pnl_agg.columns: total += pnl_agg[k]
            return total

        stmt.loc['Revenue'] = ""
        stmt.loc['Sales of Goods'] = safe_sum([('Revenue', 'Sales of Goods')])
        stmt.loc['Cost of Goods Sold'] = ""
        stmt.loc['Raw Materials'] = safe_sum([('COGS', 'Raw Materials')])
        stmt.loc['Direct Labor'] = safe_sum([('COGS', 'Direct Labor')])
        stmt.loc['Gross Profit'] = stmt.loc['Sales of Goods'] + stmt.loc['Raw Materials'] + stmt.loc['Direct Labor']
        stmt.loc['Operating Expenses'] = ""
        stmt.loc['Salaries & Wages