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
        color: #1a1a1a;
    }
    .financial-table th {
        text-align: right;
        border-bottom: 1px solid #000;
        padding: 8px;
        font-weight: bold;
    }
    .financial-table td {
        padding: 6px 8px;
        border: none;
    }
    .financial-table .row-header {
        text-align: left;
        width: 40%;
    }
    .financial-table .section-header {
        font-weight: bold;
        text-decoration: underline;
        padding-top: 15px;
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
        padding-left: 20px;
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
    if pd.isna(val): return "-"
    if val < 0: return f"({abs(val):,.0f})"
    return f"{val:,.0f}"

def render_financial_statement(df, title):
    html = f"<h3>{title}</h3><table class='financial-table'>"
    html += "<thead><tr><th class='row-header'>Account</th>"
    for col in df.columns: html += f"<th>{col}</th>"
    html += "</tr></thead><tbody>"
    for index, row in df.iterrows():
        clean_index = str(index).strip()
        row_class = "section-header" if clean_index in ['Revenue', 'Operating Expenses', 'Operating Activities'] else \
                    "total-row" if clean_index in ['Gross Profit', 'Net Cash Flow', 'Total OpEx'] else \
                    "grand-total" if clean_index in ['Net Income', 'Ending Cash Balance'] else "indent"
        html += f"<tr class='{row_class}'><td class='row-header'>{clean_index}</td>"
        is_header = row_class == "section-header"
        if is_header: 
            for _ in df.columns: html += "<td></td>"
        else:
            for col in df.columns: html += f"<td style='text-align: right;'>{format_banker(row[col])}</td>"
        html += "</tr>"
    html += "</tbody></table>"
    st.markdown(html, unsafe_allow_html=True)

# 2. FINANCIAL ENGINE
def generate_financials():
    df_units = pd.read_sql("SELECT * FROM production_unit", engine)
    df_parts = pd.read_sql("SELECT * FROM part_master", engine)
    df_bom = pd.read_sql("SELECT * FROM bom_items", engine)
    df_opex = pd.read_sql("SELECT * FROM opex_staffing_plan", engine)
    df_roles = pd.read_sql("SELECT * FROM opex_roles", engine)
    try:
        df_gen_exp = pd.read_sql("SELECT * FROM opex_general_expenses", engine)
        df_gen_exp['month_date'] = pd.to_datetime(df_gen_exp['month_date'])
    except:
        df_gen_exp = pd.DataFrame(columns=['month_date', 'expense_type', 'amount'])
    config = pd.read_sql("SELECT * FROM global_config", engine)
    
    df_units['build_date'] = pd.to_datetime(df_units['build_date'])
    df_opex['month_date'] = pd.to_datetime(df_opex['month_date'])
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
    st.warning("⚠️ Database Empty or Connection Failed")
    if st.button("🚀 INITIALIZE CLOUD DATABASE", type="primary"):
        seed_db.run_seed()
        st.rerun()
    st.stop()

# 4. VISUAL LAYOUT
st.sidebar.title("IdleX CFO Console")
view = st.sidebar.radio("Navigation", ["Executive Dashboard", "Financial Statements", "Production & Sales", "OpEx Planning", "BOM & Supply Chain"])

# Global Filter
years = sorted(df_pnl['Date'].dt.year.unique().tolist()) if not df_pnl.empty else []
st.sidebar.divider()
selected_period = st.sidebar.selectbox("Fiscal Year:", ["All Time"] + years)

if selected_period == "All Time":
    pnl_view, cash_view = df_pnl, df_cash
else:
    pnl_view = df_pnl[df_pnl['Date'].dt.year == selected_period]
    cash_view = df_cash[df_cash['Date'].dt.year == selected_period]

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

elif view == "Financial Statements":
    st.title("Financial Statements")
    if not pnl_view.empty:
        col1, col2 = st.columns(2)
        with col1: freq = st.radio("Period Aggregation:", ["Monthly", "Quarterly", "Yearly"], horizontal=True, index=1)
        freq_map = {"Monthly": "ME", "Quarterly": "QE", "Yearly": "YE"}
        
        # P&L
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
        stmt.loc['Salaries & Wages'] = safe_sum([('OpEx', 'Salaries & Wages')])
        opex_cols = [c for c in pnl_agg.columns if c[0] == 'OpEx' and c[1] != 'Salaries & Wages']
        for col in opex_cols: stmt.loc[col[1]] = safe_sum([col])
        stmt.loc['Total OpEx'] = safe_sum([('OpEx', c[1]) for c in opex_cols]) + stmt.loc['Salaries & Wages']
        stmt.loc['Net Income'] = stmt.loc['Gross Profit'] + stmt.loc['Total OpEx']
        
        render_financial_statement(stmt, "")
        st.markdown("---")
        
        # Cash Flow
        st.header("Statement of Cash Flows")
        cash_view_indexed = cash_view.set_index('Date')
        cash_agg = cash_view.groupby([pd.Grouper(key='Date', freq=freq_map[freq]), 'Category']).sum()['Amount'].unstack().fillna(0)
        if freq == "Monthly": cash_agg.index = cash_agg.index.strftime('%Y-%b')
        elif freq == "Quarterly": cash_agg.index = cash_agg.index.to_period("Q").astype(str)
        else: cash_agg.index = cash_agg.index.strftime('%Y')
        
        cf = pd.DataFrame(columns=cash_agg.index)
        cf.loc['Operating Activities'] = ""
        cf.loc['Cash from Customers'] = cash_agg.get('Cash from Customers', 0)
        cf.loc['Supplier Payments'] = cash_agg.get('Supplier Deposits', 0) + cash_agg.get('Supplier Settlements', 0)
        cf.loc['Payroll Paid'] = cash_agg.get('Payroll Paid', 0)
        cf.loc['OpEx Paid'] = cash_agg.get('OpEx Paid', 0)
        cf.loc['Net Cash Flow'] = cf.sum()
        end_bals = cash_view_indexed.resample(freq_map[freq])['Cash_Balance'].last()
        end_bals.index = cf.columns
        cf.loc['Ending Cash Balance'] = end_bals
        
        render_financial_statement(cf, "")

elif view == "Production & Sales":
    st.title("Production & Sales Mix")
    c1, c2 = st.columns([2, 1])
    with c1:
        st.subheader("Production Manifest")
        df_units = pd.read_sql("SELECT * FROM production_unit", engine)
        edited = st.data_editor(df_units.sort_values('build_date'), column_config={"id": st.column_config.NumberColumn(disabled=True)}, hide_index=True, height=500, use_container_width=True)
        if st.button("💾 Save Changes"):
            with engine.connect() as conn:
                for _, r in edited.iterrows():
                    conn.execute(text("UPDATE production_unit SET sales_channel=:c, status=:s WHERE id=:i"), {"c": r['sales_channel'], "s": r['status'], "i": r['id']})
                conn.commit()
            st.success("Saved!")
            st.rerun()
    with c2:
        st.subheader("Smart Planner")
        start_date = st.date_input("Production Start", value=date(2026, 1, 1))
        df_units['Month'] = pd.to_datetime(df_units['build_date']).dt.strftime('%Y-%m')
        exist = df_units.groupby('Month').size()
        dates = pd.date_range('2026-01-01', '2027-12-01', freq='MS')
        plan = [{"Month": d.date(), "Target": int(exist.get(d.strftime('%Y-%m'), 0))} for d in dates]
        edit_plan = st.data_editor(pd.DataFrame(plan), hide_index=True, height=400)
        if st.button("🚀 Smart Regenerate"):
            with engine.connect() as conn:
                conn.execute(text("DELETE FROM production_unit WHERE status = 'PLANNED'"))
                last_sn = conn.execute(text("SELECT serial_number FROM production_unit ORDER BY id DESC LIMIT 1")).scalar()
                sn = int(''.join(filter(str.isdigit, last_sn))) + 1 if last_sn else 1
                for _, r in edit_plan.iterrows():
                    tgt = r['Target']
                    if tgt == 0: continue
                    m_str = r['Month'].strftime('%Y-%m')
                    locked = conn.execute(text(f"SELECT COUNT(*) FROM production_unit WHERE strftime('%Y-%m', build_date) = '{m_str}' AND status != 'PLANNED'")).scalar()
                    build = tgt - locked
                    if build <= 0: continue
                    dt_obj = r['Month']
                    thresh = start_date if dt_obj.year==start_date.year and dt_obj.month==start_date.month else None
                    if date(dt_obj.year, dt_obj.month, calendar.monthrange(dt_obj.year, dt_obj.month)[1]) < start_date: continue
                    wd = get_workdays(dt_obj.year, dt_obj.month, thresh)
                    if not wd: continue
                    d_qty = math.floor(build*0.25)
                    pool = ['DIRECT']*d_qty + ['DEALER']*(build-d_qty)
                    d_idx = 0
                    for t in pool:
                        conn.execute(text("INSERT INTO production_unit (serial_number, build_date, sales_channel, status) VALUES (:s, :b, :c, 'PLANNED')"), {"s": f"IDX-{sn:04d}", "b": wd[d_idx], "c": t})
                        sn += 1
                        d_idx = (d_idx + 1) % len(wd)
                conn.commit()
            st.success("Done!")
            st.rerun()

elif view == "OpEx Planning":
    st.title("OpEx Budget")
    t1, t2 = st.tabs(["Headcount", "R&D Expenses"])
    with t1:
        st.subheader("Headcount Planner")
        df_r = pd.read_sql("SELECT * FROM opex_roles", engine)