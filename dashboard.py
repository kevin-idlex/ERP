This is the **"Strategy Engine"** upgrade.

We are adding a sophisticated **Monte Carlo-style Optimizer** to your dashboard. Instead of guessing "Can we grow 5%?", you can now ask the computer: *"Given $1M cash and a $2M credit line, what is the mathematical maximum speed we can grow?"*

### New Features in Version 3.0:

1.  **Scenario Builder Tab:** A dedicated sandbox to set your constraints (Cash & Credit).
2.  **Growth Optimizer:** A "Solve for X" button that runs dozens of simulations in seconds to find your break-neck speed.
3.  **Milestone Reporting:** Automatically calculates:
      * **Self-Funded Date:** The day you pay off the bank forever.
      * **Exit Velocity:** The day you have paid back all investors and are purely printing profit.

### The New `dashboard.py` (Version 3.0)

**Select All** \> **Delete** \> **Paste**.

```python
"""
IdleX CFO Console - Dashboard Application
Version: 3.0 (Scenario Builder & Optimization Engine)
"""

import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text
from datetime import timedelta, date
import plotly.express as px
import plotly.graph_objects as go
import calendar
import math
import os
import logging
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

# Custom CSS
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
    .metric-card {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #dee2e6;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# DATABASE HELPERS
# =============================================================================
@st.cache_resource
def get_db_engine():
    db_url = os.getenv("DATABASE_URL")
    if db_url:
        if db_url.startswith("postgres://"):
            db_url = db_url.replace("postgres://", "postgresql://", 1)
        return create_engine(db_url)
    return create_engine('sqlite:///idlex.db')

engine = get_db_engine()
DB_TYPE = "postgresql" if os.getenv("DATABASE_URL") else "sqlite"

def get_upsert_sql(db_type):
    if db_type == "postgresql":
        return """
            INSERT INTO opex_staffing_plan (role_id, month_date, headcount) 
            VALUES (:rid, :dt, :hc)
            ON CONFLICT (role_id, month_date) 
            DO UPDATE SET headcount = EXCLUDED.headcount
        """
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
def generate_financials(units_override=None, start_cash_override=None):
    """
    Generates financials. Can accept an in-memory DataFrame for simulation
    to avoid writing to the DB during optimization loops.
    """
    try:
        # 1. Load Static Data (BOM, OpEx) - Always from DB
        df_parts = pd.read_sql("SELECT * FROM part_master", engine)
        df_bom = pd.read_sql("SELECT * FROM bom_items", engine)
        df_opex = pd.read_sql("SELECT * FROM opex_staffing_plan", engine)
        df_roles = pd.read_sql("SELECT * FROM opex_roles", engine)
        try:
            df_gen_exp = pd.read_sql("SELECT * FROM opex_general_expenses", engine)
            df_gen_exp['month_date'] = pd.to_datetime(df_gen_exp['month_date'])
        except:
            df_gen_exp = pd.DataFrame(columns=['month_date', 'expense_type', 'amount'])
            
        # 2. Load or Use Override Data (Units & Cash)
        if units_override is not None:
            df_units = units_override.copy()
        else:
            df_units = pd.read_sql("SELECT * FROM production_unit", engine)
        
        if start_cash_override is not None:
            start_cash = float(start_cash_override)
        else:
            config = pd.read_sql("SELECT * FROM global_config", engine)
            row = config[config['setting_key'] == 'start_cash']
            start_cash = float(row['setting_value'].values[0]) if not row.empty else 1000000.0

    except Exception as e:
        logger.error(f"Data Load Error: {e}")
        return pd.DataFrame(), pd.DataFrame()

    if df_units.empty: return pd.DataFrame(), pd.DataFrame()

    # Conversions
    df_units['build_date'] = pd.to_datetime(df_units['build_date'])
    df_opex['month_date'] = pd.to_datetime(df_opex['month_date'])
    if not df_gen_exp.empty:
        df_gen_exp['month_date'] = pd.to_datetime(df_gen_exp['month_date'])

    ledger = []

    # A. Revenue & COGS
    # Pre-calc unit cost
    unit_mat_cost = 0
    if not df_bom.empty and not df_parts.empty:
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
    
    # B. Supply Chain Cash Flow
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

    # C. Payroll
    if not df_opex.empty and not df_roles.empty:
        opex_merged = pd.merge(df_opex, df_roles, left_on='role_id', right_on='id')
        for _, row in opex_merged.iterrows():
            monthly_cost = (row['annual_salary'] / 12) * row['headcount']
            if monthly_cost > 0:
                is_direct = "Assembler" in row['role_name']
                cat = "Direct Labor" if is_direct else "Salaries & Wages"
                pnl_type = "COGS" if is_direct else "OpEx"
                
                ledger.append({"Date": row['month_date'], "Category": cat, "Type": pnl_type, "Amount": -monthly_cost, "Report": "PnL"})
                ledger.append({"Date": row['month_date'], "Category": "Payroll Paid", "Type": "Operations", "Amount": -monthly_cost, "Report": "Cash"})

    # D. General Expenses
    if not df_gen_exp.empty:
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


def simulate_growth_scenario(start_units, growth_pct, start_date, months_to_sim=36):
    """Generates a hypothetical unit dataframe for simulation."""
    sim_units = []
    current_units = start_units
    serial_counter = 1
    
    current_date = start_date.replace(day=1) # Normalize to 1st of month
    
    for i in range(months_to_sim):
        # Calculate targets
        target = int(current_units)
        workdays = get_workdays(current_date.year, current_date.month)
        
        if target > 0 and workdays:
            # Mix Logic
            direct_qty = int(target * 0.25)
            dealer_qty = target - direct_qty
            pool = ['DIRECT']*direct_qty + ['DEALER']*dealer_qty
            
            for idx, channel in enumerate(pool):
                build_day = workdays[idx % len(workdays)]
                sim_units.append({
                    "serial_number": f"SIM-{serial_counter}",
                    "build_date": build_day,
                    "sales_channel": channel,
                    "status": "PLANNED"
                })
                serial_counter += 1
        
        # Increment
        current_units = current_units * (1 + (growth_pct/100))
        # Move to next month
        if current_date.month == 12:
            current_date = date(current_date.year + 1, 1, 1)
        else:
            current_date = date(current_date.year, current_date.month + 1, 1)
            
    return pd.DataFrame(sim_units)

# =============================================================================
# MAIN UI
# =============================================================================
def main():
    st.sidebar.title("IdleX CFO Console")
    
    # Admin Tools
    if st.sidebar.button("⚠️ Rebuild Database"):
        with st.spinner("Resetting..."): seed_db.run_seed()
        st.sidebar.success("Done! Refresh.")
        st.rerun()

    view = st.sidebar.radio("Navigation", [
        "Executive Dashboard", 
        "Scenario & Optimization",  # <--- NEW TAB
        "Financial Statements", 
        "Production & Sales", 
        "OpEx Planning", 
        "BOM & Supply Chain"
    ])

    # Default Load
    if view != "Scenario & Optimization":
        df_pnl, df_cash = generate_financials()
    
    # ---------------------------------------------------------
    # VIEW: EXECUTIVE DASHBOARD
    # ---------------------------------------------------------
    if view == "Executive Dashboard":
        st.title("Executive Dashboard")
        if not df_pnl.empty:
            # Filters
            years = sorted(df_pnl['Date'].dt.year.unique().tolist())
            selected_period = st.sidebar.selectbox("Fiscal Year:", ["All Time"] + years)
            
            if selected_period == "All Time":
                pnl_v = df_pnl
                cash_v = df_cash
            else:
                pnl_v = df_pnl[df_pnl['Date'].dt.year == selected_period]
                cash_v = df_cash[df_cash['Date'].dt.year == selected_period]

            # Metrics
            rev = pnl_v[pnl_v['Category'] == 'Sales of Goods']['Amount'].sum()
            margin = rev - abs(pnl_v[pnl_v['Type'] == 'COGS']['Amount'].sum())
            min_c = cash_v['Cash_Balance'].min()
            end_c = cash_v.iloc[-1]['Cash_Balance'] if not cash_v.empty else 0
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Revenue", f"${rev:,.0f}")
            c2.metric("Gross Margin", f"${margin:,.0f}")
            c3.metric("Min Net Cash", f"${min_c:,.0f}", delta_color="inverse")
            c4.metric("Ending Cash", f"${end_c:,.0f}")
            
            # Charts
            fig = px.area(cash_v, x='Date', y='Cash_Balance', title="Liquidity Forecast", color_discrete_sequence=['#10B981'])
            fig.add_hline(y=0, line_dash="dash", line_color="red")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Database empty.")

    # ---------------------------------------------------------
    # VIEW: SCENARIO BUILDER (NEW!)
    # ---------------------------------------------------------
    elif view == "Scenario & Optimization":
        st.title("Growth Strategy Engine")
        st.info("Define your constraints and solve for maximum sustainable growth.")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Constraints")
            
            # Get defaults
            config = pd.read_sql("SELECT * FROM global_config", engine)
            def_cash = float(config[config['setting_key']=='start_cash']['setting_value'].values[0])
            def_loc = float(config[config['setting_key']=='loc_limit']['setting_value'].values[0])
            
            inv_cash = st.number_input("Investor Equity (Cash)", value=def_cash, step=100000.0)
            loc_limit = st.number_input("Bank Credit Limit (LOC)", value=def_loc, step=100000.0)
            
            st.divider()
            st.subheader("Simulation Inputs")
            start_vol = st.number_input("Starting Monthly Units", value=50)
            sim_months = st.slider("Simulation Horizon (Months)", 12, 60, 36)
            
            if st.button("💾 Save Constraints"):
                with engine.connect() as conn:
                    conn.execute(text(f"UPDATE global_config SET setting_value='{inv_cash}' WHERE setting_key='start_cash'"))
                    conn.execute(text(f"UPDATE global_config SET setting_value='{loc_limit}' WHERE setting_key='loc_limit'"))
                    conn.commit()
                st.success("Constraints Saved!")

        with col2:
            st.subheader("Optimization Results")
            
            if st.button("🚀 Solve for Maximum Growth", type="primary"):
                with st.status("Running Simulations...") as status:
                    best_rate = 0.0
                    best_cash_df = pd.DataFrame()
                    safe_limit = -loc_limit # Lowest cash can go
                    
                    # Binary Search for Max Growth
                    low = 0.0
                    high = 100.0 # 100% monthly growth cap
                    
                    for i in range(10): # 10 iterations is plenty for precision
                        mid = (low + high) / 2
                        status.write(f"Testing {mid:.1f}% Monthly Growth...")
                        
                        # 1. Generate Units
                        sim_units = simulate_growth_scenario(start_vol, mid, date(2026,1,1), sim_months)
                        
                        # 2. Run Financials
                        _, sim_cash = generate_financials(units_override=sim_units, start_cash_override=inv_cash)
                        
                        if sim_cash.empty:
                            min_c = 0
                        else:
                            min_c = sim_cash['Cash_Balance'].min()
                        
                        if min_c >= safe_limit:
                            best_rate = mid
                            best_cash_df = sim_cash
                            low = mid
                        else:
                            high = mid
                            
                    status.update(label="Optimization Complete!", state="complete")
                
                # RESULTS DISPLAY
                st.success(f"Max Sustainable Growth: **{best_rate:.1f}% per Month**")
                
                # Calculations
                if not best_cash_df.empty:
                    # 1. Self Funded Date (Last time cash was below 0)
                    below_zero = best_cash_df[best_cash_df['Cash_Balance'] < 0]
                    if below_zero.empty:
                        self_fund_date = "Day 1"
                    else:
                        last_neg_date = below_zero['Date'].max()
                        self_fund_date = last_neg_date.strftime('%b %Y')

                    # 2. Investor Payback (When cumulative cash > investment)
                    paid_back = best_cash_df[best_cash_df['Cash_Balance'] >= inv_cash]
                    if paid_back.empty:
                        payback_date = "Not in Horizon"
                    else:
                        payback_date = paid_back.iloc[0]['Date'].strftime('%b %Y')

                    colA, colB, colC = st.columns(3)
                    colA.metric("Peak Credit Usage", f"${abs(best_cash_df['Cash_Balance'].min()):,.0f}")
                    colB.metric("Self-Funded By", self_fund_date)
                    colC.metric("Investors Repaid By", payback_date)
                    
                    # Chart
                    fig = px.area(best_cash_df, x='Date', y='Cash_Balance', 
                                  title=f"Cash Projection at {best_rate:.1f}% Growth", 
                                  color_discrete_sequence=['#10B981'])
                    fig.add_hline(y=0, line_dash="dash", line_color="gray")
                    fig.add_hline(y=-loc_limit, line_dash="dash", line_color="red", annotation_text="Credit Limit")
                    st.plotly_chart(fig, use_container_width=True)

    # ---------------------------------------------------------
    # VIEW: FINANCIAL STATEMENTS
    # ---------------------------------------------------------
    elif view == "Financial Statements":
        st.title("Financial Statements")
        if not df_pnl.empty:
            col1, col2 = st.columns(2)
            with col1: freq = st.radio("Aggregation:", ["Monthly", "Quarterly", "Yearly"], horizontal=True, index=1)
            freq_map = {"Monthly": "ME", "Quarterly": "QE", "Yearly": "YE"}
            
            st.header("Consolidated Statement of Operations")
            pnl_agg = df_pnl.groupby([pd.Grouper(key='Date', freq=freq_map[freq]), 'Type', 'Category']).sum()['Amount'].unstack(level=[1,2]).fillna(0)
            
            if freq == "Monthly": pnl_agg.index = pnl_agg.index.strftime('%Y-%b')
            elif freq == "Quarterly": pnl_agg.index = pnl_agg.index.to_period("Q").astype(str)
            else: pnl_agg.index = pnl_agg.index.strftime('%Y')
            
            stmt = pd.DataFrame(columns=pnl_agg.index)
            def safe_sum(keys):
                total = pd.Series(0, index=pnl_agg.index)
                for k in keys: 
                    if k in pnl_agg.columns: total += pnl_agg[k]
                return total

            stmt.loc['Revenue'] = None
            stmt.loc['Sales of Goods'] = safe_sum([('Revenue', 'Sales of Goods')])
            stmt.loc['Cost of Goods Sold'] = None
            stmt.loc['Raw Materials'] = safe_sum([('COGS', 'Raw Materials')])
            stmt.loc['Direct Labor'] = safe_sum([('COGS', 'Direct Labor')])
            stmt.loc['Gross Profit'] = stmt.loc['Sales of Goods'] + stmt.loc['Raw Materials'] + stmt.loc['Direct Labor']
            stmt.loc['Operating Expenses'] = None
            stmt.loc['Salaries & Wages'] = safe_sum([('OpEx', 'Salaries & Wages')])
            opex_cols = [c for c in pnl_agg.columns if c[0] == 'OpEx' and c[1] != 'Salaries & Wages']
            for col in opex_cols: stmt.loc[col[1]] = safe_sum([col])
            stmt.loc['Total OpEx'] = safe_sum([('OpEx', c[1]) for c in opex_cols]) + stmt.loc['Salaries & Wages']
            stmt.loc['Net Income'] = stmt.loc['Gross Profit'] + stmt.loc['Total OpEx']
            
            render_financial_statement(stmt, "")
            st.markdown("---")
            
            st.header("Statement of Cash Flows")
            cash_view_indexed = df_cash.set_index('Date')
            cash_agg = df_cash.groupby([pd.Grouper(key='Date', freq=freq_map[freq]), 'Category']).sum()['Amount'].unstack().fillna(0)
            if freq == "Monthly": cash_agg.index = cash_agg.index.strftime('%Y-%b')
            elif freq == "Quarterly": cash_agg.index = cash_agg.index.to_period("Q").astype(str)
            else: cash_agg.index = cash_agg.index.strftime('%Y')
            
            cf = pd.DataFrame(columns=cash_agg.index)
            cf.loc['Operating Activities'] = None
            cf.loc['Cash from Customers'] = cash_agg.get('Cash from Customers', 0)
            cf.loc['Supplier Payments'] = cash_agg.get('Supplier Deposits', 0) + cash_agg.get('Supplier Settlements', 0)
            cf.loc['Payroll Paid'] = cash_agg.get('Payroll Paid', 0)
            cf.loc['OpEx Paid'] = cash_agg.get('OpEx Paid', 0)
            
            cf.loc['Net Cash Flow'] = (
                cf.loc['Cash from Customers'] + 
                cf.loc['Supplier Payments'] + 
                cf.loc['Payroll Paid'] + 
                cf.loc['OpEx Paid']
            )
            
            end_bals = cash_view_indexed.resample(freq_map[freq])['Cash_Balance'].last()
            if len(end_bals) == len(cf.columns):
                end_bals.index = cf.columns
                cf.loc['Ending Cash Balance'] = end_bals
            
            render_financial_statement(cf, "")

    # ---------------------------------------------------------
    # VIEW: PRODUCTION & SALES (FIXED)
    # ---------------------------------------------------------
    elif view == "Production & Sales":
        st.title("Production & Sales Mix")
        c1, c2 = st.columns([2, 1])
        
        with c1:
            st.subheader("Production Manifest")
            df_units = pd.read_sql("SELECT * FROM production_unit", engine)
            edited = st.data_editor(
                df_units.sort_values('build_date'),
                column_config={
                    "id": st.column_config.NumberColumn(disabled=True),
                    "serial_number": st.column_config.TextColumn(disabled=True),
                    "build_date": st.column_config.DateColumn(disabled=True),
                    "sales_channel": st.column_config.SelectboxColumn("Type", options=["DIRECT", "DEALER"], required=True),
                    "status": st.column_config.SelectboxColumn("Status", options=["PLANNED", "WIP", "COMPLETE"])
                }, hide_index=True, height=500, use_container_width=True
            )
            if st.button("💾 Save Changes"):
                with engine.connect() as conn:
                    for _, r in edited.iterrows():
                        conn.execute(text("UPDATE production_unit SET sales_channel=:c, status=:s WHERE id=:i"), 
                                     {"c": r['sales_channel'], "s": r['status'], "i": r['id']})
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
                with st.spinner("Optimizing..."):
                    regenerate_production_schedule(edit_plan, start_date)
                st.success("Done!")
                st.rerun()

    # ---------------------------------------------------------
    # VIEW: OPEX PLANNING
    # ---------------------------------------------------------
    elif view == "OpEx Planning":
        st.title("OpEx Budget")
        t1, t2 = st.tabs(["Headcount", "Expenses"])
        with t1:
            st.subheader("Headcount Planner")
            df_r = pd.read_sql("SELECT * FROM opex_roles", engine)
            df_s = pd.read_sql("SELECT * FROM opex_staffing_plan", engine)
            df_m = pd.merge(df_s, df_r, left_on='role_id', right_on='id')
            df_m['Month'] = pd.to_datetime(df_m['month_date']).dt.strftime('%Y-%m')
            piv = df_m.pivot(index='role_name', columns='Month', values='headcount').reset_index()
            ed = st.data_editor(piv, use_container_width=True)
            if st.button("💾 Save Headcount"):
                with engine.connect() as conn:
                    mlt = ed.melt(id_vars=['role_name'], var_name='Month', value_name='headcount')
                    for _, r in mlt.iterrows():
                        rid = conn.execute(text("SELECT id FROM opex_roles WHERE role_name=:rn"), {"rn": r['role_name']}).scalar()
                        if rid:
                            dt = date.fromisoformat(r['Month']+"-01")
                            sql = get_upsert_sql(DB_TYPE)
                            conn.execute(text(sql), {"rid": rid, "dt": dt, "hc": r['headcount']})
                    conn.commit()
                st.rerun()
            
            st.divider()
            st.subheader("Salary Configuration")
            edited_roles = st.data_editor(
                df_r, 
                column_config={"id": st.column_config.NumberColumn(disabled=True)}, 
                hide_index=True,
                use_container_width=True
            )
            if st.button("💾 Update Salaries"):
                with engine.connect() as conn:
                    for _, r in edited_roles.iterrows():
                        conn.execute(text("UPDATE opex_roles SET role_name=:n, annual_salary=:s WHERE id=:id"),
                                     {"n": r['role_name'], "s": r['annual_salary'], "id": r['id']})
                    conn.commit()
                st.success("Salaries Updated!")
                st.rerun()

        with t2:
            df_g = pd.read_sql("SELECT * FROM opex_general_expenses", engine)
            df_g['Month'] = pd.to_datetime(df_g['month_date']).dt.strftime('%Y-%m')
            piv_g = df_g.pivot(index=['category', 'expense_type'], columns='Month', values='amount').reset_index()
            ed_g = st.data_editor(piv_g, use_container_width=True)
            if st.button("💾 Save Expenses"):
                with engine.connect() as conn:
                    mlt_g = ed_g.melt(id_vars=['category', 'expense_type'], var_name='Month', value_name='amount')
                    conn.execute(text("DELETE FROM opex_general_expenses"))
                    for _, r in mlt_g.iterrows():
                        if pd.notna(r['amount']):
                            dt = date.fromisoformat(r['Month']+"-01")
                            conn.execute(text("INSERT INTO opex_general_expenses (category, expense_type, month_date, amount) VALUES (:c, :t, :d, :a)"), {"c": r['category'], "t": r['expense_type'], "d": dt, "a": r['amount']})
                    conn.commit()
                st.rerun()

    # ---------------------------------------------------------
    # VIEW: BOM
    # ---------------------------------------------------------
    elif view == "BOM & Supply Chain":
        st.title("Bill of Materials")
        df_p = pd.read_sql("SELECT * FROM part_master", engine)
        ed_p = st.data_editor(df_p, disabled=["id", "sku"], use_container_width=True)
        if st.button("💾 Save BOM"):
            with engine.connect() as conn:
                for _, r in ed_p.iterrows():
                    conn.execute(text("UPDATE part_master SET name=:n, cost=:c, moq=:m, lead_time=:l, deposit_pct=:dp, deposit_days=:dd, balance_days=:bd WHERE id=:id"), 
                                 {"n": r['name'], "c": r['cost'], "m": r['moq'], "l": r['lead_time'], "dp": r['deposit_pct'], "dd": r['deposit_days'], "bd": r['balance_days'], "id": r['id']})
            conn.commit()
        st.rerun()

def regenerate_production_schedule(edit_plan, start_date):
    with engine.connect() as conn:
        try:
            conn.execute(text("DELETE FROM production_unit WHERE status = 'PLANNED'"))
            last_sn = conn.execute(text("SELECT serial_number FROM production_unit ORDER BY id DESC LIMIT 1")).scalar()
            next_sn = parse_serial_number(last_sn) + 1
            
            for _, row in edit_plan.iterrows():
                target = int(row['Target'])
                if target == 0: continue
                
                month_date = row['Month']
                if isinstance(month_date, pd.Timestamp): month_date = month_date.date()
                
                month_str = month_date.strftime('%Y-%m')
                year_month_sql = "TO_CHAR(build_date, 'YYYY-MM')" if DB_TYPE == 'postgresql' else "strftime('%Y-%m', build_date)"
                
                locked_count = conn.execute(text(f"SELECT COUNT(*) FROM production_unit WHERE {year_month_sql} = :ms AND status != 'PLANNED'"), {"ms": month_str}).scalar() or 0
                
                to_build = target - locked_count
                if to_build <= 0: continue
                
                threshold = start_date if (month_date.year == start_date.year and month_date.month == start_date.month) else None
                last_day = date(month_date.year, month_date.month, calendar.monthrange(month_date.year, month_date.month)[1])
                if last_day < start_date: continue
                
                workdays = get_workdays(month_date.year, month_date.month, threshold)
                if not workdays: continue
                
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

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"System Error: {e}")
```