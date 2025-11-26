"""
IdleX ERP - Enterprise Edition
Version: 7.0 (Unified Scenario & Financials)
"""

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
import math

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# BRAND CONFIGURATION
# =============================================================================
BRAND_NAVY = "#1E3466"
BRAND_BLUE = "#3A77D8"
BRAND_SLATE = "#A5ABB5"
BRAND_WHITE = "#FFFFFF"
MSRP_PRICE = 8500.00
DEALER_DISCOUNT_RATE = 0.75
DIRECT_SALES_TARGET_PCT = 0.25

st.set_page_config(page_title="IdleX ERP", layout="wide", page_icon="⚡")

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');
    html, body, [class*="css"] {{ font-family: 'Roboto', sans-serif; }}
    
    h1, h2, h3 {{ color: {BRAND_NAVY} !important; font-weight: 700; }}
    
    .financial-table {{
        font-family: 'Georgia', serif; font-size: 15px; border-collapse: collapse;
        width: 100%; color: #000000 !important; background-color: #ffffff; border: 1px solid #e0e0e0;
    }}
    .financial-table th {{
        text-align: right; background-color: {BRAND_NAVY}; color: white !important;
        padding: 12px; border-bottom: 3px solid {BRAND_BLUE};
    }}
    .financial-table td {{ padding: 10px 12px; border-bottom: 1px solid #f0f0f0; }}
    .financial-table .row-header {{ text-align: left; font-weight: 500; color: {BRAND_NAVY} !important; }}
    .financial-table .section-header {{ font-weight: bold; background-color: #f1f3f6; }}
    .financial-table .grand-total {{ font-weight: bold; color: {BRAND_BLUE} !important; border-top: 2px solid {BRAND_NAVY}; border-bottom: 3px double {BRAND_NAVY}; }}
    .financial-table .indent {{ padding-left: 25px; }}
</style>
""", unsafe_allow_html=True)

# =============================================================================
# DATABASE
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

# =============================================================================
# UTILITIES
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
    html = f"<h3 style='color:{BRAND_NAVY}; border-bottom: 2px solid {BRAND_BLUE};'>{title}</h3>"
    html += "<div style='border:1px solid #ddd; overflow-x:auto;'><table class='financial-table'>"
    html += "<thead><tr><th class='row-header'>Account</th>"
    for col in df.columns: html += f"<th>{col}</th>"
    html += "</tr></thead><tbody>"
    
    headers = ['Revenue', 'Cost of Goods Sold', 'Operating Expenses', 'Operating Activities']
    totals = ['Gross Profit', 'Net Cash Flow', 'Total OpEx']
    grands = ['Net Income', 'Ending Cash Balance']

    for index, row in df.iterrows():
        clean = str(index).strip()
        cls = "indent"
        if clean in headers: cls = "section-header"
        elif clean in totals: cls = "total-row"
        elif clean in grands: cls = "grand-total"
        
        html += f"<tr class='{cls}'><td class='row-header'>{clean}</td>"
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
# CORE LOGIC
# =============================================================================
def generate_financials(units_override=None, start_cash_override=None):
    try:
        df_parts = pd.read_sql("SELECT * FROM part_master", engine)
        df_bom = pd.read_sql("SELECT * FROM bom_items", engine)
        df_opex = pd.read_sql("SELECT * FROM opex_staffing_plan", engine)
        df_roles = pd.read_sql("SELECT * FROM opex_roles", engine)
        try:
            df_gen_exp = pd.read_sql("SELECT * FROM opex_general_expenses", engine)
            df_gen_exp['month_date'] = pd.to_datetime(df_gen_exp['month_date'])
        except: df_gen_exp = pd.DataFrame()

        if units_override is not None:
            df_units = units_override.copy()
        else:
            df_units = pd.read_sql("SELECT * FROM production_unit", engine)

        config = pd.read_sql("SELECT * FROM global_config", engine)
        
        if start_cash_override is not None:
            start_cash = float(start_cash_override)
        else:
            row = config[config['setting_key'] == 'start_cash']
            start_cash = float(row['setting_value'].values[0]) if not row.empty else 1000000.0

    except:
        return pd.DataFrame(), pd.DataFrame()

    if df_units.empty: return pd.DataFrame(), pd.DataFrame()

    df_units['build_date'] = pd.to_datetime(df_units['build_date'])
    df_opex['month_date'] = pd.to_datetime(df_opex['month_date'])
    if not df_gen_exp.empty: df_gen_exp['month_date'] = pd.to_datetime(df_gen_exp['month_date'])
    
    ledger = []
    
    # 1. Revenue & COGS
    unit_mat_cost = 0
    if not df_bom.empty and not df_parts.empty:
        bom_with_parts = pd.merge(df_bom, df_parts, left_on='part_id', right_on='id')
        unit_mat_cost = (bom_with_parts['qty_per_unit'] * bom_with_parts['cost']).sum()

    for _, u in df_units.iterrows():
        direct = u['sales_channel'] == 'DIRECT'
        rev = MSRP_PRICE if direct else MSRP_PRICE * DEALER_DISCOUNT_RATE
        lag = 0 if direct else 30
        dt = u['build_date']
        
        ledger.append({"Date": dt, "Category": "Product Sales", "Type": "Revenue", "Amount": rev, "Report": "PnL"})
        ledger.append({"Date": dt + timedelta(days=lag), "Category": "Customer Collections", "Type": "Ops", "Amount": rev, "Report": "Cash"})
        ledger.append({"Date": dt, "Category": "Materials", "Type": "COGS", "Amount": -unit_mat_cost, "Report": "PnL"})

    # 2. Supply Chain
    monthly = df_units.groupby(pd.Grouper(key='build_date', freq='MS')).size()
    for mo, cnt in monthly.items():
        if cnt == 0: continue
        for _, p in df_parts.iterrows():
            b = df_bom[df_bom['part_id'] == p['id']]
            if b.empty: continue
            cost = b.iloc[0]['qty_per_unit'] * cnt * p['cost']
            if p['deposit_pct'] > 0:
                ledger.append({"Date": mo + timedelta(days=int(p['deposit_days'])), "Category": "Supplier Deposits", "Type": "Ops", "Amount": -cost * p['deposit_pct'], "Report": "Cash"})
            if p['deposit_pct'] < 1:
                ledger.append({"Date": mo + timedelta(days=int(p['balance_days'])), "Category": "Supplier Settlements", "Type": "Ops", "Amount": -cost * (1 - p['deposit_pct']), "Report": "Cash"})

    # 3. Payroll
    if not df_staffing.empty:
        merged = pd.merge(df_opex, df_roles, left_on='role_id', right_on='id')
        for _, r in merged.iterrows():
            cost = (r['annual_salary']/12) * r['headcount']
            if cost > 0:
                labor = "Assembler" in r['role_name']
                ledger.append({"Date": r['month_date'], "Category": "Direct Labor" if labor else "Salaries", "Type": "COGS" if labor else "OpEx", "Amount": -cost, "Report": "PnL"})
                ledger.append({"Date": r['month_date'], "Category": "Payroll", "Type": "Ops", "Amount": -cost, "Report": "Cash"})

    # 4. Expenses
    if not df_gen_exp.empty:
        for _, e in df_gen_exp.iterrows():
            if e['amount'] > 0:
                ledger.append({"Date": e['month_date'], "Category": e['category'], "Type": "OpEx", "Amount": -e['amount'], "Report": "PnL"})
                ledger.append({"Date": e['month_date'], "Category": "OpEx Paid", "Type": "Ops", "Amount": -e['amount'], "Report": "Cash"})

    if not ledger: return pd.DataFrame(), pd.DataFrame()
    df = pd.DataFrame(ledger)
    pnl = df[df['Report']=='PnL'].sort_values('Date')
    cash = df[df['Report']=='Cash'].sort_values('Date')
    cash['Cash_Balance'] = cash['Amount'].cumsum() + start_cash
    return pnl, cash

def simulate_growth_scenario(start_units, growth_pct, start_date, months_to_sim=36):
    sim_units = []
    current_units = start_units
    serial_counter = 1
    current_date = start_date.replace(day=1)
    for i in range(months_to_sim):
        target = int(current_units)
        workdays = get_workdays(current_date.year, current_date.month)
        if target > 0 and workdays:
            direct_qty = int(target * DIRECT_SALES_TARGET_PCT)
            dealer_qty = target - direct_qty
            pool = ['DIRECT']*direct_qty + ['DEALER']*dealer_qty
            for idx, channel in enumerate(pool):
                build_day = workdays[idx % len(workdays)]
                sim_units.append({"serial_number": f"SIM-{serial_counter}", "build_date": build_day, "sales_channel": channel, "status": "PLANNED"})
                serial_counter += 1
        current_units = current_units * (1 + (growth_pct/100))
        if current_date.month == 12: current_date = date(current_date.year + 1, 1, 1)
        else: current_date = date(current_date.year, current_date.month + 1, 1)
    return pd.DataFrame(sim_units)

def push_scenario_to_production(sim_units):
    with engine.connect() as conn:
        try:
            # Delete only PLANNED units to preserve history
            conn.execute(text("DELETE FROM production_unit WHERE status = 'PLANNED'"))
            
            # Get next serial number
            last_sn = conn.execute(text("SELECT serial_number FROM production_unit ORDER BY id DESC LIMIT 1")).scalar()
            next_sn = parse_serial_number(last_sn) + 1
            
            for _, row in sim_units.iterrows():
                conn.execute(text("""
                    INSERT INTO production_unit (serial_number, build_date, sales_channel, status) 
                    VALUES (:sn, :bd, :ch, 'PLANNED')
                """), {
                    "sn": f"IDX-{next_sn:04d}",
                    "bd": row['build_date'],
                    "ch": row['sales_channel']
                })
                next_sn += 1
            
            conn.commit()
            return True
        except Exception as e:
            conn.rollback()
            logger.error(f"Push failed: {e}")
            return False

# =============================================================================
# MAIN UI
# =============================================================================
def main():
    st.sidebar.image("https://via.placeholder.com/200x60?text=IdleX+ERP", use_container_width=True)
    st.sidebar.title("IdleX ERP")
    
    if st.sidebar.button("⚠️ Rebuild Database"):
        with st.spinner("Resetting..."): seed_db.run_seed()
        st.sidebar.success("Done!")
        st.rerun()
    
    st.sidebar.divider()
    
    view = st.sidebar.radio("Navigation", [
        "📊 Dashboard",
        "🚀 Strategy & Scenarios",
        "📈 Financials",
        "📦 Production & Sales",
        "💵 OpEx Budget",
        "🛠️ Supply Chain"
    ])
    
    # Default financial load (skipped for scenario tab to be faster)
    pnl, cash = pd.DataFrame(), pd.DataFrame()
    if view != "🚀 Strategy & Scenarios":
        pnl, cash = generate_financials()
    
    # --- DASHBOARD ---
    if view == "📊 Dashboard":
        st.title("Executive Dashboard")
        if not pnl.empty:
            rev = pnl[pnl['Category'] == 'Product Sales']['Amount'].sum()
            end_cash = cash.iloc[-1]['Cash_Balance']
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Revenue", f"${rev:,.0f}")
            c2.metric("Ending Cash", f"${end_cash:,.0f}")
            
            fig = px.area(cash, x='Date', y='Cash_Balance', title="Liquidity Forecast", color_discrete_sequence=[BRAND_BLUE])
            fig.add_hline(y=0, line_dash="dash", line_color="red")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data. Please Rebuild Database.")

    # --- STRATEGY ENGINE ---
    elif view == "🚀 Strategy & Scenarios":
        st.title("Growth Strategy Engine")
        
        c1, c2 = st.columns([1, 2])
        
        with c1:
            st.subheader("Constraints")
            config = pd.read_sql("SELECT * FROM global_config", engine)
            def_cash = float(config[config['setting_key']=='start_cash']['setting_value'].values[0])
            def_loc = float(config[config['setting_key']=='loc_limit']['setting_value'].values[0])
            
            # Improved Number Inputs
            inv_cash = st.number_input("Investor Equity ($)", value=int(def_cash), step=100000, format="%d")
            loc_limit = st.number_input("Bank Credit Limit ($)", value=int(def_loc), step=100000, format="%d")
            
            st.divider()
            st.subheader("Growth Drivers")
            start_vol = st.number_input("Start Monthly Units", value=50, min_value=1)
            sim_months = st.slider("Forecast Horizon (Months)", 12, 60, 36)
            start_dt = st.date_input("Start Date", value=date(2026, 1, 1))
            
            if st.button("💾 Save Constraints"):
                with engine.connect() as conn:
                    conn.execute(text(f"UPDATE global_config SET setting_value='{inv_cash}' WHERE setting_key='start_cash'"))
                    conn.execute(text(f"UPDATE global_config SET setting_value='{loc_limit}' WHERE setting_key='loc_limit'"))
                    conn.commit()
                st.success("Saved!")

        with c2:
            st.subheader("Optimization")
            if st.button("🔍 Find Max Growth Rate", type="primary"):
                with st.status("Running Simulations...") as status:
                    best_rate = 0.0
                    best_cash_df = pd.DataFrame()
                    best_units_df = pd.DataFrame()
                    
                    limit = -loc_limit
                    low, high = 0.0, 100.0
                    
                    for i in range(10):
                        mid = (low + high) / 2
                        status.write(f"Testing {mid:.1f}% Growth...")
                        sim_u = simulate_growth_scenario(start_vol, mid, start_dt, sim_months)
                        _, sim_c = generate_financials(units_override=sim_u, start_cash_override=inv_cash)
                        
                        min_c = sim_c['Cash_Balance'].min() if not sim_c.empty else 0
                        
                        if min_c >= limit:
                            best_rate = mid
                            best_cash_df = sim_c
                            best_units_df = sim_u
                            low = mid
                        else:
                            high = mid
                    
                    # Store result in session state for "Push" button
                    st.session_state['sim_result'] = best_units_df
                    st.session_state['sim_rate'] = best_rate
                    st.session_state['sim_cash'] = best_cash_df
                    
                    status.update(label="Done!", state="complete")
            
            # Show Results if available
            if 'sim_result' in st.session_state:
                rate = st.session_state['sim_rate']
                cash_df = st.session_state['sim_cash']
                
                st.success(f"**Max Sustainable Growth: {rate:.1f}% per Month**")
                
                fig = px.area(cash_df, x='Date', y='Cash_Balance', title=f"Cash Flow @ {rate:.1f}% Growth", color_discrete_sequence=[BRAND_BLUE])
                fig.add_hline(y=-loc_limit, line_dash="dash", line_color="red", annotation_text="Credit Limit")
                st.plotly_chart(fig, use_container_width=True)
                
                st.warning("⚠️ Clicking below will OVERWRITE your live production schedule with this scenario.")
                if st.button("🚀 Push Scenario to Production Schedule", type="primary"):
                    with st.spinner("Updating Database..."):
                        success = push_scenario_to_production(st.session_state['sim_result'])
                        if success:
                            st.success("Production Schedule Updated! Check 'Production & Sales' tab.")
                        else:
                            st.error("Failed to update database.")

    # --- FINANCIALS (GAAP) ---
    elif view == "📈 Financials":
        st.title("Financial Statements")
        if not pnl.empty:
            freq = st.radio("Aggregation", ["Monthly", "Quarterly", "Yearly"], horizontal=True, index=1)
            fmap = {"Monthly": "ME", "Quarterly": "QE", "Yearly": "YE"}
            
            st.header("Consolidated Statement of Operations")
            agg = pnl.groupby([pd.Grouper(key='Date', freq=fmap[freq]), 'Type', 'Category']).sum()['Amount'].unstack(level=[1,2]).fillna(0)
            if freq=="Monthly": agg.index = agg.index.strftime('%b-%y')
            elif freq=="Quarterly": agg.index = agg.index.to_period("Q").astype(str)
            else: agg.index = agg.index.strftime('%Y')
            
            stmt = pd.DataFrame(columns=agg.index)
            def ssum(keys):
                t = pd.Series(0.0, index=agg.index)
                for k in keys: 
                    if k in agg.columns: t += agg[k]
                return t
            
            stmt.loc['Revenue'] = None
            stmt.loc['Product Sales'] = ssum([('Revenue', 'Product Sales')])
            stmt.loc['Cost of Goods Sold'] = None
            stmt.loc['Materials'] = ssum([('COGS', 'Materials')])
            stmt.loc['Direct Labor'] = ssum([('COGS', 'Direct Labor')])
            stmt.loc['Gross Profit'] = stmt.loc['Product Sales'] + stmt.loc['Materials'] + stmt.loc['Direct Labor']
            stmt.loc['Operating Expenses'] = None
            stmt.loc['Salaries'] = ssum([('OpEx', 'Salaries')])
            stmt.loc['Warranty Expense'] = ssum([('OpEx', 'Warranty Expense')])
            
            # Add dynamic OpEx categories
            opex_cols = [c for c in agg.columns if c[0] == 'OpEx' and c[1] not in ['Salaries', 'Warranty Expense']]
            for c in opex_cols: stmt.loc[c[1]] = ssum([c])
            
            stmt.loc['Total OpEx'] = ssum([('OpEx', c[1]) for c in opex_cols]) + stmt.loc['Salaries'] + stmt.loc['Warranty Expense']
            stmt.loc['Net Income'] = stmt.loc['Gross Profit'] + stmt.loc['Total OpEx']
            
            render_financial_statement(stmt, "")
            
            st.markdown("---")
            st.header("Statement of Cash Flows")
            cash_agg = cash.groupby([pd.Grouper(key='Date', freq=fmap[freq]), 'Category']).sum()['Amount'].unstack().fillna(0)
            if freq == "Monthly": cash_agg.index = cash_agg.index.strftime('%Y-%b')
            elif freq == "Quarterly": cash_agg.index = cash_agg.index.to_period("Q").astype(str)
            else: cash_agg.index = cash_agg.index.strftime('%Y')
            
            cf = pd.DataFrame(columns=cash_agg.index)
            cf.loc['Operating Activities'] = None
            cf.loc['Cash from Customers'] = cash_agg.get('Customer Collections', 0)
            cf.loc['Supplier Payments'] = cash_agg.get('Supplier Deposits', 0) + cash_agg.get('Supplier Settlements', 0)
            cf.loc['Payroll Paid'] = cash_agg.get('Payroll', 0)
            cf.loc['OpEx Paid'] = cash_agg.get('OpEx Paid', 0)
            
            # Calculate Net Flow
            cf.loc['Net Cash Flow'] = cf.loc['Cash from Customers'] + cf.loc['Supplier Payments'] + cf.loc['Payroll Paid'] + cf.loc['OpEx Paid']
            
            end_bals = cash.set_index('Date').resample(fmap[freq])['Cash_Balance'].last()
            if len(end_bals) == len(cf.columns):
                end_bals.index = cf.columns
                cf.loc['Ending Cash Balance'] = end_bals
            
            render_financial_statement(cf, "")

    # --- PRODUCTION ---
    elif view == "📦 Production & Sales":
        st.title("Production & Sales Planner")
        
        c1, c2 = st.columns([1, 2])
        
        with c1:
            st.subheader("Detailed Schedule")
            df_units = pd.read_sql("SELECT * FROM production_unit", engine)
            edited = st.data_editor(
                df_units.sort_values('build_date'), 
                column_config={"id": st.column_config.NumberColumn(disabled=True)},
                hide_index=True, height=500, use_container_width=True
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
            st.subheader("Volume Planner")
            
            # Generate Monthly Grid
            df_units['Month'] = pd.to_datetime(df_units['build_date']).dt.strftime('%Y-%m')
            existing = df_units.groupby('Month').size()
            dates = pd.date_range('2026-01-01', '2027-12-01', freq='MS')
            plan = [{"Month": d.date(), "Target": int(existing.get(d.strftime('%Y-%m'), 0))} for d in dates]
            
            edit_plan = st.data_editor(
                pd.DataFrame(plan),
                column_config={
                    "Month": st.column_config.DateColumn("Month", format="MMM YYYY", disabled=True),
                    "Target": st.column_config.NumberColumn("Monthly Units", min_value=0)
                },
                hide_index=True, height=400, use_container_width=True
            )
            
            start_date = st.date_input("Start Production From", value=date(2026, 1, 1))
            
            if st.button("🚀 Regenerate Production Schedule"):
                with st.spinner("Optimizing..."):
                    # Same logic as scenario push but from manual grid
                    regenerate_production_schedule(edit_plan, start_date) # Need to define this locally or move helper
                st.success("Done!")
                st.rerun()

    # --- OPEX ---
    elif view == "💵 OpEx Budget":
        st.title("OpEx Budget")
        # (Same as v6 code - kept separate for length)
        t1, t2 = st.tabs(["Headcount", "Expenses"])
        with t1:
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
                            conn.execute(text("INSERT OR REPLACE INTO opex_staffing_plan (id, role_id, month_date, headcount) VALUES ((SELECT id FROM opex_staffing_plan WHERE role_id=:rid AND month_date=:dt), :rid, :dt, :hc)"), {"rid": rid, "dt": dt, "hc": r['headcount']})
                    conn.commit()
                st.rerun()
            st.divider()
            edited_roles = st.data_editor(df_r, column_config={"id": None}, hide_index=True, use_container_width=True)
            if st.button("💾 Update Salaries"):
                with engine.connect() as conn:
                    for _, r in edited_roles.iterrows():
                        conn.execute(text("UPDATE opex_roles SET role_name=:n, annual_salary=:s WHERE id=:id"), {"n": r['role_name'], "s": r['annual_salary'], "id": r['id']})
                    conn.commit()
                st.success("Updated!")
                st.rerun()
        
        with t2:
             df_g = pd.read_sql("SELECT * FROM opex_general_expenses", engine)
             df_g['Month'] = pd.to_datetime(df_g['month_date']).dt.strftime('%Y-%m')
             piv_g = df_g.pivot(index=['category', 'expense_type'], columns='Month', values='amount').reset_index()
             ed_g = st.data_editor(piv_g, use_container_width=True)
             if st.button("💾 Save Expenses"):
                 with engine.connect() as conn:
                     mlt = ed_g.melt(id_vars=['category', 'expense_type'], var_name='Month', value_name='amount')
                     conn.execute(text("DELETE FROM opex_general_expenses"))
                     for _, r in mlt.iterrows():
                         if pd.notna(r['amount']):
                             dt = date.fromisoformat(r['Month']+"-01")
                             conn.execute(text("INSERT INTO opex_general_expenses (category, expense_type, month_date, amount) VALUES (:c, :t, :d, :a)"), {"c": r['category'], "t": r['expense_type'], "d": dt, "a": r['amount']})
                     conn.commit()
                 st.rerun()

    # --- BOM ---
    elif view == "🛠️ Supply Chain":
        st.title("Supply Chain")
        df_p = pd.read_sql("SELECT * FROM part_master", engine)
        ed_p = st.data_editor(df_p, disabled=["id", "sku"], use_container_width=True)
        if st.button("💾 Save BOM"):
            with engine.connect() as conn:
                for _, r in ed_p.iterrows():
                    conn.execute(text("UPDATE part_master SET name=:n, cost=:c, moq=:m, lead_time=:l, deposit_pct=:dp, deposit_days=:dd, balance_days=:bd WHERE id=:id"), 
                                 {"n": r['name'], "c": r['cost'], "m": r['moq'], "l": r['lead_time'], "dp": r['deposit_pct'], "dd": r['deposit_days'], "bd": r['balance_days'], "id": r['id']})
            conn.commit()
        st.rerun()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"System Error: {e}")