"""
IdleX CFO Console - Dashboard Application
Version: 2.2 (Enhanced OpEx Planner)
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

    if df_units.empty:
        return pd.DataFrame(), pd.DataFrame()

    # Conversions
    df_units['build_date'] = pd.to_datetime(df_units['build_date'])
    df_opex['month_date'] = pd.to_datetime(df_opex['month_date'])
    df_gen_exp['month_date'] = pd.to_datetime(df_gen_exp['month_date'])
    
    start_cash = 1000000.0
    if not config.empty and 'setting_key' in config.columns:
        row = config[config['setting_key'] == 'start_cash']
        if not row.empty: start_cash = float(row['setting_value'].values[0])

    ledger = []

    # 1. Revenue & COGS
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

    # 2. Supply Chain
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

    # 4. General Expenses
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

# =============================================================================
# MAIN UI
# =============================================================================
def main():
    st.sidebar.title("IdleX CFO Console")
    if st.sidebar.button("⚠️ Rebuild Database"):
        with st.spinner("Resetting Database..."):
            seed_db.run_seed()
        st.sidebar.success("Done! Refresh page.")
        st.rerun()

    view = st.sidebar.radio("Navigation", ["Executive Dashboard", "Financial Statements", "Production & Sales", "OpEx Planning", "BOM & Supply Chain"])

    df_pnl, df_cash = generate_financials()

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
        st.title("Executive Dashboard")
        if not pnl_view.empty:
            rev = pnl_view[pnl_view['Category'] == 'Sales of Goods']['Amount'].sum()
            cogs = abs(pnl_view[pnl_view['Type'] == 'COGS']['Amount'].sum())
            margin = rev - cogs
            min_cash = cash_view['Cash_Balance'].min()
            end_cash = cash_view.iloc[-1]['Cash_Balance']
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Revenue", f"${rev:,.0f}")
            c2.metric("Gross Margin", f"${margin:,.0f}")
            c3.metric("Min Net Cash", f"${min_cash:,.0f}", delta_color="inverse")
            c4.metric("Ending Cash", f"${end_cash:,.0f}")
            
            fig = px.area(cash_view, x='Date', y='Cash_Balance', title="Liquidity Forecast", color_discrete_sequence=['#10B981'])
            fig.add_hline(y=0, line_dash="dash", line_color="red")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Database empty. Please use the Rebuild button.")

    elif view == "Financial Statements":
        st.title("Financial Statements")
        if not pnl_view.empty:
            c1, c2 = st.columns(2)
            with c1: freq = st.radio("Aggregation:", ["Monthly", "Quarterly", "Yearly"], horizontal=True, index=1)
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
            stmt.loc['Salaries & Wages'] = safe_sum([('OpEx', 'Salaries & Wages')])
            opex_cols = [c for c in pnl_agg.columns if c[0] == 'OpEx' and c[1] != 'Salaries & Wages']
            for col in opex_cols: stmt.loc[col[1]] = safe_sum([col])
            stmt.loc['Total OpEx'] = safe_sum([('OpEx', c[1]) for c in opex_cols]) + stmt.loc['Salaries & Wages']
            stmt.loc['Net Income'] = stmt.loc['Gross Profit'] + stmt.loc['Total OpEx']
            
            render_financial_statement(stmt, "")
            st.markdown("---")
            
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
            if len(end_bals) == len(cf.columns):
                end_bals.index = cf.columns
                cf.loc['Ending Cash Balance'] = end_bals
            
            render_financial_statement(cf, "")

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
                st.success("Schedule Optimized!")
                st.rerun()

    # ------------------- NEW: OPEX BUDGET WITH DYNAMIC CATEGORIES -------------------
    elif view == "OpEx Planning":
        st.title("OpEx Budget")
        tab1, tab2, tab3 = st.tabs(["A. Headcount & Payroll", "B. R&D Expenses", "C. SG&A Expenses"])
        
        # TAB A: HEADCOUNT
        with tab1:
            st.subheader("Headcount & Roles")
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
            st.subheader("Role Management")
            with st.expander("Add/Edit Roles & Salaries"):
                edited_roles = st.data_editor(df_r, column_config={"id": st.column_config.NumberColumn(disabled=True)}, hide_index=True, use_container_width=True, num_rows="dynamic")
                if st.button("💾 Update Roles"):
                    with engine.connect() as conn:
                        # Delete old mapping to avoid orphans (simple strategy)
                        conn.execute(text("DELETE FROM opex_roles")) 
                        # Re-insert
                        for _, r in edited_roles.iterrows():
                            conn.execute(text("INSERT INTO opex_roles (role_name, annual_salary) VALUES (:n, :s)"), {"n": r['role_name'], "s": r['annual_salary']})
                        # Note: This resets headcount mapping if names change. In production, use IDs.
                        conn.commit()
                    st.success("Roles Updated!")
                    st.rerun()

        # SHARED FUNCTION FOR R&D / SG&A
        def render_expense_tab(exp_type):
            st.subheader(f"{exp_type} Budget")
            
            # 1. Get Data for this Type
            df_all = pd.read_sql(f"SELECT * FROM opex_general_expenses WHERE expense_type = '{exp_type}'", engine)
            
            if not df_all.empty:
                df_all['Month'] = pd.to_datetime(df_all['month_date']).dt.strftime('%Y-%m')
                pivot_exp = df_all.pivot(index='category', columns='Month', values='amount').reset_index()
            else:
                # Initialize blank grid if empty
                dates = pd.date_range('2026-01-01', '2027-06-01', freq='MS')
                pivot_exp = pd.DataFrame(columns=['category'] + [d.strftime('%Y-%m') for d in dates])

            # 2. Editor
            edited_exp = st.data_editor(pivot_exp, use_container_width=True, num_rows="dynamic", key=f"editor_{exp_type}")
            
            # 3. Save Logic
            if st.button(f"💾 Save {exp_type} Budget"):
                with engine.connect() as conn:
                    # Wipe existing entries for this type only
                    conn.execute(text(f"DELETE FROM opex_general_expenses WHERE expense_type = '{exp_type}'"))
                    
                    # Melt & Insert
                    melted = edited_exp.melt(id_vars=['category'], var_name='Month', value_name='amount')
                    for _, r in melted.iterrows():
                        if pd.notna(r['amount']):
                            dt = date.fromisoformat(r['Month']+"-01")
                            conn.execute(text("""
                                INSERT INTO opex_general_expenses (category, expense_type, month_date, amount) 
                                VALUES (:c, :t, :d, :a)
                            """), {"c": r['category'], "t": exp_type, "d": dt, "a": r['amount']})
                    conn.commit()
                st.success(f"{exp_type} Budget Saved!")
                st.rerun()

        # TAB B: R&D
        with tab2:
            render_expense_tab("R&D")

        # TAB C: SG&A
        with tab3:
            render_expense_tab("SG&A")

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

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"System Error: {e}")