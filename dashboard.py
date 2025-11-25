import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text
from datetime import timedelta, date
import plotly.express as px
import plotly.graph_objects as go
import calendar
import math
import os

# 1. PAGE SETUP
st.set_page_config(page_title="IdleX CFO Console", layout="wide")

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

# 2. FINANCIAL ENGINE
def generate_financials():
    # Load Data
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
    
    # Convert Dates
    df_units['build_date'] = pd.to_datetime(df_units['build_date'])
    df_opex['month_date'] = pd.to_datetime(df_opex['month_date'])
    start_cash = float(config[config['setting_key']=='start_cash']['setting_value'].values[0])
    
    ledger = []
    
    # Revenue & Materials (Pre-calc Unit Cost)
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
        
        ledger.append({"Date": pnl_date, "Category": "Revenue", "Type": "Revenue", "Amount": rev_amt, "Report": "PnL"})
        ledger.append({"Date": pnl_date + timedelta(days=cash_lag), "Category": "Collections", "Type": "Cash", "Amount": rev_amt, "Report": "Cash"})
        ledger.append({"Date": pnl_date, "Category": "Material Cost", "Type": "COGS", "Amount": -unit_mat_cost, "Report": "PnL"})
    
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
                ledger.append({"Date": delivery + timedelta(days=int(part['deposit_days'])), "Category": "Material Deposit", "Type": "Cash", "Amount": -(total_po_cost * part['deposit_pct']), "Report": "Cash"})
            if part['deposit_pct'] < 1.0:
                ledger.append({"Date": delivery + timedelta(days=int(part['balance_days'])), "Category": "Material Balance", "Type": "Cash", "Amount": -(total_po_cost * (1 - part['deposit_pct'])), "Report": "Cash"})

    # OpEx
    opex_merged = pd.merge(df_opex, df_roles, left_on='role_id', right_on='id')
    for _, row in opex_merged.iterrows():
        cost = (row['annual_salary']/12) * row['headcount']
        if cost > 0:
            cat = "Direct Labor" if "Assembler" in row['role_name'] else "Payroll (SG&A/R&D)"
            pnl_type = "COGS" if "Assembler" in row['role_name'] else "OpEx"
            ledger.append({"Date": row['month_date'], "Category": cat, "Type": pnl_type, "Amount": -cost, "Report": "PnL"})
            ledger.append({"Date": row['month_date'], "Category": "Payroll", "Type": "Cash", "Amount": -cost, "Report": "Cash"})

    for _, row in df_gen_exp.iterrows():
        if row['amount'] > 0:
            ledger.append({"Date": row['month_date'], "Category": row['expense_type'], "Type": "OpEx", "Amount": -row['amount'], "Report": "PnL"})
            ledger.append({"Date": row['month_date'], "Category": "OpEx Spend", "Type": "Cash", "Amount": -row['amount'], "Report": "Cash"})

    if not ledger: return pd.DataFrame(), pd.DataFrame()
    
    df_master = pd.DataFrame(ledger)
    df_pnl = df_master[df_master['Report'] == "PnL"].sort_values('Date')
    df_cash = df_master[df_master['Report'] == "Cash"].sort_values('Date')
    df_cash['Cash_Balance'] = df_cash['Amount'].cumsum() + start_cash
    return df_pnl, df_cash

# 3. EXECUTE
try:
    df_pnl, df_cash = generate_financials()

    # 4. VISUAL LAYOUT
    st.sidebar.title("IdleX CFO Console")
    view = st.sidebar.radio("Navigation", ["Executive Dashboard", "Financial Statements", "Production & Sales", "OpEx Planning", "BOM & Supply Chain"])

    if not df_pnl.empty:
        years = sorted(df_pnl['Date'].dt.year.unique().tolist())
        st.sidebar.divider()
        selected_period = st.sidebar.selectbox("Fiscal Year:", ["All Time"] + years)
        
        if selected_period == "All Time":
            pnl_view = df_pnl
            cash_view = df_cash
        else:
            pnl_view = df_pnl[df_pnl['Date'].dt.year == selected_period]
            cash_view = df_cash[df_cash['Date'].dt.year == selected_period]
    else:
        pnl_view, cash_view = pd.DataFrame(), pd.DataFrame()

    if view == "Executive Dashboard":
        st.title(f"Executive Dashboard")
        if not pnl_view.empty:
            rev = pnl_view[pnl_view['Category']=='Revenue']['Amount'].sum()
            cogs = abs(pnl_view[pnl_view['Type']=='COGS']['Amount'].sum())
            margin = rev - cogs
            min_c = cash_view['Cash_Balance'].min()
            end_c = cash_view.iloc[-1]['Cash_Balance'] if not cash_view.empty else 0
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Revenue", f"${rev:,.0f}")
            c2.metric("Gross Margin", f"${margin:,.0f}")
            c3.metric("Min Net Cash", f"${min_c:,.0f}", delta_color="inverse")
            c4.metric("Ending Cash", f"${end_c:,.0f}")
            
            fig = px.area(cash_view, x='Date', y='Cash_Balance', title="Cash Runway", color_discrete_sequence=['#10B981'])
            fig.add_hline(y=0, line_dash="dash", line_color="red")
            st.plotly_chart(fig, use_container_width=True)

    elif view == "Financial Statements":
        st.title("Financial Statements")
        if not pnl_view.empty:
            col1, col2 = st.columns(2)
            with col1: freq = st.radio("Aggregation:", ["Monthly", "Quarterly", "Yearly"], horizontal=True)
            freq_map = {"Monthly": "ME", "Quarterly": "QE", "Yearly": "YE"}
            
            # P&L
            st.subheader("Profit & Loss")
            p_piv = pnl_view.groupby([pd.Grouper(key='Date', freq=freq_map[freq]), 'Type']).sum()['Amount'].unstack().fillna(0)
            for c in ['Revenue', 'COGS', 'OpEx']: 
                if c not in p_piv: p_piv[c] = 0
            p_piv['Net Income'] = p_piv['Revenue'] + p_piv['COGS'] + p_piv['OpEx']
            
            if freq == "Monthly": p_piv.index = p_piv.index.strftime('%Y-%b')
            elif freq == "Quarterly": p_piv.index = p_piv.index.to_period("Q").astype(str)
            else: p_piv.index = p_piv.index.strftime('%Y')
            
            st.dataframe(p_piv.style.format("${:,.0f}"), use_container_width=True)
            
            st.divider()
            st.subheader("Cash Flow")
            c_piv = cash_view.groupby([pd.Grouper(key='Date', freq=freq_map[freq]), 'Category']).sum()['Amount'].unstack().fillna(0)
            c_piv['Net Change'] = c_piv.sum(axis=1)
            if freq == "Monthly": c_piv.index = c_piv.index.strftime('%Y-%b')
            elif freq == "Quarterly": c_piv.index = c_piv.index.to_period("Q").astype(str)
            else: c_piv.index = c_piv.index.strftime('%Y')
            st.dataframe(c_piv.style.format("${:,.0f}"), use_container_width=True)

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
            st.info("Updates here preserve 'WIP' and 'COMPLETE' units.")
            
            start_date = st.date_input("Production Start", value=date(2026, 1, 1))
            df_units['Month'] = pd.to_datetime(df_units['build_date']).dt.strftime('%Y-%m')
            
            exist = df_units.groupby('Month').size()
            dates = pd.date_range('2026-01-01', '2027-12-01', freq='MS')
            plan = [{"Month": d.date(), "Target": int(exist.get(d.strftime('%Y-%m'), 0))} for d in dates]
            
            edit_plan = st.data_editor(pd.DataFrame(plan), hide_index=True, height=400)
            
            if st.button("🚀 Smart Regenerate"):
                with st.spinner("Optimizing Schedule..."):
                    with engine.connect() as conn:
                        conn.execute(text("DELETE FROM production_unit WHERE status = 'PLANNED'"))
                        last_sn = conn.execute(text("SELECT serial_number FROM production_unit ORDER BY id DESC LIMIT 1")).scalar()
                        if last_sn:
                            sn_counter = int(''.join(filter(str.isdigit, last_sn))) + 1
                        else:
                            sn_counter = 1
                        
                        for _, row in edit_plan.iterrows():
                            target = row['Target']
                            if target == 0: continue
                            
                            m_str = row['Month'].strftime('%Y-%m')
                            locked_count = conn.execute(text(f"SELECT COUNT(*) FROM production_unit WHERE strftime('%Y-%m', build_date) = '{m_str}' AND status != 'PLANNED'")).scalar()
                            
                            to_build = target - locked_count
                            if to_build <= 0: continue
                            
                            tgt = row['Month']
                            threshold = None
                            if tgt.year == start_date.year and tgt.month == start_date.month:
                                threshold = start_date
                            
                            last_day = date(tgt.year, tgt.month, calendar.monthrange(tgt.year, tgt.month)[1])
                            if last_day < start_date: continue

                            wd = get_workdays(tgt.year, tgt.month, threshold)
                            if not wd: continue
                            
                            direct = math.floor(to_build * 0.25)
                            pool = ['DIRECT']*direct + ['DEALER']*(to_build - direct)
                            
                            d_idx = 0
                            for t in pool:
                                conn.execute(text("INSERT INTO production_unit (serial_number, build_date, sales_channel, status) VALUES (:s, :b, :c, 'PLANNED')"), 
                                             {"s": f"IDX-{sn_counter:04d}", "b": wd[d_idx], "c": t})
                                sn_counter += 1
                                d_idx = (d_idx + 1) % len(wd)
                        conn.commit()
                st.success("Schedule Optimized!")
                st.rerun()

    elif view == "OpEx Planning":
        st.title("OpEx Budget")
        tab1, tab2 = st.tabs(["Headcount & Payroll", "R&D & SG&A"])
        with tab1:
            df_roles = pd.read_sql("SELECT * FROM opex_roles", engine)
            df_staff = pd.read_sql("SELECT * FROM opex_staffing_plan", engine)
            df_merged = pd.merge(df_staff, df_roles, left_on='role_id', right_on='id')
            df_merged['Month'] = pd.to_datetime(df_merged['month_date']).dt.strftime('%Y-%m')
            pivot_hc = df_merged.pivot(index='role_name', columns='Month', values='headcount').reset_index()
            edited_hc = st.data_editor(pivot_hc, use_container_width=True, num_rows="dynamic")
            if st.button("💾 Save Headcount"):
                with engine.connect() as conn:
                    melted = edited_hc.melt(id_vars=['role_name'], var_name='Month', value_name='headcount')
                    for _, row in melted.iterrows():
                        role_id = conn.execute(text("SELECT id FROM opex_roles WHERE role_name=:rn"), {"rn": row['role_name']}).scalar()
                        if role_id:
                            dt = date.fromisoformat(row['Month'] + "-01")
                            conn.execute(text("INSERT OR REPLACE INTO opex_staffing_plan (id, role_id, month_date, headcount) VALUES ((SELECT id FROM opex_staffing_plan WHERE role_id=:rid AND month_date=:dt), :rid, :dt, :hc)"), {"rid": role_id, "dt": dt, "hc": row['headcount']})
                    conn.commit()
                st.rerun()
        with tab2:
            df_gen = pd.read_sql("SELECT * FROM opex_general_expenses", engine)
            df_gen['Month'] = pd.to_datetime(df_gen['month_date']).dt.strftime('%Y-%m')
            pivot_gen = df_gen.pivot(index=['category', 'expense_type'], columns='Month', values='amount').reset_index()
            edited_gen = st.data_editor(pivot_gen, use_container_width=True, num_rows="dynamic")
            if st.button("💾 Save Expenses"):
                with engine.connect() as conn:
                    melted = edited_gen.melt(id_vars=['category', 'expense_type'], var_name='Month', value_name='amount')
                    conn.execute(text("DELETE FROM opex_general_expenses"))
                    for _, row in melted.iterrows():
                        if pd.notna(row['amount']):
                            dt = date.fromisoformat(row['Month'] + "-01")
                            conn.execute(text("INSERT INTO opex_general_expenses (category, expense_type, month_date, amount) VALUES (:c, :t, :d, :a)"), {"c": row['category'], "t": row['expense_type'], "d": dt, "a": row['amount']})
                    conn.commit()
                st.rerun()

    elif view == "BOM & Supply Chain":
        st.title("Bill of Materials")
        df_parts = pd.read_sql("SELECT * FROM part_master", engine)
        edited_df = st.data_editor(df_parts, disabled=["id", "sku"], use_container_width=True, num_rows="dynamic")
        if st.button("💾 Save BOM"):
            with engine.connect() as conn:
                for _, row in edited_df.iterrows():
                    conn.execute(text("UPDATE part_master SET name=:name, cost=:cost, moq=:moq, lead_time=:l, deposit_pct=:dp, deposit_days=:dd, balance_days=:bd WHERE id=:id"), 
                                 {"name": row['name'], "cost": row['cost'], "moq": row['moq'], "l": row['lead_time'], "dp": row['deposit_pct'], "dd": row['deposit_days'], "bd": row['balance_days'], "id": row['id']})
                conn.commit()
            st.rerun()

except Exception as e:
    st.error(f"System Error: {e}")