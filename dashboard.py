"""
IdleX ERP - Enterprise Resource Planning System
Version: 11.0 (Fixed Cash Calculations)
"""

import streamlit as st
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from datetime import datetime, date, timedelta
import plotly.express as px
import plotly.graph_objects as go
import calendar
import math
import os

# =============================================================================
# BRAND CONFIGURATION
# =============================================================================
BRAND = {
    "primary": "#1E3A5F",
    "secondary": "#3D7EAA",
    "accent": "#F4A261",
    "success": "#2A9D8F",
    "warning": "#E9C46A",
    "danger": "#E76F51",
    "light": "#F8F9FA",
    "dark": "#212529",
}

# =============================================================================
# DATABASE CONNECTION
# =============================================================================
@st.cache_resource
def get_engine():
    db_url = os.getenv("DATABASE_URL")
    if db_url:
        if db_url.startswith("postgres://"):
            db_url = db_url.replace("postgres://", "postgresql://", 1)
        return create_engine(db_url, pool_pre_ping=True)
    return create_engine("sqlite:///idlex.db")

engine = get_engine()

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
def money(val):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "-"
    try:
        v = int(round(float(val)))
        if v < 0:
            return f"({abs(v):,})"
        return f"{v:,}"
    except:
        return "-"

def pct(val, decimals=1):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "-"
    try:
        return f"{float(val) * 100:.{decimals}f}%"
    except:
        return "-"

def parse_date(val):
    if val is None:
        return None
    if isinstance(val, date):
        return val
    if isinstance(val, datetime):
        return val.date()
    if isinstance(val, str):
        for fmt in ["%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y"]:
            try:
                return datetime.strptime(val, fmt).date()
            except:
                continue
    return None

def get_workdays(year, month):
    num_days = calendar.monthrange(year, month)[1]
    days = [date(year, month, d) for d in range(1, num_days + 1)]
    return [d for d in days if d.weekday() < 5]

def get_quarter(month):
    return (month - 1) // 3 + 1

# =============================================================================
# FINANCIAL ENGINE - FIXED
# =============================================================================
def calculate_unit_material_cost(conn):
    try:
        result = conn.execute(text("""
            SELECT COALESCE(SUM(p.cost * b.qty_per_unit), 0) as total
            FROM bom_items b JOIN part_master p ON b.part_id = p.id
        """)).scalar()
        return float(result) if result else 0.0
    except:
        return 0.0

def get_pricing_for_year(conn, year):
    try:
        row = conn.execute(text("SELECT msrp, dealer_discount_pct FROM pricing_config WHERE year = :y"), {"y": year}).fetchone()
        if row:
            return {"msrp": float(row[0]), "dealer_pct": float(row[1])}
    except:
        pass
    defaults = {2026: 15500, 2027: 13500, 2028: 11500}
    return {"msrp": defaults.get(year, 13500), "dealer_pct": 0.80}

def generate_financial_ledgers(conn):
    """Generate P&L and Cash ledgers - FIXED VERSION."""
    pnl_entries = []
    cash_entries = []
    
    try:
        material_cost = calculate_unit_material_cost(conn)
        
        # REVENUE & COGS from production
        units = conn.execute(text("SELECT id, serial_number, build_date, sales_channel FROM production_unit ORDER BY build_date")).fetchall()
        
        for unit in units:
            unit_id, serial, build_date, channel = unit
            build_dt = parse_date(build_date)
            if not build_dt:
                continue
            
            year = build_dt.year
            pricing = get_pricing_for_year(conn, year)
            msrp = pricing["msrp"]
            dealer_pct = pricing["dealer_pct"]
            
            revenue = msrp if channel == "DIRECT" else msrp * dealer_pct
            
            # P&L entries (positive revenue, negative costs)
            pnl_entries.append({"Date": build_dt, "Type": "Revenue", "Category": f"{channel} Sale", "Amount": revenue, "Unit": serial})
            pnl_entries.append({"Date": build_dt, "Type": "COGS", "Category": "Material Cost", "Amount": -material_cost, "Unit": serial})
            
            # Cash entries (positive inflow, negative outflow)
            cash_entries.append({"Date": build_dt, "Type": "Inflow", "Category": "Revenue", "Amount": revenue, "Unit": serial})
            cash_entries.append({"Date": build_dt, "Type": "Outflow", "Category": "Material", "Amount": -material_cost, "Unit": serial})
        
        # PAYROLL from staffing
        staffing = conn.execute(text("""
            SELECT s.month_date, r.role_name, r.annual_salary, s.headcount
            FROM opex_staffing_plan s JOIN opex_roles r ON s.role_id = r.id
        """)).fetchall()
        
        for row in staffing:
            month_date, role, salary, headcount = row
            m_date = parse_date(month_date)
            if not m_date:
                continue
            headcount = float(headcount) if headcount else 0
            if headcount <= 0:
                continue
            
            monthly_cost = (float(salary) / 12.0) * headcount
            
            pnl_entries.append({"Date": m_date, "Type": "OpEx", "Category": f"Payroll: {role}", "Amount": -monthly_cost, "Unit": None})
            cash_entries.append({"Date": m_date, "Type": "Outflow", "Category": "Payroll", "Amount": -monthly_cost, "Unit": None})
        
        # GENERAL EXPENSES - FIXED: Only include positive amounts
        expenses = conn.execute(text("""
            SELECT month_date, category, expense_type, amount 
            FROM opex_general_expenses 
            WHERE amount > 0
        """)).fetchall()
        
        for row in expenses:
            month_date, category, exp_type, amount = row
            m_date = parse_date(month_date)
            if not m_date:
                continue
            
            amount_val = float(amount)
            if amount_val <= 0:  # Double-check
                continue
            
            pnl_entries.append({"Date": m_date, "Type": "OpEx", "Category": f"{exp_type}: {category}", "Amount": -amount_val, "Unit": None})
            cash_entries.append({"Date": m_date, "Type": "Outflow", "Category": "General", "Amount": -amount_val, "Unit": None})
        
        return pd.DataFrame(pnl_entries), pd.DataFrame(cash_entries)
    except Exception as e:
        st.error(f"Error generating ledgers: {e}")
        return pd.DataFrame(), pd.DataFrame()

def run_cash_waterfall(cash_df, starting_equity, loc_limit):
    """Process cash flow chronologically with LOC logic - SIMPLIFIED & FIXED."""
    if cash_df.empty:
        return pd.DataFrame([{
            "Date": date.today(), 
            "Cash": starting_equity, 
            "LOC_Used": 0.0,
            "LOC_Available": loc_limit,
            "Total_Liquidity": starting_equity + loc_limit,
            "Day_Inflow": 0,
            "Day_Outflow": 0,
            "Cash_Crunch": False
        }])
    
    df = cash_df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")
    
    results = []
    cash = float(starting_equity)
    loc_used = 0.0
    
    # Group by date and process
    for dt, group in df.groupby("Date"):
        # Calculate day totals
        day_inflow = group[group["Amount"] > 0]["Amount"].sum()
        day_outflow = group[group["Amount"] < 0]["Amount"].sum()  # This is negative
        
        # Separate by category for LOC eligibility
        material_outflow = group[(group["Amount"] < 0) & (group["Category"] == "Material")]["Amount"].sum()
        other_outflow = day_outflow - material_outflow  # Payroll + General (also negative)
        
        # Step 1: Add inflows to cash (but first pay down LOC if any)
        if day_inflow > 0:
            if loc_used > 0:
                loc_paydown = min(day_inflow, loc_used)
                loc_used -= loc_paydown
                day_inflow -= loc_paydown
            cash += day_inflow
        
        # Step 2: Pay non-material expenses (Payroll, General) - NO LOC allowed
        if other_outflow < 0:  # It's negative
            cash += other_outflow  # Subtracts from cash
        
        # Step 3: Pay material - CAN use LOC
        if material_outflow < 0:  # It's negative
            needed = abs(material_outflow)
            if cash >= needed:
                cash -= needed
            else:
                # Use available cash first
                shortfall = needed - max(cash, 0)
                cash = max(cash - needed, 0)  # Don't go below 0 for material if we have LOC
                
                # Draw from LOC
                loc_available = loc_limit - loc_used
                loc_draw = min(shortfall, loc_available)
                loc_used += loc_draw
                
                # If still short, cash goes negative
                remaining_shortfall = shortfall - loc_draw
                if remaining_shortfall > 0:
                    cash -= remaining_shortfall
        
        results.append({
            "Date": dt,
            "Cash": round(cash, 2),
            "LOC_Used": round(loc_used, 2),
            "LOC_Available": round(loc_limit - loc_used, 2),
            "Total_Liquidity": round(cash + (loc_limit - loc_used), 2),
            "Day_Inflow": round(day_inflow, 2),
            "Day_Outflow": round(abs(day_outflow), 2),
            "Cash_Crunch": cash < 0
        })
    
    return pd.DataFrame(results)

def calculate_revenue_impact(conn, year, msrp, dealer_margin_pct):
    try:
        result = conn.execute(text("""
            SELECT sales_channel, COUNT(*) as units
            FROM production_unit WHERE strftime('%Y', build_date) = :y
            GROUP BY sales_channel
        """), {"y": str(year)}).fetchall()
        
        direct_units = sum(r[1] for r in result if r[0] == "DIRECT")
        dealer_units = sum(r[1] for r in result if r[0] == "DEALER")
        
        dealer_price = msrp * dealer_margin_pct
        return {
            "direct_units": direct_units,
            "dealer_units": dealer_units,
            "total_units": direct_units + dealer_units,
            "direct_revenue": direct_units * msrp,
            "dealer_revenue": dealer_units * dealer_price,
            "total_revenue": direct_units * msrp + dealer_units * dealer_price,
            "dealer_price": dealer_price
        }
    except:
        return None

def apply_channel_mix_to_production(conn, year, quarter, direct_pct):
    try:
        start_month = (quarter - 1) * 3 + 1
        end_month = quarter * 3
        
        units = conn.execute(text("""
            SELECT id FROM production_unit
            WHERE strftime('%Y', build_date) = :y
            AND CAST(strftime('%m', build_date) AS INTEGER) BETWEEN :sm AND :em
            ORDER BY build_date, id
        """), {"y": str(year), "sm": start_month, "em": end_month}).fetchall()
        
        if not units:
            return 0
        
        total = len(units)
        direct_count = int(total * direct_pct)
        unit_ids = [u[0] for u in units]
        
        for i, uid in enumerate(unit_ids):
            channel = "DIRECT" if i < direct_count else "DEALER"
            conn.execute(text("UPDATE production_unit SET sales_channel = :ch WHERE id = :id"), {"ch": channel, "id": uid})
        
        conn.commit()
        return total
    except Exception as e:
        st.error(f"Error: {e}")
        return 0

# =============================================================================
# STREAMLIT UI
# =============================================================================
def main():
    st.set_page_config(page_title="IdleX ERP", page_icon="⚡", layout="wide", initial_sidebar_state="expanded")
    
    st.markdown(f"""
    <style>
        .stApp {{ background-color: {BRAND['dark']}; }}
        .main-header {{ color: {BRAND['accent']}; font-family: 'Helvetica Neue', sans-serif; font-style: italic; }}
    </style>
    """, unsafe_allow_html=True)
    
    with st.sidebar:
        st.markdown("<h1 class='main-header'>idleX</h1>", unsafe_allow_html=True)
        st.caption("Enterprise Resource Planning")
        st.divider()
        
        page = st.radio("", ["🏠 Dashboard", "📊 Financials", "📈 Production & Sales", "💼 OpEx Budget", "🔧 Supply Chain", "🔍 Debug"], label_visibility="collapsed")
        
        st.divider()
        with st.expander("⚙️ Database", expanded=False):
            if st.button("🔄 Rebuild Database", type="primary", use_container_width=True):
                try:
                    from seed_db import run_seed
                    run_seed()
                    st.success("Database rebuilt!")
                    st.cache_data.clear()
                    st.rerun()
                except Exception as e:
                    st.error(f"Rebuild failed: {e}")
            
            if st.button("🗑️ Clear Cache"):
                st.cache_data.clear()
                st.rerun()
            
            st.caption("⚠️ Rebuild resets all data")
    
    st.markdown("<h1 class='main-header'>idleX</h1>", unsafe_allow_html=True)
    st.caption("| Enterprise Resource Planning")
    
    if "Dashboard" in page:
        render_dashboard()
    elif "Financials" in page:
        render_financials()
    elif "Production" in page:
        render_production_sales()
    elif "OpEx" in page:
        render_opex()
    elif "Supply Chain" in page:
        render_supply_chain()
    elif "Debug" in page:
        render_debug()

def render_dashboard():
    st.header("Executive Dashboard")
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("⚙️ Scenario Constraints")
        equity = st.number_input("Investor Equity ($)", value=1500000, step=100000)
        loc = st.number_input("Credit Limit (LOC) ($)", value=4100000, step=100000)
        
        st.subheader("🚀 Growth Assumptions")
        growth_rate = st.slider("Monthly Growth Rate (%)", 0, 10, 2)
        st.caption("Growth model not yet applied")
    
    with col2:
        try:
            with engine.connect() as conn:
                pnl_df, cash_df = generate_financial_ledgers(conn)
                if cash_df.empty:
                    st.warning("No financial data. Rebuild database.")
                    return
                
                waterfall = run_cash_waterfall(cash_df, equity, loc)
                
                total_revenue = pnl_df[pnl_df["Type"] == "Revenue"]["Amount"].sum()
                total_cogs = abs(pnl_df[pnl_df["Type"] == "COGS"]["Amount"].sum())
                total_opex = abs(pnl_df[pnl_df["Type"] == "OpEx"]["Amount"].sum())
                gross_profit = total_revenue - total_cogs
                net_income = gross_profit - total_opex
                unit_count = conn.execute(text("SELECT COUNT(*) FROM production_unit")).scalar()
                
                # Key metrics
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Total Units", f"{unit_count:,}")
                m2.metric("Total Revenue", f"${total_revenue:,.0f}")
                m3.metric("Gross Profit", f"${gross_profit:,.0f}", f"{gross_profit/total_revenue*100:.1f}%" if total_revenue else "")
                m4.metric("Net Income", f"${net_income:,.0f}")
                
                # Cash summary
                st.markdown("---")
                min_cash = waterfall["Cash"].min()
                max_loc = waterfall["LOC_Used"].max()
                final_cash = waterfall["Cash"].iloc[-1]
                
                c1, c2, c3 = st.columns(3)
                c1.metric("Min Cash", f"${min_cash:,.0f}", delta="⚠️ Negative!" if min_cash < 0 else None, delta_color="inverse")
                c2.metric("Peak LOC Used", f"${max_loc:,.0f}")
                c3.metric("Final Cash (Dec 2028)", f"${final_cash:,.0f}")
                
                # Chart
                st.subheader("💰 Cash & LOC Forecast")
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=waterfall["Date"], y=waterfall["Cash"], name="Cash", fill="tozeroy", line=dict(color=BRAND["success"])))
                fig.add_trace(go.Scatter(x=waterfall["Date"], y=waterfall["LOC_Used"], name="LOC Used", line=dict(color=BRAND["warning"], dash="dash")))
                fig.add_hline(y=0, line_dash="dot", line_color="red")
                fig.update_layout(template="plotly_dark", height=350)
                st.plotly_chart(fig, use_container_width=True)
                
                if waterfall["Cash_Crunch"].any():
                    crunch_count = waterfall["Cash_Crunch"].sum()
                    st.error(f"⚠️ Cash goes negative on {crunch_count} days!")
                    
        except Exception as e:
            st.error(f"Error: {e}")

def render_financials():
    st.header("📊 Financial Analysis")
    
    try:
        with engine.connect() as conn:
            pnl_df, cash_df = generate_financial_ledgers(conn)
            if pnl_df.empty:
                st.warning("No data available.")
                return
            
            # Totals
            total_revenue = pnl_df[pnl_df["Type"] == "Revenue"]["Amount"].sum()
            total_cogs = abs(pnl_df[pnl_df["Type"] == "COGS"]["Amount"].sum())
            total_opex = abs(pnl_df[pnl_df["Type"] == "OpEx"]["Amount"].sum())
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Revenue", f"${total_revenue:,.0f}")
            col2.metric("Total COGS", f"${total_cogs:,.0f}")
            col3.metric("Gross Margin", f"{(total_revenue-total_cogs)/total_revenue*100:.1f}%" if total_revenue else "-")
            col4.metric("Total OpEx", f"${total_opex:,.0f}")
            
            # Monthly summary
            st.subheader("Monthly P&L")
            pnl_df["Month"] = pd.to_datetime(pnl_df["Date"]).dt.to_period("M").astype(str)
            monthly = pnl_df.groupby(["Month", "Type"])["Amount"].sum().unstack(fill_value=0)
            
            if "Revenue" in monthly.columns and "COGS" in monthly.columns:
                monthly["Gross Profit"] = monthly["Revenue"] + monthly["COGS"]  # COGS is negative
            if "OpEx" in monthly.columns and "Gross Profit" in monthly.columns:
                monthly["Net Income"] = monthly["Gross Profit"] + monthly["OpEx"]  # OpEx is negative
            
            st.dataframe(monthly.style.format("${:,.0f}"), use_container_width=True)
            
    except Exception as e:
        st.error(f"Error: {e}")

def render_production_sales():
    st.header("Production & Sales Planning")
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Manifest", "💰 Pricing", "📊 Channel Mix", "🚀 Planner"])
    
    with tab1:
        render_production_manifest()
    with tab2:
        render_pricing_config()
    with tab3:
        render_channel_mix()
    with tab4:
        st.info("Smart Planner coming soon")

def render_production_manifest():
    try:
        with engine.connect() as conn:
            summary = pd.read_sql("""
                SELECT strftime('%Y', build_date) as Year, sales_channel as Channel, COUNT(*) as Units
                FROM production_unit GROUP BY Year, Channel ORDER BY Year, Channel
            """, conn)
            
            if summary.empty:
                st.warning("No production data.")
                return
            
            pivot = summary.pivot(index="Year", columns="Channel", values="Units").fillna(0)
            pivot["Total"] = pivot.sum(axis=1)
            st.dataframe(pivot.style.format("{:,.0f}"), use_container_width=True)
            
            fig = px.bar(summary, x="Year", y="Units", color="Channel", barmode="group",
                        color_discrete_map={"DIRECT": BRAND["success"], "DEALER": BRAND["secondary"]})
            fig.update_layout(template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error: {e}")

def render_pricing_config():
    st.subheader("Pricing Configuration")
    
    try:
        with engine.connect() as conn:
            pricing = pd.read_sql("SELECT * FROM pricing_config ORDER BY year", conn)
            if pricing.empty:
                st.warning("No pricing config. Rebuild database.")
                return
            
            selected_year = st.selectbox("Select Year", pricing["year"].tolist())
            row = pricing[pricing["year"] == selected_year].iloc[0]
            
            col1, col2 = st.columns(2)
            with col1:
                new_msrp = st.number_input("MSRP ($)", value=float(row["msrp"]), step=100.0, format="%.2f")
            with col2:
                current_margin = float(row["dealer_discount_pct"]) * 100
                new_margin = st.number_input("Dealer Pays (% of MSRP)", value=current_margin, min_value=50.0, max_value=100.0, step=1.0, format="%.2f")
            
            dealer_price = new_msrp * (new_margin / 100)
            
            st.markdown("---")
            st.subheader("💵 Revenue Impact")
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Direct Price", f"${new_msrp:,.2f}")
            c2.metric("Dealer Price", f"${dealer_price:,.2f}")
            c3.metric("Dealer Keeps", f"${new_msrp - dealer_price:,.2f}")
            
            impact = calculate_revenue_impact(conn, selected_year, new_msrp, new_margin / 100)
            if impact:
                c1, c2, c3 = st.columns(3)
                c1.metric(f"Direct ({impact['direct_units']:,})", f"${impact['direct_revenue']:,.0f}")
                c2.metric(f"Dealer ({impact['dealer_units']:,})", f"${impact['dealer_revenue']:,.0f}")
                c3.metric(f"Total ({impact['total_units']:,})", f"${impact['total_revenue']:,.0f}")
            
            if st.button("💾 Save Pricing", type="primary"):
                conn.execute(text("UPDATE pricing_config SET msrp=:m, dealer_discount_pct=:d WHERE year=:y"),
                            {"y": selected_year, "m": new_msrp, "d": new_margin/100})
                conn.commit()
                st.success("Saved!")
                st.cache_data.clear()
                st.rerun()
            
            # Summary table
            st.markdown("---")
            st.subheader("All Years")
            summary = []
            for _, r in pricing.iterrows():
                summary.append({
                    "Year": int(r["year"]),
                    "MSRP": f"${float(r['msrp']):,.0f}",
                    "Dealer Pays": f"{float(r['dealer_discount_pct'])*100:.0f}%",
                    "Dealer Price": f"${float(r['msrp']) * float(r['dealer_discount_pct']):,.0f}"
                })
            st.dataframe(pd.DataFrame(summary), use_container_width=True, hide_index=True)
            
    except Exception as e:
        st.error(f"Error: {e}")

def render_channel_mix():
    st.subheader("Channel Mix by Quarter")
    
    try:
        with engine.connect() as conn:
            mix = pd.read_sql("SELECT * FROM channel_mix_config ORDER BY year, quarter", conn)
            if mix.empty:
                st.warning("No channel mix config.")
                return
            
            # Build display data
            display_data = []
            for _, row in mix.iterrows():
                direct = float(row["direct_pct"]) * 100
                display_data.append({
                    "id": int(row["id"]),
                    "Year": int(row["year"]),
                    "Quarter": int(row["quarter"]),
                    "Direct %": direct,
                    "Dealer %": 100 - direct,
                    "Label": f"Q{int(row['quarter'])} {int(row['year'])}"
                })
            
            display_df = pd.DataFrame(display_data)
            
            edited = st.data_editor(
                display_df[["Label", "Direct %", "Dealer %"]],
                column_config={
                    "Label": st.column_config.TextColumn("Quarter", disabled=True),
                    "Direct %": st.column_config.NumberColumn("Direct %", min_value=0, max_value=100, step=0.5, format="%.2f%%"),
                    "Dealer %": st.column_config.NumberColumn("Dealer %", disabled=True, format="%.2f%%"),
                },
                use_container_width=True, hide_index=True
            )
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("💾 Save Channel Mix", type="primary"):
                    for idx, row in edited.iterrows():
                        orig = display_df.iloc[idx]
                        conn.execute(text("UPDATE channel_mix_config SET direct_pct=:d WHERE id=:id"),
                                    {"d": row["Direct %"]/100, "id": orig["id"]})
                    conn.commit()
                    st.success("Saved!")
                    st.cache_data.clear()
                    st.rerun()
            
            with col2:
                if st.button("🔄 Apply to Production"):
                    total = 0
                    for idx, row in edited.iterrows():
                        orig = display_df.iloc[idx]
                        updated = apply_channel_mix_to_production(conn, orig["Year"], orig["Quarter"], row["Direct %"]/100)
                        total += updated
                    st.success(f"Updated {total} units!")
                    st.cache_data.clear()
                    st.rerun()
            
    except Exception as e:
        st.error(f"Error: {e}")

def render_opex():
    st.header("💼 OpEx Budget")
    tab1, tab2 = st.tabs(["👥 Staffing", "📋 General Expenses"])
    
    with tab1:
        try:
            with engine.connect() as conn:
                roles = pd.read_sql("SELECT * FROM opex_roles ORDER BY annual_salary DESC", conn)
                st.subheader("Roles")
                st.dataframe(roles.style.format({"annual_salary": "${:,.0f}"}), use_container_width=True)
                
                staffing = pd.read_sql("""
                    SELECT s.month_date, r.role_name, s.headcount
                    FROM opex_staffing_plan s JOIN opex_roles r ON s.role_id = r.id
                    ORDER BY s.month_date
                """, conn)
                if not staffing.empty:
                    st.subheader("Staffing Plan")
                    pivot = staffing.pivot(index="month_date", columns="role_name", values="headcount")
                    st.dataframe(pivot, use_container_width=True)
        except Exception as e:
            st.error(f"Error: {e}")
    
    with tab2:
        render_general_expenses()

def render_general_expenses():
    st.subheader("General Expenses")
    
    try:
        with engine.connect() as conn:
            expenses = pd.read_sql("""
                SELECT id, month_date, expense_type, category, amount 
                FROM opex_general_expenses ORDER BY month_date, category
            """, conn)
            
            if expenses.empty:
                st.info("No expenses found.")
                return
            
            # Summary by type
            by_type = expenses.groupby("expense_type")["amount"].sum()
            cols = st.columns(len(by_type))
            for i, (exp_type, total) in enumerate(by_type.items()):
                cols[i].metric(f"Total {exp_type}", f"${total:,.0f}")
            
            st.markdown("---")
            
            # Editable expenses
            st.caption("Edit amounts directly. Set to 0 to remove expense from calculations.")
            
            edited = st.data_editor(
                expenses[["month_date", "expense_type", "category", "amount"]],
                column_config={
                    "month_date": st.column_config.TextColumn("Month", disabled=True),
                    "expense_type": st.column_config.TextColumn("Type", disabled=True),
                    "category": st.column_config.TextColumn("Category", disabled=True),
                    "amount": st.column_config.NumberColumn("Amount ($)", min_value=0, format="$%.0f"),
                },
                use_container_width=True, hide_index=True
            )
            
            if st.button("💾 Save Expense Changes", type="primary"):
                for idx, row in edited.iterrows():
                    orig_id = expenses.iloc[idx]["id"]
                    new_amount = row["amount"]
                    conn.execute(text("UPDATE opex_general_expenses SET amount=:a WHERE id=:id"),
                                {"a": new_amount, "id": orig_id})
                conn.commit()
                st.success("Expenses saved!")
                st.cache_data.clear()
                st.rerun()
                
    except Exception as e:
        st.error(f"Error: {e}")

def render_supply_chain():
    st.header("🔧 Supply Chain")
    tab1, tab2 = st.tabs(["📦 Parts", "🧩 BOM"])
    
    with tab1:
        try:
            with engine.connect() as conn:
                parts = pd.read_sql("SELECT * FROM part_master ORDER BY sku", conn)
                st.dataframe(parts.style.format({"cost": "${:,.2f}", "deposit_pct": "{:.0%}"}), use_container_width=True)
        except Exception as e:
            st.error(f"Error: {e}")
    
    with tab2:
        try:
            with engine.connect() as conn:
                bom = pd.read_sql("""
                    SELECT p.sku, p.name, b.qty_per_unit, p.cost, (b.qty_per_unit * p.cost) as extended
                    FROM bom_items b JOIN part_master p ON b.part_id = p.id ORDER BY extended DESC
                """, conn)
                if not bom.empty:
                    total = bom["extended"].sum()
                    st.metric("BOM Cost per Unit", f"${total:,.2f}")
                    st.dataframe(bom.style.format({"cost": "${:,.2f}", "extended": "${:,.2f}"}), use_container_width=True)
        except Exception as e:
            st.error(f"Error: {e}")

def render_debug():
    """Debug view to trace cash calculations."""
    st.header("🔍 Financial Debug")
    
    equity = st.number_input("Equity", value=1500000, step=100000, key="debug_equity")
    loc = st.number_input("LOC Limit", value=4100000, step=100000, key="debug_loc")
    
    try:
        with engine.connect() as conn:
            pnl_df, cash_df = generate_financial_ledgers(conn)
            
            if cash_df.empty:
                st.warning("No data")
                return
            
            # Show expense totals from database
            st.subheader("1. Database Expense Totals")
            
            # General expenses
            gen_exp = conn.execute(text("SELECT SUM(amount) FROM opex_general_expenses WHERE amount > 0")).scalar()
            st.write(f"General Expenses (DB): **${gen_exp:,.0f}**" if gen_exp else "General Expenses: $0")
            
            # Payroll total
            payroll = conn.execute(text("""
                SELECT SUM(r.annual_salary / 12.0 * s.headcount)
                FROM opex_staffing_plan s JOIN opex_roles r ON s.role_id = r.id
                WHERE s.headcount > 0
            """)).scalar()
            st.write(f"Payroll (DB): **${payroll:,.0f}**" if payroll else "Payroll: $0")
            
            st.markdown("---")
            st.subheader("2. Generated Cash Entries Summary")
            
            # Summarize cash_df
            summary = cash_df.groupby("Category")["Amount"].sum().sort_values()
            st.dataframe(summary.reset_index().style.format({"Amount": "${:,.0f}"}))
            
            total_inflow = cash_df[cash_df["Amount"] > 0]["Amount"].sum()
            total_outflow = cash_df[cash_df["Amount"] < 0]["Amount"].sum()
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Inflows", f"${total_inflow:,.0f}")
            c2.metric("Total Outflows", f"${abs(total_outflow):,.0f}")
            c3.metric("Net", f"${total_inflow + total_outflow:,.0f}")
            
            st.markdown("---")
            st.subheader("3. Cash Waterfall Result")
            
            waterfall = run_cash_waterfall(cash_df, equity, loc)
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Min Cash", f"${waterfall['Cash'].min():,.0f}")
            c2.metric("Max LOC Used", f"${waterfall['LOC_Used'].max():,.0f}")
            c3.metric("Final Cash", f"${waterfall['Cash'].iloc[-1]:,.0f}")
            
            st.markdown("---")
            st.subheader("4. Monthly Cash Flow Detail")
            
            # Monthly aggregation
            cash_df["Month"] = pd.to_datetime(cash_df["Date"]).dt.to_period("M").astype(str)
            monthly = cash_df.groupby(["Month", "Category"])["Amount"].sum().unstack(fill_value=0)
            monthly["Total"] = monthly.sum(axis=1)
            
            st.dataframe(monthly.style.format("${:,.0f}"), use_container_width=True)
            
            st.markdown("---")
            st.subheader("5. Raw Cash Entries (First 100)")
            st.dataframe(cash_df.head(100))
            
    except Exception as e:
        st.error(f"Error: {e}")
        import traceback
        st.code(traceback.format_exc())

if __name__ == "__main__":
    main()
