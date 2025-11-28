"""
IdleX ERP - Enterprise Resource Planning System
Version: 12.0 (Production - Full Featured)
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
import base64

# =============================================================================
# BRAND CONFIGURATION (IdleX 2025 Brandbook)
# =============================================================================
BRAND = {
    "primary": "#1E3A5F",      # Deep Navy
    "secondary": "#5B8DB8",    # Steel Blue  
    "accent": "#F4A261",       # Warm Amber
    "success": "#2A9D8F",      # Teal Green
    "warning": "#E9C46A",      # Golden Yellow
    "danger": "#E76F51",       # Coral Red
    "light": "#F8F9FA",
    "dark": "#1a1a2e",
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
def money(val, show_cents=False):
    """Format as currency. Negative values shown in parentheses."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "-"
    try:
        v = float(val)
        if show_cents:
            if v < 0:
                return f"(${abs(v):,.2f})"
            return f"${v:,.2f}"
        else:
            v = int(round(v))
            if v < 0:
                return f"(${abs(v):,})"
            return f"${v:,}"
    except:
        return "-"

def pct(val, decimals=1):
    """Format decimal as percentage."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "-"
    try:
        return f"{float(val) * 100:.{decimals}f}%"
    except:
        return "-"

def parse_date(val):
    """Parse various date formats."""
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
    """Get weekdays in a month."""
    num_days = calendar.monthrange(year, month)[1]
    days = [date(year, month, d) for d in range(1, num_days + 1)]
    return [d for d in days if d.weekday() < 5]

def get_quarter(month):
    return (month - 1) // 3 + 1

def get_logo_base64(filename):
    """Load logo as base64 for embedding."""
    try:
        # Try local file first
        if os.path.exists(filename):
            with open(filename, "rb") as f:
                return base64.b64encode(f.read()).decode()
        # Try in same directory as script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        filepath = os.path.join(script_dir, filename)
        if os.path.exists(filepath):
            with open(filepath, "rb") as f:
                return base64.b64encode(f.read()).decode()
    except:
        pass
    return None

# =============================================================================
# FINANCIAL ENGINE
# =============================================================================
def calculate_unit_material_cost(conn):
    """Calculate BOM cost per unit."""
    try:
        result = conn.execute(text("""
            SELECT COALESCE(SUM(p.cost * b.qty_per_unit), 0)
            FROM bom_items b JOIN part_master p ON b.part_id = p.id
        """)).scalar()
        return float(result) if result else 0.0
    except:
        return 0.0

def get_pricing_for_year(conn, year):
    """Get pricing config for a year."""
    try:
        row = conn.execute(text(
            "SELECT msrp, dealer_discount_pct FROM pricing_config WHERE year = :y"
        ), {"y": year}).fetchone()
        if row:
            return {"msrp": float(row[0]), "dealer_pct": float(row[1])}
    except:
        pass
    defaults = {2026: 15500, 2027: 13500, 2028: 11500}
    return {"msrp": defaults.get(year, 13500), "dealer_pct": 0.80}

def generate_financial_ledgers(conn):
    """Generate complete P&L and Cash flow ledgers."""
    pnl_entries = []
    cash_entries = []
    
    try:
        material_cost = calculate_unit_material_cost(conn)
        
        # Production revenue & COGS
        units = conn.execute(text("""
            SELECT serial_number, build_date, sales_channel 
            FROM production_unit ORDER BY build_date
        """)).fetchall()
        
        for serial, build_date, channel in units:
            build_dt = parse_date(build_date)
            if not build_dt:
                continue
            
            pricing = get_pricing_for_year(conn, build_dt.year)
            revenue = pricing["msrp"] if channel == "DIRECT" else pricing["msrp"] * pricing["dealer_pct"]
            
            pnl_entries.append({"Date": build_dt, "Type": "Revenue", "Category": f"{channel}", "Amount": revenue})
            pnl_entries.append({"Date": build_dt, "Type": "COGS", "Category": "Materials", "Amount": material_cost})
            cash_entries.append({"Date": build_dt, "Type": "Inflow", "Category": "Revenue", "Amount": revenue})
            cash_entries.append({"Date": build_dt, "Type": "Outflow", "Category": "Material", "Amount": material_cost})
        
        # Payroll
        staffing = conn.execute(text("""
            SELECT s.month_date, r.role_name, r.annual_salary, s.headcount
            FROM opex_staffing_plan s JOIN opex_roles r ON s.role_id = r.id
            WHERE s.headcount > 0
        """)).fetchall()
        
        for month_date, role, salary, headcount in staffing:
            m_date = parse_date(month_date)
            if not m_date:
                continue
            monthly = (float(salary) / 12.0) * float(headcount)
            pnl_entries.append({"Date": m_date, "Type": "OpEx", "Category": "Payroll", "Amount": monthly})
            cash_entries.append({"Date": m_date, "Type": "Outflow", "Category": "Payroll", "Amount": monthly})
        
        # General expenses
        expenses = conn.execute(text("""
            SELECT month_date, category, expense_type, amount 
            FROM opex_general_expenses WHERE amount > 0
        """)).fetchall()
        
        for month_date, category, exp_type, amount in expenses:
            m_date = parse_date(month_date)
            if not m_date:
                continue
            pnl_entries.append({"Date": m_date, "Type": "OpEx", "Category": exp_type, "Amount": float(amount)})
            cash_entries.append({"Date": m_date, "Type": "Outflow", "Category": "General", "Amount": float(amount)})
        
        return pd.DataFrame(pnl_entries), pd.DataFrame(cash_entries)
    except Exception as e:
        st.error(f"Ledger error: {e}")
        return pd.DataFrame(), pd.DataFrame()

def run_cash_waterfall(cash_df, starting_equity, loc_limit):
    """Process cash flow with LOC logic."""
    if cash_df.empty:
        return pd.DataFrame([{
            "Date": date.today(), "Cash": starting_equity, "LOC_Used": 0.0,
            "LOC_Available": loc_limit, "Total_Liquidity": starting_equity + loc_limit,
            "Cash_Crunch": False
        }])
    
    df = cash_df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")
    
    results = []
    cash = float(starting_equity)
    loc_used = 0.0
    
    for dt, group in df.groupby("Date"):
        inflows = group[group["Type"] == "Inflow"]["Amount"].sum()
        material_out = group[(group["Type"] == "Outflow") & (group["Category"] == "Material")]["Amount"].sum()
        other_out = group[(group["Type"] == "Outflow") & (group["Category"] != "Material")]["Amount"].sum()
        
        # Inflows pay down LOC first
        if inflows > 0 and loc_used > 0:
            paydown = min(inflows, loc_used)
            loc_used -= paydown
            inflows -= paydown
        cash += inflows
        
        # Non-material expenses (no LOC)
        cash -= other_out
        
        # Material can use LOC
        if material_out > 0:
            if cash >= material_out:
                cash -= material_out
            else:
                shortfall = material_out - max(cash, 0)
                cash = max(cash - material_out, 0)
                loc_avail = loc_limit - loc_used
                loc_draw = min(shortfall, loc_avail)
                loc_used += loc_draw
                if shortfall > loc_draw:
                    cash -= (shortfall - loc_draw)
        
        results.append({
            "Date": dt, "Cash": round(cash, 2), "LOC_Used": round(loc_used, 2),
            "LOC_Available": round(loc_limit - loc_used, 2),
            "Total_Liquidity": round(cash + loc_limit - loc_used, 2),
            "Cash_Crunch": cash < 0
        })
    
    return pd.DataFrame(results)

def apply_channel_mix(conn, year, quarter, direct_pct):
    """Apply channel mix to production units."""
    try:
        sm, em = (quarter - 1) * 3 + 1, quarter * 3
        units = conn.execute(text("""
            SELECT id FROM production_unit
            WHERE strftime('%Y', build_date) = :y
            AND CAST(strftime('%m', build_date) AS INTEGER) BETWEEN :sm AND :em
            ORDER BY build_date, id
        """), {"y": str(year), "sm": sm, "em": em}).fetchall()
        
        if not units:
            return 0
        
        direct_count = int(len(units) * direct_pct)
        for i, (uid,) in enumerate(units):
            ch = "DIRECT" if i < direct_count else "DEALER"
            conn.execute(text("UPDATE production_unit SET sales_channel=:ch WHERE id=:id"), {"ch": ch, "id": uid})
        conn.commit()
        return len(units)
    except Exception as e:
        st.error(f"Channel mix error: {e}")
        return 0

# =============================================================================
# STREAMLIT UI
# =============================================================================
def main():
    st.set_page_config(
        page_title="IdleX ERP",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown(f"""
    <style>
        .stApp {{
            background: linear-gradient(180deg, {BRAND['dark']} 0%, #16213e 100%);
        }}
        .main-header {{
            color: {BRAND['accent']};
            font-family: 'Helvetica Neue', sans-serif;
        }}
        section[data-testid="stSidebar"] {{
            background: linear-gradient(180deg, {BRAND['primary']} 0%, {BRAND['dark']} 100%);
        }}
        .stMetric {{
            background: rgba(255,255,255,0.05);
            padding: 1rem;
            border-radius: 8px;
        }}
        div[data-testid="stMetricValue"] {{
            color: {BRAND['light']};
        }}
    </style>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        # Logo
        logo_b64 = get_logo_base64("logo_white.png")
        if logo_b64:
            st.markdown(f'<img src="data:image/png;base64,{logo_b64}" style="width:180px;margin-bottom:10px;">', unsafe_allow_html=True)
        else:
            st.markdown("<h1 style='color:#F4A261;font-style:italic;'>idleX</h1>", unsafe_allow_html=True)
        
        st.caption("Enterprise Resource Planning")
        st.divider()
        
        page = st.radio(
            "Navigation",
            ["🏠 Dashboard", "📊 Financials", "📈 Production & Sales", "💼 OpEx Budget", "🔧 Supply Chain"],
            label_visibility="collapsed"
        )
        
        st.divider()
        
        with st.expander("⚙️ Database", expanded=False):
            if st.button("🔄 Rebuild Database", type="primary", use_container_width=True):
                try:
                    from seed_db import run_seed
                    run_seed()
                    st.success("✓ Rebuilt!")
                    st.cache_data.clear()
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")
            
            if st.button("🗑️ Clear Cache", use_container_width=True):
                st.cache_data.clear()
                st.rerun()
            
            st.caption("⚠️ Rebuild resets all data")
    
    # Main content header
    col1, col2 = st.columns([3, 1])
    with col1:
        logo_b64 = get_logo_base64("logo_white.png")
        if logo_b64:
            st.markdown(f'<img src="data:image/png;base64,{logo_b64}" style="width:200px;">', unsafe_allow_html=True)
        else:
            st.markdown("<h1 style='color:#F4A261;font-style:italic;'>idleX</h1>", unsafe_allow_html=True)
        st.caption("Enterprise Resource Planning")
    
    # Route to page
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

# =============================================================================
# DASHBOARD
# =============================================================================
def render_dashboard():
    st.header("Executive Dashboard")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("⚙️ Scenario Constraints")
        equity = st.number_input("Investor Equity ($)", value=1_500_000, step=100_000, format="%d")
        loc = st.number_input("Credit Limit (LOC) ($)", value=4_100_000, step=100_000, format="%d")
    
    with col2:
        try:
            with engine.connect() as conn:
                pnl_df, cash_df = generate_financial_ledgers(conn)
                
                if pnl_df.empty:
                    st.warning("No data. Click 'Rebuild Database' in sidebar.")
                    return
                
                # Calculate metrics
                revenue = pnl_df[pnl_df["Type"] == "Revenue"]["Amount"].sum()
                cogs = pnl_df[pnl_df["Type"] == "COGS"]["Amount"].sum()
                opex = pnl_df[pnl_df["Type"] == "OpEx"]["Amount"].sum()
                gross_profit = revenue - cogs
                net_income = gross_profit - opex
                units = conn.execute(text("SELECT COUNT(*) FROM production_unit")).scalar()
                
                # Metrics row
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Total Units", f"{units:,}")
                m2.metric("Total Revenue", money(revenue))
                m3.metric("Gross Profit", money(gross_profit), f"{gross_profit/revenue*100:.1f}%" if revenue else "")
                m4.metric("Net Income", money(net_income))
                
                # Cash waterfall
                waterfall = run_cash_waterfall(cash_df, equity, loc)
                
                st.markdown("---")
                c1, c2, c3 = st.columns(3)
                c1.metric("Min Cash", money(waterfall["Cash"].min()), 
                         delta="⚠️ Negative" if waterfall["Cash"].min() < 0 else "✓ Positive",
                         delta_color="inverse" if waterfall["Cash"].min() < 0 else "normal")
                c2.metric("Peak LOC Used", money(waterfall["LOC_Used"].max()))
                c3.metric("Final Cash", money(waterfall["Cash"].iloc[-1]))
                
                # Chart
                st.subheader("💰 Cash & Credit Forecast")
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=waterfall["Date"], y=waterfall["Cash"], 
                    name="Cash", fill="tozeroy", 
                    line=dict(color=BRAND["success"], width=2)
                ))
                fig.add_trace(go.Scatter(
                    x=waterfall["Date"], y=waterfall["LOC_Used"], 
                    name="LOC Used", 
                    line=dict(color=BRAND["warning"], width=2, dash="dash")
                ))
                fig.add_hline(y=0, line_dash="dot", line_color="red", opacity=0.5)
                fig.update_layout(
                    template="plotly_dark",
                    height=350,
                    margin=dict(l=0, r=0, t=30, b=0),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02)
                )
                st.plotly_chart(fig, use_container_width=True)
                
        except Exception as e:
            st.error(f"Dashboard error: {e}")

# =============================================================================
# FINANCIALS
# =============================================================================
def render_financials():
    st.header("📊 Financial Analysis")
    
    try:
        with engine.connect() as conn:
            pnl_df, _ = generate_financial_ledgers(conn)
            
            if pnl_df.empty:
                st.warning("No financial data available.")
                return
            
            # Summary metrics
            revenue = pnl_df[pnl_df["Type"] == "Revenue"]["Amount"].sum()
            cogs = pnl_df[pnl_df["Type"] == "COGS"]["Amount"].sum()
            opex = pnl_df[pnl_df["Type"] == "OpEx"]["Amount"].sum()
            gross_profit = revenue - cogs
            net_income = gross_profit - opex
            
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Revenue", money(revenue))
            c2.metric("COGS", money(cogs))
            c3.metric("Gross Profit", money(gross_profit))
            c4.metric("OpEx", money(opex))
            c5.metric("Net Income", money(net_income))
            
            st.markdown("---")
            
            # Monthly P&L - properly formatted
            st.subheader("Monthly P&L Statement")
            
            pnl_df["Month"] = pd.to_datetime(pnl_df["Date"]).dt.to_period("M").astype(str)
            
            # Pivot and calculate
            monthly = pnl_df.groupby(["Month", "Type"])["Amount"].sum().unstack(fill_value=0)
            
            # Ensure columns exist
            for col in ["Revenue", "COGS", "OpEx"]:
                if col not in monthly.columns:
                    monthly[col] = 0
            
            # Calculate derived columns
            monthly["Gross Profit"] = monthly["Revenue"] - monthly["COGS"]
            monthly["Net Income"] = monthly["Gross Profit"] - monthly["OpEx"]
            
            # Reorder columns
            monthly = monthly[["Revenue", "COGS", "Gross Profit", "OpEx", "Net Income"]]
            
            # Format for display
            st.dataframe(
                monthly.style.format("${:,.0f}").background_gradient(
                    subset=["Net Income"], cmap="RdYlGn", vmin=-500000, vmax=500000
                ),
                use_container_width=True,
                height=400
            )
            
            # Annual summary
            st.subheader("Annual Summary")
            pnl_df["Year"] = pd.to_datetime(pnl_df["Date"]).dt.year
            annual = pnl_df.groupby(["Year", "Type"])["Amount"].sum().unstack(fill_value=0)
            
            for col in ["Revenue", "COGS", "OpEx"]:
                if col not in annual.columns:
                    annual[col] = 0
            
            annual["Gross Profit"] = annual["Revenue"] - annual["COGS"]
            annual["Gross Margin %"] = (annual["Gross Profit"] / annual["Revenue"] * 100).round(1)
            annual["Net Income"] = annual["Gross Profit"] - annual["OpEx"]
            annual["Net Margin %"] = (annual["Net Income"] / annual["Revenue"] * 100).round(1)
            
            annual = annual[["Revenue", "COGS", "Gross Profit", "Gross Margin %", "OpEx", "Net Income", "Net Margin %"]]
            
            st.dataframe(
                annual.style.format({
                    "Revenue": "${:,.0f}",
                    "COGS": "${:,.0f}",
                    "Gross Profit": "${:,.0f}",
                    "Gross Margin %": "{:.1f}%",
                    "OpEx": "${:,.0f}",
                    "Net Income": "${:,.0f}",
                    "Net Margin %": "{:.1f}%",
                }),
                use_container_width=True
            )
            
    except Exception as e:
        st.error(f"Error: {e}")

# =============================================================================
# PRODUCTION & SALES
# =============================================================================
def render_production_sales():
    st.header("📈 Production & Sales Planning")
    
    tab1, tab2, tab3 = st.tabs(["📋 Production Manifest", "💰 Pricing Config", "📊 Channel Mix"])
    
    with tab1:
        render_manifest()
    with tab2:
        render_pricing()
    with tab3:
        render_channel_mix()

def render_manifest():
    try:
        with engine.connect() as conn:
            # Summary
            summary = pd.read_sql("""
                SELECT strftime('%Y', build_date) as Year, 
                       sales_channel as Channel, 
                       COUNT(*) as Units
                FROM production_unit 
                GROUP BY Year, Channel 
                ORDER BY Year, Channel
            """, conn)
            
            if summary.empty:
                st.warning("No production data.")
                return
            
            # Pivot table
            pivot = summary.pivot(index="Year", columns="Channel", values="Units").fillna(0).astype(int)
            pivot["Total"] = pivot.sum(axis=1)
            
            st.subheader("Production by Year")
            st.dataframe(pivot.style.format("{:,}"), use_container_width=True)
            
            # Chart
            fig = px.bar(
                summary, x="Year", y="Units", color="Channel", barmode="group",
                color_discrete_map={"DIRECT": BRAND["success"], "DEALER": BRAND["secondary"]}
            )
            fig.update_layout(template="plotly_dark", height=300)
            st.plotly_chart(fig, use_container_width=True)
            
    except Exception as e:
        st.error(f"Error: {e}")

def render_pricing():
    st.subheader("Pricing Configuration")
    
    try:
        with engine.connect() as conn:
            pricing = pd.read_sql("SELECT * FROM pricing_config ORDER BY year", conn)
            
            if pricing.empty:
                st.warning("No pricing data. Rebuild database.")
                return
            
            # Display all years
            for _, row in pricing.iterrows():
                year = int(row["year"])
                msrp = float(row["msrp"])
                dealer_pct = float(row["dealer_discount_pct"])
                
                with st.expander(f"**{year}** - MSRP: {money(msrp)} | Dealer Pays: {dealer_pct*100:.0f}%", expanded=(year == 2026)):
                    c1, c2 = st.columns(2)
                    new_msrp = c1.number_input(f"MSRP ({year})", value=msrp, step=100.0, key=f"msrp_{year}")
                    new_pct = c2.number_input(f"Dealer Pays % ({year})", value=dealer_pct*100, min_value=50.0, max_value=100.0, step=1.0, key=f"pct_{year}")
                    
                    dealer_price = new_msrp * (new_pct / 100)
                    
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Direct Price", money(new_msrp))
                    c2.metric("Dealer Price", money(dealer_price))
                    c3.metric("Dealer Margin", money(new_msrp - dealer_price))
                    
                    # Revenue preview
                    result = conn.execute(text("""
                        SELECT sales_channel, COUNT(*) FROM production_unit 
                        WHERE strftime('%Y', build_date) = :y GROUP BY sales_channel
                    """), {"y": str(year)}).fetchall()
                    
                    direct_u = sum(r[1] for r in result if r[0] == "DIRECT")
                    dealer_u = sum(r[1] for r in result if r[0] == "DEALER")
                    
                    if direct_u + dealer_u > 0:
                        st.caption(f"📊 {year} Revenue: {money(direct_u * new_msrp + dealer_u * dealer_price)} ({direct_u:,} direct + {dealer_u:,} dealer)")
                    
                    if st.button(f"💾 Save {year}", key=f"save_{year}"):
                        conn.execute(text("UPDATE pricing_config SET msrp=:m, dealer_discount_pct=:d WHERE year=:y"),
                                    {"y": year, "m": new_msrp, "d": new_pct/100})
                        conn.commit()
                        st.success(f"Saved {year}!")
                        st.cache_data.clear()
                        st.rerun()
                        
    except Exception as e:
        st.error(f"Error: {e}")

def render_channel_mix():
    st.subheader("Channel Mix by Quarter")
    st.caption("Direct % increases over time as brand awareness grows")
    
    try:
        with engine.connect() as conn:
            mix = pd.read_sql("SELECT * FROM channel_mix_config ORDER BY year, quarter", conn)
            
            if mix.empty:
                st.warning("No channel mix data.")
                return
            
            # Build editable dataframe
            data = []
            for _, row in mix.iterrows():
                direct = float(row["direct_pct"]) * 100
                data.append({
                    "id": int(row["id"]),
                    "Year": int(row["year"]),
                    "Quarter": int(row["quarter"]),
                    "Direct %": direct,
                    "Dealer %": 100 - direct,
                    "Label": f"Q{int(row['quarter'])} {int(row['year'])}"
                })
            
            df = pd.DataFrame(data)
            
            # Editable table
            edited = st.data_editor(
                df[["Label", "Direct %", "Dealer %"]],
                column_config={
                    "Label": st.column_config.TextColumn("Quarter", disabled=True),
                    "Direct %": st.column_config.NumberColumn("Direct %", min_value=0, max_value=100, step=1, format="%.1f%%"),
                    "Dealer %": st.column_config.NumberColumn("Dealer %", disabled=True, format="%.1f%%"),
                },
                use_container_width=True,
                hide_index=True
            )
            
            c1, c2 = st.columns(2)
            
            with c1:
                if st.button("💾 Save Channel Mix", type="primary"):
                    for idx, row in edited.iterrows():
                        conn.execute(text("UPDATE channel_mix_config SET direct_pct=:d WHERE id=:id"),
                                    {"d": row["Direct %"]/100, "id": df.iloc[idx]["id"]})
                    conn.commit()
                    st.success("Saved!")
                    st.cache_data.clear()
                    st.rerun()
            
            with c2:
                if st.button("🔄 Apply to Production"):
                    total = 0
                    for idx, row in edited.iterrows():
                        orig = df.iloc[idx]
                        updated = apply_channel_mix(conn, orig["Year"], orig["Quarter"], row["Direct %"]/100)
                        total += updated
                    st.success(f"Updated {total:,} units!")
                    st.cache_data.clear()
                    st.rerun()
            
            # Visualization
            st.markdown("---")
            chart_df = df.melt(id_vars=["Label"], value_vars=["Direct %", "Dealer %"], var_name="Channel", value_name="Pct")
            fig = px.bar(chart_df, x="Label", y="Pct", color="Channel", barmode="stack",
                        color_discrete_map={"Direct %": BRAND["success"], "Dealer %": BRAND["secondary"]})
            fig.update_layout(template="plotly_dark", height=250)
            st.plotly_chart(fig, use_container_width=True)
            
    except Exception as e:
        st.error(f"Error: {e}")

# =============================================================================
# OPEX BUDGET
# =============================================================================
def render_opex():
    st.header("💼 OpEx Budget")
    
    tab1, tab2 = st.tabs(["👥 Staffing Plan", "📋 General Expenses"])
    
    with tab1:
        render_staffing()
    with tab2:
        render_expenses()

def render_staffing():
    st.subheader("Staffing Plan")
    
    try:
        with engine.connect() as conn:
            # Roles
            roles = pd.read_sql("SELECT * FROM opex_roles ORDER BY annual_salary DESC", conn)
            
            st.markdown("##### Role Definitions")
            
            # Editable roles
            edited_roles = st.data_editor(
                roles[["id", "role_name", "annual_salary"]].rename(columns={
                    "role_name": "Role", "annual_salary": "Annual Salary"
                }),
                column_config={
                    "id": None,  # Hide ID
                    "Role": st.column_config.TextColumn("Role", width="medium"),
                    "Annual Salary": st.column_config.NumberColumn("Annual Salary", format="$%d", step=5000),
                },
                use_container_width=True,
                hide_index=True
            )
            
            if st.button("💾 Save Role Changes"):
                for idx, row in edited_roles.iterrows():
                    conn.execute(text("UPDATE opex_roles SET role_name=:n, annual_salary=:s WHERE id=:id"),
                                {"n": row["Role"], "s": row["Annual Salary"], "id": roles.iloc[idx]["id"]})
                conn.commit()
                st.success("Roles saved!")
                st.cache_data.clear()
                st.rerun()
            
            st.markdown("---")
            st.markdown("##### Monthly Headcount")
            
            # Staffing pivot
            staffing = pd.read_sql("""
                SELECT s.id, s.month_date, r.role_name, s.headcount
                FROM opex_staffing_plan s 
                JOIN opex_roles r ON s.role_id = r.id
                ORDER BY s.month_date, r.role_name
            """, conn)
            
            if not staffing.empty:
                pivot = staffing.pivot(index="month_date", columns="role_name", values="headcount").fillna(0)
                st.dataframe(pivot.style.format("{:.1f}"), use_container_width=True, height=400)
                
                # Total headcount chart
                pivot["Total"] = pivot.sum(axis=1)
                fig = px.area(pivot.reset_index(), x="month_date", y="Total", title="Total Headcount Over Time")
                fig.update_layout(template="plotly_dark", height=250)
                st.plotly_chart(fig, use_container_width=True)
            
    except Exception as e:
        st.error(f"Error: {e}")

def render_expenses():
    st.subheader("General Expenses")
    
    try:
        with engine.connect() as conn:
            expenses = pd.read_sql("""
                SELECT id, month_date, expense_type, category, amount 
                FROM opex_general_expenses 
                ORDER BY expense_type, category, month_date
            """, conn)
            
            if expenses.empty:
                st.info("No expenses.")
                return
            
            # Summary by type
            by_type = expenses.groupby("expense_type")["amount"].sum()
            cols = st.columns(len(by_type))
            for i, (t, amt) in enumerate(by_type.items()):
                cols[i].metric(f"Total {t}", money(amt))
            
            st.markdown("---")
            
            # Editable expenses
            edited = st.data_editor(
                expenses[["month_date", "expense_type", "category", "amount"]].rename(columns={
                    "month_date": "Month", "expense_type": "Type", "category": "Category", "amount": "Amount"
                }),
                column_config={
                    "Month": st.column_config.TextColumn("Month", disabled=True),
                    "Type": st.column_config.TextColumn("Type", disabled=True),
                    "Category": st.column_config.TextColumn("Category", disabled=True),
                    "Amount": st.column_config.NumberColumn("Amount ($)", min_value=0, format="$%d"),
                },
                use_container_width=True,
                hide_index=True,
                height=400
            )
            
            if st.button("💾 Save Expense Changes", type="primary"):
                for idx, row in edited.iterrows():
                    conn.execute(text("UPDATE opex_general_expenses SET amount=:a WHERE id=:id"),
                                {"a": row["Amount"], "id": expenses.iloc[idx]["id"]})
                conn.commit()
                st.success("Expenses saved!")
                st.cache_data.clear()
                st.rerun()
                
    except Exception as e:
        st.error(f"Error: {e}")

# =============================================================================
# SUPPLY CHAIN
# =============================================================================
def render_supply_chain():
    st.header("🔧 Supply Chain")
    
    tab1, tab2 = st.tabs(["📦 Part Master", "🧩 Bill of Materials"])
    
    with tab1:
        try:
            with engine.connect() as conn:
                parts = pd.read_sql("SELECT * FROM part_master ORDER BY sku", conn)
                
                st.subheader("Part Master")
                st.dataframe(
                    parts.style.format({
                        "cost": "${:,.2f}",
                        "deposit_pct": "{:.0%}"
                    }),
                    use_container_width=True
                )
                
                # Safiery highlight
                safiery = parts[parts["lead_time"] >= 100]
                if not safiery.empty:
                    st.subheader("🔋 Long Lead Time Parts (Safiery)")
                    st.dataframe(safiery[["sku", "name", "cost", "lead_time", "deposit_pct"]], use_container_width=True)
                    
        except Exception as e:
            st.error(f"Error: {e}")
    
    with tab2:
        try:
            with engine.connect() as conn:
                bom = pd.read_sql("""
                    SELECT p.sku, p.name, b.qty_per_unit as qty, p.cost, 
                           (b.qty_per_unit * p.cost) as extended
                    FROM bom_items b 
                    JOIN part_master p ON b.part_id = p.id 
                    ORDER BY extended DESC
                """, conn)
                
                if not bom.empty:
                    total = bom["extended"].sum()
                    st.metric("BOM Cost per Unit", money(total, show_cents=True))
                    
                    st.dataframe(
                        bom.style.format({
                            "cost": "${:,.2f}",
                            "extended": "${:,.2f}"
                        }),
                        use_container_width=True
                    )
                    
                    # Pie chart
                    fig = px.pie(bom.head(6), values="extended", names="name", title="Cost Breakdown (Top 6)")
                    fig.update_layout(template="plotly_dark", height=300)
                    st.plotly_chart(fig, use_container_width=True)
                    
        except Exception as e:
            st.error(f"Error: {e}")

if __name__ == "__main__":
    main()
