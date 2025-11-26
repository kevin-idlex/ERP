"""
IdleX CFO Console - Dashboard Application
Version: 3.4 (Live Constraint Metrics + Auto-Optimization)
Includes: Scenario & Optimization Engine, Smart Production Scheduler, GAAP Financials
"""

import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text
from datetime import timedelta, date
import plotly.express as px
import calendar
import os
import logging
import seed_db

# =============================================================================
# CONFIGURATION
# =============================================================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Business Constants
MSRP_PRICE = 8500.00
DEALER_DISCOUNT_RATE = 0.75
DIRECT_SALES_TARGET_PCT = 0.25
DEALER_PAYMENT_LAG_DAYS = 30

# Optimizer Constants
OPTIMIZER_ITERATIONS = 15
GROWTH_RATE_MIN = 0.0
GROWTH_RATE_MAX = 100.0

# =============================================================================
# PAGE SETUP
# =============================================================================
st.set_page_config(page_title="IdleX CFO Console", layout="wide")

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
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 10px;
        padding: 15px;
        margin: 5px 0;
        border-left: 4px solid #10B981;
    }
    .metric-card.warning {
        border-left-color: #F59E0B;
    }
    .metric-card.danger {
        border-left-color: #EF4444;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# DATABASE CONNECTION
# =============================================================================
@st.cache_resource
def get_db_engine():
    """Create database engine with PostgreSQL/SQLite support."""
    db_url = os.getenv("DATABASE_URL")
    if db_url:
        if db_url.startswith("postgres://"):
            db_url = db_url.replace("postgres://", "postgresql://", 1)
        return create_engine(db_url)
    return create_engine('sqlite:///idlex.db')


def get_db_type():
    """Detect database type."""
    db_url = os.getenv("DATABASE_URL")
    if db_url and ("postgresql" in db_url or "postgres" in db_url):
        return "postgresql"
    return "sqlite"


engine = get_db_engine()
DB_TYPE = get_db_type()


def get_year_month_sql(column: str) -> str:
    """Generate database-specific year-month extraction SQL."""
    if DB_TYPE == "postgresql":
        return f"TO_CHAR({column}, 'YYYY-MM')"
    return f"strftime('%Y-%m', {column})"


def get_upsert_sql() -> str:
    """Generate database-specific upsert SQL for staffing plan."""
    if DB_TYPE == "postgresql":
        return """
            INSERT INTO opex_staffing_plan (role_id, month_date, headcount) 
            VALUES (:rid, :dt, :hc)
            ON CONFLICT (role_id, month_date) 
            DO UPDATE SET headcount = EXCLUDED.headcount
        """
    return """
        INSERT OR REPLACE INTO opex_staffing_plan 
        (id, role_id, month_date, headcount) 
        VALUES (
            (SELECT id FROM opex_staffing_plan WHERE role_id=:rid AND month_date=:dt), 
            :rid, :dt, :hc
        )
    """


def get_config_upsert_sql() -> str:
    """Generate database-specific upsert SQL for global_config."""
    if DB_TYPE == "postgresql":
        return """
            INSERT INTO global_config (setting_key, setting_value) 
            VALUES (:key, :val)
            ON CONFLICT (setting_key) 
            DO UPDATE SET setting_value = EXCLUDED.setting_value
        """
    return """
        INSERT OR REPLACE INTO global_config (setting_key, setting_value) 
        VALUES (:key, :val)
    """


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
def get_workdays(year: int, month: int, start_threshold: date = None) -> list:
    """Get list of workdays (Mon-Fri) for a given month."""
    num_days = calendar.monthrange(year, month)[1]
    days = [date(year, month, d) for d in range(1, num_days + 1)]
    valid_days = [d for d in days if d.weekday() < 5]
    if start_threshold:
        valid_days = [d for d in valid_days if d >= start_threshold]
    return valid_days


def format_banker(val) -> str:
    """Format numbers in accounting style with parentheses for negatives."""
    if pd.isna(val) or val is None or val == "":
        return ""
    if isinstance(val, str):
        return val
    try:
        if val < 0:
            return f"({abs(val):,.0f})"
        return f"{val:,.0f}"
    except (TypeError, ValueError):
        return str(val)


def render_financial_statement(df: pd.DataFrame, title: str) -> None:
    """Render DataFrame as a GAAP-style financial statement."""
    html = f"<h3>{title}</h3><div style='border:1px solid #ddd; overflow-x:auto;'>"
    html += "<table class='financial-table'>"
    html += "<thead><tr><th class='row-header'>Account</th>"
    
    for col in df.columns:
        html += f"<th>{col}</th>"
    html += "</tr></thead><tbody>"
    
    section_headers = ['Revenue', 'Cost of Goods Sold', 'Operating Expenses', 'Operating Activities']
    total_rows = ['Gross Profit', 'Net Cash Flow', 'Total OpEx']
    grand_total_rows = ['Net Income', 'Ending Cash Balance']

    for index, row in df.iterrows():
        clean_index = str(index).strip()
        is_header = clean_index in section_headers
        
        if is_header:
            row_class = "section-header"
        elif clean_index in total_rows:
            row_class = "total-row"
        elif clean_index in grand_total_rows:
            row_class = "grand-total"
        else:
            row_class = "indent"
        
        html += f"<tr class='{row_class}'><td class='row-header'>{clean_index}</td>"
        
        if is_header:
            for _ in df.columns:
                html += "<td></td>"
        else:
            for col in df.columns:
                html += f"<td style='text-align: right;'>{format_banker(row[col])}</td>"
        html += "</tr>"
        
    html += "</tbody></table></div>"
    st.markdown(html, unsafe_allow_html=True)


def parse_serial_number(serial: str) -> int:
    """Safely extract numeric portion from serial number."""
    if not serial:
        return 0
    try:
        digits = ''.join(filter(str.isdigit, serial))
        return int(digits) if digits else 0
    except (ValueError, TypeError):
        return 0


# =============================================================================
# FINANCIAL ENGINE
# =============================================================================
def generate_financials(units_override: pd.DataFrame = None, start_cash_override: float = None) -> tuple:
    """
    Generate P&L and Cash Flow statements.
    
    Args:
        units_override: Optional DataFrame to use instead of database units (for simulation)
        start_cash_override: Optional starting cash value (for simulation)
    
    Returns:
        Tuple of (df_pnl, df_cash) DataFrames
    """
    try:
        df_parts = pd.read_sql("SELECT * FROM part_master", engine)
        df_bom = pd.read_sql("SELECT * FROM bom_items", engine)
        df_opex = pd.read_sql("SELECT * FROM opex_staffing_plan", engine)
        df_roles = pd.read_sql("SELECT * FROM opex_roles", engine)
        
        try:
            df_gen_exp = pd.read_sql("SELECT * FROM opex_general_expenses", engine)
            df_gen_exp['month_date'] = pd.to_datetime(df_gen_exp['month_date'])
        except Exception:
            df_gen_exp = pd.DataFrame(columns=['month_date', 'category', 'expense_type', 'amount'])
        
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

    if df_units.empty:
        return pd.DataFrame(), pd.DataFrame()

    # Convert dates
    df_units['build_date'] = pd.to_datetime(df_units['build_date'])
    df_opex['month_date'] = pd.to_datetime(df_opex['month_date'])

    ledger = []

    # -------------------------------------------------------------------------
    # A. Calculate unit material cost (optimized merge)
    # -------------------------------------------------------------------------
    unit_mat_cost = 0
    if not df_bom.empty and not df_parts.empty:
        bom_with_parts = pd.merge(df_bom, df_parts, left_on='part_id', right_on='id')
        unit_mat_cost = (bom_with_parts['qty_per_unit'] * bom_with_parts['cost']).sum()

    # -------------------------------------------------------------------------
    # B. Revenue & COGS
    # -------------------------------------------------------------------------
    for _, unit in df_units.iterrows():
        is_direct = unit['sales_channel'] == 'DIRECT'
        rev_amt = MSRP_PRICE if is_direct else MSRP_PRICE * DEALER_DISCOUNT_RATE
        pnl_date = unit['build_date']
        cash_lag = 0 if is_direct else DEALER_PAYMENT_LAG_DAYS

        # Revenue
        ledger.append({
            "Date": pnl_date,
            "Category": "Sales of Goods",
            "Type": "Revenue",
            "Amount": rev_amt,
            "Report": "PnL"
        })
        
        # Cash collection
        ledger.append({
            "Date": pnl_date + timedelta(days=cash_lag),
            "Category": "Cash from Customers",
            "Type": "Operations",
            "Amount": rev_amt,
            "Report": "Cash"
        })
        
        # COGS
        ledger.append({
            "Date": pnl_date,
            "Category": "Raw Materials",
            "Type": "COGS",
            "Amount": -unit_mat_cost,
            "Report": "PnL"
        })

    # -------------------------------------------------------------------------
    # C. Supply Chain Cash Flows
    # -------------------------------------------------------------------------
    monthly_builds = df_units.groupby(pd.Grouper(key='build_date', freq='MS')).size()
    
    for month_start, count in monthly_builds.items():
        if count == 0:
            continue
        delivery = month_start
        
        for _, part in df_parts.iterrows():
            bom_row = df_bom[df_bom['part_id'] == part['id']]
            if bom_row.empty:
                continue
            
            total_po_cost = bom_row.iloc[0]['qty_per_unit'] * count * part['cost']
            
            # Deposit
            if part['deposit_pct'] > 0:
                ledger.append({
                    "Date": delivery + timedelta(days=int(part['deposit_days'])),
                    "Category": "Supplier Deposits",
                    "Type": "Operations",
                    "Amount": -(total_po_cost * part['deposit_pct']),
                    "Report": "Cash"
                })
            
            # Balance
            if part['deposit_pct'] < 1.0:
                ledger.append({
                    "Date": delivery + timedelta(days=int(part['balance_days'])),
                    "Category": "Supplier Settlements",
                    "Type": "Operations",
                    "Amount": -(total_po_cost * (1 - part['deposit_pct'])),
                    "Report": "Cash"
                })

    # -------------------------------------------------------------------------
    # D. Payroll
    # -------------------------------------------------------------------------
    if not df_opex.empty and not df_roles.empty:
        opex_merged = pd.merge(df_opex, df_roles, left_on='role_id', right_on='id')
        
        for _, row in opex_merged.iterrows():
            monthly_cost = (row['annual_salary'] / 12) * row['headcount']
            
            if monthly_cost > 0:
                is_direct_labor = "Assembler" in row['role_name']
                category = "Direct Labor" if is_direct_labor else "Salaries & Wages"
                pnl_type = "COGS" if is_direct_labor else "OpEx"
                
                ledger.append({
                    "Date": row['month_date'],
                    "Category": category,
                    "Type": pnl_type,
                    "Amount": -monthly_cost,
                    "Report": "PnL"
                })
                
                ledger.append({
                    "Date": row['month_date'],
                    "Category": "Payroll Paid",
                    "Type": "Operations",
                    "Amount": -monthly_cost,
                    "Report": "Cash"
                })

    # -------------------------------------------------------------------------
    # E. General Expenses
    # -------------------------------------------------------------------------
    if not df_gen_exp.empty:
        for _, row in df_gen_exp.iterrows():
            if row['amount'] > 0:
                ledger.append({
                    "Date": row['month_date'],
                    "Category": row['category'],
                    "Type": "OpEx",
                    "Amount": -row['amount'],
                    "Report": "PnL"
                })
                
                ledger.append({
                    "Date": row['month_date'],
                    "Category": "OpEx Paid",
                    "Type": "Operations",
                    "Amount": -row['amount'],
                    "Report": "Cash"
                })

    # -------------------------------------------------------------------------
    # F. Build Output
    # -------------------------------------------------------------------------
    if not ledger:
        return pd.DataFrame(), pd.DataFrame()
    
    df_master = pd.DataFrame(ledger)
    df_pnl = df_master[df_master['Report'] == "PnL"].sort_values('Date')
    df_cash = df_master[df_master['Report'] == "Cash"].sort_values('Date')
    df_cash['Cash_Balance'] = df_cash['Amount'].cumsum() + start_cash
    
    return df_pnl, df_cash


# =============================================================================
# SIMULATION ENGINE
# =============================================================================
def simulate_growth_scenario(start_units: int, growth_pct: float, start_date: date, months: int = 36) -> pd.DataFrame:
    """
    Generate simulated production units for growth scenario analysis.
    """
    sim_units = []
    current_units = start_units
    serial_counter = 1
    current_date = start_date.replace(day=1)
    
    for _ in range(months):
        target = int(current_units)
        workdays = get_workdays(current_date.year, current_date.month)
        
        if target > 0 and workdays:
            direct_qty = int(target * DIRECT_SALES_TARGET_PCT)
            dealer_qty = target - direct_qty
            pool = ['DIRECT'] * direct_qty + ['DEALER'] * dealer_qty
            
            for idx, channel in enumerate(pool):
                build_day = workdays[idx % len(workdays)]
                sim_units.append({
                    "serial_number": f"SIM-{serial_counter:05d}",
                    "build_date": build_day,
                    "sales_channel": channel,
                    "status": "SIMULATED"
                })
                serial_counter += 1
        
        current_units = current_units * (1 + (growth_pct / 100))
        
        if current_date.month == 12:
            current_date = date(current_date.year + 1, 1, 1)
        else:
            current_date = date(current_date.year, current_date.month + 1, 1)
    
    return pd.DataFrame(sim_units)


def find_max_growth_rate(start_units: int, start_cash: float, credit_limit: float, 
                         start_date: date, months: int) -> tuple:
    """
    Binary search to find maximum sustainable growth rate.
    
    Returns:
        Tuple of (best_growth_rate, best_cash_dataframe, total_units, total_revenue)
    """
    best_rate = 0.0
    best_cash_df = pd.DataFrame()
    best_pnl_df = pd.DataFrame()
    floor_limit = -credit_limit
    
    low, high = GROWTH_RATE_MIN, GROWTH_RATE_MAX
    
    for _ in range(OPTIMIZER_ITERATIONS):
        mid = (low + high) / 2
        
        sim_units = simulate_growth_scenario(start_units, mid, start_date, months)
        sim_pnl, sim_cash = generate_financials(units_override=sim_units, start_cash_override=start_cash)
        
        min_cash = sim_cash['Cash_Balance'].min() if not sim_cash.empty else 0
        
        if min_cash >= floor_limit:
            best_rate = mid
            best_cash_df = sim_cash
            best_pnl_df = sim_pnl
            low = mid
        else:
            high = mid
    
    # Calculate summary metrics
    total_units = len(simulate_growth_scenario(start_units, best_rate, start_date, months))
    total_revenue = best_pnl_df[best_pnl_df['Category'] == 'Sales of Goods']['Amount'].sum() if not best_pnl_df.empty else 0
    
    return best_rate, best_cash_df, total_units, total_revenue


def quick_scenario_metrics(start_units: int, growth_pct: float, start_cash: float, 
                           credit_limit: float, start_date: date, months: int) -> dict:
    """
    Quickly calculate key metrics for a given scenario.
    
    Returns:
        Dictionary with total_units, total_revenue, min_cash, is_feasible
    """
    sim_units = simulate_growth_scenario(start_units, growth_pct, start_date, months)
    sim_pnl, sim_cash = generate_financials(units_override=sim_units, start_cash_override=start_cash)
    
    if sim_cash.empty:
        return {
            'total_units': 0,
            'total_revenue': 0,
            'min_cash': start_cash,
            'end_cash': start_cash,
            'is_feasible': True
        }
    
    total_revenue = sim_pnl[sim_pnl['Category'] == 'Sales of Goods']['Amount'].sum() if not sim_pnl.empty else 0
    min_cash = sim_cash['Cash_Balance'].min()
    end_cash = sim_cash.iloc[-1]['Cash_Balance']
    
    return {
        'total_units': len(sim_units),
        'total_revenue': total_revenue,
        'min_cash': min_cash,
        'end_cash': end_cash,
        'is_feasible': min_cash >= -credit_limit
    }


# =============================================================================
# PRODUCTION SCHEDULER
# =============================================================================
def regenerate_production_schedule(edit_plan: pd.DataFrame, start_date: date) -> None:
    """Regenerate production schedule based on target plan."""
    with engine.connect() as conn:
        try:
            conn.execute(text("DELETE FROM production_unit WHERE status = 'PLANNED'"))
            
            last_sn = conn.execute(
                text("SELECT serial_number FROM production_unit ORDER BY id DESC LIMIT 1")
            ).scalar()
            next_sn = parse_serial_number(last_sn) + 1
            
            for _, row in edit_plan.iterrows():
                target_count = int(row['Target'])
                if target_count == 0:
                    continue
                
                month_date = row['Month']
                if isinstance(month_date, pd.Timestamp):
                    month_date = month_date.date()
                
                month_str = month_date.strftime('%Y-%m')
                
                year_month_sql = get_year_month_sql('build_date')
                locked_count = conn.execute(
                    text(f"SELECT COUNT(*) FROM production_unit WHERE {year_month_sql} = :ms AND status != 'PLANNED'"),
                    {"ms": month_str}
                ).scalar() or 0
                
                to_build = target_count - locked_count
                if to_build <= 0:
                    continue
                
                is_start_month = (month_date.year == start_date.year and month_date.month == start_date.month)
                threshold = start_date if is_start_month else None
                
                last_day = date(month_date.year, month_date.month, calendar.monthrange(month_date.year, month_date.month)[1])
                if last_day < start_date:
                    continue
                
                workdays = get_workdays(month_date.year, month_date.month, threshold)
                if not workdays:
                    continue
                
                direct_count = int(to_build * DIRECT_SALES_TARGET_PCT)
                pool = ['DIRECT'] * direct_count + ['DEALER'] * (to_build - direct_count)
                
                for idx, channel in enumerate(pool):
                    build_day = workdays[idx % len(workdays)]
                    conn.execute(
                        text("INSERT INTO production_unit (serial_number, build_date, sales_channel, status) VALUES (:sn, :bd, :ch, 'PLANNED')"),
                        {"sn": f"IDX-{next_sn:04d}", "bd": build_day, "ch": channel}
                    )
                    next_sn += 1
            
            conn.commit()
            
        except Exception as e:
            conn.rollback()
            raise


# =============================================================================
# EXPENSE TAB HELPER
# =============================================================================
def render_expense_tab(expense_type: str) -> None:
    """Render an expense budget tab (R&D or SG&A)."""
    st.subheader(f"{expense_type} Budget")
    
    df_expenses = pd.read_sql(
        text("SELECT * FROM opex_general_expenses WHERE expense_type = :et"),
        engine,
        params={"et": expense_type}
    )
    
    if not df_expenses.empty:
        df_expenses['Month'] = pd.to_datetime(df_expenses['month_date']).dt.strftime('%Y-%m')
        pivot_exp = df_expenses.pivot(index='category', columns='Month', values='amount').reset_index()
    else:
        dates = pd.date_range('2026-01-01', '2027-06-01', freq='MS')
        pivot_exp = pd.DataFrame(columns=['category'] + [d.strftime('%Y-%m') for d in dates])
    
    edited_exp = st.data_editor(
        pivot_exp,
        use_container_width=True,
        num_rows="dynamic",
        key=f"expense_editor_{expense_type}"
    )
    
    if st.button(f"💾 Save {expense_type} Budget", key=f"save_btn_{expense_type}"):
        with engine.connect() as conn:
            try:
                conn.execute(
                    text("DELETE FROM opex_general_expenses WHERE expense_type = :et"),
                    {"et": expense_type}
                )
                
                melted = edited_exp.melt(id_vars=['category'], var_name='Month', value_name='amount')
                
                for _, r in melted.iterrows():
                    if pd.notna(r['amount']) and pd.notna(r['category']):
                        dt = date.fromisoformat(r['Month'] + "-01")
                        conn.execute(
                            text("""
                                INSERT INTO opex_general_expenses (category, expense_type, month_date, amount) 
                                VALUES (:c, :t, :d, :a)
                            """),
                            {"c": r['category'], "t": expense_type, "d": dt, "a": r['amount']}
                        )
                
                conn.commit()
                st.success(f"{expense_type} budget saved!")
                st.rerun()
            except Exception as e:
                conn.rollback()
                st.error(f"Save failed: {e}")


# =============================================================================
# HEADCOUNT GRID BUILDER
# =============================================================================
def build_headcount_grid(df_roles: pd.DataFrame, df_staffing: pd.DataFrame) -> pd.DataFrame:
    """Build a complete headcount grid showing ALL roles."""
    if df_roles.empty:
        return pd.DataFrame(columns=['role_name'])
    
    months = pd.date_range('2026-01-01', '2027-12-01', freq='MS')
    month_strings = [m.strftime('%Y-%m') for m in months]
    
    grid = df_roles[['id', 'role_name']].copy()
    grid = grid.rename(columns={'id': 'role_id'})
    
    for m in month_strings:
        grid[m] = 0.0
    
    if not df_staffing.empty:
        df_staffing['Month'] = pd.to_datetime(df_staffing['month_date']).dt.strftime('%Y-%m')
        
        for _, staff_row in df_staffing.iterrows():
            role_mask = grid['role_id'] == staff_row['role_id']
            month_col = staff_row['Month']
            if month_col in grid.columns:
                grid.loc[role_mask, month_col] = staff_row['headcount']
    
    grid = grid.drop(columns=['role_id'])
    
    return grid


# =============================================================================
# MAIN APPLICATION
# =============================================================================
def main():
    # =========================================================================
    # SIDEBAR
    # =========================================================================
    st.sidebar.title("IdleX CFO Console")
    
    if st.sidebar.button("⚠️ Rebuild Database"):
        with st.spinner("Resetting Database..."):
            seed_db.run_seed()
        st.sidebar.success("Done! Refresh page.")
        st.rerun()

    view = st.sidebar.radio(
        "Navigation",
        [
            "Executive Dashboard",
            "Scenario & Optimization",
            "Financial Statements",
            "Production & Sales",
            "OpEx Planning",
            "BOM & Supply Chain"
        ]
    )

    # Generate financials for most views
    df_pnl, df_cash = pd.DataFrame(), pd.DataFrame()
    if view not in ["Scenario & Optimization"]:
        df_pnl, df_cash = generate_financials()

    # =========================================================================
    # EXECUTIVE DASHBOARD
    # =========================================================================
    if view == "Executive Dashboard":
        st.title("Executive Dashboard")
        st.caption("📊 Showing actual production data from database")
        
        if not df_pnl.empty:
            years = sorted(df_pnl['Date'].dt.year.unique().tolist())
            st.sidebar.divider()
            selected_period = st.sidebar.selectbox("Fiscal Year:", ["All Time"] + years)
            
            if selected_period == "All Time":
                pnl_view, cash_view = df_pnl, df_cash
            else:
                pnl_view = df_pnl[df_pnl['Date'].dt.year == selected_period]
                cash_view = df_cash[df_cash['Date'].dt.year == selected_period]
            
            rev = pnl_view[pnl_view['Category'] == 'Sales of Goods']['Amount'].sum()
            cogs = abs(pnl_view[pnl_view['Type'] == 'COGS']['Amount'].sum())
            margin = rev - cogs
            min_cash = cash_view['Cash_Balance'].min() if not cash_view.empty else 0
            end_cash = cash_view.iloc[-1]['Cash_Balance'] if not cash_view.empty else 0
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Revenue", f"${rev:,.0f}")
            c2.metric("Gross Margin", f"${margin:,.0f}")
            c3.metric("Min Net Cash", f"${min_cash:,.0f}", delta_color="inverse")
            c4.metric("Ending Cash", f"${end_cash:,.0f}")
            
            fig = px.area(
                cash_view, x='Date', y='Cash_Balance',
                title="Liquidity Forecast",
                color_discrete_sequence=['#10B981']
            )
            fig.add_hline(y=0, line_dash="dash", line_color="red")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Database is empty. Click 'Rebuild Database' in the sidebar.")

    # =========================================================================
    # SCENARIO & OPTIMIZATION
    # =========================================================================
    elif view == "Scenario & Optimization":
        st.title("Growth Strategy Engine")
        st.caption("🔮 Simulation tool — adjust parameters to see projected outcomes")
        
        # Load current config
        try:
            config = pd.read_sql("SELECT * FROM global_config", engine)
            cash_row = config[config['setting_key'] == 'start_cash']
            loc_row = config[config['setting_key'] == 'loc_limit']
            def_cash = float(cash_row['setting_value'].values[0]) if not cash_row.empty else 1000000.0
            def_loc = float(loc_row['setting_value'].values[0]) if not loc_row.empty else 500000.0
        except Exception:
            def_cash = 1000000.0
            def_loc = 500000.0
        
        # Initialize session state for results
        if 'opt_results' not in st.session_state:
            st.session_state.opt_results = None
        
        c1, c2 = st.columns([1, 2])
        
        with c1:
            st.subheader("📝 Constraints")
            
            inv_cash = st.number_input(
                "Investor Equity ($)", 
                value=def_cash, 
                step=100000.0, 
                format="%.0f",
                key="inv_cash"
            )
            loc_limit = st.number_input(
                "Bank Credit Limit ($)", 
                value=def_loc, 
                step=100000.0, 
                format="%.0f",
                key="loc_limit"
            )
            start_vol = st.number_input(
                "Starting Monthly Units", 
                value=50, 
                min_value=1,
                key="start_vol"
            )
            sim_months = st.slider(
                "Forecast Horizon (Months)", 
                12, 60, 36,
                key="sim_months"
            )
            sim_start = st.date_input(
                "Start Date", 
                value=date(2026, 1, 1),
                key="sim_start"
            )
            
            st.divider()
            
            # SAVE & OPTIMIZE BUTTON
            if st.button("💾 Save & Optimize", type="primary", use_container_width=True):
                # Save constraints to database
                with engine.connect() as conn:
                    try:
                        upsert_sql = get_config_upsert_sql()
                        conn.execute(text(upsert_sql), {"key": "start_cash", "val": str(inv_cash)})
                        conn.execute(text(upsert_sql), {"key": "loc_limit", "val": str(loc_limit)})
                        conn.commit()
                    except Exception as e:
                        conn.rollback()
                        st.error(f"Save failed: {e}")
                
                # Run optimization
                with st.spinner("Optimizing growth rate..."):
                    best_rate, best_cash_df, total_units, total_revenue = find_max_growth_rate(
                        start_units=start_vol,
                        start_cash=inv_cash,
                        credit_limit=loc_limit,
                        start_date=sim_start,
                        months=sim_months
                    )
                    
                    st.session_state.opt_results = {
                        'rate': best_rate,
                        'cash_df': best_cash_df,
                        'units': total_units,
                        'revenue': total_revenue,
                        'inv_cash': inv_cash,
                        'loc_limit': loc_limit
                    }
                
                st.rerun()
            
            # ─────────────────────────────────────────────────────────────────
            # LIVE RESULTS PANEL (in constraints column)
            # ─────────────────────────────────────────────────────────────────
            if st.session_state.opt_results:
                res = st.session_state.opt_results
                
                st.divider()
                st.subheader("📈 Optimized Projection")
                
                # Max Growth Rate - highlighted
                st.metric(
                    "Max Growth Rate",
                    f"{res['rate']:.1f}% /month",
                    delta=f"{res['rate']*12:.0f}% annualized"
                )
                
                # Key metrics
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Units", f"{res['units']:,}")
                with col2:
                    st.metric("Total Revenue", f"${res['revenue']:,.0f}")
                
                # Cash metrics
                if not res['cash_df'].empty:
                    min_cash = res['cash_df']['Cash_Balance'].min()
                    end_cash = res['cash_df'].iloc[-1]['Cash_Balance']
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Min Cash", f"${min_cash:,.0f}")
                    with col2:
                        st.metric("End Cash", f"${end_cash:,.0f}")
                    
                    # Feasibility indicator
                    headroom = min_cash + res['loc_limit']
                    if headroom > 100000:
                        st.success(f"✅ ${headroom:,.0f} credit headroom")
                    elif headroom > 0:
                        st.warning(f"⚠️ ${headroom:,.0f} credit headroom")
                    else:
                        st.error(f"❌ Exceeds limit by ${abs(headroom):,.0f}")

        with c2:
            st.subheader("📊 Projection Results")
            
            if st.session_state.opt_results and not st.session_state.opt_results['cash_df'].empty:
                res = st.session_state.opt_results
                
                # Main chart
                fig = px.area(
                    res['cash_df'], x='Date', y='Cash_Balance',
                    title=f"Cash Flow at {res['rate']:.1f}% Monthly Growth",
                    color_discrete_sequence=['#10B981']
                )
                fig.add_hline(y=-res['loc_limit'], line_dash="dash", line_color="red",
                              annotation_text=f"Credit Limit (-${res['loc_limit']:,.0f})")
                fig.add_hline(y=0, line_dash="dot", line_color="gray")
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
                
            else:
                st.info("👆 Click **Save & Optimize** to calculate maximum sustainable growth rate")
            
            # ─────────────────────────────────────────────────────────────────
            # MANUAL SCENARIO TEST
            # ─────────────────────────────────────────────────────────────────
            st.divider()
            st.subheader("🧪 Manual Scenario Test")
            
            test_rate = st.slider("Test Growth Rate (%/month)", 0.0, 50.0, 10.0, 0.5)
            
            if st.button("📊 Run Scenario"):
                metrics = quick_scenario_metrics(
                    start_units=start_vol,
                    growth_pct=test_rate,
                    start_cash=inv_cash,
                    credit_limit=loc_limit,
                    start_date=sim_start,
                    months=sim_months
                )
                
                # Display results
                col1, col2, col3 = st.columns(3)
                col1.metric("Units Produced", f"{metrics['total_units']:,}")
                col2.metric("Total Revenue", f"${metrics['total_revenue']:,.0f}")
                col3.metric("Min Cash", f"${metrics['min_cash']:,.0f}")
                
                if metrics['is_feasible']:
                    st.success(f"✅ Scenario is FEASIBLE at {test_rate:.1f}% growth")
                else:
                    st.error(f"❌ Scenario EXCEEDS credit limit at {test_rate:.1f}% growth")
                
                # Show chart
                sim_units = simulate_growth_scenario(start_vol, test_rate, sim_start, sim_months)
                _, sim_cash = generate_financials(units_override=sim_units, start_cash_override=inv_cash)
                
                if not sim_cash.empty:
                    fig = px.area(
                        sim_cash, x='Date', y='Cash_Balance',
                        title=f"Scenario: {test_rate:.1f}% Monthly Growth",
                        color_discrete_sequence=['#3B82F6' if metrics['is_feasible'] else '#EF4444']
                    )
                    fig.add_hline(y=-loc_limit, line_dash="dash", line_color="red")
                    fig.add_hline(y=0, line_dash="dot", line_color="gray")
                    st.plotly_chart(fig, use_container_width=True)

    # =========================================================================
    # FINANCIAL STATEMENTS
    # =========================================================================
    elif view == "Financial Statements":
        st.title("Financial Statements")
        
        if not df_pnl.empty:
            col1, col2 = st.columns(2)
            with col1:
                freq = st.radio("Period Aggregation:", ["Monthly", "Quarterly", "Yearly"], horizontal=True, index=1)
            
            freq_map = {"Monthly": "ME", "Quarterly": "QE", "Yearly": "YE"}
            
            st.header("Consolidated Statement of Operations")
            
            pnl_agg = df_pnl.groupby(
                [pd.Grouper(key='Date', freq=freq_map[freq]), 'Type', 'Category']
            ).sum()['Amount'].unstack(level=[1, 2]).fillna(0)
            
            if freq == "Monthly":
                pnl_agg.index = pnl_agg.index.strftime('%Y-%b')
            elif freq == "Quarterly":
                pnl_agg.index = pnl_agg.index.to_period("Q").astype(str)
            else:
                pnl_agg.index = pnl_agg.index.strftime('%Y')
            
            stmt = pd.DataFrame(columns=pnl_agg.index)
            
            def safe_sum(keys):
                total = pd.Series(0.0, index=pnl_agg.index)
                for k in keys:
                    if k in pnl_agg.columns:
                        total += pnl_agg[k]
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
            for col in opex_cols:
                stmt.loc[col[1]] = safe_sum([col])
            
            stmt.loc['Total OpEx'] = safe_sum([('OpEx', c[1]) for c in opex_cols]) + stmt.loc['Salaries & Wages']
            stmt.loc['Net Income'] = stmt.loc['Gross Profit'] + stmt.loc['Total OpEx']
            
            render_financial_statement(stmt, "")
            st.markdown("---")
            
            st.header("Statement of Cash Flows")
            
            cash_agg = df_cash.groupby(
                [pd.Grouper(key='Date', freq=freq_map[freq]), 'Category']
            ).sum()['Amount'].unstack().fillna(0)
            
            if freq == "Monthly":
                cash_agg.index = cash_agg.index.strftime('%Y-%b')
            elif freq == "Quarterly":
                cash_agg.index = cash_agg.index.to_period("Q").astype(str)
            else:
                cash_agg.index = cash_agg.index.strftime('%Y')
            
            cf = pd.DataFrame(columns=cash_agg.index)
            cf.loc['Operating Activities'] = None
            cf.loc['Cash from Customers'] = cash_agg.get('Cash from Customers', pd.Series(0, index=cash_agg.index))
            cf.loc['Supplier Payments'] = cash_agg.get('Supplier Deposits', 0) + cash_agg.get('Supplier Settlements', 0)
            cf.loc['Payroll Paid'] = cash_agg.get('Payroll Paid', pd.Series(0, index=cash_agg.index))
            cf.loc['OpEx Paid'] = cash_agg.get('OpEx Paid', pd.Series(0, index=cash_agg.index))
            
            cf.loc['Net Cash Flow'] = (
                cf.loc['Cash from Customers'] +
                cf.loc['Supplier Payments'] +
                cf.loc['Payroll Paid'] +
                cf.loc['OpEx Paid']
            )
            
            cash_indexed = df_cash.set_index('Date')
            end_bals = cash_indexed.resample(freq_map[freq])['Cash_Balance'].last()
            
            if len(end_bals) == len(cf.columns):
                end_bals.index = cf.columns
                cf.loc['Ending Cash Balance'] = end_bals
            else:
                cf.loc['Ending Cash Balance'] = None
            
            render_financial_statement(cf, "")
        else:
            st.info("No financial data available.")

    # =========================================================================
    # PRODUCTION & SALES
    # =========================================================================
    elif view == "Production & Sales":
        st.title("Production & Sales Mix")
        
        c1, c2 = st.columns([2, 1])
        
        with c1:
            st.subheader("Production Manifest")
            df_units = pd.read_sql("SELECT * FROM production_unit", engine)
            
            edited = st.data_editor(
                df_units.sort_values('build_date'),
                column_config={"id": st.column_config.NumberColumn(disabled=True)},
                hide_index=True,
                height=500,
                use_container_width=True
            )
            
            if st.button("💾 Save Changes"):
                with engine.connect() as conn:
                    try:
                        for _, r in edited.iterrows():
                            conn.execute(
                                text("UPDATE production_unit SET sales_channel=:c, status=:s WHERE id=:i"),
                                {"c": r['sales_channel'], "s": r['status'], "i": r['id']}
                            )
                        conn.commit()
                        st.success("Saved!")
                        st.rerun()
                    except Exception as e:
                        conn.rollback()
                        st.error(f"Save failed: {e}")
        
        with c2:
            st.subheader("Smart Planner")
            start_date = st.date_input("Production Start", value=date(2026, 1, 1))
            
            df_units['Month'] = pd.to_datetime(df_units['build_date']).dt.strftime('%Y-%m')
            existing = df_units.groupby('Month').size()
            
            dates = pd.date_range('2026-01-01', '2027-12-01', freq='MS')
            plan = [
                {"Month": d.date(), "Target": int(existing.get(d.strftime('%Y-%m'), 0))}
                for d in dates
            ]
            
            edit_plan = st.data_editor(pd.DataFrame(plan), hide_index=True, height=400)
            
            if st.button("🚀 Smart Regenerate"):
                with st.spinner("Regenerating schedule..."):
                    try:
                        regenerate_production_schedule(edit_plan, start_date)
                        st.success("Done!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed: {e}")

    # =========================================================================
    # OPEX PLANNING
    # =========================================================================
    elif view == "OpEx Planning":
        st.title("OpEx Budget")
        
        tab1, tab2, tab3 = st.tabs(["A. Headcount & Payroll", "B. R&D Expenses", "C. SG&A Expenses"])
        
        with tab1:
            st.subheader("Headcount Planner")
            
            df_roles = pd.read_sql("SELECT * FROM opex_roles", engine)
            df_staffing = pd.read_sql("SELECT * FROM opex_staffing_plan", engine)
            
            headcount_grid = build_headcount_grid(df_roles, df_staffing)
            
            edited_grid = st.data_editor(
                headcount_grid, 
                use_container_width=True,
                disabled=['role_name'],
                key="headcount_editor"
            )
            
            if st.button("💾 Save Headcount"):
                with engine.connect() as conn:
                    try:
                        role_map = dict(zip(df_roles['role_name'], df_roles['id']))
                        
                        month_cols = [c for c in edited_grid.columns if c != 'role_name']
                        melted = edited_grid.melt(
                            id_vars=['role_name'], 
                            value_vars=month_cols,
                            var_name='Month', 
                            value_name='headcount'
                        )
                        
                        for _, r in melted.iterrows():
                            role_id = role_map.get(r['role_name'])
                            
                            if role_id and pd.notna(r['headcount']):
                                dt = date.fromisoformat(r['Month'] + "-01")
                                conn.execute(
                                    text(get_upsert_sql()), 
                                    {"rid": role_id, "dt": dt, "hc": float(r['headcount'])}
                                )
                        
                        conn.commit()
                        st.success("Headcount saved!")
                        st.rerun()
                    except Exception as e:
                        conn.rollback()
                        st.error(f"Save failed: {e}")
            
            st.divider()
            
            with st.expander("➕ Add/Edit Roles & Salaries", expanded=False):
                st.caption("Add new roles or edit salaries. New roles will appear in the headcount grid after saving.")
                
                edited_roles = st.data_editor(
                    df_roles,
                    column_config={
                        "id": st.column_config.NumberColumn("ID", disabled=True),
                        "role_name": st.column_config.TextColumn("Role Name", required=True),
                        "annual_salary": st.column_config.NumberColumn("Annual Salary ($)", required=True, min_value=0)
                    },
                    hide_index=True,
                    use_container_width=True,
                    num_rows="dynamic",
                    key="roles_editor"
                )
                
                if st.button("💾 Update Roles & Salaries"):
                    with engine.connect() as conn:
                        try:
                            existing_ids = set(df_roles['id'].tolist())
                            edited_ids = set()
                            
                            for _, r in edited_roles.iterrows():
                                if pd.isna(r['role_name']) or str(r['role_name']).strip() == '':
                                    continue
                                    
                                if pd.notna(r.get('id')) and r['id'] in existing_ids:
                                    conn.execute(
                                        text("UPDATE opex_roles SET role_name=:n, annual_salary=:s WHERE id=:id"),
                                        {"n": r['role_name'], "s": r['annual_salary'], "id": r['id']}
                                    )
                                    edited_ids.add(r['id'])
                                else:
                                    conn.execute(
                                        text("INSERT INTO opex_roles (role_name, annual_salary) VALUES (:n, :s)"),
                                        {"n": r['role_name'], "s": r['annual_salary']}
                                    )
                            
                            deleted_ids = existing_ids - edited_ids
                            for del_id in deleted_ids:
                                conn.execute(text("DELETE FROM opex_staffing_plan WHERE role_id = :id"), {"id": del_id})
                                conn.execute(text("DELETE FROM opex_roles WHERE id = :id"), {"id": del_id})
                            
                            conn.commit()
                            st.success("Roles updated!")
                            st.rerun()
                        except Exception as e:
                            conn.rollback()
                            st.error(f"Update failed: {e}")
        
        with tab2:
            render_expense_tab("R&D")
        
        with tab3:
            render_expense_tab("SG&A")

    # =========================================================================
    # BOM & SUPPLY CHAIN
    # =========================================================================
    elif view == "BOM & Supply Chain":
        st.title("Bill of Materials")
        
        df_parts = pd.read_sql("SELECT * FROM part_master", engine)
        
        edited_parts = st.data_editor(
            df_parts,
            disabled=["id", "sku"],
            use_container_width=True
        )
        
        if st.button("💾 Save BOM"):
            with engine.connect() as conn:
                try:
                    for _, r in edited_parts.iterrows():
                        conn.execute(
                            text("""
                                UPDATE part_master 
                                SET name=:n, cost=:c, moq=:m, lead_time=:l, 
                                    deposit_pct=:dp, deposit_days=:dd, balance_days=:bd 
                                WHERE id=:id
                            """),
                            {
                                "n": r['name'], "c": r['cost'], "m": r['moq'],
                                "l": r['lead_time'], "dp": r['deposit_pct'],
                                "dd": r['deposit_days'], "bd": r['balance_days'],
                                "id": r['id']
                            }
                        )
                    conn.commit()
                    st.success("BOM saved!")
                    st.rerun()
                except Exception as e:
                    conn.rollback()
                    st.error(f"Save failed: {e}")


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"System Error: {e}")
        logger.exception("Unhandled exception in main application")