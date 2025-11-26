"""
IdleX CFO Console - Dashboard Application
Version: 3.5 (Push to Plan + Mobile Responsive)
Includes: Scenario → Production Push, Mobile UI, GAAP Financials
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
st.set_page_config(
    page_title="IdleX CFO Console", 
    layout="wide",
    initial_sidebar_state="collapsed"  # Better for mobile
)

# Mobile-responsive CSS
st.markdown("""
<style>
    /* Base styles */
    .financial-table {
        font-family: 'Georgia', serif;
        font-size: 14px;
        border-collapse: collapse;
        width: 100%;
        color: #000000 !important;
        background-color: #ffffff;
        margin-bottom: 20px;
        overflow-x: auto;
        display: block;
    }
    .financial-table th {
        text-align: right;
        border-bottom: 2px solid #000;
        padding: 8px;
        font-weight: bold;
        color: #000000 !important;
        white-space: nowrap;
    }
    .financial-table td {
        padding: 6px 8px;
        border: none;
        color: #000000 !important;
        white-space: nowrap;
    }
    .financial-table tr:hover {
        background-color: #f5f5f5;
    }
    .financial-table .row-header {
        text-align: left;
        min-width: 150px;
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
    
    /* Result cards */
    .result-card {
        background: linear-gradient(135deg, #1e3a5f 0%, #0d1b2a 100%);
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
        border-left: 4px solid #10B981;
    }
    .result-card.pending {
        border-left-color: #F59E0B;
        background: linear-gradient(135deg, #3d2e0a 0%, #1a1a0a 100%);
    }
    
    /* Push to plan button */
    .push-button {
        background: linear-gradient(135deg, #10B981 0%, #059669 100%);
        color: white;
        font-weight: bold;
        padding: 12px 24px;
        border-radius: 8px;
        border: none;
        width: 100%;
    }
    
    /* Mobile responsive */
    @media (max-width: 768px) {
        .block-container {
            padding: 1rem 0.5rem !important;
        }
        
        .stMetric {
            padding: 0.5rem !important;
        }
        
        .stMetric label {
            font-size: 0.75rem !important;
        }
        
        .stMetric [data-testid="stMetricValue"] {
            font-size: 1.2rem !important;
        }
        
        h1 {
            font-size: 1.5rem !important;
        }
        
        h2 {
            font-size: 1.25rem !important;
        }
        
        h3 {
            font-size: 1.1rem !important;
        }
        
        .financial-table {
            font-size: 12px;
        }
        
        .financial-table th, .financial-table td {
            padding: 4px 6px;
        }
        
        /* Stack columns on mobile */
        [data-testid="column"] {
            width: 100% !important;
            flex: 1 1 100% !important;
        }
        
        /* Reduce chart height on mobile */
        .js-plotly-plot {
            height: 250px !important;
        }
        
        /* Better touch targets */
        .stButton button {
            min-height: 48px;
            font-size: 1rem;
        }
        
        .stNumberInput input {
            font-size: 16px !important; /* Prevents zoom on iOS */
        }
        
        /* Scrollable data editors */
        [data-testid="stDataEditor"] {
            max-height: 300px;
            overflow: auto;
        }
    }
    
    /* Tablet */
    @media (min-width: 769px) and (max-width: 1024px) {
        .block-container {
            padding: 1rem 1rem !important;
        }
    }
    
    /* Hide hamburger text on mobile */
    @media (max-width: 768px) {
        .css-1d391kg {
            padding-top: 1rem;
        }
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
    if DB_TYPE == "postgresql":
        return f"TO_CHAR({column}, 'YYYY-MM')"
    return f"strftime('%Y-%m', {column})"


def get_upsert_sql() -> str:
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
    num_days = calendar.monthrange(year, month)[1]
    days = [date(year, month, d) for d in range(1, num_days + 1)]
    valid_days = [d for d in days if d.weekday() < 5]
    if start_threshold:
        valid_days = [d for d in valid_days if d >= start_threshold]
    return valid_days


def format_banker(val) -> str:
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


def format_compact(val: float) -> str:
    """Format large numbers compactly for mobile."""
    if val >= 1_000_000:
        return f"${val/1_000_000:.1f}M"
    elif val >= 1_000:
        return f"${val/1_000:.0f}K"
    else:
        return f"${val:,.0f}"


def render_financial_statement(df: pd.DataFrame, title: str) -> None:
    html = f"<h3>{title}</h3><div style='overflow-x:auto;'><table class='financial-table'>"
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

    df_units['build_date'] = pd.to_datetime(df_units['build_date'])
    df_opex['month_date'] = pd.to_datetime(df_opex['month_date'])

    ledger = []

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
            
            if part['deposit_pct'] > 0:
                ledger.append({"Date": delivery + timedelta(days=int(part['deposit_days'])), "Category": "Supplier Deposits", "Type": "Operations", "Amount": -(total_po_cost * part['deposit_pct']), "Report": "Cash"})
            
            if part['deposit_pct'] < 1.0:
                ledger.append({"Date": delivery + timedelta(days=int(part['balance_days'])), "Category": "Supplier Settlements", "Type": "Operations", "Amount": -(total_po_cost * (1 - part['deposit_pct'])), "Report": "Cash"})

    if not df_opex.empty and not df_roles.empty:
        opex_merged = pd.merge(df_opex, df_roles, left_on='role_id', right_on='id')
        
        for _, row in opex_merged.iterrows():
            monthly_cost = (row['annual_salary'] / 12) * row['headcount']
            
            if monthly_cost > 0:
                is_direct_labor = "Assembler" in row['role_name']
                category = "Direct Labor" if is_direct_labor else "Salaries & Wages"
                pnl_type = "COGS" if is_direct_labor else "OpEx"
                
                ledger.append({"Date": row['month_date'], "Category": category, "Type": pnl_type, "Amount": -monthly_cost, "Report": "PnL"})
                ledger.append({"Date": row['month_date'], "Category": "Payroll Paid", "Type": "Operations", "Amount": -monthly_cost, "Report": "Cash"})

    if not df_gen_exp.empty:
        for _, row in df_gen_exp.iterrows():
            if row['amount'] > 0:
                ledger.append({"Date": row['month_date'], "Category": row['category'], "Type": "OpEx", "Amount": -row['amount'], "Report": "PnL"})
                ledger.append({"Date": row['month_date'], "Category": "OpEx Paid", "Type": "Operations", "Amount": -row['amount'], "Report": "Cash"})

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
    """Generate simulated production units for growth scenario analysis."""
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
                    "status": "PLANNED"
                })
                serial_counter += 1
        
        current_units = current_units * (1 + (growth_pct / 100))
        
        if current_date.month == 12:
            current_date = date(current_date.year + 1, 1, 1)
        else:
            current_date = date(current_date.year, current_date.month + 1, 1)
    
    return pd.DataFrame(sim_units)


def find_max_growth_rate(start_units: int, start_cash: float, credit_limit: float, 
                         start_date: date, months: int) -> dict:
    """Binary search to find maximum sustainable growth rate."""
    best_rate = 0.0
    best_cash_df = pd.DataFrame()
    best_pnl_df = pd.DataFrame()
    best_units_df = pd.DataFrame()
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
            best_units_df = sim_units
            low = mid
        else:
            high = mid
    
    total_units = len(best_units_df)
    total_revenue = best_pnl_df[best_pnl_df['Category'] == 'Sales of Goods']['Amount'].sum() if not best_pnl_df.empty else 0
    
    # Build monthly summary for display
    if not best_units_df.empty:
        best_units_df['build_date'] = pd.to_datetime(best_units_df['build_date'])
        monthly_summary = best_units_df.groupby(best_units_df['build_date'].dt.to_period('M')).size().reset_index()
        monthly_summary.columns = ['Month', 'Units']
        monthly_summary['Month'] = monthly_summary['Month'].astype(str)
    else:
        monthly_summary = pd.DataFrame(columns=['Month', 'Units'])
    
    return {
        'rate': best_rate,
        'cash_df': best_cash_df,
        'pnl_df': best_pnl_df,
        'units_df': best_units_df,
        'total_units': total_units,
        'total_revenue': total_revenue,
        'monthly_summary': monthly_summary,
        'start_cash': start_cash,
        'credit_limit': credit_limit
    }


def push_scenario_to_production(units_df: pd.DataFrame) -> tuple:
    """
    Push simulated units to actual production plan.
    
    Returns:
        Tuple of (success: bool, message: str, count: int)
    """
    if units_df.empty:
        return False, "No units to push", 0
    
    with engine.connect() as conn:
        try:
            # Delete existing PLANNED units
            result = conn.execute(text("DELETE FROM production_unit WHERE status = 'PLANNED'"))
            deleted_count = result.rowcount
            
            # Get next serial number
            last_sn = conn.execute(
                text("SELECT serial_number FROM production_unit ORDER BY id DESC LIMIT 1")
            ).scalar()
            next_sn = parse_serial_number(last_sn) + 1
            
            # Insert new planned units
            inserted_count = 0
            for _, row in units_df.iterrows():
                build_date = row['build_date']
                if isinstance(build_date, pd.Timestamp):
                    build_date = build_date.date()
                
                conn.execute(
                    text("""
                        INSERT INTO production_unit (serial_number, build_date, sales_channel, status) 
                        VALUES (:sn, :bd, :ch, 'PLANNED')
                    """),
                    {
                        "sn": f"IDX-{next_sn:04d}",
                        "bd": build_date,
                        "ch": row['sales_channel']
                    }
                )
                next_sn += 1
                inserted_count += 1
            
            conn.commit()
            
            return True, f"Replaced {deleted_count} existing planned units with {inserted_count} new units", inserted_count
            
        except Exception as e:
            conn.rollback()
            logger.error(f"Push to production failed: {e}")
            return False, str(e), 0


# =============================================================================
# PRODUCTION SCHEDULER
# =============================================================================
def regenerate_production_schedule(edit_plan: pd.DataFrame, start_date: date) -> None:
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
# HELPER COMPONENTS
# =============================================================================
def render_expense_tab(expense_type: str) -> None:
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
    
    edited_exp = st.data_editor(pivot_exp, use_container_width=True, num_rows="dynamic", key=f"expense_editor_{expense_type}")
    
    if st.button(f"💾 Save {expense_type}", key=f"save_btn_{expense_type}"):
        with engine.connect() as conn:
            try:
                conn.execute(text("DELETE FROM opex_general_expenses WHERE expense_type = :et"), {"et": expense_type})
                melted = edited_exp.melt(id_vars=['category'], var_name='Month', value_name='amount')
                for _, r in melted.iterrows():
                    if pd.notna(r['amount']) and pd.notna(r['category']):
                        dt = date.fromisoformat(r['Month'] + "-01")
                        conn.execute(text("INSERT INTO opex_general_expenses (category, expense_type, month_date, amount) VALUES (:c, :t, :d, :a)"),
                                     {"c": r['category'], "t": expense_type, "d": dt, "a": r['amount']})
                conn.commit()
                st.success(f"{expense_type} saved!")
                st.rerun()
            except Exception as e:
                conn.rollback()
                st.error(f"Save failed: {e}")


def build_headcount_grid(df_roles: pd.DataFrame, df_staffing: pd.DataFrame) -> pd.DataFrame:
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
    st.sidebar.title("IdleX CFO")
    
    if st.sidebar.button("⚠️ Reset DB"):
        with st.spinner("Resetting..."):
            seed_db.run_seed()
        st.sidebar.success("Done!")
        st.rerun()

    view = st.sidebar.radio(
        "Navigate",
        ["📊 Dashboard", "🚀 Scenario Planner", "📈 Financials", "�icing Production", "💰 OpEx", "🔧 BOM"]
    )

    # Generate financials for most views
    df_pnl, df_cash = pd.DataFrame(), pd.DataFrame()
    if view not in ["🚀 Scenario Planner"]:
        df_pnl, df_cash = generate_financials()

    # =========================================================================
    # EXECUTIVE DASHBOARD
    # =========================================================================
    if view == "📊 Dashboard":
        st.title("Executive Dashboard")
        
        if not df_pnl.empty:
            years = sorted(df_pnl['Date'].dt.year.unique().tolist())
            selected_period = st.selectbox("Period:", ["All Time"] + years, key="dash_period")
            
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
            
            # Responsive metrics - 2x2 grid on mobile
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Revenue", format_compact(rev))
                st.metric("Min Cash", format_compact(min_cash))
            with col2:
                st.metric("Gross Margin", format_compact(margin))
                st.metric("End Cash", format_compact(end_cash))
            
            fig = px.area(cash_view, x='Date', y='Cash_Balance', title="Cash Forecast", color_discrete_sequence=['#10B981'])
            fig.add_hline(y=0, line_dash="dash", line_color="red")
            fig.update_layout(height=300, margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data. Click Reset DB in sidebar.")

    # =========================================================================
    # SCENARIO PLANNER (with Push to Plan)
    # =========================================================================
    elif view == "🚀 Scenario Planner":
        st.title("Growth Scenario Planner")
        
        # Load config
        try:
            config = pd.read_sql("SELECT * FROM global_config", engine)
            cash_row = config[config['setting_key'] == 'start_cash']
            loc_row = config[config['setting_key'] == 'loc_limit']
            def_cash = float(cash_row['setting_value'].values[0]) if not cash_row.empty else 1000000.0
            def_loc = float(loc_row['setting_value'].values[0]) if not loc_row.empty else 500000.0
        except Exception:
            def_cash = 1000000.0
            def_loc = 500000.0
        
        # Initialize session state
        if 'scenario_results' not in st.session_state:
            st.session_state.scenario_results = None
        if 'push_confirmed' not in st.session_state:
            st.session_state.push_confirmed = False
        
        # ─────────────────────────────────────────────────────────────────────
        # STEP 1: CONSTRAINTS INPUT
        # ─────────────────────────────────────────────────────────────────────
        st.subheader("① Set Constraints")
        
        # Responsive layout - stack on mobile
        col1, col2 = st.columns(2)
        with col1:
            inv_cash = st.number_input("Investor Equity ($)", value=def_cash, step=100000.0, format="%.0f")
            start_vol = st.number_input("Starting Units/Month", value=50, min_value=1)
        with col2:
            loc_limit = st.number_input("Credit Limit ($)", value=def_loc, step=100000.0, format="%.0f")
            sim_months = st.slider("Horizon (Months)", 12, 60, 36)
        
        sim_start = st.date_input("Start Date", value=date(2026, 1, 1))
        
        # OPTIMIZE BUTTON
        if st.button("🔍 Save & Find Max Growth", type="primary", use_container_width=True):
            # Save to DB
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
            with st.spinner("Finding optimal growth rate..."):
                results = find_max_growth_rate(
                    start_units=start_vol,
                    start_cash=inv_cash,
                    credit_limit=loc_limit,
                    start_date=sim_start,
                    months=sim_months
                )
                st.session_state.scenario_results = results
                st.session_state.push_confirmed = False
            
            st.rerun()
        
        # ─────────────────────────────────────────────────────────────────────
        # STEP 2: RESULTS DISPLAY
        # ─────────────────────────────────────────────────────────────────────
        if st.session_state.scenario_results:
            res = st.session_state.scenario_results
            
            st.divider()
            st.subheader("② Review Results")
            
            # Key metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📈 Max Growth", f"{res['rate']:.1f}%/mo")
            with col2:
                st.metric("🏭 Total Units", f"{res['total_units']:,}")
            with col3:
                st.metric("💵 Revenue", format_compact(res['total_revenue']))
            
            # Cash metrics
            if not res['cash_df'].empty:
                min_cash = res['cash_df']['Cash_Balance'].min()
                end_cash = res['cash_df'].iloc[-1]['Cash_Balance']
                headroom = min_cash + res['credit_limit']
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Min Cash", format_compact(min_cash))
                with col2:
                    st.metric("End Cash", format_compact(end_cash))
                with col3:
                    if headroom > 100000:
                        st.metric("Headroom", format_compact(headroom), delta="OK")
                    else:
                        st.metric("Headroom", format_compact(headroom), delta="Tight", delta_color="inverse")
                
                # Cash chart
                fig = px.area(res['cash_df'], x='Date', y='Cash_Balance', 
                              title=f"Cash Flow @ {res['rate']:.1f}%/mo Growth",
                              color_discrete_sequence=['#10B981'])
                fig.add_hline(y=-res['credit_limit'], line_dash="dash", line_color="red", 
                              annotation_text="Credit Limit")
                fig.add_hline(y=0, line_dash="dot", line_color="gray")
                fig.update_layout(height=300, margin=dict(l=0, r=0, t=40, b=0))
                st.plotly_chart(fig, use_container_width=True)
                
                # Monthly breakdown (collapsible)
                with st.expander("📅 Monthly Production Plan"):
                    st.dataframe(res['monthly_summary'], use_container_width=True, hide_index=True)
            
            # ─────────────────────────────────────────────────────────────────
            # STEP 3: PUSH TO PLAN
            # ─────────────────────────────────────────────────────────────────
            st.divider()
            st.subheader("③ Push to Production Plan")
            
            if not res['units_df'].empty:
                unit_count = len(res['units_df'])
                
                # Warning box
                st.warning(f"""
                    **Ready to push {unit_count:,} units to production plan?**
                    
                    This will:
                    - ❌ Delete all existing PLANNED units
                    - ✅ Insert {unit_count:,} new PLANNED units
                    - 📊 Update the Executive Dashboard
                """)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("🚀 Push to Plan", type="primary", use_container_width=True):
                        success, message, count = push_scenario_to_production(res['units_df'])
                        
                        if success:
                            st.session_state.push_confirmed = True
                            st.success(f"✅ {message}")
                            st.balloons()
                        else:
                            st.error(f"❌ Failed: {message}")
                
                with col2:
                    if st.button("🔄 Try Different Scenario", use_container_width=True):
                        st.session_state.scenario_results = None
                        st.session_state.push_confirmed = False
                        st.rerun()
                
                if st.session_state.push_confirmed:
                    st.success("✅ Production plan updated! View in 🏭 Production tab.")
        
        else:
            st.info("👆 Enter constraints and click **Save & Find Max Growth** to start.")

    # =========================================================================
    # FINANCIAL STATEMENTS
    # =========================================================================
    elif view == "📈 Financials":
        st.title("Financial Statements")
        
        if not df_pnl.empty:
            freq = st.radio("Period:", ["Monthly", "Quarterly", "Yearly"], horizontal=True, index=1)
            freq_map = {"Monthly": "ME", "Quarterly": "QE", "Yearly": "YE"}
            
            st.header("Income Statement")
            
            pnl_agg = df_pnl.groupby([pd.Grouper(key='Date', freq=freq_map[freq]), 'Type', 'Category']).sum()['Amount'].unstack(level=[1, 2]).fillna(0)
            
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
            
            st.divider()
            st.header("Cash Flow Statement")
            
            cash_agg = df_cash.groupby([pd.Grouper(key='Date', freq=freq_map[freq]), 'Category']).sum()['Amount'].unstack().fillna(0)
            
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
            cf.loc['Net Cash Flow'] = cf.loc['Cash from Customers'] + cf.loc['Supplier Payments'] + cf.loc['Payroll Paid'] + cf.loc['OpEx Paid']
            
            cash_indexed = df_cash.set_index('Date')
            end_bals = cash_indexed.resample(freq_map[freq])['Cash_Balance'].last()
            
            if len(end_bals) == len(cf.columns):
                end_bals.index = cf.columns
                cf.loc['Ending Cash Balance'] = end_bals
            else:
                cf.loc['Ending Cash Balance'] = None
            
            render_financial_statement(cf, "")
        else:
            st.info("No data available.")

    # =========================================================================
    # PRODUCTION & SALES
    # =========================================================================
    elif view == "🏭 Production":
        st.title("Production Plan")
        
        df_units = pd.read_sql("SELECT * FROM production_unit", engine)
        
        # Summary stats
        if not df_units.empty:
            total = len(df_units)
            planned = len(df_units[df_units['status'] == 'PLANNED'])
            other = total - planned
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Units", total)
            col2.metric("Planned", planned)
            col3.metric("WIP/Complete", other)
        
        st.subheader("Production Manifest")
        
        edited = st.data_editor(
            df_units.sort_values('build_date'),
            column_config={"id": st.column_config.NumberColumn(disabled=True)},
            hide_index=True,
            use_container_width=True
        )
        
        if st.button("💾 Save Changes", use_container_width=True):
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
                    st.error(f"Failed: {e}")
        
        # Manual planner
        with st.expander("📅 Manual Schedule Builder"):
            start_date = st.date_input("Start", value=date(2026, 1, 1))
            
            df_units['Month'] = pd.to_datetime(df_units['build_date']).dt.strftime('%Y-%m')
            existing = df_units.groupby('Month').size()
            
            dates = pd.date_range('2026-01-01', '2027-12-01', freq='MS')
            plan = [{"Month": d.date(), "Target": int(existing.get(d.strftime('%Y-%m'), 0))} for d in dates]
            
            edit_plan = st.data_editor(pd.DataFrame(plan), hide_index=True)
            
            if st.button("🚀 Regenerate Schedule"):
                with st.spinner("Building..."):
                    try:
                        regenerate_production_schedule(edit_plan, start_date)
                        st.success("Done!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed: {e}")

    # =========================================================================
    # OPEX PLANNING
    # =========================================================================
    elif view == "💰 OpEx":
        st.title("OpEx Budget")
        
        tab1, tab2, tab3 = st.tabs(["Headcount", "R&D", "SG&A"])
        
        with tab1:
            df_roles = pd.read_sql("SELECT * FROM opex_roles", engine)
            df_staffing = pd.read_sql("SELECT * FROM opex_staffing_plan", engine)
            
            headcount_grid = build_headcount_grid(df_roles, df_staffing)
            
            edited_grid = st.data_editor(headcount_grid, use_container_width=True, disabled=['role_name'])
            
            if st.button("💾 Save Headcount", use_container_width=True):
                with engine.connect() as conn:
                    try:
                        role_map = dict(zip(df_roles['role_name'], df_roles['id']))
                        month_cols = [c for c in edited_grid.columns if c != 'role_name']
                        melted = edited_grid.melt(id_vars=['role_name'], value_vars=month_cols, var_name='Month', value_name='headcount')
                        
                        for _, r in melted.iterrows():
                            role_id = role_map.get(r['role_name'])
                            if role_id and pd.notna(r['headcount']):
                                dt = date.fromisoformat(r['Month'] + "-01")
                                conn.execute(text(get_upsert_sql()), {"rid": role_id, "dt": dt, "hc": float(r['headcount'])})
                        
                        conn.commit()
                        st.success("Saved!")
                        st.rerun()
                    except Exception as e:
                        conn.rollback()
                        st.error(f"Failed: {e}")
            
            with st.expander("➕ Manage Roles"):
                edited_roles = st.data_editor(df_roles, column_config={"id": st.column_config.NumberColumn(disabled=True)},
                                              hide_index=True, num_rows="dynamic")
                
                if st.button("💾 Update Roles"):
                    with engine.connect() as conn:
                        try:
                            existing_ids = set(df_roles['id'].tolist())
                            edited_ids = set()
                            
                            for _, r in edited_roles.iterrows():
                                if pd.isna(r['role_name']) or str(r['role_name']).strip() == '':
                                    continue
                                if pd.notna(r.get('id')) and r['id'] in existing_ids:
                                    conn.execute(text("UPDATE opex_roles SET role_name=:n, annual_salary=:s WHERE id=:id"),
                                                 {"n": r['role_name'], "s": r['annual_salary'], "id": r['id']})
                                    edited_ids.add(r['id'])
                                else:
                                    conn.execute(text("INSERT INTO opex_roles (role_name, annual_salary) VALUES (:n, :s)"),
                                                 {"n": r['role_name'], "s": r['annual_salary']})
                            
                            for del_id in (existing_ids - edited_ids):
                                conn.execute(text("DELETE FROM opex_staffing_plan WHERE role_id = :id"), {"id": del_id})
                                conn.execute(text("DELETE FROM opex_roles WHERE id = :id"), {"id": del_id})
                            
                            conn.commit()
                            st.success("Updated!")
                            st.rerun()
                        except Exception as e:
                            conn.rollback()
                            st.error(f"Failed: {e}")
        
        with tab2:
            render_expense_tab("R&D")
        
        with tab3:
            render_expense_tab("SG&A")

    # =========================================================================
    # BOM
    # =========================================================================
    elif view == "🔧 BOM":
        st.title("Bill of Materials")
        
        df_parts = pd.read_sql("SELECT * FROM part_master", engine)
        
        edited_parts = st.data_editor(df_parts, disabled=["id", "sku"], use_container_width=True)
        
        if st.button("💾 Save BOM", use_container_width=True):
            with engine.connect() as conn:
                try:
                    for _, r in edited_parts.iterrows():
                        conn.execute(text("""UPDATE part_master SET name=:n, cost=:c, moq=:m, lead_time=:l, 
                                    deposit_pct=:dp, deposit_days=:dd, balance_days=:bd WHERE id=:id"""),
                                     {"n": r['name'], "c": r['cost'], "m": r['moq'], "l": r['lead_time'],
                                      "dp": r['deposit_pct'], "dd": r['deposit_days'], "bd": r['balance_days'], "id": r['id']})
                    conn.commit()
                    st.success("Saved!")
                    st.rerun()
                except Exception as e:
                    conn.rollback()
                    st.error(f"Failed: {e}")


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Error: {e}")
        logger.exception("Unhandled exception")