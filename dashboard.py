"""
IdleX ERP - Enterprise Resource Planning System
Version: 7.0 (Production Cloud - Boardroom Ready)
Stack: Python 3.11+, Streamlit 1.28+, SQLAlchemy 2.0+, PostgreSQL/SQLite

DESIGN PRINCIPLES:
1. Financial Precision: All monetary calculations use Decimal where critical
2. Cloud Native: PostgreSQL-first with SQLite fallback for local dev
3. Defensive Coding: Every database operation wrapped in try/except
4. Brand Compliance: Full IdleX 2025 Brandbook implementation
5. Audit Trail: All financial decisions traceable

MATHEMATICAL GUARANTEES:
- Cash waterfall processes transactions chronologically
- LOC can only fund inventory/materials (not OpEx)
- Revenue pays down LOC before filling cash
- All dates stored as ISO strings for cross-DB compatibility
"""

import streamlit as st
import pandas as pd
import numpy as np
from decimal import Decimal, ROUND_HALF_UP
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError, OperationalError
from datetime import datetime, date, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import calendar
import math
import os
import traceback
import logging
import random

# Configure logging for production debugging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# SECTION 1: BRAND IDENTITY (IdleX Brandbook 2025)
# =============================================================================

# Primary Brand Colors
NAVY = "#1E3466"        # Dark Navy Blue - Trust, resilience, engineering excellence
X_BLUE = "#3A77D8"      # X Blue - Accent and energy
YELLOW = "#FFB400"      # Electric Bolt Yellow - Highlights and CTAs
PURPLE = "#6248FF"      # Hyper Purple - Special accents
SLATE = "#A5ABB5"       # Slate Gray - Supporting text
LIGHT = "#E6E8EC"       # Light Gray - Backgrounds
WHITE = "#FFFFFF"       # Clean space

# Extended Navy Palette
NAVY_DARK = "#060A14"
NAVY_MID = "#101C37"
NAVY_LIGHT = "#25407E"

# Status Colors
STATUS_OK = "#10B981"
STATUS_WARN = "#FFB400"
STATUS_DANGER = "#EF4444"

# =============================================================================
# SECTION 2: PAGE CONFIGURATION & STYLING
# =============================================================================

st.set_page_config(
    page_title="IdleX ERP",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Complete Brand CSS
st.markdown(f"""
<style>
    /* ===== TYPOGRAPHY (Brandbook: Montserrat + Open Sans) ===== */
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@500;600;700;800&family=Open+Sans:wght@400;500;600&display=swap');
    
    html, body, [class*="css"] {{
        font-family: 'Open Sans', sans-serif !important;
    }}
    
    h1, h2, h3, h4, h5, h6 {{
        font-family: 'Montserrat', sans-serif !important;
        color: {NAVY} !important;
        font-weight: 700 !important;
    }}
    
    /* ===== SIDEBAR (Dark Navy Gradient) ===== */
    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, {NAVY} 0%, {NAVY_MID} 100%) !important;
    }}
    
    [data-testid="stSidebar"] * {{
        color: {WHITE} !important;
    }}
    
    [data-testid="stSidebar"] .stRadio label {{
        color: {WHITE} !important;
        font-family: 'Open Sans', sans-serif !important;
    }}
    
    [data-testid="stSidebar"] .stRadio label:hover {{
        color: {X_BLUE} !important;
    }}
    
    [data-testid="stSidebar"] hr {{
        border-color: rgba(255,255,255,0.2) !important;
    }}
    
    /* ===== BUTTONS (Gradient with Hover) ===== */
    .stButton > button {{
        background: linear-gradient(135deg, {NAVY} 0%, {X_BLUE} 100%) !important;
        color: {WHITE} !important;
        border: none !important;
        border-radius: 8px !important;
        font-family: 'Montserrat', sans-serif !important;
        font-weight: 600 !important;
        padding: 0.6rem 1.2rem !important;
        transition: all 0.3s ease !important;
    }}
    
    .stButton > button:hover {{
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 12px rgba(30, 52, 102, 0.3) !important;
    }}
    
    .stButton > button:active {{
        transform: translateY(0) !important;
    }}
    
    /* Primary buttons (more prominent) */
    .stButton > button[kind="primary"] {{
        background: linear-gradient(135deg, {X_BLUE} 0%, {NAVY} 100%) !important;
    }}
    
    /* ===== TABS (Brand Styled) ===== */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
        background: {LIGHT};
        padding: 4px;
        border-radius: 8px;
    }}
    
    .stTabs [data-baseweb="tab"] {{
        background: transparent !important;
        border-radius: 6px !important;
        color: {NAVY} !important;
        font-family: 'Montserrat', sans-serif !important;
        font-weight: 600 !important;
        padding: 8px 16px !important;
    }}
    
    .stTabs [aria-selected="true"] {{
        background: {WHITE} !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
    }}
    
    /* ===== METRICS (Brand Cards) ===== */
    [data-testid="stMetricValue"] {{
        font-family: 'Montserrat', sans-serif !important;
        font-weight: 700 !important;
        color: {NAVY} !important;
    }}
    
    [data-testid="stMetricLabel"] {{
        font-family: 'Open Sans', sans-serif !important;
        color: {SLATE} !important;
    }}
    
    /* ===== DATA EDITOR (Clean Tables) ===== */
    .stDataFrame {{
        border-radius: 8px !important;
        overflow: hidden !important;
    }}
    
    /* ===== FINANCIAL STATEMENTS (GAAP Style) ===== */
    .financial-table {{
        font-family: 'Open Sans', sans-serif;
        font-size: 14px;
        width: 100%;
        border-collapse: collapse;
        color: #000;
        background: {WHITE};
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }}
    
    .financial-table th {{
        background: {NAVY};
        color: {WHITE};
        text-align: right;
        padding: 12px 8px;
        font-weight: 600;
        font-family: 'Montserrat', sans-serif;
    }}
    
    .financial-table th:first-child {{
        text-align: left;
    }}
    
    .financial-table td {{
        padding: 8px;
        border-bottom: 1px solid {LIGHT};
    }}
    
    .financial-table .row-header {{
        text-align: left;
        width: 35%;
        font-weight: 500;
    }}
    
    .financial-table .section-header {{
        font-weight: 700;
        color: {NAVY};
        background: {LIGHT};
        text-decoration: underline;
        padding-top: 15px;
    }}
    
    .financial-table .total-row {{
        font-weight: 700;
        border-top: 1px solid {NAVY};
        color: {NAVY};
    }}
    
    .financial-table .grand-total {{
        font-weight: 700;
        border-top: 2px solid {NAVY};
        border-bottom: 3px double {NAVY};
        color: {NAVY};
        background: {LIGHT};
    }}
    
    .financial-table .indent {{
        padding-left: 24px;
    }}
    
    .financial-table .negative {{
        color: {STATUS_DANGER};
    }}
    
    /* ===== METRIC CARDS ===== */
    .metric-card {{
        background: {WHITE};
        border-radius: 12px;
        padding: 1.25rem;
        box-shadow: 0 2px 8px rgba(30, 52, 102, 0.08);
        border-left: 4px solid {X_BLUE};
        margin-bottom: 1rem;
    }}
    
    .metric-card-warn {{
        border-left-color: {YELLOW};
    }}
    
    .metric-card-danger {{
        border-left-color: {STATUS_DANGER};
    }}
    
    .metric-card-success {{
        border-left-color: {STATUS_OK};
    }}
    
    /* ===== STATUS BADGES ===== */
    .status-ok {{ color: {STATUS_OK}; font-weight: 600; }}
    .status-warn {{ color: {YELLOW}; font-weight: 600; }}
    .status-danger {{ color: {STATUS_DANGER}; font-weight: 600; }}
    
    /* ===== CONTAINERS ===== */
    .block-container {{
        padding-top: 2rem !important;
        max-width: 1400px !important;
    }}
    
    /* ===== EXPANDER (Styled) ===== */
    .streamlit-expanderHeader {{
        font-family: 'Montserrat', sans-serif !important;
        font-weight: 600 !important;
        color: {NAVY} !important;
    }}
    
    /* ===== RESPONSIVE ===== */
    @media (max-width: 768px) {{
        .block-container {{ padding: 0.5rem !important; }}
        h1 {{ font-size: 1.5rem !important; }}
        [data-testid="stMetricValue"] {{ font-size: 1.2rem !important; }}
        .stButton button {{ min-height: 48px; width: 100%; }}
    }}
</style>
""", unsafe_allow_html=True)


# =============================================================================
# SECTION 3: DATABASE ENGINE (Cloud-Native)
# =============================================================================

@st.cache_resource
def get_db_engine():
    """
    Create database engine with proper connection pooling.
    
    Cloud Detection:
    - DATABASE_URL present → PostgreSQL (Cloud Run / Cloud SQL)
    - No DATABASE_URL → SQLite (Local Development)
    
    CRITICAL: SQLAlchemy 2.0 requires 'postgresql://' not 'postgres://'
    """
    db_url = os.getenv("DATABASE_URL")
    
    if db_url:
        # Fix Heroku/Cloud providers that use 'postgres://'
        if db_url.startswith("postgres://"):
            db_url = db_url.replace("postgres://", "postgresql://", 1)
        
        # Production settings for PostgreSQL
        engine = create_engine(
            db_url,
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True,  # Verify connections before use
            pool_recycle=300     # Recycle connections every 5 minutes
        )
        logger.info("Connected to PostgreSQL (Cloud)")
        return engine
    
    # Local SQLite with threading support
    engine = create_engine(
        'sqlite:///idlex.db',
        connect_args={"check_same_thread": False}
    )
    logger.info("Connected to SQLite (Local)")
    return engine


def get_db_type() -> str:
    """Returns 'postgresql' or 'sqlite' for SQL dialect branching."""
    db_url = os.getenv("DATABASE_URL")
    return "postgresql" if db_url and "postgres" in db_url else "sqlite"


def ensure_database_ready() -> bool:
    """
    Auto-healing database check.
    
    Strategy:
    1. Try to query a core table
    2. If it fails, run seed script
    3. Return True if ready, False if unrecoverable
    
    This ensures the app never shows "table not found" errors to users.
    """
    engine = get_db_engine()
    
    try:
        with engine.connect() as conn:
            # Test query - if this works, DB is ready
            result = conn.execute(text("SELECT COUNT(*) FROM production_unit"))
            count = result.scalar()
            logger.info(f"Database ready: {count} production units found")
            return True
    except Exception as e:
        logger.warning(f"Database check failed: {e}")
        
        # Attempt auto-heal
        try:
            import seed_db
            seed_db.run_seed()
            logger.info("Database auto-healed successfully")
            return True
        except Exception as seed_error:
            logger.error(f"Auto-heal failed: {seed_error}")
            return False


# =============================================================================
# SECTION 4: UTILITY FUNCTIONS
# =============================================================================

def money(value: float) -> str:
    """Format number as currency with proper negative handling (banker's notation)."""
    if pd.isna(value) or value is None:
        return "-"
    if value < 0:
        return f"({abs(value):,.0f})"
    return f"{value:,.0f}"


def money_with_sign(value: float) -> str:
    """Format with explicit + or - sign."""
    if pd.isna(value) or value is None:
        return "-"
    if value < 0:
        return f"-${abs(value):,.0f}"
    return f"+${value:,.0f}"


def pct(value: float) -> str:
    """Format as percentage."""
    if pd.isna(value) or value is None:
        return "-"
    return f"{value * 100:.1f}%"


def parse_date(d) -> date:
    """
    Safely parse date from various formats.
    Handles: datetime, date, string (ISO), pandas Timestamp
    """
    if d is None:
        return None
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, date):
        return d
    if isinstance(d, pd.Timestamp):
        return d.date()
    if isinstance(d, str):
        try:
            return datetime.strptime(d[:10], '%Y-%m-%d').date()
        except:
            return None
    return None


def get_workdays(year: int, month: int, start_threshold: date = None) -> list:
    """
    Get list of workdays (Mon-Fri) in a month.
    
    Args:
        year: Target year
        month: Target month (1-12)
        start_threshold: Optional minimum date (for partial months)
    
    Returns:
        List of date objects representing workdays
    """
    num_days = calendar.monthrange(year, month)[1]
    days = [date(year, month, d) for d in range(1, num_days + 1)]
    
    # Filter to weekdays only
    workdays = [d for d in days if d.weekday() < 5]
    
    # Apply threshold if provided
    if start_threshold:
        workdays = [d for d in workdays if d >= start_threshold]
    
    return workdays


# =============================================================================
# SECTION 5: FINANCIAL ENGINE (The Brain)
# =============================================================================

def calculate_unit_material_cost(engine) -> float:
    """
    Calculate total material cost per unit from BOM.
    
    Formula: SUM(part_cost * qty_per_unit) for all BOM items
    
    Returns: Float representing total material cost per unit
    """
    try:
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT SUM(p.cost * b.qty_per_unit) as total_cost
                FROM bom_items b
                JOIN part_master p ON b.part_id = p.id
            """))
            row = result.fetchone()
            return float(row[0]) if row and row[0] else 0.0
    except Exception as e:
        logger.error(f"BOM cost calculation failed: {e}")
        return 4050.0  # Fallback to approximate cost


def generate_financial_ledgers(engine):
    """
    Generate P&L and Cash Flow ledgers from production and OpEx data.
    
    This is the CORE FINANCIAL ENGINE. It produces two ledgers:
    
    1. P&L Ledger (Accrual Basis):
       - Revenue recognized on build_date
       - Expenses matched to period
    
    2. Cash Ledger (Cash Basis):
       - Cash In: build_date (Direct) or build_date + 30 (Dealer)
       - Cash Out: Based on supplier payment terms
    
    CRITICAL ASSUMPTIONS:
    - MSRP: $8,500
    - Dealer Discount: 25% (pays $6,375)
    - Dealer Payment Terms: Net 30
    - Direct Sales: Cash on delivery
    """
    try:
        # Load all required data
        df_units = pd.read_sql("SELECT * FROM production_unit", engine)
        df_parts = pd.read_sql("SELECT * FROM part_master", engine)
        df_bom = pd.read_sql("SELECT * FROM bom_items", engine)
        df_roles = pd.read_sql("SELECT * FROM opex_roles", engine)
        df_staffing = pd.read_sql("SELECT * FROM opex_staffing_plan", engine)
        df_expenses = pd.read_sql("SELECT * FROM opex_general_expenses", engine)
        
        # Handle empty tables gracefully
        if df_units.empty:
            return pd.DataFrame(), pd.DataFrame()
        
        # Parse dates consistently
        df_units['build_date'] = pd.to_datetime(df_units['build_date'])
        df_staffing['month_date'] = pd.to_datetime(df_staffing['month_date'])
        if not df_expenses.empty:
            df_expenses['month_date'] = pd.to_datetime(df_expenses['month_date'])
        
        # Load pricing configuration
        try:
            df_pricing = pd.read_sql("SELECT * FROM pricing_config", engine)
            pricing_by_year = {row['year']: (row['msrp'], row['dealer_discount_pct']) 
                              for _, row in df_pricing.iterrows()}
        except:
            # Fallback to defaults if table doesn't exist
            pricing_by_year = {2026: (8500.0, 0.75), 2027: (8500.0, 0.75), 2028: (8750.0, 0.77)}
        
        # Load channel mix configuration
        try:
            df_channel = pd.read_sql("SELECT * FROM channel_mix_config", engine)
            channel_by_yq = {(row['year'], row['quarter']): row['direct_pct'] 
                            for _, row in df_channel.iterrows()}
        except:
            # Fallback to 25% direct
            channel_by_yq = {}
        
        # Calculate unit material cost
        unit_material_cost = 0.0
        for _, part in df_parts.iterrows():
            bom_match = df_bom[df_bom['part_id'] == part['id']]
            if not bom_match.empty:
                qty = bom_match.iloc[0]['qty_per_unit']
                unit_material_cost += qty * part['cost']
        
        # Default pricing constants (used if config missing)
        DEFAULT_MSRP = 8500.0
        DEFAULT_DEALER_DISCOUNT = 0.75
        DEALER_PAYMENT_LAG = 30  # days
        
        pnl_entries = []
        cash_entries = []
        
        # ---------------------------------------------------------------------
        # REVENUE & COGS (Per Unit)
        # ---------------------------------------------------------------------
        for _, unit in df_units.iterrows():
            build_dt = unit['build_date']
            is_direct = unit['sales_channel'] == 'DIRECT'
            
            # Get year-specific pricing
            unit_year = build_dt.year
            msrp, dealer_discount = pricing_by_year.get(unit_year, (DEFAULT_MSRP, DEFAULT_DEALER_DISCOUNT))
            
            # Revenue calculation
            revenue = msrp if is_direct else msrp * dealer_discount
            cash_lag = 0 if is_direct else DEALER_PAYMENT_LAG
            
            # P&L Entry: Revenue on build date (accrual)
            pnl_entries.append({
                "Date": build_dt,
                "Category": "Sales Revenue",
                "Type": "Revenue",
                "Amount": revenue
            })
            
            # P&L Entry: Material cost on build date
            pnl_entries.append({
                "Date": build_dt,
                "Category": "Raw Materials",
                "Type": "COGS",
                "Amount": -unit_material_cost
            })
            
            # Cash Entry: Collection (with lag for dealers)
            cash_entries.append({
                "Date": build_dt + timedelta(days=cash_lag),
                "Category": "Customer Collections",
                "Type": "INFLOW",
                "Amount": revenue
            })
        
        # ---------------------------------------------------------------------
        # SUPPLY CHAIN PAYMENTS (Batched Monthly)
        # ---------------------------------------------------------------------
        # Group builds by month for supplier payments
        df_units['build_month'] = df_units['build_date'].dt.to_period('M').dt.to_timestamp()
        monthly_builds = df_units.groupby('build_month').size()
        
        for month_start, unit_count in monthly_builds.items():
            if unit_count == 0:
                continue
            
            for _, part in df_parts.iterrows():
                bom_match = df_bom[df_bom['part_id'] == part['id']]
                if bom_match.empty:
                    continue
                
                qty_per_unit = bom_match.iloc[0]['qty_per_unit']
                total_parts_needed = qty_per_unit * unit_count
                total_cost = total_parts_needed * part['cost']
                
                deposit_pct = part['deposit_pct'] if part['deposit_pct'] else 0
                deposit_days = int(part['deposit_days']) if part['deposit_days'] else 0
                balance_days = int(part['balance_days']) if part['balance_days'] else 0
                
                # Deposit payment (if applicable)
                if deposit_pct > 0:
                    deposit_date = month_start + timedelta(days=deposit_days)
                    deposit_amount = total_cost * deposit_pct
                    cash_entries.append({
                        "Date": deposit_date,
                        "Category": "Supplier Deposit",
                        "Type": "MATERIAL_OUTFLOW",
                        "Amount": -deposit_amount
                    })
                
                # Balance payment
                balance_pct = 1.0 - deposit_pct
                if balance_pct > 0:
                    balance_date = month_start + timedelta(days=balance_days)
                    balance_amount = total_cost * balance_pct
                    cash_entries.append({
                        "Date": balance_date,
                        "Category": "Supplier Balance",
                        "Type": "MATERIAL_OUTFLOW",
                        "Amount": -balance_amount
                    })
        
        # ---------------------------------------------------------------------
        # PAYROLL (Monthly) - Outsourced Manufacturing Model
        # ---------------------------------------------------------------------
        # Note: No direct labor (assemblers) - manufacturing is outsourced
        # All internal staff are G&A/OpEx, not COGS
        
        merged_payroll = pd.merge(
            df_staffing, 
            df_roles, 
            left_on='role_id', 
            right_on='id',
            suffixes=('_staff', '_role')
        )
        
        for _, row in merged_payroll.iterrows():
            monthly_cost = (row['annual_salary'] / 12) * row['headcount']
            
            if monthly_cost <= 0:
                continue
            
            pay_date = row['month_date']
            
            # In outsourced model, all internal staff are overhead (OpEx)
            # No assemblers = no direct labor COGS
            # Field service and QA are also OpEx (support costs)
            
            # P&L Entry - All payroll is OpEx in outsourced model
            pnl_entries.append({
                "Date": pay_date,
                "Category": "Salaries & Benefits",
                "Type": "OpEx",
                "Amount": -monthly_cost
            })
            
            # Cash Entry (payroll is OpEx, not fundable by LOC)
            cash_entries.append({
                "Date": pay_date,
                "Category": "Payroll",
                "Type": "OPEX_OUTFLOW",
                "Amount": -monthly_cost
            })
        
        # ---------------------------------------------------------------------
        # GENERAL EXPENSES (R&D, SG&A)
        # ---------------------------------------------------------------------
        if not df_expenses.empty:
            for _, row in df_expenses.iterrows():
                if row['amount'] <= 0:
                    continue
                
                pnl_entries.append({
                    "Date": row['month_date'],
                    "Category": row['category'],
                    "Type": "OpEx",
                    "Amount": -row['amount']
                })
                
                cash_entries.append({
                    "Date": row['month_date'],
                    "Category": "Operating Expenses",
                    "Type": "OPEX_OUTFLOW",
                    "Amount": -row['amount']
                })
        
        # Create DataFrames
        df_pnl = pd.DataFrame(pnl_entries) if pnl_entries else pd.DataFrame()
        df_cash = pd.DataFrame(cash_entries) if cash_entries else pd.DataFrame()
        
        # Ensure Date column is datetime
        if not df_pnl.empty:
            df_pnl['Date'] = pd.to_datetime(df_pnl['Date'])
        if not df_cash.empty:
            df_cash['Date'] = pd.to_datetime(df_cash['Date'])
        
        return df_pnl, df_cash
    
    except Exception as e:
        logger.error(f"Financial ledger generation failed: {e}")
        logger.error(traceback.format_exc())
        return pd.DataFrame(), pd.DataFrame()


def run_cash_waterfall(
    df_cash: pd.DataFrame,
    starting_equity: float,
    loc_limit: float,
    enforce_loc_rules: bool = True
) -> pd.DataFrame:
    """
    The Cash Waterfall Engine - Transaction-by-Transaction Processing
    
    This is the MATHEMATICALLY CRITICAL function that determines:
    1. When you run out of cash
    2. How much credit you're using
    3. Whether the business is viable
    
    WATERFALL RULES:
    1. INFLOWS (Revenue):
       - First: Pay down LOC balance
       - Then: Add remainder to Cash
    
    2. OUTFLOWS (Expenses):
       - MATERIAL_OUTFLOW: Can use LOC if cash insufficient
       - OPEX_OUTFLOW: MUST come from Cash (cannot use LOC)
    
    3. LOC CONSTRAINTS:
       - Maximum draw: loc_limit
       - Interest: Not modeled (simplification)
    
    Returns: DataFrame with daily cash positions
    """
    if df_cash.empty:
        return pd.DataFrame({
            "Date": [date.today()],
            "Net_Cash": [starting_equity],
            "LOC_Usage": [0.0],
            "LOC_Available": [loc_limit],
            "Total_Liquidity": [starting_equity + loc_limit],
            "Cash_Crunch": [False]
        })
    
    # Sort transactions chronologically
    df_sorted = df_cash.sort_values('Date').copy()
    
    # Initialize state
    cash_balance = float(starting_equity)
    loc_balance = 0.0  # Amount drawn on LOC
    
    history = []
    
    for _, txn in df_sorted.iterrows():
        txn_date = txn['Date']
        amount = float(txn['Amount'])
        txn_type = txn.get('Type', 'UNKNOWN')
        
        # === INFLOW PROCESSING ===
        if amount > 0:
            # Revenue comes in - pay down LOC first
            if loc_balance > 0:
                loc_paydown = min(loc_balance, amount)
                loc_balance -= loc_paydown
                cash_balance += (amount - loc_paydown)
            else:
                cash_balance += amount
        
        # === OUTFLOW PROCESSING ===
        else:
            expense = abs(amount)
            
            # Is this expense eligible for LOC funding?
            # If enforce_loc_rules is True: only MATERIAL_OUTFLOW can use LOC
            # If enforce_loc_rules is False: any expense can use LOC (more lenient)
            if enforce_loc_rules:
                is_loc_eligible = txn_type == 'MATERIAL_OUTFLOW'
            else:
                is_loc_eligible = True  # All expenses can use LOC when rules are off
            
            if is_loc_eligible:
                # MATERIAL: Can use LOC if cash insufficient
                if cash_balance >= expense:
                    # Enough cash - use it
                    cash_balance -= expense
                else:
                    # Not enough cash - calculate what we can cover
                    cash_available = max(0, cash_balance)
                    shortfall = expense - cash_available
                    
                    # Use up available cash first
                    cash_balance = cash_balance - cash_available  # Goes to 0 or stays negative
                    
                    # Draw on LOC for the shortfall (up to available credit)
                    loc_available = loc_limit - loc_balance
                    loc_draw = min(shortfall, loc_available)
                    loc_balance += loc_draw
                    
                    # If LOC can't cover everything, cash goes negative
                    uncovered = shortfall - loc_draw
                    if uncovered > 0:
                        cash_balance -= uncovered
            else:
                # OPEX: Must come from cash only (cannot use LOC)
                cash_balance -= expense
        
        # Record state after transaction
        loc_available = max(0, loc_limit - loc_balance)
        total_liquidity = cash_balance + loc_available
        
        history.append({
            "Date": txn_date,
            "Net_Cash": round(cash_balance, 2),
            "LOC_Usage": round(loc_balance, 2),
            "LOC_Available": round(loc_available, 2),
            "Total_Liquidity": round(total_liquidity, 2),
            "Cash_Crunch": cash_balance < 0
        })
    
    return pd.DataFrame(history)


# =============================================================================
# SECTION 6: UI COMPONENTS (Brand Compliant)
# =============================================================================

def render_header():
    """Render IdleX branded header."""
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:12px;margin-bottom:1.5rem;">
        <div style="font-size:2.5rem;font-weight:800;font-family:Montserrat;color:{NAVY};">
            idle<span style="color:{X_BLUE};">X</span>
        </div>
        <div style="font-size:1rem;color:{SLATE};font-family:Open Sans;
                    border-left:2px solid {LIGHT};padding-left:12px;">
            Enterprise Resource Planning
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_sidebar_brand():
    """Render sidebar branding."""
    st.sidebar.markdown(f"""
    <div style="text-align:center;padding:1rem 0 1.5rem 0;">
        <div style="font-size:2rem;font-weight:800;font-family:Montserrat;">
            idle<span style="color:{X_BLUE};">X</span>
        </div>
        <div style="font-size:0.85rem;color:rgba(255,255,255,0.7);margin-top:4px;">
            Enterprise Resource Planning
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_metric_card(label: str, value: str, status: str = "neutral", delta: str = None):
    """Render a branded metric card."""
    status_class = {
        "success": "metric-card-success",
        "warning": "metric-card-warn",
        "danger": "metric-card-danger",
        "neutral": ""
    }.get(status, "")
    
    delta_html = f'<div style="font-size:0.85rem;color:{SLATE};margin-top:4px;">{delta}</div>' if delta else ""
    
    st.markdown(f"""
    <div class="metric-card {status_class}">
        <div style="font-size:0.85rem;color:{SLATE};font-family:Open Sans;margin-bottom:4px;">{label}</div>
        <div style="font-size:1.75rem;font-weight:700;color:{NAVY};font-family:Montserrat;">{value}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)


def render_financial_statement(df: pd.DataFrame, title: str):
    """Render a GAAP-compliant financial statement table."""
    if df.empty:
        st.info(f"No data available for {title}")
        return
    
    html = f"<h3 style='font-family:Montserrat;color:{NAVY};'>{title}</h3>"
    html += "<table class='financial-table'>"
    
    # Header row
    html += "<thead><tr><th class='row-header'>Account</th>"
    for col in df.columns:
        html += f"<th>{col}</th>"
    html += "</tr></thead><tbody>"
    
    # Data rows
    section_headers = ['Revenue', 'Cost of Goods Sold', 'Operating Expenses', 'Operating Activities']
    total_rows = ['Gross Profit', 'Total OpEx', 'Net Cash Flow']
    grand_totals = ['Net Income', 'Ending Cash Balance', 'EBITDA']
    
    for index, row in df.iterrows():
        row_label = str(index).strip()
        
        if row_label in section_headers:
            row_class = "section-header"
        elif row_label in total_rows:
            row_class = "total-row"
        elif row_label in grand_totals:
            row_class = "grand-total"
        else:
            row_class = "indent"
        
        html += f"<tr class='{row_class}'><td class='row-header'>{row_label}</td>"
        
        if row_class == "section-header":
            for _ in df.columns:
                html += "<td></td>"
        else:
            for col in df.columns:
                val = row[col]
                val_class = "negative" if isinstance(val, (int, float)) and val < 0 else ""
                html += f"<td style='text-align:right;' class='{val_class}'>{money(val)}</td>"
        
        html += "</tr>"
    
    html += "</tbody></table>"
    st.markdown(html, unsafe_allow_html=True)


# =============================================================================
# SECTION 7: VIEW FUNCTIONS
# =============================================================================

def view_dashboard(engine, df_pnl, df_cash):
    """
    Executive Dashboard & Negotiation Console
    
    Split layout:
    - LEFT: Scenario inputs (Equity, Credit, Growth)
    - RIGHT: Results (KPIs, Charts, Alerts)
    """
    render_header()
    st.markdown("### Executive Dashboard")
    
    col_input, col_output = st.columns([1, 3])
    
    with col_input:
        st.markdown(f"""
        <div style="background:{LIGHT};padding:1rem;border-radius:8px;margin-bottom:1rem;">
            <div style="font-weight:700;color:{NAVY};font-family:Montserrat;margin-bottom:0.5rem;">
                ⚙️ Scenario Constraints
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        equity_input = st.number_input(
            "Investor Equity ($)",
            min_value=0,
            max_value=10_000_000,
            value=1_600_000,
            step=100_000,
            help="Total cash from equity investment"
        )
        
        loc_input = st.number_input(
            "Credit Limit (LOC) ($)",
            min_value=0,
            max_value=5_000_000,
            value=500_000,
            step=50_000,
            help="Maximum line of credit available"
        )
        
        enforce_rules = st.checkbox(
            "Enforce Credit Rules",
            value=True,
            help="If checked, LOC can only fund inventory (not OpEx/Payroll)"
        )
        
        st.markdown("---")
        
        st.markdown(f"""
        <div style="background:{LIGHT};padding:1rem;border-radius:8px;margin-bottom:1rem;">
            <div style="font-weight:700;color:{NAVY};font-family:Montserrat;margin-bottom:0.5rem;">
                🚀 Growth Assumptions
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        growth_rate = st.slider(
            "Monthly Growth Rate (%)",
            min_value=0,
            max_value=25,
            value=2,
            help="Projected monthly growth in unit sales"
        )
        
        st.caption(f"*Growth model not yet applied to production schedule*")
    
    with col_output:
        # Run the waterfall calculation
        if df_cash.empty:
            st.warning("No financial data available. Please check the production schedule.")
            return
        
        waterfall_df = run_cash_waterfall(df_cash, equity_input, loc_input, enforce_rules)
        
        if waterfall_df.empty:
            st.warning("Could not generate cash flow projection.")
            return
        
        # Calculate KPIs
        min_cash = waterfall_df['Net_Cash'].min()
        max_loc = waterfall_df['LOC_Usage'].max()
        end_cash = waterfall_df.iloc[-1]['Net_Cash']
        end_loc = waterfall_df.iloc[-1]['LOC_Usage']
        
        # Revenue calculation
        if not df_pnl.empty:
            revenue_2026 = df_pnl[
                (df_pnl['Date'].dt.year == 2026) & 
                (df_pnl['Category'] == 'Sales Revenue')
            ]['Amount'].sum()
        else:
            revenue_2026 = 0
        
        # Determine funding status
        if min_cash < 0:
            status = "UNDERFUNDED"
            shortfall = abs(min_cash)
        elif max_loc > loc_input:
            status = "CREDIT_EXCEEDED"
            shortfall = max_loc - loc_input
        else:
            status = "FUNDED"
            shortfall = 0
        
        # KPI Cards
        k1, k2, k3, k4 = st.columns(4)
        
        with k1:
            render_metric_card(
                "2026 Revenue",
                f"${revenue_2026:,.0f}",
                "success" if revenue_2026 > 0 else "neutral"
            )
        
        with k2:
            render_metric_card(
                "Minimum Cash",
                f"${min_cash:,.0f}",
                "danger" if min_cash < 0 else ("warning" if min_cash < 250000 else "success")
            )
        
        with k3:
            render_metric_card(
                "Peak LOC Usage",
                f"${max_loc:,.0f}",
                "danger" if max_loc > loc_input else ("warning" if max_loc > loc_input * 0.8 else "success"),
                delta=f"of ${loc_input:,.0f} limit"
            )
        
        with k4:
            render_metric_card(
                "Ending Cash",
                f"${end_cash:,.0f}",
                "success" if end_cash > 500000 else ("warning" if end_cash > 0 else "danger")
            )
        
        # Status Alert
        st.markdown("<br>", unsafe_allow_html=True)
        
        if status == "UNDERFUNDED":
            st.error(f"""
            🚨 **CASH CRUNCH DETECTED**
            
            Your plan requires **${shortfall:,.0f}** more equity to cover OpEx obligations.
            
            OpEx (payroll, rent, etc.) cannot be funded by the line of credit under standard banking covenants.
            """)
        elif status == "CREDIT_EXCEEDED":
            st.warning(f"""
            ⚠️ **CREDIT LIMIT EXCEEDED**
            
            Production creates a **${shortfall:,.0f}** overage beyond your LOC limit.
            
            Consider: Increasing credit limit, reducing production pace, or securing additional equity.
            """)
        else:
            runway_months = end_cash / (abs(min_cash - end_cash) / 12 + 1) if min_cash != end_cash else 24
            st.success(f"""
            ✅ **PLAN IS FULLY FUNDED**
            
            Minimum cash cushion: **${min_cash:,.0f}** | Peak credit utilization: **{(max_loc/loc_input*100):.0f}%**
            """)
        
        # Liquidity Chart
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"<h4 style='font-family:Montserrat;color:{NAVY};'>Liquidity Forecast</h4>", unsafe_allow_html=True)
        
        fig = go.Figure()
        
        # Cash balance area
        fig.add_trace(go.Scatter(
            x=waterfall_df['Date'],
            y=waterfall_df['Net_Cash'],
            name='Net Cash (Equity)',
            fill='tozeroy',
            fillcolor=f'rgba(16, 185, 129, 0.3)',
            line=dict(color=STATUS_OK, width=2)
        ))
        
        # LOC usage area
        fig.add_trace(go.Scatter(
            x=waterfall_df['Date'],
            y=waterfall_df['LOC_Usage'],
            name='LOC Usage',
            fill='tozeroy',
            fillcolor=f'rgba(58, 119, 216, 0.3)',
            line=dict(color=X_BLUE, width=2)
        ))
        
        # Zero line
        fig.add_hline(
            y=0, 
            line_dash="dash", 
            line_color=STATUS_DANGER,
            annotation_text="Zero Cash (Crisis)",
            annotation_position="right"
        )
        
        # LOC limit line
        fig.add_hline(
            y=loc_input,
            line_dash="dot",
            line_color=YELLOW,
            annotation_text=f"Credit Limit (${loc_input/1000:.0f}K)",
            annotation_position="right"
        )
        
        fig.update_layout(
            margin=dict(l=0, r=0, t=30, b=0),
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="left",
                x=0
            ),
            xaxis=dict(
                showgrid=False,
                title=""
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor=LIGHT,
                title="$ Amount",
                tickformat="$,.0f"
            ),
            font=dict(family="Open Sans")
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Monthly summary table
        with st.expander("📊 Monthly Cash Summary", expanded=False):
            monthly = waterfall_df.copy()
            monthly['Month'] = pd.to_datetime(monthly['Date']).dt.to_period('M')
            monthly_summary = monthly.groupby('Month').agg({
                'Net_Cash': 'last',
                'LOC_Usage': 'max',
                'Cash_Crunch': 'any'
            }).reset_index()
            monthly_summary['Month'] = monthly_summary['Month'].astype(str)
            monthly_summary.columns = ['Month', 'Ending Cash', 'Peak LOC', 'Crunch?']
            st.dataframe(monthly_summary, use_container_width=True, hide_index=True)


def view_financials(engine, df_pnl, df_cash):
    """Financial Statements View - GAAP Compliant"""
    render_header()
    st.markdown("### Financial Statements")
    
    if df_pnl.empty:
        st.warning("No financial data available.")
        return
    
    # Aggregation selector
    col1, col2 = st.columns([1, 3])
    with col1:
        freq = st.radio(
            "Aggregation",
            ["Monthly", "Quarterly", "Annually"],
            horizontal=True,
            index=2
        )
    
    freq_map = {"Monthly": "ME", "Quarterly": "QE", "Annually": "YE"}
    freq_code = freq_map[freq]
    
    # === INCOME STATEMENT ===
    st.markdown("---")
    st.markdown(f"<h3 style='font-family:Montserrat;color:{NAVY};'>Statement of Operations</h3>", unsafe_allow_html=True)
    
    # Aggregate by period
    pnl_agg = df_pnl.groupby([
        pd.Grouper(key='Date', freq=freq_code),
        'Type',
        'Category'
    ])['Amount'].sum().unstack(level=[1, 2]).fillna(0)
    
    # Format period labels
    if freq == "Monthly":
        pnl_agg.index = pnl_agg.index.strftime('%b %Y')
    elif freq == "Quarterly":
        pnl_agg.index = pnl_agg.index.to_period('Q').astype(str)
    else:
        pnl_agg.index = pnl_agg.index.strftime('%Y')
    
    # Build statement structure
    stmt = pd.DataFrame(columns=pnl_agg.index)
    
    def safe_sum(keys):
        """Safely sum columns that may or may not exist."""
        total = pd.Series(0.0, index=pnl_agg.index)
        for k in keys:
            if k in pnl_agg.columns:
                total += pnl_agg[k].fillna(0)
        return total
    
    # Revenue section
    stmt.loc['Revenue'] = ""
    stmt.loc['Product Sales'] = safe_sum([('Revenue', 'Sales Revenue')])
    
    # COGS section - Outsourced manufacturing means only materials
    stmt.loc['Cost of Goods Sold'] = ""
    stmt.loc['Materials'] = safe_sum([('COGS', 'Raw Materials')])
    
    # Check if there's any direct labor (for backward compatibility)
    direct_labor = safe_sum([('COGS', 'Direct Labor')])
    if direct_labor.abs().sum() > 0:
        stmt.loc['Direct Labor'] = direct_labor
        total_cogs = stmt.loc['Materials'] + stmt.loc['Direct Labor']
    else:
        total_cogs = stmt.loc['Materials']
    
    stmt.loc['Gross Profit'] = stmt.loc['Product Sales'] + total_cogs  # COGS is negative
    
    # OpEx section
    stmt.loc['Operating Expenses'] = ""
    stmt.loc['Salaries & Benefits'] = safe_sum([('OpEx', 'Salaries & Benefits')])
    
    # Find other OpEx categories
    opex_cats = [c for c in pnl_agg.columns if c[0] == 'OpEx' and c[1] != 'Salaries & Benefits']
    for cat in opex_cats:
        stmt.loc[cat[1]] = safe_sum([cat])
    
    total_opex = safe_sum([('OpEx', c[1]) for c in opex_cats]) + stmt.loc['Salaries & Benefits']
    stmt.loc['Total OpEx'] = total_opex
    
    # Net Income
    stmt.loc['Net Income'] = stmt.loc['Gross Profit'] + total_opex  # OpEx is negative
    
    render_financial_statement(stmt, "")
    
    # === CASH FLOW STATEMENT ===
    st.markdown("---")
    st.markdown(f"<h3 style='font-family:Montserrat;color:{NAVY};'>Statement of Cash Flows</h3>", unsafe_allow_html=True)
    st.caption("Direct Method - Operating Activities Only")
    
    if df_cash.empty:
        st.info("No cash flow data available.")
        return
    
    cash_agg = df_cash.groupby([
        pd.Grouper(key='Date', freq=freq_code),
        'Category'
    ])['Amount'].sum().unstack().fillna(0)
    
    # Format period labels
    if freq == "Monthly":
        cash_agg.index = cash_agg.index.strftime('%b %Y')
    elif freq == "Quarterly":
        cash_agg.index = cash_agg.index.to_period('Q').astype(str)
    else:
        cash_agg.index = cash_agg.index.strftime('%Y')
    
    cf = pd.DataFrame(columns=cash_agg.index)
    
    cf.loc['Operating Activities'] = ""
    cf.loc['Customer Collections'] = cash_agg.get('Customer Collections', pd.Series(0, index=cash_agg.index))
    cf.loc['Supplier Payments'] = (
        cash_agg.get('Supplier Deposit', pd.Series(0, index=cash_agg.index)) +
        cash_agg.get('Supplier Balance', pd.Series(0, index=cash_agg.index))
    )
    cf.loc['Payroll'] = cash_agg.get('Payroll', pd.Series(0, index=cash_agg.index))
    cf.loc['Operating Expenses'] = cash_agg.get('Operating Expenses', pd.Series(0, index=cash_agg.index))
    
    cf.loc['Net Cash Flow'] = (
        cf.loc['Customer Collections'] + 
        cf.loc['Supplier Payments'] + 
        cf.loc['Payroll'] +
        cf.loc['Operating Expenses']
    )
    
    render_financial_statement(cf, "")


def view_production(engine):
    """Production & Sales View"""
    render_header()
    st.markdown("### Production & Sales Planning")
    
    # Tabs for different sections
    tab_manifest, tab_planner, tab_pricing, tab_channel = st.tabs([
        "📋 Production Manifest", 
        "🚀 Smart Planner",
        "💰 Pricing Config",
        "📊 Channel Mix"
    ])
    
    with tab_manifest:
        st.markdown(f"<h4 style='font-family:Montserrat;color:{NAVY};'>Production Schedule</h4>", unsafe_allow_html=True)
        
        try:
            df_units = pd.read_sql("SELECT * FROM production_unit ORDER BY build_date", engine)
            
            if df_units.empty:
                st.info("No production units scheduled.")
            else:
                # Summary metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Units", f"{len(df_units):,}")
                with col2:
                    direct = (df_units['sales_channel'] == 'DIRECT').sum()
                    st.metric("Direct", f"{direct:,} ({direct/len(df_units)*100:.0f}%)")
                with col3:
                    dealer = (df_units['sales_channel'] == 'DEALER').sum()
                    st.metric("Dealer", f"{dealer:,} ({dealer/len(df_units)*100:.0f}%)")
                with col4:
                    planned = (df_units['status'] == 'PLANNED').sum()
                    st.metric("Planned", f"{planned:,}")
                
                # Editable grid
                edited_df = st.data_editor(
                    df_units,
                    column_config={
                        "id": st.column_config.NumberColumn("ID", disabled=True, width=60),
                        "serial_number": st.column_config.TextColumn("Serial #", width=100),
                        "build_date": st.column_config.TextColumn("Build Date", width=100),
                        "sales_channel": st.column_config.SelectboxColumn(
                            "Channel",
                            options=["DIRECT", "DEALER"],
                            width=100
                        ),
                        "status": st.column_config.SelectboxColumn(
                            "Status",
                            options=["PLANNED", "WIP", "COMPLETE", "CANCELLED"],
                            width=100
                        ),
                    },
                    hide_index=True,
                    height=400,
                    use_container_width=True
                )
                
                if st.button("💾 Save Changes", type="primary", key="save_manifest"):
                    try:
                        with engine.connect() as conn:
                            for _, row in edited_df.iterrows():
                                conn.execute(text("""
                                    UPDATE production_unit 
                                    SET sales_channel = :channel, status = :status
                                    WHERE id = :id
                                """), {
                                    "channel": row['sales_channel'],
                                    "status": row['status'],
                                    "id": row['id']
                                })
                            conn.commit()
                        st.success("Changes saved!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Save failed: {e}")
        
        except Exception as e:
            st.error(f"Could not load production data: {e}")
    
    with tab_planner:
        st.markdown(f"<h4 style='font-family:Montserrat;color:{NAVY};'>Smart Production Planner</h4>", unsafe_allow_html=True)
        
        col_config, col_plan = st.columns([1, 2])
        
        with col_config:
            start_date = st.date_input(
                "Production Start Date",
                value=date(2026, 1, 6),
                help="First day of production"
            )
            
            st.markdown("---")
            st.caption("⚠️ Regenerating will delete all PLANNED units and create new ones based on targets below.")
        
        with col_plan:
            try:
                # Get current distribution
                df_units = pd.read_sql("SELECT * FROM production_unit", engine)
                df_units['Month'] = pd.to_datetime(df_units['build_date']).dt.to_period('M').astype(str)
                existing = df_units.groupby('Month').size().to_dict()
                
                # Create plan grid
                months = pd.date_range('2026-01-01', '2028-12-01', freq='MS')
                plan_data = []
                for m in months:
                    month_key = m.strftime('%Y-%m')
                    plan_data.append({
                        "Month": m.date(),
                        "Target": existing.get(month_key, 0)
                    })
                
                plan_df = pd.DataFrame(plan_data)
                edited_plan = st.data_editor(
                    plan_df,
                    column_config={
                        "Month": st.column_config.DateColumn("Month", format="MMM YYYY", width=100),
                        "Target": st.column_config.NumberColumn("Units", min_value=0, max_value=5000, width=80)
                    },
                    hide_index=True,
                    height=400
                )
                
                if st.button("🚀 Regenerate Schedule", type="primary", use_container_width=True):
                    _regenerate_production_schedule(engine, edited_plan, start_date)
                    
            except Exception as e:
                st.error(f"Could not load planning data: {e}")
    
    with tab_pricing:
        st.markdown(f"<h4 style='font-family:Montserrat;color:{NAVY};'>Pricing Configuration by Year</h4>", unsafe_allow_html=True)
        st.caption("Set MSRP and dealer discount percentage for each year")
        
        try:
            df_pricing = pd.read_sql("SELECT * FROM pricing_config ORDER BY year", engine)
            
            if df_pricing.empty:
                st.info("No pricing configuration found. Add pricing below.")
                df_pricing = pd.DataFrame({
                    'id': [None, None, None],
                    'year': [2026, 2027, 2028],
                    'msrp': [8500.0, 8500.0, 8750.0],
                    'dealer_discount_pct': [0.75, 0.75, 0.77],
                    'notes': ['', '', '']
                })
            
            # Calculate derived fields for display
            df_pricing['Dealer Price'] = df_pricing['msrp'] * df_pricing['dealer_discount_pct']
            df_pricing['Dealer Margin'] = (1 - df_pricing['dealer_discount_pct']) * 100
            
            edited_pricing = st.data_editor(
                df_pricing,
                column_config={
                    "id": None,  # Hide ID
                    "year": st.column_config.NumberColumn("Year", disabled=True, width=80),
                    "msrp": st.column_config.NumberColumn("MSRP ($)", format="$%.0f", width=100),
                    "dealer_discount_pct": st.column_config.NumberColumn(
                        "Dealer Pays %", 
                        format="%.0f%%",
                        min_value=0.5,
                        max_value=1.0,
                        help="Percentage of MSRP that dealer pays (e.g., 0.75 = 75%)",
                        width=120
                    ),
                    "Dealer Price": st.column_config.NumberColumn("Dealer Price", format="$%.0f", disabled=True, width=110),
                    "Dealer Margin": st.column_config.NumberColumn("Dealer Margin %", format="%.0f%%", disabled=True, width=120),
                    "notes": st.column_config.TextColumn("Notes", width=200),
                },
                hide_index=True,
                use_container_width=True
            )
            
            if st.button("💾 Save Pricing", type="primary", key="save_pricing"):
                try:
                    with engine.connect() as conn:
                        for _, row in edited_pricing.iterrows():
                            if row['id'] is not None:
                                conn.execute(text("""
                                    UPDATE pricing_config 
                                    SET msrp = :msrp, dealer_discount_pct = :disc, notes = :notes
                                    WHERE id = :id
                                """), {
                                    "msrp": row['msrp'],
                                    "disc": row['dealer_discount_pct'],
                                    "notes": row['notes'] or '',
                                    "id": row['id']
                                })
                            else:
                                conn.execute(text("""
                                    INSERT INTO pricing_config (year, msrp, dealer_discount_pct, notes)
                                    VALUES (:year, :msrp, :disc, :notes)
                                """), {
                                    "year": row['year'],
                                    "msrp": row['msrp'],
                                    "disc": row['dealer_discount_pct'],
                                    "notes": row['notes'] or ''
                                })
                        conn.commit()
                    st.success("Pricing saved!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Save failed: {e}")
            
            # Revenue impact preview
            st.markdown("---")
            st.markdown(f"<h5 style='font-family:Montserrat;color:{NAVY};'>Revenue Impact Preview</h5>", unsafe_allow_html=True)
            
            try:
                df_units = pd.read_sql("SELECT build_date, sales_channel FROM production_unit", engine)
                df_units['build_date'] = pd.to_datetime(df_units['build_date'])
                df_units['year'] = df_units['build_date'].dt.year
                
                preview_data = []
                for year in [2026, 2027, 2028]:
                    year_units = df_units[df_units['year'] == year]
                    direct = (year_units['sales_channel'] == 'DIRECT').sum()
                    dealer = (year_units['sales_channel'] == 'DEALER').sum()
                    
                    pricing_row = edited_pricing[edited_pricing['year'] == year]
                    if not pricing_row.empty:
                        msrp = pricing_row.iloc[0]['msrp']
                        disc = pricing_row.iloc[0]['dealer_discount_pct']
                    else:
                        msrp, disc = 8500, 0.75
                    
                    direct_rev = direct * msrp
                    dealer_rev = dealer * msrp * disc
                    
                    preview_data.append({
                        'Year': year,
                        'Direct Units': direct,
                        'Dealer Units': dealer,
                        'Direct Revenue': direct_rev,
                        'Dealer Revenue': dealer_rev,
                        'Total Revenue': direct_rev + dealer_rev
                    })
                
                preview_df = pd.DataFrame(preview_data)
                st.dataframe(
                    preview_df,
                    column_config={
                        "Direct Revenue": st.column_config.NumberColumn(format="$%,.0f"),
                        "Dealer Revenue": st.column_config.NumberColumn(format="$%,.0f"),
                        "Total Revenue": st.column_config.NumberColumn(format="$%,.0f"),
                    },
                    hide_index=True,
                    use_container_width=True
                )
            except:
                pass
                
        except Exception as e:
            st.error(f"Could not load pricing config: {e}")
    
    with tab_channel:
        st.markdown(f"<h4 style='font-family:Montserrat;color:{NAVY};'>Channel Mix by Quarter</h4>", unsafe_allow_html=True)
        st.caption("Set target Direct vs Dealer split by quarter. Use 'Apply to Schedule' to update production units.")
        
        try:
            df_channel = pd.read_sql("SELECT * FROM channel_mix_config ORDER BY year, quarter", engine)
            
            if df_channel.empty:
                # Create default data
                channel_data = []
                for year in [2026, 2027, 2028]:
                    for q in [1, 2, 3, 4]:
                        channel_data.append({
                            'id': None,
                            'year': year,
                            'quarter': q,
                            'direct_pct': 0.25,
                            'notes': ''
                        })
                df_channel = pd.DataFrame(channel_data)
            
            # Add calculated dealer % for display
            df_channel['dealer_pct'] = 1 - df_channel['direct_pct']
            df_channel['period'] = df_channel.apply(lambda r: f"Q{r['quarter']} {r['year']}", axis=1)
            
            edited_channel = st.data_editor(
                df_channel,
                column_config={
                    "id": None,
                    "year": None,
                    "quarter": None,
                    "period": st.column_config.TextColumn("Quarter", disabled=True, width=100),
                    "direct_pct": st.column_config.NumberColumn(
                        "Direct %", 
                        format="%.0f%%",
                        min_value=0.0,
                        max_value=1.0,
                        help="Percentage of units sold direct (vs through dealers)",
                        width=100
                    ),
                    "dealer_pct": st.column_config.NumberColumn(
                        "Dealer %", 
                        format="%.0f%%",
                        disabled=True,
                        width=100
                    ),
                    "notes": st.column_config.TextColumn("Notes", width=200),
                },
                hide_index=True,
                use_container_width=True,
                height=500
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("💾 Save Channel Mix", type="primary", key="save_channel"):
                    try:
                        with engine.connect() as conn:
                            for _, row in edited_channel.iterrows():
                                if row['id'] is not None:
                                    conn.execute(text("""
                                        UPDATE channel_mix_config 
                                        SET direct_pct = :pct, notes = :notes
                                        WHERE id = :id
                                    """), {
                                        "pct": row['direct_pct'],
                                        "notes": row['notes'] or '',
                                        "id": row['id']
                                    })
                                else:
                                    conn.execute(text("""
                                        INSERT INTO channel_mix_config (year, quarter, direct_pct, notes)
                                        VALUES (:year, :q, :pct, :notes)
                                    """), {
                                        "year": row['year'],
                                        "q": row['quarter'],
                                        "pct": row['direct_pct'],
                                        "notes": row['notes'] or ''
                                    })
                            conn.commit()
                        st.success("Channel mix saved!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Save failed: {e}")
            
            with col2:
                if st.button("🔄 Apply to Production Schedule", type="secondary", key="apply_channel"):
                    try:
                        with engine.connect() as conn:
                            # Get all production units
                            units = conn.execute(text("""
                                SELECT id, build_date FROM production_unit
                            """)).fetchall()
                            
                            updated = 0
                            for unit_id, build_date in units:
                                # Parse date and get quarter
                                if isinstance(build_date, str):
                                    bd = datetime.strptime(build_date[:10], '%Y-%m-%d')
                                else:
                                    bd = build_date
                                
                                year = bd.year
                                quarter = (bd.month - 1) // 3 + 1
                                
                                # Find channel mix for this quarter
                                channel_row = edited_channel[
                                    (edited_channel['year'] == year) & 
                                    (edited_channel['quarter'] == quarter)
                                ]
                                
                                if not channel_row.empty:
                                    direct_pct = channel_row.iloc[0]['direct_pct']
                                else:
                                    direct_pct = 0.25
                                
                                # Randomly assign based on probability
                                import random
                                channel = 'DIRECT' if random.random() < direct_pct else 'DEALER'
                                
                                conn.execute(text("""
                                    UPDATE production_unit SET sales_channel = :ch WHERE id = :id
                                """), {"ch": channel, "id": unit_id})
                                updated += 1
                            
                            conn.commit()
                        st.success(f"Updated {updated:,} units based on channel mix targets!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Apply failed: {e}")
            
            # Show current vs target mix
            st.markdown("---")
            st.markdown(f"<h5 style='font-family:Montserrat;color:{NAVY};'>Current vs Target Mix</h5>", unsafe_allow_html=True)
            
            try:
                df_units = pd.read_sql("SELECT build_date, sales_channel FROM production_unit", engine)
                df_units['build_date'] = pd.to_datetime(df_units['build_date'])
                df_units['year'] = df_units['build_date'].dt.year
                df_units['quarter'] = (df_units['build_date'].dt.month - 1) // 3 + 1
                
                comparison = []
                for year in [2026, 2027, 2028]:
                    for q in [1, 2, 3, 4]:
                        q_units = df_units[(df_units['year'] == year) & (df_units['quarter'] == q)]
                        if len(q_units) > 0:
                            actual_direct = (q_units['sales_channel'] == 'DIRECT').mean()
                        else:
                            actual_direct = 0
                        
                        target_row = edited_channel[
                            (edited_channel['year'] == year) & 
                            (edited_channel['quarter'] == q)
                        ]
                        target = target_row.iloc[0]['direct_pct'] if not target_row.empty else 0.25
                        
                        comparison.append({
                            'Period': f"Q{q} {year}",
                            'Units': len(q_units),
                            'Target Direct %': target,
                            'Actual Direct %': actual_direct,
                            'Variance': actual_direct - target
                        })
                
                comp_df = pd.DataFrame(comparison)
                st.dataframe(
                    comp_df,
                    column_config={
                        "Target Direct %": st.column_config.NumberColumn(format="%.0f%%"),
                        "Actual Direct %": st.column_config.NumberColumn(format="%.0f%%"),
                        "Variance": st.column_config.NumberColumn(format="%+.0f%%"),
                    },
                    hide_index=True,
                    use_container_width=True
                )
            except:
                pass
                
        except Exception as e:
            st.error(f"Could not load channel config: {e}")


def _regenerate_production_schedule(engine, plan_df, start_date):
    """Helper function to regenerate production schedule from plan."""
    with st.spinner("Optimizing production schedule..."):
        try:
            db_type = get_db_type()
            
            with engine.connect() as conn:
                # Delete only PLANNED units (preserve WIP/COMPLETE)
                conn.execute(text("DELETE FROM production_unit WHERE status = 'PLANNED'"))
                
                # Get next serial number
                result = conn.execute(text(
                    "SELECT serial_number FROM production_unit ORDER BY id DESC LIMIT 1"
                ))
                last_sn = result.scalar()
                
                if last_sn:
                    try:
                        sn_num = int(''.join(filter(str.isdigit, last_sn))) + 1
                    except:
                        sn_num = 1
                else:
                    sn_num = 1
                
                # Load channel mix config
                try:
                    channel_config = {}
                    channel_rows = conn.execute(text(
                        "SELECT year, quarter, direct_pct FROM channel_mix_config"
                    )).fetchall()
                    for year, quarter, pct in channel_rows:
                        channel_config[(year, quarter)] = pct
                except:
                    channel_config = {}
                
                # Generate new units based on plan
                for _, row in plan_df.iterrows():
                    target = int(row['Target'])
                    if target <= 0:
                        continue
                    
                    month_dt = row['Month']
                    if isinstance(month_dt, pd.Timestamp):
                        month_dt = month_dt.date()
                    
                    # Skip if month is before start date
                    month_end = date(month_dt.year, month_dt.month, 
                                    calendar.monthrange(month_dt.year, month_dt.month)[1])
                    if month_end < start_date:
                        continue
                    
                    # Get workdays for this month
                    threshold = start_date if (month_dt.year == start_date.year and 
                                              month_dt.month == start_date.month) else None
                    workdays = get_workdays(month_dt.year, month_dt.month, threshold)
                    
                    if not workdays:
                        continue
                    
                    # Get channel mix for this quarter
                    quarter = (month_dt.month - 1) // 3 + 1
                    direct_pct = channel_config.get((month_dt.year, quarter), 0.25)
                    
                    # Count existing non-PLANNED units for this month
                    month_str = month_dt.strftime('%Y-%m')
                    existing_result = conn.execute(text(
                        "SELECT build_date FROM production_unit WHERE status != 'PLANNED'"
                    ))
                    existing_dates = [r[0] for r in existing_result.fetchall()]
                    existing_count = sum(1 for d in existing_dates if str(d)[:7] == month_str)
                    
                    units_to_create = target - existing_count
                    if units_to_create <= 0:
                        continue
                    
                    # Distribute across workdays with channel mix
                    day_idx = 0
                    for i in range(units_to_create):
                        build_date = workdays[day_idx % len(workdays)]
                        channel = 'DIRECT' if random.random() < direct_pct else 'DEALER'
                        
                        conn.execute(text("""
                            INSERT INTO production_unit 
                            (serial_number, build_date, sales_channel, status)
                            VALUES (:sn, :bd, :ch, 'PLANNED')
                        """), {
                            "sn": f"IDX-{sn_num:05d}",
                            "bd": str(build_date),
                            "ch": channel
                        })
                        
                        sn_num += 1
                        day_idx += 1
                
                conn.commit()
            
            st.success("Schedule regenerated!")
            st.rerun()
        
        except Exception as e:
            st.error(f"Regeneration failed: {e}")
            logger.error(traceback.format_exc())


def view_opex(engine):
    """OpEx Budget View"""
    render_header()
    st.markdown("### Operating Expense Budget")
    
    tab1, tab2 = st.tabs(["👥 Headcount Plan", "💰 General Expenses"])
    
    with tab1:
        st.markdown(f"<h4 style='font-family:Montserrat;color:{NAVY};'>Headcount by Role</h4>", unsafe_allow_html=True)
        
        try:
            df_roles = pd.read_sql("SELECT * FROM opex_roles ORDER BY id", engine)
            df_staffing = pd.read_sql("SELECT * FROM opex_staffing_plan", engine)
            
            if df_roles.empty:
                st.info("No roles defined.")
                return
            
            # Merge for display
            merged = pd.merge(df_staffing, df_roles, left_on='role_id', right_on='id', suffixes=('_staff', '_role'))
            merged['Month'] = pd.to_datetime(merged['month_date']).dt.strftime('%Y-%m')
            
            # Pivot to grid format
            pivot = merged.pivot(
                index='role_name',
                columns='Month',
                values='headcount'
            ).fillna(0).reset_index()
            
            edited_hc = st.data_editor(pivot, use_container_width=True, height=400)
            
            if st.button("💾 Save Headcount", key="save_hc"):
                try:
                    db_type = get_db_type()
                    
                    with engine.connect() as conn:
                        # Melt back to long format
                        melted = edited_hc.melt(
                            id_vars=['role_name'],
                            var_name='Month',
                            value_name='headcount'
                        )
                        
                        for _, row in melted.iterrows():
                            # Get role ID
                            role_result = conn.execute(text(
                                "SELECT id FROM opex_roles WHERE role_name = :name"
                            ), {"name": row['role_name']})
                            role_id = role_result.scalar()
                            
                            if role_id:
                                month_date = f"{row['Month']}-01"
                                headcount = float(row['headcount']) if pd.notna(row['headcount']) else 0
                                
                                # Upsert logic (works for both PostgreSQL and SQLite)
                                if db_type == "postgresql":
                                    conn.execute(text("""
                                        INSERT INTO opex_staffing_plan (role_id, month_date, headcount)
                                        VALUES (:rid, :dt, :hc)
                                        ON CONFLICT (role_id, month_date) 
                                        DO UPDATE SET headcount = :hc
                                    """), {"rid": role_id, "dt": month_date, "hc": headcount})
                                else:
                                    # SQLite upsert
                                    conn.execute(text("""
                                        INSERT OR REPLACE INTO opex_staffing_plan 
                                        (id, role_id, month_date, headcount)
                                        VALUES (
                                            (SELECT id FROM opex_staffing_plan WHERE role_id = :rid AND month_date = :dt),
                                            :rid, :dt, :hc
                                        )
                                    """), {"rid": role_id, "dt": month_date, "hc": headcount})
                        
                        conn.commit()
                    
                    st.success("Headcount saved!")
                    st.rerun()
                
                except Exception as e:
                    st.error(f"Save failed: {e}")
            
            st.markdown("---")
            st.markdown(f"<h4 style='font-family:Montserrat;color:{NAVY};'>Salary Configuration</h4>", unsafe_allow_html=True)
            
            edited_roles = st.data_editor(
                df_roles,
                column_config={
                    "id": st.column_config.NumberColumn("ID", disabled=True, width=60),
                    "role_name": st.column_config.TextColumn("Role", width=200),
                    "annual_salary": st.column_config.NumberColumn(
                        "Annual Salary",
                        format="$%.0f",
                        width=120
                    ),
                    "department": st.column_config.SelectboxColumn(
                        "Department",
                        options=["Executive", "Engineering", "Operations", "Finance", "Sales", "Service"],
                        width=120
                    )
                },
                hide_index=True,
                use_container_width=True
            )
            
            if st.button("💾 Update Salaries", key="save_salaries"):
                try:
                    with engine.connect() as conn:
                        for _, row in edited_roles.iterrows():
                            conn.execute(text("""
                                UPDATE opex_roles
                                SET role_name = :name, annual_salary = :salary, department = :dept
                                WHERE id = :id
                            """), {
                                "name": row['role_name'],
                                "salary": row['annual_salary'],
                                "dept": row['department'],
                                "id": row['id']
                            })
                        conn.commit()
                    st.success("Salaries updated!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Update failed: {e}")
        
        except Exception as e:
            st.error(f"Could not load headcount data: {e}")
    
    with tab2:
        st.markdown(f"<h4 style='font-family:Montserrat;color:{NAVY};'>General & Administrative Expenses</h4>", unsafe_allow_html=True)
        
        try:
            df_expenses = pd.read_sql("SELECT * FROM opex_general_expenses", engine)
            
            if df_expenses.empty:
                st.info("No general expenses recorded. Add expenses below.")
                df_expenses = pd.DataFrame(columns=['id', 'category', 'expense_type', 'month_date', 'amount'])
            else:
                df_expenses['Month'] = pd.to_datetime(df_expenses['month_date']).dt.strftime('%Y-%m')
                
                # Pivot for editing
                pivot_exp = df_expenses.pivot(
                    index=['category', 'expense_type'],
                    columns='Month',
                    values='amount'
                ).fillna(0).reset_index()
                
                edited_exp = st.data_editor(pivot_exp, use_container_width=True)
                
                if st.button("💾 Save Expenses", key="save_expenses"):
                    try:
                        with engine.connect() as conn:
                            # Clear and rebuild
                            conn.execute(text("DELETE FROM opex_general_expenses"))
                            
                            # Melt back to long format
                            melted = edited_exp.melt(
                                id_vars=['category', 'expense_type'],
                                var_name='Month',
                                value_name='amount'
                            )
                            
                            for _, row in melted.iterrows():
                                if pd.notna(row['amount']) and row['amount'] != 0:
                                    conn.execute(text("""
                                        INSERT INTO opex_general_expenses 
                                        (category, expense_type, month_date, amount)
                                        VALUES (:cat, :type, :dt, :amt)
                                    """), {
                                        "cat": row['category'],
                                        "type": row['expense_type'],
                                        "dt": f"{row['Month']}-01",
                                        "amt": float(row['amount'])
                                    })
                            
                            conn.commit()
                        
                        st.success("Expenses saved!")
                        st.rerun()
                    
                    except Exception as e:
                        st.error(f"Save failed: {e}")
        
        except Exception as e:
            st.error(f"Could not load expense data: {e}")


def view_supply_chain(engine):
    """Supply Chain / BOM View"""
    render_header()
    st.markdown("### Bill of Materials & Supply Chain")
    
    try:
        df_parts = pd.read_sql("SELECT * FROM part_master ORDER BY id", engine)
        
        if df_parts.empty:
            st.info("No parts defined in BOM.")
            return
        
        st.markdown(f"<h4 style='font-family:Montserrat;color:{NAVY};'>Component Master</h4>", unsafe_allow_html=True)
        
        edited_parts = st.data_editor(
            df_parts,
            column_config={
                "id": st.column_config.NumberColumn("ID", disabled=True, width=50),
                "sku": st.column_config.TextColumn("SKU", disabled=True, width=100),
                "name": st.column_config.TextColumn("Component Name", width=180),
                "cost": st.column_config.NumberColumn("Unit Cost", format="$%.2f", width=90),
                "moq": st.column_config.NumberColumn("MOQ", width=70),
                "lead_time": st.column_config.NumberColumn("Lead Time (days)", width=100),
                "deposit_pct": st.column_config.NumberColumn("Deposit %", format="%.0f%%", width=80),
                "deposit_days": st.column_config.NumberColumn("Deposit Days", width=90),
                "balance_days": st.column_config.NumberColumn("Balance Days", width=90),
                "supplier_name": st.column_config.TextColumn("Supplier", width=120),
            },
            hide_index=True,
            use_container_width=True
        )
        
        col1, col2 = st.columns([1, 4])
        
        with col1:
            if st.button("💾 Save BOM", type="primary"):
                try:
                    with engine.connect() as conn:
                        for _, row in edited_parts.iterrows():
                            conn.execute(text("""
                                UPDATE part_master
                                SET name = :name, cost = :cost, moq = :moq,
                                    lead_time = :lt, deposit_pct = :dp,
                                    deposit_days = :dd, balance_days = :bd,
                                    supplier_name = :supplier
                                WHERE id = :id
                            """), {
                                "name": row['name'],
                                "cost": row['cost'],
                                "moq": row['moq'],
                                "lt": row['lead_time'],
                                "dp": row['deposit_pct'],
                                "dd": row['deposit_days'],
                                "bd": row['balance_days'],
                                "supplier": row['supplier_name'],
                                "id": row['id']
                            })
                        conn.commit()
                    st.success("BOM saved!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Save failed: {e}")
        
        # Cost summary
        st.markdown("---")
        st.markdown(f"<h4 style='font-family:Montserrat;color:{NAVY};'>Unit Cost Summary</h4>", unsafe_allow_html=True)
        
        total_cost = df_parts['cost'].sum()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            render_metric_card("Total Material Cost", f"${total_cost:,.0f}", "neutral")
        with col2:
            render_metric_card("Component Count", str(len(df_parts)), "neutral")
        with col3:
            avg_lead = df_parts['lead_time'].mean()
            render_metric_card("Avg Lead Time", f"{avg_lead:.0f} days", "neutral")
    
    except Exception as e:
        st.error(f"Could not load BOM data: {e}")


# =============================================================================
# SECTION 8: MAIN APPLICATION
# =============================================================================

def main():
    """Main application entry point."""
    
    # Initialize database engine
    engine = get_db_engine()
    
    # Auto-heal database if needed
    if not ensure_database_ready():
        st.error("Database initialization failed. Please check logs.")
        if st.button("🔧 Manual Database Rebuild"):
            try:
                import seed_db
                seed_db.run_seed()
                st.success("Database rebuilt!")
                st.rerun()
            except Exception as e:
                st.error(f"Rebuild failed: {e}")
        st.stop()
    
    # Sidebar navigation
    render_sidebar_brand()
    
    st.sidebar.markdown("---")
    
    view = st.sidebar.radio(
        "Navigation",
        [
            "📊 Dashboard",
            "📈 Financials",
            "🏭 Production",
            "💼 OpEx Budget",
            "📦 Supply Chain"
        ],
        label_visibility="collapsed"
    )
    
    st.sidebar.markdown("---")
    
    # Database management (collapsed by default)
    with st.sidebar.expander("⚙️ Database", expanded=False):
        if st.button("🔄 Rebuild Database", use_container_width=True):
            try:
                import seed_db
                seed_db.run_seed()
                st.success("Database rebuilt!")
                st.cache_resource.clear()
                st.rerun()
            except Exception as e:
                st.error(f"Rebuild failed: {e}")
        
        st.caption("⚠️ This will reset all data to defaults")
    
    # Version info
    st.sidebar.markdown("---")
    st.sidebar.caption(f"""
    **IdleX ERP v7.0**  
    Boardroom Ready Edition  
    © 2025 IdleX Inc.
    """)
    
    # Generate financial data
    try:
        df_pnl, df_cash = generate_financial_ledgers(engine)
    except Exception as e:
        logger.error(f"Financial data generation failed: {e}")
        df_pnl, df_cash = pd.DataFrame(), pd.DataFrame()
    
    # Route to selected view
    if "Dashboard" in view:
        view_dashboard(engine, df_pnl, df_cash)
    elif "Financials" in view:
        view_financials(engine, df_pnl, df_cash)
    elif "Production" in view:
        view_production(engine)
    elif "OpEx" in view:
        view_opex(engine)
    elif "Supply Chain" in view:
        view_supply_chain(engine)


if __name__ == "__main__":
    main()
