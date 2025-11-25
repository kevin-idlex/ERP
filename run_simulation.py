import pandas as pd
from sqlalchemy import create_engine
from datetime import timedelta

# CONNECT TO YOUR DATABASE
engine = create_engine('sqlite:///idlex.db')

print("\n--- IDLEX CASH FLOW SIMULATION ---")

# 1. LOAD DATA FROM DATABASE
# We use pandas to read SQL tables directly into easy-to-use lists (DataFrames)
df_units = pd.read_sql("SELECT * FROM production_unit", engine)
df_parts = pd.read_sql("SELECT * FROM part_master", engine)
df_bom = pd.read_sql("SELECT * FROM bom_items", engine)
config = pd.read_sql("SELECT * FROM global_config", engine)

# Convert dates to datetime objects so we can do math
df_units['build_date'] = pd.to_datetime(df_units['build_date'])
start_cash = float(config[config['setting_key']=='start_cash']['setting_value'].values[0])

# 2. CALCULATE CASH IN (Revenue)
# Logic: Direct = Paid on Build Day. Dealer = Paid 30 Days Later.
ledger = []
msrp = 8500.00
dealer_discount = 0.25

print(f"Simulating Revenue for {len(df_units)} units...")

for idx, unit in df_units.iterrows():
    if unit['sales_channel'] == 'DIRECT':
        amount = msrp
        date = unit['build_date']
    else:
        amount = msrp * (1 - dealer_discount)
        date = unit['build_date'] + timedelta(days=30)
        
    ledger.append({
        "Date": date,
        "Category": "Revenue",
        "Amount": amount,
        "Note": f"Unit {unit['serial_number']}"
    })

# 3. CALCULATE CASH OUT (Purchasing)
# Logic: Group all demand by month, buy in batch on the 1st, pay deposit.
print("Simulating Supply Chain Orders...")

# A. Calculate total demand for every part
# We assume we order for the whole month's build on the 1st of that month
monthly_builds = df_units.groupby(pd.Grouper(key='build_date', freq='MS')).size()

for month_start, unit_count in monthly_builds.items():
    if unit_count == 0: continue
    
    # Needs to be delivered by the 1st of the build month
    delivery_deadline = month_start 
    
    for idx, part in df_parts.iterrows():
        # How many do we need? (Qty per unit * Unit Count)
        # Find BOM qty for this part
        bom_entry = df_bom[df_bom['part_id'] == part['id']]
        if bom_entry.empty: continue
        qty_needed = bom_entry.iloc[0]['qty_per_unit'] * unit_count
        
        # Calculate Order Date
        lead_time = int(part['lead_time'])
        order_date = delivery_deadline - timedelta(days=lead_time)
        
        total_cost = qty_needed * part['cost']
        
        # EVENT 1: DEPOSIT
        if part['deposit_pct'] > 0:
            dep_amount = total_cost * part['deposit_pct']
            # Deposit Date = Order Date + deposit_days (usually negative)
            # Actually, per your rules: deposit_days is usually negative relative to delivery
            # Let's stick to the simpler math: Pay Deposit ON Order Date
            ledger.append({
                "Date": order_date,
                "Category": "Material Deposit",
                "Amount": -dep_amount,
                "Note": f"Dep: {part['name']} ({qty_needed} units)"
            })
            
        # EVENT 2: BALANCE
        rem_pct = 1.0 - part['deposit_pct']
        if rem_pct > 0:
            bal_amount = total_cost * rem_pct
            # Balance Date = Delivery Date + balance_days
            bal_date = delivery_deadline + timedelta(days=int(part['balance_days']))
            ledger.append({
                "Date": bal_date,
                "Category": "Material Balance",
                "Amount": -bal_amount,
                "Note": f"Bal: {part['name']}"
            })

# 4. CRUNCH THE NUMBERS
df_cash = pd.DataFrame(ledger)
df_cash = df_cash.sort_values('Date')

# Running Balance
df_cash['Cash_Balance'] = df_cash['Amount'].cumsum() + start_cash

# 5. PRINT THE REPORT
print("\n" + "="*40)
print("FINAL RESULTS")
print("="*40)

lowest_point = df_cash['Cash_Balance'].min()
lowest_date = df_cash.loc[df_cash['Cash_Balance'].idxmin()]['Date']

print(f"Starting Cash: ${start_cash:,.2f}")
print(f"Lowest Cash Point: ${lowest_point:,.2f} on {lowest_date.date()}")

if lowest_point < 0:
    print(f"WARNING: YOU NEED A LOC OF ${abs(lowest_point):,.2f}")
else:
    print("Result: You never run out of cash.")

print("\n--- CRITICAL MONTHLY SUMMARY ---")
# Resample to Monthly buckets
df_cash.set_index('Date', inplace=True)
monthly_summary = df_cash.resample('ME')['Amount'].sum()
monthly_balance = df_cash.resample('ME')['Cash_Balance'].last()

summary_df = pd.concat([monthly_summary, monthly_balance], axis=1)
summary_df.columns = ['Net Flow', 'End Balance']

pd.options.display.float_format = '${:,.2f}'.format
print(summary_df)