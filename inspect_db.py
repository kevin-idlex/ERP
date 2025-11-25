import pandas as pd
from sqlalchemy import create_engine, text

# Connect
engine = create_engine('sqlite:///idlex.db')

def analyze_database():
    print("="*50)
    print("   IDLEX DATABASE DIAGNOSTIC REPORT")
    print("="*50)
    
    with engine.connect() as conn:
        # 1. PRODUCTION SCHEDULE ANALYSIS
        print("\n[1] PRODUCTION SCHEDULE")
        try:
            units = pd.read_sql("SELECT * FROM production_unit", conn)
            if units.empty:
                print("❌ TABLE IS EMPTY")
            else:
                print(f"✅ Total Units Scheduled: {len(units)}")
                
                # Breakdown by Year
                units['year'] = pd.to_datetime(units['build_date']).dt.year
                by_year = units.groupby('year').size()
                print("\n   Volume by Year:")
                print(by_year.to_string())
                
                # Breakdown by Status
                print("\n   Status Breakdown:")
                print(units['status'].value_counts().to_string())
                
                # Check for dates
                earliest = units['build_date'].min()
                latest = units['build_date'].max()
                print(f"\n   Timeline: {earliest} to {latest}")
        except Exception as e:
            print(f"❌ Error reading table: {e}")

        # 2. OPEX & STAFFING ANALYSIS
        print("\n" + "-"*50)
        print("[2] OPEX & STAFFING")
        try:
            roles = pd.read_sql("SELECT * FROM opex_roles", conn)
            staff = pd.read_sql("SELECT * FROM opex_staffing_plan", conn)
            expenses = pd.read_sql("SELECT * FROM opex_general_expenses", conn)
            
            print(f"✅ Roles Defined: {len(roles)} (CEO, CTO, etc.)")
            print(f"✅ Monthly Headcount Entries: {len(staff)}")
            print(f"✅ R&D/SG&A Budget Entries: {len(expenses)}")
            
            if not expenses.empty:
                total_rnd = expenses[expenses['amount'] > 0]['amount'].sum()
                print(f"   Total Non-Payroll Budget Loaded: ${total_rnd:,.0f}")
        except Exception as e:
            print(f"❌ Error reading OpEx tables: {e}")

        # 3. BOM ANALYSIS
        print("\n" + "-"*50)
        print("[3] SUPPLY CHAIN (BOM)")
        try:
            parts = pd.read_sql("SELECT * FROM part_master", conn)
            bom = pd.read_sql("SELECT * FROM bom_items", conn)
            
            print(f"✅ Unique Parts in Master: {len(parts)}")
            print(f"✅ BOM Links Created: {len(bom)}")
            
            # Check for expensive items
            expensive = parts[parts['cost'] > 500]
            print(f"\n   High Value Items (> $500):")
            for _, row in expensive.iterrows():
                print(f"   - {row['name']}: ${row['cost']:,.2f} (Deposit: {row['deposit_pct']*100}%)")
                
        except Exception as e:
            print(f"❌ Error reading BOM: {e}")

    print("\n" + "="*50)
    print("DIAGNOSTIC COMPLETE")

if __name__ == "__main__":
    analyze_database()