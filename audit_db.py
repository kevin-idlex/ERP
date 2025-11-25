import pandas as pd
from sqlalchemy import create_engine, text

# Connect
engine = create_engine('sqlite:///idlex.db')

def audit_system():
    print("--- IDLEX DATA STRUCTURE AUDIT ---\n")
    
    with engine.connect() as conn:
        # 1. Check for Duplicate Serials
        dupes = conn.execute(text("SELECT serial_number, COUNT(*) FROM production_unit GROUP BY serial_number HAVING COUNT(*) > 1")).fetchall()
        if dupes:
            print(f"❌ CRITICAL ERROR: Found {len(dupes)} duplicate serial numbers!")
        else:
            print("✅ Serial Number Integrity: OK")

        # 2. Check for Invalid Dates
        null_dates = conn.execute(text("SELECT COUNT(*) FROM production_unit WHERE build_date IS NULL")).scalar()
        if null_dates > 0:
            print(f"❌ DATA ERROR: Found {null_dates} units with no build date.")
        else:
            print("✅ Date Integrity: OK")

        # 3. Check Production Continuity
        # Do we have gaps in serial numbers? (e.g. 1, 2, 4...)
        serials = pd.read_sql("SELECT serial_number FROM production_unit", conn)
        if not serials.empty:
            # Extract number from "IDX-0001"
            serials['num'] = serials['serial_number'].str.extract('(\d+)').astype(float)
            serials = serials.sort_values('num')
            
            min_s = int(serials['num'].min())
            max_s = int(serials['num'].max())
            count = len(serials)
            
            if (max_s - min_s + 1) != count:
                print(f"⚠️ WARNING: Serial Number Gaps Detected. Range: {min_s}-{max_s}, Count: {count}")
                print("   (This happens when we 'Nuke and Pave' the schedule repeatedly)")
            else:
                print("✅ Serial Continuity: Perfect")
        
        # 4. Verify Status
        status_counts = pd.read_sql("SELECT status, COUNT(*) as qty FROM production_unit GROUP BY status", conn)
        print("\nCurrent Status Breakdown:")
        print(status_counts)

if __name__ == "__main__":
    audit_system()