"""
IdleX ERP - Schema & Seeding Tests (seed_db.py)
Tests: S01-S08

These tests guarantee the seed data matches the intended business model.
"""

import pytest
from sqlalchemy import text, inspect
import pandas as pd


# =============================================================================
# S01: Core Tables Exist with Expected Columns
# =============================================================================

class TestSchemaExistence:
    """Verify all required tables and columns exist."""
    
    REQUIRED_TABLES = [
        "global_config",
        "part_master",
        "bom_items", 
        "production_unit",
        "opex_roles",
        "opex_staffing_plan",
        "opex_general_expenses",
        "pricing_config",
        "channel_mix_config",
        "inventory_balance",
        "work_center",
        "routing_step",
        "work_center_assignment",
        "covenant_config",
        "fleet",
        "unit_fleet_assignment",
        "warranty_policy",
        "unit_warranty_event",
        "service_plan",
        "unit_service_subscription",
        "audit_log",
        "external_data_import",
        "scenario_header",
        "scenario_growth_profile",
        "scenario_cash_timeseries",
    ]
    
    def test_s01_all_tables_exist(self, engine_sqlite):
        """All core tables must exist after seeding."""
        inspector = inspect(engine_sqlite)
        existing_tables = inspector.get_table_names()
        
        for table in self.REQUIRED_TABLES:
            assert table in existing_tables, f"Missing table: {table}"
    
    def test_s01_production_unit_columns(self, engine_sqlite):
        """production_unit must have required columns."""
        inspector = inspect(engine_sqlite)
        columns = {col['name'] for col in inspector.get_columns('production_unit')}
        
        required = {'id', 'serial_number', 'build_date', 'sales_channel', 'status'}
        assert required.issubset(columns), f"Missing columns: {required - columns}"
    
    def test_s01_part_master_columns(self, engine_sqlite):
        """part_master must have cost and lead time columns."""
        inspector = inspect(engine_sqlite)
        columns = {col['name'] for col in inspector.get_columns('part_master')}
        
        required = {'id', 'sku', 'name', 'cost', 'lead_time', 'moq', 
                   'deposit_pct', 'deposit_days', 'balance_days'}
        assert required.issubset(columns), f"Missing columns: {required - columns}"


# =============================================================================
# S02: Global Config Values
# =============================================================================

class TestGlobalConfig:
    """Verify critical global configuration values."""
    
    def test_s02_start_cash(self, engine_sqlite):
        """Starting cash should be $1.6M."""
        with engine_sqlite.connect() as conn:
            result = conn.execute(text(
                "SELECT setting_value FROM global_config WHERE setting_key = 'start_cash'"
            )).fetchone()
        
        assert result is not None, "start_cash not found"
        assert result[0] == "1600000", f"Expected 1600000, got {result[0]}"
    
    def test_s02_loc_limit(self, engine_sqlite):
        """LOC limit should be $500K."""
        with engine_sqlite.connect() as conn:
            result = conn.execute(text(
                "SELECT setting_value FROM global_config WHERE setting_key = 'loc_limit'"
            )).fetchone()
        
        assert result is not None, "loc_limit not found"
        assert result[0] == "500000", f"Expected 500000, got {result[0]}"
    
    def test_s02_msrp_price(self, engine_sqlite):
        """Default MSRP should be $8,500."""
        with engine_sqlite.connect() as conn:
            result = conn.execute(text(
                "SELECT setting_value FROM global_config WHERE setting_key = 'msrp_price'"
            )).fetchone()
        
        assert result is not None, "msrp_price not found"
        assert result[0] == "8500", f"Expected 8500, got {result[0]}"
    
    def test_s02_dealer_discount(self, engine_sqlite):
        """Default dealer discount should be 0.75 (75% of MSRP)."""
        with engine_sqlite.connect() as conn:
            result = conn.execute(text(
                "SELECT setting_value FROM global_config WHERE setting_key = 'dealer_discount'"
            )).fetchone()
        
        assert result is not None, "dealer_discount not found"
        assert result[0] == "0.75", f"Expected 0.75, got {result[0]}"


# =============================================================================
# S03: BOM Cost Calculation
# =============================================================================

class TestBOMCost:
    """Verify BOM and material cost calculations."""
    
    def test_s03_bom_items_exist(self, engine_sqlite):
        """BOM should have items linked to parts."""
        with engine_sqlite.connect() as conn:
            count = conn.execute(text(
                "SELECT COUNT(*) FROM bom_items"
            )).scalar()
        
        assert count > 0, "No BOM items found"
    
    def test_s03_bom_cost_calculation(self, engine_sqlite):
        """Unit material cost = sum(cost × qty_per_unit)."""
        with engine_sqlite.connect() as conn:
            result = conn.execute(text("""
                SELECT SUM(p.cost * b.qty_per_unit) as total_cost
                FROM bom_items b
                JOIN part_master p ON b.part_id = p.id
            """)).scalar()
        
        assert result is not None, "Could not calculate BOM cost"
        assert 3000 < result < 5000, f"Unit cost ${result:,.2f} outside expected range $3,000-$5,000"
    
    def test_s03_key_parts_qty(self, engine_sqlite):
        """Verify quantities for key parts."""
        with engine_sqlite.connect() as conn:
            # Battery should be qty 3
            battery = conn.execute(text("""
                SELECT b.qty_per_unit 
                FROM bom_items b 
                JOIN part_master p ON b.part_id = p.id 
                WHERE p.sku = 'BAT-SS-48V'
            """)).scalar()
            
            assert battery == 3, f"Battery qty should be 3, got {battery}"
            
            # DCDC should be qty 1
            dcdc = conn.execute(text("""
                SELECT b.qty_per_unit 
                FROM bom_items b 
                JOIN part_master p ON b.part_id = p.id 
                WHERE p.sku = 'DCDC-SCOTTY'
            """)).scalar()
            
            assert dcdc == 1, f"DCDC qty should be 1, got {dcdc}"


# =============================================================================
# S04-S05: Production Plan Counts
# =============================================================================

class TestProductionPlan:
    """Verify production unit counts match plan."""
    
    def test_s04_units_have_required_fields(self, engine_sqlite):
        """All units should have build_date, sales_channel, status."""
        with engine_sqlite.connect() as conn:
            null_count = conn.execute(text("""
                SELECT COUNT(*) FROM production_unit 
                WHERE build_date IS NULL 
                   OR sales_channel IS NULL 
                   OR status IS NULL
            """)).scalar()
        
        assert null_count == 0, f"{null_count} units have NULL required fields"
    
    def test_s04_all_status_planned(self, engine_sqlite):
        """All seeded units should have status PLANNED."""
        with engine_sqlite.connect() as conn:
            non_planned = conn.execute(text("""
                SELECT COUNT(*) FROM production_unit WHERE status != 'PLANNED'
            """)).scalar()
        
        assert non_planned == 0, f"{non_planned} units have non-PLANNED status"
    
    def test_s05_total_units_2026(self, engine_sqlite, expected_unit_counts):
        """2026 should have ~2,187 units."""
        with engine_sqlite.connect() as conn:
            count = conn.execute(text("""
                SELECT COUNT(*) FROM production_unit 
                WHERE substr(build_date, 1, 4) = '2026'
            """)).scalar()
        
        # Allow 5% tolerance for workday distribution variations
        expected = expected_unit_counts[2026]
        assert abs(count - expected) / expected < 0.05, \
            f"2026 units: expected ~{expected}, got {count}"
    
    def test_s05_total_units_2027(self, engine_sqlite, expected_unit_counts):
        """2027 should have ~11,322 units."""
        with engine_sqlite.connect() as conn:
            count = conn.execute(text("""
                SELECT COUNT(*) FROM production_unit 
                WHERE substr(build_date, 1, 4) = '2027'
            """)).scalar()
        
        expected = expected_unit_counts[2027]
        assert abs(count - expected) / expected < 0.05, \
            f"2027 units: expected ~{expected}, got {count}"
    
    def test_s05_total_units_2028(self, engine_sqlite, expected_unit_counts):
        """2028 should have ~31,351 units."""
        with engine_sqlite.connect() as conn:
            count = conn.execute(text("""
                SELECT COUNT(*) FROM production_unit 
                WHERE substr(build_date, 1, 4) = '2028'
            """)).scalar()
        
        expected = expected_unit_counts[2028]
        assert abs(count - expected) / expected < 0.05, \
            f"2028 units: expected ~{expected}, got {count}"
    
    def test_s05_total_units_all_years(self, engine_sqlite, expected_unit_counts):
        """Total units across all years should be ~44,860."""
        with engine_sqlite.connect() as conn:
            count = conn.execute(text(
                "SELECT COUNT(*) FROM production_unit"
            )).scalar()
        
        expected = expected_unit_counts["total"]
        assert abs(count - expected) / expected < 0.05, \
            f"Total units: expected ~{expected}, got {count}"


# =============================================================================
# S06: Staffing Plan
# =============================================================================

class TestStaffingPlan:
    """Verify staffing plan is correctly seeded."""
    
    def test_s06_staffing_months_exist(self, engine_sqlite):
        """Should have 36 months of staffing data (Jan 2026 - Dec 2028)."""
        with engine_sqlite.connect() as conn:
            months = conn.execute(text("""
                SELECT DISTINCT month FROM opex_staffing_plan ORDER BY month
            """)).fetchall()
        
        assert len(months) == 36, f"Expected 36 months, got {len(months)}"
        
        # Check first and last
        assert months[0][0] == "2026-01-01", f"First month should be 2026-01-01"
        assert months[-1][0] == "2028-12-01", f"Last month should be 2028-12-01"
    
    def test_s06_ceo_headcount_constant(self, engine_sqlite):
        """CEO headcount should be 1 for all months."""
        with engine_sqlite.connect() as conn:
            # Get CEO role id
            ceo_id = conn.execute(text(
                "SELECT id FROM opex_roles WHERE role_name = 'CEO'"
            )).scalar()
            
            if ceo_id:
                headcounts = conn.execute(text("""
                    SELECT headcount FROM opex_staffing_plan 
                    WHERE role_id = :ceo_id
                """), {"ceo_id": ceo_id}).fetchall()
                
                for hc in headcounts:
                    assert hc[0] == 1, f"CEO headcount should be 1, got {hc[0]}"
    
    def test_s06_no_assemblers(self, engine_sqlite):
        """Assembler roles should have 0 headcount (outsourced manufacturing)."""
        with engine_sqlite.connect() as conn:
            assembler_hc = conn.execute(text("""
                SELECT SUM(s.headcount) 
                FROM opex_staffing_plan s
                JOIN opex_roles r ON s.role_id = r.id
                WHERE r.role_name LIKE '%Assembler%'
            """)).scalar()
        
        assert assembler_hc == 0 or assembler_hc is None, \
            f"Assembler headcount should be 0, got {assembler_hc}"


# =============================================================================
# S07: Pricing Config
# =============================================================================

class TestPricingConfig:
    """Verify pricing configuration."""
    
    def test_s07_pricing_2026(self, engine_sqlite, expected_pricing):
        """2026 pricing: MSRP=$8,500, dealer=75%."""
        with engine_sqlite.connect() as conn:
            row = conn.execute(text("""
                SELECT msrp, dealer_discount_pct FROM pricing_config WHERE year = 2026
            """)).fetchone()
        
        assert row is not None, "2026 pricing not found"
        assert row[0] == expected_pricing[2026]["msrp"]
        assert row[1] == expected_pricing[2026]["dealer_discount"]
    
    def test_s07_pricing_2028(self, engine_sqlite, expected_pricing):
        """2028 pricing: MSRP=$8,750, dealer=77%."""
        with engine_sqlite.connect() as conn:
            row = conn.execute(text("""
                SELECT msrp, dealer_discount_pct FROM pricing_config WHERE year = 2028
            """)).fetchone()
        
        assert row is not None, "2028 pricing not found"
        assert row[0] == expected_pricing[2028]["msrp"]
        assert row[1] == expected_pricing[2028]["dealer_discount"]


# =============================================================================
# S08: Channel Mix Config
# =============================================================================

class TestChannelMixConfig:
    """Verify channel mix configuration."""
    
    def test_s08_channel_mix_exists(self, engine_sqlite):
        """Should have 12 quarters of channel mix (2026-2028)."""
        with engine_sqlite.connect() as conn:
            count = conn.execute(text(
                "SELECT COUNT(*) FROM channel_mix_config"
            )).scalar()
        
        assert count == 12, f"Expected 12 quarters, got {count}"
    
    def test_s08_channel_mix_values(self, engine_sqlite, expected_channel_mix):
        """Channel mix values should match seed data."""
        with engine_sqlite.connect() as conn:
            rows = conn.execute(text("""
                SELECT year, quarter, direct_pct FROM channel_mix_config
            """)).fetchall()
        
        for year, quarter, direct_pct in rows:
            expected = expected_channel_mix.get((year, quarter))
            if expected is not None:
                assert abs(direct_pct - expected) < 0.01, \
                    f"Q{quarter} {year}: expected {expected}, got {direct_pct}"
    
    def test_s08_channel_mix_progression(self, engine_sqlite):
        """Direct % should generally increase over time."""
        with engine_sqlite.connect() as conn:
            rows = conn.execute(text("""
                SELECT year, quarter, direct_pct FROM channel_mix_config
                ORDER BY year, quarter
            """)).fetchall()
        
        # First quarter should have lower direct % than last
        first_direct = rows[0][2]
        last_direct = rows[-1][2]
        
        assert last_direct > first_direct, \
            f"Direct % should grow: {first_direct} -> {last_direct}"
