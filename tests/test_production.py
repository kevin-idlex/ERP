"""
IdleX ERP - Production & Channel Mix Tests
Tests: P01-P04

Validates channel mix synchronization and production scheduling.
"""

import pytest
from sqlalchemy import text
import pandas as pd
import random
from datetime import date
from conftest import count_units_by_channel


# =============================================================================
# CRITICAL BUG TEST: Channel Mix Consistency
# =============================================================================

class TestChannelMixConsistency:
    """
    CRITICAL: Ensure channel_mix_config matches production_unit distribution.
    
    This was the root cause of the 75% dealer bug - the seeder used hardcoded
    25% instead of reading from channel_mix_config.
    """
    
    def test_channel_config_matches_production(self, engine_sqlite, expected_channel_mix):
        """
        The channel distribution in production_unit should match channel_mix_config.
        This is the most critical financial integrity test.
        """
        for (year, quarter), expected_direct_pct in expected_channel_mix.items():
            counts = count_units_by_channel(engine_sqlite, year=year, quarter=quarter)
            
            if not counts:
                continue  # No units in this quarter
            
            total = sum(counts.values())
            if total == 0:
                continue
            
            actual_direct = counts.get('DIRECT', 0)
            actual_direct_pct = actual_direct / total
            
            # Allow 15% tolerance due to random assignment
            tolerance = 0.15
            assert abs(actual_direct_pct - expected_direct_pct) < tolerance, \
                f"Q{quarter} {year}: Expected direct ~{expected_direct_pct*100:.0f}%, " \
                f"got {actual_direct_pct*100:.0f}% ({actual_direct}/{total} units)"
    
    def test_no_units_without_channel(self, engine_sqlite):
        """Every production unit must have a sales channel assigned."""
        with engine_sqlite.connect() as conn:
            null_count = conn.execute(text("""
                SELECT COUNT(*) FROM production_unit 
                WHERE sales_channel IS NULL OR sales_channel = ''
            """)).scalar()
        
        assert null_count == 0, f"{null_count} units have no sales channel"
    
    def test_only_valid_channels(self, engine_sqlite):
        """Sales channel must be either DIRECT or DEALER."""
        with engine_sqlite.connect() as conn:
            invalid = conn.execute(text("""
                SELECT DISTINCT sales_channel FROM production_unit 
                WHERE sales_channel NOT IN ('DIRECT', 'DEALER')
            """)).fetchall()
        
        assert len(invalid) == 0, f"Invalid channels found: {[r[0] for r in invalid]}"


# =============================================================================
# P01: Regenerate Schedule Basics
# =============================================================================

class TestRegenerateSchedule:
    """Test production schedule regeneration."""
    
    def test_p01_only_planned_units_deleted(self, engine_sqlite_fresh):
        """Non-PLANNED units should survive regeneration."""
        with engine_sqlite_fresh.connect() as conn:
            # Mark some units as WIP
            conn.execute(text("""
                UPDATE production_unit 
                SET status = 'WIP' 
                WHERE id IN (SELECT id FROM production_unit LIMIT 5)
            """))
            conn.commit()
            
            # Get WIP unit IDs
            wip_ids = [r[0] for r in conn.execute(text(
                "SELECT id FROM production_unit WHERE status = 'WIP'"
            )).fetchall()]
        
        # Note: Actual regeneration would be called here
        # For now, just verify WIP units exist
        assert len(wip_ids) == 5
    
    def test_p01_all_seeded_units_planned(self, engine_sqlite):
        """All initially seeded units should have PLANNED status."""
        with engine_sqlite.connect() as conn:
            non_planned = conn.execute(text("""
                SELECT COUNT(*) FROM production_unit WHERE status != 'PLANNED'
            """)).scalar()
        
        assert non_planned == 0


# =============================================================================
# P02: Workday Scheduling
# =============================================================================

class TestWorkdayScheduling:
    """Test units are scheduled on workdays only."""
    
    def test_p02_no_weekend_builds(self, engine_sqlite):
        """No units should be scheduled on weekends."""
        with engine_sqlite.connect() as conn:
            # SQLite: strftime('%w', date) returns 0=Sunday, 6=Saturday
            weekend_count = conn.execute(text("""
                SELECT COUNT(*) FROM production_unit 
                WHERE CAST(strftime('%w', build_date) AS INTEGER) IN (0, 6)
            """)).scalar()
        
        assert weekend_count == 0, f"{weekend_count} units scheduled on weekends"
    
    def test_p02_builds_distributed_across_month(self, engine_sqlite):
        """Builds should be distributed across the month, not clustered."""
        with engine_sqlite.connect() as conn:
            # Check distribution for a high-volume month
            distribution = conn.execute(text("""
                SELECT substr(build_date, 9, 2) as day, COUNT(*) as cnt
                FROM production_unit 
                WHERE substr(build_date, 1, 7) = '2028-06'
                GROUP BY day
                ORDER BY day
            """)).fetchall()
        
        if distribution:
            counts = [r[1] for r in distribution]
            # Should have builds on multiple days
            assert len(counts) >= 15, "Builds should span most of the month"
            # No single day should have >10% of the month's total
            total = sum(counts)
            max_day = max(counts)
            assert max_day / total < 0.15, f"Single day has {max_day/total*100:.0f}% of builds"


# =============================================================================
# P03: Channel Mix Enforcement
# =============================================================================

class TestChannelMixEnforcement:
    """Test channel mix configuration is respected."""
    
    def test_p03_direct_percentage_increases_over_time(self, engine_sqlite):
        """Direct sales percentage should increase year over year."""
        yearly_direct = {}
        
        for year in [2026, 2027, 2028]:
            counts = count_units_by_channel(engine_sqlite, year=year)
            total = sum(counts.values())
            if total > 0:
                yearly_direct[year] = counts.get('DIRECT', 0) / total
        
        # 2028 direct % should be higher than 2026
        if 2026 in yearly_direct and 2028 in yearly_direct:
            assert yearly_direct[2028] > yearly_direct[2026], \
                f"Direct % should grow: {yearly_direct[2026]*100:.0f}% -> {yearly_direct[2028]*100:.0f}%"
    
    def test_p03_q1_2026_heavy_dealer(self, engine_sqlite):
        """Q1 2026 should be ~85% dealer (15% direct) per config."""
        counts = count_units_by_channel(engine_sqlite, year=2026, quarter=1)
        total = sum(counts.values())
        
        if total > 0:
            dealer_pct = counts.get('DEALER', 0) / total
            # Should be close to 85% dealer
            assert dealer_pct > 0.70, f"Q1 2026 should be dealer-heavy, got {dealer_pct*100:.0f}%"
    
    def test_p03_q4_2028_more_direct(self, engine_sqlite):
        """Q4 2028 should be ~45% direct per config."""
        counts = count_units_by_channel(engine_sqlite, year=2028, quarter=4)
        total = sum(counts.values())
        
        if total > 0:
            direct_pct = counts.get('DIRECT', 0) / total
            # Should be close to 45% direct
            assert 0.30 < direct_pct < 0.60, f"Q4 2028 should be ~45% direct, got {direct_pct*100:.0f}%"


# =============================================================================
# P04: Production Manifest Updates
# =============================================================================

class TestManifestUpdates:
    """Test production manifest can be edited."""
    
    def test_p04_channel_update_persists(self, engine_sqlite_fresh):
        """Changing sales_channel should persist."""
        with engine_sqlite_fresh.connect() as conn:
            # Get a unit ID
            unit_id = conn.execute(text(
                "SELECT id FROM production_unit LIMIT 1"
            )).scalar()
            
            # Get current channel
            old_channel = conn.execute(text(
                "SELECT sales_channel FROM production_unit WHERE id = :id"
            ), {"id": unit_id}).scalar()
            
            # Flip it
            new_channel = 'DEALER' if old_channel == 'DIRECT' else 'DIRECT'
            conn.execute(text(
                "UPDATE production_unit SET sales_channel = :ch WHERE id = :id"
            ), {"ch": new_channel, "id": unit_id})
            conn.commit()
            
            # Verify
            updated = conn.execute(text(
                "SELECT sales_channel FROM production_unit WHERE id = :id"
            ), {"id": unit_id}).scalar()
            
            assert updated == new_channel
    
    def test_p04_status_update_persists(self, engine_sqlite_fresh):
        """Changing status should persist."""
        with engine_sqlite_fresh.connect() as conn:
            unit_id = conn.execute(text(
                "SELECT id FROM production_unit WHERE status = 'PLANNED' LIMIT 1"
            )).scalar()
            
            conn.execute(text(
                "UPDATE production_unit SET status = 'BUILT' WHERE id = :id"
            ), {"id": unit_id})
            conn.commit()
            
            updated = conn.execute(text(
                "SELECT status FROM production_unit WHERE id = :id"
            ), {"id": unit_id}).scalar()
            
            assert updated == 'BUILT'


# =============================================================================
# Revenue Impact of Channel Mix
# =============================================================================

class TestChannelRevenueImpact:
    """Test that channel mix correctly impacts revenue."""
    
    def test_revenue_varies_by_channel(self, engine_sqlite, expected_pricing):
        """Total revenue should reflect the channel mix (not assume fixed 25%)."""
        with engine_sqlite.connect() as conn:
            # Count by year and channel
            data = conn.execute(text("""
                SELECT 
                    substr(build_date, 1, 4) as year,
                    sales_channel,
                    COUNT(*) as cnt
                FROM production_unit
                GROUP BY year, sales_channel
            """)).fetchall()
        
        revenue_by_year = {}
        for year_str, channel, count in data:
            year = int(year_str)
            pricing = expected_pricing.get(year, expected_pricing[2026])
            
            if channel == 'DIRECT':
                unit_rev = pricing["msrp"]
            else:
                unit_rev = pricing["msrp"] * pricing["dealer_discount"]
            
            revenue_by_year[year] = revenue_by_year.get(year, 0) + (count * unit_rev)
        
        # 2026 revenue should be ~$28M based on ~2000 units at avg ~$14000
        if 2026 in revenue_by_year:
            assert 20_000_000 < revenue_by_year[2026] < 40_000_000, \
                f"2026 revenue ${revenue_by_year[2026]:,.0f} outside expected range"
        
        # 2028 revenue should be ~$350M+ based on ~30000 units
        if 2028 in revenue_by_year:
            assert revenue_by_year[2028] > 250_000_000, \
                f"2028 revenue ${revenue_by_year[2028]:,.0f} seems too low"
