# IdleX ERP - Financial Integrity Testing Guide
## CFO Board Presentation Validation Checklist

**Version:** 1.0  
**Date:** November 27, 2025  
**Purpose:** Ensure all financial calculations are accurate for investor presentations

---

## 🐛 CRITICAL BUG IDENTIFIED

### Channel Mix Inconsistency (FIXED)

**Bug:** Production Manifest showed 75% dealer while Channel Mix Config showed different percentages.

**Root Cause:** `seed_db.py` used hardcoded 25% direct split instead of reading from `channel_mix_config`.

**Impact:**
- Revenue forecasts using channel config were wrong
- Cash timing predictions incorrect (dealers have 30-day payment lag)
- Margin calculations per channel inaccurate

**Fix Applied:** Modified `seed_db.py` to use quarterly channel mix percentages when creating production units.

---

## 📋 AUTOMATED TEST SUITE

Run the full test suite:
```bash
cd D:\Dropbox\IdleX\IdleX_ERP
python test_financial_integrity.py
```

Run individual tests:
```bash
python test_financial_integrity.py --test channel_mix
python test_financial_integrity.py --test revenue
python test_financial_integrity.py --test bom
```

### Test Categories

| Test | What It Validates | CFO Risk |
|------|-------------------|----------|
| `channel_mix` | Config matches production units | Revenue forecasts |
| `revenue` | MSRP × volume × channel | Top-line accuracy |
| `cash_timing` | Direct=immediate, Dealer=30-day | Cash position |
| `bom` | Material costs per unit | COGS accuracy |
| `opex` | Staffing costs | Operating budget |
| `consistency` | Cross-module data matches | All reports |
| `pricing` | MSRP and margins reasonable | Unit economics |
| `working_capital` | Lead times and deposits | Cash needs |

---

## 🔍 MANUAL VALIDATION CHECKLIST

### Before Every Board Presentation

#### 1. Channel Mix Validation
- [ ] Go to Production & Sales Planning → Channel Mix tab
- [ ] Note the Direct % for each quarter
- [ ] Go to Production Manifest tab
- [ ] Verify Direct/Dealer split in header matches config
- [ ] If mismatch, click "Apply to Production Schedule"

#### 2. Revenue Sanity Check
- [ ] Calculate: Units × MSRP × Channel Mix
- [ ] Compare to P&L "Product Sales" line
- [ ] Variance should be < 1%

**Manual Calculation:**
```
Direct Revenue = Direct Units × $8,500
Dealer Revenue = Dealer Units × $8,500 × 0.75
Total Revenue = Direct Revenue + Dealer Revenue
```

#### 3. Cash Flow Timing
- [ ] Direct sales: Cash received on build date
- [ ] Dealer sales: Cash received 30 days after build
- [ ] High dealer months = lower immediate cash

#### 4. Gross Margin Check
- [ ] Material Cost per Unit: ~$3,800 (from BOM)
- [ ] Direct Price: $8,500 → Margin: 55%
- [ ] Dealer Price: $6,375 → Margin: 40%
- [ ] Blended margin depends on channel mix

#### 5. Unit Economics Validation
| Metric | Expected Range | Red Flag |
|--------|---------------|----------|
| Material Cost | $3,500-$4,500 | < $3,000 or > $5,000 |
| Direct Gross Margin | 50-60% | < 45% |
| Dealer Gross Margin | 35-45% | < 30% |
| OpEx/Unit (at scale) | $800-$1,200 | > $1,500 |

---

## 🚨 KNOWN CALCULATION RISKS

### 1. Cash Waterfall Logic
The cash waterfall has specific rules:
- Revenue pays down LOC before filling cash
- Materials can use LOC if cash insufficient
- OpEx MUST come from cash only

**Validation:** If cash goes negative while LOC available, check if it's OpEx (correct) or materials (bug).

### 2. Pricing Year Boundaries
MSRP changes by year. Units near year-end may use wrong pricing.

**Validation:** Check December units use current year's MSRP, January units use next year's.

### 3. Lead Time Working Capital
Long lead time parts (batteries: 90 days, HVAC: 120 days) require:
- Deposits paid early
- Balance paid on delivery
- This creates cash timing gaps

**Validation:** Production ramp-up months should show higher cash burn.

---

## 📊 KEY METRICS FOR BOARD DECK

### 2026 Projections (Verify These)
- Units: ~2,000
- Revenue: ~$15M
- Gross Margin: 45-50% blended
- OpEx: ~$3M
- EBITDA: Breakeven by Q4

### 2027 Projections
- Units: ~12,000
- Revenue: ~$85M
- Gross Margin: 48-52%
- OpEx: ~$8M
- EBITDA: ~$20M+

### 2028 Projections
- Units: ~30,000
- Revenue: ~$216M
- Gross Margin: 50-55%
- OpEx: ~$15M
- EBITDA: ~$70M+

---

## 🔄 DEPLOYMENT PROCESS

After fixing bugs, redeploy:

```cmd
cd /d D:\Dropbox\IdleX\IdleX_ERP
gcloud builds submit --tag us-east1-docker.pkg.dev/idlex-erp/cloud-run-source-deploy/idlex-erp
gcloud run deploy idlex-erp --image us-east1-docker.pkg.dev/idlex-erp/cloud-run-source-deploy/idlex-erp --region us-east1
```

After deployment, click **Manual Database Rebuild** to regenerate data with fixes.

---

## ✅ SIGN-OFF CHECKLIST

Before presenting to board:

- [ ] All automated tests pass
- [ ] Channel mix matches across modules
- [ ] Revenue calculation verified manually
- [ ] Cash waterfall looks reasonable
- [ ] No negative cash without explanation
- [ ] Gross margins in expected range
- [ ] OpEx scales with revenue (not linear with units)
- [ ] Working capital needs explained for ramp-up

---

## 📞 SUPPORT

If calculations look wrong:
1. Check Cloud Run logs: `gcloud run services logs read idlex-erp --region us-east1 --limit 50`
2. Run test suite locally
3. Compare manual calculations to system output
4. Document variance and root cause
