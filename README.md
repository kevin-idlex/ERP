# IdleX ERP

Enterprise Resource Planning system for IdleX - 48V eAPU manufacturing.

## Features

- **Dashboard**: Executive overview with cash flow projections
- **Financials**: P&L statements, revenue analysis, margin tracking
- **Production**: Manufacturing schedule, unit manifests
- **Pricing**: MSRP configuration, dealer pricing, revenue impact
- **Channel Mix**: Direct vs dealer sales mix by quarter
- **OpEx**: Staffing plans, general expenses
- **Supply Chain**: Part master, BOM management

## Quick Start

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Seed the database
python seed_db.py

# Run the app
streamlit run dashboard.py
```

### Cloud Run Deployment

```bash
# Build and deploy
gcloud run deploy idlex-erp \
  --source . \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars DATABASE_URL=your-postgres-url
```

## Tech Stack

- **Frontend**: Streamlit
- **Database**: SQLite (local) / PostgreSQL (cloud)
- **Charts**: Plotly
- **ORM**: SQLAlchemy

## File Structure

```
IdleX_ERP/
├── dashboard.py      # Main Streamlit application
├── seed_db.py        # Database seeder with sample data
├── requirements.txt  # Python dependencies
├── Dockerfile        # Container configuration
├── logo_white.png    # IdleX logo (white)
├── logo_blue.png     # IdleX logo (blue)
└── tests/            # Test suite
    ├── test_financials.py
    ├── test_production.py
    └── ...
```

## Configuration

### Pricing (per year)
- 2026: $15,500 MSRP (dealer pays 80%)
- 2027: $13,500 MSRP (dealer pays 80%)
- 2028: $11,500 MSRP (dealer pays 80%)

### Channel Mix (direct sales %)
- Q1 2026: 15% → Q4 2028: 45%

### Funding
- Equity: $1,500,000
- LOC: $4,100,000

## License

Proprietary - IdleX Inc.
