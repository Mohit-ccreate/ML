# Data

- `raw/` - Raw NSE OHLC + Option Chain (gitignored, local only)
- `processed/` - Features (gitignored)

To fetch:
```bash
python src/data/nse_data.py --symbol NIFTY --days 365
python src/data/features.py --symbol NIFTY
```
