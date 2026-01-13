# NYC Yellow Taxi — Revenue Drivers & Anomaly Monitoring (Python)


## Overview
This repository contains a **Python analytics portfolio project** built around a realistic business question:

> **What drives taxi revenue by time and location — and how can we quickly detect and investigate unusual days?**

The work starts with careful **EDA + data validation**, then progresses into an analyst-style workflow:
**KPI layer → revenue driver analysis (day / hour / zone) → anomaly flags → drill-down → recommendations**.

---

## What’s included
- **Notebook:** `EDA_Analysis.ipynb`
- **Utility modules:** `data_prep.py`, `eda.py` (reusable EDA + validation utilities)
- **Dataset:** `2017_Yellow_Taxi_Trip_Data.csv` (Yellow Taxi trip records extract)

---

## Data (high level)
NYC TLC Yellow Taxi trip records include operational and financial fields such as:
- pickup / drop-off timestamps
- taxi zones (`PULocationID`, `DOLocationID`)
- trip distance, passenger count
- itemised charges (fare, surcharges, tolls, tips) and `total_amount`
- payment type and vendor identifiers

Important nuance: **`tip_amount` is recorded for card tips** (cash tips are not captured), so tipping analysis should be interpreted accordingly.

Reference:
- TLC Trip Record Data: https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page
- Yellow Taxi data dictionary: https://www.nyc.gov/assets/tlc/downloads/pdf/data_dictionary_trip_records_yellow.pdf

