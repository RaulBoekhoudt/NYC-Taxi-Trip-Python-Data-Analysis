# NYC Yellow Taxi — Revenue Drivers & Anomaly Monitoring (Python)

![Project overview](Images/NYC_Taxi.jpg)

## Overview
This repository contains a **Python analytics portfolio project** built around a realistic business question:

> **What drives taxi revenue by time and location — and how can we quickly detect and investigate unusual days?**

The work starts with careful **EDA + data validation**, then progresses into an analyst-style workflow:
**KPI layer → revenue driver analysis (day / hour / zone) → anomaly flags → drill-down → recommendations**.

---

## What’s included
- **Notebook:** `EDA_Analysis.ipynb`
- **Helper modules:** `data_prep.py`, `eda.py` (reusable EDA + validation utilities)
- **Dataset:** `2017_Yellow_Taxi_Trip_Data.csv` (Yellow Taxi trip records extract)
- **Images:** `Images/` (screenshots referenced in this README)

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

---

## How to run locally
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

pip install pandas numpy matplotlib seaborn jupyter
jupyter notebook
```

Open `EDA_Analysis.ipynb` and run all cells.

---

## Project goals (what this analysis is designed to answer)
1) **When** does revenue happen?
- Hour-of-day peaks and day-of-week patterns

2) **Where** does revenue happen?
- Zone concentration and “top contributor” pickup areas

3) **What explains higher-value trips?**
- Relationship between revenue and distance/duration/payment behaviour

4) **Which days are unusual, and why?**
- A simple, explainable anomaly flagging method plus drill-down views

---

## EDA techniques used (and how they’re applied)
This project is intentionally **not** “just plotting graphs”. The EDA phase follows an analyst workflow: validate → understand structure → check distributions → isolate outliers → test relationships → summarise.

### 1) Structure & sanity checks (fast triage)
The notebook uses a compact “first look” checklist to establish:
- dataset shape and sample records
- data types (dates vs numeric vs categorical)
- a quick scan for missing values and obvious parsing issues

**Why it matters:** reviewers expect you to confirm you understand the dataset *before* drawing conclusions.

---

### 2) Automatic feature type detection (categorical vs numeric)
Instead of manually guessing column types, the notebook programmatically splits fields into:
- **categorical**
- **numeric**
- **categorical-but-cardinal** (high-uniqueness IDs)

This uses threshold-based logic (`cat_th`, `car_th`) and then corrects edge cases where numeric-coded categories exist (e.g., `passenger_count` can look categorical depending on frequency).

**Why it matters:** it avoids common mistakes like treating IDs as numeric features or treating “small integer codes” as continuous measures.

---

### 3) Categorical EDA (frequency + ratio + visual diagnostics)
For each categorical column, the EDA produces:
- frequency tables (counts and ratios)
- plots for quick distribution scanning (e.g., payment types, vendor mix)

**Why it matters:** categorical distributions often reveal data quality issues (rare codes, unexpected categories) and operational patterns (behaviour differences by payment type/vendor).

---

### 4) Numerical EDA (descriptive stats + distribution shape)
For each numeric column, the notebook generates:
- descriptive statistics (mean/median/std, min/max, quantiles)
- histograms to understand shape (skew, spikes, truncation)
- box plots to visualise spread and extreme values


**Why it matters:** taxi data is naturally heavy-tailed (distance, totals, tips). Good EDA shows you’ve looked for skew/outliers instead of trusting averages blindly.

---

### 5) Outlier identification (IQR-based thresholds)
The notebook uses an **IQR / quantile threshold approach** to identify columns that contain outliers:
- compute lower/upper bounds from Q1/Q3 (configurable)
- flag variables with outliers via a reusable check function

**Why it matters:** this is a practical, explainable method used routinely in business data cleaning (and it’s more robust than naïve z-scores on skewed distributions).

---

### 6) Pairwise relationship exploration (multivariate EDA)
To move beyond single-variable charts, the notebook uses:
- a **pairplot** for quick multivariate scanning of numeric relationships
- targeted plots to explore specific behaviours (tips by vendor, mean tips by passenger count)


**Why it matters:** strong portfolio EDA includes at least one multivariate step to test plausible drivers (distance/duration/tip/revenue patterns).

---

### 7) Early “business metric” framing during EDA
Even before the formal KPI layer, the notebook introduces revenue-oriented views:
- total revenue by **day of week**
- total revenue by **month**

**Why it matters:** it keeps the EDA anchored to the project’s end goal (revenue drivers), instead of becoming exploratory-but-directionless.

---

## Revenue drivers & anomaly monitoring (portfolio core)
The final section turns EDA into a repeatable analysis you could plausibly deliver to stakeholders.

### 1) KPI layer (daily, hourly, zone aggregates)
Builds analysis-ready tables for:
- trips, revenue, average revenue per trip
- summaries of distance and duration
- time features (date, hour, weekday)
- pickup-zone rollups

---

### 2) Revenue drivers (time patterns)
#### By hour of day
Identifies peak revenue windows and whether they’re driven by higher trip volume or higher value per trip.


#### By day of week
Surfaces weekday vs weekend differences and supports staffing/supply planning.


---

### 3) Revenue drivers (zone concentration)
Quantifies how much revenue is concentrated in a small number of pickup zones.


---

### 4) Explaining higher-value trips
Uses interpretable signals:
- distance and duration vs revenue
- payment type mix and its implications (e.g., tip capture differences)


---

### 5) Anomaly detection (robust and explainable)
Implements a lightweight monitoring step using a **rolling median + MAD “modified z-score”**:
- flags unusual days for both revenue and trip counts
- avoids overreacting to outliers compared with mean/std on skewed data
- supports quick drill-down (hourly + zone breakdown)


---

## What this project demonstrates (skills signal)
- **Practical EDA**: automated type detection, distribution analysis, multivariate checks
- **Data quality thinking**: sanity checks + outlier identification (not just pretty charts)
- **Business framing**: revenue-focused KPIs and driver analysis
- **Analyst workflow**: anomaly flags + drill-down for investigation
- **Communication**: conclusions and recommendations written from observed evidence

---

## Example recommendations (based on patterns surfaced)
- **Shift planning:** align staffing and vehicle availability to peak hourly windows rather than relying on daily averages.
- **Zone focus:** prioritise operations around the small set of pickup zones that contribute a disproportionate share of revenue.
- **Monitoring process:** when an anomaly is flagged, drill down by hour and pickup zone to distinguish:
  - city-wide demand shocks (e.g., weather/events),
  - localised disruptions (closures/incidents),
  - or data quality issues.

---

## Tech stack
- Python (pandas, NumPy)
- Matplotlib / Seaborn
- Jupyter Notebook
