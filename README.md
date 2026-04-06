# groundwater_depth

Groundwater forecasting and seasonal groundwater-depth prediction using BGS Future Flows groundwater datasets.

This repository includes:

- scripts for seasonal groundwater analysis
- a multi-agent workflow for station discovery, cleaning, plotting, and prediction
- generated CSV tables and PNG figures
- a lightweight browser app for selecting a station and year

## Main Files

- `groundwater_multi_agent.py`: the main 3-agent workflow
- `groundwater_web_app.py`: dependency-light local web app
- `README_groundwater_agents.md`: detailed command-line usage guide
- `FF-RGroundwater-TL33_4-GWL.csv`: example BGS groundwater input file

## The 3 Agents

1. Station discovery agent
- lists the BGS Future Flows groundwater stations
- saves a station catalog CSV
- downloads station datasets from BGS

2. Seasonal analysis agent
- groups monthly values into `DJF`, `MAM`, `JJA`, `SON`
- removes outliers from ensemble members
- calculates cleaned seasonal means
- fits seasonal linear regression models
- saves tables and plots

3. Prediction agent
- accepts a station and target year
- predicts seasonal groundwater depth
- returns an annual mean across seasons

## Quick Start

### Command line workflow

List the known stations:

```bash
python3 groundwater_multi_agent.py stations
```

Analyze a local station CSV:

```bash
venv_plot/bin/python groundwater_multi_agent.py analyze \
  --input-csv FF-RGroundwater-TL33_4-GWL.csv \
  --station TL33_4 \
  --output-dir outputs/TL33_4
```

Predict groundwater depth for a selected year:

```bash
venv_plot/bin/python groundwater_multi_agent.py predict \
  --model-json outputs/TL33_4/TL33_4_regression_models.json \
  --year 2050
```

### Local web app

Run the browser app:

```bash
venv_plot/bin/python groundwater_web_app.py
```

Then open:

```text
http://127.0.0.1:8000
```

The web app lets a client:

- choose a station
- upload a groundwater CSV directly
- enter a year
- view predicted groundwater depth for `DJF`, `MAM`, `JJA`, `SON`
- view the annual mean
- download the prediction result as a CSV file

## Current Example Output

The repository already contains generated outputs for `TL33_4` (Therfield Rectory) in `outputs/TL33_4/`.

## Data Source

British Geological Survey Future Flows groundwater level forecasting:

- https://www.bgs.ac.uk/geology-projects/environmental-modelling/groundwater-level-forecasting/
- legacy station list: https://www2.bgs.ac.uk/groundwater/change/FutureFlows/sites.html

## Notes

- `venv_plot/` is intentionally excluded from git because it is a local virtual environment.
- The current predictive method is seasonal linear regression.
- Additional models such as polynomial or robust regression can be added later.
