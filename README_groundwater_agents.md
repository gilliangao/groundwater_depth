# Groundwater Multi-Agent Workflow

This project now includes a single script, `groundwater_multi_agent.py`, that splits the work into three agents:

1. `StationDiscoveryAgent`
   - lists the 24 BGS Future Flows groundwater stations
   - saves a station catalog CSV
   - downloads the BGS level/climate zip files and extracts the CSV files

2. `SeasonalAnalysisAgent`
   - reads a station groundwater CSV
   - groups monthly data into seasons (`DJF`, `MAM`, `JJA`, `SON`)
   - removes ensemble outliers with the IQR rule
   - calculates cleaned seasonal means
   - fits seasonal linear regressions
   - saves tables and PNG plots

3. `GroundwaterPredictionAgent`
   - reads the saved regression model JSON
   - predicts groundwater values for a user-selected year
   - returns seasonal predictions and an annual mean

## Station Catalog

A ready-to-use catalog is saved in `bgs_future_flows_stations.csv`.

To print the stations:

```bash
python3 groundwater_multi_agent.py stations
```

To save the catalog again:

```bash
python3 groundwater_multi_agent.py stations --save-csv bgs_future_flows_stations.csv
```

## Download a Station Dataset

```bash
python3 groundwater_multi_agent.py download --station TL33_4 --dataset levels --output-dir downloads/TL33_4
```

`--dataset` can be `levels`, `climate`, or `both`.

## Analyze a Downloaded Station CSV

Use the plotting environment already in this folder:

```bash
venv_plot/bin/python groundwater_multi_agent.py analyze \
  --input-csv FF-RGroundwater-TL33_4-GWL.csv \
  --station TL33_4 \
  --output-dir outputs/TL33_4
```

This creates:

- seasonal means CSV
- cleaned seasonal means CSV
- seasonal trend CSV
- regression model JSON
- combined trend PNG
- one PNG for each season
- predicted seasonal means CSV for the modeled year range

## Predict a Year for One Station

```bash
venv_plot/bin/python groundwater_multi_agent.py predict \
  --model-json outputs/TL33_4/TL33_4_regression_models.json \
  --year 2050
```

## Run Agent 2 and Agent 3 Together

```bash
venv_plot/bin/python groundwater_multi_agent.py run \
  --station TL33_4 \
  --input-csv FF-RGroundwater-TL33_4-GWL.csv \
  --year 2050 \
  --output-dir outputs/TL33_4
```

## Notes

- The built-in station list comes from the BGS Future Flows stations page.
- The downloader assumes the BGS zip naming pattern used by the official site.
- The current prediction model is `linear_regression` for each season.
- If you want, we can extend this next with a small web app so a client can choose a station and year from a form instead of the command line.
