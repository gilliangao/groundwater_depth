"""
Groundwater multi-agent workflow using the OpenAI Agents SDK.

Mirrors the functionality of groundwater_multi_agent.py but orchestrates
three specialised agents via the OpenAI Agents SDK:

  1. StationDiscovery  – lists BGS Future Flows stations, refreshes from
                         the BGS website, saves the station catalog, and
                         downloads + extracts zip datasets.
  2. SeasonalAnalysis  – reads a groundwater CSV, removes ensemble outliers,
                         builds seasonal means, fits linear regressions, and
                         writes CSVs / plots.
  3. Prediction        – applies a previously saved regression model JSON to
                         produce groundwater-level forecasts for a given year.

Usage:
    python groundwater_openai_agents.py stations [--save-csv PATH] [--refresh-from-web]
    python groundwater_openai_agents.py download --station TL33_4 [--dataset levels|climate|both] [--output-dir PATH]
    python groundwater_openai_agents.py analyze --input-csv PATH [--station CODE] [--output-dir PATH]
    python groundwater_openai_agents.py predict --model-json PATH --year YEAR [--output-csv PATH]
    python groundwater_openai_agents.py run --station CODE --input-csv PATH --year YEAR [--output-dir PATH]

Environment:
    OPENAI_API_KEY – required, set in .env (loaded automatically).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import tempfile
import urllib.request
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from dotenv import load_dotenv

load_dotenv()

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "tygron-mpl-cache")
)

try:
    import matplotlib
except ModuleNotFoundError:
    matplotlib = None

plt = None

try:
    import numpy as np
except ModuleNotFoundError:
    np = None

try:
    from scipy import stats
except ModuleNotFoundError:
    stats = None

from agents import Agent, Runner, function_tool  # openai-agents


# ---------------------------------------------------------------------------
# Constants (identical to groundwater_multi_agent.py)
# ---------------------------------------------------------------------------

BGS_SITES_URL = "https://www2.bgs.ac.uk/groundwater/change/FutureFlows/sites.html"
BGS_DOWNLOAD_BASE = "https://www2.bgs.ac.uk/groundwater/downloads/hydrogeology"
SEASONS = ("DJF", "MAM", "JJA", "SON")
SEASON_LABELS = {
    "DJF": "Winter (DJF)",
    "MAM": "Spring (MAM)",
    "JJA": "Summer (JJA)",
    "SON": "Autumn (SON)",
}
SEASON_COLORS = {
    "DJF": "#2d6a4f",
    "MAM": "#1b4965",
    "JJA": "#c1121f",
    "SON": "#9c6644",
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StationRecord:
    dataset_code: str
    wellmaster_id: str
    location: str
    aquifer: str
    grid_reference: str
    _levels_zip: str = ""
    _climate_zip: str = ""

    @property
    def levels_zip_name(self) -> str:
        return self._levels_zip or f"FutureFlowsHydrology-{self.dataset_code}-GWL.zip"

    @property
    def climate_zip_name(self) -> str:
        return self._climate_zip or f"FutureFlowsClimate-{self.dataset_code}.zip"

    @property
    def levels_url(self) -> str:
        return f"{BGS_DOWNLOAD_BASE}/{self.levels_zip_name}"

    @property
    def climate_url(self) -> str:
        return f"{BGS_DOWNLOAD_BASE}/{self.climate_zip_name}"


@dataclass
class SeasonalRegressionResult:
    season: str
    slope: float
    intercept: float
    r_squared: float
    p_value: float
    std_error: float


def _s(code: str, wid: str, loc: str, aq: str, grid: str,
        lzip: str = "", czip: str = "") -> StationRecord:
    return StationRecord(code, wid, loc, aq, grid, lzip, czip)


BGS_STATIONS: Tuple[StationRecord, ...] = (
    _s("SY68_34",  "SY68/34",  "Ashton Farm",              "Chalk",                      "3661 0880"),
    _s("TA10_63",  "TA10/63",  "Aylesby",                  "Chalk",                      "5194 4071"),
    _s("SU81_1",   "SU81/1",   "Chilgrove House",          "Chalk",                      "4835 1143"),
    _s("SU34_8D",  "SU34/8D",  "Clanville Lodge Gate",     "Chalk",                      "4322 1490",
       "FutureFlowsHydrology-SU34_8d-GWL.zip"),
    _s("SE94_5",   "SE94/5",   "Dalton Holme",             "Chalk",                      "4965 4453"),
    _s("ST88_62A", "ST88/62A", "Didmarton 1",              "Inferior Oolite",            "3827 1874",
       "FutureFlowsHydrology-ST88_62a-GWL.zip"),
    _s("SD27_6B",  "SD27/6B",  "Furness Abbey",            "Permo-Triassic Sandstone",   "3216 4717",
       "FF-ZOOMQ3D-SpatialChange-GWL.zip"),
    _s("TL89_37",  "TL89/37",  "Grimes Graves",            "Chalk",                      "5817 2900"),
    _s("SJ62_112", "SJ62/112", "Heathlanes",               "Permo-Triassic Sandstone",   "3619 3210"),
    _s("SK17_13",  "SK17/13",  "Hucklow South",            "Carboniferous Limestone",    "4177 3775"),
    _s("TR14_9",   "TR14/9",   "Little Bucket Farm",       "Chalk",                      "6122 1469"),
    _s("SJ15_13",  "SJ15/13",  "Llanfair Dyffryn Clwyd",  "Permo-Triassic Sandstone",   "3137 3555"),
    _s("TQ41_82",  "TQ41/82",  "Lower Barn Cottage",       "Lower Greensand",            "5437 1132"),
    _s("TF03_37",  "TF03/37",  "New Red Lion",             "Lincolnshire Limestone",     "5088 3303"),
    _s("NX97_2",   "NX97/2",   "Newbridge",                "Permo-Triassic Sandstone",   "2951 5788"),
    _s("SU17_57",  "SU17/57",  "Rockley",                  "Chalk",                      "4165 1717"),
    _s("NY63_2",   "NY63/2",   "Skirwith",                 "Permo-Triassic Sandstone",   "3613 5325"),
    _s("SU78_45A", "SU78/45A", "Stonor Park",              "Chalk",                      "4741 1892",
       "FutureFlowsHydrology-SU78_45a-GWL.zip"),
    _s("NZ21_29",  "NZ21/29",  "Swan House",               "Magnesian Limestone",        "4252 5199"),
    _s("TL33_4",   "TL33/4",   "Therfield Rectory",        "Chalk",                      "5333 2372"),
    _s("TF81_2A",  "TF81/2A",  "Washpit Farm",             "Chalk",                      "5813 3196",
       "FutureFlowsHydrology-TF81_2a-GWL.zip"),
    _s("TQ25_13",  "TQ25/13",  "Well House Inn",           "Chalk",                      "5258 1552"),
    _s("TV59_7C",  "TV59/7C",  "West Dean No. 3",          "Chalk",                      "5529 0992",
       "FutureFlowsHydrology-TV59_7c-GWL.zip"),
    _s("SU01_5B",  "SU01/5B",  "West Woodyates Manor",    "Chalk",                      "4016 1194",
       "FutureFlowsHydrology-SU01_5b-GWL.zip"),
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _get_station(station_code: str, stations: Tuple[StationRecord, ...] = BGS_STATIONS) -> StationRecord:
    lookup = station_code.strip().upper().replace("/", "_")
    for s in stations:
        if s.dataset_code.upper() == lookup:
            return s
    raise KeyError(f"Unknown station '{station_code}'.")


def _extract_zip(archive_path: Path, output_dir: Path) -> List[Path]:
    extracted: List[Path] = []
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.namelist():
            out = output_dir / Path(member).name
            with archive.open(member) as src, out.open("wb") as dst:
                dst.write(src.read())
            extracted.append(out)
    return extracted


def _validate_csv(file_path: Path) -> None:
    if file_path.suffix.lower() != ".csv":
        return
    with file_path.open("r", newline="", encoding="utf-8-sig") as fh:
        headers = next(csv.reader(fh))
    if "Date" not in headers or len(headers) < 4:
        raise ValueError(f"{file_path} does not look like a BGS groundwater ensemble CSV.")


def _season_for_month(month: int) -> str:
    if month in (12, 1, 2):
        return "DJF"
    if month in (3, 4, 5):
        return "MAM"
    if month in (6, 7, 8):
        return "JJA"
    return "SON"


def _remove_outliers(values: Sequence[float]) -> List[float]:
    if len(values) < 4:
        return list(values)
    q1 = float(np.percentile(values, 25))
    q3 = float(np.percentile(values, 75))
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return [v for v in values if lo <= v <= hi]


def _require_deps(needs_plot: bool = False) -> None:
    missing = []
    if np is None:
        missing.append("numpy")
    if stats is None:
        missing.append("scipy")
    if needs_plot and matplotlib is None:
        missing.append("matplotlib")
    if missing:
        raise ModuleNotFoundError(f"Missing: {', '.join(missing)}.")


def _pyplot():
    global plt
    _require_deps(needs_plot=True)
    if plt is None:
        matplotlib.use("Agg")
        import matplotlib.pyplot as _plt
        plt = _plt
    return plt


# ---------------------------------------------------------------------------
# Tool functions – Station Discovery Agent
# ---------------------------------------------------------------------------

@function_tool
def list_stations() -> str:
    """Return a CSV-formatted list of all known BGS Future Flows groundwater stations."""
    lines = ["dataset_code,wellmaster_id,location,aquifer,grid_reference"]
    for s in BGS_STATIONS:
        lines.append(f"{s.dataset_code},{s.wellmaster_id},{s.location},{s.aquifer},{s.grid_reference}")
    return "\n".join(lines)


@function_tool
def refresh_stations_from_website(url: str = BGS_SITES_URL) -> str:
    """Fetch the BGS Future Flows stations page and parse the station list. Returns CSV."""
    with urllib.request.urlopen(url, timeout=30) as resp:
        html = resp.read().decode("utf-8", errors="ignore")

    pattern = re.compile(
        r"(?P<wellmaster>[A-Z]{1,2}\d{2}/[0-9A-Z]+)\s+"
        r"(?P<grid>\d{4}\s+\d{4})\s+"
        r"(?P<location>[^<\n]+?)\s+"
        r"(?P<aquifer>Chalk|Inferior Oolite|Permo-Triassic Sandstone|Carboniferous Limestone|"
        r"Lower Greensand|Lincolnshire Limestone|Magnesian Limestone)",
        re.IGNORECASE,
    )

    lines = ["dataset_code,wellmaster_id,location,aquifer,grid_reference"]
    seen: set[str] = set()
    for m in pattern.finditer(html):
        wid = m.group("wellmaster")
        code = wid.replace("/", "_")
        if code in seen:
            continue
        seen.add(code)
        loc = " ".join(m.group("location").split())
        grid = " ".join(m.group("grid").split())
        lines.append(f"{code},{wid},{loc},{m.group('aquifer')},{grid}")

    if len(lines) == 1:
        raise RuntimeError("Could not parse the BGS stations page – structure may have changed.")
    return "\n".join(lines)


@function_tool
def save_station_catalog(output_csv_path: str) -> str:
    """Save the built-in BGS station catalog to a CSV file. Returns the saved path."""
    out = Path(output_csv_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["dataset_code", "wellmaster_id", "location", "aquifer",
                         "grid_reference", "levels_url", "climate_url"],
        )
        writer.writeheader()
        for s in BGS_STATIONS:
            row = asdict(s)
            row["levels_url"] = s.levels_url
            row["climate_url"] = s.climate_url
            writer.writerow(row)
    return str(out)


@function_tool
def download_station_dataset(station_code: str, output_dir: str, dataset: str = "levels") -> str:
    """
    Download BGS Future Flows zip file(s) for a station and extract CSVs.

    dataset: 'levels', 'climate', or 'both'.
    Returns a newline-separated list of extracted file paths.
    """
    station = _get_station(station_code)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    targets: List[Tuple[str, str]] = []
    if dataset in {"levels", "both"}:
        targets.append((station.levels_url, station.levels_zip_name))
    if dataset in {"climate", "both"}:
        targets.append((station.climate_url, station.climate_zip_name))

    extracted_paths: List[str] = []
    for url, archive_name in targets:
        archive_path = out / archive_name
        urllib.request.urlretrieve(url, archive_path)
        for ep in _extract_zip(archive_path, out):
            _validate_csv(ep)
            extracted_paths.append(str(ep))

    return "\n".join(extracted_paths)


# ---------------------------------------------------------------------------
# Tool functions – Seasonal Analysis Agent
# ---------------------------------------------------------------------------

@function_tool
def analyze_groundwater_csv(
    input_csv_path: str,
    output_dir: str,
    station_code: str = "",
    start_year: int = 2000,
    end_year: int = 2080,
) -> str:
    """
    Read a BGS groundwater ensemble CSV, remove outliers, compute seasonal means,
    fit linear regressions, and write output CSVs and PNG plots.

    Returns JSON with paths to all output files.
    """
    _require_deps(needs_plot=True)

    input_csv = Path(input_csv_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- read CSV ---
    rows: List[Dict[str, Any]] = []
    ensemble_cols: List[str] = []
    inferred_code: Optional[str] = None

    with input_csv.open("r", newline="", encoding="utf-8-sig") as fh:
        reader = csv.reader(fh)
        headers = next(reader)
        if len(headers) < 4:
            raise ValueError("Expected at least 4 columns in the groundwater CSV.")
        ensemble_cols = headers[3:]
        for raw in reader:
            if not raw:
                continue
            date = datetime.strptime(raw[2], "%d/%m/%Y")
            if not (start_year <= date.year <= end_year):
                continue
            inferred_code = inferred_code or raw[0]
            rows.append({"date": date, "values": [float(v) for v in raw[3:]]})

    if not rows:
        raise ValueError(f"No rows in {input_csv} for {start_year}-{end_year}.")

    code = (station_code or (inferred_code or "").replace("/", "_") or input_csv.stem)

    # --- seasonal means ---
    seasonal_values: Dict[Tuple[int, str], List[List[float]]] = {}
    for row in rows:
        season = _season_for_month(row["date"].month)
        key = (row["date"].year, season)
        seasonal_values.setdefault(key, [[] for _ in ensemble_cols])
        for i, v in enumerate(row["values"]):
            seasonal_values[key][i].append(v)

    seasonal_means: Dict[Tuple[int, str], List[float]] = {}
    for key, per_ens in seasonal_values.items():
        seasonal_means[key] = [float(np.mean(s)) for s in per_ens]
    seasonal_means = dict(sorted(seasonal_means.items()))

    # --- outlier removal & per-season series ---
    cleaned: Dict[str, Dict[str, List[float]]] = {s: {"years": [], "means": []} for s in SEASONS}
    for (year, season), values in seasonal_means.items():
        filtered = _remove_outliers(values)
        if not filtered:
            continue
        cleaned[season]["years"].append(year)
        cleaned[season]["means"].append(float(np.mean(filtered)))

    # --- regressions ---
    regressions: Dict[str, SeasonalRegressionResult] = {}
    for season in SEASONS:
        years_arr = np.asarray(cleaned[season]["years"], dtype=float)
        means_arr = np.asarray(cleaned[season]["means"], dtype=float)
        if len(years_arr) < 2:
            raise ValueError(f"Not enough data for season {season}.")
        slope, intercept, r_val, p_val, std_err = stats.linregress(years_arr, means_arr)
        regressions[season] = SeasonalRegressionResult(
            season=season, slope=float(slope), intercept=float(intercept),
            r_squared=float(r_val ** 2), p_value=float(p_val), std_error=float(std_err),
        )

    # --- write CSVs ---
    seasonal_means_csv = out_dir / f"{code}_seasonal_means.csv"
    cleaned_means_csv = out_dir / f"{code}_seasonal_cleaned_means.csv"
    trends_csv = out_dir / f"{code}_seasonal_trends.csv"
    predictions_csv = out_dir / f"{code}_predicted_seasonal_means.csv"
    models_json = out_dir / f"{code}_regression_models.json"
    combined_plot = out_dir / f"{code}_seasonal_trends.png"

    with seasonal_means_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["Year", "Season", *ensemble_cols])
        for (year, season), vals in seasonal_means.items():
            w.writerow([year, season, *vals])

    with cleaned_means_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["Year", "Season", "Mean_After_Outlier_Removal"])
        w.writeheader()
        for season in SEASONS:
            for year, mean in zip(cleaned[season]["years"], cleaned[season]["means"]):
                w.writerow({"Year": year, "Season": season, "Mean_After_Outlier_Removal": mean})

    with trends_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["Season", "Slope", "Intercept", "R_squared", "P_value", "Std_Error"])
        w.writeheader()
        for season in SEASONS:
            r = regressions[season]
            w.writerow({"Season": r.season, "Slope": r.slope, "Intercept": r.intercept,
                         "R_squared": r.r_squared, "P_value": r.p_value, "Std_Error": r.std_error})

    all_years = sorted({y for s in SEASONS for y in cleaned[s]["years"]})
    with predictions_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["Year", *SEASONS, "Annual_Mean"])
        w.writeheader()
        for year in all_years:
            row = {"Year": year}
            season_vals = []
            for season in SEASONS:
                pred = regressions[season].slope * year + regressions[season].intercept
                row[season] = pred
                season_vals.append(pred)
            row["Annual_Mean"] = float(np.mean(season_vals))
            w.writerow(row)

    models_json.write_text(json.dumps({
        "station_code": code,
        "source_csv": str(input_csv),
        "year_range": {"start": start_year, "end": end_year},
        "algorithm": "linear_regression",
        "seasons": {s: asdict(r) for s, r in regressions.items()},
    }, indent=2), encoding="utf-8")

    # --- plots ---
    plot = _pyplot()
    plot.figure(figsize=(12, 8))
    for season in SEASONS:
        years_arr = np.asarray(cleaned[season]["years"], dtype=float)
        means_arr = np.asarray(cleaned[season]["means"], dtype=float)
        color = SEASON_COLORS[season]
        model = regressions[season]
        trend = model.slope * years_arr + model.intercept
        plot.scatter(years_arr, means_arr, color=color, alpha=0.65, s=18, label=f"{SEASON_LABELS[season]} data")
        plot.plot(years_arr, trend, color=color, linewidth=2,
                  label=f"{SEASON_LABELS[season]} trend (R²={model.r_squared:.3f})")
    plot.xlabel("Year")
    plot.ylabel("Groundwater level mean")
    plot.title("Seasonal groundwater trends after ensemble outlier removal")
    plot.grid(True, alpha=0.3)
    plot.legend()
    plot.tight_layout()
    plot.savefig(combined_plot, dpi=300)
    plot.close()

    for season in SEASONS:
        years_arr = np.asarray(cleaned[season]["years"], dtype=float)
        means_arr = np.asarray(cleaned[season]["means"], dtype=float)
        model = regressions[season]
        trend = model.slope * years_arr + model.intercept
        color = SEASON_COLORS[season]
        plot.figure(figsize=(10, 6))
        plot.scatter(years_arr, means_arr, color=color, alpha=0.7, s=24, label="Cleaned seasonal mean")
        plot.plot(years_arr, trend, color="#111111", linewidth=2, label="Linear regression")
        plot.xlabel("Year")
        plot.ylabel("Groundwater level mean")
        plot.title(f"{code} {SEASON_LABELS[season]} groundwater trend")
        plot.grid(True, alpha=0.3)
        plot.legend()
        plot.tight_layout()
        plot.savefig(out_dir / f"{code}_{season}_trend.png", dpi=300)
        plot.close()

    return json.dumps({
        "seasonal_means_csv": str(seasonal_means_csv),
        "cleaned_means_csv": str(cleaned_means_csv),
        "trends_csv": str(trends_csv),
        "predictions_csv": str(predictions_csv),
        "models_json": str(models_json),
        "combined_plot": str(combined_plot),
    }, indent=2)


# ---------------------------------------------------------------------------
# Tool functions – Prediction Agent
# ---------------------------------------------------------------------------

@function_tool
def predict_groundwater_level(model_json_path: str, year: int) -> str:
    """
    Apply a saved regression model JSON to predict seasonal groundwater levels for a year.
    Returns a JSON object with predicted values per season and the annual mean.
    """
    payload = json.loads(Path(model_json_path).read_text(encoding="utf-8"))
    seasons = payload["seasons"]
    prediction: Dict[str, Any] = {
        "Year": year,
        "Station": payload["station_code"],
        "Algorithm": payload["algorithm"],
    }
    season_values: List[float] = []
    for season in SEASONS:
        value = seasons[season]["slope"] * year + seasons[season]["intercept"]
        prediction[season] = value
        season_values.append(value)
    prediction["Annual_Mean"] = float(sum(season_values) / len(season_values))
    return json.dumps(prediction, indent=2)


@function_tool
def save_prediction_csv(prediction_json: str, output_csv_path: str) -> str:
    """Save the prediction dict (JSON string) to a CSV file. Returns the saved path."""
    prediction = json.loads(prediction_json)
    out = Path(output_csv_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(prediction.keys()))
        writer.writeheader()
        writer.writerow(prediction)
    return str(out)


# ---------------------------------------------------------------------------
# Agent definitions
# ---------------------------------------------------------------------------

station_discovery_agent = Agent(
    name="StationDiscoveryAgent",
    instructions=(
        "You are a BGS Future Flows groundwater station discovery agent. "
        "You can list all known stations, refresh that list from the BGS website, "
        "save the catalog to CSV, and download + extract dataset zip files for a given station. "
        "Always confirm the station code exists before attempting a download."
    ),
    tools=[list_stations, refresh_stations_from_website, save_station_catalog, download_station_dataset],
)

seasonal_analysis_agent = Agent(
    name="SeasonalAnalysisAgent",
    instructions=(
        "You are a groundwater seasonal analysis agent. "
        "Given a BGS groundwater ensemble CSV file, you remove outliers across the ensemble, "
        "compute seasonal (DJF/MAM/JJA/SON) means, fit linear regression models per season, "
        "and produce output CSVs and PNG plots. "
        "Always tell the user where the output files were saved."
    ),
    tools=[analyze_groundwater_csv],
)

prediction_agent = Agent(
    name="PredictionAgent",
    instructions=(
        "You are a groundwater prediction agent. "
        "Given a regression model JSON file produced by the SeasonalAnalysisAgent and a target year, "
        "you predict seasonal groundwater levels and the annual mean. "
        "Optionally save the prediction to a CSV file."
    ),
    tools=[predict_groundwater_level, save_prediction_csv],
)


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def _run_agent(agent: Agent, prompt: str) -> str:
    result = Runner.run_sync(agent, prompt)
    return result.final_output


def cmd_stations(args: argparse.Namespace) -> int:
    parts = ["List all BGS Future Flows groundwater stations."]
    if args.refresh_from_web:
        parts = ["Refresh the station list from the BGS website and show results."]
    if args.save_csv:
        parts.append(f"Then save the catalog to {args.save_csv}.")
    print(_run_agent(station_discovery_agent, " ".join(parts)))
    return 0


def cmd_download(args: argparse.Namespace) -> int:
    prompt = (
        f"Download the {args.dataset} dataset for station {args.station} "
        f"into the directory {args.output_dir}. "
        "List the extracted files when done."
    )
    print(_run_agent(station_discovery_agent, prompt))
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    station_hint = f" Use station code {args.station}." if args.station else ""
    prompt = (
        f"Analyze the groundwater CSV at {args.input_csv}."
        f"{station_hint} "
        f"Save outputs to {args.output_dir}. "
        f"Year range: {args.start_year} to {args.end_year}."
    )
    print(_run_agent(seasonal_analysis_agent, prompt))
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    parts = [f"Predict groundwater levels for year {args.year} using the model at {args.model_json}."]
    if args.output_csv:
        parts.append(f"Save the prediction CSV to {args.output_csv}.")
    print(_run_agent(prediction_agent, " ".join(parts)))
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    out_dir = Path(args.output_dir) / args.station
    out_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: analyze
    analyze_prompt = (
        f"Analyze the groundwater CSV at {args.input_csv} for station {args.station}. "
        f"Save outputs to {out_dir}."
    )
    analysis_output = _run_agent(seasonal_analysis_agent, analyze_prompt)
    print("=== Analysis ===")
    print(analysis_output)

    # Extract model JSON path from agent output (look for _regression_models.json)
    model_match = re.search(r"[\w./\-]+_regression_models\.json", analysis_output)
    if not model_match:
        model_json = out_dir / f"{args.station}_regression_models.json"
    else:
        model_json = Path(model_match.group(0))

    # Step 2: predict
    pred_csv = out_dir / f"{args.station}_prediction_{args.year}.csv"
    predict_prompt = (
        f"Predict groundwater levels for year {args.year} using the model at {model_json}. "
        f"Save the prediction CSV to {pred_csv}."
    )
    prediction_output = _run_agent(prediction_agent, predict_prompt)
    print("\n=== Prediction ===")
    print(prediction_output)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="OpenAI Agents SDK groundwater workflow for BGS Future Flows station discovery, "
                    "seasonal analysis, and prediction."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("stations", help="List BGS Future Flows stations.")
    sp.add_argument("--save-csv", type=Path)
    sp.add_argument("--refresh-from-web", action="store_true")

    dp = sub.add_parser("download", help="Download dataset zip and extract CSVs for a station.")
    dp.add_argument("--station", required=True)
    dp.add_argument("--dataset", choices=["levels", "climate", "both"], default="levels")
    dp.add_argument("--output-dir", type=Path, default=Path("downloads"))

    ap = sub.add_parser("analyze", help="Seasonal analysis: outlier removal, regression, plots.")
    ap.add_argument("--input-csv", type=Path, required=True)
    ap.add_argument("--station")
    ap.add_argument("--output-dir", type=Path, default=Path("outputs"))
    ap.add_argument("--start-year", type=int, default=2000)
    ap.add_argument("--end-year", type=int, default=2080)

    pp = sub.add_parser("predict", help="Predict groundwater level for a target year.")
    pp.add_argument("--model-json", type=Path, required=True)
    pp.add_argument("--year", type=int, required=True)
    pp.add_argument("--output-csv", type=Path)

    rp = sub.add_parser("run", help="Run analysis + prediction together.")
    rp.add_argument("--station", required=True)
    rp.add_argument("--input-csv", type=Path, required=True)
    rp.add_argument("--year", type=int, required=True)
    rp.add_argument("--output-dir", type=Path, default=Path("outputs"))

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    dispatch = {
        "stations": cmd_stations,
        "download": cmd_download,
        "analyze": cmd_analyze,
        "predict": cmd_predict,
        "run": cmd_run,
    }
    return dispatch[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
