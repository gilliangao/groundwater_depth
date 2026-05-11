"""
Bulk ingest of all BGS Future Flows groundwater datasets.

Scrapes the BGS website for the exact download URLs (preserving the
case-sensitive zip filenames the server uses), downloads every station's
levels zip, extracts the CSV, runs the seasonal analysis pipeline, and
writes all outputs into outputs/<station_code>/.

Also produces a master summary CSV: outputs/all_stations_summary.csv

Usage:
    python3 groundwater_bulk_ingest.py [options]

Options:
    --output-dir PATH        Root directory for all outputs  [default: outputs]
    --download-dir PATH      Where to cache downloaded zips  [default: downloads]
    --start-year INT         [default: 2000]
    --end-year   INT         [default: 2080]
    --workers INT            Parallel download threads       [default: 4]
    --skip-existing          Skip stations whose model JSON already exists
    --dry-run                Parse the website and print what would be downloaded
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import tempfile
import threading
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "tygron-mpl-cache"))

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_PLOT = True
except ModuleNotFoundError:
    plt = None
    HAS_PLOT = False

try:
    import numpy as np
    HAS_NUMPY = True
except ModuleNotFoundError:
    np = None
    HAS_NUMPY = False

try:
    from scipy import stats
    HAS_SCIPY = True
except ModuleNotFoundError:
    stats = None
    HAS_SCIPY = False


BGS_SITES_URL = "https://www2.bgs.ac.uk/groundwater/change/FutureFlows/sites.html"
BGS_DOWNLOAD_BASE = "https://www2.bgs.ac.uk/groundwater/downloads/hydrogeology"
SEASONS = ("DJF", "MAM", "JJA", "SON")
SEASON_LABELS = {"DJF": "Winter (DJF)", "MAM": "Spring (MAM)", "JJA": "Summer (JJA)", "SON": "Autumn (SON)"}
SEASON_COLORS = {"DJF": "#2d6a4f", "MAM": "#1b4965", "JJA": "#c1121f", "SON": "#9c6644"}

_print_lock = threading.Lock()


def log(msg: str) -> None:
    with _print_lock:
        print(msg, flush=True)


# ---------------------------------------------------------------------------
# Website scraping — reads exact zip filenames directly from the HTML
# ---------------------------------------------------------------------------

@dataclass
class ScrapedStation:
    """A station record built directly from what the BGS page says."""
    dataset_code: str          # e.g. TL33_4
    wellmaster_id: str         # e.g. TL33/4
    location: str              # e.g. Therfield Rectory
    aquifer: str
    grid_reference: str
    levels_zip: str            # exact filename the server uses
    climate_zip: str


def scrape_all_stations(url: str = BGS_SITES_URL, timeout: int = 30) -> List[ScrapedStation]:
    """
    Parse the BGS Future Flows sites page and return every station with its
    exact (case-preserving) zip filenames.
    """
    log(f"Fetching station list from {url} …")
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        html = resp.read().decode("utf-8", errors="ignore")

    # Each row: wellmaster | grid | <a>location</a> | aquifer | <a href=…>Levels</a> | <a href=…>Climate</a>
    row_pat = re.compile(
        r"<tr>\s*"
        r"<td>([^<]+)</td>\s*"          # wellmaster_id
        r"<td>([^<]+)</td>\s*"          # grid_reference
        r"<td><a[^>]*>([^<]+)</a></td>\s*"  # location
        r"<td>([^<]+)</td>\s*"          # aquifer
        r"<td><a href=\"([^\"]+)\">Levels</a></td>\s*"   # levels href
        r"<td><a href=\"([^\"]+)\">Climate</a></td>",
        re.DOTALL,
    )

    stations: List[ScrapedStation] = []
    for m in row_pat.finditer(html):
        wellmaster = m.group(1).strip()
        grid = m.group(2).strip()
        location = m.group(3).strip()
        aquifer = m.group(4).strip()
        levels_zip = m.group(5).split("/")[-1]
        climate_zip = m.group(6).split("/")[-1]
        dataset_code = wellmaster.replace("/", "_")
        stations.append(ScrapedStation(
            dataset_code=dataset_code,
            wellmaster_id=wellmaster,
            location=location,
            aquifer=aquifer,
            grid_reference=grid,
            levels_zip=levels_zip,
            climate_zip=climate_zip,
        ))

    if not stations:
        raise RuntimeError(
            "Could not parse any stations from the BGS page. "
            "The page structure may have changed."
        )

    log(f"Found {len(stations)} stations on the BGS website.")
    return stations


# ---------------------------------------------------------------------------
# Download + extraction
# ---------------------------------------------------------------------------

def download_and_extract(
    station: ScrapedStation,
    download_dir: Path,
    skip_if_exists: bool = True,
) -> Optional[Path]:
    """
    Download the levels zip for a station, extract the GWL CSV, and return
    its path.  Returns None if the download or extraction fails.
    """
    zip_path = download_dir / station.levels_zip
    url = f"{BGS_DOWNLOAD_BASE}/{station.levels_zip}"

    if skip_if_exists and zip_path.exists():
        log(f"  [{station.dataset_code}] zip already cached, skipping download.")
    else:
        log(f"  [{station.dataset_code}] Downloading {url} …")
        try:
            urllib.request.urlretrieve(url, zip_path)
        except Exception as exc:
            log(f"  [{station.dataset_code}] DOWNLOAD FAILED: {exc}")
            return None

    # Extract the time-series GWL CSV (the one without UKCP09 in its name)
    try:
        with zipfile.ZipFile(zip_path) as zf:
            members = zf.namelist()
            # Prefer the plain time-series CSV: ends with -GWL.csv but NOT UKCP09
            gwl_members = [
                m for m in members
                if m.lower().endswith("-gwl.csv") and "ukcp09" not in m.lower()
            ]
            # Fall back to any CSV if the preferred form isn't found
            if not gwl_members:
                gwl_members = [m for m in members if m.lower().endswith(".csv")]
            if not gwl_members:
                log(f"  [{station.dataset_code}] No CSV found inside zip.")
                return None
            member = gwl_members[0]
            out_path = download_dir / Path(member).name
            with zf.open(member) as src, out_path.open("wb") as dst:
                dst.write(src.read())
            log(f"  [{station.dataset_code}] Extracted → {out_path.name}")
            return out_path
    except Exception as exc:
        log(f"  [{station.dataset_code}] EXTRACTION FAILED: {exc}")
        return None


# ---------------------------------------------------------------------------
# Seasonal analysis (self-contained — no dependency on groundwater_multi_agent)
# ---------------------------------------------------------------------------

@dataclass
class RegressionResult:
    season: str
    slope: float
    intercept: float
    r_squared: float
    p_value: float
    std_error: float


def _season_for_month(month: int) -> str:
    if month in (12, 1, 2):
        return "DJF"
    if month in (3, 4, 5):
        return "MAM"
    if month in (6, 7, 8):
        return "JJA"
    return "SON"


def _iqr_filter(values: Sequence[float]) -> List[float]:
    if len(values) < 4:
        return list(values)
    q1 = float(np.percentile(values, 25))
    q3 = float(np.percentile(values, 75))
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return [v for v in values if lo <= v <= hi]


def analyze_station(
    input_csv: Path,
    output_dir: Path,
    station_code: str,
    start_year: int,
    end_year: int,
) -> Optional[Dict[str, Path]]:
    """
    Full seasonal analysis for one station.  Returns a dict of output paths
    or None on error.
    """
    if not (HAS_NUMPY and HAS_SCIPY):
        log(f"  [{station_code}] Skipping analysis: numpy/scipy not available.")
        return None

    output_dir.mkdir(parents=True, exist_ok=True)

    # --- read CSV ---
    rows: List[Dict] = []
    ensemble_cols: List[str] = []
    try:
        with input_csv.open("r", newline="", encoding="utf-8-sig") as fh:
            reader = csv.reader(fh)
            headers = next(reader)
            if len(headers) < 4:
                log(f"  [{station_code}] Unexpected CSV format, skipping.")
                return None
            ensemble_cols = headers[3:]
            for raw in reader:
                if not raw:
                    continue
                try:
                    date = datetime.strptime(raw[2], "%d/%m/%Y")
                except ValueError:
                    continue
                if not (start_year <= date.year <= end_year):
                    continue
                rows.append({"date": date, "values": [float(v) for v in raw[3:]]})
    except Exception as exc:
        log(f"  [{station_code}] CSV read error: {exc}")
        return None

    if not rows:
        log(f"  [{station_code}] No data rows for {start_year}-{end_year}, skipping.")
        return None

    # --- seasonal bucket → means ---
    buckets: Dict[Tuple[int, str], List[List[float]]] = {}
    for row in rows:
        season = _season_for_month(row["date"].month)
        key = (row["date"].year, season)
        buckets.setdefault(key, [[] for _ in ensemble_cols])
        for i, v in enumerate(row["values"]):
            buckets[key][i].append(v)

    seasonal_means: Dict[Tuple[int, str], List[float]] = dict(sorted(
        {k: [float(np.mean(col)) for col in cols] for k, cols in buckets.items()}.items()
    ))

    # --- outlier removal & per-season series ---
    cleaned: Dict[str, Dict[str, List[float]]] = {s: {"years": [], "means": []} for s in SEASONS}
    for (year, season), values in seasonal_means.items():
        filtered = _iqr_filter(values)
        if not filtered:
            continue
        cleaned[season]["years"].append(year)
        cleaned[season]["means"].append(float(np.mean(filtered)))

    # --- regressions ---
    regressions: Dict[str, RegressionResult] = {}
    for season in SEASONS:
        years_arr = np.asarray(cleaned[season]["years"], dtype=float)
        means_arr = np.asarray(cleaned[season]["means"], dtype=float)
        if len(years_arr) < 2:
            log(f"  [{station_code}] Not enough data for season {season}, skipping regression.")
            return None
        slope, intercept, r_val, p_val, std_err = stats.linregress(years_arr, means_arr)
        regressions[season] = RegressionResult(
            season=season, slope=float(slope), intercept=float(intercept),
            r_squared=float(r_val ** 2), p_value=float(p_val), std_error=float(std_err),
        )

    # --- write outputs ---
    def _w(name: str) -> Path:
        return output_dir / f"{station_code}_{name}"

    seasonal_means_csv = _w("seasonal_means.csv")
    cleaned_means_csv  = _w("seasonal_cleaned_means.csv")
    trends_csv         = _w("seasonal_trends.csv")
    predictions_csv    = _w("predicted_seasonal_means.csv")
    models_json        = _w("regression_models.json")
    combined_plot      = _w("seasonal_trends.png")

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
            row: Dict = {"Year": year}
            season_vals = []
            for season in SEASONS:
                pred = regressions[season].slope * year + regressions[season].intercept
                row[season] = round(pred, 6)
                season_vals.append(pred)
            row["Annual_Mean"] = round(float(np.mean(season_vals)), 6)
            w.writerow(row)

    models_json.write_text(json.dumps({
        "station_code": station_code,
        "source_csv": str(input_csv),
        "year_range": {"start": start_year, "end": end_year},
        "algorithm": "linear_regression",
        "seasons": {
            s: {"season": r.season, "slope": r.slope, "intercept": r.intercept,
                "r_squared": r.r_squared, "p_value": r.p_value, "std_error": r.std_error}
            for s, r in regressions.items()
        },
    }, indent=2), encoding="utf-8")

    # --- plots ---
    if HAS_PLOT:
        try:
            fig, ax = plt.subplots(figsize=(12, 8))
            for season in SEASONS:
                years_arr = np.asarray(cleaned[season]["years"], dtype=float)
                means_arr = np.asarray(cleaned[season]["means"], dtype=float)
                color = SEASON_COLORS[season]
                model = regressions[season]
                trend = model.slope * years_arr + model.intercept
                ax.scatter(years_arr, means_arr, color=color, alpha=0.65, s=18,
                           label=f"{SEASON_LABELS[season]} data")
                ax.plot(years_arr, trend, color=color, linewidth=2,
                        label=f"{SEASON_LABELS[season]} trend (R²={model.r_squared:.3f})")
            ax.set_xlabel("Year")
            ax.set_ylabel("Groundwater level mean")
            ax.set_title(f"{station_code} – seasonal groundwater trends")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(combined_plot, dpi=150)
            plt.close(fig)

            for season in SEASONS:
                years_arr = np.asarray(cleaned[season]["years"], dtype=float)
                means_arr = np.asarray(cleaned[season]["means"], dtype=float)
                model = regressions[season]
                trend = model.slope * years_arr + model.intercept
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.scatter(years_arr, means_arr, color=SEASON_COLORS[season], alpha=0.7, s=24,
                           label="Cleaned seasonal mean")
                ax.plot(years_arr, trend, color="#111111", linewidth=2, label="Linear regression")
                ax.set_xlabel("Year")
                ax.set_ylabel("Groundwater level mean")
                ax.set_title(f"{station_code} {SEASON_LABELS[season]} trend")
                ax.grid(True, alpha=0.3)
                ax.legend()
                fig.tight_layout()
                fig.savefig(output_dir / f"{station_code}_{season}_trend.png", dpi=150)
                plt.close(fig)
        except Exception as exc:
            log(f"  [{station_code}] Plot error (non-fatal): {exc}")

    return {
        "seasonal_means_csv": seasonal_means_csv,
        "cleaned_means_csv":  cleaned_means_csv,
        "trends_csv":         trends_csv,
        "predictions_csv":    predictions_csv,
        "models_json":        models_json,
        "combined_plot":      combined_plot,
    }


# ---------------------------------------------------------------------------
# Master summary
# ---------------------------------------------------------------------------

def write_summary(
    results: List[Dict],
    output_path: Path,
) -> None:
    """Write a master CSV summarising regression results for every station."""
    fieldnames = [
        "station_code", "wellmaster_id", "location", "aquifer",
        "status",
        "DJF_slope", "DJF_r2", "MAM_slope", "MAM_r2",
        "JJA_slope", "JJA_r2", "SON_slope", "SON_r2",
        "predictions_csv", "models_json",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in sorted(results, key=lambda x: x["station_code"]):
            w.writerow(r)
    log(f"\nSummary written → {output_path}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_bulk_ingest(
    output_dir: Path,
    download_dir: Path,
    start_year: int,
    end_year: int,
    workers: int,
    skip_existing: bool,
    dry_run: bool,
) -> None:
    download_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    stations = scrape_all_stations()

    if dry_run:
        print("\n=== DRY RUN — stations that would be downloaded ===")
        print(f"{'Code':<14} {'Location':<26} {'Aquifer':<30} {'Levels zip'}")
        print("-" * 90)
        for s in stations:
            print(f"{s.dataset_code:<14} {s.location:<26} {s.aquifer:<30} {s.levels_zip}")
        print(f"\nTotal: {len(stations)} stations")
        return

    summary_rows: List[Dict] = []
    lock = threading.Lock()

    def process_station(station: ScrapedStation) -> None:
        code = station.dataset_code
        station_out = output_dir / code
        model_json = station_out / f"{code}_regression_models.json"

        if skip_existing and model_json.exists():
            log(f"  [{code}] Already processed, skipping.")
            with lock:
                summary_rows.append({
                    "station_code": code, "wellmaster_id": station.wellmaster_id,
                    "location": station.location, "aquifer": station.aquifer,
                    "status": "skipped (existing)",
                    "DJF_slope": "", "DJF_r2": "", "MAM_slope": "", "MAM_r2": "",
                    "JJA_slope": "", "JJA_r2": "", "SON_slope": "", "SON_r2": "",
                    "predictions_csv": str(station_out / f"{code}_predicted_seasonal_means.csv"),
                    "models_json": str(model_json),
                })
            return

        csv_path = download_and_extract(station, download_dir, skip_if_exists=skip_existing)
        if csv_path is None:
            with lock:
                summary_rows.append({
                    "station_code": code, "wellmaster_id": station.wellmaster_id,
                    "location": station.location, "aquifer": station.aquifer,
                    "status": "download_failed",
                    "DJF_slope": "", "DJF_r2": "", "MAM_slope": "", "MAM_r2": "",
                    "JJA_slope": "", "JJA_r2": "", "SON_slope": "", "SON_r2": "",
                    "predictions_csv": "", "models_json": "",
                })
            return

        log(f"  [{code}] Running seasonal analysis …")
        outputs = analyze_station(csv_path, station_out, code, start_year, end_year)

        if outputs is None:
            with lock:
                summary_rows.append({
                    "station_code": code, "wellmaster_id": station.wellmaster_id,
                    "location": station.location, "aquifer": station.aquifer,
                    "status": "analysis_failed",
                    "DJF_slope": "", "DJF_r2": "", "MAM_slope": "", "MAM_r2": "",
                    "JJA_slope": "", "JJA_r2": "", "SON_slope": "", "SON_r2": "",
                    "predictions_csv": "", "models_json": "",
                })
            return

        # read back regression results for the summary
        models = json.loads(outputs["models_json"].read_text(encoding="utf-8"))
        row: Dict = {
            "station_code": code, "wellmaster_id": station.wellmaster_id,
            "location": station.location, "aquifer": station.aquifer,
            "status": "ok",
            "predictions_csv": str(outputs["predictions_csv"]),
            "models_json": str(outputs["models_json"]),
        }
        for season in SEASONS:
            s_data = models["seasons"].get(season, {})
            row[f"{season}_slope"] = round(s_data.get("slope", 0), 8)
            row[f"{season}_r2"] = round(s_data.get("r_squared", 0), 6)
        with lock:
            summary_rows.append(row)
        log(f"  [{code}] Done.")

    log(f"\nProcessing {len(stations)} stations with {workers} parallel workers …\n")
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_station, s): s for s in stations}
        for future in as_completed(futures):
            exc = future.exception()
            if exc:
                station = futures[future]
                log(f"  [{station.dataset_code}] Unhandled error: {exc}")

    summary_path = output_dir / "all_stations_summary.csv"
    write_summary(summary_rows, summary_path)

    ok = sum(1 for r in summary_rows if r["status"] == "ok")
    skipped = sum(1 for r in summary_rows if "skipped" in r["status"])
    failed = len(summary_rows) - ok - skipped
    log(f"\n=== Bulk ingest complete ===")
    log(f"  Succeeded : {ok}")
    log(f"  Skipped   : {skipped}")
    log(f"  Failed    : {failed}")
    log(f"  Summary   : {summary_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Bulk download and analyze all BGS Future Flows groundwater datasets."
    )
    p.add_argument("--output-dir",   type=Path, default=Path("outputs"),   help="Root for analysis outputs")
    p.add_argument("--download-dir", type=Path, default=Path("downloads"), help="Cache directory for zips/CSVs")
    p.add_argument("--start-year",   type=int,  default=2000)
    p.add_argument("--end-year",     type=int,  default=2080)
    p.add_argument("--workers",      type=int,  default=4,                 help="Parallel download threads")
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip stations whose model JSON already exists")
    p.add_argument("--dry-run", action="store_true",
                   help="Print what would be downloaded without doing anything")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    run_bulk_ingest(
        output_dir=args.output_dir,
        download_dir=args.download_dir,
        start_year=args.start_year,
        end_year=args.end_year,
        workers=args.workers,
        skip_existing=args.skip_existing,
        dry_run=args.dry_run,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
