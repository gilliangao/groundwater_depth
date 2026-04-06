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
from typing import Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "tygron-mpl-cache"))

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


@dataclass(frozen=True)
class StationRecord:
    dataset_code: str
    wellmaster_id: str
    location: str
    aquifer: str
    grid_reference: str

    @property
    def levels_zip_name(self) -> str:
        return f"FutureFlowsHydrology-{self.dataset_code}-GWL.zip"

    @property
    def climate_zip_name(self) -> str:
        return f"FutureFlowsClimate-{self.dataset_code}.zip"

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


BGS_STATIONS: Tuple[StationRecord, ...] = (
    StationRecord("SY68_34", "SY68/34", "Ashton Farm", "Chalk", "3661 0880"),
    StationRecord("TA10_63", "TA10/63", "Aylesby", "Chalk", "5194 4071"),
    StationRecord("SU81_1", "SU81/1", "Chilgrove House", "Chalk", "4835 1143"),
    StationRecord("SU34_8D", "SU34/8D", "Clanville Lodge Gate", "Chalk", "4322 1490"),
    StationRecord("SE94_5", "SE94/5", "Dalton Holme", "Chalk", "4965 4453"),
    StationRecord("ST88_62A", "ST88/62A", "Didmarton 1", "Inferior Oolite", "3827 1874"),
    StationRecord("SD27_6B", "SD27/6B", "Furness Abbey", "Permo-Triassic Sandstone", "3216 4717"),
    StationRecord("TL89_37", "TL89/37", "Grimes Graves", "Chalk", "5817 2900"),
    StationRecord("SJ62_112", "SJ62/112", "Heathlanes", "Permo-Triassic Sandstone", "3619 3210"),
    StationRecord("SK17_13", "SK17/13", "Hucklow South", "Carboniferous Limestone", "4177 3775"),
    StationRecord("TR14_9", "TR14/9", "Little Bucket Farm", "Chalk", "6122 1469"),
    StationRecord("SJ15_13", "SJ15/13", "Llanfair Dyffryn Clwyd", "Permo-Triassic Sandstone", "3137 3555"),
    StationRecord("TQ41_82", "TQ41/82", "Lower Barn Cottage", "Lower Greensand", "5437 1132"),
    StationRecord("TF03_37", "TF03/37", "New Red Lion", "Lincolnshire Limestone", "5088 3303"),
    StationRecord("NX97_2", "NX97/2", "Newbridge", "Permo-Triassic Sandstone", "2951 5788"),
    StationRecord("SU17_57", "SU17/57", "Rockley", "Chalk", "4165 1717"),
    StationRecord("NY63_2", "NY63/2", "Skirwith", "Permo-Triassic Sandstone", "3613 5325"),
    StationRecord("SU78_45A", "SU78/45A", "Stonor Park", "Chalk", "4741 1892"),
    StationRecord("NZ21_29", "NZ21/29", "Swan House", "Magnesian Limestone", "4252 5199"),
    StationRecord("TL33_4", "TL33/4", "Therfield Rectory", "Chalk", "5333 2372"),
    StationRecord("TF81_2A", "TF81/2A", "Washpit Farm", "Chalk", "5813 3196"),
    StationRecord("TQ25_13", "TQ25/13", "Well House Inn", "Chalk", "5258 1552"),
    StationRecord("TV59_7C", "TV59/7C", "West Dean No. 3", "Chalk", "5529 0992"),
    StationRecord("SU01_5B", "SU01/5B", "West Woodyates Manor", "Chalk", "4016 1194"),
)


class StationDiscoveryAgent:
    """Agent 1: station catalog and station CSV download."""

    def __init__(self, stations: Sequence[StationRecord] = BGS_STATIONS) -> None:
        self._stations = tuple(stations)

    def list_stations(self) -> List[StationRecord]:
        return list(self._stations)

    def save_catalog(self, output_csv: Path) -> Path:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with output_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "dataset_code",
                    "wellmaster_id",
                    "location",
                    "aquifer",
                    "grid_reference",
                    "levels_url",
                    "climate_url",
                ],
            )
            writer.writeheader()
            for station in self._stations:
                row = asdict(station)
                row["levels_url"] = station.levels_url
                row["climate_url"] = station.climate_url
                writer.writerow(row)
        return output_csv

    def try_refresh_from_website(self, url: str = BGS_SITES_URL, timeout: int = 30) -> List[StationRecord]:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            html = response.read().decode("utf-8", errors="ignore")

        pattern = re.compile(
            r"(?P<wellmaster>[A-Z]{1,2}\d{2}/[0-9A-Z]+)\s+"
            r"(?P<grid>\d{4}\s+\d{4})\s+"
            r"(?P<location>[^<\n]+?)\s+"
            r"(?P<aquifer>Chalk|Inferior Oolite|Permo-Triassic Sandstone|Carboniferous Limestone|Lower Greensand|Lincolnshire Limestone|Magnesian Limestone)",
            re.IGNORECASE,
        )

        refreshed: List[StationRecord] = []
        seen = set()
        for match in pattern.finditer(html):
            wellmaster_id = match.group("wellmaster")
            dataset_code = wellmaster_id.replace("/", "_")
            if dataset_code in seen:
                continue
            seen.add(dataset_code)
            refreshed.append(
                StationRecord(
                    dataset_code=dataset_code,
                    wellmaster_id=wellmaster_id,
                    location=" ".join(match.group("location").split()),
                    aquifer=match.group("aquifer"),
                    grid_reference=" ".join(match.group("grid").split()),
                )
            )

        if not refreshed:
            raise RuntimeError(
                "Could not parse the BGS stations page. The website structure may have changed."
            )
        return refreshed

    def download_station_dataset(
        self,
        station_code: str,
        output_dir: Path,
        dataset: str = "levels",
    ) -> List[Path]:
        station = self._get_station(station_code)
        output_dir.mkdir(parents=True, exist_ok=True)

        targets = []
        if dataset in {"levels", "both"}:
            targets.append((station.levels_url, station.levels_zip_name))
        if dataset in {"climate", "both"}:
            targets.append((station.climate_url, station.climate_zip_name))

        extracted_paths: List[Path] = []
        for url, archive_name in targets:
            archive_path = output_dir / archive_name
            urllib.request.urlretrieve(url, archive_path)
            for extracted_path in self._extract_zip(archive_path, output_dir):
                self._validate_downloaded_file(extracted_path)
                extracted_paths.append(extracted_path)
        return extracted_paths

    def _extract_zip(self, archive_path: Path, output_dir: Path) -> List[Path]:
        extracted: List[Path] = []
        with zipfile.ZipFile(archive_path) as archive:
            for member in archive.namelist():
                extracted_path = output_dir / Path(member).name
                with archive.open(member) as source, extracted_path.open("wb") as target:
                    target.write(source.read())
                extracted.append(extracted_path)
        return extracted

    def _get_station(self, station_code: str) -> StationRecord:
        lookup = station_code.strip().upper().replace("/", "_")
        for station in self._stations:
            if station.dataset_code.upper() == lookup:
                return station
        raise KeyError(f"Unknown station '{station_code}'. Run the 'stations' command to list valid station codes.")

    @staticmethod
    def _validate_downloaded_file(file_path: Path) -> None:
        if file_path.suffix.lower() != ".csv":
            return
        with file_path.open("r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.reader(handle)
            headers = next(reader)
        if "Date" not in headers or len(headers) < 4:
            raise ValueError(f"Downloaded file {file_path} does not look like a BGS groundwater ensemble CSV.")


class SeasonalAnalysisAgent:
    """Agent 2: outlier removal, seasonal aggregation, plots, and regression."""

    def __init__(self, start_year: int = 2000, end_year: int = 2080) -> None:
        _require_analysis_dependencies(needs_plot=False)
        self.start_year = start_year
        self.end_year = end_year

    def analyze(self, input_csv: Path, output_dir: Path, station_code: Optional[str] = None) -> Dict[str, Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        rows, ensemble_cols, inferred_station = self._read_groundwater_csv(input_csv)
        station_code = station_code or inferred_station or input_csv.stem

        seasonal_means = self._build_seasonal_means(rows, ensemble_cols)
        cleaned_series = self._remove_outliers_and_average(seasonal_means)
        regressions = self._fit_regressions(cleaned_series)

        seasonal_means_csv = output_dir / f"{station_code}_seasonal_means.csv"
        cleaned_means_csv = output_dir / f"{station_code}_seasonal_cleaned_means.csv"
        trends_csv = output_dir / f"{station_code}_seasonal_trends.csv"
        predictions_csv = output_dir / f"{station_code}_predicted_seasonal_means.csv"
        models_json = output_dir / f"{station_code}_regression_models.json"
        combined_plot = output_dir / f"{station_code}_seasonal_trends.png"

        self._write_seasonal_means_csv(seasonal_means, ensemble_cols, seasonal_means_csv)
        self._write_cleaned_means_csv(cleaned_series, cleaned_means_csv)
        self._write_trends_csv(regressions, trends_csv)
        self._write_models_json(regressions, station_code, input_csv, models_json)
        self._write_predictions_csv(regressions, cleaned_series, predictions_csv)
        self._plot_combined(cleaned_series, regressions, combined_plot)
        self._plot_per_season(cleaned_series, regressions, output_dir, station_code)

        return {
            "seasonal_means_csv": seasonal_means_csv,
            "cleaned_means_csv": cleaned_means_csv,
            "trends_csv": trends_csv,
            "predictions_csv": predictions_csv,
            "models_json": models_json,
            "combined_plot": combined_plot,
        }

    def _read_groundwater_csv(
        self, input_csv: Path
    ) -> Tuple[List[Dict[str, object]], List[str], Optional[str]]:
        rows: List[Dict[str, object]] = []
        with input_csv.open("r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.reader(handle)
            headers = next(reader)
            if len(headers) < 4:
                raise ValueError("Expected at least 4 columns in the groundwater CSV.")
            ensemble_cols = headers[3:]
            station_code: Optional[str] = None

            for raw_row in reader:
                if not raw_row:
                    continue
                date = datetime.strptime(raw_row[2], "%d/%m/%Y")
                if not (self.start_year <= date.year <= self.end_year):
                    continue
                values = [float(value) for value in raw_row[3:]]
                station_code = station_code or raw_row[0]
                rows.append(
                    {
                        "station_code": raw_row[0],
                        "hydmod_id": raw_row[1],
                        "date": date,
                        "values": values,
                    }
                )

        if not rows:
            raise ValueError(
                f"No rows found in {input_csv} for years {self.start_year}-{self.end_year}."
            )
        normalized_station = station_code.replace("/", "_") if station_code else None
        return rows, ensemble_cols, normalized_station

    def _build_seasonal_means(
        self, rows: Sequence[Dict[str, object]], ensemble_cols: Sequence[str]
    ) -> Dict[Tuple[int, str], List[float]]:
        seasonal_values: Dict[Tuple[int, str], List[List[float]]] = {}

        for row in rows:
            date = row["date"]
            assert isinstance(date, datetime)
            values = row["values"]
            assert isinstance(values, list)
            season = self._season_for_month(date.month)
            key = (date.year, season)
            seasonal_values.setdefault(key, [[] for _ in ensemble_cols])
            for idx, value in enumerate(values):
                seasonal_values[key][idx].append(value)

        seasonal_means: Dict[Tuple[int, str], List[float]] = {}
        for key, per_ensemble in seasonal_values.items():
            seasonal_means[key] = [float(np.mean(series)) for series in per_ensemble]
        return dict(sorted(seasonal_means.items()))

    def _remove_outliers_and_average(
        self, seasonal_means: Dict[Tuple[int, str], List[float]]
    ) -> Dict[str, Dict[str, List[float]]]:
        cleaned = {season: {"years": [], "means": []} for season in SEASONS}
        for (year, season), values in seasonal_means.items():
            filtered = self._remove_outliers(values)
            if not filtered:
                continue
            cleaned[season]["years"].append(year)
            cleaned[season]["means"].append(float(np.mean(filtered)))
        return cleaned

    def _fit_regressions(
        self, cleaned_series: Dict[str, Dict[str, List[float]]]
    ) -> Dict[str, SeasonalRegressionResult]:
        results: Dict[str, SeasonalRegressionResult] = {}
        for season in SEASONS:
            years = np.asarray(cleaned_series[season]["years"], dtype=float)
            means = np.asarray(cleaned_series[season]["means"], dtype=float)
            if len(years) < 2:
                raise ValueError(f"Not enough data points to fit regression for season {season}.")
            slope, intercept, r_value, p_value, std_error = stats.linregress(years, means)
            results[season] = SeasonalRegressionResult(
                season=season,
                slope=float(slope),
                intercept=float(intercept),
                r_squared=float(r_value**2),
                p_value=float(p_value),
                std_error=float(std_error),
            )
        return results

    def _write_seasonal_means_csv(
        self,
        seasonal_means: Dict[Tuple[int, str], List[float]],
        ensemble_cols: Sequence[str],
        output_csv: Path,
    ) -> None:
        with output_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["Year", "Season", *ensemble_cols])
            for (year, season), values in seasonal_means.items():
                writer.writerow([year, season, *values])

    def _write_cleaned_means_csv(
        self, cleaned_series: Dict[str, Dict[str, List[float]]], output_csv: Path
    ) -> None:
        with output_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["Year", "Season", "Mean_After_Outlier_Removal"])
            writer.writeheader()
            for season in SEASONS:
                for year, mean in zip(cleaned_series[season]["years"], cleaned_series[season]["means"]):
                    writer.writerow(
                        {
                            "Year": year,
                            "Season": season,
                            "Mean_After_Outlier_Removal": mean,
                        }
                    )

    def _write_trends_csv(
        self, regressions: Dict[str, SeasonalRegressionResult], output_csv: Path
    ) -> None:
        with output_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["Season", "Slope", "Intercept", "R_squared", "P_value", "Std_Error"],
            )
            writer.writeheader()
            for season in SEASONS:
                row = regressions[season]
                writer.writerow(
                    {
                        "Season": row.season,
                        "Slope": row.slope,
                        "Intercept": row.intercept,
                        "R_squared": row.r_squared,
                        "P_value": row.p_value,
                        "Std_Error": row.std_error,
                    }
                )

    def _write_models_json(
        self,
        regressions: Dict[str, SeasonalRegressionResult],
        station_code: str,
        input_csv: Path,
        output_json: Path,
    ) -> None:
        payload = {
            "station_code": station_code,
            "source_csv": str(input_csv),
            "year_range": {"start": self.start_year, "end": self.end_year},
            "algorithm": "linear_regression",
            "seasons": {season: asdict(result) for season, result in regressions.items()},
        }
        output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _write_predictions_csv(
        self,
        regressions: Dict[str, SeasonalRegressionResult],
        cleaned_series: Dict[str, Dict[str, List[float]]],
        output_csv: Path,
    ) -> None:
        all_years = sorted({year for season in SEASONS for year in cleaned_series[season]["years"]})
        with output_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["Year", *SEASONS, "Annual_Mean"])
            writer.writeheader()
            for year in all_years:
                row = {"Year": year}
                season_values = []
                for season in SEASONS:
                    prediction = regressions[season].slope * year + regressions[season].intercept
                    row[season] = prediction
                    season_values.append(prediction)
                row["Annual_Mean"] = float(np.mean(season_values))
                writer.writerow(row)

    def _plot_combined(
        self,
        cleaned_series: Dict[str, Dict[str, List[float]]],
        regressions: Dict[str, SeasonalRegressionResult],
        output_png: Path,
    ) -> None:
        plot = _load_matplotlib_pyplot()
        plot.figure(figsize=(12, 8))
        for season in SEASONS:
            years = np.asarray(cleaned_series[season]["years"], dtype=float)
            means = np.asarray(cleaned_series[season]["means"], dtype=float)
            color = SEASON_COLORS[season]
            model = regressions[season]
            trend_line = model.slope * years + model.intercept

            plot.scatter(years, means, color=color, alpha=0.65, s=18, label=f"{SEASON_LABELS[season]} data")
            plot.plot(
                years,
                trend_line,
                color=color,
                linewidth=2,
                label=f"{SEASON_LABELS[season]} trend (R^2={model.r_squared:.3f})",
            )

        plot.xlabel("Year")
        plot.ylabel("Groundwater level mean")
        plot.title("Seasonal groundwater trends after ensemble outlier removal")
        plot.grid(True, alpha=0.3)
        plot.legend()
        plot.tight_layout()
        plot.savefig(output_png, dpi=300)
        plot.close()

    def _plot_per_season(
        self,
        cleaned_series: Dict[str, Dict[str, List[float]]],
        regressions: Dict[str, SeasonalRegressionResult],
        output_dir: Path,
        station_code: str,
    ) -> None:
        plot = _load_matplotlib_pyplot()
        for season in SEASONS:
            years = np.asarray(cleaned_series[season]["years"], dtype=float)
            means = np.asarray(cleaned_series[season]["means"], dtype=float)
            model = regressions[season]
            trend_line = model.slope * years + model.intercept
            color = SEASON_COLORS[season]

            plot.figure(figsize=(10, 6))
            plot.scatter(years, means, color=color, alpha=0.7, s=24, label="Cleaned seasonal mean")
            plot.plot(years, trend_line, color="#111111", linewidth=2, label="Linear regression")
            plot.xlabel("Year")
            plot.ylabel("Groundwater level mean")
            plot.title(f"{station_code} {SEASON_LABELS[season]} groundwater trend")
            plot.grid(True, alpha=0.3)
            plot.legend()
            plot.tight_layout()
            plot.savefig(output_dir / f"{station_code}_{season}_trend.png", dpi=300)
            plot.close()

    @staticmethod
    def _remove_outliers(values: Sequence[float]) -> List[float]:
        if len(values) < 4:
            return list(values)
        q1 = float(np.percentile(values, 25))
        q3 = float(np.percentile(values, 75))
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        return [value for value in values if lower <= value <= upper]

    @staticmethod
    def _season_for_month(month: int) -> str:
        if month in (12, 1, 2):
            return "DJF"
        if month in (3, 4, 5):
            return "MAM"
        if month in (6, 7, 8):
            return "JJA"
        return "SON"


class GroundwaterPredictionAgent:
    """Agent 3: year-based prediction for a selected station."""

    def predict(self, model_json: Path, year: int) -> Dict[str, float]:
        _require_analysis_dependencies(needs_plot=False)
        payload = json.loads(model_json.read_text(encoding="utf-8"))
        seasons = payload["seasons"]
        prediction = {"Year": year, "Station": payload["station_code"], "Algorithm": payload["algorithm"]}
        season_values: List[float] = []
        for season in SEASONS:
            slope = seasons[season]["slope"]
            intercept = seasons[season]["intercept"]
            value = slope * year + intercept
            prediction[season] = value
            season_values.append(value)
        prediction["Annual_Mean"] = float(np.mean(season_values))
        return prediction

    def save_prediction(self, prediction: Dict[str, float], output_csv: Path) -> Path:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with output_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(prediction.keys()))
            writer.writeheader()
            writer.writerow(prediction)
        return output_csv


class GroundwaterWorkflowManager:
    """Coordinator that runs the three agents in sequence."""

    def __init__(self) -> None:
        self.station_agent = StationDiscoveryAgent()
        self.analysis_agent = SeasonalAnalysisAgent()
        self.prediction_agent = GroundwaterPredictionAgent()

    def run(self, station_code: str, input_csv: Path, year: int, output_dir: Path) -> Dict[str, object]:
        analysis_outputs = self.analysis_agent.analyze(input_csv=input_csv, output_dir=output_dir, station_code=station_code)
        prediction = self.prediction_agent.predict(analysis_outputs["models_json"], year)
        prediction_path = self.prediction_agent.save_prediction(
            prediction,
            output_dir / f"{station_code}_prediction_{year}.csv",
        )
        return {
            "analysis_outputs": analysis_outputs,
            "prediction": prediction,
            "prediction_csv": prediction_path,
        }


def _require_analysis_dependencies(needs_plot: bool) -> None:
    missing = []
    if np is None:
        missing.append("numpy")
    if stats is None:
        missing.append("scipy")
    if needs_plot and matplotlib is None:
        missing.append("matplotlib")
    if missing:
        joined = ", ".join(missing)
        raise ModuleNotFoundError(
            f"Missing required analysis dependencies: {joined}. "
            "Use the project's plotting environment, for example `venv_plot/bin/python`."
        )


def _load_matplotlib_pyplot():
    global plt
    _require_analysis_dependencies(needs_plot=True)
    if plt is None:
        assert matplotlib is not None
        matplotlib.use("Agg")
        import matplotlib.pyplot as pyplot

        plt = pyplot
    return plt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Multi-agent groundwater workflow for BGS Future Flows station discovery, seasonal analysis, and prediction."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    stations_parser = subparsers.add_parser("stations", help="List the known BGS Future Flows groundwater stations.")
    stations_parser.add_argument("--save-csv", type=Path, help="Optional path to save the station catalog as CSV.")
    stations_parser.add_argument(
        "--refresh-from-web",
        action="store_true",
        help="Try to refresh the station list from the BGS website instead of using the built-in catalog.",
    )

    download_parser = subparsers.add_parser("download", help="Download level/climate zip files and extract CSVs for a station.")
    download_parser.add_argument("--station", required=True, help="Station code such as TL33_4 or WellMaster ID such as TL33/4.")
    download_parser.add_argument("--dataset", choices=["levels", "climate", "both"], default="levels")
    download_parser.add_argument("--output-dir", type=Path, default=Path("downloads"))

    analyze_parser = subparsers.add_parser("analyze", help="Remove outliers, build seasonal plots, and fit regression models.")
    analyze_parser.add_argument("--input-csv", type=Path, required=True)
    analyze_parser.add_argument("--station", help="Optional station code override.")
    analyze_parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    analyze_parser.add_argument("--start-year", type=int, default=2000)
    analyze_parser.add_argument("--end-year", type=int, default=2080)

    predict_parser = subparsers.add_parser("predict", help="Predict groundwater level statistics for a chosen year.")
    predict_parser.add_argument("--model-json", type=Path, required=True)
    predict_parser.add_argument("--year", type=int, required=True)
    predict_parser.add_argument("--output-csv", type=Path)

    run_parser = subparsers.add_parser("run", help="Run agent 2 and agent 3 together on a local station CSV.")
    run_parser.add_argument("--station", required=True)
    run_parser.add_argument("--input-csv", type=Path, required=True)
    run_parser.add_argument("--year", type=int, required=True)
    run_parser.add_argument("--output-dir", type=Path, default=Path("outputs"))

    return parser


def cmd_stations(args: argparse.Namespace) -> int:
    agent = StationDiscoveryAgent()
    stations = agent.try_refresh_from_website() if args.refresh_from_web else agent.list_stations()

    if args.save_csv:
        agent = StationDiscoveryAgent(stations)
        saved = agent.save_catalog(args.save_csv)
        print(f"Saved {len(stations)} stations to {saved}")

    print("dataset_code,wellmaster_id,location,aquifer,grid_reference")
    for station in stations:
        print(
            f"{station.dataset_code},{station.wellmaster_id},{station.location},{station.aquifer},{station.grid_reference}"
        )
    return 0


def cmd_download(args: argparse.Namespace) -> int:
    agent = StationDiscoveryAgent()
    extracted = agent.download_station_dataset(args.station, args.output_dir, dataset=args.dataset)
    print(f"Downloaded and extracted {len(extracted)} file(s):")
    for path in extracted:
        print(path)
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    agent = SeasonalAnalysisAgent(start_year=args.start_year, end_year=args.end_year)
    outputs = agent.analyze(args.input_csv, args.output_dir, station_code=args.station)
    print("Analysis outputs:")
    for label, path in outputs.items():
        print(f"{label}: {path}")
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    agent = GroundwaterPredictionAgent()
    prediction = agent.predict(args.model_json, args.year)
    print(json.dumps(prediction, indent=2))
    if args.output_csv:
        path = agent.save_prediction(prediction, args.output_csv)
        print(f"Saved prediction CSV to {path}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    manager = GroundwaterWorkflowManager()
    result = manager.run(args.station, args.input_csv, args.year, args.output_dir)
    print("Analysis outputs:")
    for label, path in result["analysis_outputs"].items():
        print(f"{label}: {path}")
    print("Prediction:")
    print(json.dumps(result["prediction"], indent=2))
    print(f"prediction_csv: {result['prediction_csv']}")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "stations":
        return cmd_stations(args)
    if args.command == "download":
        return cmd_download(args)
    if args.command == "analyze":
        return cmd_analyze(args)
    if args.command == "predict":
        return cmd_predict(args)
    if args.command == "run":
        return cmd_run(args)
    parser.error("Unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
