from __future__ import annotations

import csv
import html
import io
import json
import tempfile
from email.parser import BytesParser
from email.policy import default
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, Optional
from urllib.parse import parse_qs, urlparse

from groundwater_multi_agent import (
    BGS_STATIONS,
    GroundwaterPredictionAgent,
    ensure_station_model,
)

APP_TITLE = "Groundwater Depth Predictor"
BASE_DIR = Path(__file__).resolve().parent


def station_lookup() -> Dict[str, object]:
    return {station.dataset_code: station for station in BGS_STATIONS}


def normalize_station_code(value: str) -> str:
    return value.strip().upper().replace("/", "_")


def infer_station_code_from_csv_bytes(payload: bytes, fallback_name: str = "uploaded_station") -> str:
    text = payload.decode("utf-8-sig", errors="replace")
    reader = csv.reader(io.StringIO(text))
    next(reader, None)
    first_row = next(reader, None)
    if first_row and first_row[0].strip():
        return normalize_station_code(first_row[0])
    fallback = Path(fallback_name).stem or "uploaded_station"
    return normalize_station_code(fallback)


def analyze_uploaded_csv(file_bytes: bytes, filename: str, year: int, station_code: str = "") -> Dict[str, float]:
    chosen_station = normalize_station_code(station_code) if station_code.strip() else infer_station_code_from_csv_bytes(file_bytes, filename)
    with tempfile.NamedTemporaryFile(prefix=f"{chosen_station}_", suffix=".csv", delete=False) as handle:
        handle.write(file_bytes)
        temp_path = Path(handle.name)

    try:
        output_dir = BASE_DIR / "outputs" / chosen_station
        from groundwater_multi_agent import SeasonalAnalysisAgent

        SeasonalAnalysisAgent().analyze(
            input_csv=temp_path,
            output_dir=output_dir,
            station_code=chosen_station,
        )
        model_json = output_dir / f"{chosen_station}_regression_models.json"
        return GroundwaterPredictionAgent().predict(model_json, year)
    finally:
        temp_path.unlink(missing_ok=True)


def prediction_to_csv_bytes(result: Dict[str, float]) -> bytes:
    fieldnames = ["Year", "Station", "Algorithm", "DJF", "MAM", "JJA", "SON", "Annual_Mean"]
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerow({key: result.get(key, "") for key in fieldnames})
    return buffer.getvalue().encode("utf-8")


def hidden_prediction_inputs(result: Dict[str, float]) -> str:
    keys = ["Year", "Station", "Algorithm", "DJF", "MAM", "JJA", "SON", "Annual_Mean"]
    return "".join(
        f'<input type="hidden" name="{html.escape(key)}" value="{html.escape(str(result.get(key, "")))}" />'
        for key in keys
    )


def render_page(
    station_code: str,
    year: int,
    result: Optional[Dict[str, float]],
    error: str,
    upload_station_code: str = "",
    source_label: str = "Saved station model",
) -> str:
    stations = station_lookup()
    options = []
    for code, station in stations.items():
        selected = " selected" if code == station_code else ""
        label = f"{station.location} ({code})"
        options.append(f'<option value="{html.escape(code)}"{selected}>{html.escape(label)}</option>')

    station = stations.get(station_code)
    metadata = ""
    if station is not None:
        metadata = (
            f"<div class=\"meta-card\">"
            f"<div><strong>Location:</strong> {html.escape(station.location)}</div>"
            f"<div><strong>Aquifer:</strong> {html.escape(station.aquifer)}</div>"
            f"<div><strong>WellMaster ID:</strong> {html.escape(station.wellmaster_id)}</div>"
            f"<div><strong>Grid Reference:</strong> {html.escape(station.grid_reference)}</div>"
            f"</div>"
        )

    result_html = ""
    if error:
        result_html = f"<section class=\"panel error\"><h2>Could not calculate prediction</h2><p>{html.escape(error)}</p></section>"
    elif result is not None:
        result_json = html.escape(json.dumps(result, indent=2))
        download_form = f"""
        <form method="post" action="/download" class="download-form">
          {hidden_prediction_inputs(result)}
          <button type="submit" class="secondary-button">Download prediction CSV</button>
        </form>
        """
        result_html = f"""
        <section class="panel results">
          <h2>Prediction for {html.escape(str(result['Station']))} in {year}</h2>
          <p class="source-note">Source: {html.escape(source_label)}</p>
          <div class="metrics">
            <div class="metric"><span class="label">DJF</span><span class="value">{float(result['DJF']):.2f}</span></div>
            <div class="metric"><span class="label">MAM</span><span class="value">{float(result['MAM']):.2f}</span></div>
            <div class="metric"><span class="label">JJA</span><span class="value">{float(result['JJA']):.2f}</span></div>
            <div class="metric"><span class="label">SON</span><span class="value">{float(result['SON']):.2f}</span></div>
            <div class="metric annual"><span class="label">Annual Mean</span><span class="value">{float(result['Annual_Mean']):.2f}</span></div>
          </div>
          {download_form}
          <details>
            <summary>Show raw prediction JSON</summary>
            <pre>{result_json}</pre>
          </details>
        </section>
        """

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{APP_TITLE}</title>
  <style>
    :root {{
      --paper: #f7f3ea;
      --ink: #14213d;
      --accent: #d97706;
      --accent-dark: #9a3412;
      --card: rgba(255,255,255,0.78);
      --line: rgba(20, 33, 61, 0.12);
      --error: #8f1d21;
      --soft: rgba(20, 33, 61, 0.04);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Georgia, "Times New Roman", serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(217,119,6,0.18), transparent 30%),
        radial-gradient(circle at bottom right, rgba(47,111,79,0.18), transparent 28%),
        linear-gradient(160deg, #efe7d7 0%, var(--paper) 45%, #e7efe8 100%);
      min-height: 100vh;
    }}
    .wrap {{ max-width: 1120px; margin: 0 auto; padding: 32px 20px 60px; }}
    .hero {{
      padding: 28px;
      border: 1px solid var(--line);
      background: rgba(255,255,255,0.58);
      backdrop-filter: blur(10px);
      border-radius: 24px;
      box-shadow: 0 20px 50px rgba(20,33,61,0.08);
      margin-bottom: 24px;
    }}
    .eyebrow {{ letter-spacing: 0.18em; text-transform: uppercase; font-size: 0.8rem; color: var(--accent-dark); margin-bottom: 8px; }}
    h1 {{ font-size: clamp(2rem, 4vw, 4rem); margin: 0 0 12px; line-height: 0.96; }}
    .sub {{ max-width: 52rem; font-size: 1.05rem; line-height: 1.6; margin: 0; }}
    .grid {{ display: grid; grid-template-columns: 1.15fr 0.85fr; gap: 24px; align-items: start; }}
    .stack {{ display: grid; gap: 24px; }}
    .panel {{ border: 1px solid var(--line); background: var(--card); border-radius: 22px; padding: 24px; box-shadow: 0 12px 30px rgba(20,33,61,0.06); }}
    form {{ display: grid; gap: 16px; }}
    label {{ display: grid; gap: 8px; font-weight: 600; }}
    select, input, button {{ font: inherit; padding: 14px 16px; border-radius: 14px; border: 1px solid var(--line); background: white; }}
    input[type="file"] {{ padding: 10px 12px; }}
    button {{ cursor: pointer; background: linear-gradient(135deg, var(--accent) 0%, #eab308 100%); color: #1f2937; font-weight: 700; border: none; }}
    .secondary-button {{ background: linear-gradient(135deg, #1b4965 0%, #61a5c2 100%); color: white; }}
    .meta-card {{ display: grid; gap: 10px; padding: 18px; background: var(--soft); border-radius: 18px; border: 1px solid var(--line); margin-top: 16px; }}
    .results h2, .panel h2 {{ margin-top: 0; }}
    .metrics {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 14px; }}
    .metric {{ padding: 18px; border-radius: 18px; background: linear-gradient(180deg, rgba(255,255,255,0.95), rgba(20,33,61,0.03)); border: 1px solid var(--line); }}
    .metric.annual {{ grid-column: 1 / -1; background: linear-gradient(135deg, rgba(47,111,79,0.15), rgba(255,255,255,0.9)); }}
    .label {{ display: block; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.08em; color: var(--accent-dark); }}
    .value {{ display: block; font-size: 2rem; margin-top: 6px; font-weight: 700; }}
    .source-note {{ margin-top: -2px; margin-bottom: 16px; color: #4b5563; font-size: 0.95rem; }}
    .error {{ border-color: rgba(143,29,33,0.2); }}
    .error p {{ color: var(--error); }}
    .hint {{ margin: 0; color: #4b5563; line-height: 1.55; }}
    details {{ margin-top: 18px; }}
    pre {{ overflow-x: auto; padding: 12px; border-radius: 12px; background: #f5f5f5; }}
    .footer {{ margin-top: 18px; font-size: 0.92rem; opacity: 0.82; }}
    .download-form {{ margin-top: 18px; }}
    @media (max-width: 900px) {{ .grid {{ grid-template-columns: 1fr; }} .metrics {{ grid-template-columns: 1fr 1fr; }} }}
    @media (max-width: 560px) {{ .wrap {{ padding: 18px 14px 40px; }} .hero, .panel {{ border-radius: 18px; padding: 18px; }} .metrics {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <div class="eyebrow">Groundwater Forecasting</div>
      <h1>{APP_TITLE}</h1>
      <p class="sub">Choose a BGS Future Flows station and a target year, or upload a groundwater CSV directly. The app predicts seasonal groundwater depth and gives you a downloadable CSV result.</p>
    </section>

    <div class="grid">
      <div class="stack">
        <section class="panel">
          <h2>Use a saved station model</h2>
          <form method="get" action="/">
            <label>
              Station
              <select name="station">{''.join(options)}</select>
            </label>
            <label>
              Year
              <input type="number" min="1951" max="2100" step="1" name="year" value="{year}" />
            </label>
            <button type="submit">Predict from saved model</button>
          </form>
          {metadata}
          <p class="footer">Tip: Therfield Rectory (`TL33_4`) already has generated model outputs in this repository, so it works immediately.</p>
        </section>

        <section class="panel">
          <h2>Upload a groundwater CSV</h2>
          <form method="post" action="/" enctype="multipart/form-data">
            <label>
              Target Year
              <input type="number" min="1951" max="2100" step="1" name="year" value="{year}" />
            </label>
            <label>
              Optional Station Code Override
              <input type="text" name="upload_station" value="{html.escape(upload_station_code)}" placeholder="Example: TL33_4" />
            </label>
            <label>
              CSV File
              <input type="file" name="csv_file" accept=".csv,text/csv" required />
            </label>
            <button type="submit">Analyze uploaded CSV</button>
          </form>
          <p class="hint">If the station code field is left empty, the app will try to infer the station code from the first data row of the uploaded CSV.</p>
        </section>
      </div>
      {result_html or '<section class="panel"><h2>Prediction output</h2><p>Submit either form to view the seasonal groundwater prediction and download it as CSV.</p></section>'}
    </div>
  </div>
</body>
</html>
"""


class GroundwaterRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/download":
            self.send_error(405, "Use POST to download prediction CSV")
            return
        if parsed.path not in {"/", "/index.html"}:
            self.send_error(404, "Not found")
            return

        stations = station_lookup()
        default_station = next(iter(stations))
        params = parse_qs(parsed.query)
        station_code = normalize_station_code(params.get("station", [default_station])[0])
        if station_code not in stations:
            station_code = default_station

        try:
            year = int(params.get("year", ["2050"])[0])
        except ValueError:
            year = 2050

        result = None
        error = ""
        source_label = "Saved station model"
        if parsed.query:
            try:
                model_json = ensure_station_model(station_code, base_dir=BASE_DIR)
                result = GroundwaterPredictionAgent().predict(model_json, year)
            except Exception as exc:
                error = str(exc)

        body = render_page(
            station_code=station_code,
            year=year,
            result=result,
            error=error,
            source_label=source_label,
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/download":
            self._handle_download_post()
            return
        if parsed.path not in {"/", "/index.html"}:
            self.send_error(404, "Not found")
            return

        try:
            fields, files = self._parse_post_request()
            year = int(fields.get("year", "2050"))
            upload_station = fields.get("upload_station", "")
            upload = files.get("csv_file")
            if upload is None or not upload["content"]:
                raise ValueError("Please choose a CSV file to upload.")
            result = analyze_uploaded_csv(
                upload["content"],
                upload["filename"],
                year,
                station_code=upload_station,
            )
            station_code = normalize_station_code(str(result["Station"]))
            body = render_page(
                station_code=station_code,
                year=year,
                result=result,
                error="",
                upload_station_code=upload_station,
                source_label=f"Uploaded CSV: {upload['filename']}",
            ).encode("utf-8")
        except Exception as exc:
            station_code = normalize_station_code(fields.get("upload_station", "TL33_4")) if "fields" in locals() else "TL33_4"
            body = render_page(
                station_code=station_code,
                year=2050,
                result=None,
                error=str(exc),
                upload_station_code=fields.get("upload_station", "") if "fields" in locals() else "",
            ).encode("utf-8")

        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _handle_download_post(self) -> None:
        fields, _files = self._parse_post_request()
        try:
            result = {
                "Year": int(fields["Year"]),
                "Station": fields["Station"],
                "Algorithm": fields.get("Algorithm", "linear_regression"),
                "DJF": float(fields["DJF"]),
                "MAM": float(fields["MAM"]),
                "JJA": float(fields["JJA"]),
                "SON": float(fields["SON"]),
                "Annual_Mean": float(fields["Annual_Mean"]),
            }
        except (KeyError, ValueError) as exc:
            self.send_error(400, f"Invalid prediction download request: {exc}")
            return

        payload = prediction_to_csv_bytes(result)
        filename = f"{normalize_station_code(str(result['Station']))}_prediction_{result['Year']}.csv"
        self.send_response(200)
        self.send_header("Content-Type", "text/csv; charset=utf-8")
        self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _parse_post_request(self) -> tuple[Dict[str, str], Dict[str, Dict[str, bytes | str]]]:
        content_type = self.headers.get("Content-Type", "")
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)

        if content_type.startswith("multipart/form-data"):
            return self._parse_multipart(content_type, body)
        if content_type.startswith("application/x-www-form-urlencoded"):
            params = parse_qs(body.decode("utf-8"), keep_blank_values=True)
            fields = {key: values[0] if values else "" for key, values in params.items()}
            return fields, {}
        raise ValueError(f"Unsupported content type: {content_type}")

    def _parse_multipart(self, content_type: str, body: bytes) -> tuple[Dict[str, str], Dict[str, Dict[str, bytes | str]]]:
        message = BytesParser(policy=default).parsebytes(
            f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode("utf-8") + body
        )
        fields: Dict[str, str] = {}
        files: Dict[str, Dict[str, bytes | str]] = {}
        for part in message.iter_parts():
            disposition = part.get_content_disposition()
            if disposition != "form-data":
                continue
            name = part.get_param("name", header="content-disposition")
            if not name:
                continue
            filename = part.get_filename()
            payload = part.get_payload(decode=True) or b""
            if filename:
                files[name] = {"filename": filename, "content": payload}
            else:
                fields[name] = payload.decode("utf-8", errors="replace")
        return fields, files

    def log_message(self, format: str, *args) -> None:
        return


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run the local groundwater depth web app.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), GroundwaterRequestHandler)
    print(f"Serving {APP_TITLE} at http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
