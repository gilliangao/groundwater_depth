from __future__ import annotations

import html
import json
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


def render_page(station_code: str, year: int, result: Optional[Dict[str, float]], error: str) -> str:
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
            f"<div class='meta-card'>"
            f"<div><strong>Location:</strong> {html.escape(station.location)}</div>"
            f"<div><strong>Aquifer:</strong> {html.escape(station.aquifer)}</div>"
            f"<div><strong>WellMaster ID:</strong> {html.escape(station.wellmaster_id)}</div>"
            f"<div><strong>Grid Reference:</strong> {html.escape(station.grid_reference)}</div>"
            f"</div>"
        )

    result_html = ""
    if error:
        result_html = f"<section class='panel error'><h2>Could not calculate prediction</h2><p>{html.escape(error)}</p></section>"
    elif result is not None:
        result_json = html.escape(json.dumps(result, indent=2))
        result_html = f"""
        <section class='panel results'>
          <h2>Prediction for {html.escape(station_code)} in {year}</h2>
          <div class='metrics'>
            <div class='metric'><span class='label'>DJF</span><span class='value'>{result['DJF']:.2f}</span></div>
            <div class='metric'><span class='label'>MAM</span><span class='value'>{result['MAM']:.2f}</span></div>
            <div class='metric'><span class='label'>JJA</span><span class='value'>{result['JJA']:.2f}</span></div>
            <div class='metric'><span class='label'>SON</span><span class='value'>{result['SON']:.2f}</span></div>
            <div class='metric annual'><span class='label'>Annual Mean</span><span class='value'>{result['Annual_Mean']:.2f}</span></div>
          </div>
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
      --good: #2f6f4f;
      --error: #8f1d21;
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
    .wrap {{ max-width: 1100px; margin: 0 auto; padding: 32px 20px 60px; }}
    .hero {{
      padding: 28px;
      border: 1px solid var(--line);
      background: rgba(255,255,255,0.58);
      backdrop-filter: blur(10px);
      border-radius: 24px;
      box-shadow: 0 20px 50px rgba(20,33,61,0.08);
      margin-bottom: 24px;
    }}
    .eyebrow {{
      letter-spacing: 0.18em;
      text-transform: uppercase;
      font-size: 0.8rem;
      color: var(--accent-dark);
      margin-bottom: 8px;
    }}
    h1 {{ font-size: clamp(2rem, 4vw, 4rem); margin: 0 0 12px; line-height: 0.96; }}
    .sub {{ max-width: 50rem; font-size: 1.05rem; line-height: 1.6; margin: 0; }}
    .grid {{ display: grid; grid-template-columns: 1.1fr 0.9fr; gap: 24px; align-items: start; }}
    .panel {{
      border: 1px solid var(--line);
      background: var(--card);
      border-radius: 22px;
      padding: 24px;
      box-shadow: 0 12px 30px rgba(20,33,61,0.06);
    }}
    form {{ display: grid; gap: 16px; }}
    label {{ display: grid; gap: 8px; font-weight: 600; }}
    select, input, button {{
      font: inherit;
      padding: 14px 16px;
      border-radius: 14px;
      border: 1px solid var(--line);
      background: white;
    }}
    button {{
      cursor: pointer;
      background: linear-gradient(135deg, var(--accent) 0%, #eab308 100%);
      color: #1f2937;
      font-weight: 700;
      border: none;
    }}
    button:hover {{ filter: saturate(1.05) contrast(1.02); }}
    .meta-card {{
      display: grid;
      gap: 10px;
      padding: 18px;
      background: rgba(20,33,61,0.04);
      border-radius: 18px;
      border: 1px solid var(--line);
      margin-top: 16px;
    }}
    .results h2, .panel h2 {{ margin-top: 0; }}
    .metrics {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 14px; }}
    .metric {{
      padding: 18px;
      border-radius: 18px;
      background: linear-gradient(180deg, rgba(255,255,255,0.95), rgba(20,33,61,0.03));
      border: 1px solid var(--line);
    }}
    .metric.annual {{ grid-column: 1 / -1; background: linear-gradient(135deg, rgba(47,111,79,0.15), rgba(255,255,255,0.9)); }}
    .label {{ display: block; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.08em; color: var(--accent-dark); }}
    .value {{ display: block; font-size: 2rem; margin-top: 6px; font-weight: 700; }}
    .error {{ border-color: rgba(143,29,33,0.2); }}
    .error p {{ color: var(--error); }}
    pre {{ overflow-x: auto; padding: 12px; border-radius: 12px; background: #f5f5f5; }}
    .footer {{ margin-top: 18px; font-size: 0.92rem; opacity: 0.82; }}
    @media (max-width: 850px) {{
      .grid {{ grid-template-columns: 1fr; }}
      .metrics {{ grid-template-columns: 1fr 1fr; }}
    }}
    @media (max-width: 560px) {{
      .wrap {{ padding: 18px 14px 40px; }}
      .hero, .panel {{ border-radius: 18px; padding: 18px; }}
      .metrics {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <div class="eyebrow">Groundwater Forecasting</div>
      <h1>{APP_TITLE}</h1>
      <p class="sub">Choose a BGS Future Flows station and a target year. The app uses the saved seasonal linear regression models from this repository and automatically builds a model from a local station CSV when one is available.</p>
    </section>

    <div class="grid">
      <section class="panel">
        <h2>Run a prediction</h2>
        <form method="get" action="/">
          <label>
            Station
            <select name="station">{''.join(options)}</select>
          </label>
          <label>
            Year
            <input type="number" min="1951" max="2100" step="1" name="year" value="{year}" />
          </label>
          <button type="submit">Predict groundwater depth</button>
        </form>
        {metadata}
        <p class="footer">Tip: the Therfield Rectory station (`TL33_4`) already has generated model outputs in this repository, so it works immediately.</p>
      </section>
      {result_html or '<section class="panel"><h2>Prediction output</h2><p>Submit the form to view the seasonal groundwater prediction.</p></section>'}
    </div>
  </div>
</body>
</html>
"""


class GroundwaterRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path not in {"/", "/index.html"}:
            self.send_error(404, "Not found")
            return

        stations = station_lookup()
        default_station = next(iter(stations))
        params = parse_qs(parsed.query)
        station_code = params.get("station", [default_station])[0].strip().upper().replace("/", "_")
        if station_code not in stations:
            station_code = default_station

        try:
            year = int(params.get("year", ["2050"])[0])
        except ValueError:
            year = 2050

        result = None
        error = ""
        if parsed.query:
            try:
                model_json = ensure_station_model(station_code, base_dir=BASE_DIR)
                result = GroundwaterPredictionAgent().predict(model_json, year)
            except Exception as exc:
                error = str(exc)

        body = render_page(station_code=station_code, year=year, result=result, error=error).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

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
