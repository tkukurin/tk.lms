# %%
from __future__ import annotations

import http.server, json, sys, threading, webbrowser
from dataclasses import dataclass, field
from pathlib import Path

import simple_parsing as sp

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Cfg:
  records: Path | None = None
  vibes: Path | None = None
  port: int = 8877
  no_open: bool = False
  datadir: Path = field(
    default_factory=lambda: ROOT / "data" / "out"
                            / "model-meta-crawler")


def find_latest(datadir: Path, glob: str) -> Path | None:
  candidates = sorted(datadir.rglob(glob), key=lambda p: p.stat().st_mtime, reverse=True)
  return candidates[0] if candidates else None


def build_handler(cfg: Cfg, records_path: Path | None,
                  vibes_path: Path | None):
  viewer = Path(__file__).resolve().parent / "viewer.html"
  viewer_html = viewer.read_text(encoding="utf-8")

  # Inject auto-load script before </body>
  records_json = "null"
  vibes_json = "null"
  if records_path and records_path.exists():
    records_json = f"[{','.join(records_path.read_text().splitlines())}]"
  if vibes_path and vibes_path.exists():
    vibes_json = f"[{','.join(vibes_path.read_text().splitlines())}]"

  boot_script = f"""
<script>
(function() {{
  const _records = {records_json};
  const _vibes = {vibes_json};
  if (_records) {{
    allRecords = _records;
    flatRows = allRecords.map(flattenRecord);
    initTable();
  }}
  if (_vibes) {{
    vibeSignals = _vibes;
    vibesByModel = {{}};
    for (const s of vibeSignals) {{
      const k = s.model_key || "";
      if (!vibesByModel[k]) vibesByModel[k] = [];
      vibesByModel[k].push(s);
    }}
  }}
  updateStatus();
}})();
</script>
"""
  patched_html = viewer_html.replace("</body>", boot_script + "</body>")

  class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
      if self.path == "/" or self.path == "/index.html":
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(patched_html.encode())
      else:
        self.send_error(404)

    def log_message(self, fmt, *args):
      pass  # silence request logs

  return Handler


def main() -> None:
  cfg = sp.parse(Cfg)

  records_path = cfg.records or find_latest(
    cfg.datadir, "normalized/modelsdev_records.jsonl")
  vibes_path = cfg.vibes or find_latest(
    cfg.datadir, "vibes/vibes_signals.jsonl")

  print(f"records: {records_path or 'none found'}")
  print(f"vibes:   {vibes_path or 'none found'}")

  Handler = build_handler(cfg, records_path, vibes_path)
  server = http.server.HTTPServer(("127.0.0.1", cfg.port), Handler)
  url = f"http://127.0.0.1:{cfg.port}"
  print(f"serving on {url}")

  if not cfg.no_open:
    threading.Timer(0.3, lambda: webbrowser.open(url)).start()

  try: server.serve_forever()
  except KeyboardInterrupt: pass
  server.server_close()
  print("\nshutdown")


if __name__ == "__main__":
  main()
