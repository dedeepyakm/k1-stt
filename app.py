"""Operator console: load the STT pipeline once, then serve the web UI.

    python app.py                 start and open the console in the browser
    python app.py --no-browser    start the server only
"""
import argparse
import threading
import time
import webbrowser

from stt.config import load_config
from stt.pipeline import STTPipeline
from stt.server import create_server


def main() -> None:
    parser = argparse.ArgumentParser(description="K1 voice command console")
    parser.add_argument("--config", help="path to a config.yaml (default: project config)")
    parser.add_argument("--no-browser", action="store_true", help="do not open a browser")
    args = parser.parse_args()

    cfg = load_config(args.config)
    print("Loading model (one time) ...")
    t0 = time.perf_counter()
    with STTPipeline(cfg) as pipe:
        server = create_server(pipe, cfg.ui.host, cfg.ui.port, cfg.ui.history_size)
        host = "127.0.0.1" if cfg.ui.host in ("0.0.0.0", "") else cfg.ui.host
        url = f"http://{host}:{cfg.ui.port}"
        t = pipe.transcriber
        print(f"Ready in {time.perf_counter() - t0:.1f}s on {t.device}/{t.compute_type} "
              f"| outputs: {', '.join(cfg.output.sinks)}")
        print(f"Console: {url}   (Ctrl+C to stop)")
        if not args.no_browser:
            threading.Timer(0.5, webbrowser.open, args=(url,)).start()
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()
    print("Console stopped.")


if __name__ == "__main__":
    main()