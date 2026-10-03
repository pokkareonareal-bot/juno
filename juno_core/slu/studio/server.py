"""The studio's web server: a page, a JSON API, and a stream of events.

    python -m juno_core.slu studio            # then open the URL it prints

Standard library only. It listens on 127.0.0.1 and nowhere else, and two
things keep other web pages from using it -- a page on any site can try to
talk to localhost, and this one controls a microphone:

  - every API call must carry a token minted when the server starts, which
    is only ever handed to the studio's own page (other origins cannot read
    that page, so cannot learn it);
  - requests whose Host header is not this server's own address are refused,
    which defeats DNS rebinding (a hostile name that resolves to 127.0.0.1).
"""

from __future__ import annotations

import json
import mimetypes
import secrets
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

STATIC = Path(__file__).resolve().parent / "static"


def make_server(studio, host: str = "127.0.0.1", port: int = 8765) -> ThreadingHTTPServer:
    token = secrets.token_urlsafe(24)

    class Handler(BaseHTTPRequestHandler):
        server_version = "JunoStudio/1"

        def log_message(self, fmt, *args):     # quiet: the page shows what matters
            pass

        # -- guards ------------------------------------------------------------

        def _host_ok(self) -> bool:
            port_ = self.server.server_address[1]
            allowed = {f"127.0.0.1:{port_}", f"localhost:{port_}", f"[::1]:{port_}"}
            return self.headers.get("Host", "") in allowed

        def _token_ok(self, supplied: str | None) -> bool:
            return bool(supplied) and secrets.compare_digest(supplied, token)

        def _send(self, status: int, body: bytes, ctype: str, extra: dict | None = None) -> None:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, status: int = 200) -> None:
            self._send(status, json.dumps(obj, default=str, allow_nan=False).encode(),
                       "application/json")

        # -- GET ---------------------------------------------------------------

        def do_GET(self):
            if not self._host_ok():
                return self._send(HTTPStatus.MISDIRECTED_REQUEST, b"wrong host", "text/plain")
            url = urlparse(self.path)
            if url.path in ("/", "/index.html"):
                page = (STATIC / "index.html").read_text(encoding="utf-8")
                page = page.replace("__STUDIO_TOKEN__", token)
                csp = ("default-src 'self'; script-src 'self'; style-src 'self'; "
                       "img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'")
                return self._send(200, page.encode(), "text/html; charset=utf-8",
                                  {"Content-Security-Policy": csp})
            if url.path.startswith("/static/"):
                return self._static(url.path[len("/static/"):])
            if url.path == "/api/events":
                if not self._token_ok(parse_qs(url.query).get("t", [None])[0]):
                    return self._send(403, b"forbidden", "text/plain")
                return self._events()
            if not self._token_ok(self.headers.get("X-Studio-Token")):
                return self._send(403, b"forbidden", "text/plain")
            routes = {"/api/status": studio.status, "/api/models": studio.list_models,
                      "/api/sessions": studio.list_sessions}
            fn = routes.get(url.path)
            if fn is None:
                return self._send(404, b"not found", "text/plain")
            self._call(fn)

        def _static(self, name: str) -> None:
            path = (STATIC / name).resolve()
            if STATIC not in path.parents or not path.is_file():
                return self._send(404, b"not found", "text/plain")
            ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            self._send(200, path.read_bytes(), ctype)

        def _events(self) -> None:
            q = studio.events.subscribe()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Accel-Buffering", "no")
            self.end_headers()
            try:
                self._event("status", studio.status())
                while True:
                    try:
                        kind, data = q.get(timeout=15)
                    except Exception:
                        self.wfile.write(b": keep-alive\n\n")
                        self.wfile.flush()
                        continue
                    self._event(kind, data)
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass
            finally:
                studio.events.unsubscribe(q)

        def _event(self, kind: str, data) -> None:
            payload = json.dumps(data, default=str, allow_nan=False)
            self.wfile.write(f"event: {kind}\ndata: {payload}\n\n".encode())
            self.wfile.flush()

        # -- POST --------------------------------------------------------------

        def do_POST(self):
            if not self._host_ok():
                return self._send(HTTPStatus.MISDIRECTED_REQUEST, b"wrong host", "text/plain")
            if not self._token_ok(self.headers.get("X-Studio-Token")):
                return self._send(403, b"forbidden", "text/plain")
            if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                return self._send(415, b"json only", "text/plain")
            try:
                length = min(int(self.headers.get("Content-Length", "0")), 1_000_000)
                body = json.loads(self.rfile.read(length) or b"{}")
            except (ValueError, json.JSONDecodeError):
                return self._json({"ok": False, "error": "bad JSON"}, 400)
            b = body.get
            routes = {
                "/api/try/start": lambda: studio.start_try(),
                "/api/listen/stop": lambda: studio.stop_listening(),
                "/api/try/state": lambda: studio.set_state(b("awaiting_answer"), b("timer_running")),
                "/api/try/label": lambda: studio.label(b("id"), b("addressed"), b("intent") or None),
                "/api/try/save": lambda: studio.save_labelled(b("consent"), b("room", ""),
                                                              b("speaker", "")),
                "/api/model": lambda: studio.set_model(b("path")),
                "/api/session/start": lambda: studio.start_session(
                    b("speakers") or [], b("room", ""), b("consent"), bool(b("confirmed")),
                    b("encoders")),
                "/api/session/record": lambda: studio.record_step(int(b("step"))),
                "/api/session/free": lambda: studio.record_free(b("label"), b("intent") or None,
                                                                b("speaker", "")),
                "/api/session/stop": lambda: studio.stop_recording(),
                "/api/session/goto": lambda: studio.goto_step(int(b("step"))),
                "/api/session/discard": lambda: studio.discard(b("id")),
                "/api/session/finish": lambda: studio.finish_session(),
                "/api/session/abandon": lambda: studio.abandon_session(),
                "/api/evaluate": lambda: studio.evaluate(b("model"), b("sessions") or [],
                                                         bool(b("awaiting_answer"))),
                "/api/train": lambda: studio.train(b("name", ""), b("encoder", ""),
                                                   b("targets", "gold"), b("sessions") or [],
                                                   bool(b("include_base")),
                                                   bool(b("allow_internal"))),
            }
            fn = routes.get(urlparse(self.path).path)
            if fn is None:
                return self._send(404, b"not found", "text/plain")
            self._call(fn)

        def _call(self, fn) -> None:
            try:
                result = fn()
            except (ValueError, KeyError, RuntimeError, FileNotFoundError, IndexError) as exc:
                msg = exc.args[0] if isinstance(exc, KeyError) and exc.args else str(exc)
                return self._json({"ok": False, "error": msg}, 400)
            except SystemExit as exc:
                return self._json({"ok": False, "error": str(exc)}, 400)
            except Exception as exc:      # noqa: BLE001 - report, never crash the server
                return self._json({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, 500)
            self._json({"ok": True, "result": result})

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    server.token = token                  # for tests and for printing the URL
    return server


def studio_main(args) -> int:
    import webbrowser

    from juno_core.slu.studio.engine import DEFAULT_ENCODERS, Studio

    config = None
    if args.config or Path("config.yaml").exists():
        from juno_core.config import load_config

        config = load_config(args.config)
    aliases = [a.strip() for a in (args.aliases or "").split(",") if a.strip()]
    if not aliases and config is not None:
        aliases = list((config.get("intent") or {}).get("assistant_aliases") or [])
    studio = Studio(config, root=Path.cwd(), stt=args.stt, aliases=aliases, name=args.name,
                    encoders=args.encoder or DEFAULT_ENCODERS, model=args.model,
                    simulate=args.simulate, judge=args.judge)
    studio.load(background=True)
    server = make_server(studio, port=args.port)
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    print(f"Juno Studio: {url}", flush=True)
    print("Models load in the background (the first time takes a minute). Ctrl+C to stop.",
          flush=True)
    if not args.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        studio.stop_listening()
        server.server_close()
    return 0
