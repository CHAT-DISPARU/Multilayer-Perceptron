# +------------------------------------+
# | CHAT-DISPARU // HEADER             |
# |       _---~~(~~-_.                 |
# |     _{        )   )                |
# |   ,   ) -~~- ( ,-' )_              |
# |  (  `-,_..`., )-- '_,)             |
# | ( ` _)  (  -~( -_ `,  }            |
# | (_-  _  ~_-~~~~`,  ,' )            |
# |   `~ -^(    __;-,((()))            |
# |         ~~~~ {_ -_(())             |
# |                `\  }               |
# |                  { }               |
# |                                    |
# | File    : gui.py                   |
# | Login   : CHAT-DISPARU             |
# | Email   : gaetanfranca@proton.me   |
# | Created : 21/11/2026 21:12:10      |
# | Updated : 21/11/2026 21:12:10      |
# | END HEADER                         |
# +------------------------------------+

import json
import os
import sys
import threading
import csv
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from webui import api

ROOT = Path(__file__).resolve().parent
STATIC = (ROOT / "webui" / "static").resolve()
PORT = 8765
MIME = {
	".html": "text/html; charset=utf-8",
	".css": "text/css; charset=utf-8",
	".js": "application/javascript; charset=utf-8",
	".svg": "image/svg+xml",
}


class Handler(BaseHTTPRequestHandler):
	def _send(self, code, body, ctype="application/json"):
		if isinstance(body, (dict, list)):
			body = json.dumps(body).encode()
		self.send_response(code)
		self.send_header("Content-Type", ctype)
		self.send_header("Content-Length", str(len(body)))
		self.end_headers()
		self.wfile.write(body)

	def do_GET(self):
		path = urlparse(self.path).path
		if path == "/":
			path = "/index.html"
		file = (STATIC / path.lstrip("/")).resolve()
		if STATIC not in file.parents or not file.is_file():
			return self._send(404, {"error": "not found"})
		self._send(200, file.read_bytes(), MIME.get(file.suffix, "application/octet-stream"))

	def do_POST(self):
		route = urlparse(self.path).path
		handler = api.ROUTES.get(route)
		if handler is None:
			return self._send(404, {"error": "unknown route"})
		try:
			length = int(self.headers.get("Content-Length", 0))
			payload = json.loads(self.rfile.read(length) or b"{}")
			self._send(200, handler(payload))
		except (BrokenPipeError, ConnectionResetError):
			pass
		except json.JSONDecodeError:
			self._send(400, {"error": "invalid JSON"})
		except (ValueError, OSError, KeyError, TypeError, AttributeError, csv.Error) as e:
			self._send(400, {"error": str(e)})

	def log_message(self, *args):
		pass


def main():
	os.chdir(ROOT)
	try:
		server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
	except OSError as e:
		print(f"[ERROR] Cannot start the server on port {PORT}: {e.strerror}", file=sys.stderr)
		print(f"> Another instance is probably already running (try: lsof -i :{PORT})", file=sys.stderr)
		sys.exit(1)
	url = f"http://127.0.0.1:{PORT}"
	print(f"> MLP Studio running on {url} (Ctrl+C to stop)")
	threading.Timer(0.5, webbrowser.open, args=(url,)).start()
	try:
		server.serve_forever()
	except KeyboardInterrupt:
		print("\n> bye")


if __name__ == "__main__":
	main()
