from http.server import BaseHTTPRequestHandler
import os
import httpx

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        try:
            with httpx.Client(timeout=10.0) as client:
                client.post(
                    "https://api-inference.huggingface.co/models/chahatsaini1309/moodmatch-emotion-model",
                    headers={
                        "Authorization": f"Bearer {os.environ['HF_TOKEN']}",
                        "x-wait-for-model": "true",
                    },
                    json={"inputs": "warmup"},
                )
        except Exception:
            pass

        self.send_response(200)
        self.send_header("Content-type", "text/plain")
        self.end_headers()
        self.wfile.write(b"ok")

    def log_message(self, format, *args):
        pass  # silence default request logs
