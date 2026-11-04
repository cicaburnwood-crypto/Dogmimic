"""G1 console with model-backed offline and SDK adapters."""
import argparse
import importlib
import json
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ACTIONS = {'jump_60cm': '立定跳远 60 cm', 'funky_chicken': 'Funky Chicken'}
ROOT = Path(__file__).parent / 'web'

class OfflineBackend:
    def __init__(self):
        from backend import create_offline_backend
        self.backend=create_offline_backend()
    def status(self):
        return self.backend.status()
    def execute(self, action):
        raise RuntimeError('G1 尚未连接')

class Controller:
    def __init__(self, backend):
        self.backend = backend
        self.lock = threading.Lock()
        self.events = []
    def status(self):
        with self.lock:
            return {**self.backend.status(), 'events': self.events[-30:]}
    def execute(self, action):
        with self.lock:
            if action not in ACTIONS:
                raise ValueError('未知动作')
            state = self.backend.status()
            if state.get('connected') is not True:
                raise ValueError('G1 尚未连接')
            if state.get('stable_standing') is not True:
                raise ValueError('请在稳定站立时执行动作')
            # Adapter rechecks current posture atomically when accepting the action.
            receipt = self.backend.execute(action)
            self.events.append({'time': datetime.now(timezone.utc).isoformat(),
                                'text': '已提交：' + ACTIONS[action]})
            return {'ok': True, 'receipt': receipt}

def handler(controller):
    class Handler(BaseHTTPRequestHandler):
        def respond(self, code, payload):
            data = json.dumps(payload, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(data)
        def do_GET(self):
            path = self.path.split('?')[0]
            if path == '/api/status':
                try:
                    self.respond(200, controller.status())
                except Exception:
                    self.respond(503, {'error': '设备状态读取失败'})
                return
            if path == '/models.json':
                self.respond(200, controller.status().get('models', []))
                return
            name = 'index.html' if path == '/' else path.lstrip('/')
            if name not in {'index.html', 'style.css', 'app.js', 'robot.svg', 'favicon.svg', 'models.json'}:
                self.send_error(404)
                return
            types = {'html': 'text/html', 'css': 'text/css', 'js': 'text/javascript', 'svg': 'image/svg+xml', 'json': 'application/json'}
            self.send_response(200)
            self.send_header('Content-Type', types[name.rsplit('.', 1)[1]] + '; charset=utf-8')
            self.end_headers()
            self.wfile.write((ROOT / name).read_bytes())
        def do_POST(self):
            if self.path != '/api/action':
                self.send_error(404)
                return
            # Same-origin transport validation; not an operator unlock step.
            origin = self.headers.get('Origin')
            if origin != 'http://' + self.headers.get('Host', '') or self.headers.get('Content-Type') != 'application/json':
                self.respond(403, {'error': '请求来源不匹配'})
                return
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 1024:
                    raise ValueError('请求格式错误')
                body = json.loads(self.rfile.read(length))
                self.respond(200, controller.execute(body.get('action')))
            except (ValueError, TypeError, AttributeError) as exc:
                self.respond(409, {'error': str(exc)})
            except Exception:
                self.respond(503, {'error': '动作接口调用失败，请查看设备端记录'})
    return Handler

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8767)
    parser.add_argument('--backend', help='Adapter factory as module:factory; default loads models without DDS')
    args = parser.parse_args()
    if args.backend:
        module, factory = args.backend.split(':', 1)
        backend = getattr(importlib.import_module(module), factory)()
    else:
        backend = OfflineBackend()
    server = ThreadingHTTPServer(('127.0.0.1', args.port), handler(Controller(backend)))
    print(f'G1 console: http://127.0.0.1:{args.port}', flush=True)
    server.serve_forever()
