"""Loopback-only B2 control console, accessed through authenticated SSH forwarding."""
import argparse
import json
import mimetypes
import secrets
import signal
import threading
import time
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit
from backend import OfflineBackend,SDKBackend
from control import Engine,Rejected

ROOT=Path(__file__).resolve().parent


class Application:
    def __init__(self,backend,log_dir):
        self.backend=backend;self.engine=Engine(backend);self.lock=threading.RLock()
        self.csrf=secrets.token_urlsafe(32);self.seen=OrderedDict();self.stop_event=threading.Event()
        self.log_dir=Path(log_dir);self.log_dir.mkdir(parents=True,exist_ok=True)
        self.last_logged=0;self.thread=threading.Thread(target=self.run,daemon=True);self.thread.start()

    def run(self):
        deadline=time.monotonic()
        while not self.stop_event.is_set():
            with self.lock:
                try:self.engine.tick()
                except Exception as e:self.engine._failure(e)
                trace=getattr(self.backend,'drain_trace',lambda:[])()
                if trace:
                    for record in trace:
                        name='tilt_locomotion_trace.jsonl' if record.get('skill')=='tilt_locomotion' else 'attitude_trace.jsonl' if record.get('skill')=='attitude' else 'hindstand_trace.jsonl'
                        with (self.log_dir/name).open('a') as f:f.write(json.dumps(record,ensure_ascii=False)+'\n')
                for event in reversed(self.engine.events):
                    if event['id']>self.last_logged:
                        with (self.log_dir/'events.jsonl').open('a') as f:f.write(json.dumps(event,ensure_ascii=False)+'\n')
                        self.last_logged=event['id']
            deadline+=.02
            self.stop_event.wait(max(0,deadline-time.monotonic()))
            if time.monotonic()-deadline>.1:deadline=time.monotonic()

    def status(self):
        with self.lock:
            result=self.engine.status();result['csrf']=self.csrf
            result['ownership']=dict(name=getattr(self.backend,'owner_name',None),
                                      error=getattr(self.backend,'owner_error',None))
            result['motor_packets_sent']=getattr(self.backend,'writer_packets',0)
            return result

    def command(self,body):
        if set(body)-{'id','action','value','revision'}:raise Rejected('请求包含不支持的字段')
        key=body.get('id',secrets.token_hex(16))
        if not isinstance(key,str) or not 8<=len(key)<=80:raise Rejected('请求必须带唯一编号')
        if body.get('action') not in ('select','hindstand','attitude'):raise Rejected('未知网页操作')
        with self.lock:
            signature=json.dumps([body.get('action'),body.get('value'),body.get('revision')],sort_keys=True,allow_nan=False)
            if key in self.seen:
                if self.seen[key][0]!=signature:raise Rejected('同一请求编号不能改变操作')
                return self.seen[key][1]
            self.engine.telemetry=self.backend.telemetry()
            self.engine.request(body.get('action'),body.get('value'),body.get('revision'))
            result={'ok':True,'state':self.engine.status()}
            self.seen[key]=(signature,result)
            while len(self.seen)>256:self.seen.popitem(last=False)
            return result

    def close(self):
        self.stop_event.set();self.thread.join(2);self.backend.close()


def make_handler(app,port):
    allowed={f'127.0.0.1:{port}',f'localhost:{port}'}
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass

        def send(self,code,body,content_type='application/json; charset=utf-8'):
            data=json.dumps(body,ensure_ascii=False,allow_nan=False).encode() if isinstance(body,dict) else body
            self.send_response(code);self.send_header('Content-Type',content_type)
            self.send_header('Content-Length',str(len(data)));self.send_header('Cache-Control','no-store')
            self.send_header('X-Content-Type-Options','nosniff');self.send_header('Referrer-Policy','no-referrer')
            self.send_header('Content-Security-Policy',"default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'")
            self.end_headers()
            try:self.wfile.write(data)
            except (BrokenPipeError,ConnectionResetError):pass

        def valid_host(self):
            if self.headers.get('Host') not in allowed:
                self.send(403,{'error':'仅接受通过 SSH 隧道访问的本机来源'});return False
            return True

        def do_GET(self):
            if not self.valid_host():return
            path=urlsplit(self.path).path
            if path=='/api/status':self.send(200,app.status());return
            if path=='/api/health':self.send(200,{'ok':True,'robot_connected':app.status()['connected']});return
            names={'/':'index.html','/app.js':'app.js','/style.css':'style.css','/robot.svg':'robot.svg','/favicon.svg':'favicon.svg'}
            if path not in names:self.send(404,{'error':'Not found'});return
            p=ROOT/'web'/names[path]
            self.send(200,p.read_bytes(),mimetypes.guess_type(p)[0] or 'application/octet-stream')

        def do_POST(self):
            if not self.valid_host():return
            origin=self.headers.get('Origin','')
            if origin not in {'http://'+host for host in allowed}:
                self.send(403,{'error':'跨站控制请求被拒绝'});return
            if not secrets.compare_digest(self.headers.get('X-B2-CSRF',''),app.csrf):
                self.send(403,{'error':'控制会话已失效，请刷新页面'});return
            if self.path!='/api/command':self.send(404,{'error':'Not found'});return
            try:
                length=int(self.headers.get('Content-Length','0'))
                if not 0<length<=65536:raise ValueError('请求长度超限')
                body=json.loads(self.rfile.read(length))
                if not isinstance(body,dict):raise ValueError('无效请求')
                self.send(200,app.command(body))
            except Rejected as e:self.send(409,{'ok':False,'error':str(e)})
            except (ValueError,TypeError) as e:self.send(400,{'ok':False,'error':str(e)})
            except Exception:
                self.send(503,{'ok':False,'error':'机器人控制接口异常，操作未确认；请查看状态'})
    return Handler


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backend',choices=['offline','sdk'],default='offline')
    p.add_argument('--bundle',default=str(ROOT.parents[1]/'deployments/b2_192_168_123_164_20261101'))
    p.add_argument('--interface',default='eth0');p.add_argument('--port',type=int,default=8765)
    p.add_argument('--log-dir',default=str(ROOT/'logs'));args=p.parse_args()
    backend=SDKBackend(args.bundle,args.interface) if args.backend=='sdk' else OfflineBackend(bundle=args.bundle)
    app=Application(backend,args.log_dir)
    server=ThreadingHTTPServer(('127.0.0.1',args.port),make_handler(app,args.port));server.daemon_threads=True
    def stop(*_):threading.Thread(target=server.shutdown,daemon=True).start()
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    print(f'B2 Console: http://127.0.0.1:{args.port} | backend={args.backend}',flush=True)
    try:server.serve_forever(poll_interval=.1)
    finally:server.server_close();app.close()


if __name__=='__main__':main()
