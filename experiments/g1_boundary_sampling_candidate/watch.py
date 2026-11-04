from pathlib import Path
import json,time,sys,os,signal
p=Path(sys.argv[1]);a=json.loads((p/'launch.json').read_text())
def alive(pid):
 try:return Path('/proc/%s/stat'%pid).read_text().split(') ')[1][0]!='Z'
 except FileNotFoundError:return False
while alive(a['train_pid']):time.sleep(10)
if not (p/'DONE').exists():
 (p/'TRAIN_FAILED.json').write_text(json.dumps(dict(time=time.time(),reason='training exited without DONE; inspect train.log')))
 if alive(a['eval_pid']):
  cmd=Path('/proc/%s/cmdline'%a['eval_pid']).read_bytes()
  if str(Path(__file__).resolve().with_name('eval_worker.py')).encode() in cmd:os.kill(a['eval_pid'],signal.SIGTERM)
else:
 while alive(a['eval_pid']) and not (p/'EVAL_DONE.json').exists():time.sleep(10)
(p/'pipeline_timing.json').write_text(json.dumps(dict(finished=time.time(),elapsed=time.time()-a['started'],eval_done=(p/'EVAL_DONE.json').exists())))
