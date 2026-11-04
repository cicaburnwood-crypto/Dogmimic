from pathlib import Path
import os,sys,subprocess,json,time,datetime,hashlib,shutil
root=Path(__file__).resolve().parents[2];base=Path(__file__).resolve().parent;os.chdir(root)
assert json.loads((root/'runs/g1_boundary_sampling_cuda_smoke/report.json').read_text())['passed']
assert (root/'runs/g1_jump_effective_v7_20261101_205801/EVAL_DONE.json').exists()
stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S');out=root/'runs'/('g1_jump_boundary_v8_'+stamp);out.mkdir()
source=root/'runs/g1_jump_effective_v7_20261101_205801/best.pt'
(out/'source.pt').write_bytes(source.read_bytes())
assert hashlib.sha256((base/'evaluate_jump.py').read_bytes()).hexdigest()==hashlib.sha256((root/'experiments/g1_jump_robust_v5_candidate/evaluate_jump.py').read_bytes()).hexdigest()
env=os.environ.copy();env.update(CUDA_DEVICE_ORDER='PCI_BUS_ID',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='4',PYTHONUNBUFFERED='1',TORCHINDUCTOR_COMPILE_THREADS='2',INITIAL_POLICY=str(out/'source.pt'))
py=str(root/'.venv/bin/python');train=[py,'-m','torch.distributed.run','--standalone','--nproc_per_node=2',str(base/'train.py'),'--motions','data/processed/g1_standing_jump_060m.npz','--out',str(out),'--num-envs','1024','--buffer-steps','512','--warmup-steps','128','--iterations','40000','--wall-hours','6','--lr','0.0001','--use-critic','false','--lr-decay-iters','40000','--wm-batch','1024','--pi-batch','2048','--log-every','100','--ckpt-every','1000','--eval-every','1000']
f=(out/'train.log').open('w');t=subprocess.Popen(train,env=dict(env,CUDA_VISIBLE_DEVICES='4,5'),stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
g=(out/'eval.log').open('w');e=subprocess.Popen([py,str(base/'eval_worker.py'),str(out)],env=dict(env,CUDA_VISIBLE_DEVICES='6'),stdout=g,stderr=subprocess.STDOUT,start_new_session=True)
archive=out/'source';archive.mkdir()
for fsrc in base.glob('*'):
 if fsrc.suffix in ('.py','.md') or fsrc.name=='measured_start.json':
  (archive/fsrc.name).write_bytes(fsrc.read_bytes())
shutil.copytree(base/'engine',archive/'engine',ignore=shutil.ignore_patterns('__pycache__'))
launch=dict(world_model_control='four_consumed_5ms_targets',reset_pose_distribution=dict(measured_endpoint=.25,reference_endpoint=.25,continuous=.5),out=str(out),started=time.time(),started_iso=datetime.datetime.now().astimezone().isoformat(),train_pid=t.pid,eval_pid=e.pid,train_gpus=[4,5],eval_gpu=6,iterations=40000,training_hours_limit=6,command=train,source_sha256=hashlib.sha256(Path(env['INITIAL_POLICY']).read_bytes()).hexdigest())
(base/'latest_launch.json').write_text(json.dumps(launch,indent=2));(out/'launch.json').write_text(json.dumps(launch,indent=2));print(json.dumps(launch),flush=True)

subprocess.Popen([py,str(base/'watch.py'),str(out)],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
