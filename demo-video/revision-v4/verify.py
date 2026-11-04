from pathlib import Path
import cv2,numpy as np,subprocess,json,hashlib
R=Path(__file__).resolve().parent
new=R/'output/EchoNav_full_demo_mosaic_manual_v4.mp4'
old=R.parent/'replacement-v3/output/EchoNav_full_demo_mosaic_manual_v3.mp4'
p=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-show_chapters','-show_format','-of','json',str(new)]))
v=next(s for s in p['streams'] if s['codec_type']=='video');a=next(s for s in p['streams'] if s['codec_type']=='audio')
assert (v['width'],v['height'],v['r_frame_rate'],int(v['nb_frames']))==(1920,1080,'60/1',14922)
assert abs(float(v['duration'])-248.7)<1e-5 and abs(float(a['duration'])-248.7)<.03
assert len(p['chapters'])==8

def frame(path,t):
 cap=cv2.VideoCapture(str(path));cap.set(cv2.CAP_PROP_POS_FRAMES,round(t*60));ok,im=cap.read();cap.release();assert ok;return im
checks=[]
for t in [1,8,69,79,160,204,210,220,248]:
 x=frame(old,t);y=frame(new,t);assert np.array_equal(x,y),t;checks.append(t)
# Compare a copied prefix of AAC packets: all earlier narration remains untouched.
pre=[]
for path in [old,new]:
 data=subprocess.check_output(['ffmpeg','-v','error','-i',str(path),'-t','206','-map','0:a:0','-c','copy','-f','adts','pipe:1'])
 pre.append(hashlib.sha256(data).hexdigest())
assert pre[0]==pre[1],pre
raw=subprocess.check_output(['ffmpeg','-v','error','-i',str(new),'-ss','206.1','-map','0:a:0','-f','f32le','-acodec','pcm_f32le','pipe:1'])
samples=np.frombuffer(raw,np.float32);peak=float(np.max(np.abs(samples)));rms=float(np.sqrt(np.mean(samples**2)))
assert peak<1e-5,peak
# Export frames from the actual final encode for visual review.
for t in [90.2,92.575,93.05,94.45,130.2,131.95,135.95,137.55,138.45,139.95,140.1,163.966667,167.591667,169.591667,170.216667,172.091667,175.466667]:
 y=frame(new,t);cv2.imwrite(str(R/'inspect'/f'final-{t:.3f}.png'),y)
subprocess.run(['ffmpeg','-v','error','-i',str(new),'-f','null','-'],check=True)
report={'file':str(new),'duration_seconds':248.7,'frames':14922,'width':1920,'height':1080,'fps':60,'chapters':len(p['chapters']),'full_decode_passed':True,'unchanged_segments_exact_decoded_frame_checks_seconds':checks,'audio_prefix_206_seconds_AAC_identical':True,'audio_prefix_sha256':pre[0],'muted_interval_seconds':[206.1,248.7],'muted_peak_linear':peak,'muted_rms_linear':rms,'size_bytes':new.stat().st_size,'sha256':hashlib.sha256(new.read_bytes()).hexdigest()}
(R/'output/verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
