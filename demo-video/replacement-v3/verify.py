from pathlib import Path
import json,subprocess,hashlib,cv2,numpy as np,sys
R=Path(__file__).resolve().parent;new=R/'output/EchoNav_full_demo_mosaic_manual_v3.mp4';old=R.parent/'redaction/manual-v2/output/EchoNav_full_demo_mosaic_manual_v2.mp4'
probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-show_chapters','-show_format','-of','json',str(new)]));v=next(s for s in probe['streams'] if s['codec_type']=='video');a=next(s for s in probe['streams'] if s['codec_type']=='audio')
assert (v['width'],v['height'],v['r_frame_rate'],int(v['nb_frames']))==(1920,1080,'60/1',14922)
assert abs(float(v['duration'])-248.7)<1e-4 and abs(float(a['duration'])-248.7)<.03
assert len(probe['chapters'])==8
hashes=[]
for p in [old,new]:
 data=subprocess.check_output(['ffmpeg','-v','error','-i',str(p),'-map','0:a:0','-c','copy','-f','adts','pipe:1']);hashes.append(hashlib.sha256(data).hexdigest())
assert hashes[0]==hashes[1]
def fr(p,f):
 c=cv2.VideoCapture(str(p));c.set(cv2.CAP_PROP_POS_FRAMES,f);ok,im=c.read();c.release();assert ok;return im
# Unedited video segments are copied. Confirm their final placement and decoded pixels.
checks=[]
for t in [1,79,90,162,180,204,220,248]:
 x=fr(old,round(t*60));y=fr(new,round(t*60));assert np.array_equal(x,y),t;checks.append(t)
# The last method-overview slide begins at frame 3649, and ends at frame 4662.
roi=(80,650,355,959);sample_errors=[]
for f in [3648,3649,4140,4661,4662]:
 x=fr(old,f);y=fr(new,f);cv2.imwrite(str(R/'inspect'/f'final-frame-{f}.png'),y)
 if 3649<=f<4662:
  outside=np.ones((1080,1920),bool);outside[644:966,74:362]=False
  mae=float(np.abs(x.astype(float)-y.astype(float))[outside].mean());assert mae<1.5,(f,mae)
  delta=float(np.abs(x[701:907,80:355].astype(float)-y[701:907,80:355].astype(float)).mean());assert delta>20,(f,delta)
  sample_errors.append({'frame':f,'outside_edit_mean_pixel_error':mae,'replacement_mean_pixel_change':delta})
 else:
  mae=float(np.abs(x.astype(float)-y.astype(float)).mean());assert mae<1.5,(f,mae)
subprocess.run(['ffmpeg','-hide_banner','-v','error','-i',str(new),'-f','null','-'],check=True)
report={'file':str(new),'duration_seconds':248.7,'frames':14922,'resolution':[1920,1080],'fps':60,'audio_bitstream_identical_to_v2':True,'audio_sha256':hashes[1],'full_decode_passed':True,'unchanged_segments_pixel_checks_seconds':checks,'replacement_checks':sample_errors,'size_bytes':new.stat().st_size,'sha256':hashlib.sha256(new.read_bytes()).hexdigest(),'replacement':json.loads((R/'replacement.json').read_text())}
(R/'output/verification.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:report[k] for k in ['duration_seconds','frames','audio_bitstream_identical_to_v2','full_decode_passed','size_bytes','sha256']},indent=2),flush=True)
