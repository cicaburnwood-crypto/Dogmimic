from pathlib import Path
import subprocess,json,hashlib,cv2,numpy as np
R=Path(__file__).resolve().parent
NEW=R/'output/EchoNav_full_demo_mosaic_manual_v5.mp4'
OLD=R.parent/'revision-v4/output/EchoNav_full_demo_mosaic_manual_v4.mp4'
def audio_hash(p):
 return subprocess.check_output(['ffmpeg','-v','error','-i',str(p),'-map','0:a:0','-c','copy','-f','hash','-hash','sha256','-']).decode().strip()
def frame(p,t):
 c=cv2.VideoCapture(str(p));c.set(cv2.CAP_PROP_POS_FRAMES,round(t*60));ok,im=c.read();c.release();assert ok;(R/'inspect/final-frames').mkdir(exist_ok=True)
 return im
def main():
 probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-show_chapters','-show_format','-of','json',str(NEW)]))
 v=next(z for z in probe['streams'] if z['codec_type']=='video');a=next(z for z in probe['streams'] if z['codec_type']=='audio')
 assert (v['width'],v['height'],v['nb_frames'],v['avg_frame_rate'])==(1920,1080,'14922','60/1')
 assert abs(float(probe['format']['duration'])-248.7)<.00001 and len(probe['chapters'])==8
 h0,h1=audio_hash(OLD),audio_hash(NEW);assert h0==h1
 raw=subprocess.check_output(['ffmpeg','-v','error','-i',str(NEW),'-map','0:a:0','-af','atrim=start_sample=9892800','-f','f32le','-'])
 samples=np.frombuffer(raw,dtype=np.float32);peak=float(np.abs(samples).max());assert peak==0
 unchanged={}
 for t in [1,69,79,160,180,204,220,248]:
  b0,b1=frame(OLD,t),frame(NEW,t);delta=int(np.max(np.abs(b0.astype(np.int16)-b1.astype(np.int16))));unchanged[str(t)]=delta;assert delta==0,(t,delta)
 for t in [97,97.766667,111,111.9,113,115.4,120.2,123,125,130,135.1,135.7,137,140.8,142.2]:
  cv2.imwrite(str(R/'inspect/final-frames'/f'{t:.3f}.jpg'),frame(NEW,t))
 # Decode every frame and audio packet, independent of the container counts.
 subprocess.run(['ffmpeg','-v','error','-threads','4','-i',str(NEW),'-map','0:v:0','-map','0:a:0','-f','null','-'],check=True)
 result={'duration_seconds':float(probe['format']['duration']),'resolution':[1920,1080],'fps':60,'frames':14922,'chapters':len(probe['chapters']),'audio_payload_sha256':h1,'audio_identical_to_v4':True,'mute_from_seconds':206.1,'muted_tail_peak':peak,'unchanged_segment_sample_max_pixel_difference':unchanged,'full_decode':'passed','bytes':NEW.stat().st_size,'sha256':hashlib.sha256(NEW.read_bytes()).hexdigest()}
 (R/'output/verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
