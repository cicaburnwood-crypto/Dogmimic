from pathlib import Path
import json,subprocess,hashlib,cv2,numpy as np,sys
R=Path(__file__).resolve().parent
NEW=R/'output/EchoNav_full_demo_mosaic_manual_v6.mp4'
OLD=R.parent/'revision-v5/output/EchoNav_full_demo_mosaic_manual_v5.mp4'
def ahash(p):return subprocess.check_output(['ffmpeg','-v','error','-i',str(p),'-map','0:a:0','-c','copy','-f','hash','-hash','sha256','-']).decode().strip()
def frame(p,t):
 c=cv2.VideoCapture(str(p));c.set(1,round(t*60));ok,im=c.read();c.release();assert ok;return im
def main():
 p=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-show_format','-show_chapters','-of','json',str(NEW)]));v=next(x for x in p['streams'] if x['codec_type']=='video')
 assert (v['width'],v['height'],v['nb_frames'],v['avg_frame_rate'])==(1920,1080,'14922','60/1')
 assert float(p['format']['duration'])==248.7 and len(p['chapters'])==8
 h0,h1=ahash(OLD),ahash(NEW);assert h0==h1
 raw=subprocess.check_output(['ffmpeg','-v','error','-i',str(NEW),'-map','0:a:0','-af','atrim=start_sample=9892800','-f','f32le','-']);samples=np.frombuffer(raw,np.float32);peak=float(np.max(np.abs(samples)));assert peak==0
 # Retiming adds no frames: it relocates the existing five-second pause.
 unchanged={}
 for t in [1,69,79,160,180,204,220,248]:
  a,b=frame(OLD,t),frame(NEW,t);d=int(np.max(np.abs(a.astype(np.int16)-b.astype(np.int16))));unchanged[str(t)]=d;assert d==0
 (R/'inspect/final-frames').mkdir(exist_ok=True)
 for t in [119.4,120.15,120.2,120.7,122,124.5,125.2,125.4,127,140]:cv2.imwrite(str(R/'inspect/final-frames'/f'{t:.2f}.jpg'),frame(NEW,t))
 # All three cameras and the elapsed-time clock remain frozen together.
 a,b=frame(NEW,121.5),frame(NEW,123.5)
 camera_delta=float(np.abs(a[66:426].astype(float)-b[66:426]).mean());clock_delta=float(np.abs(a[980:1050,1300:1910].astype(float)-b[980:1050,1300:1910]).mean())
 assert camera_delta<1 and clock_delta<1,(camera_delta,clock_delta)
 subprocess.run(['ffmpeg','-v','error','-threads','4','-i',str(NEW),'-map','0:v:0','-map','0:a:0','-f','null','-'],check=True)
 r={'duration_seconds':248.7,'frames':14922,'resolution':[1920,1080],'fps':60,'chapters':8,'audio_payload_sha256':h1,'audio_identical_to_v5':True,'muted_tail_from_seconds':206.1,'muted_tail_peak':peak,'unchanged_segment_sample_pixel_max_difference':unchanged,'synchronized_camera_freeze_mean_difference':camera_delta,'frozen_clock_mean_difference':clock_delta,'explanatory_pause_global_seconds':[120.2,125.2],'source_timestamp_during_pause':9.6,'full_decode':'passed','bytes':NEW.stat().st_size,'sha256':hashlib.sha256(NEW.read_bytes()).hexdigest()}
 (R/'output/verification.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2),flush=True)
if __name__=='__main__':main()
