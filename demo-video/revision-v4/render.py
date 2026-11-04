from pathlib import Path
import cv2,json,subprocess,sys,time,numpy as np
from manual_v4 import Revision,R,V2,CARTONS,DOG1,DOG2,FIRE,POV,GLASS,TOP_PRINT,TOP_RETURN,POV_RETURN,POV_BACK
from inspect_frames import SRC,NAMES
cv2.setNumThreads(2)
OLD=R.parent/'replacement-v3/output/EchoNav_full_demo_mosaic_manual_v3.mp4'
def render(c):
 cap=cv2.VideoCapture(str(SRC/NAMES[c]));n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));m=Revision(c)
 log=(R/'inspect'/f'render-{c}.log').open('w')
 p=subprocess.Popen(['ffmpeg','-y','-hide_banner','-loglevel','warning','-f','rawvideo','-pixel_format','bgr24','-video_size','1920x1080','-framerate','60','-i','pipe:0','-an','-c:v','libx264','-preset','fast','-crf','17','-profile:v','high','-level:v','4.2','-threads','3','-pix_fmt','yuv420p','-video_track_timescale','60000','-map_metadata','-1','-movflags','+faststart',str(R/'segments'/NAMES[c])],stdin=subprocess.PIPE,stderr=log)
 st=time.monotonic();last=None
 for f in range(n):
  ok,im=cap.read();assert ok,(c,f)
  if f%2==0:last=m.edited(f,im)
  p.stdin.write(last.tobytes())
  if f%600==0:print(f'clip {c}: {f}/{n} frames, {time.monotonic()-st:.1f}s',flush=True)
 p.stdin.close();rc=p.wait();cap.release();log.close();assert rc==0
 print(f'clip {c}: done {n} frames, {time.monotonic()-st:.1f}s',flush=True)
def finish():
 timeline=json.loads((R.parent/'assembly/output/timeline.json').read_text());entries=[]
 for j in timeline:
  name=Path(j['output']).name
  if name in (NAMES[3],NAMES[4]):p=R/'segments'/name
  elif name=='02-clip-2.mp4':p=R.parent/'replacement-v3/segments'/name
  else:p=V2/'segments'/name
  streams=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-of','json',str(p)]))['streams']
  v=next(x for x in streams if x['codec_type']=='video');assert int(v['nb_frames'])==j['frames']
  entries.append(f"file '{p}'\nduration {j['duration']:.9f}\n")
 (R/'segments/concat.txt').write_text(''.join(entries))
 dest=R/'output/EchoNav_full_demo_mosaic_manual_v4.mp4'
 subprocess.run(['ffmpeg','-y','-hide_banner','-loglevel','warning','-f','concat','-safe','0','-i',str(R/'segments/concat.txt'),'-i',str(R.parent/'assembly/audio/full-mix.wav'),'-i',str(OLD),'-map','0:v:0','-map','1:a:0','-map_metadata','2','-map_chapters','2','-c:v','copy','-af','atrim=end_sample=9892800,apad=whole_len=11937600','-c:a','aac','-b:a','192k','-ar','48000','-ac','2','-movflags','+faststart',str(dest)],check=True)
 manifest={'source_version':str(OLD),'destination':str(dest),'manual_carton_labels':CARTONS,'carton_top_print':TOP_PRINT,'carton_top_return':TOP_RETURN,'whole_quadruped_first_pass':DOG1,'whole_quadruped_second_pass':DOG2,'fire_cabinet':FIRE,'glass_main':GLASS,'glass_POV':POV,'glass_POV_return':POV_RETURN,'glass_POV_backside':POV_BACK,'mute_interval_seconds':[206.1,248.7],'audio_preserved_before_seconds':206.1,'method':'human inspected keyframes; no OCR; manual foreground regions use GrabCut to retain the robot silhouette; BEV overlays protected','preserved_photo_replacement':str(R.parent/'replacement-v3/replacement.json')}
 (R/'output/edit-manifest.json').write_text(json.dumps(manifest,indent=2));print(dest,flush=True)
if __name__=='__main__':
 if sys.argv[1]=='finish':finish()
 else:render(int(sys.argv[1]))
