from pathlib import Path
import sys,cv2,numpy as np,json,subprocess,time
from manual_v5 import Revision5,R
from ours_masks import edit_camera,TRACKS
cv2.setNumThreads(2)
BASE=R/'segments/new-wheeled-source.mp4'
OUR='/media/wolfie/T9/Test Videos/TargetNextWall-BEV-Windowopeded.mp4'
START,END=2080,3662
def source_time(f):
 # Match the existing elapsed-time clock: 2x, a five-second explanatory
 # pause at 8.3 seconds, then 2x followed by 4x. Hold the replacement's last
 # frame once it finishes; the existing Complete overlay remains in place.
 if f<2332:return max(0,(f-2082)/30)
 if f<2632:return 250/30
 if f<3136:return 250/30+(f-2632)/30
 return 250/30+504/30+(f-3136)/15
class Composer:
 def __init__(self):
  self.rev=Revision5();self.cap=cv2.VideoCapture(OUR)
  self.pts=np.load(R/'inspect/ours-descriptors.npz')['pts'];self.k=-1;self.cached=-1
  self.raw=None;self.camera=None;self.bev=None
 def get_source(self,t):
  k=int(np.clip(np.searchsorted(self.pts,t,side='right')-1,0,len(self.pts)-1))
  if k<self.k:self.cap.release();self.cap=cv2.VideoCapture(OUR);self.k=-1;self.cached=-1
  while self.k<k:
   ok,self.raw=self.cap.read();assert ok,(self.k,k);self.k+=1
  if self.cached!=k:
   self.camera=edit_camera(float(self.pts[k]),cv2.resize(self.raw[270:810,:960],(640,360),interpolation=cv2.INTER_AREA))
   self.bev=self.raw[60:1020,960:1920].copy();self.cached=k
  return self.camera,self.bev
 def edit(self,f,im):
  out=self.rev.edited(f,im)
  if not START<=f<END:return out
  camera,bev=self.get_source(source_time(f))
  out[66:426,:636]=camera[:,:636]
  out[426:1066,:636]=cv2.resize(bev,(640,640),interpolation=cv2.INTER_AREA)[:,:636]
  # Preserve the prepared enlargement trajectory, located by its known red
  # border; this detects a layout element, not text or an object to redact.
  if 38.8<=f/60<=43.85:
   q=im[400:1060,644:980].astype(np.int16)
   red=(q[:,:,2]>110)&(q[:,:,1]<145)&(q[:,:,0]<145)&(q[:,:,2]-q[:,:,1]>30)
   xs=np.where(red.sum(0)>250)[0]
   if len(xs):
    right=int(xs.max()+645);top=1066-right
    out[top:1066,:right]=cv2.resize(bev,(right,right),interpolation=cv2.INTER_LINEAR)
    for sl in [(slice(top,top+6),slice(0,right)),(slice(1060,1066),slice(0,right)),(slice(top,1066),slice(0,6)),(slice(top,1066),slice(right-6,right))]:out[sl]=im[sl]
  name='2X' if f<3136 else '4X';patch=out[74:192,14:178]
  patch[:]=(patch.astype(np.float32)*.48).astype(np.uint8)
  glyph=self.rev.old.speed[name];patch[glyph]=245
  # The prepared green completion card has an opaque black background.
  q=im[260:426,:636].astype(np.int16);g=(q[:,:,1]>130)&(q[:,:,1]-q[:,:,0]>40)&(q[:,:,1]-q[:,:,2]>40)
  if f>=3595 and np.count_nonzero(g)>400:
   yy,xx=np.where(g);x0,x1=max(0,xx.min()-1),min(636,xx.max()+2);y0,y1=yy.min()+259,min(426,yy.max()+262)
   out[y0:y1,x0:x1]=im[y0:y1,x0:x1]
  return out
 def close(self):self.cap.release()
def render():
 c=cv2.VideoCapture(str(BASE));n=int(c.get(cv2.CAP_PROP_FRAME_COUNT));m=Composer();last=None
 log=(R/'inspect/render.log').open('w');dst=R/'segments/04-clip-3.mp4'
 p=subprocess.Popen(['ffmpeg','-y','-hide_banner','-loglevel','warning','-f','rawvideo','-pixel_format','bgr24','-video_size','1920x1080','-framerate','60','-i','pipe:0','-an','-c:v','libx264','-preset','fast','-crf','17','-profile:v','high','-level:v','4.2','-threads','4','-pix_fmt','yuv420p','-video_track_timescale','60000','-map_metadata','-1','-movflags','+faststart',str(dst)],stdin=subprocess.PIPE,stderr=log)
 st=time.monotonic()
 for f in range(n):
  ok,im=c.read();assert ok,f
  if f%2==0 or START<=f<END:last=m.edit(f,im)
  p.stdin.write(last.tobytes())
  if f%600==0:print(f'{f}/{n} frames, {time.monotonic()-st:.1f}s',flush=True)
 p.stdin.close();assert p.wait()==0;log.close();c.release();m.close();print(dst,flush=True)
def finish():
 root=R.parent;v2=root/'redaction/manual-v2';old=root/'revision-v4/output/EchoNav_full_demo_mosaic_manual_v4.mp4'
 timeline=json.loads((root/'assembly/output/timeline.json').read_text());entries=[]
 for j in timeline:
  name=Path(j['output']).name
  if name=='04-clip-3.mp4':p=R/'segments'/name
  elif name=='06-clip-4.mp4':p=root/'revision-v4/segments'/name
  elif name=='02-clip-2.mp4':p=root/'replacement-v3/segments'/name
  else:p=v2/'segments'/name
  probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v','-show_streams','-of','json',str(p)]))['streams'][0]
  assert int(probe['nb_frames'])==j['frames'],(name,probe['nb_frames'])
  entries.append(f"file '{p}'\nduration {j['duration']:.9f}\n")
 (R/'segments/concat.txt').write_text(''.join(entries));dst=R/'output/EchoNav_full_demo_mosaic_manual_v5.mp4'
 subprocess.run(['ffmpeg','-y','-hide_banner','-loglevel','warning','-f','concat','-safe','0','-i',str(R/'segments/concat.txt'),'-i',str(old),'-map','0:v:0','-map','1:a:0','-map_metadata','1','-map_chapters','1','-c','copy','-movflags','+faststart',str(dst)],check=True)
 manifest=dict(previous=str(old),whole_wheeled_source='/media/wolfie/T9/DemoVideosTemp/Wheeled123.mp4',nested_source=OUR,nested_replacement='Case 2, Our model camera and synchronized BEV',nested_interval_local=[START/60,END/60],nested_interval_global=[80.7+START/60,80.7+END/60],source_clock={'start_frame':2082,'pause_frames':[2332,2632],'four_times_start_frame':3136},duration_seconds=248.7,audio='copied bit-for-bit from v4; last source clip remains silent',redactions='Manually inspected image coordinates. No OCR. Prior v4 masks retained outside replacement; case 1 errors corrected and title cards excluded.',new_source_masks=TRACKS)
 (R/'output/edit-manifest.json').write_text(json.dumps(manifest,indent=2));print(dst,flush=True)
if __name__=='__main__':
 if sys.argv[1]=='finish':finish()
 else:render()
