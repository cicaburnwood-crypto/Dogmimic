"""Retimed Case 2 explanatory zoom, using the replacement source and v5 masks."""
from pathlib import Path
import sys,cv2,numpy as np,subprocess,json,time
from PIL import Image,ImageDraw
R=Path(__file__).resolve().parent
sys.path.insert(0,str(R.parent/'revision-v5'))
import render_v5 as v5
cv2.setNumThreads(2)
PAUSE_START=2370 # source timestamp 9.6 s / global 120.2 s
PAUSE_END=2670
EDIT_START=2328
EDIT_END=2672
def base_frame(f):
 if 2332<=f<PAUSE_START:return f+300
 if PAUSE_START<=f<PAUSE_END:return PAUSE_END
 return f
def source_time(f):
 if f<PAUSE_START:return max(0,(f-2082)/30)
 if f<PAUSE_END:return 9.6
 if f<3136:return 9.6+(f-PAUSE_END)/30
 return v5.source_time(f)
def ease(x):
 x=np.clip(x,0,1);return float(x*x*(3-2*x))
class Composer6(v5.Composer):
 def __init__(self):
  super().__init__()
  cap=cv2.VideoCapture(str(v5.BASE));cap.set(1,2490);ok,im=cap.read();cap.release();assert ok
  original=im[492:780,1000:1876]
  rgb=original[:,:,::-1].astype(np.float32)
  gray=rgb.mean(2);neutral=rgb.max(2)-rgb.min(2)<45
  alpha=np.clip((gray-18)/227,0,1)*neutral
  allowed=np.zeros(alpha.shape,bool);allowed[34:242,24:852]=True;alpha*=allowed
  card=Image.new('RGBA',(876,288),(0,0,0,0));d=ImageDraw.Draw(card)
  d.rounded_rectangle((2,2,873,285),radius=18,fill=(12,16,27,242),outline=(255,52,57,255),width=5)
  letters=Image.new('RGBA',card.size,(245,245,245,0));letters.putalpha(Image.fromarray(np.round(alpha*255).astype(np.uint8)))
  card=Image.alpha_composite(card,letters);self.card=np.asarray(card).copy()
  card.save(R/'inspect/restored-callout.png')
 def edit(self,f,im):
  if not EDIT_START<=f<EDIT_END:return super().edit(f,im)
  out=self.rev.edited(base_frame(f),im)
  camera,bev=self.get_source(source_time(f))
  out[66:426,:636]=camera[:,:636]
  out[426:1066,:636]=cv2.resize(bev,(640,640),interpolation=cv2.INTER_AREA)[:,:636]
  if PAUSE_START<=f<PAUSE_END:
   q=min(ease((f-PAUSE_START)/30),ease((PAUSE_END-f)/30))
   width=round(640+320*q)
   # Zoom into the useful map area, keeping the complete planned path and
   # both endpoints visible. Its aspect ratio matches the animated viewport.
   roi_w=960-60*q;roi_h=roi_w/(width/640);x0=30*q;y0=110*q
   affine=np.array([[width/roi_w,0,-x0*width/roi_w],[0,640/roi_h,-y0*640/roi_h]],np.float32)
   zoom=cv2.warpAffine(bev,affine,(width,640),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
   out[426:1066,:width]=zoom
   overlay=out.copy();cv2.rectangle(overlay,(2,428),(width-3,1063),(57,52,255),4,cv2.LINE_AA)
   out=cv2.addWeighted(overlay,q,out,1-q,0)
   a=min(ease((f-PAUSE_START-8)/22),ease((PAUSE_END-f-5)/25))
   if a>0:
    y,x=534,1000;card=self.card;alpha=(card[:,:,3:4].astype(np.float32)/255)*a
    roi=out[y:y+288,x:x+876];roi[:]=np.round(roi*(1-alpha)+card[:,:,:3][:,:,::-1]*alpha).astype(np.uint8)
    if q>.95:
     overlay=out.copy();cv2.line(overlay,(width,678),(1001,678),(57,52,255),4,cv2.LINE_AA);out=cv2.addWeighted(overlay,a,out,1-a,0)
  patch=out[74:192,14:178];patch[:]=(patch.astype(np.float32)*.48).astype(np.uint8);patch[self.rev.old.speed['2X']]=245
  return out
class BaseReader:
 def __init__(self):self.cap=cv2.VideoCapture(str(v5.BASE));self.index=-1;self.im=None
 def get(self,f):
  assert f>=self.index,(f,self.index)
  while self.index<f:
   ok,self.im=self.cap.read();assert ok,f;self.index+=1
  return self.im
 def close(self):self.cap.release()
def render():
 n=4696;reader=BaseReader();m=Composer6();last=None
 log=(R/'inspect/render.log').open('w');dst=R/'segments/04-clip-3.mp4'
 p=subprocess.Popen(['ffmpeg','-y','-hide_banner','-loglevel','warning','-f','rawvideo','-pixel_format','bgr24','-video_size','1920x1080','-framerate','60','-i','pipe:0','-an','-c:v','libx264','-preset','fast','-crf','17','-profile:v','high','-level:v','4.2','-threads','4','-pix_fmt','yuv420p','-video_track_timescale','60000','-map_metadata','-1','-movflags','+faststart',str(dst)],stdin=subprocess.PIPE,stderr=log)
 st=time.monotonic()
 for f in range(n):
  im=reader.get(base_frame(f))
  if f%2==0 or v5.START<=f<v5.END:last=m.edit(f,im)
  p.stdin.write(last.tobytes())
  if f%600==0:print(f'{f}/{n} frames, {time.monotonic()-st:.1f}s',flush=True)
 p.stdin.close();assert p.wait()==0;log.close();reader.close();m.close();print(dst,flush=True)
def finish():
 old=R.parent/'revision-v5/output/EchoNav_full_demo_mosaic_manual_v5.mp4'
 txt=(R.parent/'revision-v5/segments/concat.txt').read_text().replace(str(R.parent/'revision-v5/segments/04-clip-3.mp4'),str(R/'segments/04-clip-3.mp4'))
 (R/'segments/concat.txt').write_text(txt);dst=R/'output/EchoNav_full_demo_mosaic_manual_v6.mp4'
 subprocess.run(['ffmpeg','-y','-hide_banner','-loglevel','warning','-f','concat','-safe','0','-i',str(R/'segments/concat.txt'),'-i',str(old),'-map','0:v:0','-map','1:a:0','-map_metadata','1','-map_chapters','1','-c','copy','-movflags','+faststart',str(dst)],check=True)
 manifest={'previous_version':str(old),'output':str(dst),'explanatory_pause_global_seconds':[120.2,125.2],'explanatory_pause_source_seconds':9.6,'source':'/media/wolfie/T9/Test Videos/TargetNextWall-BEV-Windowopeded.mp4','zoom_viewport':[0,426,960,1066],'zoom_source_crop':[30,110,930,710],'explanation_card':[1000,534,1876,822],'original_explanation_text':'Predicted free space in unobserved areas enables our model to plan a clear path through the doorway.','all_comparison_panels_and_clock_pause_together':True,'pause_duration_preserved_seconds':5,'audio':'bit-for-bit copy of v5','redaction':'all v5 masks preserved and evaluated against the actual sampled source timestamps','duration_seconds':248.7}
 (R/'output/edit-manifest.json').write_text(json.dumps(manifest,indent=2));print(dst,flush=True)
if __name__=='__main__':
 if sys.argv[1]=='finish':finish()
 else:render()
