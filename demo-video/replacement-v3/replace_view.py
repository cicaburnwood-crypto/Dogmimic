from pathlib import Path
import sys,json,subprocess,time,hashlib
import cv2,numpy as np
BASE=Path(__file__).resolve().parent
V2=BASE.parent/'redaction/manual-v2'
sys.path.insert(0,str(V2))
from engine import Manual,pixels
SOURCE=BASE.parent/'assembly/segments/02-clip-2.mp4'
PHOTO=Path('/home/wolfie/Downloads/6929.jpg')
OLD=V2/'output/EchoNav_full_demo_mosaic_manual_v2.mp4'
cv2.setNumThreads(2)
# Keep the BEV panel and replace only the left camera slot, including its letterbox.
SLOT=(80,650,355,959);FIRST=3194;FRAMES=4207
photo=cv2.imread(str(PHOTO));assert photo.shape[:2]==(1280,1707)
W=275;H=round(photo.shape[0]*W/photo.shape[1]);X=80;Y=650+(309-H)//2
insert=cv2.resize(photo,(W,H),interpolation=cv2.INTER_LANCZOS4)
# Manually inspected printed labels and trademarks; no OCR or object recognition.
labels=[
 ('parcel labels',[(792,514),(957,511),(952,579),(791,584)]),
 ('carton wordmark',[(821,597),(909,596),(909,617),(819,617)]),
 ('Intel RealSense trademark',[(1054,679),(1126,680),(1126,700),(1053,699)]),
 ('device top printed label',[(1145,835),(1278,830),(1294,858),(1159,864)]),
 ('device side brand emblem',[(1145,851),(1158,849),(1166,863),(1157,896),(1145,901),(1138,884)]),
 ('platform brand emblem',[(1015,940),(1044,943),(1043,964),(1013,963)]),
 ('fire extinguisher printed label',[(529,188),(540,187),(541,211),(529,211)]),
 ('reflected extinguisher label',[(139,148),(149,148),(149,169),(139,170)])
]
mask=np.zeros((1080,1920),np.uint8)
for name,poly in labels:
 pts=np.array(poly,float)*[W/1707,H/1280]+[X,Y]
 cv2.fillConvexPoly(mask,np.round(pts).astype(np.int32),255)
mask=cv2.dilate(mask,np.ones((3,3),np.uint8));mask[:Y]=0;mask[Y+H:]=0;mask[:,:X]=0;mask[:,X+W:]=0
m=Manual(2)
def edited(f,im):
 old=m.mask(f,im);out=pixels(im,old) if old.any() else im.copy()
 if f>=FIRST:
  x0,y0,x1,y1=SLOT;out[y0:y1,x0:x1]=0
  out[Y:Y+H,X:X+W]=insert
  out=pixels(out,mask)
 return out

def preview():
 cap=cv2.VideoCapture(str(SOURCE));cap.set(cv2.CAP_PROP_POS_FRAMES,3685);ok,im=cap.read();cap.release();assert ok
 out=edited(3685,im);cv2.imwrite(str(BASE/'inspect/replacement-preview.png'),out)
 # Preserve a review image of the manually chosen locations on the unmodified source photo.
 q=photo.copy()
 for name,poly in labels:cv2.polylines(q,[np.array(poly,np.int32)],True,(0,160,255),2)
 cv2.imwrite(str(BASE/'inspect/manual-label-locations.png'),q)
 meta={'source_photo':str(PHOTO),'source_photo_sha256':hashlib.sha256(PHOTO.read_bytes()).hexdigest(),'video_interval_seconds':[3649/60,4662/60],'source_clip_interval_frames':[FIRST,FRAMES],'camera_slot':SLOT,'photo_rect':[X,Y,W,H],'preserved_BEV_rect':[356,666,632,942],'manual_targets':[z[0] for z in labels],'method':'manual image-coordinate selection applied during video compositing; no OCR'}
 (BASE/'replacement.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta,indent=2))
def render():
 cap=cv2.VideoCapture(str(SOURCE));assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT))==FRAMES
 log=(BASE/'inspect/render.log').open('w');p=subprocess.Popen(['ffmpeg','-y','-hide_banner','-loglevel','warning','-f','rawvideo','-pixel_format','bgr24','-video_size','1920x1080','-framerate','60','-i','pipe:0','-an','-c:v','libx264','-preset','fast','-crf','17','-profile:v','high','-level:v','4.2','-threads','3','-pix_fmt','yuv420p','-video_track_timescale','60000','-map_metadata','-1','-movflags','+faststart',str(BASE/'segments/02-clip-2.mp4')],stdin=subprocess.PIPE,stderr=log)
 st=time.monotonic()
 for f in range(FRAMES):
  ok,im=cap.read();assert ok;p.stdin.write(edited(f,im).tobytes())
  if f%600==0:print(f'{f}/{FRAMES} frames | {time.monotonic()-st:.1f}s',flush=True)
 p.stdin.close();rc=p.wait();cap.release();log.close();assert rc==0;print('Replacement segment rendered.',flush=True)
def finish():
 timeline=json.loads((BASE.parent/'assembly/output/timeline.json').read_text());entries=[]
 for j in timeline:
  name=Path(j['output']).name;p=(BASE/'segments'/name) if name=='02-clip-2.mp4' else (V2/'segments'/name)
  streams=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-of','json',str(p)]))['streams'];v=next(x for x in streams if x['codec_type']=='video');assert int(v['nb_frames'])==j['frames']
  entries.append(f"file '{p}'\nduration {j['duration']:.9f}\n")
 (BASE/'segments/concat.txt').write_text(''.join(entries));dest=BASE/'output/EchoNav_full_demo_mosaic_manual_v3.mp4'
 subprocess.run(['ffmpeg','-y','-hide_banner','-loglevel','warning','-f','concat','-safe','0','-i',str(BASE/'segments/concat.txt'),'-i',str(OLD),'-map','0:v:0','-map','1:a:0','-map_metadata','1','-map_chapters','1','-c','copy','-movflags','+faststart',str(dest)],check=True);print(dest)
if __name__=='__main__':
 {'preview':preview,'render':render,'finish':finish}[sys.argv[1]]()
