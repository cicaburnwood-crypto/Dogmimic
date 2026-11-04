from pathlib import Path
import sys,json,subprocess,copy
import cv2,numpy as np
from PIL import Image
R=Path(__file__).resolve().parent;B=R.parent
sys.path.insert(0,str(B/'redaction/manual-v2'))
from engine import Manual,pixels
sys.path.insert(0,str(B/'replacement-v3'))
from replace_view import labels
M=Manual(2);cv2.setNumThreads(2)
V=json.loads((B/'render/visuals.json').read_text())
PATCH=cv2.imread(str(B/'revision-v11/inspect/face-mosaic-patch.png'))
# Fill the original whole wheeled-robot image slot, without BEV or letterboxing.
PHOTO=cv2.imread('/home/wolfie/Downloads/6929.jpg')
CROP=(620,640,1688,1240);SLOT=(80,650,632,960)
x0,y0,x1,y1=CROP;crop=PHOTO[y0:y1,x0:x1]
photo=cv2.resize(crop,(552,310),interpolation=cv2.INTER_LANCZOS4)
pm=np.zeros((310,552),np.uint8)
for name,poly in labels:
 p=(np.array(poly,float)-[x0,y0])*[552/(x1-x0),310/(y1-y0)]
 cv2.fillConvexPoly(pm,np.round(p).astype(np.int32),255)
pm=cv2.dilate(pm,np.ones((5,5),np.uint8))
coarse=cv2.resize(cv2.resize(photo,(46,26),interpolation=cv2.INTER_AREA),(552,310),interpolation=cv2.INTER_NEAREST)
photo=np.where(pm[:,:,None]>0,coarse,photo)
cv2.imwrite(str(R/'inspect/robot-photo.png'),photo)

def gif_frames(path):
 im=Image.open(path);frames=[];end=[];t=0
 for k in range(im.n_frames):
  im.seek(k);frames.append(im.convert('RGBA').copy());t+=im.info.get('duration',100)/1000;end.append(t)
 return frames,np.array(end)

def raw_frame(i,f,cache):
 v=V[i]
 if 'gif' in v:
  under,over,frames,ends=cache
  local=f-(0 if i==0 else 2264)
  k=min(len(frames)-1,int(np.searchsorted(ends,local/60+1e-8,side='right')))
  x,y,w,h=v['bounds'];out=under.copy();out.alpha_composite(frames[k].resize((w,h),Image.Resampling.LANCZOS),(x,y));out.alpha_composite(over)
  out=cv2.cvtColor(np.asarray(out.convert('RGB')),cv2.COLOR_RGB2BGR)
 else:out=cache.copy()
 mask=M.mask(f,out)
 if mask.any():out=pixels(out,mask)
 if i==3:
  out[650:960,80:632]=photo
  out[670:730,1616:1668]=PATCH
 return out

def slides():
 ranges=[(0,1059,900),(1059,2264,900),(2264,3194,600),(3194,4207,720)]
 cmd=['ffmpeg','-y','-v','warning','-f','rawvideo','-pixel_format','bgr24','-video_size','1920x1080','-framerate','30','-i','pipe:0','-an','-c:v','libx264','-preset','slow','-crf','16.5','-threads','6','-pix_fmt','yuv420p','-video_track_timescale','60000',str(R/'segments/slides.mp4')]
 with (R/'inspect/slides-encode.log').open('w') as log:
  p=subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=log)
  mapping=[]
  for i,(a,b,m) in enumerate(ranges):
   v=V[i]
   cache=(Image.open(v['under']).convert('RGBA'),Image.open(v['over']).convert('RGBA'),*gif_frames(v['gif'])) if 'gif' in v else cv2.imread(v['static'])
   ns=np.arange(b-a);selected=(ns[(ns*m)%(b-a)<m]+a)[::2];assert len(selected)==m//2
   for j,f in enumerate(selected):
    out=raw_frame(i,int(f),cache);p.stdin.write(out.tobytes());mapping.append([i+1,int(f)])
    if j==min(150,len(selected)-1):cv2.imwrite(str(R/'inspect'/f'page-{i+1}.png'),out)
   print('Rendered page',i+1,flush=True)
  p.stdin.close();assert p.wait()==0
 (R/'inspect/frame-map.json').write_text(json.dumps(mapping))

if __name__=='__main__':slides()
