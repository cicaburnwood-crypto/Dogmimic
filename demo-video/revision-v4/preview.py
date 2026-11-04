from pathlib import Path
import cv2,sys,numpy as np
from PIL import Image,ImageDraw
from manual_v4 import Revision
from inspect_frames import get,R
c=int(sys.argv[1]);ts=[float(z) for z in sys.argv[2].split(',')];name=sys.argv[3]
crop=tuple(map(int,sys.argv[4].split(','))) if len(sys.argv)>4 else (0,64,640,426)
r=Revision(c);ims=[]
for t in ts:
 im=get(c,t);out=r.edited(round(t*60),im)
 x0,y0,x1,y1=crop;out=out[y0:y1,x0:x1]
 if out.shape[1]>640:out=cv2.resize(out,(640,round(out.shape[0]*640/out.shape[1])))
 ims.append(Image.fromarray(cv2.cvtColor(out,cv2.COLOR_BGR2RGB)))
w,h=ims[0].size;out=Image.new('RGB',(2*w,((len(ts)+1)//2)*(h+28)),'#222');d=ImageDraw.Draw(out)
for i,(t,im) in enumerate(zip(ts,ims)):
 x=i%2*w;y=i//2*(h+28);out.paste(im,(x,y+28));d.text((x+5,y+2),f'clip{c} local {t:.3f}s',fill='white',font_size=18)
out.save(R/'inspect'/f'{name}.jpg',quality=97)
