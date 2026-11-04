from pathlib import Path
import cv2,numpy as np,sys
from PIL import Image,ImageDraw
R=Path(__file__).resolve().parent
SRC=R.parent/'assembly/segments'
NAMES={3:'04-clip-3.mp4',4:'06-clip-4.mp4'}
def get(c,t):
 cap=cv2.VideoCapture(str(SRC/NAMES[c]));cap.set(cv2.CAP_PROP_POS_FRAMES,round(t*60));ok,im=cap.read();cap.release();assert ok
 return im

def sheet(c,ts,name,crop=(0,64,640,426),grid=True):
 ims=[]
 for t in ts:
  im=get(c,t);x0,y0,x1,y1=crop;im=im[y0:y1,x0:x1].copy()
  if grid:
   for x in range((x0//50+1)*50,x1,50):
    cv2.line(im,(x-x0,0),(x-x0,im.shape[0]),(60,150,60),1);cv2.putText(im,str(x),(x-x0+2,15),0,.37,(0,255,0),1)
   for y in range((y0//50+1)*50,y1,50):
    cv2.line(im,(0,y-y0),(im.shape[1],y-y0),(60,150,60),1);cv2.putText(im,str(y),(2,y-y0-2),0,.37,(0,255,0),1)
  if im.shape[1]>640:im=cv2.resize(im,(640,round(im.shape[0]*640/im.shape[1])))
  ims.append(Image.fromarray(cv2.cvtColor(im,cv2.COLOR_BGR2RGB)))
 w,h=ims[0].size;out=Image.new('RGB',(2*w,((len(ts)+1)//2)*(h+28)),'#222');d=ImageDraw.Draw(out)
 for i,(t,im) in enumerate(zip(ts,ims)):
  x=i%2*w;y=i//2*(h+28);out.paste(im,(x,y+28));d.text((x+5,y+2),f'clip{c} local {t:.3f}s   global {t+({3:80.7,4:161.966667}[c]):.3f}s',fill='white',font_size=18)
 out.save(R/'inspect'/f'{name}.jpg',quality=97)
if __name__=='__main__':
 sheet(int(sys.argv[1]),[float(x) for x in sys.argv[2].split(',')],sys.argv[3],tuple(map(int,sys.argv[4].split(','))) if len(sys.argv)>4 else (0,64,640,426))
