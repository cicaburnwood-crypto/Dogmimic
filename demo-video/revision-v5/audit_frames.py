from pathlib import Path
import cv2,numpy as np,sys
from PIL import Image,ImageDraw
R=Path(__file__).resolve().parent;BASE=R.parent
RAW=BASE/'assembly/segments/04-clip-3.mp4';OLD=BASE/'revision-v4/segments/04-clip-3.mp4'
def get(t,old=False):
 c=cv2.VideoCapture(str(OLD if old else RAW));c.set(cv2.CAP_PROP_POS_FRAMES,round(t*60));ok,im=c.read();c.release();assert ok;return im

def sheet(ts,name,crop=(0,64,640,426),old=False,grid=False,width=640,cols=2):
 ims=[]
 for t in ts:
  im=get(t,old)
  if grid:
   for x in range(0,1920,50):cv2.line(im,(x,0),(x,1080),(45,150,45),1);cv2.putText(im,str(x),(x+2,crop[1]+16),0,.38,(0,255,0),1)
   for y in range(0,1080,50):cv2.line(im,(0,y),(1920,y),(45,150,45),1);cv2.putText(im,str(y),(crop[0]+2,y-2),0,.38,(0,255,0),1)
  x0,y0,x1,y1=crop;im=im[y0:y1,x0:x1];im=cv2.resize(im,(width,round(im.shape[0]*width/im.shape[1])))
  ims.append(Image.fromarray(cv2.cvtColor(im,cv2.COLOR_BGR2RGB)))
 w,h=ims[0].size;out=Image.new('RGB',(w*cols,(h+28)*int(np.ceil(len(ts)/cols))),'#222');d=ImageDraw.Draw(out)
 for i,(t,im) in enumerate(zip(ts,ims)):
  x=i%cols*w;y=i//cols*(h+28);out.paste(im,(x,y+28));d.text((x+4,y+2),f'{"v4" if old else "raw"} local {t:.3f}s / global {t+80.7:.3f}s',fill='white',font_size=18)
 out.save(R/'inspect'/f'{name}.jpg',quality=97)
if __name__=='__main__':
 sheet([float(z) for z in sys.argv[1].split(',')],sys.argv[2],tuple(map(int,sys.argv[3].split(','))) if len(sys.argv)>3 else (0,64,640,426),old='old' in sys.argv,grid='grid' in sys.argv)
