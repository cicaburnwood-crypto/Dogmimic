from pathlib import Path
import cv2,numpy as np,sys
from PIL import Image,ImageDraw
R=Path(__file__).resolve().parent
SOURCE='/media/wolfie/T9/Test Videos/TargetNextWall-BEV-Windowopeded.mp4'
def sheet(ts,name,grid=False,edit=None):
 c=cv2.VideoCapture(SOURCE);out=Image.new('RGB',(1280,390*((len(ts)+1)//2)),'#222');d=ImageDraw.Draw(out)
 for i,t in enumerate(ts):
  c.set(cv2.CAP_PROP_POS_MSEC,t*1000);ok,im=c.read();assert ok
  im=cv2.resize(im[270:810,:960],(640,360),interpolation=cv2.INTER_AREA)
  if edit:im=edit(c.get(cv2.CAP_PROP_POS_MSEC)/1000,im)
  if grid:
   for x in range(0,640,50):cv2.line(im,(x,0),(x,360),(45,140,45),1);cv2.putText(im,str(x),(x+2,14),0,.38,(0,255,0),1)
   for y in range(50,360,50):cv2.line(im,(0,y),(640,y),(45,140,45),1);cv2.putText(im,str(y),(2,y-2),0,.38,(0,255,0),1)
  out.paste(Image.fromarray(cv2.cvtColor(im,cv2.COLOR_BGR2RGB)),((i%2)*640,(i//2)*390+30));d.text(((i%2)*640+4,(i//2)*390+3),f'Our model source {t:.3f}s',fill='white',font_size=20)
 c.release();out.save(R/'inspect'/f'{name}.jpg',quality=97)
if __name__=='__main__':sheet([float(t) for t in sys.argv[1].split(',')],sys.argv[2],grid='grid' in sys.argv)
