from pathlib import Path
import cv2,numpy as np,json,subprocess
from PIL import Image,ImageDraw
R=Path(__file__).resolve().parent;B=R.parent
old=cv2.VideoCapture(str(B/'compress-v14/output/refined-h264.mp4'));new=cv2.VideoCapture(str(R/'output/EchoNav_full_demo_v15_50MB.mp4'))
def get(c,n):
 if c is new:
  data=subprocess.check_output(['ffmpeg','-v','error','-ss',str(n/30),'-i',str(R/'output/EchoNav_full_demo_v15_50MB.mp4'),'-frames:v','1','-f','image2pipe','-c:v','png','-'])
  return cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_COLOR)
 c.set(1,n);ok,im=c.read();assert ok,n;return im
report=[];sheet=Image.new('RGB',(1800,920),(30,30,30));draw=ImageDraw.Draw(sheet)
for i,(p,t) in enumerate([(2,26),(4,51)]):
 raw=cv2.imread(str(R/'inspect'/f'page-{p}.png'));a=get(old,round(t*60));b=get(new,round(t*30))
 # Compare unchanged typography against original lossless source, excluding the edited photo.
 roi=(slice(20,580),slice(50,1850))
 before=float(np.abs(a[roi].astype(float)-raw[roi].astype(float)).mean());after=float(np.abs(b[roi].astype(float)-raw[roi].astype(float)).mean())
 assert after<before,(p,before,after)
 report.append(dict(page=p,source_reference='lossless PNG',old_mae=before,new_mae=after,error_reduction=1-after/before))
 for j,im in enumerate([raw,a,b]):
  # Native-pixel text crop (no scaling) shows edge retention.
  crop=im[240:660,60:660] if p==2 else im[180:600,75:675]
  sheet.paste(Image.fromarray(cv2.cvtColor(crop,cv2.COLOR_BGR2RGB)),(j*600,i*460+30))
  draw.text((j*600+8,i*460+8),['Original PNG','Previous compressed video','Restored v15'][j],fill='white')
# Verify final image slot against its prepared manually-redacted image.
f=get(new,51*30);ref=cv2.imread(str(R/'inspect/robot-photo.png'))
mae=float(np.abs(f[650:960,80:632].astype(float)-ref.astype(float)).mean());print("photo error",mae);assert mae<6
cv2.imwrite(str(R/'inspect/final-page-4.png'),f)
sheet.save(R/'inspect/text-quality-comparison.png')
(R/'inspect/quality-comparison.json').write_text(json.dumps(dict(text_checks=report,photo_mae=mae),indent=2))
print(json.dumps(report,indent=2))
