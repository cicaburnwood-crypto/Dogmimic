from pathlib import Path
import sys,subprocess,json,hashlib
import cv2,numpy as np
R=Path(__file__).resolve().parent;B=R.parent
sys.path.insert(0,str(B/'replacement-v3'))
from replace_view import labels
photo=cv2.imread('/home/wolfie/Downloads/6929.jpg');h,w=photo.shape[:2]
H=310;W=round(w*H/h);X=80+(552-W)//2;Y=650
scaled=cv2.resize(photo,(W,H),interpolation=cv2.INTER_LANCZOS4)
mask=np.zeros((H,W),np.uint8)
for name,poly in labels:
 pts=np.array(poly,float)*[W/w,H/h]
 cv2.fillConvexPoly(mask,np.round(pts).astype(np.int32),255)
mask=cv2.dilate(mask,np.ones((3,3),np.uint8))
pix=cv2.resize(cv2.resize(scaled,(round(W/12),round(H/12)),interpolation=cv2.INTER_AREA),(W,H),interpolation=cv2.INTER_NEAREST)
scaled=np.where(mask[:,:,None]>0,pix,scaled)
page=cv2.imread(str(B/'revision-v15/inspect/page-4.png'))
page[648:962,78:634]=page[640,80]
page[Y:Y+H,X:X+W]=scaled
cv2.imwrite(str(R/'inspect/page-4.png'),page)
cv2.imwrite(str(R/'inspect/full-photo-mosaic.png'),scaled)
meta={'source_photo':'/home/wolfie/Downloads/6929.jpg','source_dimensions':[w,h],'source_crop':None,'image_rect':[X,Y,W,H],'previous_slot':[80,650,632,960],'fit':'Full original photo, proportional resize; unused layout area matches slide background, no letterbox','mosaic':'Same manually located Chinese text and brand labels, transformed to full-photo layout'}
(R/'inspect/edit.json').write_text(json.dumps(meta,indent=2))
subprocess.run(['ffmpeg','-y','-v','warning','-loop','1','-framerate','30','-i',str(R/'inspect/page-4.png'),'-frames:v','360','-an','-c:v','libx264','-preset','slow','-crf','16.5','-threads','5','-pix_fmt','yuv420p','-video_track_timescale','60000',str(R/'segments/page-4.mp4')],check=True)
subprocess.run(['ffmpeg','-y','-v','warning','-i',str(B/'revision-v15/segments/slides.mp4'),'-map','0:v','-an','-frames:v','1200','-c','copy','-map_metadata','-1','-map_chapters','-1','-video_track_timescale','60000',str(R/'segments/pages-1-3.mp4')],check=True)
parts=[(B/'revision-v15/segments/title-final.mp4',5),(R/'segments/pages-1-3.mp4',40),(R/'segments/page-4.mp4',12),(B/'revision-v15/segments/experiments.mp4',7247/60)]
concat=R/'segments/concat.txt';concat.write_text(''.join(f"file '{p}'\nduration {d:.12f}\n" for p,d in parts))
old=B/'revision-v15/output/EchoNav_full_demo_v15_50MB.mp4';out=R/'output/EchoNav_full_demo_v16_50MB.mp4'
subprocess.run(['ffmpeg','-y','-v','warning','-f','concat','-safe','0','-i',str(concat),'-i',str(old),'-map','0:v','-map','1:a','-map_metadata','1','-map_chapters','1','-c','copy','-video_track_timescale','60000','-movflags','+faststart',str(out)],check=True)
print('Output',out.stat().st_size,flush=True)
