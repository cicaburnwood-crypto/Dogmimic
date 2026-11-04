from concurrent.futures import ThreadPoolExecutor
import hashlib,json,subprocess
import cv2,numpy as np
from PIL import Image,ImageDraw
from edit_v10 import ROOT,OLD,NEW,SOURCE,SEGMENT,BORDERS,edit

def evidence(p):
    audio=subprocess.check_output(['ffmpeg','-v','error','-i',str(p),'-map','0:a:0','-c','copy','-f','hash','-hash','sha256','-']).decode().strip()
    packets=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_packets','-show_entries','packet=data_hash','-show_data_hash','sha256','-of','json',str(p)]))
    return audio,[q['data_hash'] for q in packets['packets']]

with ThreadPoolExecutor(max_workers=2) as pool:
    old,new=list(pool.map(evidence,[OLD,NEW]))
assert old[0]==new[0]
assert len(old[1])==len(new[1])==14402
assert old[1][:11846]==new[1][:11846]
probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-show_format','-show_chapters','-of','json',str(NEW)]))
v=next(s for s in probe['streams'] if s['codec_type']=='video')
assert (v['width'],v['height'],v['avg_frame_rate'],v['nb_frames'])==(1920,1080,'60/1','14402')
assert abs(float(v['duration'])-240.033333333)<1e-5 and len(probe['chapters'])==8

source=cv2.VideoCapture(str(SOURCE));dest=cv2.VideoCapture(str(SEGMENT))
times=[0,8,16,24,27.2,27.3,27.5,28.5,30.2,30.4,36,42.5]
sheet=Image.new('RGB',(1920,1520),(25,25,25));draw=ImageDraw.Draw(sheet);checks=[]
for i,t in enumerate(times):
    f=round(t*60);source.set(1,f);dest.set(1,f)
    ok,im=source.read();assert ok
    ok,out=dest.read();assert ok
    expected,rect=edit(im,f)
    # Allow the small color-rounding differences from H.264 / YUV 4:2:0 encoding.
    delta=out.astype(float)-expected.astype(float)
    error=float(np.abs(delta).mean())
    psnr=float(10*np.log10(255**2/np.mean(delta**2)))
    assert error<2.0 and psnr>39,(t,error,psnr)
    mask=np.zeros((1080,1920),np.uint8)
    for x0,y0,x1,y1 in BORDERS:mask[y0:y1,x0:x1]=255
    if rect:
        x0,y0,x1,y1=rect;mask[y0:y1,x0:x1]=0
    core=cv2.erode(mask,np.ones((5,5),np.uint8))>0
    black_mean=float(out[core].mean());assert black_mean<1,(t,black_mean)
    # Protected display regions retain their content rather than being recolored.
    assert np.array_equal(expected[mask==0],im[mask==0])
    x=i%3*640;y=i//3*380
    sheet.paste(Image.fromarray(cv2.cvtColor(out,cv2.COLOR_BGR2RGB)).resize((640,360)),(x,y))
    draw.text((x+5,y+363),f'{t:.2f}s',fill='white')
    checks.append(dict(local_seconds=t,black_border_mean=black_mean,pixel_error=error,psnr_db=psnr,protected_BEV_rectangle=rect))
source.release();dest.release()
sheet.save(ROOT/'inspect/final-storyboard.jpg',quality=95)
subprocess.run(['ffmpeg','-v','error','-threads','4','-i',str(NEW),'-map','0:v:0','-map','0:a:0','-f','null','-'],check=True)
report=dict(duration_seconds=240.03333333333333,frames=14402,resolution=[1920,1080],fps=60,chapters=8,
            audio_payload_sha256=new[0],audio_identical_to_v9=True,other_video_segments_identical_to_v9=True,
            border_checks=checks,full_decode='passed',bytes=NEW.stat().st_size,sha256=hashlib.sha256(NEW.read_bytes()).hexdigest())
(ROOT/'output/verification.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2),flush=True)
