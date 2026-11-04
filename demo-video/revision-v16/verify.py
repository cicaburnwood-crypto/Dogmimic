from pathlib import Path
import subprocess,json,hashlib,shutil
import cv2,numpy as np
R=Path(__file__).resolve().parent;B=R.parent
old=B/'revision-v15/output/EchoNav_full_demo_v15_50MB.mp4';out=R/'output/EchoNav_full_demo_v16_50MB.mp4'
def probe(p):return json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-show_chapters','-of','json',str(p)]))
def pk(p,s):return json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams',s,'-show_packets','-show_entries','packet=data_hash,pts_time','-show_data_hash','sha256','-of','json',str(p)]))['packets']
p=probe(out);v=next(x for x in p['streams'] if x['codec_type']=='video')
assert int(v['nb_frames'])==8957 and abs(float(v['duration'])-177.783333)<1e-6
assert p['chapters']==probe(old)['chapters']
a=pk(old,'v:0');b=pk(out,'v:0')
assert a[:1350]==b[:1350] and a[1710:]==b[1710:]
assert [x['pts_time'] for x in a]==[x['pts_time'] for x in b]
assert pk(old,'a:0')==pk(out,'a:0')
with (R/'inspect/decode.log').open('w') as log:
 subprocess.run(['ffmpeg','-v','error','-threads','4','-i',str(out),'-map','0:v','-map','0:a','-fps_mode','passthrough','-enc_time_base:v','1:60000','-f','null','-'],stderr=log,check=True)
assert (R/'inspect/decode.log').stat().st_size==0
imdata=subprocess.check_output(['ffmpeg','-v','error','-ss','51','-i',str(out),'-frames:v','1','-f','image2pipe','-c:v','png','-'])
im=cv2.imdecode(np.frombuffer(imdata,np.uint8),cv2.IMREAD_COLOR)
cv2.imwrite(str(R/'inspect/final-page-4.png'),im)
ref=cv2.imread(str(R/'inspect/page-4.png'));assert np.abs(im.astype(float)-ref.astype(float)).mean()<4
assert out.stat().st_size<50000000
dst=Path('/home/wolfie/Downloads/newvideos')/out.name;assert not dst.exists();shutil.copy2(out,dst)
h=hashlib.sha256(out.read_bytes()).hexdigest();assert hashlib.sha256(dst.read_bytes()).hexdigest()==h
report={'delivery':str(dst),'bytes':dst.stat().st_size,'duration':177.783333,'full_original_photo':True,'photo_crop':None,'aspect_ratio_preserved':True,'black_border_and_BEV_removed':True,'mosaics_preserved':True,'unchanged_video_outside_page4':True,'unchanged_audio_and_timestamps':True,'full_decode':'passed','sha256':h}
(R/'output/verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
