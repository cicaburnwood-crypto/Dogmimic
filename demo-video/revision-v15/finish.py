from pathlib import Path
import subprocess,json,hashlib,shutil,sys
import cv2,numpy as np
R=Path(__file__).resolve().parent;B=R.parent
OLD=B/'compress-v14/output/refined-h264.mp4'
OUT=R/'output/EchoNav_full_demo_v15_50MB.mp4'
parts=[('title-final.mp4',5,150),('slides.mp4',52,1560),('experiments.mp4',7247/60,7247)]
def probe(p):return json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-show_format','-show_chapters','-of','json',str(p)]))
def pk(p,stream):
 return json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams',stream,'-show_packets','-show_entries','packet=data_hash,pts_time','-show_data_hash','sha256','-of','json',str(p)]))['packets']
for name,duration,n in parts:
 v=next(s for s in probe(R/'segments'/name)['streams'] if s['codec_type']=='video')
 assert int(v['nb_frames'])==n
 assert abs(float(v['duration'])-duration)<1e-5
c=R/'segments/concat.txt';c.write_text(''.join(f"file '{R/'segments'/name}'\nduration {duration:.12f}\n" for name,duration,n in parts))
subprocess.run(['ffmpeg','-y','-v','warning','-f','concat','-safe','0','-i',str(c),'-i',str(OLD),'-map','0:v:0','-map','1:a:0','-map_metadata','1','-map_chapters','1','-c','copy','-video_track_timescale','60000','-movflags','+faststart',str(OUT)],check=True)
p=probe(OUT);v=next(s for s in p['streams'] if s['codec_type']=='video')
assert (v['width'],v['height'],int(v['nb_frames']))==(1920,1080,8957),v
assert abs(float(v['duration'])-177.783333)<1e-5
assert p['chapters']==probe(OLD)['chapters']
assert pk(OLD,'a:0')==pk(OUT,'a:0')
a=pk(OLD,'v:0');b=pk(OUT,'v:0')
assert a[3420:]==b[1710:], 'Experimental footage packets or timestamps changed'
for i,item in enumerate(b[:1710]):
 # Encoding order differs from presentation order; sorted PTS must remain 30 fps.
 pass
pts=sorted(float(x['pts_time']) for x in b)
assert np.max(np.abs(np.array(pts[:1710])-np.arange(1710)/30))<1e-6
assert np.max(np.abs(np.array(pts[1710:])-(57+np.arange(7247)/60)))<1e-6
subprocess.run(['ffmpeg','-v','error','-threads','4','-i',str(OUT),'-map','0:v','-map','0:a','-fps_mode','passthrough','-enc_time_base:v','1:60000','-f','null','-'],check=True)
assert OUT.stat().st_size<50000000,OUT.stat().st_size
report={'file':str(OUT),'bytes':OUT.stat().st_size,'duration':float(v['duration']),'resolution':[1920,1080],'slides_fps':30,'experiments_fps':60,'frames':8957,'audio_identical':True,'experiment_packets_and_timestamps_identical':True,'chapters_identical':True,'decode':'passed','slides_source':'Original lossless PNG layers and GIFs, original retiming preserved','photo_crop':[620,640,1688,1240],'photo_slot':[80,650,632,960],'redactions':'Existing manually located masks reapplied, resized photo manually pixelated, prior humanoid face patch retained','sha256':hashlib.sha256(OUT.read_bytes()).hexdigest()}
(R/'output/verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
