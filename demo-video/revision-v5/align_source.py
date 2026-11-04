from pathlib import Path
import cv2,numpy as np,json,subprocess
R=Path(__file__).resolve().parent
NEW='/media/wolfie/T9/Test Videos/TargetNextWall-BEV-Windowopeded.mp4'
BASE=R/'segments/new-wheeled-source.mp4'
cv2.setNumThreads(2)
def descriptor(im):
    im=cv2.resize(im,(120,66),interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(im,cv2.COLOR_BGR2GRAY).astype(np.float32)
if __name__=='__main__':
    p=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v','-show_frames','-show_entries','frame=best_effort_timestamp_time','-of','json',NEW]))
    pts=np.array([float(f['best_effort_timestamp_time']) for f in p['frames']])
    c=cv2.VideoCapture(NEW);arr=[]
    while True:
        ok,im=c.read()
        if not ok:break
        arr.append(descriptor(im[270:810,0:960]))
    c.release();arr=np.array(arr)
    assert len(arr)==len(pts)
    np.savez_compressed(R/'inspect/ours-descriptors.npz',pts=pts,descs=arr)
    c=cv2.VideoCapture(str(BASE));rows=[]
    for t in [34.666667,35,36,37,38,39,39.3,39.5,42,44,44.5,45,46,47,48,49,50,51,52,52.667,53,54,55,56,57,58,59,60,60.5,61,61.4]:
        c.set(cv2.CAP_PROP_POS_FRAMES,round(t*60));ok,im=c.read();assert ok
        d=descriptor(im[66:426,0:640]);valid=np.ones(d.shape,bool)
        valid[:25,:36]=False
        if 39.3<t<44.3:valid[7:,:]=False
        if t>59.8:valid[42:,:]=False
        err=np.mean(np.abs(arr[:,valid]-d[valid]),axis=1)
        k=int(np.argmin(err));rows.append([t,k,float(pts[k]),float(err[k])])
    c.release();(R/'inspect/alignment-samples.json').write_text(json.dumps(rows,indent=2))
    print(json.dumps(rows))
