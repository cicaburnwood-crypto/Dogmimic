"""Offline narration. Source speaker notes are preserved separately verbatim."""
import json, re, time
from pathlib import Path
import numpy as np
import soundfile as sf
import onnxruntime as ort
from kokoro_onnx import Kokoro

ROOT=Path(__file__).resolve().parent
opts=ort.SessionOptions()
opts.intra_op_num_threads=4
opts.inter_op_num_threads=1
session=ort.InferenceSession(str(ROOT/'models/kokoro-v1.0.onnx'),sess_options=opts,providers=['CPUExecutionProvider'])
kokoro=Kokoro.from_session(session,str(ROOT/'models/voices-v1.0.bin'))
voice='bm_george'
scripts=json.loads((ROOT/'source/scripts.json').read_text())
def pronunciation(text):
    replacements={'EchoNav':'Echo Nav','RGB':'R G B','BEV':'B E V','mIoU':'em eye oh you','47.6':'forty-seven point six','58.4':'fifty-eight point four','86.3':'eighty-six point three','73.3':'seventy-three point three'}
    for a,b in replacements.items(): text=text.replace(a,b)
    return text.replace('’',"'")
for slide in scripts:
    i=slide['index'];chunks=[];segments=[];offset=0
    sentences=re.split(r'(?<=[.!?])\s+(?=[A-Z])',slide['script'])
    for j,sentence in enumerate(sentences):
        spoken=pronunciation(sentence)
        start=time.monotonic()
        a,sr=kokoro.create(spoken,voice=voice,speed=1.10,lang='en-gb')
        # Preserve speech, remove only excess boundary silence, add a controlled sentence gap.
        active=np.flatnonzero(np.abs(a)>0.004)
        if len(active):a=a[max(0,active[0]-int(.04*sr)):min(len(a),active[-1]+int(.12*sr))]
        segments.append({'text':sentence,'spoken':spoken,'start':offset,'end':offset+len(a)/sr})
        chunks.append(a);offset+=len(a)/sr
        if j<len(sentences)-1:
            chunks.append(np.zeros(int(.33*sr),dtype=np.float32));offset+=.33
        print(f'Slide {i} sentence {j+1}: {len(a)/sr:.2f}s generated in {time.monotonic()-start:.2f}s',flush=True)
    full=np.concatenate(chunks)
    sf.write(ROOT/f'audio/slide-{i}.wav',full,sr,subtype='PCM_24')
    slide.update(voice=voice,speed=1.10,duration=len(full)/sr,segments=segments)
    (ROOT/'audio/narration-manifest.json').write_text(json.dumps(scripts,indent=2))
print('Narration completed locally.',flush=True)
