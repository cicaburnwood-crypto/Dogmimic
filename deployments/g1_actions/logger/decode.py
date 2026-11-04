"""Decode a closed logger segment to complete JSONL; no DDS connection required."""
import argparse,dataclasses,json,sys
from record import frames,registry
p=argparse.ArgumentParser();p.add_argument('segments',nargs='+');p.add_argument('--topic');p.add_argument('--out',required=True);a=p.parse_args()
kinds=registry();count=0
with open(a.out,'w') as f:
 for path in a.segments:
  for h,payload in frames(path):
   if a.topic and h['topic']!=a.topic:continue
   data=json.loads(payload) if h['encoding']=='json' else dataclasses.asdict(kinds[h['type']].deserialize(payload))
   f.write(json.dumps(dict(h,data=data),ensure_ascii=False,separators=(',',':'))+'\n');count+=1
print(json.dumps({'records':count,'output':a.out}))
