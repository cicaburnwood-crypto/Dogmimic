"""Download public G1 retargeted CSVs; retain sizes and SHA256 for provenance."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import time
import urllib.parse
import urllib.request

out = Path('data/raw/lafan1_g1')
out.mkdir(parents=True, exist_ok=True)
base = 'https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset/resolve/main/'
api = 'https://huggingface.co/api/datasets/lvhaidong/LAFAN1_Retargeting_Dataset/tree/main/g1?limit=100'
with urllib.request.urlopen(api, timeout=60) as response:
    entries = json.load(response)
assert len(entries) == 40

def download(item):
    name = Path(item['path']).name
    target = out / name
    for attempt in range(4):
        try:
            if not target.exists() or target.stat().st_size != item['size']:
                request = urllib.request.Request(base + urllib.parse.quote(item['path']), headers={'User-Agent': 'SuperMimic-environment-setup'})
                with urllib.request.urlopen(request, timeout=90) as response:
                    target.write_bytes(response.read())
            assert target.stat().st_size == item['size']
            digest = hashlib.sha256(target.read_bytes()).hexdigest()
            print(name, target.stat().st_size, flush=True)
            return {**item, 'sha256': digest}
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2 * (attempt + 1))

with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    manifest = list(pool.map(download, entries))
Path('data/lafan1-download-manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
print('Downloaded and verified', len(manifest), 'clips', flush=True)
