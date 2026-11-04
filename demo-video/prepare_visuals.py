import json, subprocess
from pathlib import Path
from pypdf import PdfReader, PdfWriter
from pypdf.generic import ContentStream
from PIL import Image

ROOT = Path(__file__).resolve().parent
reader = PdfReader(ROOT / 'source/original.pdf')
presentation = json.loads((ROOT/'source/presentation.json').read_text())
out = ROOT/'render/layers'
out.mkdir(exist_ok=True)
path_paints = {b'S', b's', b'f', b'F', b'f*', b'B', b'B*', b'b', b'b*'}
other_paints = {b'Do', b'sh', b'Tj', b'TJ', b"'", b'"'}
manifest=[]
for index, page in enumerate(reader.pages,1):
    if index not in (1,3):
        subprocess.run(['pdftocairo','-png','-singlefile','-f',str(index),'-l',str(index),'-scale-to-x','1920','-scale-to-y','1080',str(ROOT/'source/original.pdf'),str(out/f'slide-{index}')],check=True)
        manifest.append({'index':index,'static':str(out/f'slide-{index}.png')})
        continue
    ops=ContentStream(page.get_contents(),reader).operations
    cut=next(i for i,(_,op) in enumerate(ops) if op==b'Do')
    for label in ('under','over'):
        writer=PdfWriter()
        writer.add_page(page)
        stream=ContentStream(page.get_contents(),reader)
        edited=[]
        for i,(args,op) in enumerate(stream.operations):
            omit = (i>=cut if label=='under' else i<=cut)
            if omit and op in path_paints: edited.append(([],b'n'))
            elif omit and op in other_paints: continue
            else: edited.append((args,op))
        stream.operations=edited
        writer.pages[0].replace_contents(stream)
        file=out/f'slide-{index}-{label}.pdf'
        writer.write(file)
        subprocess.run(['pdftocairo','-png','-transp','-singlefile','-scale-to-x','1920','-scale-to-y','1080',str(file),str(file.with_suffix(''))],check=True)
    element=next(e for e in presentation['slides'][index-1]['pageElements'] if 'image' in e)
    t=element['transform']; size=element['size'];factor=1920/presentation['pageSize']['width']['magnitude']
    bounds=[round(t['translateX']*factor),round(t['translateY']*factor),round(size['width']['magnitude']*t['scaleX']*factor),round(size['height']['magnitude']*t['scaleY']*factor)]
    gif=ROOT/f'source/media/image{index}.gif'
    im=Image.open(gif); duration=0
    for n in range(im.n_frames):im.seek(n);duration+=im.info['duration']/1000
    im.seek(0)
    composite=Image.open(out/f'slide-{index}-under.png').convert('RGBA')
    composite.alpha_composite(im.convert('RGBA').resize(tuple(bounds[2:]),Image.Resampling.LANCZOS),tuple(bounds[:2]))
    composite.alpha_composite(Image.open(out/f'slide-{index}-over.png').convert('RGBA'))
    composite.save(out/f'slide-{index}-check.png')
    manifest.append({'index':index,'gif':str(gif),'under':str(out/f'slide-{index}-under.png'),'over':str(out/f'slide-{index}-over.png'),'bounds':bounds,'cycle_seconds':round(duration,3),'gif_frames':im.n_frames})
(ROOT/'render/visuals.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps(manifest,indent=2))
