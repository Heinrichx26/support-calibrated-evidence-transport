"""Preserve the official MultiVerS source and checkpoint instructions."""
from pathlib import Path
import requests, json, hashlib, io
from zipfile import ZipFile

R2=Path(__file__).resolve().parents[1]
out=R2/'src/vendor/mv'
out.mkdir(parents=True,exist_ok=True)
meta=R2/'data/models/multivers'
meta.mkdir(parents=True,exist_ok=True)
commit='a6ce033f0e17ae38c1f102eae1ee4ca213fbbe2e'
url='https://codeload.github.com/dwadden/multivers/zip/'+commit
r=requests.get(url,timeout=60)
r.raise_for_status()
with ZipFile(io.BytesIO(r.content)) as archive:
    for item in archive.infolist():
        if item.is_dir():continue
        name='/'.join(item.filename.split('/')[1:])
        target=out/name
        assert target.resolve().is_relative_to(out.resolve())
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(archive.read(item))
record={'url':url,'commit':commit,'archive_sha256':hashlib.sha256(r.content).hexdigest()}
(meta/'source_tree.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
print('Official MultiVerS source',record['archive_sha256'],flush=True)
