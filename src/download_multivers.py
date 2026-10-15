"""Fetch the immutable official MultiVerS checkpoint with resumable ranges."""
from pathlib import Path
import argparse, concurrent.futures, hashlib, json, math, requests, time
R2=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');a=p.parse_args()
root=R2/'data/models/multivers';root.mkdir(parents=True,exist_ok=True)
record=json.loads((root/'checkpoint_manifest.json').read_text(encoding='utf-8'))
version='GYe58tFsyvvn1n7rib5H4wrsiq32ZY5Y'
expected_sha256='6b48f1b83eaedbc2570bc50135518ef3cf6f3c6dcdd380a9a3a433918e7f982d'
assert record['version_id']==version and record['sha256']==expected_sha256
url=record['url']
assert url.endswith('?versionId='+version)
head=requests.head(url,timeout=30);head.raise_for_status()
assert head.headers['x-amz-version-id']==version
total=int(head.headers['Content-Length'])
assert total==record['bytes']==5253768950
size=2**20 if a.smoke else 32*2**20
parts=root/'parts';parts.mkdir(exist_ok=True)
count=4 if a.smoke else math.ceil(total/size)
start=time.monotonic()
def get(i):
    first=i*size;last=min(total,(i+1)*size)-1
    target=parts/(('smoke_'+version+'_' if a.smoke else 'part')+f'{i:04d}')
    if target.exists() and target.stat().st_size==last-first+1:return i
    for retry in range(5):
        try:
            offset=target.stat().st_size if target.exists() else 0
            r=requests.get(url,headers={'Range':f'bytes={first+offset}-{last}'},stream=True,timeout=(25,60))
            r.raise_for_status();assert r.status_code==206
            assert r.headers['Content-Range'].startswith(f'bytes {first+offset}-')
            with target.open('ab' if offset else 'wb') as f:
                for chunk in r.iter_content(2**18):f.write(chunk)
            assert target.stat().st_size==last-first+1
            return i
        except Exception as e:print('Retry',i,retry,str(e)[:80],flush=True)
    raise RuntimeError(i)
with concurrent.futures.ThreadPoolExecutor(8) as pool:
    for n,i in enumerate(pool.map(get,range(count)),1):
        if n%8==0 or a.smoke:print(n,'/',count,round(time.monotonic()-start,1),'s',flush=True)
if a.smoke:
    (root/'download_smoke.json').write_text(json.dumps({'passed':True,'url':url,'bytes':count*size}),encoding='utf-8')
else:
    assert (root/'download_smoke.json').exists()
    h=hashlib.sha256();target=root/'fever_sci.ckpt'
    with target.with_suffix('.assembling').open('wb') as f:
        for i in range(count):
            with (parts/f'part{i:04d}').open('rb') as source:
                for chunk in iter(lambda:source.read(2**20),b''):f.write(chunk);h.update(chunk)
    assert h.hexdigest()==expected_sha256,'Checkpoint hash differs from the frozen experiment'
    target.with_suffix('.assembling').replace(target)
    print('Checkpoint complete',h.hexdigest(),flush=True)
