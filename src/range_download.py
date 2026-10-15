from pathlib import Path
import argparse
import concurrent.futures
import hashlib
import json
import math
import time
import requests

R2 = Path(__file__).resolve().parents[1]
SOURCES = {
    'fever': ('https://zenodo.org/api/files/91ad9fca-c15e-4069-80f9-bac6ec5c2d9b/wiki-pages.zip', 1713485474, 'wiki-pages.zip'),
    'hover': ('https://nlp.cs.unc.edu/data/hover/wiki_wo_links.db', 2156273664, 'wiki_wo_links.db')
}
parser = argparse.ArgumentParser()
parser.add_argument('dataset', choices=SOURCES)
parser.add_argument('--smoke', action='store_true')
parser.add_argument('--workers', type=int, default=8)
args = parser.parse_args()
url, total, filename = SOURCES[args.dataset]
out = R2 / 'data/raw' / args.dataset
out.mkdir(parents=True, exist_ok=True)
part_dir = out / 'download_parts'
part_dir.mkdir(exist_ok=True)
size = 2**20 if args.smoke else 32 * 2**20
count = args.workers if args.smoke else math.ceil(total / size)
start_time = time.monotonic()

def chunk(i):
    first, last = i * size, min(total, (i + 1) * size) - 1
    target = part_dir / (('smoke' if args.smoke else 'part') + f'-{i:04d}')
    expected = last - first + 1
    if target.exists() and target.stat().st_size == expected:
        return i, expected, 0
    for attempt in range(6):
        offset = target.stat().st_size if target.exists() else 0
        try:
            response = requests.get(url, headers={'Range': f'bytes={first+offset}-{last}'}, timeout=(30, 90), stream=True)
            response.raise_for_status()
            assert response.status_code == 206 and response.headers.get('Content-Range', '').startswith(f'bytes {first+offset}-')
            with target.open('ab' if offset else 'wb') as file:
                for data in response.iter_content(2**18):
                    file.write(data)
            assert target.stat().st_size == expected
            return i, expected, attempt
        except Exception as error:
            print(args.dataset, 'part', i, 'retry', attempt, str(error)[:100], flush=True)
    raise RuntimeError(f'Failed chunk {i}')

completed = 0
with concurrent.futures.ThreadPoolExecutor(args.workers) as executor:
    futures = [executor.submit(chunk, i) for i in range(count)]
    for future in concurrent.futures.as_completed(futures):
        i, length, retries = future.result()
        completed += length
        print(args.dataset, 'completed part', i, f'{completed/1e6:.1f} MB in {time.monotonic()-start_time:.1f}s', flush=True)
if args.smoke:
    (out / 'download_smoke.json').write_text(json.dumps({'bytes': completed, 'seconds': time.monotonic()-start_time, 'workers': args.workers}), encoding='utf-8')
else:
    target = out / filename
    temporary = out / (filename + '.assembling')
    sha = hashlib.sha256()
    with temporary.open('wb') as dest:
        for i in range(count):
            with (part_dir / f'part-{i:04d}').open('rb') as source:
                while data := source.read(2**20):
                    sha.update(data)
                    dest.write(data)
    assert temporary.stat().st_size == total
    temporary.replace(target)
    (out / 'download_manifest.json').write_text(json.dumps({'url': url, 'bytes': total, 'sha256': sha.hexdigest(), 'elapsed_seconds': time.monotonic()-start_time, 'workers': args.workers}, indent=2), encoding='utf-8')
    print('Completed', target, flush=True)
