"""Restore the exact upstream HealthVer/COVID-Fact processed data version.

Provider text is retrieved from its original host. This release does not
relicense or mirror source documents. A HEAD/range smoke check precedes restore.
"""
from pathlib import Path
import argparse
import hashlib
import json
import tarfile
from urllib.request import Request, urlopen

R2 = Path(__file__).resolve().parents[1]
ROOT = R2 / 'data/multivers_external'
VERSION = 'NtLPjT4MeLa6Kv8n6zEgVB0PxUs._EFD'
URL = 'https://scifact.s3.us-west-2.amazonaws.com/longchecker/latest/data.tar.gz?versionId=' + VERSION
SHA256 = '0ce25066f18572af3307438a08d0d318a734d97d1d8c27ca43b51e81ae3d683c'
SIZE = 3610222


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch(smoke):
    ROOT.mkdir(parents=True, exist_ok=True)
    gate = ROOT / 'provider_download_smoke.json'
    if smoke:
        with urlopen(Request(URL, method='HEAD'), timeout=30) as response:
            assert response.headers['x-amz-version-id'] == VERSION
            assert int(response.headers['Content-Length']) == SIZE
        with urlopen(Request(URL, headers={'Range': 'bytes=0-4095'}), timeout=30) as response:
            assert response.status == 206
            assert len(response.read()) == 4096
        gate.write_text(json.dumps({'passed': True, 'url': URL, 'version_id': VERSION,
                                    'expected_sha256': SHA256, 'expected_bytes': SIZE}, indent=2), encoding='utf-8')
        print('Immutable provider version and partial download verified.')
        return
    check = json.loads(gate.read_text())
    assert check['passed'] and check['expected_sha256'] == SHA256
    archive = ROOT / 'official_data.tar.gz'
    if not archive.exists() or digest(archive) != SHA256:
        partial = ROOT / 'official_data.tar.gz.partial'
        with urlopen(URL, timeout=60) as response, partial.open('wb') as stream:
            assert response.headers['x-amz-version-id'] == VERSION
            while chunk := response.read(2**20):
                stream.write(chunk)
        assert partial.stat().st_size == SIZE and digest(partial) == SHA256
        partial.replace(archive)
    with tarfile.open(archive) as source:
        for member in source.getmembers():
            if not member.isfile() or not member.name.startswith(('data/healthver/', 'data/covidfact/')):
                continue
            target = ROOT / member.name
            assert target.resolve().is_relative_to(ROOT.resolve())
            target.parent.mkdir(parents=True, exist_ok=True)
            with source.extractfile(member) as stream:
                content = stream.read()
            if target.exists():
                assert target.read_bytes() == content, 'Existing provider input differs from the pinned archive.'
            else:
                target.write_bytes(content)
    print('Pinned provider data restored; source text retains upstream terms.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke', action='store_true')
    fetch(parser.parse_args().smoke)
