"""Download the public mirror of the paper's Kaggle Study Three dataset."""
import hashlib
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent
tree = json.loads((ROOT / 'author/study3_mirror_tree.json').read_text(encoding='utf-8-sig'))
entries = [item for item in tree['tree'] if item['type'] == 'blob' and item['path'].split('/')[0] in {'defected', 'no_defected'}]


def fetch(item):
    target = ROOT / 'data/study3' / item['path']
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        content = target.read_bytes()
        digest = hashlib.sha1(b'blob ' + str(len(content)).encode() + b'\0' + content).hexdigest()
        if digest == item['sha']:
            return
    url = 'https://raw.githubusercontent.com/pingkung/3D-Printer-Defected-Dataset/' + tree['sha'] + '/' + item['path']
    for attempt in range(3):
        try:
            content = urllib.request.urlopen(url, timeout=90).read()
            digest = hashlib.sha1(b'blob ' + str(len(content)).encode() + b'\0' + content).hexdigest()
            if digest != item['sha']:
                raise ValueError('Checksum mismatch: ' + item['path'])
            target.write_bytes(content)
            return
        except Exception:
            if attempt == 2:
                raise


with ThreadPoolExecutor(max_workers=10) as pool:
    list(pool.map(fetch, entries))
print(f'Downloaded and verified {len(entries)} Study Three images', flush=True)
