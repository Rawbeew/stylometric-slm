# Byte-level verification of scripts
import hashlib, os, sys
from pathlib import Path

ROOT = Path(r'C:\Users\alaga\ghwork\stylometric-slm')
files = [
    'scripts/pull_gutenberg.py',
    'scripts/pull_ctext.py',
    'scripts/build_split.py',
    'scripts/push_to_hf.py',
    'notebooks/finetune_mt5.ipynb',
    'README.md',
]

print(f"{'file':<40} {'bytes':>8}  {'sha256':<12}  syntax")
print('-' * 90)
total_bytes = 0
for rel in files:
    p = ROOT / rel
    if not p.exists():
        print(f"{rel:<40} MISSING")
        continue
    size = p.stat().st_size
    total_bytes += size
    h = hashlib.sha256(p.read_bytes()).hexdigest()[:12]
    print(f"{rel:<40} {size:>8}  {h}  ", end='')
    # syntax check for .py
    if rel.endswith('.py'):
        try:
            compile(p.read_text(encoding='utf-8'), str(p), 'exec')
            print('OK')
        except SyntaxError as e:
            print(f'SYNTAX ERROR: {e}')
            sys.exit(1)
    elif rel.endswith('.ipynb'):
        try:
            import json
            json.loads(p.read_text(encoding='utf-8'))
            print('JSON OK')
        except Exception as e:
            print(f'JSON ERROR: {e}')
            sys.exit(1)
    else:
        print('(no lint)')

print(f"\ntotal: {total_bytes} bytes across {len(files)} files")
print(f"project root: {ROOT}")
