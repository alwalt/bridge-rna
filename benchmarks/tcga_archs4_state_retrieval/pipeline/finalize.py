#!/usr/bin/env python3
"""Write a checksum inventory for all benchmark deliverables."""
from pathlib import Path
import hashlib, json
H=Path(__file__).resolve().parents[1]
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(16<<20),b''): h.update(b)
 return h.hexdigest()
rows=[]
for root in [H/'results',H/'work']:
 for p in sorted(root.rglob('*')):
  if p.is_file() and p.name not in {'output_manifest.json'}:
   rows.append({'path':str(p.relative_to(H)),'bytes':p.stat().st_size,'sha256':sha(p),'regenerable':str(p.relative_to(H)).startswith('work/')})
(H/'results'/'output_manifest.json').write_text(json.dumps({'files':rows},indent=2))
print(f"Wrote checksums for {len(rows)} outputs")
