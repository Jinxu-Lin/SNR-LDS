"""Recompute CIFAR-2 native-head and matched-square diagnostics from filed arrays.
Run with the da environment from the repository root. No model fitting.
Both readouts in the paired control use heads selected from ORIGINAL signed scores.
"""
import sys, json
from pathlib import Path
import os
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
DATA=Path(os.environ.get("BALDS_DATA_ROOT", str(ROOT/"_Data"))).expanduser().resolve()
sys.path.insert(0,str(ROOT/'Codes/Figure'))
from _common import Cell, CORE, SEEDS, TRACKS, KAPPA_GRID, head_mask, keep_only
OUT=ROOT/"Paper/ICLR/Figures/section31"
records=[]
for method,_ in CORE:
 for track in TRACKS:
  for seed in SEEDS:
   c=Cell(method,'cifar2_5k',track,seed)
   native=c.a**2 if method=='das_T100' else c.a
   for pct in KAPPA_GRID:
    mask=head_mask(c.a,c.k(pct),'pos')
    values={}
    for name,a in [('original_same_head',keep_only(c.a,mask)),('square_same_head',keep_only(c.a,mask,c.a**2)),('native_head',keep_only(native,head_mask(native,c.k(pct),'pos')))]:
     per=c.per_query(a)
     values[name]={'mean':float(per.mean()),'per_query':per.tolist()}
    records.append(dict(method=method,track=track,seed=seed,head_pct=pct,**values))
   print(method,track,seed,flush=True)
result={'dataset':'cifar2_5k','seeds':list(SEEDS),'head_grid':list(KAPPA_GRID),'paired_selection':'largest original signed scores, identical membership for original and ordinary square','native_selection':'largest native scores; DAS squared exactly once','scores':'_Data/scores/{method}/cifar2_5k[_val]/seed_{seed}/scores.npy','responses':'_Data/results/gt_matrix_fm_cifar2_5k[_val]_seed_{seed}.npy','masks':'_Data/subsets/cifar2_5k_masks.pkl','aggregation':'deleted-row score sum; mean per-query Spearman, then equal mean over model seeds','records':records}
(OUT/'square_heads.json').write_text(json.dumps(result,indent=2))
for method,_ in CORE:
 for track in TRACKS:
  print(method,track,[(p,{name:round(100*np.mean([r[name]['mean'] for r in records if r['method']==method and r['track']==track and r['head_pct']==p]),2) for name in ['original_same_head','square_same_head','native_head']}) for p in [5,100]])
