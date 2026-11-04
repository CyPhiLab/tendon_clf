import sys, json
from multiprocessing import Pool
import fit
tag = sys.argv[1]; cfgs = json.load(open(sys.argv[2]))
with Pool(4) as pool:
    rows = fit.evaluate(cfgs, pool=pool)
fit.log(tag, rows)
for c, J, p, _ in sorted(rows, key=lambda r: r[1]):
    print(f'J={J:7.1f} {json.dumps({k: round(v, 2) if isinstance(v, float) else v for k, v in p.items()}) if "error" not in p else p}  {json.dumps(c)}', flush=True)
