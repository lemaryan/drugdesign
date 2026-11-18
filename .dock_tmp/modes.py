import csv
from collections import Counter, defaultdict
import statistics as st
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import rdBase
rdBase.DisableLog('rdApp.*')

rows = list(csv.DictReader(open('data/experiment_history.csv')))
def has(r,c): return r.get(c) not in (None,'','nan','NaN')

# --- yield by mode (all proposals) ---
print('=== proposal outcomes by mode ===')
tab = defaultdict(Counter)
for r in rows:
    tab[r['proposal_mode'] or '(none)'][r['proposal_status']] += 1
for mode, c in tab.items():
    tot = sum(c.values())
    inv = c['invalid']; dup = c['duplicate'] + c['dataset_duplicate']; sc = c['scored']
    print(f'{mode:<8} total={tot:6d}  invalid={inv:5d} ({100*inv/tot:4.1f}%)  dup={dup:5d} ({100*dup/tot:4.1f}%)  scored={sc:5d} ({100*sc/tot:4.1f}%)')

# --- docked performance by mode ---
docked = [r for r in rows if has(r,'docking_jak2') and has(r,'actual_objective')]
print(f'\n=== docked molecules by mode (n={len(docked)}) ===')
by = defaultdict(list)
for r in docked:
    by[r['proposal_mode'] or '(none)'].append((float(r['actual_objective']), r))
for mode, v in sorted(by.items()):
    o = sorted(x[0] for x in v)
    print(f'{mode:<8} n={len(o):4d}  best={o[0]:7.3f}  median={st.median(o):7.3f}  '
          f'mean={st.mean(o):7.3f}  <-9.40(lib best)={sum(1 for x in o if x< -9.40):3d}  <-9.0={sum(1 for x in o if x< -9.0):3d}')

# --- who owns the leaderboard ---
allv = sorted(((float(r['actual_objective']), r['proposal_mode'] or '(none)') for r in docked))
for n in (10, 20, 50, 100):
    print(f'top {n:3d}: ' + '  '.join(f'{m}={c}' for m,c in Counter(m for _,m in allv[:n]).most_common()))

# --- scaffold diversity by mode ---
print('\n=== distinct Bemis-Murcko scaffolds among docked ===')
for mode, v in sorted(by.items()):
    sc = set()
    for _, r in v:
        m = Chem.MolFromSmiles(r['smiles'])
        if m:
            try: sc.add(MurckoScaffold.MurckoScaffoldSmiles(mol=m))
            except Exception: pass
    print(f'{mode:<8} n={len(v):4d}  distinct scaffolds={len(sc):4d}  ({len(sc)/len(v):.2f} per docked molecule)')
