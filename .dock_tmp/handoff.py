import csv
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import rdBase
rdBase.DisableLog('rdApp.*')

rows = list(csv.DictReader(open('data/experiment_history.csv')))
def has(r,c): return r.get(c) not in (None,'','nan','NaN')
docked = [r for r in rows if has(r,'docking_jak2') and has(r,'actual_objective')]
for r in docked: r['_it'] = int(r['iteration']); r['_obj'] = float(r['actual_objective'])
docked.sort(key=lambda r: r['_it'])

def scaf(s):
    m = Chem.MolFromSmiles(s)
    if not m: return None
    try: return MurckoScaffold.MurckoScaffoldSmiles(mol=m)
    except Exception: return None

# seeds
seeds = {scaf(r['smiles']) for r in csv.DictReader(open('data/dockstring_seed_memory.csv'))}
seeds.discard(None)

first = {}          # scaffold -> (iteration, mode)
members = defaultdict(list)
for r in docked:
    s = scaf(r['smiles'])
    if not s: continue
    members[s].append(r)
    if s not in first: first[s] = (r['_it'], r['proposal_mode'] or '(none)')

# scaffolds discovered by explore that exploit later worked on
handoff = []
for s, (it, mode) in first.items():
    if mode != 'explore': continue
    later_exploit = [r for r in members[s] if r['proposal_mode'] == 'exploit' and r['_it'] > it]
    if later_exploit:
        disc = min(r['_obj'] for r in members[s] if r['proposal_mode'] == 'explore')
        best = min(r['_obj'] for r in later_exploit)
        handoff.append((best, disc, s, it, len(later_exploit)))
handoff.sort()
print(f'explore-discovered scaffolds that exploit later built on: {len(handoff)}')
n_disc = sum(1 for sc,(i,m) in first.items() if m == 'explore')
print('  (of %d scaffolds explore discovered)\n' % n_disc)
for best, disc, s, it, n in handoff[:10]:
    print(f'  best_exploit={best:7.3f}  explore_disc={disc:7.3f}  it={it:4d}  n_exploit={n:3d}  {s[:50]}')

# where does the overall best scaffold come from?
print('\n=== provenance of the top-20 docked molecules\' scaffolds ===')
top = sorted(docked, key=lambda r: r['_obj'])[:20]
from collections import Counter
prov = Counter()
for r in top:
    s = scaf(r['smiles'])
    origin = 'SEED (library)' if s in seeds else f'first seen in {first.get(s,(None,"?"))[1]} @ it{first.get(s,(0,0))[0]}'
    prov[origin] += 1
for k, v in prov.most_common(): print(f'  {v:3d}  {k}')
