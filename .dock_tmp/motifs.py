import csv, io, contextlib, re
from collections import Counter
from rdkit import Chem
from rdkit import rdBase
rdBase.LogToPythonStderr()

rows = list(csv.DictReader(open('data/experiment_history.csv')))
inv = [r['smiles'] for r in rows if r['proposal_status'] == 'invalid']
freq = Counter(inv)
print('=== most-repeated invalid SMILES ===')
for s, n in freq.most_common(8):
    print(f'{n:4d}x  {s[:72]}')

def kek(s):
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        Chem.MolFromSmiles(s)
    return 'kekulize' in err.getvalue().lower()

uniq = list(dict.fromkeys(inv))
kk = [s for s in uniq if kek(s)]
print(f'\n=== sample kekulization failures ({len(kk)} unique) ===')
for s in kk[:10]:
    print('  ', s[:72])

# which fused-aromatic ring patterns show up
pats = {
    'fused bicyclic (digit pair like n12 / c34)': r'[a-z]\d\d',
    'lowercase n in ring': r'n\d',
    'bridged N (n...n in same ring system)': r'n.*n',
}
print('\n=== motif frequency among unique kekulize failures ===')
for name, p in pats.items():
    print(f'  {sum(1 for s in kk if re.search(p, s)):5d}  {name}')
