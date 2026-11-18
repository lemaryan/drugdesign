import csv, io, contextlib
from collections import Counter
from rdkit import Chem
from rdkit import rdBase
rdBase.LogToPythonStderr()

rows = list(csv.DictReader(open('data/experiment_history.csv')))
inv = [r['smiles'] for r in rows if r['proposal_status'] == 'invalid']
uniq = list(dict.fromkeys(inv))
print('invalid rows', len(inv), '| unique invalid SMILES', len(uniq))

def classify(s):
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        try:
            Chem.MolFromSmiles(s)
        except Exception as exc:
            return 'exception: ' + str(exc)[:60]
    e = err.getvalue().lower()
    if 'kekulize' in e: return 'cannot kekulize aromatic ring'
    if 'ring closure' in e or 'unclosed' in e: return 'unclosed / duplicate ring-closure digit'
    if 'valence' in e: return 'explicit valence too high'
    if 'syntax' in e or 'while parsing' in e or 'failed parsing' in e: return 'SMILES syntax error'
    if 'extra close' in e or 'parenthes' in e: return 'unbalanced parentheses'
    if not e.strip(): return 'no error logged (parses in isolation)'
    return 'other: ' + e.strip().splitlines()[0][:70]

c = Counter(classify(s) for s in uniq)
tot = sum(c.values())
for k, v in c.most_common(12):
    print(f'{v:6d}  {100*v/tot:5.1f}%  {k}')
