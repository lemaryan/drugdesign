import csv, io, contextlib, re
from collections import Counter
from rdkit import Chem
from rdkit import rdBase
rdBase.LogToPythonStderr()

rows = list(csv.DictReader(open('data/experiment_history.csv')))
inv = [r['smiles'] for r in rows if r['proposal_status'] == 'invalid']
freq = Counter(inv)
uniq = list(freq)

def kek(s):
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        Chem.MolFromSmiles(s)
    return 'kekulize' in err.getvalue().lower()

kk = [s for s in uniq if kek(s)]

# substituent attached directly after an aromatic ring-closure digit on n/c
nsub = re.compile(r'[a-z]\d(?:[A-Z][a-z]?|\()')
# aromatic n carrying a ring-closure digit then a substituent atom => N-substituted pyridine-like
npyr = re.compile(r'n\d(?:C|O|N|F|Cl|Br|S)')
no_nH = [s for s in kk if '[nH]' not in s and re.search(r'n', s)]

groups = {
  'substituent written onto an aromatic ring N (e.g. cn1C, nn1C, cn1O)': [s for s in kk if npyr.search(s)],
  'aromatic n present but no [nH] anywhere': no_nH,
}
print(f'unique invalid: {len(uniq)} | unique kekulize failures: {len(kk)}')
print(f'total invalid ROWS: {len(inv)} | mean repeats per unique string: {len(inv)/len(uniq):.1f}\n')
for name, g in groups.items():
    rowcount = sum(freq[s] for s in g)
    print(f'{len(g):5d} unique / {rowcount:5d} rows  {name}')

print('\n=== how concentrated is the waste? ===')
top = freq.most_common(50)
print(f'top 50 unique invalid strings account for {sum(n for _,n in top)} of {len(inv)} invalid rows '
      f'({100*sum(n for _,n in top)/len(inv):.0f}%)')
once = sum(1 for s in uniq if freq[s] == 1)
print(f'unique strings proposed exactly once: {once} ({100*once/len(uniq):.0f}% of unique)')
