# Merck Drug Design Agent

A small, readable starter implementation for the UC Berkeley Data Discovery project.

The agent proposes SMILES with an LLM, validates molecules with RDKit, scores them with
trained surrogate models, selects a few candidates, optionally calls a docking function,
and saves each result to CSV so future iterations can use experiment history.

## Workflow

1. Ask the configured LLM for candidate SMILES.
2. Validate SMILES with RDKit and calculate QED and Morgan fingerprints.
3. Predict JAK2 and LCK docking scores using your trained Chemprop models.
4. Minimize the paper's selective-JAK2 objective, subject to a minimum QED gate.
5. Optionally run DOCKSTRING on the top candidates.
6. Save results in `data/experiment_history.csv`.
7. Use previous results as short feedback for the next LLM proposal.

**Objective:** the agent minimizes the DOCKSTRING selective-JAK2 objective from the
referenced paper:

```text
JAK2_score - min(LCK_score, -8.1) + 10 * (1 - QED)
```

The scores used during candidate selection are Chemprop predictions of DOCKSTRING
Vina affinities. More negative Vina scores indicate stronger predicted binding.
The formula rewards stronger JAK2 binding, weaker LCK binding (with the LCK term
capped at -8.1), and higher QED. Candidates also need to meet the project's extra
`MIN_QED` eligibility threshold (default `0.4`) to be selected for docking. The
objective is recorded in the `jak2_objective` column and lower values rank better.
Rows created with the older `combined_score` are retained in history but are not
compared with the new objective.

This objective and its docking-score direction come from the [DOCKSTRING paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC9364321/).

This is a research prototype, not a validated drug-discovery system. Predicted scores are not experimental evidence.

## 1. Activate an environment with Chemprop 2

```bash
conda activate drugdesign
cd /global/home/users/$USER/merck-drug-optimization
```

Copy these files into your existing repository, or clone/copy the folder there. Do not recreate the environment.

## 2. Check the runtime dependencies

Check what's already available:

```bash
python -c "import rdkit, pandas, numpy; print('basic packages OK')"
python -c "import chemprop; print('chemprop OK')"
python -c "import langchain_openai, langgraph; print('agent packages OK')"
```

Install only packages that are missing. Prefer your project's tested versions and your Savio module/conda setup:

```bash
conda install -c conda-forge rdkit pandas numpy scikit-learn
pip install langchain-openai langgraph
```

Chemprop versions have different APIs. Do not reinstall Chemprop if your trained models already work with the version in `drugdesign`.

## 3. Configure the NRP LLM API

Enter your API key without putting it directly in your code:

```bash
read -rsp "Enter your NRP API key: " NRP_API_KEY
echo
export NRP_API_KEY
export NRP_BASE_URL="https://ellm.nrp-nautilus.io/v1"
export LLM_MODEL="qwen3"
export QWEN_THINKING=0
```

Check the key is present (this does not print the key):

```bash
python -c 'import os; print("API key loaded:", bool(os.getenv("NRP_API_KEY")))'
```

## 4. Train the JAK2 and LCK models

The included `dockstring-dataset.tsv` contains SMILES and docking-score columns for both
targets. The training script validates those columns, writes one clean CSV per target,
and trains two Chemprop 2 regression models using scaffold-balanced 10:1:1
train/validation/test splits. This keeps related molecular scaffolds together when
evaluating generalization.

From the project directory, in the environment where `chemprop` works:

```bash
python train_models.py
```

The defaults match the supplied settings: message hidden size 4096, FFN hidden size 4096,
depth 9, and four FFN layers. The ensemble size is one, and other training settings use
Chemprop defaults. Training can take time and these model dimensions may require a
well-provisioned GPU. Check the Chemprop output for validation and held-out test RMSE/MAE
before using the models. The training data are Dockstring docking scores, so these are
surrogates for docking scores, not experimental binding measurements.

If the full model runs out of memory, retry with smaller dimensions. For example:

```bash
python train_models.py --message-hidden-dim 1000 --ffn-hidden-dim 1000 --depth 5 --ffn-num-layers 2
```

For a quick pipeline/performance pilot on a GPU, use a reproducible 20,000-row sample
per target, 10 epochs, and a smaller network:

```bash
python train_models.py --max-rows 20000 --epochs 10 --message-hidden-dim 500 --ffn-hidden-dim 500 --depth 3 --ffn-num-layers 2
```

This is a smoke-test baseline, not a substitute for training and evaluating on the full
dataset. Runtime still depends on the GPU and available resources.

Training writes target-specific CSVs under `models/training_data/` and checkpoints under
`models/trained/jak2/` and `models/trained/lck/`. Generated model files are not committed.
After training, point the agent at those directories:

```bash
export JAK2_MODEL_PATH="$PWD/models/trained/moderate_40434547/jak2"
export LCK_MODEL_PATH="$PWD/models/trained/moderate_40434547/lck"
```

The prediction adapter in `surrogate_models.py` uses the Chemprop 2 command-line API and
predicts a whole candidate batch at once. Keep the training and agent runs in the same
Chemprop environment. Check that both directories exist before launching the agent.

## 5. Run a safe demo (no API calls, no docking)

```bash
python agent.py --demo --iterations 2
```

This checks the molecule validation, scoring, ranking, and CSV history using a few example SMILES. It does not call the LLM, Chemprop, or DOCKSTRING. Demo results are saved separately in `data/demo_experiment_history.csv` and are not used as agent memory.

## 6. Run with the LLM and trained surrogate models

After training the checkpoints and setting the paths above:

```bash
python agent.py --iterations 2 --candidates 20 --dock-top 2
```

Docking is disabled by default. The agent and Chemprop stay in the `drugdesign`
environment. The adapter starts a separate worker in your Python 3.10 `docking`
environment, which docks selected candidates against the DOCKSTRING `JAK2` and
`LCK` targets. Vina affinities (kcal/mol) are saved to `docking_jak2` and
`docking_lck`.

Install DOCKSTRING in the `docking` environment and check that both targets
load:

```bash
conda activate docking
conda install -c conda-forge dockstring
python -c 'from dockstring import load_target; load_target("JAK2"); load_target("LCK"); print("DOCKSTRING targets OK")'
```

Then switch back to `drugdesign`, point the adapter at the Python executable in
the docking environment, and opt in to docking:

```bash
conda activate drugdesign
export DOCKING_PYTHON="$HOME/.conda/envs/docking/bin/python"
python agent.py --iterations 2 --candidates 10 --dock-top 2 --run-docking
```

Only molecules that pass the agent's filters are eligible for docking, up to
`--dock-top` per iteration. Each selected molecule is docked against both
targets, so this can add substantial runtime. Set `--cpus-per-task` in an
`sbatch` request to give Vina multiple CPUs; the adapter uses
`SLURM_CPUS_PER_TASK` when available. DOCKSTRING requires Open Babel and its
bundled AutoDock Vina executable. The adapter also accepts the path through
`DOCKING_PYTHON`; if unset, it looks for a sibling `docking` environment next to
the active conda environment.

## Files

- `llm_proposer.py`: asks the LLM for SMILES
- `molecule_tools.py`: RDKit validation and properties
- `surrogate_models.py`: adapters for your trained Chemprop models
- `scoring.py`: hard filters and ranking
- `docking.py`: optional DOCKSTRING adapter for JAK2/LCK Vina affinities
- `memory.py`: CSV experiment history and feedback
- `config.py`: settings and environment variables
- `data/experiment_history.csv`: generated history
- `tests/test_core.py`: basic unit tests

## Why some integrations are adapters

The Chemprop checkpoints and DOCKSTRING runtime are configured separately. Keep
the Chemprop version compatible with the checkpoints, and install DOCKSTRING in
the separate docking environment described above.

git status
git branch
git checkout drug-design-testing
git fetch origin
git add config.py scoring.py surrogate_models.py agent.py run_agent.sbatch docking.py llm_proposer.py memory.py README.md
git commit -m "Update JAK2 drug design workflow and HPC configuration"
git push -u origin drug-design-testing