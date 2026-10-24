# Merck Drug Design Agent

A small, readable starter implementation for the UC Berkeley Data Discovery project.

The agent proposes SMILES with an LLM, validates molecules with RDKit, scores them with
trained surrogate models, selects a few candidates, optionally calls a docking function,
and saves each result to CSV so future iterations can use experiment history.

## Workflow

1. Ask the configured LLM for candidate SMILES.
2. Validate SMILES with RDKit and calculate QED and Morgan fingerprints.
3. Predict JAK2 and LCK docking scores using your trained Chemprop models.
4. Rank candidates using hard filters and a simple combined score.
5. Optionally run DOCKSTRING on the top candidates.
6. Save results in `data/experiment_history.csv`.
7. Use previous results as short feedback for the next LLM proposal.

**Important:** docking scores are treated as negative values where more negative means stronger predicted binding. Therefore:
- JAK2: lower / more negative is better.
- LCK: higher / less negative is better, to favor selectivity over LCK.
- QED: higher is better.

This is a research prototype, not a validated drug-discovery system. Predicted scores are not experimental evidence.

## 1. Activate an environment with Chemprop 2

```bash
conda activate drugdesign
cd /global/home/users/$USER/drug_design_clean
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

## 3b. Load your GPU in, submit a sbatch for the GPU make sure you have enough time for your GPU to run

Automatically creates a GPU batch, and runs training
sbatch is hands Savio a set of instructions to run on its own. You can close your laptop, and Savio keeps working.
```bash
sbatch \
  --job-name=chemprop \
  --account=ic_cdss170fall \
  --partition=savio4_gpu \
  --qos=a5k_gpu4_ica \
  --gres=gpu:A5000:1 \
  --cpus-per-task=4 \
  --time=08:00:00 \
  --output=/global/home/users/maryanle/drug_design_clean/chemprop-%j.out \
  --chdir=/global/home/users/maryanle/drug_design_clean \
  --wrap='bash -lc "conda activate drugdesign && python train_models.py --message-hidden-dim 4096 --ffn-hidden-dim 4096 --depth 9 --ffn-num-layers 4 --epochs 9 --output-dir \$PWD/models/trained/wrapped_run"'


  sbatch \
  --job-name=chemprop \
  --account=ic_cdss170fall \
  --partition=savio4_gpu \
  --qos=a5k_gpu4_ica \
  --gres=gpu:A5000:1 \
  --cpus-per-task=4 \
  --time=08:00:00 \
  --output=/global/home/users/maryanle/drug_design_clean/chemprop-%j.out \
  --chdir=/global/home/users/maryanle/drug_design_clean \
  --wrap='bash -lc "conda activate drugdesign && python train_models.py --message-hidden-dim 4096 --ffn-hidden-dim 4096 --depth 9 --ffn-num-layers 4 --epochs 9 --output-dir \$PWD/models/trained/wrapped_run"'
```

Double check if you have a gpu request in queue and or check if it's still running
```bash
squeue -u maryanle
squeue -j 40413122
grep -E 'message_hidden_dim|ffn_hidden_dim|depth|ffn_num_layers' \
  /global/home/users/maryanle/drug_design_clean/models/trained/wrapped_run/jak2/config.toml
```

To cancel a queue, replace JOBID with one that is given
```bash
scancel JOBID
```

Check if the job has finished, replace JOBID with the one they gave you
```bash
tail -80 /global/home/users/maryanle/drug_design_clean/chemprop-JOBID.out
tail -80 /global/home/users/maryanle/drug_design_clean/chemprop-40413122.out
tail -f chemprop-JOBID.out
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

based on david's recommendations, DOING THIS START IS 12:33PM
```bash
python train_models.py \
  --message-hidden-dim 4096 \
  --ffn-hidden-dim 4096 \
  --depth 9 \
  --ffn-num-layers 4 \
  --output-dir "$PWD/models/trained/retry_20261023"
```



Training writes target-specific CSVs under `models/training_data/` and checkpoints under
`models/trained/jak2/` and `models/trained/lck/`. Generated model files are not committed.
After training, point the agent at those directories:

```bash
export JAK2_MODEL_PATH="$PWD/models/trained/jak2"
export LCK_MODEL_PATH="$PWD/models/trained/lck"
```

The prediction adapter in `surrogate_models.py` uses the Chemprop 2 command-line API and
predicts a whole candidate batch at once. Keep the training and agent runs in the same
Chemprop environment. Check that both directories exist before launching the agent.

## 5. Run a safe demo (no API calls, no docking)

```bash
python agent.py --demo --iterations 2
```

This checks the molecule validation, scoring, ranking, and CSV history using a few example SMILES. It does not call the LLM, Chemprop, or DOCKSTRING.

## 6. Run with the LLM and trained surrogate models

After training the checkpoints and setting the paths above:

```bash
python agent.py --iterations 2 --candidates 20 --dock-top 2
```

Docking is disabled by default. To enable it, configure a DOCKSTRING adapter in `docking.py`, verify it on one known molecule, then pass `--run-docking`.

## Files

- `agent.py`: simple main loop
- `llm_proposer.py`: asks the LLM for SMILES
- `molecule_tools.py`: RDKit validation and properties
- `surrogate_models.py`: adapters for your trained Chemprop models
- `scoring.py`: hard filters and ranking
- `docking.py`: optional DOCKSTRING adapter placeholder
- `memory.py`: CSV experiment history and feedback
- `config.py`: settings and environment variables
- `data/experiment_history.csv`: generated history
- `tests/test_core.py`: basic unit tests

## Why some integrations are adapters

The meeting notes specify Chemprop and DOCKSTRING, but do not specify the actual checkpoint paths, exact Chemprop version/API, DOCKSTRING installation/API, or the precise dataset schema. Those interfaces vary by installation, so the starter leaves these boundaries explicit rather than silently pretending they work. The main workflow and tests are runnable without them.
