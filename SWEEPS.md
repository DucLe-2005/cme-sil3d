# Bayesian hyperparameter search

Use the training environment with PyTorch, segmentation-models-pytorch,
scikit-learn, NumPy, and wandb installed. Authenticate with `wandb login`.
To launch a sweep directly from Python, go to the bottom of
`train_new_data_backbone.py`, comment out `train()`, and uncomment the sweep
block. It loads `sweep.yaml`, creates a sweep with `wandb.sweep(...)`, and
runs `wandb.agent(sweep_id, function=train)`. This option also requires PyYAML.
Run `python train_new_data_backbone.py`. Each launch creates a new sweep.
To return to a normal run, comment out the sweep block and restore `train()`.

Alternatively, from this directory initialize the YAML sweep:

```bash
wandb sweep sweep.yaml
```

Copy the returned `ENTITY/PROJECT/SWEEP_ID` and start the agent:

```bash
wandb agent ENTITY/PROJECT/SWEEP_ID
```

The agent automatically launches `train_new_data_backbone.py` for each trial.
`sweep.yaml` limits the entire sweep to 20 trials with `run_cap: 20`, with
30 epochs per trial. Edit these settings and the search ranges before
initializing the sweep. To limit one agent to five trials instead:

```bash
wandb agent --count 5 ENTITY/PROJECT/SWEEP_ID
```

For a team or custom project, initialize with:

```bash
wandb sweep --entity YOUR_TEAM --project YOUR_PROJECT sweep.yaml
```

To run on two GPUs, start these in separate terminals from this directory:

```bash
CUDA_VISIBLE_DEVICES=0 wandb agent ENTITY/PROJECT/SWEEP_ID
CUDA_VISIBLE_DEVICES=1 wandb agent ENTITY/PROJECT/SWEEP_ID
```

Agents share the sweep's total trial budget. On multiple machines, use the
same script, Python dependencies, and dataset contents. Set `root_dir` in
`sweep.yaml` to a location available on every machine, or keep each machine's
default dataset location. `${interpreter}` uses Python from the agent's PATH;
activate your training environment before starting it.

For a single training run, edit the `config` dictionary at the top of
`train_new_data_backbone.py`, then run:

```bash
python train_new_data_backbone.py
```

For Bayesian optimization, edit the search ranges and fixed values in
`sweep.yaml`. The agent supplies these overrides through `wandb.init()`;
the training script reads the resulting `run.config`.

The initial search space in both configurations is:

| Parameter | Search range |
| --- | --- |
| Learning rate | 1e-5 to 1e-3, log-uniform |
| Batch size | 8, 16, 32, 40 |
| BCE weight (`alpha_loss`) | 0.1 to 0.9, uniform; Dice weight is 1-alpha |

The objective is to maximize `best_val_iou`, the highest validation IoU
observed during a trial. All trials use the same 80/20 case split and seed.
Every nested sequence and both instruments from each case stay together.
Checkpoints are selected by validation IoU and saved locally under
`checkpoints/<run_id>/`, outside the W&B run directory. Checkpoints are never
uploaded to W&B. The W&B summary records the best
epoch and the exact train/validation case names.

To add more trials to an existing sweep, increase its trial budget in W&B
if needed and start another `wandb agent ENTITY/PROJECT/SWEEP_ID`.
CLI agents launch each trial in a separate process; the Python agent calls `train()` for each trial. Choose batch sizes that fit
your GPU by editing `sweep.yaml` before creating a sweep.

Validation scores guide hyperparameter selection. Use separate held-out cases
if you later need an unbiased final performance estimate.

W&B reference: https://docs.wandb.ai/guides/sweeps/define-sweep-configuration/

## Smaller dataset for faster trials

`max_cases: 10` selects ten simulated-case directories with the fixed seed,
then splits them into eight training and two validation cases. All sequences
and both instruments within a selected case stay together. Selection is
identical across trials when the dataset contents and seed are unchanged.

Edit `config["max_cases"]` in `train_new_data_backbone.py` for both single
runs and sweeps. Set `max_cases` to the number of available case directories to use all cases.
It must not exceed the number of available directories. Existing active trials
are unaffected; create a new sweep to compare all trials on the same subset.

## Separate C2 and C3 models

Set `parameters.instrument.value: c2` in `sweep.yaml` and create a sweep
with `wandb sweep sweep.yaml`. For C3, change that value to `c3`, change
`name` to `unet-c3-bayesian`, and create a second sweep. Each sweep then
optimizes its own instrument's validation IoU and saves independent models.
Keep this parameter fixed rather than searching over instruments, since
C2 and C3 scores measure different tasks. Editing YAML does not alter an
existing sweep; initialize a new sweep for these preprocessing changes.

For a single run, set `config["instrument"]` in
`train_new_data_backbone.py` to `"c2"` or `"c3"`, then run the script.
`"both"` remains available for joint training. Case selection and the 80/20
split remain identical for either instrument when seed and max_cases match.
Weight decay is disabled and is not a search parameter.

Normalization statistics are fitted once from finite, valid observation pixels
in training cases only, separately for C2 and C3. Each pixel has equal weight.
Both training and validation apply `(brightness - training_mean) / training_std`
without clipping. Invalid pixels stay zero. For zero training standard deviation,
a scale of 1 is used, so constant training pixels map to zero.
Validation never fits its own statistics. Mask processing is unchanged.

The mean and population standard deviation are logged in W&B, stored inside each
checkpoint as `normalization_stats`, and saved beside it as
`normalization_stats.json`. Use these exact statistics for inference. Initial
scanning reads the selected training samples once before training begins.
Create new sweeps for this normalization change; existing min/max checkpoints
used different preprocessing.
