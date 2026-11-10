# Optimal Compression Prediction

## Create the Conda environment

```bash
conda env create -f environment.yml
conda activate optimal-compression-prediction
```

## Add the data

```bash
mkdir -p data
```

Place the prepared features, metrics, and source datasets under `data/` using
the structure described in `data/README.md`.

## Run the complete workflow locally

```bash
./run_local.sh
```

This command uses no PBS jobs. It generates the configs, computes the label
caches, runs every single- and multi-variable CNN and RF config sequentially,
and creates the plots and label tables. Running every model locally can take a
long time.

To run only selected configs, list them after the command. For example, this
runs a paired CNN and RF single-variable experiment and then creates every
available output from that run:

```bash
./run_local.sh configs/rotated_config_1.json configs/rotated_config_48.json
```

Outputs are written under `data/runs/local_<timestamp>/`, and the newest run
path is recorded in `data/latest_run.txt`.

To regenerate plots for the newest completed run without rerunning the models:

```bash
./generate_plots.sh "$(cat data/latest_run.txt)"
```

The PBS alternative for NCAR systems remains:

```bash
./run_pipeline.sh
```
