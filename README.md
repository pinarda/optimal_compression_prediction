# Optimal Compression Prediction

Place the prepared features, metrics, source data, and precomputed labels under
`data/`.

## Create the data directory

```bash
mkdir -p data
```

## Run modeling locally without PBS

```bash
./run_local.sh configs/rotated_config_1.json configs/rotated_config_48.json
```

Pass one or more config files. The example runs the paired CNN and RF
single-variable configs directly with Python. To submit the complete workflow
through PBS instead, run `./run_pipeline.sh`.

The most recent local or PBS run path is written to `data/latest_run.txt`.

## Generate the plots after modeling finishes

```bash
./generate_plots.sh "$(cat data/latest_run.txt)"
```
