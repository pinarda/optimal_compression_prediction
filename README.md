# Optimal Compression Prediction

Place the prepared features, metrics, source data, and precomputed labels under
`data/`.

## Create the data directory

```bash
mkdir -p data
```

## Run the modeling workflow

```bash
./run_pipeline.sh
```

The path for the submitted run is written to `data/latest_run.txt`.

## Generate the plots after modeling finishes

```bash
./generate_plots.sh "$(cat data/latest_run.txt)"
```
