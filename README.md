# SEDRAS-Cleaning

This repository has two root scripts for running the SEDRAS workflow:

- `paper_data_creation.py` generates datasets from sweep settings.
- `main_evaluation.py` discovers/evaluates generated datasets.

## 1) Generate paper datasets

Run:

```bash
python paper_data_creation.py SEDRAS_2026
```

This loads the config instance:

- `configs/instances/SEDRAS_2026PaperDataCreation.py`

### How to steer generation

Edit `SEDRAS_2026_PAPER_DATA_CREATION_CONFIG` to control generation behavior, e.g.:

- output location (`save_folder`)
- sweep size (`udd_complexities`, `num_samples`)
- experiment setup (`representation_levels`, `dynamics`)
- model + search settings (`llm_model`, `max_genetic_search_configurations`, `genetic_search_max_iterations`)
- parallelism (`number_of_parallel_workers`)

## 2) Run main evaluation

Run:

```bash
python main_evaluation.py SEDRAS_2026
```

This loads the config instance:

- `configs/instances/SEDRAS_2026MainEvaluation.py`

### How to steer evaluation

Edit `SEDRAS_2026_MAIN_EVALUATION_CONFIG` to control evaluation behavior, e.g.:

- dataset source (`dataset_root`)
- filtering targets (`target_representation_levels`, `target_dynamics_levels`, `target_complexity_levels`)
- model under test (`target_llm_model`)
- runtime parallelism (`max_workers`)

## Config entry points

The root scripts map CLI names to config instances:

- `paper_data_creation.py` supports `SEDRAS_2026`
- `main_evaluation.py` supports `SEDRAS_2026` (and `Base` for baseline main evaluation config)

If you add a new config instance, update each script's `AVAILABLE_CONFIGS` mapping.
