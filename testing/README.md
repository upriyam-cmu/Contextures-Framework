# Contextures Baseline Testing Scripts

## baseline_autogluon.py

This script runs a suite of baseline machine learning models (XGBoost, LightGBM, Random Forest, Neural Network, and optionally CatBoost) on a tabular dataset using [AutoGluon](https://auto.gluon.ai/). It supports advanced hyperparameter optimization, ensemble strategies, CPU usage tracking, and ELO ranking of models. You can use your own CSV data or benchmark on OpenML datasets.

### Usage

**Run on an OpenML dataset (default: mfeat-zernike, ID 40536):**
```bash
python baseline_autogluon.py --openml 40536
```

**Run on your own CSV data:**
```bash
python baseline_autogluon.py --train path/to/your_data.csv --label target_column
```

**Disable CatBoost (if you have issues with CatBoost or want to exclude it):**
```bash
python baseline_autogluon.py --openml 40536 --disable_catboost
```

**Other useful options:**
- `--test_mode` : Run a quick test using the default OpenML dataset.
- `--time_limit 600` : Set a custom time limit for training (in seconds).
- `--ensemble_strategy stack` : Use stacking instead of bagging for ensembling.
- `--hpo_trials 10` : Set the number of hyperparameter optimization trials per model.

Run `python baseline_autogluon.py --help` for a full list of options.

---

## test_ray.py

This script is a diagnostic tool to check if your Python environment and Ray workers can successfully import key ML libraries (CatBoost, LightGBM, XGBoost). It helps debug environment issues that can cause Ray worker crashes in distributed training.

### Usage

**Run the Ray environment test:**
```bash
python test_ray.py
```

If all imports succeed in both the main process and Ray workers, your environment is correctly set up for distributed training with Ray.

---

## Notes
- Always activate your conda environment before running these scripts:
  ```bash
  conda activate agenv
  ```
- If you encounter errors related to missing libraries in Ray workers, use `test_ray.py` to debug.
- For best results, use Python 3.10+ and compatible versions of AutoGluon, CatBoost, LightGBM, and XGBoost. 