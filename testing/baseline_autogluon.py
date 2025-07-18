import argparse
import pandas as pd
import time
import os
import psutil
from autogluon.tabular import TabularPredictor
from autogluon.common import space as agspace
from collections import defaultdict
import numpy as np
import sys

# --- CatBoost availability check ---
catboost_available = False
try:
    import catboost
    catboost_available = True
except ImportError:
    catboost_available = False

try:
    import openml
except ImportError:
    openml = None

# =========================
# Version/Dependency Checks
# =========================
def check_autogluon_version():
    import autogluon.tabular
    version = autogluon.tabular.__version__
    major, minor, *_ = map(int, version.split('.'))
    if major < 1 or (major == 1 and minor < 0):
        print(f"[ERROR] AutoGluon version {version} is too old. Please upgrade to >=1.0.0.")
        sys.exit(1)
    print(f"[INFO] Using AutoGluon version {version}")

try:
    import ray
except ImportError:
    print("[WARNING] Ray is not installed. HPO will run sequentially and may be slow. Install with 'pip install ray[tune]'.")

# =========================
# ELO Ranking Utilities
# =========================
def elo_score(results, k=32):
    sorted_models = sorted(results.items(), key=lambda x: x[1], reverse=True)
    elos = {name: 1500 for name in results}
    for i in range(len(sorted_models)):
        for j in range(i+1, len(sorted_models)):
            a, a_score = sorted_models[i]
            b, b_score = sorted_models[j]
            expected_a = 1 / (1 + 10 ** ((elos[b] - elos[a]) / 400))
            expected_b = 1 / (1 + 10 ** ((elos[a] - elos[b]) / 400))
            if a_score > b_score:
                elos[a] += k * (1 - expected_a)
                elos[b] += k * (0 - expected_b)
            elif a_score < b_score:
                elos[a] += k * (0 - expected_a)
                elos[b] += k * (1 - expected_b)
            else:
                elos[a] += k * (0.5 - expected_a)
                elos[b] += k * (0.5 - expected_b)
    return elos

# =========================
# CPU Usage Tracker
# =========================
class CPUTracker:
    def __init__(self):
        self.usage = []
        self.times = []
        self.process = psutil.Process(os.getpid())
        self.start_time = time.time()
        self.running = False

    def start(self):
        self.running = True
        self._track()

    def _track(self):
        if not self.running:
            return
        self.usage.append(self.process.cpu_percent(interval=0.1))
        self.times.append(time.time() - self.start_time)
        if self.running:
            import threading
            threading.Timer(0.5, self._track).start()

    def stop(self):
        self.running = False

    def summary(self):
        return pd.DataFrame({'time': self.times, 'cpu_percent': self.usage})

# =========================
# HPO Configs (Single Dict per Model)
# =========================
def get_hpo_configs(disable_catboost=False):
    configs = {
        'GBM': {
            'learning_rate': agspace.Real(0.001, 0.2, log=True),
            'num_leaves': agspace.Int(16, 128),
            'feature_fraction': agspace.Real(0.5, 1.0),
            'bagging_fraction': agspace.Real(0.5, 1.0),
            'min_data_in_leaf': agspace.Int(5, 50),
            'num_boost_round': agspace.Int(500, 3000),
            'early_stopping_rounds': 50,
            'extra_trees': agspace.Categorical(True, False),
            'lambda_l1': agspace.Real(0, 5),
            'lambda_l2': agspace.Real(0, 5),
            'max_bin': agspace.Int(128, 512),
        },
        'CAT': {
            'iterations': agspace.Int(500, 3000),
            'learning_rate': agspace.Real(0.001, 0.2, log=True),
            'depth': agspace.Int(4, 10),
            'l2_leaf_reg': agspace.Real(1, 10),
            'bootstrap_type': agspace.Categorical('Bayesian', 'Bernoulli', 'MVS'),
            'od_wait': 50,
            'random_strength': agspace.Real(0, 2),
            'bagging_temperature': agspace.Real(0, 1),
        },
        'XGB': {
            'n_estimators': agspace.Int(500, 3000),
            'learning_rate': agspace.Real(0.001, 0.2, log=True),
            'max_depth': agspace.Int(3, 10),
            'subsample': agspace.Real(0.5, 1.0),
            'colsample_bytree': agspace.Real(0.5, 1.0),
            'gamma': agspace.Real(0, 5),
            'reg_alpha': agspace.Real(0, 2),
            'reg_lambda': agspace.Real(0, 2),
            'early_stopping_rounds': 50,
            'min_child_weight': agspace.Int(1, 10),
            'max_delta_step': agspace.Int(0, 10),
        },
        'NN_TORCH': {
            'num_epochs': agspace.Int(50, 200),
            'learning_rate': agspace.Real(0.0001, 0.01, log=True),
            'activation': agspace.Categorical('relu', 'tanh'),
            'hidden_size': agspace.Int(64, 512),
            'dropout_prob': agspace.Real(0.0, 0.5),
            'weight_decay': agspace.Real(0, 0.1),
        }
    }
    if disable_catboost or not catboost_available:
        configs.pop('CAT', None)
    return configs

# =========================
# OpenML Dataset Loader
# =========================
def load_openml_dataset(openml_id=40536):
    if openml is None:
        raise ImportError("openml package is not installed. Install with 'pip install openml'.")
    print(f"Downloading OpenML dataset {openml_id}...")
    d = openml.datasets.get_dataset(openml_id)
    df, y, _, _ = d.get_data(target=d.default_target_attribute)
    df[d.default_target_attribute] = y
    print(f"Loaded OpenML dataset '{d.name}' with shape {df.shape}")
    return df, d.default_target_attribute

# =========================
# Baseline Model Runner
# =========================
def run_baseline(
    train_data: pd.DataFrame,
    label: str,
    test_data: pd.DataFrame = None,
    time_limit: int = 1200,
    presets: str = 'best_quality',
    output_dir: str = 'AutogluonModels',
    problem_type: str = None,
    num_bag_folds: int = 8,
    num_stack_levels: int = 1,
    ensemble_strategy: str = 'bag',
    refit_full: bool = False,
    posthoc_ensemble: bool = True,
    hpo_trials: int = 20,
    disable_catboost: bool = False
):
    check_autogluon_version()
    if disable_catboost:
        print("[INFO] CatBoost is disabled by user flag.")
    elif not catboost_available:
        print("[WARNING] CatBoost is not available. It will be skipped.")
    hpo_configs = get_hpo_configs(disable_catboost=disable_catboost)
    cpu_tracker = CPUTracker()
    print(f"Starting training with detailed HPO (preset: {presets}, time_limit: {time_limit}s, folds: {num_bag_folds}, ensemble: {ensemble_strategy}, HPO trials: {hpo_trials})...")
    cpu_tracker.start()
    predictor = TabularPredictor(
        label=label,
        path=output_dir,
        problem_type=problem_type
    )
    fit_args = dict(
        train_data=train_data,
        time_limit=time_limit,
        presets=presets,
        hyperparameters=hpo_configs,
        num_bag_folds=num_bag_folds,
        num_bag_sets=1,
        num_stack_levels=num_stack_levels,
        ag_args_fit={'num_gpus': 0},
        hyperparameter_tune_kwargs={
            'num_trials': hpo_trials,
            'scheduler': 'local',
            'searcher': 'random',
        }
    )
    if ensemble_strategy == 'stack':
        fit_args['num_stack_levels'] = max(1, num_stack_levels)
    elif ensemble_strategy == 'bag':
        fit_args['num_stack_levels'] = 1
    else:
        print(f"Unknown ensemble strategy: {ensemble_strategy}, defaulting to bagging.")
        fit_args['num_stack_levels'] = 1

    # Control which models to include
    all_hpo_configs = get_hpo_configs(disable_catboost=disable_catboost)
    model_list = ['GBM', 'XGB', 'NN_TORCH', 'RF']
    if not disable_catboost and catboost_available:
        model_list.append('CAT')
    else:
        if not disable_catboost:
            print("[WARNING] CatBoost is not available and will not be used.")
    # Only include selected models in hyperparameters
    hyperparameters = {k: v for k, v in all_hpo_configs.items() if k in model_list}

    # === MODEL TRAINING HAPPENS HERE ===
    try:
        predictor.fit(
            train_data=train_data,
            time_limit=time_limit,
            presets=presets,
            hyperparameters=hyperparameters,
            num_bag_folds=num_bag_folds,
            num_bag_sets=1,
            num_stack_levels=num_stack_levels,
            ag_args_fit={'num_gpus': 0},
            hyperparameter_tune_kwargs={
                'num_trials': hpo_trials,
                'scheduler': 'local',
                'searcher': 'random',
            }
        )
    except Exception as e:
        print(f"[ERROR] AutoGluon fit failed: {e}")
        cpu_tracker.stop()
        return
    cpu_tracker.stop()
    print("Training complete. Leaderboard:")
    leaderboard = predictor.leaderboard(silent=True)
    print(leaderboard)

    if refit_full:
        print("Refitting all models on train+val (no CV ensemble) as per foundation model strategy...")
        predictor.refit_full()
        leaderboard = predictor.leaderboard(silent=True)
        print("Leaderboard after refit_full:")
        print(leaderboard)

    if posthoc_ensemble:
        print("Performing post-hoc ensembling of top HPO configurations...")
        predictor.fit_extra()
        leaderboard = predictor.leaderboard(silent=True)
        print("Leaderboard after post-hoc ensembling:")
        print(leaderboard)

    model_scores = {}
    train_val_time = leaderboard['fit_time'].max() if 'fit_time' in leaderboard else None
    n_train = len(train_data)
    median_time_per_1k = (train_val_time / n_train * 1000) if train_val_time and n_train else None
    inference_times = []
    if test_data is not None:
        print("\nEvaluating on test data:")
        for model in leaderboard['model'].values:
            start_inf = time.time()
            y_pred = predictor.predict(test_data.drop(columns=[label]), model=model)
            inf_time = time.time() - start_inf
            inference_times.append(inf_time)
            y_true = test_data[label]
            if predictor.problem_type in ['regression', 'quantile']:
                score = np.corrcoef(y_true, y_pred)[0, 1]
            else:
                score = (y_true == y_pred).mean()
            model_scores[model] = score
            print(f"Model: {model}, Score: {score:.4f}, Inference time: {inf_time:.4f}s")
    else:
        print("No test data provided. Skipping evaluation.")
        for model in leaderboard['model'].values:
            model_scores[model] = leaderboard.loc[leaderboard['model'] == model, 'score_val'].values[0]

    elos = elo_score(model_scores)
    print("\nELO Rankings:")
    for model, elo in sorted(elos.items(), key=lambda x: x[1], reverse=True):
        print(f"{model}: {elo:.1f}")

    cpu_df = cpu_tracker.summary()
    print("\nCPU Usage Over Time (first 10 rows):")
    print(cpu_df.head(10))
    cpu_df.to_csv(os.path.join(output_dir, 'cpu_usage.csv'), index=False)
    print(f"CPU usage log saved to {os.path.join(output_dir, 'cpu_usage.csv')}")

    print("\n==== Summary Metrics ====")
    print(f"Train+val time: {train_val_time:.2f} seconds")
    print(f"Median time per 1K samples: {median_time_per_1k:.4f} seconds")
    if inference_times:
        print(f"Median inference time: {np.median(inference_times):.4f} seconds")
    print("========================\n")

# =========================
# Ray V2 Migration Warnings
# =========================
# If you see RayDeprecationWarning about ray.train.get_context or ray.train.report,
# you can suppress them by setting the environment variable before running:
#   export RAY_TRAIN_ENABLE_V2_MIGRATION_WARNINGS=0
# Or in Python (at the very top, before importing ray):
#   import os; os.environ["RAY_TRAIN_ENABLE_V2_MIGRATION_WARNINGS"] = "0"
# These are only warnings and do not affect script correctness.
# =========================
# Main/Test Mode
# =========================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run baseline models (XGBoost, CatBoost, LightGBM, MLP) with AutoGluon, advanced HPO, CPU tracking, ELO ranking, and ensemble strategies.")
    parser.add_argument('--train', type=str, default=None, help='Path to training CSV file')
    parser.add_argument('--test', type=str, default=None, help='Path to test CSV file (optional)')
    parser.add_argument('--label', type=str, default=None, help='Name of the label column')
    parser.add_argument('--time_limit', type=int, default=1200, help='Time limit for training (seconds)')
    parser.add_argument('--output_dir', type=str, default='AutogluonModels', help='Directory to store models/results')
    parser.add_argument('--problem_type', type=str, default=None, help='Type of prediction problem (optional)')
    parser.add_argument('--openml', type=int, default=None, help='OpenML dataset ID to use (default: 40536 for mfeat-zernike)')
    parser.add_argument('--test_mode', action='store_true', help='Run a full test using OpenML dataset 40536 (mfeat-zernike)')
    parser.add_argument('--ensemble_strategy', type=str, default='bag', choices=['bag', 'stack'], help='Ensemble strategy: bagging or stacking')
    parser.add_argument('--num_bag_folds', type=int, default=8, help='Number of folds for bagging (cross-validation)')
    parser.add_argument('--num_stack_levels', type=int, default=1, help='Number of stacking levels')
    parser.add_argument('--refit_full', action='store_true', help='Refit foundation models on train+val (no CV ensemble)')
    parser.add_argument('--no_posthoc_ensemble', action='store_true', help='Disable post-hoc ensembling of HPO configs')
    parser.add_argument('--hpo_trials', type=int, default=20, help='Number of HPO trials per model')
    parser.add_argument('--disable_catboost', action='store_true', help='Disable CatBoost models')
    args = parser.parse_args()

    os.environ["RAY_TRAIN_ENABLE_V2_MIGRATION_WARNINGS"] = "0"

    if args.test_mode or args.openml is not None:
        if openml is None:
            raise ImportError("openml package is not installed. Install with 'pip install openml'.")
        openml_id = args.openml if args.openml is not None else 40536
        df, label = load_openml_dataset(openml_id)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        split = int(0.8 * len(df))
        train_data = df.iloc[:split].copy()
        test_data = df.iloc[split:].copy()
        run_baseline(
            train_data=train_data,
            label=label,
            test_data=test_data,
            time_limit=args.time_limit,
            output_dir=args.output_dir,
            problem_type=args.problem_type,
            num_bag_folds=args.num_bag_folds,
            num_stack_levels=args.num_stack_levels,
            ensemble_strategy=args.ensemble_strategy,
            refit_full=args.refit_full,
            posthoc_ensemble=not args.no_posthoc_ensemble,
            hpo_trials=args.hpo_trials,
            disable_catboost=args.disable_catboost
        )
    elif args.train is not None and args.label is not None:
        train_data = pd.read_csv(args.train)
        test_data = pd.read_csv(args.test) if args.test else None
        run_baseline(
            train_data=train_data,
            label=args.label,
            test_data=test_data,
            time_limit=args.time_limit,
            output_dir=args.output_dir,
            problem_type=args.problem_type,
            num_bag_folds=args.num_bag_folds,
            num_stack_levels=args.num_stack_levels,
            ensemble_strategy=args.ensemble_strategy,
            refit_full=args.refit_full,
            posthoc_ensemble=not args.no_posthoc_ensemble,
            hpo_trials=args.hpo_trials,
            disable_catboost=args.disable_catboost
        )
    else:
        print("Please provide either --train and --label, or use --test_mode or --openml for OpenML dataset.") 