import sys

def print_env_info(context):
    print(f"\n--- {context} ---")
    print("Python executable:", sys.executable)
    try:
        import catboost
        print("CatBoost version:", catboost.__version__)
    except Exception as e:
        print("CatBoost import error:", e)
    try:
        import lightgbm
        print("LightGBM version:", lightgbm.__version__)
    except Exception as e:
        print("LightGBM import error:", e)
    try:
        import xgboost
        print("XGBoost version:", xgboost.__version__)
    except Exception as e:
        print("XGBoost import error:", e)
    print("-------------------\n")

# Print in main process
print_env_info("Main Process")

# Now test in Ray worker
try:
    import ray
    ray.init(ignore_reinit_error=True)
    @ray.remote
    def worker_env_info():
        import sys
        def print_env_info_worker(context):
            print(f"\n--- {context} ---")
            print("Python executable:", sys.executable)
            try:
                import catboost
                print("CatBoost version:", catboost.__version__)
            except Exception as e:
                print("CatBoost import error:", e)
            try:
                import lightgbm
                print("LightGBM version:", lightgbm.__version__)
            except Exception as e:
                print("LightGBM import error:", e)
            try:
                import xgboost
                print("XGBoost version:", xgboost.__version__)
            except Exception as e:
                print("XGBoost import error:", e)
            print("-------------------\n")
        print_env_info_worker("Ray Worker")
        return True
    ray.get(worker_env_info.remote())
    ray.shutdown()
except ImportError:
    print("Ray is not installed. Please install Ray with 'pip install ray[tune]'.")

print("If you see all versions printed above in both Main Process and Ray Worker, your environment is set up correctly.")
