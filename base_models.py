import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score
from bayes_opt import BayesianOptimization
import pickle
import os
from concurrent.futures import ThreadPoolExecutor, as_completed


class ModelPipeline:
    def __init__(self):
        self.models = {}
        os.makedirs("models", exist_ok=True)

    def run(self, X, y, parallel=False, prefix=None):
        """
        Run all base models with hyperparameter optimization.
        If parallel is True, optimize models in parallel.
        """
        self.X = X
        self.y = y
        self.prefix = prefix
        self.models = {}
        tasks = {
            "Random Forest": (
                RandomForestRegressor,
                {
                    "param_bounds": {
                        "n_estimators": (5, 25),
                        "max_depth": (2, 5)
                    },
                    "param_transformers": {
                        "n_estimators": int,
                        "max_depth": int
                    },
                    "model_name": "random_forest.pkl",
                    "random_state": 0,
                },
            ),
            "XGBoost": (
                XGBRegressor,
                {
                    "param_bounds": {
                        "n_estimators": (5, 25),
                        "max_depth": (2, 5),
                        "learning_rate": (0.01, 0.3),
                    },
                    "param_transformers": {
                        "n_estimators": int,
                        "max_depth": int,
                    },
                    "model_name": "xgboost.pkl",
                    "random_state": 0,
                },
            ),
            "SVR": (
                SVR,
                {
                    "param_bounds": {
                        "C": (0.1, 10),
                        "epsilon": (0.01, 0.5),
                    },
                    "kernel": "linear",  # Just for speed
                    "model_name": "svr.pkl"
                },
            ),
        }

        if parallel:
            print("Running models in parallel...")
            with ThreadPoolExecutor() as executor:
                future_to_model = {
                    executor.submit(self.optimize_model, model_class, **params): model_name
                    for model_name, (model_class, params) in tasks.items()
                }
                for future in as_completed(future_to_model):
                    model_name = future_to_model[future]
                    try:
                        self.models[model_name] = future.result()
                        print(f"{model_name} completed.")
                    except Exception as e:
                        print(f"Error training {model_name}: {e}")
        else:
            print("Running models sequentially...")
            for model_name, (model_class, params) in tasks.items():
                print(f"Running {model_name}...")
                self.models[model_name] = self.optimize_model(model_class, **params)

        return self.models


    def optimize_model(self, model_class, param_bounds, param_transformers=None, model_name=None, prefix=None, **fixed_params):
        """
        Optimize a model using Bayesian Optimization.

        Parameters:
        - model_class: The model class to be instantiated.
        - param_bounds: Dictionary of parameter bounds for optimization.
        - param_transformers: Dictionary of transformations for specific parameters (e.g., int casting).
        - model_name: Name of the file to save the model.
        - fixed_params: Additional fixed parameters to pass to the model.
        """
        param_transformers = param_transformers or {}

        def eval_function(**params):
            # Apply transformations to parameters
            for param, transform in param_transformers.items():
                params[param] = transform(params[param])

            model = model_class(**params, **fixed_params)
            scores = cross_val_score(model, self.X, self.y, cv=3, scoring="neg_mean_squared_error")
            return scores.mean()

        optimizer = BayesianOptimization(
            f=eval_function,
            pbounds=param_bounds,
            random_state=0,
        )
        optimizer.maximize(init_points=5, n_iter=15)

        # Get the best parameters and apply transformations
        best_params = optimizer.max["params"]
        for param, transform in param_transformers.items():
            best_params[param] = transform(best_params[param])

        # Train the model with the best parameters
        model = model_class(**best_params, **fixed_params)
        model.fit(self.X, self.y)

        # Save the trained model
        if model_name:
            self._save_model(model, model_name, prefix=self.prefix)

        return model

    def _save_model(self, model, filename, prefix):
        """
        Save the trained model to a file.
        """
        with open(f"models/{prefix}_{filename}", "wb") as f:
            pickle.dump(model, f)