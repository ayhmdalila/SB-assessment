import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from bayes_opt import BayesianOptimization
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import pickle
import os

class MetaLearner:
    def __init__(self):
        self.meta_model = None

    def prepare_meta_data(self, base_predictions, y):
        """
        Prepare the meta-data for the meta-learner.

        Parameters:
        - base_predictions: Dictionary containing predictions from base models.
        - y: True target values.

        Returns:
        - X_meta: Features for the meta-learner.
        - y_meta: Target values for the meta-learner.
        """
        X_meta = np.column_stack(list(base_predictions.values()))
        y_meta = y
        return X_meta, y_meta

    def optimize_meta_model(self, X, y, param_bounds, n_iter=15, prefix=None):
        """
        Optimize the meta-learner model using Bayesian Optimization.

        Parameters:
        - X: Input features.
        - y: Target values.
        - param_bounds: Dictionary of hyperparameter bounds for optimization.
        - n_iter: Number of iterations for optimization.

        Returns:
        - The optimized meta-learner model.
        """
        self.meta_model = None
        def eval_function(lr, dropout_rate, hidden_units):
            # Build the model with given parameters
            model = self._build_meta_model(
                input_dim=X.shape[1],
                lr=lr,
                dropout_rate=dropout_rate,
                hidden_units=int(hidden_units)
            )
            # Perform cross-validation
            scores = cross_val_score(
                model,
                X,
                y,
                cv=3,
                scoring="neg_mean_squared_error",
                fit_params={
                    "epochs": 50,
                    "batch_size": 32,
                    "verbose": 0,
                    "callbacks": [EarlyStopping(patience=5, restore_best_weights=True)],
                },
            )
            return scores.mean()

        # Perform Bayesian Optimization
        optimizer = BayesianOptimization(
            f=eval_function,
            pbounds=param_bounds,
            random_state=0
        )
        optimizer.maximize(init_points=5, n_iter=n_iter)

        # Get the best parameters
        best_params = optimizer.max["params"]

        # Train the meta-learner with the best parameters
        self.meta_model = self._build_meta_model(
            input_dim=X.shape[1],
            lr=best_params["lr"],
            dropout_rate=best_params["dropout_rate"],
            hidden_units=int(best_params["hidden_units"])
        )
        self.meta_model.fit(
            X, y,
            epochs=100,
            batch_size=32,
            verbose=1,
            callbacks=[EarlyStopping(patience=5, restore_best_weights=True)]
        )

        
        self.meta_model.save_model(f'{prefix}_meta_learner.h5py')
        
        return self.meta_model

    def _build_meta_model(self, input_dim, lr, dropout_rate, hidden_units):
        """
        Build the neural network model for the meta-learner.

        Parameters:
        - input_dim: Number of input features.
        - lr: Learning rate.
        - dropout_rate: Dropout rate for regularization.
        - hidden_units: Number of hidden units in the dense layer.

        Returns:
        - The compiled Keras model.
        """
        model = Sequential([
            Dense(hidden_units, activation="relu", input_dim=input_dim),
            Dropout(dropout_rate),
            Dense(hidden_units, activation="relu"),
            Dropout(dropout_rate),
            Dense(1, activation="linear")
        ])
        model.compile(optimizer=Adam(learning_rate=lr), loss="mse", metrics=["mse"])
        return model

    def _save_model(self, model, filename):
        """
        Save the trained model to a file.
        """
        with open(f"meta_learner/{filename}", "wb") as f:
            pickle.dump(model, f)

    def evaluate_meta_model(self, X, y):
        """
        Evaluate the meta-learner on test data.

        Parameters:
        - X: Test features.
        - y: Test target values.

        Returns:
        - MSE of the meta-learner on the test data.
        """
        if not self.meta_model:
            raise ValueError("Meta-learner has not been trained.")
        y_pred = self.meta_model.predict(X)
        return mean_squared_error(y, y_pred)