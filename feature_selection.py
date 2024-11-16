import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE, SelectKBest, f_regression
import pickle
import os

from config import K_BEST_FEATURES, N_RFE_FEATURES

class FeatureSelector:
    def __init__(self):
        """
        Initialize FeatureSelector for performing feature selection.
        """
        self.selected_features = set()

    def combine_selected_features(self):
        """
        Finalize selected features by taking the union of all selected features from the methods.
        """
        print(f"Total selected features: {len(self.selected_features)}")
        return list(self.selected_features)

    def run(self, df, target_column, k_best=K_BEST_FEATURES, n_features_rfe=N_RFE_FEATURES):
        """
        Run all feature selection methods and return the final selected features.
        """
        self.df = df
        self.target_column = target_column
        self.k_best = k_best
        self.n_features_rfe = n_features_rfe
        self.selected_features = set()

        print("Running Tree-Based Feature Importance...")
        self.feature_importance_tree_based()
        print("Running Recursive Feature Elimination (RFE)...")
        self.recursive_feature_elimination()
        print("Running Univariate Feature Selection...")
        self.univariate_feature_selection()

        final_features = self.combine_selected_features()
        print("Feature selection completed.")
        return final_features

    def feature_importance_tree_based(self):
        """
        Select features based on importance scores from a Random Forest model.
        """
        X = self.df.drop(columns=[self.target_column])
        y = self.df[self.target_column]

        rf_model = RandomForestRegressor(random_state=0, n_jobs=-1)
        rf_model.fit(X, y)
        importance = rf_model.feature_importances_

        important_features = X.columns[np.argsort(importance)[-self.k_best:]]
        self.selected_features.update(important_features)

        # Save the model
        self._save_model(rf_model, "rf_feature_importance.pkl")
        print(f"Saved Random Forest model with important features identified.")

    def recursive_feature_elimination(self):
        """
        Perform Recursive Feature Elimination (RFE) using a Random Forest as the estimator.
        """
        X = self.df.drop(columns=[self.target_column])
        y = self.df[self.target_column]

        rf_model = RandomForestRegressor(n_estimators=20, max_depth=5, random_state=0, n_jobs=-1)  # Low estimators just for speed
        rfe = RFE(estimator=rf_model, n_features_to_select=self.n_features_rfe)
        rfe.fit(X, y)

        rfe_features = X.columns[rfe.support_]
        self.selected_features.update(rfe_features)

        # Save the model
        self._save_model(rfe, "rfe_model.pkl")
        print(f"Saved RFE model with selected features.")

    def univariate_feature_selection(self):
        """
        Perform univariate feature selection using f_regression.
        """
        X = self.df.drop(columns=[self.target_column])
        y = self.df[self.target_column]

        skb = SelectKBest(score_func=f_regression, k=self.k_best)
        skb.fit(X, y)

        skb_features = X.columns[skb.get_support()]
        self.selected_features.update(skb_features)

        # Save the selector
        self._save_model(skb, "skb_selector.pkl")
        print(f"Saved SelectKBest selector with selected features.")

    def _save_model(self, model, filename):
        """
        Save a model or transformer to a file in the feature_selectors directory.
        """
        os.makedirs("feature_selectors", exist_ok=True)
        with open(f"feature_selectors/{filename}", "wb") as f:
            pickle.dump(model, f)
