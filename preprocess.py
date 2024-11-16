import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
import pickle as pkl
import os


class DatasetPipeline:
    def __init__(self):
        self.preprocessor = None  # This will store the combined ColumnTransformer

    def run(self, path, dataset_name, target_col, encoding_dict, numerical_cols):
        self.dataset_name = dataset_name
        self.target_col = target_col

        # Load the dataset
        df = self.load_data(path)

        # Drop duplicates and rows with excessive missing data
        df = self.drop_duplicates(df)
        df = self.remove_missing_data(df)

        # Identify column groups dynamically
        categorical_columns = set(encoding_dict.get("One-Hot", []) + encoding_dict.get("Ordinal", []))
        numerical_cols = numerical_cols or list(df.select_dtypes(include=["float64", "int64"]).columns.difference([target_col]))

        # Build the preprocessing pipeline
        self.create_preprocessing_pipeline(categorical_columns, encoding_dict, numerical_cols)

        # Separate features and target
        X = df.drop(columns=[target_col])
        y = df[target_col]

        # Apply transformations
        X_transformed = self.preprocessor.fit_transform(X)

        # Save the preprocessor
        os.makedirs('pipelines', exist_ok=True)
        with open(f'pipelines/{self.dataset_name}_pipeline.pkl', 'wb') as f:
            pkl.dump(self.preprocessor, f, pkl.HIGHEST_PROTOCOL)

        # Return the transformed features and target
        df_transformed = pd.DataFrame(X_transformed)
        df_transformed[target_col] = y

        return df_transformed

    def load_data(self, path):
        return pd.read_csv(path, engine='pyarrow')

    def drop_duplicates(self, df):
        """
        Removes duplicate rows from the DataFrame.
        """
        return df.drop_duplicates()

    def remove_missing_data(self, df, threshold=0.2):
        """
        Removes rows with a high proportion of missing values.
        """
        return df.dropna(thresh=int(threshold * df.shape[1]))

    def create_preprocessing_pipeline(self, categorical_columns, encoding_dict, numerical_cols):
        """
        Create a combined ColumnTransformer for preprocessing.
        """
        # Preprocessing for numerical features
        num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

        # Preprocessing for one-hot encoded categorical features
        one_hot_columns = encoding_dict.get("One-Hot", [])
        one_hot_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(sparse_output=False, handle_unknown="ignore"))
        ])

        # Preprocessing for ordinal encoded categorical features
        ordinal_columns = encoding_dict.get("Ordinal", [])
        ordinal_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
        ])

        # Combine all preprocessing pipelines
        transformers = []
        if numerical_cols:
            transformers.append(("num", num_pipeline, numerical_cols))
        if one_hot_columns:
            transformers.append(("onehot", one_hot_pipeline, one_hot_columns))
        if ordinal_columns:
            transformers.append(("ordinal", ordinal_pipeline, ordinal_columns))

        self.preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
