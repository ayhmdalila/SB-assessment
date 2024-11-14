import pandas as pd
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
import pickle as pkl

class DatasetPipeline:
    def __init__():
        pass

    def run(self, path, usable_columns, encoding_dict):
        df = self.load_data(path, columns=usable_columns)
        df = self.drop_duplicates(df)
        df = self.handle_missing_data(df)
        df = self.encode_categorical_columns(df, encoding_dict=encoding_dict)
        df = self.scale_numerical_columns(df)
        return df

    def load_data(self, path, columns):
        return pd.read_csv(path, engine='pyarrow')[columns]

    def drop_duplicates(self, df):
        return df.drop_duplicates()
    
    def handle_missing_data(self, df, threshold = 0.2):
        df = df.dropna(thresh=int(threshold * df.shape[1]))

        # Separate columns by data type
        num_cols = df.select_dtypes(include=['float64', 'int64']).columns
        cat_cols = df.select_dtypes(include=['object']).columns

        # Impute numerical columns with median
        num_imputer = SimpleImputer(strategy='median')
        df[num_cols] = num_imputer.fit_transform(df[num_cols])

        # Impute categorical columns with the most frequent category
        cat_imputer = SimpleImputer(strategy='most_frequent')
        df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])

        with open('num_imputer.pkl', 'wb') as f:
            pkl.dump(num_imputer, f, pkl.HIGHEST_PROTOCOL)

        with open('cat_imputer.pkl', 'wb') as f:
            pkl.dump(cat_imputer, f, pkl.HIGHEST_PROTOCOL)

        return df
    
    def encode_categorical_columns(self, df, encoding_dict):
        # One encoder for all One-Hot columns, treating each column independently
        one_hot_columns = encoding_dict.get("One-Hot", [])
        if one_hot_columns:
            one_hot_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
            transformed_data = one_hot_encoder.fit_transform(df[one_hot_columns])

            # Create a DataFrame for the one-hot encoded columns
            one_hot_df = pd.DataFrame(transformed_data, columns=one_hot_encoder.get_feature_names_out(one_hot_columns))
            df.drop(one_hot_columns, axis=1, inplace=True)  # Drop the original columns
            df = pd.concat([df, one_hot_df], axis=1)  # Concatenate the one-hot encoded columns

            # Save encoder
            with open(f"encoders/one_hot_encoder.pkl", "wb") as f:
                pkl.dump(one_hot_encoder, f)
            print(f"Saved One-Hot Encoder for columns {one_hot_columns} as one_hot_encoder.pkl")

        # One encoder for all Ordinal columns, treating each column independently
        ordinal_columns = encoding_dict.get("Ordinal", [])
        if ordinal_columns:
            ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
            df[ordinal_columns] = ordinal_encoder.fit_transform(df[ordinal_columns])

            # Save encoder
            with open(f"encoders/ordinal_encoder.pkl", "wb") as f:
                pkl.dump(ordinal_encoder, f)
            print(f"Saved Ordinal Encoder for columns {ordinal_columns} as ordinal_encoder.pkl")

        return df

    def scale_numerical_columns(self, df):
        # Identify numerical columns
        num_cols = df.select_dtypes(include=['float64', 'int64']).columns
        scaler = StandardScaler()
        df[num_cols] = scaler.fit_transform(df[num_cols])

        with open('scaler.pkl', 'wb') as f:
            pkl.dump(scaler, f, pkl.HIGHEST_PROTOCOL)

        return df