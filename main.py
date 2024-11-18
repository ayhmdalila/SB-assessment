from preprocess import DatasetPipeline
from feature_selection import FeatureSelector
from base_models import ModelPipeline
from config import *

if __name__=='__main__':

    pipeline = DatasetPipeline()

    transactions_dataset = pipeline.run(TRANSACTIONS_DATASET_PATH, dataset_name='transactions', target_col = 'amount', encoding_dict = {'One-Hot': ['transaction_type_en', 'transaction_subtype_en', 'registration_type_en', 'is_freehold_text', 'property_usage_en', 'is_offplan', 'is_freehold', 'property_type_en', 'property_subtype_en'],
                                                                                                                                                            'Ordinal': ['rooms_en']}, 
                                                                                                                                                            numerical_cols=['transaction_size_sqm', 'property_size_sqm',  'total_buyer', 'total_seller', 'building_age'])


    rents_dataset = pipeline.run(RENTS_DATASET_PATH, dataset_name='rents', target_col = 'annual_amount', encoding_dict = {'One-Hot': ['is_freehold_text', 'property_type_en', 'property_subtype_en', 'property_usage_en'],
                                                                                                                                                            'Ordinal': ['area_en', 'master_project_en']}, 
                                                                                                                                                            numerical_cols=['property_size_sqm', 'total_properties', 'rooms'])


    fs = FeatureSelector()

    transactions_features = fs.run(transactions_dataset, 'amount')
    rents_features = fs.run(rents_dataset, 'annual_amount')

    transactions_X, transactions_y = transactions_dataset[transactions_features], transactions_dataset['amount']
    rents_X, rents_y = rents_dataset[rents_features], rents_dataset['annual_amount']
    model_pipeline = ModelPipeline()

    transactions_base_models = model_pipeline.run(transactions_X, transactions_y, parallel = BASE_MODEL_PARALLEL_TRAINING, prefix = 'transactions')
    rents_base_models = model_pipeline.run(rents_X, rents_y, parallel = BASE_MODEL_PARALLEL_TRAINING, prefix = 'rents')
