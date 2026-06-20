import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

class Preprocessor:
    def __init__(self):
        self.preprocessor_pipeline = None
        self.numerical_cols = []
        self.categorical_cols = []

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self.numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        self.categorical_cols = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        self.preprocessor_pipeline = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, self.numerical_cols),
                ('cat', categorical_transformer, self.categorical_cols)
            ]
        )

        X_processed = self.preprocessor_pipeline.fit_transform(X)
        
        # Reconstruct DataFrame with feature names
        feature_names = self.get_feature_names()
        return pd.DataFrame(X_processed, columns=feature_names, index=X.index)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.preprocessor_pipeline is None:
            raise ValueError("Preprocessor has not been fitted yet.")
        
        X_processed = self.preprocessor_pipeline.transform(X)
        feature_names = self.get_feature_names()
        return pd.DataFrame(X_processed, columns=feature_names, index=X.index)

    def get_feature_names(self):
        if self.preprocessor_pipeline is None:
            return []
            
        feature_names = []
        if self.numerical_cols:
            feature_names.extend(self.numerical_cols)
        
        if self.categorical_cols:
            # For older sklearn versions it's get_feature_names(), for newer it's get_feature_names_out()
            encoder = self.preprocessor_pipeline.named_transformers_['cat'].named_steps['onehot']
            if hasattr(encoder, 'get_feature_names_out'):
                cat_names = encoder.get_feature_names_out(self.categorical_cols)
            else:
                cat_names = encoder.get_feature_names(self.categorical_cols)
            feature_names.extend(cat_names)
            
        return feature_names
