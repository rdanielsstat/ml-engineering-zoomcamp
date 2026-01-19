# pipeline.py
import pandas as pd
import numpy as np

# define the LOSPipeline class
class LOSPipeline:
    def __init__(self, encoder, scaler, model, features, category_mappings = None):
        self.encoder = encoder
        self.scaler = scaler
        self.model = model
        self.features = features
        self.category_mappings = category_mappings or {}
    
    def preprocess(self, df):
        df = df.copy()
        
        # fill missing values safely
        missing_cols_abortion = ['health_service_area', 'hospital_county', 'operating_certificate_number',
                                 'permanent_facility_id', 'zip_code']
        
        if 'facility_name' in df.columns:
            facility_name_filled = df['facility_name'].fillna('')
            mask = facility_name_filled == "Redacted for Confidentiality"
            df.loc[mask, missing_cols_abortion] = df.loc[mask, missing_cols_abortion].fillna("NAA")
            df.loc[~mask, 'zip_code'] = df.loc[~mask, 'zip_code'].fillna("NAS")
        else:
            df.loc[:, 'zip_code'] = df.loc[:, 'zip_code'].fillna("NAS")
        
        for col in ['payment_typology_2', 'payment_typology_3', 'ccsr_procedure_description']:
            if col in df.columns:
                df[col] = df[col].fillna("None")
        
        for feature, mapping in self.category_mappings.items():
            if feature in df.columns:
                df[feature] = df[feature].replace(mapping)
        
        return df
    
    def predict(self, df):
        df = self.preprocess(df)
        
        for col in self.features:
            if col not in df.columns:
                df[col] = "None"
        
        df_features = df[self.features]
        df_encoded = self.encoder.transform(df_features)
        df_encoded = pd.DataFrame(df_encoded, columns=self.features)
        df_scaled = self.scaler.transform(df_encoded)
        df_scaled = pd.DataFrame(df_scaled, columns=self.features)
        log_pred = self.model.predict(df_scaled)
        return np.exp(log_pred)