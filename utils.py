from sklearn.preprocessing import LabelEncoder
import pandas as pd
import pickle

def encode_features(df):
    df_encoded = df.copy()

    # Wybór kolumn kategorycznych do zakodowania
    categorical_cols = ['locality', 'street', 'property_type', 'city']
    encoder = LabelEncoder()

    for col in categorical_cols:
        if col in df_encoded.columns:
            df_encoded[col] = encoder.fit_transform(df_encoded[col].astype(str))


    return df_encoded

def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)
    