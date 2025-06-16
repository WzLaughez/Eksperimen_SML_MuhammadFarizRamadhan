from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from joblib import dump
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
def preprocess_data(data, target_column, save_path, file_path, unnecessary_cols):
    # Konversi TotalCharges
    data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")\

    # Buang kolom tidak perlu
    data = data.drop(columns=unnecessary_cols, errors='ignore')

    # Pisah X, y
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Kolom numerik & kategorikal
    numeric_features = X.select_dtypes(include=['float64', 'int64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()

    # Simpan header awal
    pd.DataFrame(columns=X.columns).to_csv(file_path, index=False)

    # Pipeline numerik
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', MinMaxScaler())
    ])

    # Pipeline kategorik
    categorical_transformer = Pipeline([
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Gabung preprocessor
    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Fit-transform
    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

    # Simpan pipeline
    dump(preprocessor, save_path)

    # Simpan nama kolom hasil transformasi
    encoder = preprocessor.named_transformers_['cat'].named_steps['encoder']
    encoded_cols = encoder.get_feature_names_out(categorical_features)
    full_cols = numeric_features + encoded_cols.tolist()
    pd.DataFrame(columns=full_cols).to_csv("encoded_columns.csv", index=False)

    return X_train, X_test, y_train, y_test, preprocessor


if __name__ == "__main__":
    df = pd.read_csv("Customer-Churn.csv")  # Sesuaikan path ini
    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(
        data=df,
        target_column='Churn',
        save_path='preprocessor.pkl',
        file_path='kolom.csv',
        unnecessary_cols='customerID'
)
    
    # Tambahkan ini:
    if y_train.dtype == 'object':
        y_train = y_train.map({'Yes': 1, 'No': 0})
        y_test = y_test.map({'Yes': 1, 'No': 0})

    if hasattr(X_train, "toarray"):  # jika hasilnya sparse matrix
        X_train = X_train.toarray()
        X_test = X_test.toarray()
    # Ambil nama kolom yang sudah disimpan
    encoder = preprocessor.named_transformers_['cat'].named_steps['encoder']
    encoded_cols = encoder.get_feature_names_out(preprocessor.transformers_[1][2])
    full_cols = preprocessor.transformers_[0][2] + encoded_cols.tolist()

    # Simpan ke file CSV dengan nama kolom asli
    train_df = pd.DataFrame(X_train, columns=full_cols)
    train_df["target"] = y_train.values
    train_df.to_csv("preprocessing/processed.csv", index=False)
