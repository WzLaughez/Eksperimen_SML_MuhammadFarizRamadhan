from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from joblib import dump
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import pandas as pd

def preprocess_data(data, target_column, save_path, file_path, unnecessary_cols=None):

    # Membuang kolom yang tidak diperlukan
    if unnecessary_cols:
        data = data.drop(columns=unnecessary_cols, errors='ignore')
    # Coba ubah kolom object ke float jika memungkinkan (untuk kasus seperti 'TotalCharges')
    for col in data.select_dtypes(include=['object']).columns:
        # Jangan ubah kolom target
        if col == target_column:
            continue
        try:
            data[col] = pd.to_numeric(data[col])
        except ValueError:
            pass  # Tetap biarkan kolom ini sebagai object jika gagal konversi
        
    # Menentukan fitur numerik dan kategoris
    numeric_features = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
    categorical_features = data.select_dtypes(include=['object']).columns.tolist()
    column_names = data.columns

    # Membuat DataFrame kosong dengan nama kolom
    df_header = pd.DataFrame(columns=column_names)

    # Menyimpan nama kolom sebagai header tanpa data
    df_header.to_csv(file_path, index=False)
    print(f"Nama kolom berhasil disimpan ke: {file_path}")

    # Pipeline untuk fitur numerik
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', MinMaxScaler())
    ])

    # Pipeline untuk fitur kategoris
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Column Transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    # Memisahkan target
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Membagi data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Fitting dan transformasi data pada training set
    X_train = preprocessor.fit_transform(X_train)
    # Transformasi data pada testing set
    X_test = preprocessor.transform(X_test)
    # Simpan pipeline
    dump(preprocessor, save_path)

    return X_train, X_test, y_train, y_test