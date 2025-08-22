import pandas as pd
import dill
import os
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression

def load_data(path):
    return pd.read_csv(path)

def drop_cols(X):
    return X.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1)

def clean_target(df):
    """Drop rows where Survived is missing"""
    return df.dropna(subset=['Survived'])

def features_target(df):
    X = df.drop('Survived', axis=1)
    y = df['Survived']
    return X, y

def build_preprocessor(X):
    num_feats = X.select_dtypes(exclude='object').columns
    cat_feats = X.select_dtypes('object').columns

    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('OH-encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    transformer = ColumnTransformer([
        ('num_pipeline', num_pipeline, num_feats),
        ('cat_pipeline', cat_pipeline, cat_feats)
    ])

    return transformer


def save_pkl(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        dill.dump(obj, f)

def load_pkl(path):
    with open(path, 'rb') as f:
        return dill.load(f)
    

def train_model(X, y):
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    return model

def predict(model, X_test):
    return model.predict(X_test)


def training_pipeline(train_path):
    df = load_data(train_path)

    df = clean_target(df)

    X, y = features_target(df)
    X = drop_cols(X)

    preprocessor = build_preprocessor(X)
    X_transformed = preprocessor.fit_transform(X, y)

    save_pkl(preprocessor, "app/preprocessor.pkl")

    model = train_model(X_transformed, y)

    save_pkl(model, "app/model.pkl")

    print("Training complete. Preprocessor & model saved.")


def test_pipeline(test_path):
    df_test = load_data(test_path)
    X_test = drop_cols(df_test)

    preprocessor = load_pkl("app/preprocessor.pkl")
    model = load_pkl("app/model.pkl")

    X_test_transformed = preprocessor.transform(X_test)

    predictions = predict(model, X_test_transformed)
    
    resp = input("Do you want predictions to be displayed? y/n: ")
    if resp=='y':
        return f'Predictions : {predictions}'
    
    return 'Testing completed.'
    

training_pipeline("Titanic_train.csv")

print(test_pipeline("Titanic_test.csv"))
