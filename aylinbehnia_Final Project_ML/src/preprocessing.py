# import pandas as pd
# from sklearn.model_selection import train_test_split
# from src.logger import get_logger

# logger = get_logger()
# class Preprocessor:

#     def load_data(self, path):
#         return pd.read_csv(path)

#     def split(self, df, target, test_size, random_state):
#         X = df.drop(target, axis=1)
#         y = df[target]
#         return train_test_split(X, y,
#                                 test_size=test_size,
#                                 random_state=random_state)


# src/preprocessing.py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from src.logger import get_logger
import joblib
import os
from sklearn.preprocessing import MinMaxScaler

logger = get_logger()


class Preprocessor:

    # ✅ df اختیاری شد تا با Pipeline سازگار شود
    def __init__(self, df=None):
        self.df = df.copy() if df is not None else None
        self.scaler = StandardScaler()

    # -----------------------------
    # Basic ML Flow (Pipeline Mode)
    # -----------------------------

    def load_data(self, path):
        logger.info(f"Loading data from {path}")
        return pd.read_csv(path)

    def split(self, df, target, test_size=0.2, random_state=42):
        X = df.drop(target, axis=1)
        y = df[target]
        return train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y   # حرفه‌ای‌تر
        )

    def scale(self, X_train, X_test):
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        os.makedirs("models", exist_ok=True)
        joblib.dump(scaler, "models/scaler.pkl")

        return X_train_scaled, X_test_scaled

    def oversample(self, X_train, y_train):
        sm = SMOTE(random_state=42)
        X_res, y_res = sm.fit_resample(X_train, y_train)
        logger.info(f"Oversampled training data: {X_res.shape}, {y_res.shape}")
        return X_res, y_res

    # -----------------------------
    # Notebook Mode (df-based ops)
    # -----------------------------

    def remove_outliers_iqr(self, column):
        if self.df is None:
            raise ValueError("DataFrame not initialized.")
        Q1 = self.df[column].quantile(0.25)
        Q3 = self.df[column].quantile(0.75)
        IQR = Q3 - Q1
        self.df = self.df[
            (self.df[column] >= Q1 - 1.5 * IQR) &
            (self.df[column] <= Q3 + 1.5 * IQR)
        ]
        return self

    def remove_fake_zeros(self):
        if self.df is None:
            raise ValueError("DataFrame not initialized.")

        cols_with_zero_issue = [
            "Glucose",
            "BloodPressure",
            "SkinThickness",
            "Insulin",
            "BMI"
        ]

        for col in cols_with_zero_issue:
            if col in self.df.columns:
                self.df[col] = self.df[col].replace(0, np.nan)
                self.df[col] = self.df[col].fillna(self.df[col].median())

        return self
    
    
    def remove_outliers_iqr(self, column):
        Q1 = self.df[column].quantile(0.25)
        Q3 = self.df[column].quantile(0.75)
        IQR = Q3 - Q1

        self.df = self.df[
            (self.df[column] >= Q1 - 1.5 * IQR) &
            (self.df[column] <= Q3 + 1.5 * IQR)
        ]
        return self
    

    def remove_duplicates(self):
        if self.df is None:
            raise ValueError("DataFrame not initialized.")
        self.df = self.df.drop_duplicates()
        return self

    def normalize_df(self, target_col):
        if self.df is None:
            raise ValueError("DataFrame not initialized.")

        X = self.df.drop(columns=[target_col])
        y = self.df[target_col]

        X_scaled = self.scaler.fit_transform(X)

        os.makedirs("models", exist_ok=True)
        joblib.dump(self.scaler, "models/scaler.pkl")

        return X_scaled, y

    def class_count(self, target_col):
        if self.df is None:
            raise ValueError("DataFrame not initialized.")
        return self.df[target_col].value_counts()
    
    
    def minmax_scale(self, target):
        X = self.df.drop(target, axis=1)
        y = self.df[target]
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        return X_scaled, y