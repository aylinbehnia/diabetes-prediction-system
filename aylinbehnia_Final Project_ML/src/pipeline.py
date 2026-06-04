from src.config import Config
from src.preprocessing import Preprocessor
from src.models import MLModels
from src.evaluation import Evaluator
from src.logger import get_logger

import joblib
import os
from pathlib import Path

from sklearn.model_selection import cross_val_score

logger = get_logger()


class Pipeline:

    def __init__(self):
        self.cfg = Config()
        self.prep = Preprocessor()
        self.models = MLModels()
        self.evaluator = Evaluator()

    def run(self, data_path=None):

        # اگر کاربر مسیر بده → همونو استفاده کن
        # اگر نده → از config استفاده کن
        # اگر config هم نداشت → مسیر پیش‌فرض پروژه رو بساز
        if data_path is None:
            if hasattr(self.cfg, "data_path") and self.cfg.data_path:
                data_path = self.cfg.data_path
            else:
                base_dir = Path(__file__).resolve().parent.parent
                data_path = base_dir / "data" / "diabetes.csv"

        data_path = str(data_path)

        logger.info(f"Loading data from: {data_path}")

        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Dataset not found at {data_path}")

        df = self.prep.load_data(data_path)

        logger.info("Splitting data...")
        X_train, X_test, y_train, y_test = self.prep.split(
            df,
            target="Outcome",
            test_size=self.cfg.test_size,
            random_state=self.cfg.random_state
        )

        logger.info("Initializing models...")

        model_dict = {
            "DecisionTree": self.models.decision_tree(),
            "RandomForest": self.models.random_forest(),
            "LogisticRegression": self.models.logistic(),
            "KNN": self.models.knn()
        }

        scores = {}
        trained_models = {}

        logger.info("Training models...")

        for name, model in model_dict.items():

            logger.info(f"Training {name}...")

            # Train
            model.fit(X_train, y_train)

            # Cross Validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5)
            logger.in
            o(f"{name} CV Mean Accuracy: {cv_scores.mean():.3f}")

            # Evaluate
            results = self.evaluator.evaluate(model, X_test, y_test)

            scores[name] = results["accuracy"]
            trained_models[name] = model

            logger.info(f"{name} Test Accuracy: {results['accuracy']:.3f}")

            print(f"\n===== {name} Classification Report =====")
            print(results["report"])

        # -------------------------
        # Best Model Selection
        # -------------------------

        best_model_name = max(scores, key=scores.get)
        best_model = trained_models[best_model_name]

        logger.info(f"Best Model: {best_model_name}")
        logger.info(f"Best Accuracy: {scores[best_model_name]:.3f}")

        # Save best model
        model_path = self.cfg.model_path
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(best_model, model_path, protocol=4)

        logger.info("Best model saved successfully.")

        return {
            "scores": scores,
            "best_model_name": best_model_name,
            "best_model": best_model,
            "all_models": trained_models,
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": y_test,
            "dataframe": df
        }

# # src/pipeline.py
# from src.config import Config
# from src.preprocessing import Preprocessor
# from src.models import MLModels
# from src.evaluation import Evaluator
# from src.logger import get_logger
# import joblib
# import numpy as np

# logger = get_logger()

# class Pipeline:

#     def run(self, data_path):
#         cfg = Config()
#         prep = Preprocessor()
#         models = MLModels()
#         evaluator = Evaluator()

#         # Load Data
#         logger.info("Loading data...")
#         df = prep.load_data(data_path)

#         # Split
#         X_train, X_test, y_train, y_test = prep.split(df, "Outcome", cfg.test_size, cfg.random_state)

#         # Scale
#         X_train, X_test = prep.scale(X_train, X_test)

#         # Oversample
#         X_train, y_train = prep.oversample(X_train, y_train)

#         # Initialize models
#         all_models = {
#             "DecisionTree": models.decision_tree(),
#             "RandomForest": models.random_forest(),
#             "Logistic": models.logistic(),
#             "KNN": models.knn(),
#             "XGBoost": models.xgboost()
#         }

#         scores = {}
#         for name, model in all_models.items():
#             model.fit(X_train, y_train)
#             res = evaluator.evaluate(model, X_test, y_test)
#             scores[name] = res["accuracy"]
#             logger.info(f"{name}: {res['accuracy']:.3f}")
#             print(res["report"])

#         # ذخیره بهترین مدل
#         best_name = max(scores, key=scores.get)
#         joblib.dump(all_models[best_name], cfg.model_path)
#         logger.info(f"Best model: {best_name} saved at {cfg.model_path}")

#         # Return همه چیز برای نمایش نمودارها
#         return scores, X_test, y_test, all_models, df