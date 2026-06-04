# from sklearn.tree import DecisionTreeClassifier
# from sklearn.linear_model import LogisticRegression

# class MLModels:

#     def decision_tree(self):
#         return DecisionTreeClassifier()

#     def logistic(self):
#         return LogisticRegression(max_iter=1000)




# src/models.py
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import joblib
# from xgboost import XGBClassifier

class MLModels:

    def logistic(self):
        return LogisticRegression(max_iter=1000)

    def decision_tree(self):
        return DecisionTreeClassifier(max_depth=5, random_state=42)

    def random_forest(self):
        return RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)

    def svm(self):
        return SVC(kernel="rbf", probability=True)

    def knn(self):
        return KNeighborsClassifier(n_neighbors=5)

    def naive_bayes(self):
        return GaussianNB()

    def xgboost(self):
        return XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            random_state=42,
            use_label_encoder=False,
            eval_metric="logloss"
        )
    def save_model(model, name):
        joblib.dump(model, f"{name}.pkl")