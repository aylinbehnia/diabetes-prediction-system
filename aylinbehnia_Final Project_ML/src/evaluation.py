# from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# class Evaluator:

#     def evaluate(self, model, X_test, y_test):
#         y_pred = model.predict(X_test)

#         return {
#             "accuracy": accuracy_score(y_test, y_pred),
#             "confusion": confusion_matrix(y_test, y_pred),
#             "report": classification_report(y_test, y_pred)
#         }



# src/evaluation.py
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

class Evaluator:

    def evaluate(self, model, X_test, y_test):
        y_pred = model.predict(X_test)
        return {
            "accuracy": accuracy_score(y_test, y_pred),
            "confusion": confusion_matrix(y_test, y_pred),
            "report": classification_report(y_test, y_pred)
        }

    def plot_confusion_matrix(self, y_true, y_pred, title="Confusion Matrix"):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(5,4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(title)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.show()

    def plot_roc_curve(self, model, X_test, y_test, title="ROC Curve"):
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)[:, 1]
        else:  # برای SVM
            y_score = model.decision_function(X_test)
        fpr, tpr, thresholds = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(6,5))
        plt.plot(fpr, tpr, color="blue", label=f"AUC = {roc_auc:.3f}")
        plt.plot([0,1], [0,1], color="red", linestyle="--")
        plt.title(title)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.show()

    def feature_importance(self, model, feature_names, top_n=10):
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            indices = importances.argsort()[::-1][:top_n]
            plt.figure(figsize=(8,5))
            sns.barplot(x=importances[indices], y=[feature_names[i] for i in indices], palette="viridis")
            plt.title("Feature Importance")
            plt.xlabel("Importance")
            plt.ylabel("Feature")
            plt.show()
        else:
            print("Model has no feature_importances_ attribute.")
            
    def full_metrics(y_test, pred, prob):
        return {
            "accuracy": accuracy_score(y_test, pred),
            "precision": precision_score(y_test, pred),
            "recall": recall_score(y_test, pred),
            "f1": f1_score(y_test, pred),
            "roc_auc": roc_auc_score(y_test, prob)
        }       
        