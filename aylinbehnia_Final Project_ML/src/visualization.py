# # src/visualization.py
# import matplotlib.pyplot as plt
# import seaborn as sns
# import arabic_reshaper
# from bidi.algorithm import get_display
# import os
# import numpy as np

# class Visualizer:

#     # تابع فارسی‌سازی متن
#     @staticmethod
#     def fa(text):
#         return get_display(arabic_reshaper.reshape(text))

#     # نمودار Accuracy
#     def plot_accuracy(self, results):
#         plt.figure(figsize=(6,4))
#         plt.bar(results.keys(), results.values(), color="teal")
#         plt.title(self.fa("دقت مدل‌ها"))
#         plt.ylabel(self.fa("Accuracy"))
#         plt.show()

#     # ذخیره نمودار
#     def save_plot(self, filename="plot.png"):
#         os.makedirs("outputs", exist_ok=True)
#         plt.savefig(f"outputs/{filename}", dpi=300, bbox_inches='tight')
#         print("Saved in outputs/")

#     # Confusion Matrix
#     def plot_confusion(self, cm):
#         plt.figure(figsize=(5,4))
#         sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
#         plt.title(self.fa("Confusion Matrix"))
#         plt.xlabel(self.fa("Predicted"))
#         plt.ylabel(self.fa("Actual"))
#         plt.show()

# src/visualization.py

import matplotlib.pyplot as plt
import seaborn as sns
import arabic_reshaper
from bidi.algorithm import get_display
import os
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from sklearn.metrics import roc_curve, auc


class Visualizer:

    def __init__(self):
        sns.set_theme(style="whitegrid", font_scale=1.2)

    # -----------------------------
    # فارسی‌سازی متن
    # -----------------------------
    @staticmethod
    def fa(text):
        return get_display(arabic_reshaper.reshape(text))

    # -----------------------------
    # نمودار Accuracy حرفه‌ای
    # -----------------------------
    def plot_accuracy(self, scores, save=False):

        df = pd.DataFrame(list(scores.items()), columns=["Model", "Accuracy"])
        df = df.sort_values(by="Accuracy", ascending=False)

        best_model = df.iloc[0]["Model"]

        plt.figure(figsize=(10,6))
        colors = sns.color_palette("viridis", len(df))

        ax = sns.barplot(
            data=df,
            x="Model",
            y="Accuracy",
            palette=colors
        )

        # هایلایت بهترین مدل
        for i, bar in enumerate(ax.patches):
            if df.iloc[i]["Model"] == best_model:
                bar.set_edgecolor("red")
                bar.set_linewidth(3)

        # نوشتن درصد روی ستون
        for i, value in enumerate(df["Accuracy"]):
            plt.text(
                i,
                value + 0.02,
                f"{value*100:.1f}%",
                ha="center",
                fontsize=12,
                fontweight="bold"
            )

        plt.title(self.fa("مقایسه دقت مدل‌ها"), fontsize=16, fontweight="bold")
        plt.xlabel(self.fa("مدل‌ها"))
        plt.ylabel("Accuracy")
        plt.ylim(0, 1)

        legend_elements = [
            Patch(facecolor="none", edgecolor="red",
                  linewidth=3, label=self.fa("بهترین مدل"))
        ]
        plt.legend(handles=legend_elements)

        plt.tight_layout()

        if save:
            self._save("accuracy.png")

        plt.show()
        
        
    def compare_models(self, names, vals, save=False):

        cleaned_scores = {}

        for name, val in zip(names, vals):

            # اگر دیکشنری بود
            if isinstance(val, dict):
                val = val.get("accuracy", list(val.values())[0])

            # اگر Series بود
            if hasattr(val, "values"):
                val = float(val.values[0])

            # اگر DataFrame بود
            if hasattr(val, "iloc"):
                try:
                    val = float(val.iloc[0])
                except:
                    val = float(val.iloc[0, 0])

            cleaned_scores[name] = float(val)

        self.plot_accuracy(cleaned_scores, save=save)    

    # -----------------------------
    # Confusion Matrix حرفه‌ای
    # -----------------------------
    def plot_confusion(self, cm, save=False):

        plt.figure(figsize=(6,5))

        ax = sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=True,
            linewidths=1,
            linecolor="gray"
        )

        ax.set_title(self.fa("ماتریس سردرگمی"), fontsize=15)
        ax.set_xlabel(self.fa("پیش‌بینی شده"))
        ax.set_ylabel(self.fa("واقعی"))

        ax.set_xticklabels(
            [self.fa("منفی"), self.fa("مثبت")]
        )
        ax.set_yticklabels(
            [self.fa("منفی"), self.fa("مثبت")],
            rotation=0
        )

        plt.tight_layout()

        if save:
            self._save("confusion_matrix.png")

        plt.show()

    # -----------------------------
    # ROC Curve حرفه‌ای
    # -----------------------------
    def plot_roc(self, model, X_test, y_test, save=False):

        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(7,6))

        plt.plot(fpr, tpr, linewidth=3,
                 label=f"AUC = {roc_auc:.3f}")

        plt.plot([0,1], [0,1], linestyle="--")

        plt.title(self.fa("ROC Curve"))
        plt.xlabel(self.fa("نرخ مثبت کاذب"))
        plt.ylabel(self.fa("نرخ مثبت واقعی"))

        plt.legend(loc="lower right")
        plt.grid(True)

        if save:
            self._save("roc_curve.png")

        plt.show()

    # -----------------------------
    
    # Feature Importance حرفه‌ای
    # -----------------------------
    def plot_feature_importance(self, model, feature_names, save=False):

        if not hasattr(model, "feature_importances_"):
            print("Model does not support feature importance.")
            return

        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
    

    def roc_plot(self, model, X_test, y_test):
        prob = model.predict_proba(X_test)[:,1]
        fpr, tpr, _ = roc_curve(y_test, prob)
        plt.plot(fpr, tpr)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.show()

    def heatmap(self, df):
        """
        رسم ماتریس همبستگی دیتاست
        """
        plt.figure(figsize=(10, 8))
        corr = df.select_dtypes(include=['number']).corr()

        sns.heatmap(
            corr,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            linewidths=0.5
        )

        plt.title("Correlation Matrix", fontsize=14)
        plt.tight_layout()
        plt.show()

    def pairplot(self, df):
        """
        رسم Pairplot برای بررسی روابط بین ویژگی‌ها
        """
        sns.pairplot(df)
        plt.show()

    def boxplot(self, df, column):
        """
        رسم Boxplot برای یک ستون مشخص
        """
        plt.figure(figsize=(6, 4))
        sns.boxplot(x=df[column])
        plt.title(f"Boxplot of {column}")
        plt.show()    