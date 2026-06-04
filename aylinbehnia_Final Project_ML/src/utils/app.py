# import tkinter as tk
# from tkinter import ttk, messagebox
# import pandas as pd
# import joblib

# # ===============================
# # Load Model
# # ===============================
# MODEL_PATH = r"D:\aylinprograming\D&Ml\Final Project\models\saved_model.pkl"

# model = joblib.load(MODEL_PATH)
# FEATURES = list(model.feature_names_in_)

# # حذف BMI و Age از لیست برای کنترل دستی
# INPUT_FEATURES = [f for f in FEATURES if f not in ["BMI", "Age"]]

# # ===============================
# # Window Setup
# # ===============================
# root = tk.Tk()
# root.title("Diabetes Prediction AI")
# root.geometry("600x750")
# root.configure(bg="#f4f6fb")

# style = ttk.Style()
# style.theme_use("clam")

# # ===============================
# # Header
# # ===============================
# header = tk.Label(
#     root,
#     text="🩺 Diabetes Prediction Dashboard",
#     font=("Segoe UI", 16, "bold"),
#     bg="#2c7be5",
#     fg="white",
#     pady=10
# )
# header.pack(fill="x")

# # ===============================
# # Card Frame
# # ===============================
# card = ttk.Frame(root, padding=15)
# card.pack(padx=20, pady=15, fill="x")

# entries = {}

# # ===============================
# # Combobox
# # ===============================
# ttk.Label(card, text="Medical Status").pack(anchor="w")
# status_var = tk.StringVar()
# status_combo = ttk.Combobox(
#     card,
#     textvariable=status_var,
#     values=["Normal", "Pre-diabetic", "High Risk"],
#     state="readonly"
# )
# status_combo.current(0)
# status_combo.pack(fill="x", pady=5)

# # ===============================
# # Feature Entries
# # ===============================
# for feature in INPUT_FEATURES:
#     frame = ttk.Frame(card)
#     frame.pack(fill="x", pady=3)

#     ttk.Label(frame, text=feature, width=20).pack(side="left")
#     ent = ttk.Entry(frame)
#     ent.pack(side="right", fill="x", expand=True)
#     entries[feature] = ent

# # ===============================
# # BMI Spinbox
# # ===============================
# ttk.Label(card, text="BMI").pack(anchor="w", pady=(10, 0))
# bmi_var = tk.DoubleVar(value=25)
# bmi_spin = ttk.Spinbox(card, from_=10, to=60, increment=0.5, textvariable=bmi_var)
# bmi_spin.pack(fill="x")

# # ===============================
# # Age Spinbox
# # ===============================
# ttk.Label(card, text="Age").pack(anchor="w", pady=(10, 0))
# age_var = tk.IntVar(value=30)
# age_spin = ttk.Spinbox(card, from_=1, to=120, textvariable=age_var)
# age_spin.pack(fill="x")

# # ===============================
# # Checkbuttons
# # ===============================
# ttk.Label(card, text="Symptoms").pack(anchor="w", pady=(10, 0))

# symptom1 = tk.BooleanVar()
# symptom2 = tk.BooleanVar()

# ttk.Checkbutton(card, text="Frequent Urination", variable=symptom1).pack(anchor="w")
# ttk.Checkbutton(card, text="Excessive Thirst", variable=symptom2).pack(anchor="w")

# # ===============================
# # Radiobuttons
# # ===============================
# ttk.Label(card, text="Family History").pack(anchor="w", pady=(10, 0))
# family_var = tk.StringVar(value="No")

# ttk.Radiobutton(card, text="Yes", variable=family_var, value="Yes").pack(anchor="w")
# ttk.Radiobutton(card, text="No", variable=family_var, value="No").pack(anchor="w")

# # ===============================
# # Prediction Function
# # ===============================
# def predict():

#     try:
#         values = []

#         for feature in INPUT_FEATURES:
#             txt = entries[feature].get().strip()
#             if txt == "":
#                 messagebox.showerror("Error", f"Enter {feature}")
#                 return
#             values.append(float(txt))

#         # اضافه کردن BMI و Age
#         values.append(float(bmi_var.get()))
#         values.append(float(age_var.get()))

#         df = pd.DataFrame([values], columns=FEATURES)

#         pred = model.predict(df)[0]
#         prob = model.predict_proba(df)[0][1] * 100

#         if pred == 1:
#             result.set(f"🚨 High Diabetes Risk\nProbability: {prob:.2f}%")
#         else:
#             result.set(f"✅ Low Diabetes Risk\nProbability: {prob:.2f}%")

#     except Exception as e:
#         messagebox.showerror("Prediction Error", str(e))


# # ===============================
# # Buttons
# # ===============================
# btn_frame = ttk.Frame(root)
# btn_frame.pack(pady=15)

# ttk.Button(btn_frame, text="Predict", command=predict, width=15).pack(side="left", padx=5)

# def clear():
#     for e in entries.values():
#         e.delete(0, tk.END)
#     result.set("")

# ttk.Button(btn_frame, text="Clear", command=clear, width=15).pack(side="left", padx=5)

# # ===============================
# # Result Panel
# # ===============================
# result = tk.StringVar()
# result_label = ttk.Label(
#     root,
#     textvariable=result,
#     font=("Segoe UI", 11, "bold"),
#     foreground="blue",
#     wraplength=500,
#     anchor="center",
#     justify="center"
# )
# result_label.pack(pady=20)

# root.mainloop()



import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import joblib

# ===============================
# Load Model
# ===============================
MODEL_PATH = r"D:\aylinprograming\D&Ml\Final Project\models\saved_model.pkl"
model = joblib.load(MODEL_PATH)
FEATURES = list(model.feature_names_in_)
INPUT_FEATURES = [f for f in FEATURES if f not in ["BMI", "Age"]]

# ===============================
# Window Setup
# ===============================
root = tk.Tk()
root.title("Diabetes Prediction AI")
root.geometry("650x800")
root.configure(bg="#cce7ff")

style = ttk.Style()
style.theme_use("clam")

# ===============================
# Header
# ===============================
header = tk.Label(
    root,
    text="🩺 Diabetes Prediction Dashboard",
    font=("Segoe UI", 16, "bold"),
    bg="#2c7be5",
    fg="white",
    pady=12
)
header.pack(fill="x")

# ===============================
# Card Frame
# ===============================
card = ttk.Frame(root, padding=20)
card.pack(padx=20, pady=15, fill="x")

entries = {}

# ===============================
# Combobox: Gender
# ===============================
ttk.Label(card, text="جنسیت").pack(anchor="w")
gender_var = tk.StringVar()
gender_combo = ttk.Combobox(
    card,
    textvariable=gender_var,
    values=["Male", "Female"],
    state="readonly",
    font=("Segoe UI", 10)
)
gender_combo.current(0)
gender_combo.pack(fill="x", pady=5)

# ===============================
# Combobox: Medical Status
# ===============================
ttk.Label(card, text="وضعیت پزشکی").pack(anchor="w", pady=(10, 0))
status_var = tk.StringVar()
status_combo = ttk.Combobox(
    card,
    textvariable=status_var,
    values=["Normal", "Pre-diabetic", "High Risk"],
    state="readonly",
    font=("Segoe UI", 10)
)
status_combo.current(0)
status_combo.pack(fill="x", pady=5)

# ===============================
# Feature Entries
# ===============================
for feature in INPUT_FEATURES:
    frame = ttk.Frame(card)
    frame.pack(fill="x", pady=3)

    ttk.Label(frame, text=feature, width=20).pack(side="left")
    ent = ttk.Entry(frame)
    ent.pack(side="right", fill="x", expand=True)
    entries[feature] = ent

# ===============================
# BMI Spinbox
# ===============================
ttk.Label(card, text="BMI").pack(anchor="w", pady=(10, 0))
bmi_var = tk.DoubleVar(value=25)
bmi_spin = ttk.Spinbox(card, from_=10, to=60, increment=0.5, textvariable=bmi_var, font=("Segoe UI", 10))
bmi_spin.pack(fill="x")

# ===============================
# Age Spinbox
# ===============================
ttk.Label(card, text="سن").pack(anchor="w", pady=(10, 0))
age_var = tk.IntVar(value=30)
age_spin = ttk.Spinbox(card, from_=1, to=120, textvariable=age_var, font=("Segoe UI", 10))
age_spin.pack(fill="x")

# ===============================
# Checkbuttons: Symptoms
# ===============================
ttk.Label(card, text="علائم").pack(anchor="w", pady=(10, 0))
symptom1 = tk.BooleanVar()
symptom2 = tk.BooleanVar()
ttk.Checkbutton(card, text="تکرر ادرار", variable=symptom1).pack(anchor="w")
ttk.Checkbutton(card, text="تشنگی زیاد", variable=symptom2).pack(anchor="w")

# ===============================
# Radiobuttons: Family History
# ===============================
ttk.Label(card, text="سابقه خانوادگی دیابت").pack(anchor="w", pady=(10, 0))
family_var = tk.StringVar(value="No")
ttk.Radiobutton(card, text="بله", variable=family_var, value="Yes").pack(anchor="w")
ttk.Radiobutton(card, text="خیر", variable=family_var, value="No").pack(anchor="w")

# ===============================
# Prediction Function
# ===============================
def show_result(pred, prob):
    win = tk.Toplevel(root)
    win.title("نتیجه پیش‌بینی")
    win.geometry("520x350")
    win.resizable(False, False)

    frame = ttk.Frame(win, padding=15)
    frame.pack(fill="both", expand=True)

    tk.Label(
        frame,
        text="گزارش هوش مصنوعی تشخیص دیابت",
        font=("Segoe UI", 14, "bold")
    ).pack(pady=10)

    if pred == 1:
        msg = f"""🔴 احتمال ابتلا به دیابت بالا است

درصد ریسک تخمینی: {prob:.2f} %

توصیه‌ها:
• انجام آزمایش HbA1c و قند خون ناشتا
• مراجعه به پزشک متخصص غدد
• کاهش مصرف قند و کربوهیدرات ساده
• افزایش فعالیت بدنی حداقل ۳۰ دقیقه روزانه
• کنترل وزن و شاخص BMI
• پایش منظم قند خون
"""
    else:
        msg = f"""🟢 احتمال ابتلا به دیابت پایین است

درصد ریسک تخمینی: {prob:.2f} %

توصیه‌های پیشگیرانه:
• حفظ رژیم غذایی سالم
• ورزش منظم
• چکاپ سالانه قند خون
• کنترل وزن و استرس
• خواب کافی و سبک زندگی سالم
"""

    txt = tk.Text(frame, height=12, wrap="word", font=("Segoe UI", 10))
    txt.pack(fill="both", expand=True)
    txt.insert("1.0", msg)
    txt.config(state="disabled")

    ttk.Button(frame, text="بستن", command=win.destroy, width=12).pack(pady=10)

def predict():
    try:
        values = []
        for feature in INPUT_FEATURES:
            txt = entries[feature].get().strip()
            if txt == "":
                messagebox.showerror("خطا", f"لطفا {feature} را وارد کنید")
                return
            values.append(float(txt))

        values.append(float(bmi_var.get()))
        values.append(float(age_var.get()))
        df = pd.DataFrame([values], columns=FEATURES)

        pred = model.predict(df)[0]
        prob = model.predict_proba(df)[0][1] * 100

        show_result(pred, prob)

    except Exception as e:
        messagebox.showerror("Prediction Error", str(e))

# ===============================
# Buttons
# ===============================
btn_frame = ttk.Frame(root)
btn_frame.pack(pady=15)

ttk.Button(btn_frame, text="Predict", command=predict, width=20).pack(side="left", padx=10)
def clear():
    for e in entries.values():
        e.delete(0, tk.END)
ttk.Button(btn_frame, text="Clear", command=clear, width=20).pack(side="left", padx=10)

root.mainloop()