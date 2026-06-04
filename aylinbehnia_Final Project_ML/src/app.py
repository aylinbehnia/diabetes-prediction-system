import tkinter as tk
from tkinter import ttk

from tkinter import ttk, messagebox, filedialog
import pandas as pd
import joblib
from PIL import Image, ImageTk
import os
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from matplotlib import rcParams
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

import numpy as np

import arabic_reshaper
from bidi.algorithm import get_display
# ===============================
# Paths
# ===============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "models", "saved_model.pkl")
IMG_BG = os.path.join(BASE_DIR, "utils", "assets", "images", "b.png")
IMG_RESULT = os.path.join(BASE_DIR, "utils", "assets", "images", "page1.png")
ICON_MAIN = os.path.join(BASE_DIR, "utils", "assets", "icons", "page1.ico")
ICON_RESULT = os.path.join(BASE_DIR, "utils", "assets", "icons", "page2.ico")
ICON_DASHBOARD = os.path.join(BASE_DIR, "utils", "assets", "icons", "page3.ico")

# ===============================
# Load Model
# ===============================
model = joblib.load(MODEL_PATH)
FEATURES = list(model.feature_names_in_)
INPUT_FEATURES = [f for f in FEATURES if f not in ["BMI", "Age"]]

# ===============================
# Window Setup
# ===============================
root = tk.Tk()
root.title("🩺 Diabetes Prediction AI")
root.geometry("650x800")
root.resizable(False, False)
root.iconbitmap(ICON_MAIN)

# ===============================
# Background
# ===============================
bg = Image.open(IMG_BG).resize((650, 800))
bg_photo = ImageTk.PhotoImage(bg)
bg_label = tk.Label(root, image=bg_photo)
bg_label.image = bg_photo
bg_label.place(x=0, y=0, relwidth=1, relheight=1)

style = ttk.Style()
style.theme_use("clam")

# ===============================
# Light / Dark Mode
# ===============================
def toggle_theme():
    if root["bg"] == "#cce7ff":  # Light -> Dark
        root.configure(bg="#1e1e1e")
        card.configure(bg="#2e2e2e")
        header.configure(bg="#444444", fg="white")
    else:  # Dark -> Light
        root.configure(bg="#cce7ff")
        card.configure(bg="white")
        header.configure(bg="#2c7be5", fg="white")

toggle_btn = ttk.Button(root, text="Toggle Light/Dark Mode", command=toggle_theme)
toggle_btn.pack(pady=5)

# ===============================
# Header
# ===============================
header = tk.Label(
    root,
    text="🩺 داشبورد پیش‌بینی دیابت",
    font=("Segoe UI", 18, "bold"),
    bg="#2c7be5",
    fg="white",
    pady=15
)
header.pack(fill="x")

# ===============================
# Card Frame
# ===============================
card = tk.Frame(root, bg="white", bd=2, relief="groove")
card.pack(padx=20, pady=20, fill="x")

entries = {}

# ===============================
# Comboboxes
# ===============================
tk.Label(card, text="جنسیت", font=("Segoe UI", 11), bg="white").pack(anchor="w", pady=(5,0))
gender_var = tk.StringVar()
gender_combo = ttk.Combobox(card, textvariable=gender_var, values=["Male", "Female"], state="readonly")
gender_combo.current(0)
gender_combo.pack(fill="x", pady=5)

tk.Label(card, text="وضعیت پزشکی", font=("Segoe UI", 11), bg="white").pack(anchor="w", pady=(10,0))
status_var = tk.StringVar()
status_combo = ttk.Combobox(card, textvariable=status_var, values=["Normal", "Pre-diabetic", "High Risk"], state="readonly")
status_combo.current(0)
status_combo.pack(fill="x", pady=5)

# ===============================
# Feature Entries
# ===============================
for feature in INPUT_FEATURES:
    frame = tk.Frame(card, bg="white")
    frame.pack(fill="x", pady=3)

    tk.Label(frame, text=feature, width=20, anchor="w", bg="white", font=("Segoe UI", 10)).pack(side="left")
    ent = ttk.Entry(frame)
    ent.pack(side="right", fill="x", expand=True)
    entries[feature] = ent

# ===============================
# BMI & Age
# ===============================
tk.Label(card, text="BMI", font=("Segoe UI", 11), bg="white").pack(anchor="w", pady=(10,0))
bmi_var = tk.DoubleVar(value=25)
bmi_spin = ttk.Spinbox(card, from_=10, to=60, increment=0.5, textvariable=bmi_var)
bmi_spin.pack(fill="x")

tk.Label(card, text="سن", font=("Segoe UI", 11), bg="white").pack(anchor="w", pady=(10,0))
age_var = tk.IntVar(value=30)
age_spin = ttk.Spinbox(card, from_=1, to=120, textvariable=age_var)
age_spin.pack(fill="x")

# ===============================
# Symptoms
# ===============================
tk.Label(card, text="علائم", font=("Segoe UI", 11), bg="white").pack(anchor="w", pady=(10,0))
symptom1 = tk.BooleanVar()
symptom2 = tk.BooleanVar()
ttk.Checkbutton(card, text="تکرر ادرار", variable=symptom1).pack(anchor="w")
ttk.Checkbutton(card, text="تشنگی زیاد", variable=symptom2).pack(anchor="w")

# ===============================
# Family History
# ===============================
tk.Label(card, text="سابقه خانوادگی دیابت", font=("Segoe UI", 11), bg="white").pack(anchor="w", pady=(10,0))
family_var = tk.StringVar(value="No")
ttk.Radiobutton(card, text="بله", variable=family_var, value="Yes").pack(anchor="w")
ttk.Radiobutton(card, text="خیر", variable=family_var, value="No").pack(anchor="w")

# ===============================
# Save PDF Function
# ===============================
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

def save_pdf(report_text, prob):

    file_path = filedialog.asksaveasfilename(
        defaultextension=".pdf"
    )

    if file_path:

        doc = SimpleDocTemplate(file_path)
        styles = getSampleStyleSheet()

        story = []

        story.append(Paragraph(
            "گزارش تحلیل هوش مصنوعی دیابت",
            styles["Title"]
        ))

        story.append(Paragraph(
            f"Risk Probability: {prob:.2f} %",
            styles["Normal"]
        ))

        story.append(Paragraph(report_text.replace("\n", "<br/>"),
                               styles["Normal"]))

        doc.build(story)

        messagebox.showinfo("OK", "PDF ذخیره شد ✅")
        
def save_chart(fig):
    file_path = filedialog.asksaveasfilename(
        defaultextension=".png",
        filetypes=[("PNG Image", "*.png")]
    )
    if file_path:
        fig.savefig(file_path)
        messagebox.showinfo("ذخیره شد", "نمودار ذخیره شد ✅")       

# ===============================
# Show Result
# ===============================

def save_pdf(pred_text, prob):
    file_path = filedialog.asksaveasfilename(defaultextension=".pdf", filetypes=[("PDF files", "*.pdf")])
    if file_path:
        c = canvas.Canvas(file_path, pagesize=letter)
        c.setFont("Helvetica", 12)
        c.drawString(50, 750, "گزارش هوش مصنوعی تشخیص دیابت")
        c.drawString(50, 720, f"درصد ریسک تخمینی: {prob:.2f}%")
        c.drawString(50, 700, pred_text)
        c.save()
        messagebox.showinfo("ذخیره شد", f"گزارش در مسیر {file_path} ذخیره شد")

def show_result(pred, prob):

    win = tk.Toplevel(root)
    win.title("نتیجه پیش‌بینی")
    win.geometry("750x650")
    win.resizable(False, False)

    try:
        win.iconbitmap(ICON_RESULT)
    except:
        pass

    frame = tk.Frame(win, bg="white", bd=2, relief="groove")
    frame.pack(fill="both", expand=True, padx=10, pady=10)
    
    tk.Label(
    frame,
    text="🧠 Diabetes AI Diagnostic Engine v1.0",
    font=("Segoe UI", 9, "italic"),
    bg="white",
    fg="#888"
    ).pack()

    # ================= Risk Level =================

    if prob < 30:
        risk_level = "🟢 ریسک پایین"
        color = "#4CAF50"

        report_text = """
🟢 ارزیابی نهایی: ریسک پایین

احتمال ابتلا به دیابت در محدوده کم قرار دارد.

✅ توصیه‌ها:
• حفظ رژیم غذایی سالم  
• فعالیت بدنی منظم  
• انجام چکاپ سالانه  
• کنترل استرس و خواب کافی  

⚠️ این نتیجه تنها یک تحلیل هوش مصنوعی است.
"""

    elif prob < 70:
        risk_level = "🟡 ریسک متوسط"
        color = "#FFC107"

        report_text = """
🟡 ارزیابی نهایی: ریسک متوسط

برخی فاکتورهای خطر مشاهده شده است.

✅ پیشنهادات:
• کاهش مصرف قند و نوشیدنی‌های شیرین  
• افزایش فعالیت بدنی  
• کنترل وزن  
• بررسی دوره‌ای قند خون  

🔎 پیشگیری زودهنگام بسیار مهم است.
"""

    else:
        risk_level = "🔴 ریسک بالا"
        color = "#FF4C4C"

        report_text = """
🔴 ارزیابی نهایی: ریسک بالا

احتمال ابتلا به دیابت در سطح بالا تشخیص داده شده است.

⚠️ اقدامات پیشنهادی:
• انجام آزمایش HbA1c  
• مراجعه به پزشک متخصص  
• اصلاح رژیم غذایی  
• ورزش منظم  
• پایش مداوم قند خون  

❗ این نتیجه تشخیص قطعی پزشکی نیست.
"""

    rec = []

    try:
        if float(entries["Glucose"].get()) > 140:
            rec.append("🔹 قند خون بالا → آزمایش HbA1c")

        if age_var.get() > 45:
            rec.append("🔹 سن بالا → چکاپ سالانه")

        if family_var.get()=="Yes":
            rec.append("🔹 سابقه خانوادگی → رژیم کم قند")
    except:
        pass

    if rec:
        report_text += "\n\n📋 توصیه شخصی:\n" + "\n".join(rec)
    # ===== Header Result =====
    tk.Label(
        frame,
        text=f"{risk_level}\nRisk = {prob:.2f} %",
        font=("Segoe UI", 16, "bold"),
        fg=color,
        bg="white"
    ).pack(pady=10)

    progress = ttk.Progressbar(
        frame,
        orient="horizontal",
        length=400,
        mode="determinate"
    )

    progress.pack(pady=10)

    progress["value"] = prob

    # ===== Report Text Box =====
    text_box = tk.Text(
        frame,
        height=12,
        wrap="word",
        font=("Segoe UI", 11),
        bg="#f8f9fa",
        bd=0
    )

    text_box.insert("1.0", report_text)
    text_box.config(state="disabled")
    text_box.pack(fill="both", expand=True, pady=10)

    # ================= Chart Page =================

    def open_chart_page():

        chart_win = tk.Toplevel(win)
        chart_win.title("Advanced Medical Risk Dashboard")
        chart_win.geometry("700x750")  # کمی کمتر شد که فیت‌تر بشه
        chart_win.configure(bg="white")
        chart_win.iconbitmap(ICON_DASHBOARD)

        # ====== MAIN CONTAINER ======
        container = tk.Frame(chart_win, bg="white")
        container.pack(fill="both", expand=True)

        # ================= Risk Level =================
        if prob < 30:
            main_color = "#2E7D32"
            risk_label = "LOW RISK"
        elif prob < 70:
            main_color = "#F9A825"
            risk_label = "MODERATE RISK"
        else:
            main_color = "#C62828"
            risk_label = "HIGH RISK"

        # ================= Animated Donut =================
        fig, ax = plt.subplots(figsize=(4.5,4.5))  # کوچکتر شد که بالاتر بیاد
        fig.patch.set_facecolor("white")

        def draw_chart(value):
            ax.clear()
            ax.pie(
                [value, 100 - value],
                startangle=90,
                colors=[main_color, "#ECEFF1"],
                wedgeprops=dict(width=0.32)
            )
            ax.text(
                0, 0.1,
                f"{value:.1f}%",
                ha="center",
                va="center",
                fontsize=26,
                fontweight="bold",
                color=main_color
            )
            ax.text(
                0, -0.15,
                risk_label,
                ha="center",
                va="center",
                fontsize=12,
                fontweight="bold",
                color="#555"
            )
            ax.axis("equal")

        for i in range(int(prob)+1):
            draw_chart(i)
            fig.canvas.draw()
            chart_win.update()
            chart_win.after(8)

        canvas_fig = FigureCanvasTkAgg(fig, master=container)
        canvas_fig.draw()
        canvas_fig.get_tk_widget().pack(pady=(10,5))  # رفت بالاتر

        # ================= Gauge Scale =================
        tk.Label(
            container,
            text="Medical Risk Scale",
            font=("Segoe UI", 11, "bold"),
            bg="white"
        ).pack()

        scale = ttk.Progressbar(
            container,
            orient="horizontal",
            length=420,
            mode="determinate"
        )
        scale.pack(pady=5)
        scale["value"] = prob

        # ================= Feature Importance =================
        rf = model
        importances = rf.feature_importances_
        features = FEATURES
        sorted_idx = np.argsort(importances)

        fig2, ax2 = plt.subplots(figsize=(6,2.8))
        ax2.barh(
            [features[i] for i in sorted_idx],
            importances[sorted_idx]
        )
        ax2.set_title("Feature Importance (Real)")
        ax2.set_xlabel("Importance")

        canvas_fig2 = FigureCanvasTkAgg(fig2, master=container)
        canvas_fig2.draw()
        canvas_fig2.get_tk_widget().pack(pady=10)

        # ================= Save Buttons =================
        btn_frame = tk.Frame(container, bg="white")
        btn_frame.pack(pady=10)

        ttk.Button(
            btn_frame,
            text="Save Donut Chart",
            width=18,
            command=lambda: fig.savefig("donut_chart.png")
        ).pack(side="left", padx=5)

        ttk.Button(
            btn_frame,
            text="Save Feature Chart",
            width=18,
            command=lambda: fig2.savefig("feature_importance.png")
        ).pack(side="left", padx=5)

        # ================= Disclaimer =================
        tk.Label(
            container,
            text="AI-based probability analysis. Not a definitive medical diagnosis.",
            font=("Segoe UI", 9),
            bg="white",
            fg="#777",
            justify="center"
        ).pack(pady=5)
    # ================= Buttons =================

    btn_frame = tk.Frame(frame, bg="white")
    btn_frame.pack(pady=10)

    ttk.Button(
        btn_frame,
        text="نمایش نمودار",
        width=18,
        command=open_chart_page
    ).pack(side="left", padx=5)

    ttk.Button(
        btn_frame,
        text="ذخیره PDF",
        width=18,
        command=lambda: save_pdf(report_text, prob)
    ).pack(side="left", padx=5)

    ttk.Button(
        btn_frame,
        text="بستن",
        width=18,
        command=win.destroy
    ).pack(side="left", padx=5)

# ===============================
# Predict Function
# ===============================
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
btn_frame = tk.Frame(root, bg="#cce7ff")
btn_frame.pack(pady=15)

# Hover effect
def on_enter(e): e.widget.config(style="Hover.TButton")
def on_leave(e): e.widget.config(style="TButton")
style.configure("Hover.TButton", background="#4CAF50", foreground="white")

btn_predict = ttk.Button(btn_frame, text="Predict", command=predict, width=20)
btn_predict.pack(side="left", padx=10)
btn_predict.bind("<Enter>", on_enter)
btn_predict.bind("<Leave>", on_leave)

def clear():
    for e in entries.values():
        e.delete(0, tk.END)
ttk.Button(btn_frame, text="Clear", command=clear, width=20).pack(side="left", padx=10)

root.mainloop()