from pathlib import Path
import joblib

def load_model():
    try:
        # مسیر ریشه پروژه
        base_dir = Path(__file__).resolve().parent.parent

        # مسیر مدل
        model_path = base_dir / "models" / "saved_model.pkl"

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at: {model_path}")

        model = joblib.load(model_path)
        print("✅ Model Loaded Successfully")
        print(model)

    except Exception as e:
        print("❌ Error loading model:", e)


if __name__ == "__main__":
    load_model()