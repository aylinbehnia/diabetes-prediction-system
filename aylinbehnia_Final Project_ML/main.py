# from pathlib import Path
# from src.pipeline import Pipeline


# if __name__ == "__main__":
#     data_path = Path(r"D:\aylinprograming\D&Ml\Final Project\data\diabetes.csv")

#     pipe = Pipeline()
#     scores, X_test, y_test = pipe.run(data_path)

#     print("\nModel Scores:")
#     for name, score in scores.items():
#         print(f"{name}: {score:.3f}")


import sys
import os
from pathlib import Path

# اضافه کردن مسیر پروژه به PYTHONPATH
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.pipeline import Pipeline


if __name__ == "__main__":
    data_path = Path(r"D:\aylinprograming\D&Ml\Final Project\data\diabetes.csv")

    pipe = Pipeline()
    scores, X_test, y_test = pipe.run(data_path)

    print("\nModel Scores:")
    for name, score in scores.items():
        print(f"{name}: {score:.3f}")