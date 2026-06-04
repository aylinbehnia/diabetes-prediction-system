from dataclasses import dataclass

@dataclass
class Config:
    test_size: float = 0.2
    random_state: int = 42
    model_path: str = r"D:\aylinprograming\D&Ml\Final Project\models\saved_model.pkl"
    data_path: str = r"D:\aylinprograming\D&Ml\Final Project\data\diabetes.csv"
