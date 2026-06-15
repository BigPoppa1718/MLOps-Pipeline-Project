import os
from src.data_preprocessing import generate_synthetic_data
from src.train import load_config

config = load_config()
df = generate_synthetic_data(config["data"]["num_rows"], config["data"]["random_seed"])

os.makedirs("data", exist_ok=True)
df.to_csv("data/heart_disease.csv", index=False)
print("Dataset exported successfully to data/heart_disease.csv")