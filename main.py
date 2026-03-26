import yaml
from src.train import run_training

print("🚀 Starting ML IDS Project...")

with open("config.yaml") as f:
    config = yaml.safe_load(f)

run_training(config)