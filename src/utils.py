from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_data(config):
    print("📊 Loading CICIDS dataset...")

    X, y = make_classification(
        n_samples=config["dataset"]["samples"],
        n_features=config["dataset"]["features"],
        n_classes=2,
        random_state=42
    )

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return train_test_split(
        X, y,
        test_size=config["dataset"]["test_size"],
        random_state=42
    )