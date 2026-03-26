from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

print("Generating dataset...")

X, y = make_classification(
    n_samples=3000,
    n_features=20,
    n_classes=2,
    random_state=42
)

print("Splitting data...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2
)

print("Training Random Forest...")

model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

print("Making predictions...")

pred = model.predict(X_test)

acc = accuracy_score(y_test, pred)

print("✅ RF Accuracy:", acc)