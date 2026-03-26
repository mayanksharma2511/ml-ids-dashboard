from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def train_rf(X_train, X_test, y_train, y_test):
    print("🌲 Training Random Forest...")

    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)

    acc = accuracy_score(y_test, preds)

    print("🌲 RF Accuracy:", acc)

    return acc, preds, probs