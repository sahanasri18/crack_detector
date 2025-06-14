from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import joblib
from utils.preprocess import extract_features_labels

print("🔍 Extracting features and labels...")
X, y = extract_features_labels()

print("🧪 Splitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print("🌲 Training Random Forest classifier...")
model = RandomForestClassifier(n_estimators=100, class_weight='balanced', max_depth=10)
model.fit(X_train, y_train)

print("📈 Evaluating model...")
y_pred = model.predict(X_test)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("✅ Accuracy:", round(accuracy_score(y_test, y_pred)*100, 2), "%")

joblib.dump(model, "model/rf_model.joblib")
print("💾 Model saved at: model/rf_model.joblib")

