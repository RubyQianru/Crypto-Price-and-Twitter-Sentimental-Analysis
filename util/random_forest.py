# Import external libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def train_model(X_train_scaled, y_train):
  # Train random forest model
  model = RandomForestClassifier(n_estimators=100, random_state=42)
  model.fit(X_train_scaled, y_train)

  return model

def evaluate_model(model, X_test_scaled, y_test):
  # Predict and evaluate
  y_pred = model.predict(X_test_scaled)

  # Calculate metrics
  accuracy = accuracy_score(y_test, y_pred)
  precision = precision_score(y_test, y_pred)
  recall = recall_score(y_test, y_pred)
  f1 = f1_score(y_test, y_pred)
  conf_matrix = confusion_matrix(y_test, y_pred)

  print("\nModel Performance:")
  print(f"Accuracy: {accuracy:.4f}")
  print(f"Precision: {precision:.4f}")
  print(f"Recall: {recall:.4f}")
  print(f"F1 Score: {f1:.4f}")
  print("Confusion Matrix:")
  print(conf_matrix)

  return accuracy, precision, recall, f1, conf_matrix