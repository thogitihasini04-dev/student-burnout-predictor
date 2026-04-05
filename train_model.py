import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

# Load dataset
data = pd.read_csv("data/student_burnout_data.csv")

# Features & target
X = data.drop("Burnout_Risk", axis=1)
y = data["Burnout_Risk"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
model.fit(X_train, y_train)

# Save model
os.makedirs("model", exist_ok=True)
with open("model/burnout_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Burnout prediction model trained successfully")
