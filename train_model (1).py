import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib

print("Loading dataset...")
df = pd.read_csv("train.csv")

print("Dataset shape:", df.shape)

# Remove id column
df = df.drop("id", axis=1)

print("Encoding categorical columns...")
categorical_columns = [
    "gender",
    "ethnicity",
    "education_level",
    "income_level",
    "smoking_status",
    "employment_status"
]

encoders = {}

for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

print("Separating features and target...")
X = df.drop("diagnosed_diabetes", axis=1)
y = df["diagnosed_diabetes"]

print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Scaling data...")
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("Training model...")
model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

print("Saving files...")
joblib.dump(model, "model.joblib")
joblib.dump(scaler, "scaler.joblib")
joblib.dump(encoders, "encoders.joblib")

print("Done! Model saved successfully.")