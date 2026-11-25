
import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

# Create the machine_learning folder
os.makedirs("machine_learning", exist_ok=True)
os.makedirs("dataset", exist_ok=True)

# Starter training examples
data = {
    "Education": [
        "This is a university semester examination hall ticket",
        "Student marksheet and academic grade report",
        "College admission application and enrollment form",
        "University degree certificate and transcript",
        "Class timetable and subject syllabus",
        "Assignment submission and project report",
        "Student attendance and examination results",
        "Tuition fee receipt from college",
        "Scholarship application for university students",
        "Course completion certificate and academic record"
    ],

    "Finance": [
        "Bank account statement and transaction details",
        "Monthly salary slip and payment details",
        "Income tax return and tax payment receipt",
        "Loan application and repayment schedule",
        "Credit card bill and outstanding balance",
        "Invoice for business purchase and payment",
        "Bank deposit and withdrawal statement",
        "Financial audit and annual accounts report",
        "Insurance premium payment receipt",
        "Monthly electricity payment transaction"
    ],

    "Career": [
        "Resume and professional work experience",
        "Job application and candidate information",
        "Internship offer letter and joining details",
        "Employment appointment letter",
        "Professional experience certificate",
        "Job interview schedule and invitation",
        "Employee identification and company details",
        "LinkedIn professional profile and skills",
        "Training completion and career certificate",
        "Job recruitment and selection letter"
    ],

    "Medical": [
        "Hospital medical report and patient details",
        "Doctor prescription and medicine dosage",
        "Blood test laboratory report",
        "Medical examination and health checkup",
        "Hospital discharge summary",
        "Patient diagnosis and treatment details",
        "Pharmacy medicine bill",
        "Radiology scan and medical imaging report",
        "Vaccination certificate and health record",
        "Clinical laboratory test results"
    ],

    "Utility": [
        "Electricity bill and consumer number",
        "Water supply bill and payment details",
        "Internet broadband monthly bill",
        "Mobile phone recharge and billing statement",
        "Gas connection bill and payment receipt",
        "Telephone service invoice",
        "Property tax payment receipt",
        "Internet service provider account details",
        "Household utility payment receipt",
        "Cable television subscription bill"
    ],

    "Others": [
        "Personal invitation for a birthday celebration",
        "Travel itinerary and holiday schedule",
        "Event participation invitation",
        "Personal letter and general communication",
        "Restaurant food menu and price list",
        "Meeting agenda and discussion notes",
        "Product user manual and instructions",
        "General announcement and information notice",
        "Personal shopping list",
        "Event registration confirmation"
    ]
}

# Convert examples into a dataset
records = []

for category, examples in data.items():
    for example in examples:
        records.append({
            "text": example,
            "category": category
        })

df = pd.DataFrame(records)

# Save dataset
df.to_csv("dataset/document_dataset.csv", index=False)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    df["text"],
    df["category"],
    test_size=0.2,
    random_state=42,
    stratify=df["category"]
)

# Build the ML pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2)
    )),
    ("classifier", MultinomialNB())
])

# Train
model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)

print("Model training completed!")
print("Accuracy:", accuracy_score(y_test, predictions))
print("\nClassification Report:")
print(classification_report(y_test, predictions, zero_division=0))

# Save the complete pipeline
joblib.dump(model, "machine_learning/classifier.pkl")

print("\nModel saved successfully!")
