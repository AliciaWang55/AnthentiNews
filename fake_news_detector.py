import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Load dataset (make sure 'train.tsv' is in the same folder)
df = pd.read_csv("train.tsv", sep='\t', header=None, names=[
    "id", "label", "statement", "subject", "speaker", "job", "state",
    "party", "barely_true", "half_true", "mostly_true", "false", 
    "pants_on_fire", "context"
])


# Show original labels
print("Original labels:", df['label'].unique())

# Keep only relevant columns
df = df[["statement", "label"]]

# Simplify labels to 'real' and 'fake'
def simplify_label(label):
    if label in ['false', 'barely-true', 'pants-fire']:
        return 'fake'
    else:
        return 'real'

df['label'] = df['label'].apply(simplify_label)

# Show simplified label distribution
print("Simplified labels:", df['label'].value_counts())

# Drop any rows with missing values
df.dropna(inplace=True)

# Split data into train/test sets
X_train, X_test, y_train, y_test = train_test_split(
    df['statement'], df['label'], test_size=0.2, random_state=42)

# Vectorize text using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train logistic regression model
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# Predict and evaluate
y_pred = model.predict(X_test_vec)
print(classification_report(y_test, y_pred))

# Custom prediction function
def predict_fake_news(text):
    vec = vectorizer.transform([text])
    prediction = model.predict(vec)[0]
    print(f"\nPrediction: This is likely '{prediction.upper()}' news.")

# Test with custom input
predict_fake_news("Joe Biden says he created 5 million jobs in 2021.")

import gradio as gr

def gradio_predict(text):
    vec = vectorizer.transform([text])
    prediction = model.predict(vec)[0]
    return f"This is likely '{prediction.upper()}' news."

iface = gr.Interface(fn=gradio_predict, inputs="text", outputs="text", title="Fake News Detector")
iface.launch()
