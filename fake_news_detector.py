import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import gradio as gr

# Load and preprocess data
df = pd.read_csv("train.tsv", sep='\t', header=None, names=[
    "id", "label", "statement", "subject", "speaker", "job", "state",
    "party", "barely_true", "half_true", "mostly_true", "false", 
    "pants_on_fire", "context"
])

df = df[["statement", "label"]]

# Simplify labels to 'real' and 'fake'
def simplify_label(label):
    if label in ['false', 'barely-true', 'pants-fire']:
        return 'fake'
    else:
        return 'real'

df['label'] = df['label'].apply(simplify_label)

# Drop missing values
df.dropna(inplace=True)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    df['statement'], df['label'], test_size=0.2, random_state=42
)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train logistic regression model
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# Function to predict from user input
def predict_fake_news(text):
    vec = vectorizer.transform([text])
    prediction = model.predict(vec)[0]
    return f"Prediction: This is likely **{prediction.upper()}** news."

iface = gr.Interface(
    fn=predict_fake_news,
    inputs=gr.Textbox(lines=3, placeholder="Enter a news statement..."),
    outputs="text",
    title="Fake News Detector",
    description="Enter a news claim or statement to see if it's likely fake or real.",
)

# Add custom CSS for a more polished UI
iface.css = """
.gradio-container {
    background-color: #f7f7f7;
    padding: 30px;
    border-radius: 8px;
}
.gradio-header {
    font-family: 'Arial', sans-serif;
    font-size: 24px;
    color: #333;
    text-align: center;
}
.gradio-footer {
    font-size: 14px;
    text-align: center;
    color: #777;
}
"""


iface.launch()
