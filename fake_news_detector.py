import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import gradio as gr

# Sample prediction function
def predict_fake_news(text):
    vec = vectorizer.transform([text])
    prediction = model.predict(vec)[0]
    return f"Prediction: This is likely **{prediction.upper()}** news."

# --- Load & Prepare Data ---
df = pd.read_csv("train.tsv", sep='\t', header=None, names=[
    "id", "label", "statement", "subject", "speaker", "job", "state",
    "party", "barely_true", "half_true", "mostly_true", "false", 
    "pants_on_fire", "context"
])
df = df[["statement", "label"]]
df['label'] = df['label'].apply(lambda l: 'fake' if l in ['false', 'barely-true', 'pants-fire'] else 'real')
df.dropna(inplace=True)

X_train, X_test, y_train, y_test = train_test_split(
    df['statement'], df['label'], test_size=0.2, random_state=42
)
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = LogisticRegression()
model.fit(X_train_vec, y_train)

# --- Build Gradio Interface ---


with gr.Blocks(css="""
body {
    background-color: #f5f5dc;
}

#custom-button {
    background-color: #d32f2f !important;
    color: white !important;
    border-radius: 8px !important;
    font-size: 18px !important;
    padding: 12px 24px !important;
}

.box textarea {
    height: 200px !important;
    font-size: 16px !important;
}

.output-class {
    height: 200px;
    font-size: 16px;
    white-space: pre-wrap;
}

.output-class > label + div {
    background: none !important;
    border: none !important;
    box-shadow: none !important;
}

#top-image {
    display: block;
    margin-left: auto;
    margin-right: auto;
    width: 200px;
}
""") as demo:

    gr.Image("logo.png", elem_id="top-image", show_label=False)
    gr.Markdown("## 📰 AuthentiNews: Fake News Detector")
    gr.Markdown("Enter a news statement below. The model will predict if it's likely REAL or FAKE.")

    with gr.Row():
        with gr.Column():
            input_box = gr.Textbox(label="Input Statement", lines=10, elem_classes=["box"])
            clear_btn = gr.Button("Clear")
        with gr.Column():
            output_box = gr.Textbox(label="Prediction", lines=10, interactive=False, elem_classes=["box"])

    with gr.Row():
        submit_btn = gr.Button("Check News", elem_id="custom-button")

    # Button functionality
    submit_btn.click(fn=predict_fake_news, inputs=input_box, outputs=output_box)
    clear_btn.click(fn=lambda: ("", ""), inputs=None, outputs=[input_box, output_box])

demo.launch()
