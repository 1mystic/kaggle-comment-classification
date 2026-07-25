### 1. Do You Need to Rerun to Save the Trained Model?

* **If you didn't explicitly run `joblib.dump()` or `pickle.dump()` during the Kaggle execution:** Yes, unfortunately, you will need to rerun it. Python objects in RAM vanish once the notebook session terminates. Kaggle only persists files that were explicitly written to disk in `/kaggle/working/` during the run.
* **If you wrote code to save the `.joblib` / `.pkl` file inside `/kaggle/working/` during that 8-hour run:** You can simply download the file from the **Data / Output** tab of your committed notebook version without rerunning anything.

---

### 2. How to Efficiently Save the Pipeline for Hugging Face

When deploying a text classification model (especially an ensemble with TF-IDF vectorizers and feature transformers), you should save **the entire pipeline** (preprocessing + vectorizer + ensemble models) into a single bundle so live predictions don't require re-implementing feature extraction logic.

#### Recommended Code Snippet

Add this at the end of your training notebook:

```python
import joblib

# Bundle everything needed for inference into a dictionary
model_artifact = {
    'vectorizer': tfidf_vectorizer,       # Your fitted TF-IDF
    'feature_pipeline': feature_pipeline, # Any numeric/date scaler/encoders
    'models': final_ensemble_models,      # Linear + LightGBM models/weights
    'class_names': CLASS_NAMES            # {0: 'Normal', 1: 'Offensive', ...}
}

# Compress to keep file size small for fast loading on Hugging Face Spaces
joblib.dump(model_artifact, 'comment_classifier_pipeline.joblib', compress=3)

```

> **Efficiency Tip:** Using `compress=3` (or `compress=('gzip', 3)`) drastically reduces file size for high-dimensional TF-IDF matrices without noticeable loading overhead.

---

### 3. How to Avoid Another 8-Hour Waiting Period

If you have to rerun, speed it up significantly with these optimizations:

* **Parallelize LightGBM & Linear Models:** Set `n_jobs=-1` across all `LogisticRegression`, `ComplementNB`, and `lgb.LGBMClassifier` estimators to utilize all Kaggle CPU cores.
* **Pre-calculate & Save Intermediate Features:** Save the vectorized matrices (`csr_matrix`) to disk using `scipy.sparse.save_npz('X_train_tfidf.npz', X_tfidf)` so re-training hyperparameter tuning takes minutes instead of hours.
* **Switch to Kaggle GPU:** LightGBM supports GPU acceleration (`device='gpu'`), which can speed up tree building by $5\times\text{--}10\times$.

---

### 4. Live Web UI & Deployment on Hugging Face Spaces

The fastest way to deploy a live UI and get a working API for a Chrome Extension is using **Gradio** hosted on **Hugging Face Spaces**.

#### Step 1: Create a Hugging Face Space

1. Go to [Hugging Face Spaces](https://huggingface.co/spaces) $\rightarrow$ **Create new Space**.
2. Select **Gradio** as the SDK (free CPU tier is sufficient for inference).

#### Step 2: `app.py` Code (Gradio Interface)

```python
import gradio as gr
import joblib
import numpy as np
import pandas as pd

# Load saved pipeline
artifact = joblib.load('comment_classifier_pipeline.joblib')
vectorizer = artifact['vectorizer']
models = artifact['models']
class_names = artifact['class_names']

def analyze_comment(text):
    if not text.strip():
        return "Please enter a comment.", {}, ""
    
    # 1. Transform text
    X_vec = vectorizer.transform([text])
    
    # 2. Get probabilities (ensemble weighted average)
    probs = np.zeros(len(class_names))
    for model, weight in models:
        probs += weight * model.predict_proba(X_vec)[0]
    
    predicted_class_id = int(np.argmax(probs))
    predicted_label = class_names[predicted_class_id]
    
    # Format confidence dictionary for Gradio
    confidence_dict = {class_names[i]: float(probs[i]) for i in range(len(probs))}
    
    # 3. Top influential words in input
    feature_names = np.array(vectorizer.get_feature_names_out())
    nonzero_indices = X_vec.nonzero()[1]
    tokens = [feature_names[i] for i in nonzero_indices]
    
    top_words_str = ", ".join(tokens) if tokens else "None (Out of vocabulary)"
    
    return predicted_label, confidence_dict, top_words_str

# Build UI
demo = gr.Interface(
    fn=analyze_comment,
    inputs=gr.Textbox(lines=3, placeholder="Type a comment here..."),
    outputs=[
        gr.Textbox(label="Predicted Category"),
        gr.Label(label="Confidence Scores"),
        gr.Textbox(label="Key Detected Tokens")
    ],
    title="Comment Toxicity & Category Classifier",
    description="Live inference engine predicting comment classifications using a stacked ensemble."
)

if __name__ == "__main__":
    demo.launch()

```

---

### 5. Building a Chrome Extension with the Live Model

Gradio automatically exposes a REST API endpoint for your Hugging Face Space at `https://<your-username>-<space-name>.hf.space/api/predict`.

#### Chrome Extension Manifest (`manifest.json`)

```json
{
  "manifest_version": 3,
  "name": "Comment Classifier",
  "version": "1.0",
  "permissions": ["activeTab"],
  "action": {
    "default_popup": "popup.html"
  }
}

```

#### Popup Logic (`popup.js`)

```javascript
document.getElementById("analyze-btn").addEventListener("click", async () => {
    const text = document.getElementById("comment-input").value;
    
    const response = await fetch("https://YOUR_HF_USERNAME-YOUR_SPACE.hf.space/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ data: [text] })
    });

    const result = await response.json();
    // Gradio returns array corresponding to output components
    const [category, confidences, topWords] = result.data;

    document.getElementById("result").innerText = `Category: ${category}`;
});

```