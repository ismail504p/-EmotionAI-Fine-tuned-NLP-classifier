# EmotionAI — Fine-Tuned DistilBERT Emotion Detection

A Streamlit interface for the fine-tuned DistilBERT emotion-classification model included in this project.

## Features
- Single-sentence emotion prediction
- Confidence score and probability chart
- Batch prediction from CSV
- Download enriched prediction results
- Model/device status and model information
- Responsive dashboard-style interface

## Emotion labels
The included model predicts the six classes used by the DAIR.AI Emotion dataset:
`sadness`, `joy`, `love`, `anger`, `fear`, and `surprise`.

## Project structure
```text
Emotion_Detection_Streamlit/
├── app.py
├── requirements.txt
├── README.md
└── fine_tuned_emotion_model/
    ├── config.json
    ├── model.safetensors
    ├── tokenizer.json
    ├── tokenizer_config.json
    ├── special_tokens_map.json
    └── vocab.txt
```

## Run on Windows (PowerShell)

1. Extract the project ZIP.
2. Open PowerShell in the extracted project folder.
3. Create and activate a virtual environment:

```powershell
py -3.11 -m venv .venv
.venv\\Scripts\\Activate.ps1
```

4. Install dependencies:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

5. Start Streamlit:

```powershell
streamlit run app.py
```

The app will open at `http://localhost:8501`.

## Run on Google Colab
This project is designed to run locally. For Colab, install the requirements, upload/extract this project folder into `/content`, then run:

```python
!pip install -r /content/Emotion_Detection_Streamlit/requirements.txt
!streamlit run /content/Emotion_Detection_Streamlit/app.py &>/content/streamlit.log &
```

Colab does not expose the local Streamlit port as a public URL by default. Use a supported tunnel if you need to access the app in a browser.

## Notes
- The app loads the model from the included `fine_tuned_emotion_model` folder; no Hugging Face download is needed at runtime.
- First startup can take a little time because PyTorch and Transformers initialize.
- CPU inference works; an NVIDIA CUDA GPU is used automatically when available.
- The model's confidence is a probability distribution from the classifier, not a guarantee of correctness.
- Do not use the output as a clinical or mental-health diagnosis.
