
# Guitar Chord Detection

This project implements a deep learning model for detecting guitar chords from audio input.

## Project Structure

```
guitar_chord_detection/
│
├── data/
│   ├── raw/                 # Raw audio files
│   ├── processed/           # Processed spectrograms
│   └── augmented/           # Augmented data
│
├── src/
│   ├── data/
│   │   ├── preprocess.py    # Data preprocessing scripts
│   │   └── augment.py       # Data augmentation scripts
│   ├── models/
│   │   ├── cnn_model.py     # CNN model architecture
│   │   └── train.py         # Model training script
│   ├── utils/
│   │   ├── audio_utils.py   # Audio processing utilities
│   │   └── visualization.py # Visualization utilities
│   └── inference.py         # Inference script
│
├── notebooks/
│   ├── exploratory_data_analysis.ipynb
│   └── model_evaluation.ipynb
│
├── tests/
│   ├── test_preprocess.py
│   ├── test_model.py
│   └── test_inference.py
│
├── config/
│   └── model_config.yaml    # Model and training configurations
│
├── app/
│   ├── api.py               # FastAPI app for serving predictions
│   └── frontend/            # Frontend code for web app
│
├── models/                  # Saved model checkpoints
│
├── logs/                    # Training logs
│
├── requirements.txt         # Project dependencies
└── README.md                # This file
```

## Setup

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Prepare your data in the `data/raw` directory
4. Run preprocessing: `python src/data/preprocess.py`
5. Train the model: `python src/models/train.py`
6. Make predictions: `python src/inference.py`

## API Usage

To start the API server:

```bash
uvicorn app.api:app --reload
```

Then you can make POST requests to `http://localhost:8000/predict_chord/` with an audio file to get chord predictions
