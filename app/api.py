from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import torch
from PIL import Image
import io
from src.models.cnn_model import GuitarChordCNN
from src.utils.audio_utils import convert_audio_to_spectrogram

app = FastAPI()

# Load your model (you might want to do this more efficiently in a production setting)
model = GuitarChordCNN(num_classes=8)
model.load_state_dict(torch.load('models/guitar_chord_cnn.pth'))
model.eval()

@app.post("/predict/")
async def predict_chord(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    
    # Preprocess the image (implement this based on your model's requirements)
    preprocessed = preprocess_image(image)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(preprocessed)
        _, predicted = torch.max(outputs, 1)
    
    # Convert prediction to chord name (implement this based on your class mapping)
    chord = get_chord_name(predicted.item())
    
    return JSONResponse(content={"predicted_chord": chord})

@app.post("/predict_from_audio/")
async def predict_chord_from_audio(file: UploadFile = File(...)):
    contents = await file.read()
    spectrogram = convert_audio_to_spectrogram(contents)
    
    # Preprocess the spectrogram
    preprocessed = preprocess_spectrogram(spectrogram)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(preprocessed)
        _, predicted = torch.max(outputs, 1)
    
    chord = get_chord_name(predicted.item())
    
    return JSONResponse(content={"predicted_chord": chord})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)