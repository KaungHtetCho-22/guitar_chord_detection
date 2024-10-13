import torch
import librosa
import numpy as np
from model import GuitarChordCNN

def load_model(model_path, num_classes):
    model = GuitarChordCNN(num_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def audio_to_melspectrogram(audio_path, sr=22050, n_mels=128, n_fft=2048, hop_length=512):
    # Load audio file
    y, sr = librosa.load(audio_path, sr=sr)
    
    # Compute mel-spectrogram
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
    
    # Convert to log scale
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Normalize
    mel_spec_normalized = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min())
    
    return mel_spec_normalized

def preprocess_for_model(mel_spec, target_width=224):
    # Convert to tensor and add batch and channel dimensions
    mel_tensor = torch.from_numpy(mel_spec).unsqueeze(0).unsqueeze(0).float()
    
    # Resize to target width
    mel_tensor_resized = torch.nn.functional.interpolate(mel_tensor, size=(mel_spec.shape[0], target_width), mode='bilinear', align_corners=False)
    
    return mel_tensor_resized

def predict(model, mel_tensor, class_names):
    with torch.no_grad():
        outputs = model(mel_tensor)
        _, predicted = torch.max(outputs, 1)
        return class_names[predicted.item()]

def main():
    model_path = 'weights/guitar_chord_cnn.pth'
    num_classes = 8 
    class_names = ['Am', 'Bb', 'Bdim', 'C', 'Dm', 'Em', 'F', 'G']  
    model = load_model(model_path, num_classes)
    
    # Example usage
    audio_path = 'data/raw/Test/Am/Am_AcousticGuitar_RodrigoMercador_1.wav'
    mel_spec = audio_to_melspectrogram(audio_path)
    mel_tensor = preprocess_for_model(mel_spec)
    predicted_chord = predict(model, mel_tensor, class_names)
    print(f"Predicted chord: {predicted_chord}")

if __name__ == "__main__":
    main()