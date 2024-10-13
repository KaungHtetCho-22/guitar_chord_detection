import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def load_audio(file_path, sr=22050):
    """Load an audio file."""
    audio, _ = librosa.load(file_path, sr=sr)
    return audio

def audio_to_melspectrogram(audio, sr=22050, n_mels=128, n_fft=2048, hop_length=512):
    """Convert audio to mel spectrogram."""
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec_db

def process_audio_files(input_dir, output_dir, dataset_type):
    """Process all audio files in the input directory and save mel spectrograms."""
    os.makedirs(output_dir, exist_ok=True)
    
    for root, _, files in os.walk(input_dir):
        for file in tqdm(files, desc=f"Processing {dataset_type} files"):
            if file.endswith(('.wav', '.mp3')):
                file_path = os.path.join(root, file)
                
                # Load audio
                audio = load_audio(file_path)
                
                # Convert to mel spectrogram
                mel_spec = audio_to_melspectrogram(audio)
                
                # Save mel spectrogram
                output_path = os.path.join(output_dir, os.path.splitext(file)[0] + '.npy')
                np.save(output_path, mel_spec)
                
                # Optionally, save a plot of the mel spectrogram
                plt.figure(figsize=(10, 4))
                librosa.display.specshow(mel_spec, x_axis='time', y_axis='mel', sr=22050, fmax=8000)
                plt.colorbar(format='%+2.0f dB')
                plt.title(f'Mel spectrogram - {file}')
                plt.tight_layout()
                plt.savefig(output_path.replace('.npy', '.png'))
                plt.close()

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    raw_data_dir = os.path.join(base_dir, 'data', 'raw')
    processed_data_dir = os.path.join(base_dir, 'data', 'processed')

    # Process training data
    train_input_dir = os.path.join(raw_data_dir, 'Training')
    train_output_dir = os.path.join(processed_data_dir, 'train')
    process_audio_files(train_input_dir, train_output_dir, 'Training')

    # Process test data
    test_input_dir = os.path.join(raw_data_dir, 'Test')
    test_output_dir = os.path.join(processed_data_dir, 'test')
    process_audio_files(test_input_dir, test_output_dir, 'Test')

if __name__ == "__main__":
    main()