import torch
import librosa
import numpy as np
import sounddevice as sd
import torch.nn as nn

# Emotion labels
emotion_labels = ['Neutral','Calm','Happy','Sad','Angry','Fear','Disgust']

# Device
device = torch.device("cpu")

# Model
class EmotionCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1,32,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*16*16,256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256,7)
        )

    def forward(self,x):
        return self.fc(self.conv(x))

# Load Model
model = EmotionCNN()
model.load_state_dict(
    torch.load("models/voice_emotion_model.pth", map_location=device)
)
model.to(device)
model.eval()

# Record Audio
def record_audio(duration=3, sr=22050):
    print("Recording...")
    audio = sd.rec(int(duration*sr), samplerate=sr, channels=1)
    sd.wait()
    return audio.flatten()

# Feature Extraction
def extract(audio):
    mel = librosa.feature.melspectrogram(y=audio, sr=22050, n_mels=128)
    mel = librosa.power_to_db(mel)
    mel = (mel - np.mean(mel)) / (np.std(mel)+1e-6)
    if mel.shape[1] < 128:
        mel = np.pad(mel, ((0,0),(0,128-mel.shape[1])))
    else:
        mel = mel[:, :128]
    return mel

# Live Loop
while True:
    audio = record_audio()
    mel = extract(audio)
    x = torch.tensor(mel).unsqueeze(0).unsqueeze(0).float().to(device)
    with torch.no_grad():
        output = model(x)
        _, pred = torch.max(output, 1)
        emotion = emotion_labels[pred.item()]
    print("Detected Emotion:", emotion)