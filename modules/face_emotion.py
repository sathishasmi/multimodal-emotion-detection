import torch
import cv2
from torchvision import transforms
from PIL import Image
from modules.model import EmotionCNN

# Emotion labels
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
def load_model():
    model = EmotionCNN().to(device)
    model.load_state_dict(torch.load("models/face_emotion_model.pth", map_location=device))
    model.eval()
    return model

# Transform
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Main function
def run_face_emotion():
    model = load_model()

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face_img = Image.fromarray(face)
            face_tensor = transform(face_img).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(face_tensor)
                _, predicted = torch.max(output, 1)
                emotion = emotion_labels[predicted.item()]

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, emotion, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow("Emotion Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

#  Run directly
if __name__ == "__main__":
    run_face_emotion()