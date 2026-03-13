import cv2
import torch
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import pickle
import argparse
import sys

def run_recognition():
    # 1. Setup Device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Running on device: {device}')

    # 2. Load Models
    print("Loading FaceNet model...")
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    
    print("Loading MTCNN...")
    mtcnn = MTCNN(keep_all=True, device=device)

    # 3. Load Classifier
    try:
        with open('svm_classifier.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('label_encoder.pkl', 'rb') as f:
            le = pickle.load(f)
        print("Classifier loaded successfully.")
    except FileNotFoundError:
        print("Error: Model files not found. Please run the training notebook first to generate 'svm_classifier.pkl' and 'label_encoder.pkl'.")
        sys.exit(1)

    # 4. Initialize Webcam
    cap = cv2.VideoCapture(0)
    
    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        sys.exit(1)
        
    print("Starting video stream. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
            
        # Convert to RGB for MTCNN/FaceNet
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        
        # Detect faces
        try:
            boxes, _ = mtcnn.detect(pil_img)
        except Exception as e:
            # Sometimes MTCNN fails on empty frames or specific errors
            print(f"Detection error: {e}")
            boxes = None

        if boxes is not None:
            for box in boxes:
                # Extract coordinates
                x1, y1, x2, y2 = [int(b) for b in box]
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Safe crop
                h, w, _ = frame.shape
                # padding could be added here if needed
                x1_c, y1_c = max(0, x1), max(0, y1)
                x2_c, y2_c = min(w, x2), min(h, y2)
                
                if x2_c > x1_c and y2_c > y1_c:
                    face_img = pil_img.crop((x1_c, y1_c, x2_c, y2_c))
                    face_img = face_img.resize((160, 160))
                    
                    # Preprocess for FaceNet
                    # Convert to tensor and normalize (whiten)
                    face_tensor = torch.tensor(np.array(face_img)).float()
                    
                    # (H, W, C) -> (C, H, W)
                    face_tensor = face_tensor.permute(2, 0, 1)
                    
                    # Standardize: (x - 127.5) / 128.0
                    face_tensor = (face_tensor - 127.5) / 128.0
                    
                    # Add batch dimension and move to device
                    face_tensor = face_tensor.unsqueeze(0).to(device)
                    
                    # Get embedding
                    input_embed = resnet(face_tensor).detach().cpu().numpy()
                    
                    # Classification
                    prediction = model.predict(input_embed)
                    prob = model.predict_proba(input_embed)
                    
                    name = le.inverse_transform(prediction)[0]
                    confidence = prob[0][prediction[0]]
                    
                    # Overlay Text
                    label_text = f"{name}: {confidence*100:.1f}%"
                    cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow('Face Recognition System', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    run_recognition()
