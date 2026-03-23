import cv2
import numpy as np
from insightface.app import FaceAnalysis

def main():
    app = FaceAnalysis(providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=-1, det_size=(640, 640))

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        faces = app.get(frame)

        for face in faces:
            box = face.bbox.astype(int)
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # NEW: get normalized embedding (ArcFace feature vector)
            emb = face.normed_embedding  # or face.embedding depending on version [web:116][web:121]
            # Print its length once
            print("Embedding dim:", emb.shape if isinstance(emb, np.ndarray) else len(emb))
            # To avoid spamming, break after first face
            break

        cv2.imshow("Webcam - press q to quit", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    for _ in range(4):
        cv2.waitKey(1)

if __name__ == "__main__":
    main()
