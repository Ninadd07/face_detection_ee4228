import cv2
from insightface.app import FaceAnalysis

def main():
    app = FaceAnalysis(providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=-1, det_size=(640, 640))

    img = cv2.imread("test_2.jpg")

    if img is None:
        print("Could not read image. Make sure test_2.jpg is in this folder.")
        return

    faces = app.get(img)
    print(f"Detected {len(faces)} face(s).")

    for face in faces:
        box = face.bbox.astype(int)
        x1, y1, x2, y2 = box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow("Faces", img)

    # --- Close on 'q' (Mac-friendly) ---
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # 'q' or ESC
            break

    cv2.destroyAllWindows()
    # extra waitKey calls help on macOS to actually close the window
    for _ in range(4):
        cv2.waitKey(1)

if __name__ == "__main__":
    main()
