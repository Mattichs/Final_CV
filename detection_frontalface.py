# detection_only.py
import cv2
from pathlib import Path

def main():
    images_dir = Path("test_images")   # folder with your test images
    out_dir = Path("Detected_Faces")
    out_dir.mkdir(exist_ok=True)

    # Load Haar cascade (assumes XML is in the same folder or installed in cv2.data)
    cascade_path = "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        print(f"❌ Error loading cascade: {cascade_path}")
        return

    # Loop through images
    for img_path in sorted(images_dir.iterdir()):
        if img_path.suffix.lower() not in [".jpg", ".png", ".jpeg", ".bmp"]:
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"⚠️ Could not open {img_path.name}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=6,
            minSize=(40, 40)
        )

        # Draw rectangles
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Save output
        out_path = out_dir / img_path.name
        cv2.imwrite(str(out_path), img)
        print(f"✅ Processed {img_path.name} → {out_path}")

if __name__ == "__main__":
    main()
