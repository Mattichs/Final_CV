# detection_only.py
import cv2
from pathlib import Path

def main():
    images_dir = Path("test_images")   # folder with your test images
    out_dir = Path("Detected_AllFaces")
    out_dir.mkdir(exist_ok=True)

    
    frontal_path = "haarcascade_frontalface_default.xml"
    side_path = "haarcascade_profileface.xml"
    frontal_cascade = cv2.CascadeClassifier(frontal_path)
    side_cascade = cv2.CascadeClassifier(side_path)

    if frontal_cascade.empty():
        print(f"❌ Could not load {frontal_path}"); return
    if side_cascade.empty():
        print(f"❌ Could not load {side_path}"); return

    params = dict(scaleFactor=1.1, minNeighbors=6, minSize=(20,20))
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

        # Frontal faces (green)
        for (x, y, w, h) in frontal_cascade.detectMultiScale(gray, **params):
            cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)

        # Side profiles (blue, left facing)
        for (x, y, w, h) in side_cascade.detectMultiScale(gray, **params):
            cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)


        # Side profiles (red, right facing → detect on flipped image)
        flipped = cv2.flip(gray, 1)
        w_img = gray.shape[1]
        for (x, y, w, h) in side_cascade.detectMultiScale(flipped, **params):
            # map back coordinates
            x_real = w_img - x - w
            cv2.rectangle(img, (x_real,y), (x_real+w,y+h), (0,0,255), 2)

        # Save output
        out_path = out_dir / img_path.name
        cv2.imwrite(str(out_path), img)
        print(f"✅ Processed {img_path.name} → {out_path}")

if __name__ == "__main__":
    main()
