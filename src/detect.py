import cv2
import sys
from pathlib import Path

def main():
    if len(sys.argv) != 3:
        print(f"Uso: {sys.argv[0]} <modello.xml> <cartella_immagini>")
        sys.exit(1)

    cascade_path = sys.argv[1]
    folder_path  = sys.argv[2]

    # Carica il classificatore
    cascade = cv2.CascadeClassifier(cascade_path)
    if cascade.empty():
        print(f"Errore: impossibile caricare il file XML: {cascade_path}")
        sys.exit(1)

    # Itera su tutti i file della cartella
    for entry in Path(folder_path).iterdir():
        file_path = str(entry)

        # Carica immagine
        image = cv2.imread(file_path)
        if image is None:
            print(f"Errore: impossibile aprire {file_path}")
            continue

        # Conversione in grigio
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        # Rilevamento
        objects = cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=8,
            flags=0,
            minSize=(100, 100)
        )

        # Disegna rettangoli
        for (x, y, w, h) in objects:
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Mostra risultato
        cv2.imshow("Rilevamento", image)
        cv2.waitKey(0)  # Premi un tasto per passare allimmagine successiva

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
