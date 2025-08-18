"""  
Questo file serve a creare un file di testo che contiene le informazioni sulle immagini e le loro etichette.
Il file di output sarà utilizzato per l'addestramento di un modello di rilevamento degli oggetti.
"""


import json
import os

# Percorsi delle cartelle
image_dir = "train/apple/img"
label_dir = "train/apple/label"
output_txt_path = "positives.txt"

lines = []

# Scorri tutte le immagini nella cartella delle immagini
for filename in os.listdir(image_dir):
    if not filename.lower().endswith(".jpg"):
        continue

    image_path = os.path.join(image_dir, filename)
    json_filename = filename.replace(".jpg", ".json")
    json_path = os.path.join(label_dir, json_filename)

    if not os.path.exists(json_path):
        print(f"File JSON mancante per {filename}, salto...")
        continue

    with open(json_path, "r") as f:
        data = json.load(f)

    objects = data.get("objects", [])
    num_objects = len(objects)

    if num_objects == 0:
        continue

    coords = []
    for obj in objects:
        points = obj.get("points", {})
        exterior = points.get("exterior", [])
        if len(exterior) != 2:
            continue

        x1, y1 = exterior[0]
        x2, y2 = exterior[1]
        x = min(x1, x2)
        y = min(y1, y2)
        w = abs(x2 - x1)
        h = abs(y2 - y1)
        coords.extend([str(x), str(y), str(w), str(h)])

    line = f"{image_path} {num_objects} " + " ".join(coords)
    lines.append(line)

# Salva nel file di output
with open(output_txt_path, "w") as f:
    f.write("\n".join(lines))

print(f"File 'positives.txt' generato con {len(lines)} immagini.")
