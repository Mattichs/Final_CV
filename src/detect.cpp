#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <filesystem> // C++17

using namespace cv;
using namespace std;

int main(int argc, char** argv) {
    if (argc != 3) {
        cout << "Uso: " << argv[0] << " <modello.xml> <cartella_immagini>" << endl;
        return -1;
    }

    string cascadePath = argv[1];
    string folderPath  = argv[2];

    // Carica il classificatore
    CascadeClassifier cascade;
    if (!cascade.load(cascadePath)) {
        cerr << "Errore: impossibile caricare il file XML: " << cascadePath << endl;
        return -1;
    }


    // Itera su tutti i file della cartella
    for (const auto& entry : std::filesystem::directory_iterator(folderPath)) {
        string filePath = entry.path().string();

        // Carica immagine
        Mat image = imread(filePath);
        if (image.empty()) {
            cerr << "Errore: impossibile aprire " << filePath << endl;
            continue;
        }

        // Conversione in grigio
        Mat gray;
        cvtColor(image, gray, COLOR_BGR2GRAY);
        equalizeHist(gray, gray);

        // Rilevamento
        vector<Rect> objects;
        cascade.detectMultiScale(gray, objects, 1.1, 5, 0, Size(150, 150));

        // Disegna rettangoli
        for (size_t i = 0; i < objects.size(); i++) {
            rectangle(image, objects[i], Scalar(0, 255, 0), 2);
        }

        // Mostra risultato
        imshow("Rilevamento", image);
        waitKey(0); // Premi un tasto per passare all'immagine successiva

    }

    return 0;
}
