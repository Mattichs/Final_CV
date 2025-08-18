/* // this function label images 

// INPUT: 
// - image file path
// - json file path
// OUTPUT: 
// -show labeled images on the screen

#include <string>
#include <filesystem>
#include <iostream>
#include <fstream>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/ml.hpp>

#include "json.hpp"

using json = nlohmann::json;


using namespace cv;
using namespace cv::ml;


int main() {
    // Example input paths
    std::string imageFilePath = "train/strawberry/img";
    std::string jsonFilePath = "train/strawberry/label";

    std::vector<cv::Mat> all_crops;

    // Cartella output per i ritagli
    std::string outDir = "negative";
    int cropCounter = 0;
    int maxCount = 50;

    try {
        for(const auto& entry : std::filesystem::directory_iterator(imageFilePath)) {
            if (entry.is_regular_file()) {
                // read image
                cv::Mat image = cv::imread(entry.path().string());
                if(image.empty()) {
                    std::cerr << "Could not read the image: " << entry.path() << std::endl;
                    continue;
                }

                // read json file
                std::string from = "jpg";
                std::string to = "json";
                std::string filename = entry.path().filename().string();
                size_t pos = filename.find(from);
                if (pos != std::string::npos) {
                    filename.replace(pos, from.length(), to);
                }
                std::ifstream file(jsonFilePath + "/" + filename);
                json j;
                file >> j;
                // leggere numero elementi dell'attributo objects e trovare i quadrati e stamparli sull'immagine
                // ho angolo in alto a sinistra e angolo in basso a destra
                                
                for(const auto& object : j["objects"]) {
                    int x1 = object["points"]["exterior"][0][0];
                    int y1 = object["points"]["exterior"][0][1];
                    int x2 = object["points"]["exterior"][1][0];
                    int y2 = object["points"]["exterior"][1][1];
                    
                    //cv::rectangle(image, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(255, 0, 0), 2);
                    cv::Rect roi(cv::Point(x1, y1), cv::Point(x2, y2));
                
                    roi &= cv::Rect(0, 0, image.cols, image.rows);

                    if (roi.width > 0 && roi.height > 0 && cropCounter < maxCount) {
                        cv::Mat crop = image(roi).clone(); // clone per copia indipendente
                        all_crops.push_back(crop);

                        // Nome file di output
                        std::string outName = outDir + "/crop_" + std::to_string(cropCounter++) + ".jpg";

                        // Salva ritaglio
                        cv::imwrite(outName, crop);
                    }
                
                }
                //std::cout << "Reading JSON from: " << filename << std::endl;
                //cv::imshow("Image", image);
                //cv::waitKey(0);
            }
        }
        std::cout << "Numero totale di ritagli: " << all_crops.size() << std::endl;
        // Mostra i ritagli 
         for (size_t i = 0; i < all_crops.size(); ++i) {
            cv::imshow("Crop " + std::to_string(i), all_crops[i]);
            cv::waitKey(0); // mostra ogni 0.5 secondi
        } 
    } catch(const std::filesystem::filesystem_error& e) {
        std::cerr << "Error reading: " << e.what() << std::endl;
    }


    return 0;
} */
#include <opencv2/opencv.hpp>
#include "json.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>

namespace fs = std::filesystem;
using json = nlohmann::json;

int main() {
    std::string trainPath = "train";
    std::string excludeFolder = "strawberry"; // frutto da escludere
    int maxROIs = 150;
    std::string outDir = "temp";

    for (const auto &fruitDir : fs::directory_iterator(trainPath)) {
        
        if (!fruitDir.is_directory()) continue;

        std::string fruitName = fruitDir.path().filename().string();
        if (fruitName == excludeFolder) {
            std::cout << "Salto cartella: " << fruitName << "\n";
            continue;
        }

        std::cout << "Processo cartella: " << fruitName << "\n";
        int roiCount = 0;
        fs::path labelDir = fruitDir.path() / "label";
        fs::path imgDir   = fruitDir.path() / "img";

        if (!fs::exists(labelDir) || !fs::exists(imgDir)) {
            std::cerr << "Cartelle mancanti in " << fruitName << "\n";
            continue;
        }

        for (const auto &jsonFile : fs::directory_iterator(labelDir)) {
            std::cout << roiCount <<"\n";
            if (roiCount >= maxROIs) break;
            if (jsonFile.path().extension() != ".json") continue;

            std::ifstream in(jsonFile.path());
            if (!in.is_open()) continue;

            json j;
            in >> j;
            in.close();

            // Trova il file immagine corrispondente
            fs::path imgPath = imgDir / jsonFile.path().filename().replace_extension(".jpg");
            cv::Mat img = cv::imread(imgPath.string());
            /* cv:imshow("Image", img);
            cv::waitKey(0); */
            if (img.empty()) {
                std::cerr << "Errore apertura immagine: " << imgPath << "\n";
                continue;
            }
            std::cout << "" << j["annotations"].size() << "\n";
            // Leggi tutte le annotazioni
            for (const auto &object : j["objects"]) {
                if (roiCount >= maxROIs) break;
                
                int x1 = object["points"]["exterior"][0][0];
                int y1 = object["points"]["exterior"][0][1];
                int x2 = object["points"]["exterior"][1][0];
                int y2 = object["points"]["exterior"][1][1];

                //cv::rectangle(image, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(255, 0, 0), 2);
                cv::Rect roi(cv::Point(x1, y1), cv::Point(x2, y2));
            
                roi &= cv::Rect(0, 0, img.cols, img.rows);

                if (roi.width > 0 && roi.height > 0) {
                    cv::Mat crop = img(roi).clone(); // clone per copia indipendente

                    // Nome file di output
                    std::string outName = outDir + "/crop_"  + fruitName + std::to_string(roiCount++) + ".jpg";

                    // Salva ritaglio
                    cv::imwrite(outName, crop);
                }
            }
        }
    }

    return 0;
}
