// this function label images 
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
    std::string imageFilePath = "train/apple/img";
    std::string jsonFilePath = "train/apple/label";

    std::vector<cv::Mat> all_crops;

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

                    if (roi.width > 0 && roi.height > 0) {
                        cv::Mat crop = image(roi).clone(); // clone per copia indipendente
                        all_crops.push_back(crop);
                    }
                
                }
                //std::cout << "Reading JSON from: " << filename << std::endl;
                //cv::imshow("Image", image);
                //cv::waitKey(0);
            }
        }
        std::cout << "Numero totale di ritagli: " << all_crops.size() << std::endl;
        // Mostra i ritagli 
        /* for (size_t i = 0; i < all_crops.size(); ++i) {
            cv::imshow("Crop " + std::to_string(i), all_crops[i]);
            cv::waitKey(0); // mostra ogni 0.5 secondi
        } */
    } catch(const std::filesystem::filesystem_error& e) {
        std::cerr << "Error reading: " << e.what() << std::endl;
    }


    return 0;
}