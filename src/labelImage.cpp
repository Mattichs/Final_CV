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
#include <opencv2/opencv.hpp>

#include "json.hpp"

using json = nlohmann::json;

int main() {
    // Example input paths
    std::string imageFilePath = "../train/apple/img";
    std::string jsonFilePath = "../train/apple/label";

    try {
        for(const auto& entry : std::filesystem::directory_iterator(imageFilePath)) {
            if (entry.is_regular_file()) {
                // read image
                cv::Mat image = imread(entry.path().string(), cv::IMREAD_GRAYSCALE);
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
                    //std::string classTitle = object["classTitle"];
                    cv::rectangle(image, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(255, 0, 0), 2);
                }
                std::cout << "Reading JSON from: " << filename << std::endl;
                cv::imshow("Image", image);
                cv::waitKey(0);
            }
            /* if (entry.is_regular_file()) {
                std::ifstream file(entry.path());
                json j;
                file >> j;
                std::cout << "File: " << entry.path().string() << std::endl;
                std::cout << "Objects: " << j["objects"][0]["classTitle"] << std::endl;
            } */
        }
    } catch(const std::filesystem::filesystem_error& e) {
        std::cerr << "Error reading: " << e.what() << std::endl;
    }

    return 0;
}