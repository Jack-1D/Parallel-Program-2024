#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>
#include <cmath>

const float EPSILON = 5e-3; // Define a tolerance for floating-point comparison

// Function to read a flat array of floats from a file
std::vector<float> readArrayFromFile(const std::string &filename, size_t &size) {
    std::ifstream infile(filename, std::ios::binary);
    if (!infile) {
        throw std::runtime_error("Unable to open file: " + filename);
    }

    infile.seekg(0, std::ios::end);
    size_t fileSize = infile.tellg();
    infile.seekg(0, std::ios::beg);

    size = fileSize / sizeof(float);
    std::vector<float> data(size);
    infile.read(reinterpret_cast<char *>(data.data()), fileSize);

    if (!infile) {
        throw std::runtime_error("Error reading file: " + filename);
    }

    return data;
}

int main() {
    try {
        size_t size1, size2;
        
        // Read arrays from files
        std::vector<float> array1 = readArrayFromFile("1.txt", size1);
        std::vector<float> array2 = readArrayFromFile("2.txt", size2);

        // Check if sizes match
        if (size1 != size2) {
            std::cerr << "Arrays have different sizes: " << size1 << " vs " << size2 << std::endl;
            return 1;
        }

        // Compare arrays
        bool differencesFound = false;
        for (size_t i = 0; i < size1; ++i) {
            if (std::fabs(array1[i] - array2[i]) > EPSILON) {
                if (!differencesFound) {
                    std::cout << "Differences found:\n";
                    differencesFound = true;
                }
                std::cout << "Position: " << i
                          << ", File1: " << std::setprecision(6) << array1[i]
                          << ", File2: " << std::setprecision(6) << array2[i]
                          << std::endl;
            }
        }

        if (!differencesFound) {
            std::cout << "No differences found." << std::endl;
        }
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}