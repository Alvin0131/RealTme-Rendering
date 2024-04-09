#include <iostream>
#include <opencv2/opencv.hpp>
#include <chrono>  // for high_resolution_clock
#include <cmath>
#include <vector>
#include <omp.h>

using namespace std;
using namespace cv;



float determinant(float matrix[3][3]) {
    return matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[2][1] * matrix[1][2]) -
           matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0]) +
           matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0]);
}

float calculateCovarianceMatrixDeterminant(const Mat& src, const Point& center, int neighborhoodSize) {
    vector<Vec3f> colors;
    for (int y = max(center.y - neighborhoodSize / 2, 0); y <= min(center.y + neighborhoodSize / 2, src.rows - 1); ++y) {
        for (int x = max(center.x - neighborhoodSize / 2, 0); x <= min(center.x + neighborhoodSize / 2, src.cols - 1); ++x) {
            Vec3b color = src.at<Vec3b>(Point(x, y));
            colors.push_back(Vec3f(color[0], color[1], color[2]));
        }
    }

    Vec3f mean(0, 0, 0);
    for (const auto& color : colors) {
        mean += color;
    }
    mean /= static_cast<float>(colors.size());

    float covMatrix[3][3] = {0};
    for (const auto& color : colors) {
        Vec3f diff = color - mean;
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                covMatrix[i][j] += diff[i] * diff[j];
            }
        }
    }
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            covMatrix[i][j] /= colors.size();
        }
    }
    return determinant(covMatrix);
}
Mat extendImageBorders(const Mat& src, int borderSize) {
    Mat extended;
    copyMakeBorder(src, extended, borderSize, borderSize, borderSize, borderSize, BORDER_REFLECT);
    return extended;
}
// Adapted code for adaptive Gaussian filtering using the provided function
Mat adaptiveGaussianFilter(const Mat& src, int neighborhoodSize, double factorRatio) {
    int halfSize = neighborhoodSize / 2;
    // Extend borders of the source image
    Mat extendedSrc = extendImageBorders(src, halfSize);
    Mat extendedDst = extendedSrc.clone();
    
    for (int y = halfSize; y < extendedSrc.rows - halfSize; y++) {
        for (int x = halfSize; x < extendedSrc.cols - halfSize; x++) {
            float det = calculateCovarianceMatrixDeterminant(extendedSrc, Point(x, y), neighborhoodSize);
            double sigma = 0.5 + 1.0 / (1.0 + det * factorRatio);
            GaussianBlur(extendedSrc(Rect(x - halfSize, y - halfSize, neighborhoodSize, neighborhoodSize)),
                         extendedDst(Rect(x - halfSize, y - halfSize, neighborhoodSize, neighborhoodSize)),
                         Size(0, 0), sigma, sigma);
        }
    }

    // Crop the extended borders to get the final image
    Mat dst;
    extendedDst(Rect(halfSize, halfSize, src.cols, src.rows)).copyTo(dst);
    return dst;
}
    
cv::Mat adaptiveGaussianFilter(const cv::Mat& src, int neighborhoodSize, double factorRatio);

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <ImagePath>\n";
        return -1;
    }

    cv::Mat source = cv::imread(argv[1], cv::IMREAD_COLOR);
    if (source.empty()) {
        std::cout << "Error loading the image\n";
        return -1;
    }
    cv::imshow("Source Image", source);
    
    int cutWidth = 12; // Width to cut from the side

    auto begin = std::chrono::high_resolution_clock::now();
    const int iter = 1; // Adjusted for single adaptive filter application

    cv::Mat destination = adaptiveGaussianFilter(source, 5, 0.01); // Example usage with adaptiveGaussianFilter
    int width = destination.cols;
    int height = destination.rows;
    cv::Rect leftRect(0, 0, width - cutWidth, height); // Cut right side
    cv::Rect rightRect(cutWidth, 0, width - cutWidth, height); // Cut left side

    // Prepare the left and right eye images
    cv::Mat leftEye = destination(leftRect).clone();
    cv::Mat rightEye = destination(rightRect).clone();
    std::vector<cv::Mat> leftChannels(3);
    cv::split(leftEye, leftChannels);
    leftChannels[1] = cv::Mat::zeros(leftEye.size(), CV_8UC1); // Green
    leftChannels[0] = cv::Mat::zeros(leftEye.size(), CV_8UC1); // Blue
    cv::merge(leftChannels, leftEye);

    // Keep only cyan channels (green and blue) for the right eye image
    std::vector<cv::Mat> rightChannels(3);
    cv::split(rightEye, rightChannels);
    rightChannels[2] = cv::Mat::zeros(rightEye.size(), CV_8UC1); // Red (zero out)
    cv::merge(rightChannels, rightEye);
    cv::Mat anaglyph = leftEye + rightEye;

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - begin;

    cv::imshow("Processed Image", anaglyph);

    std::cout << "Total time: " << diff.count() << " s\n";
    std::cout << "Time for 1 iteration: " << diff.count()/iter << " s\n"; // Reporting for a single iteration
    std::cout << "IPS: " << iter/diff.count() << std::endl;

    cv::waitKey(0); // Wait for a key press before exiting
    return 0;
}
