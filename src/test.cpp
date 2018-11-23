#include <opencv2/opencv.hpp>
#include "qrcode.h"

using namespace cv;

int test_qrcode_cv(cv::Mat src) {
#if (CV_VERSION_MAJOR > 3)
    cv::QRCodeDetector qrdc;
    std::string dec = qrdc.detectAndDecode(src);
    std::cout << "decode: " << dec << std::endl;
#endif
    return 0;
}

int test_qrcode(cv::Mat src) {
    pc::QRCodeDetector qrdc;
    std::string dec = qrdc.detectAndDecode(src);
    std::cout << "decode: " << dec << std::endl;
    return 0;
}

int main(int argc, char *argv[]) {
    Mat image = imread("../images/version_1_down.jpg");
    test_qrcode_cv(image);
    test_qrcode(image);
    return 0;
}