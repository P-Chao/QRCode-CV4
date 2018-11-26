/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#ifndef PC_QRCODE_H
#define PC_QRCODE_H

#include "opencv2/core.hpp"

/**
@defgroup objdetect Object Detection

Haar Feature-based Cascade Classifier for Object Detection
----------------------------------------------------------

The object detector described below has been initially proposed by Paul Viola @cite Viola01 and
improved by Rainer Lienhart @cite Lienhart02 .

First, a classifier (namely a *cascade of boosted classifiers working with haar-like features*) is
trained with a few hundred sample views of a particular object (i.e., a face or a car), called
positive examples, that are scaled to the same size (say, 20x20), and negative examples - arbitrary
images of the same size.

After a classifier is trained, it can be applied to a region of interest (of the same size as used
during the training) in an input image. The classifier outputs a "1" if the region is likely to show
the object (i.e., face/car), and "0" otherwise. To search for the object in the whole image one can
move the search window across the image and check every location using the classifier. The
classifier is designed so that it can be easily "resized" in order to be able to find the objects of
interest at different sizes, which is more efficient than resizing the image itself. So, to find an
object of an unknown size in the image the scan procedure should be done several times at different
scales.

The word "cascade" in the classifier name means that the resultant classifier consists of several
simpler classifiers (*stages*) that are applied subsequently to a region of interest until at some
stage the candidate is rejected or all the stages are passed. The word "boosted" means that the
classifiers at every stage of the cascade are complex themselves and they are built out of basic
classifiers using one of four different boosting techniques (weighted voting). Currently Discrete
Adaboost, Real Adaboost, Gentle Adaboost and Logitboost are supported. The basic classifiers are
decision-tree classifiers with at least 2 leaves. Haar-like features are the input to the basic
classifiers, and are calculated as described below. The current algorithm uses the following
Haar-like features:

![image](pics/haarfeatures.png)

The feature used in a particular classifier is specified by its shape (1a, 2b etc.), position within
the region of interest and the scale (this scale is not the same as the scale used at the detection
stage, though these two scales are multiplied). For example, in the case of the third line feature
(2c) the response is calculated as the difference between the sum of image pixels under the
rectangle covering the whole feature (including the two white stripes and the black stripe in the
middle) and the sum of the image pixels under the black stripe multiplied by 3 in order to
compensate for the differences in the size of areas. The sums of pixel values over a rectangular
regions are calculated rapidly using integral images (see below and the integral description).

To see the object detector at work, have a look at the facedetect demo:
<https://github.com/opencv/opencv/tree/master/samples/cpp/dbt_face_detection.cpp>

The following reference is for the detection part only. There is a separate application called
opencv_traincascade that can train a cascade of boosted classifiers from a set of samples.

@note In the new C++ interface it is also possible to use LBP (local binary pattern) features in
addition to Haar-like features. .. [Viola01] Paul Viola and Michael J. Jones. Rapid Object Detection
using a Boosted Cascade of Simple Features. IEEE CVPR, 2001. The paper is available online at
<http://research.microsoft.com/en-us/um/people/viola/Pubs/Detect/violaJones_CVPR2001.pdf>

@{
    @defgroup objdetect_c C API
@}
 */

typedef struct CvHaarClassifierCascade CvHaarClassifierCascade;

namespace pc
{

template<int N = 4>
class Quad
{
public:
    cv::Point2f p[N];  //point
    float a[N];    //

    float area() const {
        if (N < 3) { return 0; }
        float a = 0;
        for (size_t i = 0; i < N; ++i) {
            const Point2f& p0 = p[i];
            const Point2f& p1 = p[(i + 1) % N];
            a += p0.x * p1.y - p0.y * p1.x;
        }
        if (a < 0) { a = -a; }
        a *= float(0.5);
        return a;
    }
};

class QRQuad : public Quad<4> {
public:
    cv::Point2f center; // cros conner
    float areas;
    float l[4]; // length of line
    float a[4]; // angle of line
    float dir[4]; // direction of line
    bool selectSelf() {
        if (areas < 4) {
            return false;
        }
        if (diffAngle(a[0], a[2]) > 30) {
            return false;
        }
        if (diffAngle(a[1], a[3]) > 30) {
            return false;
        }
        float lmax = 0, lmin = 1e10;
        for (int i = 0; i < 4; ++i) {
            if (l[i] > lmax) {
                lmax = l[i];
            }
            if (l[i] < lmin) {
                lmin = l[i];
            }
        }
        if (lmax / lmin > 2) {
            return false;
        }
        return true;
    }
    bool selectSimilar(QRQuad& quad) {
        // centerlocation
        float rr = pointL2Norm(center, quad.center);
        if (rr > l[1] / 10) {
            return false;
        }
        float anglethresh = 8;
        rr = diffAngle(a[0], quad.a[0]);
        if (rr > 80) rr -= 90;
        if (rr < anglethresh) return false;
        rr = diffAngle(a[1], quad.a[1]);
        if (rr > 80) rr -= 90;
        if (rr < anglethresh) return false;
        rr = diffAngle(a[2], quad.a[2]);
        if (rr > 80) rr -= 90;
        if (rr < anglethresh) return false;
        rr = diffAngle(a[3], quad.a[3]);
        if (rr > 80) rr -= 90;
        if (rr < anglethresh) return false;
        rr = diffAngle(dir[0], quad.dir[0]);
        if (rr > 80) rr -= 90;
        if (rr < anglethresh) return false;
        rr = diffAngle(dir[1], quad.dir[1]);
        if (rr > 80) rr -= 90;
        if (rr < anglethresh) return false;
        rr = diffAngle(dir[2], quad.dir[2]);
        if (rr > 80) rr -= 90;
        if (rr < anglethresh) return false;
        rr = diffAngle(dir[3], quad.dir[3]);
        if (rr > 80) rr -= 90;
        if (rr < anglethresh) return false;
        // area
        rr = areas / quad.areas;
        if (rr < 4 || rr > 12) return false;
        return true;
    }
    static float pointL2Norm(cv::Point2f a, cv::Point2f b) {
        return sqrt((a.x - b.x)*(a.x - b.x) + (a.y - b.y)*(a.y - b.y));
    }
    static float lineAngle(cv::Point2f a, cv::Point2f b) {
        if (a.x == b.x) {
            return 90;
        }
        float radv = atan((b.y-a.y)/(b.x-a.x));
        return radv * 180 / CV_PI;
    }
    float diffAngle(float degtan1, float degtan2) {
        float rr = abs(degtan1 - degtan2);
        if (rr > 90) {
            return rr - 90;
        }
        return rr;
    }

    cv::Point2f intersectionLines(cv::Point2f a1, cv::Point2f a2, cv::Point2f b1, cv::Point2f b2)
    {
        cv::Point2f result_square_angle(
            ((a1.x * a2.y - a1.y * a2.x) * (b1.x - b2.x) -
            (b1.x * b2.y - b1.y * b2.x) * (a1.x - a2.x)) /
                ((a1.x - a2.x) * (b1.y - b2.y) -
            (a1.y - a2.y) * (b1.x - b2.x)),
                    ((a1.x * a2.y - a1.y * a2.x) * (b1.y - b2.y) -
            (b1.x * b2.y - b1.y * b2.x) * (a1.y - a2.y)) /
                        ((a1.x - a2.x) * (b1.y - b2.y) -
            (a1.y - a2.y) * (b1.x - b2.x))
        );
        return result_square_angle;
    }

    QRQuad(cv::Point_<float> points[4]) {
        p[0] = points[0];
        p[1] = points[1];
        p[2] = points[2];
        p[3] = points[3];
        center = intersectionLines(p[0], p[2], p[1], p[3]);
        areas = area();
        l[0] = pointL2Norm(p[0], p[1]);
        l[1] = pointL2Norm(p[2], p[1]);
        l[2] = pointL2Norm(p[2], p[3]);
        l[3] = pointL2Norm(p[0], p[3]);
        a[0] = lineAngle(p[0], p[1]);
        a[1] = lineAngle(p[2], p[1]);
        a[2] = lineAngle(p[2], p[3]);
        a[3] = lineAngle(p[0], p[3]);
        dir[0] = lineAngle(center, p[0]);
        dir[1] = lineAngle(center, p[1]);
        dir[2] = lineAngle(center, p[2]);
        dir[3] = lineAngle(center, p[3]);
    }
};

class CV_EXPORTS_W QRCodeDetector
{
public:
    CV_WRAP QRCodeDetector();
    ~QRCodeDetector();

    /** @brief sets the epsilon used during the horizontal scan of QR code stop marker detection.
     @param epsX Epsilon neighborhood, which allows you to determine the horizontal pattern
     of the scheme 1:1:3:1:1 according to QR code standard.
    */
    CV_WRAP void setEpsX(double epsX);
    /** @brief sets the epsilon used during the vertical scan of QR code stop marker detection.
     @param epsY Epsilon neighborhood, which allows you to determine the vertical pattern
     of the scheme 1:1:3:1:1 according to QR code standard.
     */
    CV_WRAP void setEpsY(double epsY);

    /** @brief Detects QR code in image and returns the quadrangle containing the code.
     @param img grayscale or color (BGR) image containing (or not) QR code.
     @param points Output vector of vertices of the minimum-area quadrangle containing the code.
     */
    CV_WRAP bool detect(cv::InputArray img, cv::OutputArray points) const;

    /** @brief Decodes QR code in image once it's found by the detect() method.
     Returns UTF8-encoded output string or empty string if the code cannot be decoded.

     @param img grayscale or color (BGR) image containing QR code.
     @param points Quadrangle vertices found by detect() method (or some other algorithm).
     @param straight_qrcode The optional output image containing rectified and binarized QR code
     */
    CV_WRAP std::string decode(cv::InputArray img, cv::InputArray points, cv::OutputArray straight_qrcode = cv::noArray());

    /** @brief Both detects and decodes QR code

     @param img grayscale or color (BGR) image containing QR code.
     @param points opiotnal output array of vertices of the found QR code quadrangle. Will be empty if not found.
     @param straight_qrcode The optional output image containing rectified and binarized QR code
     */
    CV_WRAP std::string detectAndDecode(cv::InputArray img, cv::OutputArray points=cv::noArray(),
                                        cv::OutputArray straight_qrcode = cv::noArray());
protected:
    struct Impl;
    cv::Ptr<Impl> p;
};

//! @} objdetect
}

#include "opencv2/objdetect/detection_based_tracker.hpp"

#endif
