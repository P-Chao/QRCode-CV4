/*********************************************************************
 * This file is distributed as part of the C++ port of the APRIL tags
 * library. The code is licensed under GPLv2.
 *
 * Original author: Edwin Olson <ebolson@umich.edu>
 * C++ port and modifications: Matt Zucker <mzucker1@swarthmore.edu>
 ********************************************************************/

#include "DebugImage.h"
#include "AprilTypes.h"

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#ifdef _WIN32
#include "Profiler.h"
int gettimeofday(struct timeval *tp, void *tzp)
{
    time_t clock;
    struct tm tm;
    SYSTEMTIME wtm;
    GetLocalTime(&wtm);
    tm.tm_year = wtm.wYear - 1900;
    tm.tm_mon = wtm.wMonth - 1;
    tm.tm_mday = wtm.wDay;
    tm.tm_hour = wtm.wHour;
    tm.tm_min = wtm.wMinute;
    tm.tm_sec = wtm.wSecond;
    tm.tm_isdst = -1;
    clock = mktime(&tm);
    tp->tv_sec = clock;
    tp->tv_usec = wtm.wMilliseconds * 1000;
    return (0);
}
#endif

cv::Mat rescaleImageIntensity(const cv::Mat& img, ScaleType type) {

  cv::Mat rval;

  if (type == ScaleNone) {

    if (img.depth() == CV_8U) {
      rval = img.clone();
    } else {
      double fmin, fmax;
      cv::minMaxLoc(img, &fmin, &fmax);
      if (fmax - fmin <= 1) {
        rval = cv::Mat(img.rows, img.cols, CV_8U);
        cv::convertScaleAbs(img, rval, 255);
      } else {
        img.convertTo(rval, CV_8U);
      }
    }

  } else {
  
    cv::Mat fsrc;
    
    if (img.depth() != at::REAL_IMAGE_TYPE) {
      img.convertTo(fsrc, at::REAL_IMAGE_TYPE);
    } else {
      fsrc = img;
    }
    
    cv::Mat tmp;
    
    if (type == ScaleMinMax) {
      double fmin, fmax;
      cv::minMaxLoc(fsrc, &fmin, &fmax);
      tmp = 255*((fsrc-fmin)/(fmax-fmin));
    } else {
      at::real fmag = cv::norm(fsrc, cv::NORM_INF);
      tmp = 127 + 127*fsrc/fmag;
    }

    tmp.convertTo(rval, CV_8U);

  }

  return rval;

}

void labelImage(cv::Mat& img, const std::string& text) {

  cv::putText(img, text, cv::Point(4, 20),
              cv::FONT_HERSHEY_SIMPLEX,
              0.6, CV_RGB(0,0,0), 3, CV_AA);

  cv::putText(img, text, cv::Point(4, 20),
              cv::FONT_HERSHEY_SIMPLEX,
              0.6, CV_RGB(255,255,255), 1, CV_AA);

}

int resizeToDisplay(const cv::Mat& src, cv::Mat& dst,
                    int width, int height) {

  int fw = width  / src.cols;
  int fh = height / src.rows;

  int f = std::min(fw,fh);

  if (f > 1) {

    cv::Mat tmp;
    cv::resize(src, tmp, cv::Size(0,0), f, f, CV_INTER_NN);
    dst = tmp;
    return f;

  } else {

    dst = src.clone();
    return 1;

  }


}

void labelAndWaitForKey(const std::string& window,
                        const std::string& text,
                        const cv::Mat& img,
                        ScaleType type,
                        bool resize,
                        int w, int h) {
  
  cv::Mat rescaled = rescaleImageIntensity(img, type);

  if (resize) { resizeToDisplay(rescaled, rescaled, w, h); }

  labelImage(rescaled, text);
  
  cv::imshow(window, rescaled);
  cv::waitKey();


}
