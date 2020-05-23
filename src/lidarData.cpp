
#include <iostream>
#include <algorithm>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "lidarData.hpp"


using namespace std;

// remove Lidar points based on min. and max distance in X, Y and Z
void cropLidarPoints(std::vector<LidarPoint>& lidarPoints, float minX, float maxX, float maxY, float minZ, float maxZ, float minR)
{
    std::vector<LidarPoint> newLidarPts;
    for (auto it = lidarPoints.begin(); it != lidarPoints.end(); ++it) {

        if ((*it).x >= minX && (*it).x <= maxX && (*it).z >= minZ && (*it).z <= maxZ && (*it).z <= 0.0 && abs((*it).y) <= maxY && (*it).r >= minR)  // Check if Lidar point is outside of boundaries
        {
            newLidarPts.push_back(*it);
        }
    }

    lidarPoints = newLidarPts;
}



// Load Lidar points from a given location and store them in a vector
void loadLidarFromFile(vector<LidarPoint>& lidarPoints, string filename)
{
    // allocate 4 MB buffer (only ~130*4*4 KB are needed)
    unsigned long num = 1000000;
    float* data = (float*)malloc(num * sizeof(float));

    // pointers
    float* px = data + 0;
    float* py = data + 1;
    float* pz = data + 2;
    float* pr = data + 3;

    // load point cloud
    FILE* stream;
    stream = fopen(filename.c_str(), "rb");
    num = fread(data, sizeof(float), num, stream) / 4;

    for (int32_t i = 0; i < num; i++) {
        LidarPoint lpt;
        lpt.x = *px; lpt.y = *py; lpt.z = *pz; lpt.r = *pr;
        lidarPoints.push_back(lpt);
        px += 4; py += 4; pz += 4; pr += 4;
    }
    fclose(stream);
}


void showLidarTopview(BoundingBox* currBB, cv::Size worldSize, cv::Size imageSize, bool bWait)
{
    // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

    // plot Lidar points into image
    int top = 1e8, left = 1e8, bottom = 0.0, right = 0.0;
    float xwmin, ywmin, ywmax;
    xwmin = ywmin = ywmax = 1e8;
    for (auto it = currBB->lidarPoints.begin(); it != currBB->lidarPoints.end(); ++it)
    {

        float xw = (*it).x; // world position in m with x facing forward from sensor
        float yw = (*it).y; // world position in m with y facing left from sensor

        xwmin = xwmin < xw ? xwmin : xw;
        ywmin = ywmin < yw ? ywmin : yw;
        ywmax = ywmax < yw ? ywmax : yw;

        int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
        int x = (-yw * imageSize.height / worldSize.height) + imageSize.width / 2;

        top = top < y ? top : y;
        left = left < x ? left : x;
        bottom = bottom > y ? bottom : y;
        right = right > x ? right : x;

        cv::circle(topviewImg, cv::Point(x, y), 5, cv::Scalar(0, 0, 255), -1);
    }

    cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom), cv::Scalar(0, 0, 0), 2);

    //augment object with key data
    char str1[200], str2[200];
    sprintf(str1, "id=%d, #pts=%d", currBB->boxID, (int)currBB->lidarPoints.size());
    cv::putText(topviewImg, str1, cv::Point2f(left - 250, bottom + 50), cv::FONT_ITALIC, 2, cv::Scalar(255, 0, 0));
    sprintf(str2, "xmin=%2.2f m, #yw=%2.2f m", xwmin, ywmax - ywmin);
    cv::putText(topviewImg, str2, cv::Point2f(left - 250, bottom + 125), cv::FONT_ITALIC, 2, cv::Scalar(255, 0, 0));

    // plot distance markers
    float lineSpacing = 2.0; // gap between distance markers
    int nMarkers = floor(worldSize.height / lineSpacing);
    for (size_t i = 0; i < nMarkers; ++i)
    {
        int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
    }

    // display image
    string windowName = "Top-View Perspective of LiDAR data";
    cv::namedWindow(windowName, 2);
    cv::imshow(windowName, topviewImg);
    if (bWait)
    {
        cv::waitKey(0); // wait for key to be pressed
    }
}

void showLidarImgOverlay(cv::Mat& img, std::vector<LidarPoint>& lidarPoints, cv::Mat& P_rect_xx, cv::Mat& R_rect_xx, cv::Mat& RT, cv::Mat* extVisImg)
{
    // init image for visualization
    cv::Mat visImg;
    if (extVisImg == nullptr)
    {
        visImg = img.clone();
    }
    else
    {
        visImg = *extVisImg;
    }

    cv::Mat overlay = visImg.clone();

    // find max. x-value
    double maxVal = 0.0;
    for (auto it = lidarPoints.begin(); it != lidarPoints.end(); ++it)
    {
        maxVal = maxVal < it->x ? it->x : maxVal;
    }

    cv::Mat X(4, 1, cv::DataType<double>::type);
    cv::Mat Y(3, 1, cv::DataType<double>::type);
    for (auto it = lidarPoints.begin(); it != lidarPoints.end(); ++it) {

        X.at<double>(0, 0) = it->x;
        X.at<double>(1, 0) = it->y;
        X.at<double>(2, 0) = it->z;
        X.at<double>(3, 0) = 1;

        Y = P_rect_xx * R_rect_xx * RT * X;
        cv::Point pt;
        pt.x = Y.at<double>(0, 0) / Y.at<double>(0, 2);
        pt.y = Y.at<double>(1, 0) / Y.at<double>(0, 2);

        float val = it->x;
        int red = min(255, (int)(255 * abs((val - maxVal) / maxVal)));
        int green = min(255, (int)(255 * (1 - abs((val - maxVal) / maxVal))));
        cv::circle(overlay, pt, 5, cv::Scalar(0, green, red), -1);
    }

    float opacity = 0.6;
    cv::addWeighted(overlay, opacity, visImg, 1 - opacity, 0, visImg);

    // return augmented image or wait if no image has been provided
    if (extVisImg == nullptr)
    {
        string windowName = "LiDAR data on image overlay";
        cv::namedWindow(windowName, 3);
        cv::imshow(windowName, visImg);
        cv::waitKey(0); // wait for key to be pressed
    }
    else
    {
        extVisImg = &visImg;
    }
}