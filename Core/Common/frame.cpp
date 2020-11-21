#include "Common/frame.hpp"

namespace ds_slam
{

Frame::Frame()
{
    id = 0;
    poseValid = true;
    camToWorld = SE3();
    AffG2L = AffLight(0, 0);
    timestamp = 0;
    marginalizedAt = -1;
    movedByOpt = 0;
    statistics_outlierResOnThis = statistics_goodResOnThis = 0;
    trackingRef = 0;
    camToTrackingRef = SE3();

    is_kf = false;
    fx = fy = cx = cy = 0;
    fxi = fyi = cxi = cyi = 0;
}

Frame::~Frame()
{
    for(std::vector<Point *>::iterator iter = points.begin(); iter != points.end(); )
    {
        delete *iter;
        iter = points.erase(iter);
    }
    points.clear();
}

void Frame::RemoveOutlier(Point *outPoint)
{
    for(std::vector<Point *>::iterator iter = points.begin(); iter != points.end(); )
    {
        if(*iter == outPoint)
        {
            //delete *iter; // delete outlier point
            //iter = points.erase(iter);

            (*iter)->status = Point::PointStatus::INVALID;
            break;
        }
        else
            iter++;
    }
}

void Frame::CreateCvMat(Eigen::Vector3f *dI, int width, int height)
{
    image.create(height, width, CV_8UC1);
    for(int i = 0; i < height; i++)
    {
        for(int j = 0; j < width; j++)
        {
            image.at<uchar>(i, j) = dI[i * width + j][0];
        }
    }

    //cv::imwrite("test_image.jpg", image);
}

}