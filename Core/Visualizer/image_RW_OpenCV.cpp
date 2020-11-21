#include <opencv2/highgui/highgui.hpp>
#include "Visualizer/image_RW.h"


namespace ds_slam
{
namespace visualizer
{
MinimalImageB *ReadImageBW_8U(std::string filename)
{
    cv::Mat m = cv::imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
    if (m.rows * m.cols == 0)
    {
        printf("cv::imread could not read image %s! this may segfault. \n", filename.c_str());
        return 0;
    }
    if (m.type() != CV_8U)
    {
        printf("cv::imread did something strange! this may segfault. \n");
        return 0;
    }
    MinimalImageB *img = new MinimalImageB(m.cols, m.rows);
    memcpy(img->data, m.data, m.rows * m.cols);
    return img;
}

MinimalImageB3 *ReadImageRGB_8U(std::string filename)
{
    cv::Mat m = cv::imread(filename, CV_LOAD_IMAGE_COLOR);
    if (m.rows * m.cols == 0)
    {
        printf("cv::imread could not read image %s! this may segfault. \n", filename.c_str());
        return 0;
    }
    if (m.type() != CV_8UC3)
    {
        printf("cv::imread did something strange! this may segfault. \n");
        return 0;
    }
    MinimalImageB3 *img = new MinimalImageB3(m.cols, m.rows);
    memcpy(img->data, m.data, 3 * m.rows * m.cols);
    return img;
}

MinimalImage<unsigned short> *ReadImageBW_16U(std::string filename)
{
    cv::Mat m = cv::imread(filename, CV_LOAD_IMAGE_UNCHANGED);
    if (m.rows * m.cols == 0)
    {
        printf("cv::imread could not read image %s! this may segfault. \n", filename.c_str());
        return 0;
    }
    if (m.type() != CV_16U)
    {
        printf("readImageBW_16U called on image that is not a 16bit grayscale image. this may segfault. \n");
        return 0;
    }
    MinimalImage<unsigned short> *img = new MinimalImage<unsigned short>(m.cols, m.rows);
    memcpy(img->data, m.data, 2 * m.rows * m.cols);
    return img;
}

MinimalImageB *ReadStreamBW_8U(char *data, int numBytes)
{
    cv::Mat m = cv::imdecode(cv::Mat(numBytes, 1, CV_8U, data), CV_LOAD_IMAGE_GRAYSCALE);
    if (m.rows * m.cols == 0)
    {
        printf("cv::imdecode could not read stream (%d bytes)! this may segfault. \n", numBytes);
        return 0;
    }
    if (m.type() != CV_8U)
    {
        printf("cv::imdecode did something strange! this may segfault. \n");
        return 0;
    }
    MinimalImageB *img = new MinimalImageB(m.cols, m.rows);
    memcpy(img->data, m.data, m.rows * m.cols);
    return img;
}

void WriteImage(std::string filename, MinimalImageB *img)
{
    cv::imwrite(filename, cv::Mat(img->h, img->w, CV_8U, img->data));
}

void WriteImage(std::string filename, MinimalImageB3 *img)
{
    cv::imwrite(filename, cv::Mat(img->h, img->w, CV_8UC3, img->data));
}

void WriteImage(std::string filename, MinimalImageF *img)
{
    cv::imwrite(filename, cv::Mat(img->h, img->w, CV_32F, img->data));
}

void WriteImage(std::string filename, MinimalImageF3 *img)
{
    cv::imwrite(filename, cv::Mat(img->h, img->w, CV_32FC3, img->data));
}

} // namespace visualizer
} // namespace ds_slam