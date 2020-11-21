#ifndef __IMAGE_AND_EXPOSURE_HPP__
#define __IMAGE_AND_EXPOSURE_HPP__


#include <cstring>
#include <iostream>

namespace ds_slam
{

class ImageAndExposure
{
public:
    // image with timestamp
    inline ImageAndExposure(int w_, int h_, double timestamp_ = 0) : w(w_), h(h_), timestamp(timestamp_)
    {
        image = new float[w * h];
        exposure_time = 1;
    }

    //
    inline ~ImageAndExposure()
    {
        delete[] image;
    }

    //
    inline void CopyMetaTo(ImageAndExposure &other)
    {
        other.exposure_time = exposure_time;
    }

    //
    inline ImageAndExposure *GetDeepCopy()
    {
        ImageAndExposure *img = new ImageAndExposure(w, h, timestamp);
        img->exposure_time = exposure_time;
        memcpy(img->image, image, w * h * sizeof(float));
        return img;
    }

    //
    float *image; // irradiance. between 0 and 256
    int w; // width 
    int h; // height;
    double timestamp;
    float exposure_time; // exposure time in ms.
};

} // namespace ds_slam
#endif