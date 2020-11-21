#ifndef __UNDISTORT_HPP__
#define __UNDISTORT_HPP__


#include "Utils/image_and_exposure.hpp"
#include "Utils/minimal_image.hpp"
#include "Utils/num_type.h"



namespace ds_slam
{

class PhotometricUndistorter
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    /**
     * @brief construct a new photometric undistorter
     * @param gammaFile gamma file
     * @param noiseImage noise image(discard)
     * @param vignetteImage vignette image
     * @param width
     * @param height 
     */
    PhotometricUndistorter(std::string gammaFile, std::string noiseImage, std::string vignetteImage, int width, int height);

    ~PhotometricUndistorter();

    // removes readout noise, and converts to irradiance.
    // affine normalizes values to 0 <= I < 256.
    // raw irradiance = a*I + b.
    // output will be written in [output].
    template <typename T>
    void ProcessFrame(T *image_in, float exposure_time, float factor = 1);

    // undistort image
    void UnMapFloatImage(float *image);

    // get gamma map
    float *GetG()
    {
        if (!valid)
            return 0;
        else
            return G;
    };

    // undistorted image
    ImageAndExposure *output;

private:
    float G[256 * 256]; // gamma map
    int GDepth;
    float *vignetteMap;
    float *vignetteMapInv;
    int w, h;
    bool valid;
};

class Undistort
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    virtual ~Undistort();

    /**
     * @brief create undistorter
     * @param calibFilename calibration filename 
     * @param gammaFilename gamma filename 
     * @param vignetteFilename vignette filename 
     * @return undistorter ptr
     */
    static Undistort *GetUndistorterForFile(std::string calibFilename, std::string gammaFilename, std::string vignetteFilename);

    /**
     * @brief distort coordinates
     * @param in_x input x coordinates
     * @param in_y input y coordinates
     * @param out_x output x coordinates
     * @param out_y output y coordinates
     * @param n 
     */
    virtual void DistortCoordinates(float *in_x, float *in_y, float *out_x, float *out_y, int n) const = 0;

    /**
     * @brief undistort image
     * @param image_raw input raw image
     * @param exposure time
     * @param timestamp 
     * @param factor 
     * @return undistorted image
     */
    template <typename T>
    ImageAndExposure *UndistortImage(const MinimalImage<T> *image_raw, float exposure = 0, double timestamp = 0, float factor = 1) const;

    /**
     * @brief load photometric calibration
     * @param file 
     * @param noiseImage 
     * @param vignetteImage 
     */
    void LoadPhotometricCalibration(std::string file, std::string noiseImage, std::string vignetteImage);


    inline const Mat33 GetK() const { return K; };
    inline const Eigen::Vector2i GetSize() const { return Eigen::Vector2i(w, h); };
    inline const VecX GetOriginalParameter() const { return parsOrg; };
    inline const Eigen::Vector2i GetOriginalSize() { return Eigen::Vector2i(wOrg, hOrg); };
    inline bool IsValid() { return valid; };

    // photometric undistorter
    PhotometricUndistorter *photometricUndist;

protected:
    void ApplyBlurNoise(float *img) const;
    void MakeOptimalK_crop();
    void MakeOptimalK_full();
    void ReadFromFile(const char *configFileName, int nPars, std::string prefix = "");

    int w, h, wOrg, hOrg, wUp, hUp;
    int upsampleUndistFactor;
    Mat33 K; //intrinsic matrix
    VecX parsOrg;
    bool valid;
    bool passthrough;

    // remap
    float *remapX;
    float *remapY;
};

class UndistortFOV : public Undistort
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    UndistortFOV(const char *configFileName, bool noprefix);
    ~UndistortFOV();
    void DistortCoordinates(float *in_x, float *in_y, float *out_x, float *out_y, int n) const;
};

class UndistortRadTan : public Undistort
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    UndistortRadTan(const char *configFileName, bool noprefix);
    ~UndistortRadTan();
    void DistortCoordinates(float *in_x, float *in_y, float *out_x, float *out_y, int n) const;
};

class UndistortEquidistant : public Undistort
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    UndistortEquidistant(const char *configFileName, bool noprefix);
    ~UndistortEquidistant();
    void DistortCoordinates(float *in_x, float *in_y, float *out_x, float *out_y, int n) const;
};

class UndistortPinhole : public Undistort
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    UndistortPinhole(const char *configFileName, bool noprefix);
    ~UndistortPinhole();
    void DistortCoordinates(float *in_x, float *in_y, float *out_x, float *out_y, int n) const;

private:
    float inputCalibration[8];
};

class UndistortKB : public Undistort
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    UndistortKB(const char *configFileName, bool noprefix);
    ~UndistortKB();
    void DistortCoordinates(float *in_x, float *in_y, float *out_x, float *out_y, int n) const;
};

} // namespace ds_slam

#endif