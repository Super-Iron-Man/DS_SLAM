#ifndef __KEYFRAME_DISPLAY_HPP__
#define __KEYFRAME_DISPLAY_HPP__

#include <sstream>
#include <fstream>
#include <Eigen/Core>
#include <pangolin/pangolin.h>

#include "Utils/num_type.h"


namespace ds_slam
{

class CalibHessian;
class FrameHessian;
class Frame;

namespace visualizer
{

template <int ppp>
struct InputPointSparse
{
    float u;
    float v;
    float idpeth;
    float idepth_hessian;
    float relObsBaseline;
    int numGoodRes;
    unsigned char color[ppp];
    unsigned char status;
};

struct MyVertex
{
    float point[3];
    unsigned char color[4];
};

// stores a pointcloud associated to a Keyframe.
class KeyFrameDisplay
{

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    KeyFrameDisplay();
    ~KeyFrameDisplay();

    // copies points from KF over to internal buffer,
    // keeping some additional information so we can render it differently.
    void SetFromKF(FrameHessian *fh, CalibHessian *HCalib);

    // copies points from KF over to internal buffer,
    // keeping some additional information so we can render it differently.
    void SetFromF(Frame *fs, CalibHessian *HCalib);

    // copies & filters internal data to GL buffer for rendering. if nothing to do: does nothing.
    bool RefreshPC(bool canRefresh, float scaledTH, float absTH, int mode, float minBS, int sparsity);

    // renders cam & pointcloud.
    void DrawCam(float lineWidth = 1, float *color = 0, float sizeFactor = 1);
    void DrawPC(float pointSize);

    int id;
    bool active;
    SE3 camToWorld;

    inline bool operator<(const KeyFrameDisplay &other) const
    {
        return (id < other.id);
    }

private:
    float fx, fy, cx, cy;
    float fxi, fyi, cxi, cyi;
    int width, height;

    float my_scaledTH, my_absTH, my_scale;
    int my_sparsifyFactor;
    int my_displayMode;
    float my_minRelBS;
    bool needRefresh;

    int numSparsePoints;
    int numSparseBufferSize;
    InputPointSparse<MAX_RES_PER_POINT> *originalInputSparse;

    bool bufferValid;
    int numGLBufferPoints;
    int numGLBufferGoodPoints;
    pangolin::GlBuffer vertexBuffer;
    pangolin::GlBuffer colorBuffer;
};

} // namespace visualizer
} // namespace ds_slam

#endif