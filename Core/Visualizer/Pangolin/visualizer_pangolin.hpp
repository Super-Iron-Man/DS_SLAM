#ifndef __VIEWER_PANGOLIN_HPP__
#define __VIEWER_PANGOLIN_HPP__

#include <vector>
#include <map>
#include <deque>
#include <pangolin/pangolin.h>
#include <boost/thread.hpp>

#include "Utils/minimal_image.hpp"
#include "Visualizer/visualizer_3D.hpp"
#include "keyframe_display.hpp"


namespace ds_slam
{

class FrameHessian;
class CalibHessian;
class Frame;


namespace visualizer
{

class KeyFrameDisplay;

struct GraphConnection
{
    KeyFrameDisplay *from;
    KeyFrameDisplay *to;
    int fwdMarg, bwdMarg, fwdAct, bwdAct;
};

class ViewerPangolin : public Visualizer3D
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    ViewerPangolin(int w, int h, bool startRunThread = true);
    virtual ~ViewerPangolin();

    void Run();
    void Close();

    void AddImageToDisplay(std::string name, MinimalImageB3 *image);
    void ClearAllImagesToDisplay();

    // Visualizer3D Function
    virtual void PublishGraph(const std::map<uint64_t,
                                             Eigen::Vector2i,
                                             std::less<uint64_t>,
                                             Eigen::aligned_allocator<std::pair<const uint64_t, Eigen::Vector2i>>> &connectivity) override;
    virtual void PublishKeyframes(std::vector<FrameHessian *> &frames, bool final, CalibHessian *HCalib) override;
    virtual void PublishCamPose(Frame *frame, CalibHessian *HCalib) override;
    virtual void PushLiveFrame(FrameHessian *image) override;
    virtual void PushDepthImage(MinimalImageB3 *image) override;
    virtual bool NeedPushDepthImage() override;
    virtual void Join() override;
    virtual void Reset() override;

private:
    bool needReset;
    void ResetInternal();
    void DrawConstraints();

    boost::thread runThread;
    bool running;
    int w; 
    int h;

    // images rendering
    boost::mutex openImagesMutex;
    MinimalImageB3 *internalVideoImg;
    MinimalImageB3 *internalKFImg;
    MinimalImageB3 *internalResImg;
    bool videoImgChanged, kfImgChanged, resImgChanged;

    // 3D model rendering
    boost::mutex model3DMutex;
    KeyFrameDisplay *currentCam;
    std::vector<KeyFrameDisplay *> keyframes;
    std::vector<Vec3f, Eigen::aligned_allocator<Vec3f>> allFramePoses;
    std::map<int, KeyFrameDisplay *> keyframesByKFID;
    std::vector<GraphConnection, Eigen::aligned_allocator<GraphConnection>> connections;

    // render settings
    bool settings_showKFCameras;
    bool settings_showCurrentCamera;
    bool settings_showTrajectory;
    bool settings_showFullTrajectory;
    bool settings_showActiveConstraints;
    bool settings_showAllConstraints;

    // parameter settings
    float settings_scaledVarTH;
    float settings_absVarTH;
    int settings_pointCloudMode;
    float settings_minRelBS;
    int settings_sparsity;

    // timings
    struct timeval last_track;
    struct timeval last_map;

    std::deque<float> lastNTrackingMs;
    std::deque<float> lastNMappingMs;
};

} // namespace visualizer
} // namespace ds_slam

#endif