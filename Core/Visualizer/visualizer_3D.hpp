#ifndef __VIEWER_3D_HPP__
#define __VIEWER_3D_HPP__


#include <vector>
#include <string>
#include <map>

#include "Utils/num_type.h"
#include "Utils/minimal_image.hpp"

namespace cv
{
class Mat;
}

namespace ds_slam
{

class FrameHessian;
class CalibHessian;
class Frame;

namespace visualizer
{

/* ======================= Some typical usecases: ===============
 *
 * (1) always get the pose of the most recent frame:
 *     -> Implement [PublishCamPose].
 *
 * (2) always get the depthmap of the most recent keyframe
 *     -> Implement [pushDepthImageFloat] (use inverse depth in [image], and pose / frame information from [KF]).
 *
 * (3) accumulate final model
 *     -> Implement [PublishKeyframes] (skip for final!=false), and accumulate frames.
 *
 * (4) get evolving model in real-time
 *     -> Implement [PublishKeyframes] (Update all frames for final==false).
 *
 *
 *
 *
 * ==================== How to use the structs: ===================
 * [Frame]: minimal struct kept for each frame ever tracked.
 *      ->camToWorld = camera to world transformation
 *      ->poseValid = false if [camToWorld] is invalid (only happens for frames during initialization).
 *      ->trackingRef = Shell of the frame this frame was tracked on.
 *      ->id = ID of that frame, starting with 0 for the very first frame.
 *
 *      ->incoming_id = ID passed into [AddActiveFrame( ImageAndExposure* image, int id )].
 *	->timestamp = timestamp passed into [AddActiveFrame( ImageAndExposure* image, int id )] as image.timestamp.
 *
 * [FrameHessian]
 *      ->immaturePoints: contains points that have not been "activated" (they do however have a depth initialization).
 *      ->pointHessians: contains active points.
 *      ->pointHessiansMarginalized: contains marginalized points.
 *      ->pointHessiansOut: contains outlier points.
 *
 *      ->frameID: incremental ID for keyframes only.
 *      ->shell: corresponding [Frame] struct.
 *
 *
 * [CalibHessian]
 *      ->fxl(), fyl(), cxl(), cyl(): get optimized, most recent (pinhole) camera intrinsics.
 *
 *
 * [PointHessian]
 * 	->u,v: pixel-coordinates of point.
 *      ->idepth_scaled: inverse depth of point.
 *                       DO NOT USE [idepth], since it may be scaled with [SCALE_IDEPTH] ... however that is currently set to 1 so never mind.
 *      ->host: pointer to host-frame of point.
 *      ->status: current status of point (ACTIVE=0, INACTIVE, OUTLIER, OOB, MARGINALIZED)
 *      ->numGoodResiduals: number of non-outlier residuals supporting this point (approximate).
 *      ->maxRelBaseline: value roughly proportional to the relative baseline this point was observed with (0 = no baseline).
 *                        points for which this value is low are badly contrained.
 *      ->idepth_hessian: hessian value (inverse variance) of inverse depth.
 *
 * [ImmaturePoint]
 * 	->u,v: pixel-coordinates of point.
 *      ->idepth_min, idepth_max: the initialization sais that the inverse depth of this point is very likely
 *        between these two thresholds (their mean being the best guess)
 *      ->host: pointer to host-frame of point.
 */


class Visualizer3D
{
public:
    Visualizer3D() {}

    virtual ~Visualizer3D() {}


    /* Usage:
     * Called once after each new Keyframe is inserted & optimized.
     * [connectivity] contains for each frame-frame pair the number of [0] active residuals in between them,
     * and [1] the number of marginalized reisduals between them.
     * frame-frame pairs are encoded as HASH_IDX = [(int)hostFrameKFID << 32 + (int)targetFrameKFID].
     * the [***frameKFID] used for hashing correspond to the [FrameHessian]->frameID.
     *
     * Calling:
     * Always called, no overhead if not used.
     */
    virtual void PublishGraph(const std::map<uint64_t,
                                             Eigen::Vector2i,
                                             std::less<uint64_t>,
                                             Eigen::aligned_allocator<std::pair<const uint64_t, Eigen::Vector2i>>> &connectivity) {}

    /* Usage:
    * Called after each new Keyframe is inserted & optimized, with all keyframes that were part of the active window during
    * that optimization in [frames] (with final=false). Use to access the new frame pose and points.
    * Also called just before a frame is marginalized (with final=true) with only that frame in [frames]; at that point,
    * no further updates will ever occur to it's optimized values (pose & idepth values of it's points).
    *
    * If you want always all most recent values for all frames, just use the [final=false] calls.
    * If you only want to get the final model, but don't care about it being delay-free, only use the
    * [final=true] calls, to save compute.
    *
    * Calling:
    * Always called, negligible overhead if not used.
    */
    virtual void PublishKeyframes(std::vector<FrameHessian *> &frames,
                                  bool final,
                                  CalibHessian *HCalib) {}

    /* Usage:
    * Called once for each tracked frame, with the real-time, low-delay frame pose.
    *
    * Calling:
    * Always called, no overhead if not used.
    */
    virtual void PublishCamPose(Frame *frame, CalibHessian *HCalib) {}

    /* Usage:
    * Called once for each new frame, before it is tracked (i.e., it doesn't have a pose yet).
    *
    * Calling:
    * Always called, no overhead if not used.
    */
    virtual void PushLiveFrame(FrameHessian *image) {}

    /* called once after a new keyframe is created, with the color-coded, forward-warped inverse depthmap for that keyframe,
    * which is used for initial alignment of future frames. Meant for visualization.
    *
    * Calling:
    * Needs to prepare the depth image, so it is only called if [needPushDepthImage()] returned true.
    */
    virtual void PushDepthImage(MinimalImageB3 *image) {}


    virtual bool NeedPushDepthImage() { return false; }

    /* Usage:
    * called once after a new keyframe is created, with the forward-warped inverse depthmap for that keyframe.
    * (<= 0 for pixels without inv. depth value, >0 for pixels with inv. depth value)
    *
    * Calling:
    * Always called, almost no overhead if not used.
    */
    virtual void PushDepthImageFloat(MinimalImageF *image, FrameHessian *KF) {}

    /* call on Finish */
    virtual void Join() {}

    /* call on reset */
    virtual void Reset() {}

};

} // namespace visualizer
} // namespace ds_slam

#endif