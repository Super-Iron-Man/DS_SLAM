#ifndef __VIEWER_SIMPLE_HPP__
#define __VIEWER_SIMPLE_HPP__


#include <boost/thread.hpp>
#include "Utils/minimal_image.hpp"
#include "Visualizer/visualizer_3D.hpp"
#include "Common/frame.hpp"
#include "Common/hessian_blocks.hpp"


namespace ds_slam
{

class FrameHessian;
class CalibHessian;
class Frame;


namespace visualizer
{

class ViewerSample : public Visualizer3D
{
public:
    inline ViewerSample()
    {
        printf("OUT: Created ViewerSample\n");
    }

    virtual ~ViewerSample()
    {
        printf("OUT: Destroyed ViewerSample\n");
    }

    virtual void PublishGraph(const std::map<uint64_t,
                                             Eigen::Vector2i,
                                             std::less<uint64_t>,
                                             Eigen::aligned_allocator<std::pair<const uint64_t, Eigen::Vector2i>>> &connectivity) override
    {
        printf("OUT: got graph with %d edges\n", (int)connectivity.size());

        int maxWrite = 5;

        for (const std::pair<uint64_t, Eigen::Vector2i> &p : connectivity)
        {
            int idHost = p.first >> 32;
            int idTarget = p.first & ((uint64_t)0xFFFFFFFF);
            printf("OUT: Example Edge %d -> %d has %d active and %d marg residuals\n", idHost, idTarget, p.second[0], p.second[1]);
            maxWrite--;
            if (maxWrite == 0)
                break;
        }
    }

    virtual void PublishKeyframes(std::vector<FrameHessian *> &frames, bool final, CalibHessian *HCalib) override
    {
        for (FrameHessian *f : frames)
        {
            printf("OUT: KF %d (%s) (id %d, tme %f): %d active, %d marginalized, %d immature points. CameraToWorld:\n",
                   f->frameID,
                   final ? "final" : "non-final",
                   f->shell->incoming_id,
                   f->shell->timestamp,
                   (int)f->pointHessians.size(), (int)f->pointHessiansMarginalized.size(), (int)f->immaturePoints.size());
            std::cout << f->shell->camToWorld.matrix3x4() << "\n";

            int maxWrite = 5;
            for (PointHessian *p : f->pointHessians)
            {
                printf("OUT: Example Point x=%.1f, y=%.1f, idepth=%f, idepth std.dev. %f, %d inlier-residuals\n",
                       p->u, p->v, p->idepth_scaled, sqrt(1.0f / p->idepth_hessian), p->numGoodResiduals);
                maxWrite--;
                if (maxWrite == 0)
                    break;
            }
        }
    }

    virtual void PublishCamPose(Frame *frame, CalibHessian *HCalib) override
    {
        printf("OUT: Current Frame %d (time %f, internal ID %d). CameraToWorld:\n",
               frame->incoming_id,
               frame->timestamp,
               frame->id);
        std::cout << frame->camToWorld.matrix3x4() << "\n";
    }

    virtual void PushLiveFrame(FrameHessian *image) override
    {
        // can be used to get the raw image / intensity pyramid.
    }

    virtual void PushDepthImage(MinimalImageB3 *image) override
    {
        // can be used to get the raw image with depth overlay.
    }
    
    virtual bool NeedPushDepthImage() override
    {
        return false;
    }

    virtual void PushDepthImageFloat(MinimalImageF *image, FrameHessian *KF) override
    {
        printf("OUT: Predicted depth for KF %d (id %d, time %f, internal frame-ID %d). CameraToWorld:\n",
               KF->frameID,
               KF->shell->incoming_id,
               KF->shell->timestamp,
               KF->shell->id);
        std::cout << KF->shell->camToWorld.matrix3x4() << "\n";

        int maxWrite = 5;
        for (int y = 0; y < image->h; y++)
        {
            for (int x = 0; x < image->w; x++)
            {
                if (image->at(x, y) <= 0)
                    continue;

                printf("OUT: Example Idepth at pixel (%d,%d): %f.\n", x, y, image->at(x, y));
                maxWrite--;
                if (maxWrite == 0)
                    break;
            }
            if (maxWrite == 0)
                break;
        }
    }
};

} // namespace visualizer
} // namespace ds_slam

#endif