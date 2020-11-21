#pragma once


#include <iostream>
#include <fstream>
#include <deque>
#include <vector>
#include <math.h>

#include "Utils/num_type.h"
#include "Utils/global_calib.h"
#include "Utils/index_thread_reduce.hpp"
#include "Utils/image_and_exposure.hpp"
#include "Utils/misc.h"
#include "Common/hessian_blocks.hpp"
#include "Common/residuals.hpp"
#include "Common/pixel_selector.hpp"
#include "Common/immature_point.hpp"
#include "Common/frame.hpp"
#include "Common/point.hpp"
#include "Initializer/coarse_initializer.hpp"
#include "Tracker/coarse_tracker.hpp"
#include "Optimization/energy_function.hpp"
#include "LoopClosing/brief_extractor.hpp"
#include "LoopClosing/pose_graph.hpp"
#include "Visualizer/visualizer_3D.hpp"



namespace ds_slam
{

class FullSystem
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // create slam system
    FullSystem(std::string &vocFile);

    // release slam system
    virtual ~FullSystem();

    // set original calibration result
    void SetOriginalCalib(const VecXf &originalCalib, int originalW, int originalH);

    // set gamma function
    void SetGammaFunction(float *BInv);

    // adds a new frame, and creates point & residual structs.
    void AddActiveFrame(ImageAndExposure *image, int id);

    // wait sub-threads finished
    void BlockUntilMappingIsFinished();

    // print slam result
    void PrintResult(std::string file);

    // debug functions
    void DebugPlotTracking();
    void DebugPlot(std::string name);

    // print frame
    void PrintFrameLifetimes();

    // shower
    std::vector<visualizer::Visualizer3D *> viewers;

    // flag
    bool isLost;
    bool initFailed;
    bool initialized;
    bool linearizeOperation;
    bool relocated;

    // frames
    std::vector<Frame *> allFrameHistory;
    std::vector<Frame *> allKeyFramesHistory;

private:

    //////////////////////////////////////////////////////////////////////////////// main pipeline functions
    // system initialize
    void InitializeFromInitializer(FrameHessian *newFrame);

    // coarse track (optimaze frame pose)
    Vec4 TrackNewCoarse(FrameHessian *fh);

    // deliver the tracked frame to mapping
    void DeliverTrackedFrame(FrameHessian *fh, bool needKF);

    // mapping loop
    void MappingLoop();

    // make key/non-key frame
    void MakeKeyFrame(FrameHessian *fh);
    void MakeNonKeyFrame(FrameHessian *fh);

    // coarse trace (optimaze point)
    void TraceNewCoarse(FrameHessian *fh);

    // set precalc values.
    void SetPrecalcValues();

    // active immature points in window by new keyframe
    void ActivatePointsMT();

    // sliding window optimization
    float Optimize(int mnumOptIts);

    // remove outiliers
    void RemoveOutliers();

    // flag point to be removed
    void FlagPointsForRemoval();

    // get point cloud of a keyframe
    void GetKFPointCloud(FrameHessian *frame, bool onlyCorner);

    // update intrinsic of keyframe
    void UpdateCalib(Frame *frame);

    // flag frames to be marginalized
    void FlagFramesForMarginalization(FrameHessian *newFH);

    // make immature points
    void MakeNewTraces(FrameHessian *newFrame, float *gtDepth);

    // marginalizes a frame. drops / marginalizes points & residuals.
    void MarginalizeFrame(FrameHessian *frame);

    // loop closing loop
    void LoopClosingLoop();

    //////////////////////////////////////////////////////////////////////////////// sub-functions
    // active immature points
    void ActivatePointsMT_Reductor(std::vector<PointHessian *> *optimized, std::vector<ImmaturePoint *> *toOptimize,
                                   int min, int max, Vec10 *stats, int tid);
    PointHessian *OptimizeImmaturePoint(ImmaturePoint *point, int minObs, ImmaturePointTemporaryResidual *residuals);

    // optimization functions
    void SolveSystem(int iteration, double lambda);
    Vec3 LinearizeAll(bool fixLinearization);
    void SetNewFrameEnergyTH();
    void LinearizeAllReductor(bool fixLinearization, std::vector<PointFrameResidual *> *toRemove,
                               int min, int max, Vec10 *stats, int tid);
    bool DoStepFromBackup(float stepfacC, float stepfacT, float stepfacR, float stepfacA, float stepfacD);
    void BackupState(bool backupLastStep);
    void LoadSateBackup();
    double CalcLEnergy();
    double CalcMEnergy();
    void ApplyResReductor(bool copyJacobians, int min, int max, Vec10 *stats, int tid);
    std::vector<VecX> GetNullspaces(std::vector<VecX> &nullspaces_pose,
                                    std::vector<VecX> &nullspaces_scale,
                                    std::vector<VecX> &nullspaces_affA,
                                    std::vector<VecX> &nullspaces_affB);

    // print functions
    void PrintOptRes(const Vec3 &res, double resL, double resM, double resPrior, double LExact, float a, float b);
    void PrintLogLine();
    void PrintEigenValLine();

    //////////////////////////////////////////////////////////////////////////////// members
    // calibration
    CalibHessian Hcalib;

    // log stream
    std::ofstream *calibLog;
    std::ofstream *numsLog;
    std::ofstream *eigenAllLog;
    std::ofstream *eigenPLog;
    std::ofstream *eigenALog;
    std::ofstream *DiagonalLog;
    std::ofstream *variancesLog;
    std::ofstream *nullspacesLog;
    std::ofstream *coarseTrackingLog;

    // statistics
    long int statistics_lastNumOptIts;
    long int statistics_numDroppedPoints;
    long int statistics_numActivatedPoints;
    long int statistics_numCreatedPoints;
    long int statistics_numForceDroppedResBwd;
    long int statistics_numForceDroppedResFwd;
    long int statistics_numMargResFwd;
    long int statistics_numMargResBwd;
    float statistics_lastFineTrackRMSE;

    // initializer
    CoarseInitializer *coarseInitializer;
    Vec5 lastCoarseRMSE;

    //changed by tracker-thread. protected by trackMutex
    boost::mutex trackMutex;

    // mutex for tracker exchange.
    boost::mutex coarseTrackerSwapMutex;   // if tracker sees that there is a new reference, tracker locks [coarseTrackerSwapMutex] and swaps the two.
    CoarseTracker *coarseTracker_forNewKF; // set as as reference. protected by [coarseTrackerSwapMutex].
    CoarseTracker *coarseTracker;          // always used to track new frames. protected by [trackMutex].
    float minIdJetVisTracker, maxIdJetVisTracker;
    float minIdJetVisDebug, maxIdJetVisDebug;

    //changed by mapping-thread. protected by mapMutex
    boost::mutex mapMutex;

    // tracking / mapping synchronization. All protected by trackMapSyncMutex.
    boost::mutex trackMapSyncMutex;
    boost::condition_variable trackedFrameSignal;
    boost::condition_variable mappedFrameSignal;
    std::deque<FrameHessian *> unmappedTrackedFrames;
    int needNewKFAfter; // Otherwise, a new KF is *needed that has ID bigger than [needNewKFAfter]*.
    boost::thread mappingThread;
    bool runMapping;
    bool needToKetchupMapping;

    // windows frame and residual
    std::vector<FrameHessian *> frameHessians;
    std::vector<PointFrameResidual *> activeResiduals;

    // mutex for camToWorl's in shells (these are always in a good configuration).
    boost::mutex shellPoseMutex;

    // energy function (optimazation)
    EnergyFunction *ef;

    // thread pool
    IndexThreadReduce<Vec10> treadReduce;

    // point selector and map
    PixelSelector *pixelSelector;
    int *selectionMap;
    int *cornerMap;

    // distance map
    CoarseDistanceMap *coarseDistanceMap;

    // point min distance
    float currentMinActDist;

    // all residual
    std::vector<float> allResVec;

    // last reference stop id
    int lastRefStopID;

    // loop closing
    PoseGraph *poseGraph;
    boost::mutex loopClosingMutex;
    std::deque<Frame *> unmappedKeyFrames;
    bool runLoopClosing;
    boost::thread loopClosingThread;
    std::string vocabularyFile;
    int sequence;
    long latest_optimized_KF_id;
};
}

