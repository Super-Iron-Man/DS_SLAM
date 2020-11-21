#include <stdio.h>
#include <cmath>
#include <algorithm>
#include <Eigen/LU>
#include <Eigen/SVD>
#include <Eigen/Eigenvalues>
#include <boost/make_shared.hpp>

#include "FullSystem/full_system.hpp"
#include "Optimization/energy_function_structs.hpp"
#include "Visualizer/image_display.h"



namespace ds_slam
{

int FrameHessian::instanceCounter = 0;
int PointHessian::instanceCounter = 0;
int CalibHessian::instanceCounter = 0;


////////////////////////////////////////////////////////////////////////  main pipeline function
FullSystem::FullSystem(std::string &vocFile)
{
    int retstat = 0;
    if (setting_logStuff)
    {
        retstat += system("rm -rf logs");
        retstat += system("mkdir logs");

        retstat += system("rm -rf mats");
        retstat += system("mkdir mats");

        calibLog = new std::ofstream();
        calibLog->open("logs/calibLog.txt", std::ios::trunc | std::ios::out);
        calibLog->precision(12);

        numsLog = new std::ofstream();
        numsLog->open("logs/numsLog.txt", std::ios::trunc | std::ios::out);
        numsLog->precision(10);

        coarseTrackingLog = new std::ofstream();
        coarseTrackingLog->open("logs/coarseTrackingLog.txt", std::ios::trunc | std::ios::out);
        coarseTrackingLog->precision(10);

        eigenAllLog = new std::ofstream();
        eigenAllLog->open("logs/eigenAllLog.txt", std::ios::trunc | std::ios::out);
        eigenAllLog->precision(10);

        eigenPLog = new std::ofstream();
        eigenPLog->open("logs/eigenPLog.txt", std::ios::trunc | std::ios::out);
        eigenPLog->precision(10);

        eigenALog = new std::ofstream();
        eigenALog->open("logs/eigenALog.txt", std::ios::trunc | std::ios::out);
        eigenALog->precision(10);

        DiagonalLog = new std::ofstream();
        DiagonalLog->open("logs/diagonal.txt", std::ios::trunc | std::ios::out);
        DiagonalLog->precision(10);

        variancesLog = new std::ofstream();
        variancesLog->open("logs/variancesLog.txt", std::ios::trunc | std::ios::out);
        variancesLog->precision(10);

        nullspacesLog = new std::ofstream();
        nullspacesLog->open("logs/nullspacesLog.txt", std::ios::trunc | std::ios::out);
        nullspacesLog->precision(10);
    }
    else
    {
        nullspacesLog = 0;
        variancesLog = 0;
        DiagonalLog = 0;
        eigenALog = 0;
        eigenPLog = 0;
        eigenAllLog = 0;
        numsLog = 0;
        calibLog = 0;
    }

    assert(retstat != 293847);

    selectionMap = new int[wG[0] * hG[0]];
    cornerMap = new int[wG[0] * hG[0]];

    coarseDistanceMap = new CoarseDistanceMap(wG[0], hG[0]);
    coarseTracker = new CoarseTracker(wG[0], hG[0]);
    coarseTracker_forNewKF = new CoarseTracker(wG[0], hG[0]);
    coarseInitializer = new CoarseInitializer(wG[0], hG[0]);
    pixelSelector = new PixelSelector(wG[0], hG[0]);

    statistics_lastNumOptIts = 0;
    statistics_numDroppedPoints = 0;
    statistics_numActivatedPoints = 0;
    statistics_numCreatedPoints = 0;
    statistics_numForceDroppedResBwd = 0;
    statistics_numForceDroppedResFwd = 0;
    statistics_numMargResFwd = 0;
    statistics_numMargResBwd = 0;

    lastCoarseRMSE.setConstant(100);

    currentMinActDist = 2;
    initialized = false;

    ef = new EnergyFunction();
    ef->red = &this->treadReduce;

    isLost = false;
    initFailed = false;
    relocated = false;

    needNewKFAfter = -1;
    linearizeOperation = true;
    lastRefStopID = 0;

    minIdJetVisDebug = -1;
    maxIdJetVisDebug = -1;
    minIdJetVisTracker = -1;
    maxIdJetVisTracker = -1;

    runMapping = true;
    mappingThread = boost::thread(&FullSystem::MappingLoop, this);

    if(setting_enableLoopClosing)
    {
        vocabularyFile = vocFile;
        poseGraph = new PoseGraph(&allKeyFramesHistory, &viewers);
        poseGraph->LoadVocabulary(vocabularyFile);

        unmappedKeyFrames.clear();
        runLoopClosing = true;
        loopClosingThread = boost::thread(&FullSystem::LoopClosingLoop, this);
        
    }
    sequence = 0;
    latest_optimized_KF_id = 0;
}

void FullSystem::BlockUntilMappingIsFinished()
{
    boost::unique_lock<boost::mutex> lock(trackMapSyncMutex);
    runMapping = false;
    trackedFrameSignal.notify_all();
    lock.unlock();

    mappingThread.join();

    if(setting_enableLoopClosing)
    {
        runLoopClosing = false;
        loopClosingThread.join();
    }
}

FullSystem::~FullSystem()
{
    BlockUntilMappingIsFinished();

    if (setting_logStuff)
    {
        calibLog->close();
        delete calibLog;
        numsLog->close();
        delete numsLog;
        coarseTrackingLog->close();
        delete coarseTrackingLog;
        eigenAllLog->close();
        delete eigenAllLog;
        eigenPLog->close();
        delete eigenPLog;
        eigenALog->close();
        delete eigenALog;
        DiagonalLog->close();
        delete DiagonalLog;
        variancesLog->close();
        delete variancesLog;
        nullspacesLog->close();
        delete nullspacesLog;
    }

    for (Frame *s : allFrameHistory)
        delete s;
    for (FrameHessian *fh : unmappedTrackedFrames)
        delete fh;

    delete[] selectionMap;
    delete[] cornerMap;

    delete coarseDistanceMap;
    delete coarseTracker;
    delete coarseTracker_forNewKF;
    delete coarseInitializer;
    delete pixelSelector;
    delete ef;

    if(setting_enableLoopClosing)
        delete poseGraph;
}

void FullSystem::SetOriginalCalib(const VecXf &originalCalib, int originalW, int originalH)
{
    // Hcalib has be set!!!
}

void FullSystem::SetGammaFunction(float *BInv)
{
    if (BInv == 0)
        return;

    // copy BInv.
    memcpy(Hcalib.Binv, BInv, sizeof(float) * 256);

    // invert.
    for (int i = 1; i < 255; i++)
    {
        // find val, such that Binv[val] = i.
        // I dont care about speed for this, so do it the stupid way.

        for (int s = 1; s < 255; s++)
        {
            if (BInv[s] <= i && BInv[s + 1] >= i)
            {
                Hcalib.B[i] = s + (i - BInv[s]) / (BInv[s + 1] - BInv[s]);
                break;
            }
        }
    }
    Hcalib.B[0] = 0;
    Hcalib.B[255] = 255;
}

void FullSystem::AddActiveFrame(ImageAndExposure *image, int id)
{
    if (isLost)
        return;
    boost::unique_lock<boost::mutex> lock(trackMutex);

    Frame *shell = new Frame();
    shell->timestamp = image->timestamp;
    shell->incoming_id = id;
    shell->id = allFrameHistory.size();
    shell->sequence = sequence;
    allFrameHistory.push_back(shell); // add into allFrameHistory

    FrameHessian *fh = new FrameHessian();
    fh->shell = shell;
    fh->ab_exposure = image->exposure_time;
    fh->MakeImages(image->image, &Hcalib); //make pyramid images

    if (!initialized) // use initializer
    {
        if (coarseInitializer->frameID < 0) // first frame set. fh is kept by coarseInitializer.
        {
            coarseInitializer->SetFirst(&Hcalib, fh);
        }
        else if (coarseInitializer->TrackFrame(fh, viewers)) // if snapped
        {
            InitializeFromInitializer(fh); // initialization finish
            lock.unlock();
            DeliverTrackedFrame(fh, true); // system start!
        }
        else
        {
            // still initializing
            fh->shell->poseValid = false;
            delete fh;
        }
        return;
    }
    else // do front-end operation
    {
        // swap tracking reference
        if (coarseTracker_forNewKF->refFrameID > coarseTracker->refFrameID)
        {
            boost::unique_lock<boost::mutex> crlock(coarseTrackerSwapMutex);
            CoarseTracker *tmp = coarseTracker;
            coarseTracker = coarseTracker_forNewKF;
            coarseTracker_forNewKF = tmp;
        }

        // coarse track with reference key-frame
        Vec4 tres = TrackNewCoarse(fh);
        if (!std::isfinite((double)tres[0]) || !std::isfinite((double)tres[1]) || !std::isfinite((double)tres[2]) || !std::isfinite((double)tres[3]))
        {
            printf("Initial Tracking failed: LOST!\n");
            isLost = true;
            return;
        }

        // key-frame conditions
        bool needToMakeKF = false;
        if (setting_keyframesPerSecond > 0)
        {
            needToMakeKF = allFrameHistory.size() == 1 ||
                           (fh->shell->timestamp - allKeyFramesHistory.back()->timestamp) > 0.95f / setting_keyframesPerSecond;
        }
        else
        {
            Vec2 refToFh = AffLight::FromToVecExposure(coarseTracker->lastRef->ab_exposure, fh->ab_exposure,
                                                       coarseTracker->lastRef_aff_g2l, fh->shell->AffG2L);

            
            needToMakeKF = (allFrameHistory.size() == 1) ||
                           (setting_kfGlobalWeight * setting_maxShiftWeightT * sqrtf((double)tres[1]) / (wG[0] + hG[0]) +
                            setting_kfGlobalWeight * setting_maxShiftWeightR * sqrtf((double)tres[2]) / (wG[0] + hG[0]) +
                            setting_kfGlobalWeight * setting_maxShiftWeightRT * sqrtf((double)tres[3]) / (wG[0] + hG[0]) +
                            setting_kfGlobalWeight * setting_maxAffineWeight * fabs(logf((float)refToFh[0])) > 1) ||
                           (2 * coarseTracker->firstCoarseRMSE < tres[0]);
        }

        // show
        for (visualizer::Visualizer3D *ow : viewers)
            ow->PublishCamPose(fh->shell, &Hcalib);

        lock.unlock();
        DeliverTrackedFrame(fh, needToMakeKF);
        return;
    }
}

void FullSystem::DeliverTrackedFrame(FrameHessian *fh, bool needKF)
{
    if (linearizeOperation)
    {
        if (goStepByStep && lastRefStopID != coarseTracker->refFrameID)
        {
            MinimalImageF3 img(wG[0], hG[0], fh->dI);
            visualizer::DisplayImage("frameToTrack", &img);
            while (true)
            {
                char k = visualizer::WaitKey(0);
                if (k == ' ')
                    break;
                handleKey(k);
            }
            lastRefStopID = coarseTracker->refFrameID;
        }
        else
            handleKey(visualizer::WaitKey(1));

        if (needKF)
            MakeKeyFrame(fh);
        else
            MakeNonKeyFrame(fh);
    }
    else
    {
        boost::unique_lock<boost::mutex> lock(trackMapSyncMutex);
        unmappedTrackedFrames.push_back(fh);
        if (needKF)
            needNewKFAfter = fh->shell->trackingRef->id;
        trackedFrameSignal.notify_all();

        while (coarseTracker_forNewKF->refFrameID == -1 && coarseTracker->refFrameID == -1)
        {
            mappedFrameSignal.wait(lock);
        }

        lock.unlock();
    }
}

void FullSystem::MappingLoop()
{
    boost::unique_lock<boost::mutex> lock(trackMapSyncMutex);

    while (runMapping)
    {
        while (unmappedTrackedFrames.size() == 0)
        {
            trackedFrameSignal.wait(lock);
            if (!runMapping)
                return;
        }

        FrameHessian *fh = unmappedTrackedFrames.front();
        unmappedTrackedFrames.pop_front();

        // guaranteed to make a KF for the very first two tracked frames.
        if (allKeyFramesHistory.size() <= 2)
        {
            lock.unlock();
            MakeKeyFrame(fh);
            lock.lock();
            mappedFrameSignal.notify_all();
            continue;
        }

        if (unmappedTrackedFrames.size() > 3)
            needToKetchupMapping = true;

        if (unmappedTrackedFrames.size() > 0) // if there are other frames to tracke, do that first.
        {
            lock.unlock();
            MakeNonKeyFrame(fh);
            lock.lock();

            if (needToKetchupMapping && unmappedTrackedFrames.size() > 0)
            {
                FrameHessian *fh = unmappedTrackedFrames.front();
                unmappedTrackedFrames.pop_front();
                {
                    boost::unique_lock<boost::mutex> crlock(shellPoseMutex);
                    assert(fh->shell->trackingRef != 0);
                    fh->shell->camToWorld = fh->shell->trackingRef->camToWorld * fh->shell->camToTrackingRef;
                    fh->SetEvalPTScaled(fh->shell->camToWorld.inverse(), fh->shell->AffG2L);
                    fh->shell->Twc = fh->shell->camToWorld;
                }
                delete fh; // free FrameHessian
            }
        }
        else
        {
            if (setting_realTimeMaxKF || needNewKFAfter >= frameHessians.back()->shell->id)
            {
                lock.unlock();
                MakeKeyFrame(fh);
                needToKetchupMapping = false;
                lock.lock();
            }
            else
            {
                lock.unlock();
                MakeNonKeyFrame(fh);
                lock.lock();
            }
        }
        mappedFrameSignal.notify_all();
    }
    printf("MAPPING FINISHED!\n");
}

void FullSystem::MakeNonKeyFrame(FrameHessian *fh)
{
    // needs to be set by mapping thread. no lock required since we are in mapping thread.
    {
        boost::unique_lock<boost::mutex> crlock(shellPoseMutex);
        assert(fh->shell->trackingRef != 0);
        fh->shell->camToWorld = fh->shell->trackingRef->camToWorld * fh->shell->camToTrackingRef;
        fh->SetEvalPTScaled(fh->shell->camToWorld.inverse(), fh->shell->AffG2L);
        fh->shell->Twc = fh->shell->camToWorld;
    }

    // update immature points
    TraceNewCoarse(fh);
    delete fh;
}

void FullSystem::MakeKeyFrame(FrameHessian *fh)
{
    // needs to be set by mapping thread
    {
        boost::unique_lock<boost::mutex> crlock(shellPoseMutex);
        assert(fh->shell->trackingRef != 0);
        fh->shell->camToWorld = fh->shell->trackingRef->camToWorld * fh->shell->camToTrackingRef;
        fh->SetEvalPTScaled(fh->shell->camToWorld.inverse(), fh->shell->AffG2L);
        fh->shell->Twc = fh->shell->camToWorld;
    }

    // update immature points
    TraceNewCoarse(fh);

    boost::unique_lock<boost::mutex> lock(mapMutex);

    // flag frames to be marginalized
    FlagFramesForMarginalization(fh);

    // add new frame to hessian struct
    fh->idx = frameHessians.size();
    frameHessians.push_back(fh);
    fh->frameID = allKeyFramesHistory.size();
    allKeyFramesHistory.push_back(fh->shell);
    ef->InsertFrame(fh, &Hcalib);
    fh->shell->is_kf = true;
    fh->shell->index = fh->frameID;

    // set pre-calculated value
    SetPrecalcValues();

    // add new residuals for old points
    int numFwdResAdde = 0;
    for (FrameHessian *fh1 : frameHessians) // go through all active frames
    {
        if (fh1 == fh)
            continue;
        for (PointHessian *ph : fh1->pointHessians)
        {
            PointFrameResidual *r = new PointFrameResidual(ph, fh1, fh);
            r->SetState(ResState::IN);
            ph->residuals.push_back(r);
            ef->InsertResidual(r);
            ph->lastResiduals[1] = ph->lastResiduals[0];
            ph->lastResiduals[0] = std::pair<PointFrameResidual *, ResState>(r, ResState::IN);
            numFwdResAdde += 1;
        }
    }

    // activate immature points
    ActivatePointsMT();
    ef->MakeIDX();

    // optimaze all
    fh->frameEnergyTH = frameHessians.back()->frameEnergyTH;
    float rmse = Optimize(setting_maxOptIterations);

    // figure out if initialization failed
    if (allKeyFramesHistory.size() <= 4)
    {
        if (allKeyFramesHistory.size() == 2 && rmse > 20 * benchmark_initializerSlackFactor)
        {
            printf("I THINK INITIALIZATINO FAILED! Resetting.\n");
            initFailed = true;
        }
        if (allKeyFramesHistory.size() == 3 && rmse > 13 * benchmark_initializerSlackFactor)
        {
            printf("I THINK INITIALIZATINO FAILED! Resetting.\n");
            initFailed = true;
        }
        if (allKeyFramesHistory.size() == 4 && rmse > 9 * benchmark_initializerSlackFactor)
        {
            printf("I THINK INITIALIZATINO FAILED! Resetting.\n");
            initFailed = true;
        }
    }

    // optimaze failed, reset system
    if (isLost) //todo
    {
        initialized = false;
        relocated = true;
        sequence++; // flag new sequence
        return;
    }

    // remove outlier point
    RemoveOutliers();

    // update reference key-frame for front-end
    {
        boost::unique_lock<boost::mutex> crlock(coarseTrackerSwapMutex);
        coarseTracker_forNewKF->MakeK(&Hcalib);
        coarseTracker_forNewKF->SetCoarseTrackingRef(frameHessians);

        coarseTracker_forNewKF->DebugPlotIDepthMap(&minIdJetVisTracker, &maxIdJetVisTracker, viewers);
        coarseTracker_forNewKF->DebugPlotIDepthMapFloat(viewers);
    }

    // debug plot
    DebugPlot("post optimize");

    // marginalize points
    FlagPointsForRemoval();
    GetKFPointCloud(fh, true); // get point cloud of new keyframe
    UpdateCalib(fh->shell); // update intrinsic parameter
    ef->DropPointsF();
    GetNullspaces(ef->lastNullspaces_pose,
                  ef->lastNullspaces_scale,
                  ef->lastNullspaces_affA,
                  ef->lastNullspaces_affB);
    ef->MarginalizePointsF();

    // add new immature points
    MakeNewTraces(fh, 0);

    // record the relative poses for building a covisibility graph
    for (FrameHessian *fh1 : frameHessians)
    {
        for(FrameHessian *fh2 : frameHessians)
        {
            if(fh1->shell->index < fh2->shell->index)
            {
                fh2->shell->mutexPoseRel.lock();
                SE3 T_fh1_fh2 = fh1->shell->Twc.inverse() * fh2->shell->Twc;
                fh2->shell->poseRel[fh1->shell] = Sim3(T_fh1_fh2.matrix());
                fh2->shell->mutexPoseRel.unlock();
            }
        }
    }

    // show result
    for (visualizer::Visualizer3D *ow : viewers)
    {
        ow->PublishGraph(ef->connectivityMap);
        ow->PublishKeyframes(frameHessians, false, &Hcalib);
    }

    // marginalize frames
    for (unsigned int i = 0; i < frameHessians.size(); i++)
    {
        if (frameHessians[i]->flaggedForMarginalization)
        {
            MarginalizeFrame(frameHessians[i]);
            i = 0;
        }
    }

    PrintLogLine();
    //PrintEigenValLine();

    if (setting_enableLoopClosing && (int)fh->shell->index > setting_minKFLoopClosing)
    {
        fh->shell->CreateCvMat(fh->dI, wG[0], hG[0]);
        // add current key-frame to loop closing
        loopClosingMutex.lock();
        unmappedKeyFrames.push_back(fh->shell);
        loopClosingMutex.unlock();
    }
}

void FullSystem::LoopClosingLoop()
{
    if(!setting_enableLoopClosing)
        return;
    while(runLoopClosing)
    {
        if(unmappedKeyFrames.size())
        {
            if(!runLoopClosing)
                return;

            loopClosingMutex.lock();
            Frame *frame = unmappedKeyFrames.front();
            unmappedKeyFrames.pop_front();
            latest_optimized_KF_id = frame->index;
            if(unmappedKeyFrames.size() > 20)
            {
                printf("The thread of loop closing is blocking, clear buffer!\n");
                unmappedKeyFrames.clear();
            }
            loopClosingMutex.unlock();

            KeyFrame *keyframe = new KeyFrame(frame);
            poseGraph->AddKeyFrame(keyframe, true);
        }
        usleep(1000);
    }
}


////////////////////////////////////////////////////////////////////////  sub-function

void FullSystem::InitializeFromInitializer(FrameHessian *newFrame)
{
    boost::unique_lock<boost::mutex> lock(mapMutex);

    // add firstframe.
    FrameHessian *firstFrame = coarseInitializer->firstFrame;
    firstFrame->idx = frameHessians.size();
    frameHessians.push_back(firstFrame);
    firstFrame->frameID = allKeyFramesHistory.size();
    firstFrame->shell->is_kf = true;
    firstFrame->shell->index = firstFrame->frameID;
    allKeyFramesHistory.push_back(firstFrame->shell);
    ef->InsertFrame(firstFrame, &Hcalib);
    SetPrecalcValues();

    //int numPointsTotal = MakePixelStatus(firstFrame->dI, selectionMap, wG[0], hG[0], setting_desiredDensity);
    //int numPointsTotal = pixelSelector->MakeMaps(firstFrame->dIp, selectionMap,setting_desiredDensity);

    firstFrame->pointHessians.reserve(wG[0] * hG[0] * 0.2f);
    firstFrame->pointHessiansMarginalized.reserve(wG[0] * hG[0] * 0.2f);
    firstFrame->pointHessiansOut.reserve(wG[0] * hG[0] * 0.2f);

    float sumID = 1e-5, numID = 1e-5;
    for (int i = 0; i < coarseInitializer->numPoints[0]; i++)
    {
        sumID += coarseInitializer->points[0][i].iR;
        numID++;
    }
    float rescaleFactor = 1 / (sumID / numID);

    // randomly sub-select the points I need.
    float keepPercentage = setting_desiredPointDensity / coarseInitializer->numPoints[0];

    if (!setting_debugout_runquiet)
        printf("Initialization: keep %.1f%% (need %d, have %d)!\n", 100 * keepPercentage,
               (int)(setting_desiredPointDensity), coarseInitializer->numPoints[0]);

    for (int i = 0; i < coarseInitializer->numPoints[0]; i++)
    {
        if (rand() / (float)RAND_MAX > keepPercentage)
            continue;

        Pnt *point = coarseInitializer->points[0] + i;
        ImmaturePoint *pt = new ImmaturePoint(point->u + 0.5f, point->v + 0.5f, firstFrame, point->my_type, 0, &Hcalib);

        if (!std::isfinite(pt->energyTH))
        {
            delete pt;
            continue;
        }

        pt->idepth_max = pt->idepth_min = 1;
        PointHessian *ph = new PointHessian(pt);
        if (!std::isfinite(ph->energyTH))
        {
            delete ph;
            delete pt;
            continue;
        }

        Point *pnt = new Point(pt); // create map point
        ph->shell = pnt;
        pnt->host->points.push_back(pnt);
        pnt->idepth = ph->idepth_scaled;
        delete pt;

        ph->SetIdepthScaled(point->iR * rescaleFactor);
        ph->SetIdepthZero(ph->idepth);
        ph->hasDepthPrior = true;
        ph->SetPointStatus(PointHessian::ACTIVE);

        firstFrame->pointHessians.push_back(ph); // convert to active point
        ef->InsertPoint(ph);
    }

    SE3 firstToNew = coarseInitializer->thisToNext;
    firstToNew.translation() /= rescaleFactor;

    // really no lock required, as we are initializing.
    {
        boost::unique_lock<boost::mutex> crlock(shellPoseMutex);
        firstFrame->shell->camToWorld = SE3();
        firstFrame->shell->AffG2L = AffLight(0, 0);
        firstFrame->SetEvalPTScaled(firstFrame->shell->camToWorld.inverse(), firstFrame->shell->AffG2L);
        firstFrame->shell->trackingRef = 0;
        firstFrame->shell->camToTrackingRef = SE3();
        firstFrame->shell->Twc = SE3();
        firstFrame->shell->TwcOpti = Sim3();
        //if(setting_enableLoopClosing)
        //{
        //    firstFrame->shell->CreateCvMat(firstFrame->dI, wG[0], hG[0]);
        //    // add current key-frame to loop closing
        //    loopClosingMutex.lock();
        //    unmappedKeyFrames.push_back(firstFrame->shell);
        //    loopClosingMutex.unlock();
        //}

        newFrame->shell->camToWorld = firstToNew.inverse();
        newFrame->shell->AffG2L = AffLight(0, 0);
        newFrame->SetEvalPTScaled(newFrame->shell->camToWorld.inverse(), newFrame->shell->AffG2L);
        newFrame->shell->trackingRef = firstFrame->shell;
        newFrame->shell->camToTrackingRef = firstToNew.inverse();
        newFrame->shell->Twc = newFrame->shell->camToWorld;
    }

    initialized = true;
    printf("INITIALIZE FROM INITIALIZER (%d pts)!\n", (int)firstFrame->pointHessians.size());
}

Vec4 FullSystem::TrackNewCoarse(FrameHessian *fh)
{

    assert(allFrameHistory.size() > 0);
    // set pose initialization.

    for (visualizer::Visualizer3D *ow : viewers)
        ow->PushLiveFrame(fh);

    FrameHessian *lastF = coarseTracker->lastRef;

    AffLight aff_last_2_l = AffLight(0, 0);

    std::vector<SE3, Eigen::aligned_allocator<SE3>> lastF_2_fh_tries;
    if (allFrameHistory.size() == 2)
    {
        for (unsigned int i = 0; i < lastF_2_fh_tries.size(); i++)
            lastF_2_fh_tries.push_back(SE3());
    }
    else
    {
        Frame *slast = allFrameHistory[allFrameHistory.size() - 2];
        Frame *sprelast = allFrameHistory[allFrameHistory.size() - 3];
        SE3 slast_2_sprelast;
        SE3 lastF_2_slast;
        { // lock on global pose consistency!
            boost::unique_lock<boost::mutex> crlock(shellPoseMutex);
            slast_2_sprelast = sprelast->camToWorld.inverse() * slast->camToWorld;
            lastF_2_slast = slast->camToWorld.inverse() * lastF->shell->camToWorld;
            aff_last_2_l = slast->AffG2L;
        }
        SE3 fh_2_slast = slast_2_sprelast; // assumed to be the same as fh_2_slast.

        // get last delta-movement.
        lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast);                        // assume constant motion.
        lastF_2_fh_tries.push_back(fh_2_slast.inverse() * fh_2_slast.inverse() * lastF_2_slast); // assume double motion (frame skipped)
        lastF_2_fh_tries.push_back(SE3::exp(fh_2_slast.log() * 0.5).inverse() * lastF_2_slast);  // assume half motion.
        lastF_2_fh_tries.push_back(lastF_2_slast);                                               // assume zero motion.
        lastF_2_fh_tries.push_back(SE3());                                                       // assume zero motion FROM KF.

        // just try a TON of different initializations (all rotations). In the end,
        // if they don't work they will only be tried on the coarsest level, which is super fast anyway.
        // also, if tracking rails here we loose, so we really, really want to avoid that.
        for (float rotDelta = 0.02; rotDelta < 0.05; rotDelta++)
        {
            lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1, rotDelta, 0, 0), Vec3(0, 0, 0)));                  // assume constant motion.
            lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1, 0, rotDelta, 0), Vec3(0, 0, 0)));                  // assume constant motion.
            lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1, 0, 0, rotDelta), Vec3(0, 0, 0)));                  // assume constant motion.
            lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1, -rotDelta, 0, 0), Vec3(0, 0, 0)));                 // assume constant motion.
            lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1, 0, -rotDelta, 0), Vec3(0, 0, 0)));                 // assume constant motion.
            lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1, 0, 0, -rotDelta), Vec3(0, 0, 0)));                 // assume constant motion.
            lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1, rotDelta, rotDelta, 0), Vec3(0, 0, 0)));           // assume constant motion.
            lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1, 0, rotDelta, rotDelta), Vec3(0, 0, 0)));           // assume constant motion.
            lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1, rotDelta, 0, rotDelta), Vec3(0, 0, 0)));           // assume constant motion.
            lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1, -rotDelta, rotDelta, 0), Vec3(0, 0, 0)));          // assume constant motion.
            lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1, 0, -rotDelta, rotDelta), Vec3(0, 0, 0)));          // assume constant motion.
            lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1, -rotDelta, 0, rotDelta), Vec3(0, 0, 0)));          // assume constant motion.
            lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1, rotDelta, -rotDelta, 0), Vec3(0, 0, 0)));          // assume constant motion.
            lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1, 0, rotDelta, -rotDelta), Vec3(0, 0, 0)));          // assume constant motion.
            lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1, rotDelta, 0, -rotDelta), Vec3(0, 0, 0)));          // assume constant motion.
            lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1, -rotDelta, -rotDelta, 0), Vec3(0, 0, 0)));         // assume constant motion.
            lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1, 0, -rotDelta, -rotDelta), Vec3(0, 0, 0)));         // assume constant motion.
            lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1, -rotDelta, 0, -rotDelta), Vec3(0, 0, 0)));         // assume constant motion.
            lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1, -rotDelta, -rotDelta, -rotDelta), Vec3(0, 0, 0))); // assume constant motion.
            lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1, -rotDelta, -rotDelta, rotDelta), Vec3(0, 0, 0)));  // assume constant motion.
            lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1, -rotDelta, rotDelta, -rotDelta), Vec3(0, 0, 0)));  // assume constant motion.
            lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1, -rotDelta, rotDelta, rotDelta), Vec3(0, 0, 0)));   // assume constant motion.
            lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1, rotDelta, -rotDelta, -rotDelta), Vec3(0, 0, 0)));  // assume constant motion.
            lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1, rotDelta, -rotDelta, rotDelta), Vec3(0, 0, 0)));   // assume constant motion.
            lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1, rotDelta, rotDelta, -rotDelta), Vec3(0, 0, 0)));   // assume constant motion.
            lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1, rotDelta, rotDelta, rotDelta), Vec3(0, 0, 0)));    // assume constant motion.
        }

        if (!slast->poseValid || !sprelast->poseValid || !lastF->shell->poseValid)
        {
            lastF_2_fh_tries.clear();
            lastF_2_fh_tries.push_back(SE3());
        }
    }

    Vec3 flowVecs = Vec3(100, 100, 100);
    SE3 lastF_2_fh = SE3();
    AffLight AffG2L = AffLight(0, 0);

    // as long as maxResForImmediateAccept is not reached, I'll continue through the options.
    // I'll keep track of the so-far best achieved residual for each level in achievedRes.
    // If on a coarse level, tracking is WORSE than achievedRes, we will not continue to save time.

    Vec5 achievedRes = Vec5::Constant(NAN);
    bool haveOneGood = false;
    int tryIterations = 0;
    for (unsigned int i = 0; i < lastF_2_fh_tries.size(); i++)
    {
        AffLight aff_g2l_this = aff_last_2_l;
        SE3 lastF_2_fh_this = lastF_2_fh_tries[i];
        bool trackingIsGood = coarseTracker->TrackNewestCoarse(
            fh, lastF_2_fh_this, aff_g2l_this,
            pyrLevelsUsed - 1,
            achievedRes); // in each level has to be at least as good as the last try.
        tryIterations++;

        if (i != 0)
        {
            printf("RE-TRACK ATTEMPT %d with initOption %d and start-lvl %d (ab %f %f): %f %f %f %f %f -> %f %f %f %f %f \n",
                   i,
                   i, pyrLevelsUsed - 1,
                   aff_g2l_this.a, aff_g2l_this.b,
                   achievedRes[0],
                   achievedRes[1],
                   achievedRes[2],
                   achievedRes[3],
                   achievedRes[4],
                   coarseTracker->lastResiduals[0],
                   coarseTracker->lastResiduals[1],
                   coarseTracker->lastResiduals[2],
                   coarseTracker->lastResiduals[3],
                   coarseTracker->lastResiduals[4]);
        }

        // do we have a new winner?
        if (trackingIsGood && std::isfinite((float)coarseTracker->lastResiduals[0]) && !(coarseTracker->lastResiduals[0] >= achievedRes[0]))
        {
            //printf("take over. minRes %f -> %f!\n", achievedRes[0], coarseTracker->lastResiduals[0]);
            flowVecs = coarseTracker->lastFlowIndicators;
            AffG2L = aff_g2l_this;
            lastF_2_fh = lastF_2_fh_this;
            haveOneGood = true;
        }

        // take over achieved res (always).
        if (haveOneGood)
        {
            for (int i = 0; i < 5; i++)
            {
                if (!std::isfinite((float)achievedRes[i]) || achievedRes[i] > coarseTracker->lastResiduals[i]) // take over if achievedRes is either bigger or NAN.
                    achievedRes[i] = coarseTracker->lastResiduals[i];
            }
        }

        if (haveOneGood && achievedRes[0] < lastCoarseRMSE[0] * setting_reTrackThreshold)
            break;
    }

    if (!haveOneGood)
    {
        printf("BIG ERROR! tracking failed entirely. Take predictred pose and hope we may somehow recover.\n");
        flowVecs = Vec3(0, 0, 0);
        AffG2L = aff_last_2_l;
        lastF_2_fh = lastF_2_fh_tries[0];
    }

    lastCoarseRMSE = achievedRes;

    // no lock required, as fh is not used anywhere yet.
    fh->shell->camToTrackingRef = lastF_2_fh.inverse();
    fh->shell->trackingRef = lastF->shell;
    fh->shell->AffG2L = AffG2L;
    fh->shell->camToWorld = fh->shell->trackingRef->camToWorld * fh->shell->camToTrackingRef; // initial frame pose

    if (coarseTracker->firstCoarseRMSE < 0)
        coarseTracker->firstCoarseRMSE = achievedRes[0];

    if (!setting_debugout_runquiet)
        printf("Coarse Tracker tracked ab = %f %f (exp %f). Res %f!\n", AffG2L.a, AffG2L.b, fh->ab_exposure, achievedRes[0]);

    if (setting_logStuff)
    {
        (*coarseTrackingLog) << std::setprecision(16)
                             << fh->shell->id << " "
                             << fh->shell->timestamp << " "
                             << fh->ab_exposure << " "
                             << fh->shell->camToWorld.log().transpose() << " "
                             << AffG2L.a << " "
                             << AffG2L.b << " "
                             << achievedRes[0] << " "
                             << tryIterations << "\n";
    }

    return Vec4(achievedRes[0], flowVecs[0], flowVecs[1], flowVecs[2]);
}

void FullSystem::TraceNewCoarse(FrameHessian *fh)
{
    boost::unique_lock<boost::mutex> lock(mapMutex);

    int trace_total = 0, trace_good = 0, trace_oob = 0, trace_out = 0, trace_skip = 0, trace_badcondition = 0, trace_uninitialized = 0;

    Mat33f K = Mat33f::Identity();
    K(0, 0) = Hcalib.fxl();
    K(1, 1) = Hcalib.fyl();
    K(0, 2) = Hcalib.cxl();
    K(1, 2) = Hcalib.cyl();

    for (FrameHessian *host : frameHessians) // go through all active frames
    {
        SE3 hostToNew = fh->PRE_worldToCam * host->PRE_camToWorld;
        Mat33f KRKi = K * hostToNew.rotationMatrix().cast<float>() * K.inverse();
        Vec3f Kt = K * hostToNew.translation().cast<float>();

        Vec2f aff = AffLight::FromToVecExposure(host->ab_exposure, fh->ab_exposure, host->AffG2L(), fh->AffG2L()).cast<float>();

        for (ImmaturePoint *ph : host->immaturePoints) // go throught all immature points
        {
            ph->TraceOn(fh, KRKi, Kt, aff, &Hcalib, false);

            if (ph->lastTraceStatus == ImmaturePointStatus::IPS_GOOD)
                trace_good++;
            if (ph->lastTraceStatus == ImmaturePointStatus::IPS_BADCONDITION)
                trace_badcondition++;
            if (ph->lastTraceStatus == ImmaturePointStatus::IPS_OOB)
                trace_oob++;
            if (ph->lastTraceStatus == ImmaturePointStatus::IPS_OUTLIER)
                trace_out++;
            if (ph->lastTraceStatus == ImmaturePointStatus::IPS_SKIPPED)
                trace_skip++;
            if (ph->lastTraceStatus == ImmaturePointStatus::IPS_UNINITIALIZED)
                trace_uninitialized++;
            trace_total++;
        }
    }
    //printf("ADD: TRACE: %'d points. %'d (%.0f%%) good. %'d (%.0f%%) skip. %'d (%.0f%%) badcond. %'d (%.0f%%) oob. %'d (%.0f%%) out. %'d (%.0f%%) uninit.\n",
    //       trace_total,
    //       trace_good, 100 * trace_good / (float)trace_total,
    //       trace_skip, 100 * trace_skip / (float)trace_total,
    //       trace_badcondition, 100 * trace_badcondition / (float)trace_total,
    //       trace_oob, 100 * trace_oob / (float)trace_total,
    //       trace_out, 100 * trace_out / (float)trace_total,
    //       trace_uninitialized, 100 * trace_uninitialized / (float)trace_total);
}

void FullSystem::SetPrecalcValues()
{
    for (FrameHessian *fh : frameHessians)
    {
        fh->targetPrecalc.resize(frameHessians.size());
        for (unsigned int i = 0; i < frameHessians.size(); i++)
            fh->targetPrecalc[i].Set(fh, frameHessians[i], &Hcalib);
    }

    ef->SetDeltaF(&Hcalib);
}

void FullSystem::ActivatePointsMT_Reductor(std::vector<PointHessian *> *optimized,
                                           std::vector<ImmaturePoint *> *toOptimize,
                                           int min, int max, Vec10 *stats, int tid)
{
    ImmaturePointTemporaryResidual *tr = new ImmaturePointTemporaryResidual[frameHessians.size()];
    for (int k = min; k < max; k++)
    {
        (*optimized)[k] = OptimizeImmaturePoint((*toOptimize)[k], 1, tr);
    }
    delete[] tr;
}

void FullSystem::ActivatePointsMT()
{
    if (ef->nPoints < setting_desiredPointDensity * 0.66)
        currentMinActDist -= 0.8;
    if (ef->nPoints < setting_desiredPointDensity * 0.8)
        currentMinActDist -= 0.5;
    else if (ef->nPoints < setting_desiredPointDensity * 0.9)
        currentMinActDist -= 0.2;
    else if (ef->nPoints < setting_desiredPointDensity)
        currentMinActDist -= 0.1;

    if (ef->nPoints > setting_desiredPointDensity * 1.5)
        currentMinActDist += 0.8;
    if (ef->nPoints > setting_desiredPointDensity * 1.3)
        currentMinActDist += 0.5;
    if (ef->nPoints > setting_desiredPointDensity * 1.15)
        currentMinActDist += 0.2;
    if (ef->nPoints > setting_desiredPointDensity)
        currentMinActDist += 0.1;

    if (currentMinActDist < 0)
        currentMinActDist = 0;
    if (currentMinActDist > 4)
        currentMinActDist = 4;

    if (!setting_debugout_runquiet)
        printf("SPARSITY:  MinActDist %f (need %d points, have %d points)!\n",
               currentMinActDist, (int)(setting_desiredPointDensity), ef->nPoints);

    FrameHessian *newestHs = frameHessians.back();

    // make dist map.
    coarseDistanceMap->MakeK(&Hcalib);
    coarseDistanceMap->MakeDistanceMap(frameHessians, newestHs);

    //coarseTracker->debugPlotDistMap("distMap");

    std::vector<ImmaturePoint *> toOptimize;
    toOptimize.reserve(20000);

    for (FrameHessian *host : frameHessians) // go through all active frames
    {
        if (host == newestHs)
            continue;

        SE3 fhToNew = newestHs->PRE_worldToCam * host->PRE_camToWorld;
        Mat33f KRKi = (coarseDistanceMap->K[1] * fhToNew.rotationMatrix().cast<float>() * coarseDistanceMap->Ki[0]);
        Vec3f Kt = (coarseDistanceMap->K[1] * fhToNew.translation().cast<float>());

        for (unsigned int i = 0; i < host->immaturePoints.size(); i += 1)
        {
            ImmaturePoint *ph = host->immaturePoints[i];
            ph->idxInImmaturePoints = i;

            // delete points that have never been traced successfully, or that are outlier on the last trace.
            if (!std::isfinite(ph->idepth_max) || ph->lastTraceStatus == IPS_OUTLIER)
            {
                // immature_invalid_deleted++;
                delete ph; // remove immature point
                host->immaturePoints[i] = 0;
                continue;
            }

            // can activate only if this is true.
            bool canActivate = (ph->lastTraceStatus == IPS_GOOD || ph->lastTraceStatus == IPS_SKIPPED || ph->lastTraceStatus == IPS_BADCONDITION || ph->lastTraceStatus == IPS_OOB) &&
                               ph->lastTracePixelInterval < 8 && ph->quality > setting_minTraceQuality && (ph->idepth_max + ph->idepth_min) > 0;

            // if I cannot activate the point, skip it. Maybe also delete it.
            if (!canActivate)
            {
                // if point will be out afterwards, delete it instead.
                if (ph->host->flaggedForMarginalization || ph->lastTraceStatus == IPS_OOB)
                {
                    // immature_notReady_deleted++;
                    delete ph; // remove immature point
                    host->immaturePoints[i] = 0;
                }
                // immature_notReady_skipped++;
                continue;
            }

            // see if we need to activate point due to distance map.
            Vec3f ptp = KRKi * Vec3f(ph->u, ph->v, 1) + Kt * (0.5f * (ph->idepth_max + ph->idepth_min));
            int u = ptp[0] / ptp[2] + 0.5f;
            int v = ptp[1] / ptp[2] + 0.5f;

            if ((u > 0 && v > 0 && u < wG[1] && v < hG[1]))
            {
                float dist = coarseDistanceMap->fwdWarpedIDDistFinal[u + wG[1] * v] + (ptp[0] - floorf((float)(ptp[0]))); // distance
                if (dist >= currentMinActDist * ph->my_type) // final distance = pyramid level * distance
                {
                    coarseDistanceMap->AddIntoDistFinal(u, v);
                    toOptimize.push_back(ph); // seleted immature point to optimize
                }
            }
            else
            {
                delete ph; // remove immature point
                host->immaturePoints[i] = 0;
            }
        }
    }

    // printf("ACTIVATE: %d. (del %d, notReady %d, marg %d, good %d, marg-skip %d)\n",
    //        (int)toOptimize.size(), immature_deleted, immature_notReady, immature_needMarg, immature_want, immature_margskip);

    std::vector<PointHessian *> optimized;
    optimized.resize(toOptimize.size());

    if (multiThreading)
        treadReduce.reduce(boost::bind(&FullSystem::ActivatePointsMT_Reductor, this, &optimized, &toOptimize, _1, _2, _3, _4), 0, toOptimize.size(), 50);

    else
        ActivatePointsMT_Reductor(&optimized, &toOptimize, 0, toOptimize.size(), 0, 0);

    for (unsigned k = 0; k < toOptimize.size(); k++)
    {
        PointHessian *newpoint = optimized[k];
        ImmaturePoint *ph = toOptimize[k];

        if (newpoint != 0 && newpoint != (PointHessian *)((long)(-1)))
        {
            newpoint->host->immaturePoints[ph->idxInImmaturePoints] = 0;
            newpoint->host->pointHessians.push_back(newpoint); // convert to active point
            Point *pnt = new Point(ph); // create map point
            newpoint->shell = pnt;
            pnt->host->points.push_back(pnt);
            pnt->idepth = newpoint->idepth_scaled;
            ef->InsertPoint(newpoint);
            for (PointFrameResidual *r : newpoint->residuals)
                ef->InsertResidual(r);
            assert(newpoint->efPoint != 0);
            delete ph; // remove immature point
        }
        else if (newpoint == (PointHessian *)((long)(-1)) || ph->lastTraceStatus == IPS_OOB)
        {
            delete ph; // remove immature point
            ph->host->immaturePoints[ph->idxInImmaturePoints] = 0;
        }
        else
        {
            assert(newpoint == 0 || newpoint == (PointHessian *)((long)(-1)));
        }
    }

    for (FrameHessian *host : frameHessians)
    {
        for (int i = 0; i < (int)host->immaturePoints.size(); i++)
        {
            if (host->immaturePoints[i] == 0)
            {
                host->immaturePoints[i] = host->immaturePoints.back();
                host->immaturePoints.pop_back();
                i--;
            }
        }
    }
}

void FullSystem::FlagPointsForRemoval()
{
    assert(EFIndicesValid);

    std::vector<FrameHessian *> fhsToKeepPoints;
    std::vector<FrameHessian *> fhsToMargPoints;

    // separate the keeped frames and marginalized frames
    for (int i = ((int)frameHessians.size()) - 1; i >= 0 && i >= ((int)frameHessians.size()); i--)
        if (!frameHessians[i]->flaggedForMarginalization)
            fhsToKeepPoints.push_back(frameHessians[i]);

    for (int i = 0; i < (int)frameHessians.size(); i++)
        if (frameHessians[i]->flaggedForMarginalization)
            fhsToMargPoints.push_back(frameHessians[i]);

    //ef->SetAdjointsF();
    //ef->SetDeltaF(&Hcalib);
    int flag_oob = 0, flag_in = 0, flag_inin = 0, flag_nores = 0;

    for (FrameHessian *host : frameHessians) // go through all active frames
    {
        for (unsigned int i = 0; i < host->pointHessians.size(); i++)
        {
            PointHessian *ph = host->pointHessians[i];
            if (ph == 0)
                continue;

            if (ph->idepth_scaled < 0 || ph->residuals.size() == 0)
            {
                host->pointHessiansOut.push_back(ph); // removed point
                ph->efPoint->stateFlag = EFPointStatus::PS_DROP;
                host->pointHessians[i] = 0;
                flag_nores++;

                // remove outlier point
                assert(ph->shell->host == host->shell);
                host->shell->RemoveOutlier(ph->shell);
            }
            else if (ph->IsOOB(fhsToKeepPoints, fhsToMargPoints) || host->flaggedForMarginalization)
            {
                flag_oob++;
                if (ph->IsInlierNew())
                {
                    flag_in++;
                    int ngoodRes = 0;
                    for (PointFrameResidual *r : ph->residuals)
                    {
                        r->ResetOOB();
                        r->Linearize(&Hcalib);
                        r->efResidual->isLinearized = false;
                        r->ApplyRes(true);
                        if (r->efResidual->IsActive())
                        {
                            r->efResidual->FixLinearizationF(ef); // flag Linearized!!!
                            ngoodRes++;
                        }
                    }
                    if (ph->idepth_hessian > setting_minIdepthH_marg)
                    {
                        flag_inin++;
                        ph->efPoint->stateFlag = EFPointStatus::PS_MARGINALIZE;
                        host->pointHessiansMarginalized.push_back(ph); // marginalized point
                    }
                    else
                    {
                        ph->efPoint->stateFlag = EFPointStatus::PS_DROP;
                        host->pointHessiansOut.push_back(ph); // removed point

                        // remove outlier point
                        assert(ph->shell->host == host->shell);
                        host->shell->RemoveOutlier(ph->shell);
                    }
                }
                else
                {
                    host->pointHessiansOut.push_back(ph); // removed point
                    ph->efPoint->stateFlag = EFPointStatus::PS_DROP;

                    // remove outlier point
                    assert(ph->shell->host == host->shell);
                    host->shell->RemoveOutlier(ph->shell);
                }

                host->pointHessians[i] = 0; // remove active point
            }
        }

        for (int i = 0; i < (int)host->pointHessians.size(); i++)
        {
            if (host->pointHessians[i] == 0)
            {
                host->pointHessians[i] = host->pointHessians.back();
                host->pointHessians.pop_back();
                i--;
            }
        }
    }
}

void FullSystem::GetKFPointCloud(FrameHessian *frame, bool onlyCorner)
{
    for (FrameHessian *host : frameHessians)
    {
        if(host == frame)
        {
            // add points of current frame
            for (PointHessian *ph : host->pointHessians)
            {
                if(onlyCorner)
                {
                    if(ph->shell->isCorner)
                        frame->shell->window_PC.push_back(ph->shell);
                }
                else
                    frame->shell->window_PC.push_back(ph->shell);
            }
        }
        else
        {
            // add points of other frame
            for (PointHessian *ph : host->pointHessians)
            {
                for (PointFrameResidual *res : ph->residuals)
                {
                    if(res->target == frame)
                    {
                        if(onlyCorner)
                        {
                            if(ph->shell->isCorner)
                                frame->shell->window_PC.push_back(ph->shell);
                        }
                        else
                            frame->shell->window_PC.push_back(ph->shell);
                    }
                }
            }
        }
    }
}

void FullSystem::UpdateCalib(Frame *frame)
{
    frame->fx = Hcalib.fxl();
    frame->fy = Hcalib.fyl();
    frame->cx = Hcalib.cxl();
    frame->cy = Hcalib.cyl();
    frame->fxi = Hcalib.fxli();
    frame->fyi = Hcalib.fyli();
    frame->cxi = Hcalib.cxli();
    frame->cyi = Hcalib.cyli();
}

void FullSystem::MakeNewTraces(FrameHessian *newFrame, float *gtDepth)
{
    pixelSelector->allowFast = true;
    //int numPointsTotal = MakePixelStatus(newFrame->dI, selectionMap, wG[0], hG[0], setting_desiredDensity);
    int numPointsTotal = pixelSelector->MakeMaps(newFrame, selectionMap, cornerMap, setting_desiredImmatureDensity);

    newFrame->pointHessians.reserve(numPointsTotal * 1.2f);
    //fh->pointHessiansInactive.reserve(numPointsTotal*1.2f);
    newFrame->pointHessiansMarginalized.reserve(numPointsTotal * 1.2f);
    newFrame->pointHessiansOut.reserve(numPointsTotal * 1.2f);

    for (int y = patternPadding + 1; y < hG[0] - patternPadding - 2; y++)
    {
        for (int x = patternPadding + 1; x < wG[0] - patternPadding - 2; x++)
        {
            int i = x + y * wG[0];
            if (selectionMap[i] == 0)
                continue;

            ImmaturePoint *impt = new ImmaturePoint(x, y, newFrame, selectionMap[i], cornerMap[i], &Hcalib);
            if (!std::isfinite(impt->energyTH))
                delete impt;
            else
                newFrame->immaturePoints.push_back(impt); // create immature point
        }
    }
    //printf("MADE %d IMMATURE POINTS!\n", (int)newFrame->immaturePoints.size());
}

void FullSystem::PrintResult(std::string file)
{
    boost::unique_lock<boost::mutex> lock(trackMutex);
    boost::unique_lock<boost::mutex> crlock(shellPoseMutex);

    std::ofstream myfile;
    myfile.open(file.c_str());
    myfile << std::setprecision(15);

    for (Frame *s : allFrameHistory)
    {
        if (!s->poseValid)
            continue;

        if (setting_onlyLogKFPoses && s->marginalizedAt == s->id)
            continue;

        myfile << s->timestamp
               << " " << s->camToWorld.translation().transpose()
               << " " << s->camToWorld.so3().unit_quaternion().x()
               << " " << s->camToWorld.so3().unit_quaternion().y()
               << " " << s->camToWorld.so3().unit_quaternion().z()
               << " " << s->camToWorld.so3().unit_quaternion().w()
               << "\n";
    }
    myfile.close();
}

void FullSystem::PrintLogLine()
{
    if (frameHessians.size() == 0)
        return;

    if (!setting_debugout_runquiet)
        printf("LOG %d: %.3f fine. Res: %d A, %d L, %d M; (%'d / %'d) forceDrop. a=%f, b=%f. Window %d (%d)\n",
               allKeyFramesHistory.back()->id,
               statistics_lastFineTrackRMSE,
               ef->resInA,
               ef->resInL,
               ef->resInM,
               (int)statistics_numForceDroppedResFwd,
               (int)statistics_numForceDroppedResBwd,
               allKeyFramesHistory.back()->AffG2L.a,
               allKeyFramesHistory.back()->AffG2L.b,
               frameHessians.back()->shell->id - frameHessians.front()->shell->id,
               (int)frameHessians.size());

    if (!setting_logStuff)
        return;

    if (numsLog != 0)
    {
        (*numsLog) << allKeyFramesHistory.back()->id << " "
                   << statistics_lastFineTrackRMSE << " "
                   << (int)statistics_numCreatedPoints << " "
                   << (int)statistics_numActivatedPoints << " "
                   << (int)statistics_numDroppedPoints << " "
                   << (int)statistics_lastNumOptIts << " "
                   << ef->resInA << " "
                   << ef->resInL << " "
                   << ef->resInM << " "
                   << statistics_numMargResFwd << " "
                   << statistics_numMargResBwd << " "
                   << statistics_numForceDroppedResFwd << " "
                   << statistics_numForceDroppedResBwd << " "
                   << frameHessians.back()->AffG2L().a << " "
                   << frameHessians.back()->AffG2L().b << " "
                   << frameHessians.back()->shell->id - frameHessians.front()->shell->id << " "
                   << (int)frameHessians.size() << "\n";
        numsLog->flush();
    }
}

void FullSystem::PrintEigenValLine()
{
    if (!setting_logStuff)
        return;
    if (ef->lastHS.rows() < 12)
        return;

    MatXX Hp = ef->lastHS.bottomRightCorner(ef->lastHS.cols() - CPARS, ef->lastHS.cols() - CPARS);
    MatXX Ha = ef->lastHS.bottomRightCorner(ef->lastHS.cols() - CPARS, ef->lastHS.cols() - CPARS);
    int n = Hp.cols() / 8;
    assert(Hp.cols() % 8 == 0);

    // sub-select
    for (int i = 0; i < n; i++)
    {
        MatXX tmp6 = Hp.block(i * 8, 0, 6, n * 8);
        Hp.block(i * 6, 0, 6, n * 8) = tmp6;

        MatXX tmp2 = Ha.block(i * 8 + 6, 0, 2, n * 8);
        Ha.block(i * 2, 0, 2, n * 8) = tmp2;
    }
    for (int i = 0; i < n; i++)
    {
        MatXX tmp6 = Hp.block(0, i * 8, n * 8, 6);
        Hp.block(0, i * 6, n * 8, 6) = tmp6;

        MatXX tmp2 = Ha.block(0, i * 8 + 6, n * 8, 2);
        Ha.block(0, i * 2, n * 8, 2) = tmp2;
    }

    VecX eigenvaluesAll = ef->lastHS.eigenvalues().real();
    VecX eigenP = Hp.topLeftCorner(n * 6, n * 6).eigenvalues().real();
    VecX eigenA = Ha.topLeftCorner(n * 2, n * 2).eigenvalues().real();
    VecX diagonal = ef->lastHS.diagonal();

    std::sort(eigenvaluesAll.data(), eigenvaluesAll.data() + eigenvaluesAll.size());
    std::sort(eigenP.data(), eigenP.data() + eigenP.size());
    std::sort(eigenA.data(), eigenA.data() + eigenA.size());

    int nz = std::max(100, setting_maxFrames * 10);

    if (eigenAllLog != 0)
    {
        VecX ea = VecX::Zero(nz);
        ea.head(eigenvaluesAll.size()) = eigenvaluesAll;
        (*eigenAllLog) << allKeyFramesHistory.back()->id << " " << ea.transpose() << "\n";
        eigenAllLog->flush();
    }
    if (eigenALog != 0)
    {
        VecX ea = VecX::Zero(nz);
        ea.head(eigenA.size()) = eigenA;
        (*eigenALog) << allKeyFramesHistory.back()->id << " " << ea.transpose() << "\n";
        eigenALog->flush();
    }
    if (eigenPLog != 0)
    {
        VecX ea = VecX::Zero(nz);
        ea.head(eigenP.size()) = eigenP;
        (*eigenPLog) << allKeyFramesHistory.back()->id << " " << ea.transpose() << "\n";
        eigenPLog->flush();
    }

    if (DiagonalLog != 0)
    {
        VecX ea = VecX::Zero(nz);
        ea.head(diagonal.size()) = diagonal;
        (*DiagonalLog) << allKeyFramesHistory.back()->id << " " << ea.transpose() << "\n";
        DiagonalLog->flush();
    }

    if (variancesLog != 0)
    {
        VecX ea = VecX::Zero(nz);
        ea.head(diagonal.size()) = ef->lastHS.inverse().diagonal();
        (*variancesLog) << allKeyFramesHistory.back()->id << " " << ea.transpose() << "\n";
        variancesLog->flush();
    }

    std::vector<VecX> &nsp = ef->lastNullspaces_forLogging;
    (*nullspacesLog) << allKeyFramesHistory.back()->id << " ";
    for (unsigned int i = 0; i < nsp.size(); i++)
        (*nullspacesLog) << nsp[i].dot(ef->lastHS * nsp[i]) << " " << nsp[i].dot(ef->lastbS) << " ";
    (*nullspacesLog) << "\n";
    nullspacesLog->flush();
}

void FullSystem::PrintFrameLifetimes()
{
    if (!setting_logStuff)
        return;

    boost::unique_lock<boost::mutex> lock(trackMutex);

    std::ofstream *lg = new std::ofstream();
    lg->open("logs/lifetimeLog.txt", std::ios::trunc | std::ios::out);
    lg->precision(15);

    for (Frame *s : allFrameHistory)
    {
        (*lg) << s->id
              << " " << s->marginalizedAt
              << " " << s->statistics_goodResOnThis
              << " " << s->statistics_outlierResOnThis
              << " " << s->movedByOpt;

        (*lg) << "\n";
    }

    lg->close();
    delete lg;
}



}
