#include "LoopClosing/pose_graph.hpp"
#include "Utils/utility.hpp"
#include "Utils/tic_toc.h"


using namespace DVision;
using namespace DBoW2;
using namespace Eigen;

#define DEBUG_IMAGE 0
#define USE_KEYFRAME_LIST 0
string POSE_GRAPH_SAVE_PATH = "./";

namespace ds_slam
{

PoseGraph::PoseGraph(std::vector<Frame *> *allKeyFramesHistory_, std::vector<visualizer::Visualizer3D *> *viewers_) 
    : allKeyFramesHistory(allKeyFramesHistory_), viewers(viewers_)
{
    t_drift = Eigen::Vector3d(0, 0, 0);
    yaw_drift = 0;
    r_drift = Eigen::Matrix3d::Identity();
    w_t_vio = Eigen::Vector3d(0, 0, 0);
    w_r_vio = Eigen::Matrix3d::Identity();
    latest_loop_index = 0;

    earliest_loop_index = -1;
    sequence_cnt = 0;
    sequence_loop.push_back(0);
    base_sequence = 1;

    run_optimization = true;
#if USE_KEYFRAME_LIST
    t_optimization = std::thread(&PoseGraph::Optimize7DoF, this); // separation mode
#else
    t_optimization = std::thread(&PoseGraph::OptimizePoseGraph, this); // coupling model
#endif
}

PoseGraph::~PoseGraph()
{
    run_optimization = false;
    t_optimization.join();

    delete voc;
}

void PoseGraph::LoadVocabulary(std::string voc_path)
{
#if 1
    // Brief
    voc = new BriefVocabulary();
    voc->loadBin(voc_path);
    db.setVocabulary(*voc, false, 0);
#else
    // ORB
    voc_orb = new ORBVocabulary();
    voc_orb->loadFromBinaryFile(voc_path);
    db_orb.setVocabulary(*voc, false, 0);
#endif
}

void PoseGraph::AddKeyFrame(KeyFrame *cur_kf, bool flag_detect_loop)
{
    //shift to base frame
    Vector3d vio_P_cur;
    Matrix3d vio_R_cur;
    if (sequence_cnt != cur_kf->sequence)
    {
        sequence_cnt++; // clear statue for new sequence
        sequence_loop.push_back(0);
        w_t_vio = Eigen::Vector3d(0, 0, 0);
        w_r_vio = Eigen::Matrix3d::Identity();
        m_drift.lock();
        t_drift = Eigen::Vector3d(0, 0, 0);
        r_drift = Eigen::Matrix3d::Identity();
        m_drift.unlock();
    }

    cur_kf->GetVoPose(vio_P_cur, vio_R_cur);
    vio_P_cur = w_r_vio * vio_P_cur + w_t_vio;
    vio_R_cur = w_r_vio * vio_R_cur;
    cur_kf->UpdateVoPose(vio_P_cur, vio_R_cur);
    int loop_index = -1;
    if (flag_detect_loop)
    {
        loop_index = DetectLoop(cur_kf, cur_kf->index);
    }
    else
    {
        AddKeyFrameIntoVoc(cur_kf);
    }
    if (loop_index != -1)
    {
        latest_loop_index = cur_kf->index;
        //printf(" %d detect loop with %d \n", cur_kf->index, loop_index);

        KeyFrame *old_kf = GetKeyFrame(loop_index);
        if (cur_kf->FindConnection(old_kf))
        {
            // record the earliset loop index of history keyframes
            if (earliest_loop_index > loop_index || earliest_loop_index == -1)
                earliest_loop_index = loop_index;

            Vector3d relative_t;
            Quaterniond relative_q;
            relative_t = cur_kf->GetLoopRelativeT();
            relative_q = (cur_kf->GetLoopRelativeQ()).toRotationMatrix();

            // record the relative poses between current frame and history loop frame
            cur_kf->frame->mutexPoseRel.lock();
            Matrix4d T_ref_cur;
            T_ref_cur.setIdentity();
            T_ref_cur.block(0, 0, 3, 3) = relative_q.matrix();
            T_ref_cur.block(0, 3, 3, 1) = relative_t;
            cur_kf->frame->poseRel[old_kf->frame] = Frame::RELPOSE(Sim3(T_ref_cur), Mat77::Identity(), true);
            cur_kf->frame->mutexPoseRel.unlock();

            // calculate the shift
            Vector3d w_P_old, w_P_cur, vio_P_cur;
            Matrix3d w_R_old, w_R_cur, vio_R_cur;
            old_kf->GetVoPose(w_P_old, w_R_old);
            cur_kf->GetVoPose(vio_P_cur, vio_R_cur);
            w_P_cur = w_R_old * relative_t + w_P_old;
            w_R_cur = w_R_old * relative_q;
#if 0
            double shift_yaw;
            Matrix3d shift_r;
            Vector3d shift_t;
            shift_yaw = Utility::R2ypr(w_R_cur).x() - Utility::R2ypr(vio_R_cur).x(); // calculate the shift (4DOF)
            shift_r = Utility::ypr2R(Vector3d(shift_yaw, 0, 0));
            shift_t = w_P_cur - w_R_cur * vio_R_cur.transpose() * vio_P_cur;
#else
            Matrix3d shift_r = w_R_cur * vio_R_cur.transpose(); // calculate the shift
            Vector3d shift_t = w_P_cur - w_R_cur * vio_R_cur.transpose() * vio_P_cur;
#endif
            // shift pose of whole sequence to the world frame, relocation!!!
            if (old_kf->sequence != cur_kf->sequence && sequence_loop[cur_kf->sequence] == 0)
            {
                w_r_vio = shift_r;
                w_t_vio = shift_t;
                vio_P_cur = w_r_vio * vio_P_cur + w_t_vio;
                vio_R_cur = w_r_vio * vio_R_cur;
                cur_kf->UpdateVoPose(vio_P_cur, vio_R_cur);
                list<KeyFrame *>::iterator it = keyframelist.begin();
                for (; it != keyframelist.end(); it++)
                {
                    if ((*it)->sequence == cur_kf->sequence)
                    {
                        Vector3d vio_P_cur;
                        Matrix3d vio_R_cur;
                        (*it)->GetVoPose(vio_P_cur, vio_R_cur);
                        vio_P_cur = w_r_vio * vio_P_cur + w_t_vio;
                        vio_R_cur = w_r_vio * vio_R_cur;
                        (*it)->UpdateVoPose(vio_P_cur, vio_R_cur); // reset pose of keyframe
                    }
                }
                sequence_loop[cur_kf->sequence] = 1; // flag relocation success for the sequence

                // shift all keyframe of system
                for (std::vector<Frame *>::iterator itr = allKeyFramesHistory->begin();
                     itr != allKeyFramesHistory->end(); itr++)
                {
                    if((*itr)->sequence == cur_kf->sequence)
                    {
                        Vector3d vio_P_cur = (*itr)->TwcOpti.translation();
                        Matrix3d vio_R_cur = (*itr)->TwcOpti.rotationMatrix();
                        vio_P_cur = w_r_vio * vio_P_cur + w_t_vio;
                        vio_R_cur = w_r_vio * vio_R_cur;
                        Matrix4d vio_T_cur;
                        vio_T_cur.setIdentity();
                        vio_T_cur.block(0, 0, 3, 3) = vio_R_cur;
                        vio_T_cur.block(0, 3, 3, 1) = vio_P_cur;
                        (*itr)->poseMutex.lock();
                        (*itr)->TwcOpti = Sim3(vio_T_cur);
                        (*itr)->poseMutex.unlock();
                    }
                }
            }
            m_optimize_buf.lock();
            optimize_buf.push(cur_kf->index); // feeding to optimization
            m_optimize_buf.unlock();
        }
    }
    m_keyframelist.lock();
    Vector3d P;
    Matrix3d R;
    cur_kf->GetVoPose(P, R);
    P = r_drift * P + t_drift;
    R = r_drift * R;
    cur_kf->UpdatePose(P, R); // remove drift and update pose
    keyframelist.push_back(cur_kf);
    m_keyframelist.unlock();
}

KeyFrame *PoseGraph::GetKeyFrame(int index)
{
    //unique_lock<mutex> lock(m_keyframelist);
    list<KeyFrame *>::iterator it = keyframelist.begin();
    for (; it != keyframelist.end(); it++)
    {
        if ((*it)->index == index)
            break;
    }
    if (it != keyframelist.end())
        return *it;
    else
        return NULL;
}

Frame *PoseGraph::GetKeyFrameHistory(int index)
{
    vector<Frame *>::iterator it;
    for (; it != (*allKeyFramesHistory).end(); it++)
    {
        if ((*it)->index == index)
            break;
    }
    if (it != (*allKeyFramesHistory).end())
        return *it;
    else
        return NULL;
}

int PoseGraph::DetectLoop(KeyFrame *keyframe, int frame_index)
{
    // put image into image_pool; for visualization
    cv::Mat compressed_image;
    if (DEBUG_IMAGE)
    {
        int feature_num = keyframe->keypoints.size();
        cv::resize(keyframe->image, compressed_image, cv::Size(376, 240));
        putText(compressed_image, "feature_num:" + to_string(feature_num), cv::Point2f(10, 10), CV_FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255));
        image_pool[frame_index] = compressed_image;
    }

    //first query; then add this frame into database!
    QueryResults ret;
    db.query(keyframe->brief_descriptors, ret, 0, frame_index - 50); // configure parameter
    //printf("query time: %f", t_query.toc());
    //cout << "Searching for Image " << frame_index << ". " << ret << endl;
    db.add(keyframe->brief_descriptors);
    //printf("add feature time: %f", t_add.toc());
    // ret[0] is the nearest neighbour's score. threshold change with neighour score

    // visual loop result
    cv::Mat loop_result;
    if (DEBUG_IMAGE)
    {
        loop_result = compressed_image.clone();
        if (ret.size() > 0)
            putText(loop_result, "neighbour score:" + to_string(ret[0].Score), cv::Point2f(10, 50), CV_FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255));

        for (unsigned int i = 0; i < ret.size(); i++)
        {
            int tmp_index = ret[i].Id;
            auto it = image_pool.find(tmp_index);
            cv::Mat tmp_image = (it->second).clone();
            putText(tmp_image, "index:  " + to_string(tmp_index) + "loop score:" + to_string(ret[i].Score), cv::Point2f(10, 50), CV_FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255));
            cv::hconcat(loop_result, tmp_image, loop_result);
        }
    }

    // a good match with its nerghbour
    bool find_loop = false;
    if (ret.size() >= 1 && ret[0].Score > 0.05)
    {
        for (unsigned int i = 1; i < ret.size(); i++)
        {
            //if (ret[i].Score > ret[0].Score * 0.3)
            if (ret[i].Score > 0.015)
            {
                find_loop = true;
                int tmp_index = ret[i].Id;
                if (DEBUG_IMAGE && 0)
                {
                    auto it = image_pool.find(tmp_index);
                    cv::Mat tmp_image = (it->second).clone();
                    putText(tmp_image, "loop score:" + to_string(ret[i].Score), cv::Point2f(10, 50), CV_FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255));
                    cv::hconcat(loop_result, tmp_image, loop_result);
                }
            }
        }
    }
    if (DEBUG_IMAGE)
    {
        cv::imshow("loop_result", loop_result);
        cv::waitKey(20);
    }

    if (find_loop && frame_index > 50)
    {
        int min_index = -1;
        for (unsigned int i = 0; i < ret.size(); i++)
        {
            if (min_index == -1 || ((int)ret[i].Id < min_index && ret[i].Score > 0.015))
                min_index = ret[i].Id;
        }
        return min_index;
    }
    else
        return -1;
}

void PoseGraph::AddKeyFrameIntoVoc(KeyFrame *keyframe)
{
    // put image into image_pool; for visualization
    cv::Mat compressed_image;
    if (DEBUG_IMAGE)
    {
        int feature_num = keyframe->keypoints.size();
        cv::resize(keyframe->image, compressed_image, cv::Size(376, 240));
        putText(compressed_image, "feature_num:" + to_string(feature_num), cv::Point2f(10, 10), CV_FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255));
        image_pool[keyframe->index] = compressed_image;
    }

    db.add(keyframe->brief_descriptors);
}

void PoseGraph::Optimize7DoF()
{
    while (run_optimization)
    {
        int cur_index = -1;
        int first_looped_index = -1;
        m_optimize_buf.lock();
        while (!optimize_buf.empty()) // select the newest loop frame
        {
            cur_index = optimize_buf.front();
            first_looped_index = earliest_loop_index;
            optimize_buf.pop();
        }
        m_optimize_buf.unlock();

        if(!run_optimization)
            break;
        if (cur_index != -1)
        {
            printf("optimize pose graph \n");
            TicToc tmp_t;
            m_keyframelist.lock();
            KeyFrame *cur_kf = GetKeyFrame(cur_index);

            int max_length = cur_index + 1;

            // w^t_i   w^q_i
            double t_array[max_length][3];
            Quaterniond q_array[max_length];
            double euler_array[max_length][3];
            double scale_array[max_length];
            double sequence_array[max_length];

            ceres::Problem problem;
            ceres::Solver::Options options;
            options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
            //options.minimizer_progress_to_stdout = true;
            //options.max_solver_time_in_seconds = SOLVER_TIME * 3;
            options.max_num_iterations = 5;
            ceres::Solver::Summary summary;
            ceres::LossFunction *loss_function;
            loss_function = new ceres::HuberLoss(0.1);
            //loss_function = new ceres::CauchyLoss(1.0);
            ceres::LocalParameterization *angle_local_parameterization = AngleLocalParameterization::Create();

            list<KeyFrame *>::iterator it;

            int i = 0;
            for (it = keyframelist.begin(); it != keyframelist.end(); it++)
            {
                if ((*it)->index < first_looped_index)
                    continue;
                (*it)->local_index = i; // mark local loop index
                Quaterniond tmp_q;
                Matrix3d tmp_r;
                Vector3d tmp_t;
                (*it)->GetVoPose(tmp_t, tmp_r);
                tmp_q = tmp_r;
                t_array[i][0] = tmp_t(0);
                t_array[i][1] = tmp_t(1);
                t_array[i][2] = tmp_t(2);
                q_array[i] = tmp_q;

                Vector3d euler_angle = Utility::R2ypr(tmp_q.toRotationMatrix());
                euler_array[i][0] = euler_angle.x();
                euler_array[i][1] = euler_angle.y();
                euler_array[i][2] = euler_angle.z();
                scale_array[i] = 1.0;

                sequence_array[i] = (*it)->sequence;

                problem.AddParameterBlock(euler_array[i], 3, angle_local_parameterization);
                problem.AddParameterBlock(t_array[i], 3);
                problem.AddParameterBlock(scale_array + i, 1);

                if ((*it)->index == first_looped_index || (*it)->sequence == 0) // first index must be fixed for consistency graph!
                {
                    problem.SetParameterBlockConstant(euler_array[i]);
                    problem.SetParameterBlockConstant(t_array[i]);
                }

                //add edge
                for (int j = 1; j < 5; j++)
                {
                    if (i - j >= 0 && sequence_array[i] == sequence_array[i - j])
                    {
                        //Vector3d euler_conncected = Utility::R2ypr(q_array[i - j].toRotationMatrix());
                        Vector3d relative_t(t_array[i][0] - t_array[i - j][0], t_array[i][1] - t_array[i - j][1], t_array[i][2] - t_array[i - j][2]);
                        relative_t = q_array[i - j].inverse() * relative_t;
                        double relative_yaw = euler_array[i][0] - euler_array[i - j][0];
                        double relative_pitch = euler_array[i][1] - euler_array[i - j][1];
                        double relative_roll = euler_array[i][2] - euler_array[i - j][2];
                        ceres::CostFunction *cost_function = SevenDOFError::Create(relative_t.x(), relative_t.y(), relative_t.z(),
                                                                                  relative_yaw, relative_pitch, relative_roll,
                                                                                  0);
                        problem.AddResidualBlock(cost_function,
                                                 NULL,
                                                 euler_array[i - j],
                                                 t_array[i - j],
                                                 scale_array + i - j,
                                                 euler_array[i],
                                                 t_array[i],
                                                 scale_array + i);
                    }
                }

                //add loop edge
                if ((*it)->has_loop)
                {
                    assert((*it)->loop_index >= first_looped_index);
                    int connected_index = GetKeyFrame((*it)->loop_index)->local_index;
                    //Vector3d euler_conncected = Utility::R2ypr(q_array[connected_index].toRotationMatrix());
                    Vector3d relative_t;
                    relative_t = (*it)->GetLoopRelativeT();
                    Eigen::Quaterniond relative_Q = (*it)->GetLoopRelativeQ();
                    Vector3d relative_ypr = Utility::R2ypr(relative_Q.matrix());
                    ceres::CostFunction *cost_function = SevenDOFWeightError::Create(relative_t.x(), relative_t.y(), relative_t.z(),
                                                                                     relative_ypr.x(), relative_ypr.y(), relative_ypr.z(),
                                                                                     0);
                    problem.AddResidualBlock(cost_function,
                                             loss_function,
                                             euler_array[connected_index],
                                             t_array[connected_index],
                                             scale_array + connected_index,
                                             euler_array[i],
                                             t_array[i],
                                             scale_array + i);
                }

                if ((*it)->index == cur_index)
                    break;
                i++;
            }
            m_keyframelist.unlock();

            ceres::Solve(options, &problem, &summary);
            //std::cout << summary.BriefReport() << "\n";

            //printf("pose optimization time: %f \n", tmp_t.toc());
            //for (int j = 0 ; j < i; j++)
            //{
            //    printf("optimize i: %d p: %f, %f, %f\n", j, t_array[j][0], t_array[j][1], t_array[j][2]);
            //}
            
            // update local loop pose
            m_keyframelist.lock();
            i = 0;
            for (it = keyframelist.begin(); it != keyframelist.end(); it++)
            {
                if ((*it)->index < first_looped_index)
                    continue;
                Quaterniond tmp_q;
                tmp_q = Utility::ypr2R(Vector3d(euler_array[i][0], euler_array[i][1], euler_array[i][2]));
                Vector3d tmp_t = Vector3d(t_array[i][0], t_array[i][1], t_array[i][2]);
                Matrix3d tmp_r = tmp_q.toRotationMatrix();
                (*it)->UpdatePose(tmp_t, tmp_r);
                (*it)->UpdateScale(scale_array[i]);
                if ((*it)->index == cur_index)
                    break;
                i++;
            }

            // calculate drift of current loop frame
            Vector3d cur_t, vio_t;
            Matrix3d cur_r, vio_r;
            cur_kf->GetPose(cur_t, cur_r);
            cur_kf->GetVoPose(vio_t, vio_r);
            m_drift.lock();
            yaw_drift = Utility::R2ypr(cur_r).x() - Utility::R2ypr(vio_r).x();
            r_drift = Utility::ypr2R(Vector3d(yaw_drift, 0, 0));
            t_drift = cur_t - r_drift * vio_t;
            m_drift.unlock();
            //cout << "t_drift " << t_drift.transpose() << endl;
            //cout << "r_drift " << Utility::R2ypr(r_drift).transpose() << endl;
            //cout << "yaw drift " << yaw_drift << endl;

            // update later pose of current loop frame
            it++;
            for (; it != keyframelist.end(); it++) 
            {
                Vector3d P;
                Matrix3d R;
                (*it)->GetVoPose(P, R);
                P = r_drift * P + t_drift;
                R = r_drift * R;
                (*it)->UpdatePose(P, R);
            }
            m_keyframelist.unlock();
            UpdatePath();
        }

        std::chrono::milliseconds dura(2000);
        std::this_thread::sleep_for(dura);
    }
}

void PoseGraph::OptimizePoseGraph()
{
    while (run_optimization)
    {
        int cur_index = -1;
        int first_looped_index = -1;
        m_optimize_buf.lock();
        while (!optimize_buf.empty()) // select the newest loop frame
        {
            cur_index = optimize_buf.front();
            first_looped_index = earliest_loop_index;
            optimize_buf.pop();
        }
        m_optimize_buf.unlock();

        if(!run_optimization)
            break;
        if (cur_index != -1)
        {
            printf("optimize pose graph \n");
            TicToc tmp_t;
            m_keyframelist.lock();
            KeyFrame *cur_kf = GetKeyFrame(cur_index);

            int max_length = cur_index + 1;

            // w^t_i   w^q_i
            double t_array[max_length][3];
            Quaterniond q_array[max_length];
            double euler_array[max_length][3];
            double scale_array[max_length];
            int sequence_array[max_length];

            ceres::Problem problem;
            ceres::Solver::Options options;
            options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
            //options.minimizer_progress_to_stdout = true;
            //options.max_solver_time_in_seconds = SOLVER_TIME * 3;
            options.max_num_iterations = 5;
            ceres::Solver::Summary summary;
            ceres::LossFunction *loss_function;
            loss_function = new ceres::HuberLoss(0.1);
            //loss_function = new ceres::CauchyLoss(1.0);
            ceres::LocalParameterization *angle_local_parameterization = AngleLocalParameterization::Create();

            vector<Frame *>::iterator it;
            int i = 0;
            for (it = (*allKeyFramesHistory).begin(); it != (*allKeyFramesHistory).end(); it++)
            {
                if ((*it)->index < first_looped_index)
                    continue;
                (*it)->local_index = i; // mark local loop index
                Quaterniond tmp_q;
                Matrix3d tmp_r;
                Vector3d tmp_t;
                //(*it)->GetVoPose(tmp_t, tmp_r);
                tmp_t = (*it)->TwcOpti.translation();
                tmp_r = (*it)->TwcOpti.rotationMatrix();
                tmp_q = tmp_r;
                t_array[i][0] = tmp_t(0);
                t_array[i][1] = tmp_t(1);
                t_array[i][2] = tmp_t(2);
                q_array[i] = tmp_q;

                Vector3d euler_angle = Utility::R2ypr(tmp_q.toRotationMatrix());
                euler_array[i][0] = euler_angle.x();
                euler_array[i][1] = euler_angle.y();
                euler_array[i][2] = euler_angle.z();
                scale_array[i] = 1.0;

                sequence_array[i] = (*it)->sequence;

                problem.AddParameterBlock(euler_array[i], 3, angle_local_parameterization);
                problem.AddParameterBlock(t_array[i], 3);
                problem.AddParameterBlock(scale_array + i, 1);

                if ((*it)->index == first_looped_index || (*it)->sequence == 0) // first index must be fixed for consistency graph!
                {
                    problem.SetParameterBlockConstant(euler_array[i]);
                    problem.SetParameterBlockConstant(t_array[i]);
                }

#if 0
                // construct pose graph by order
                // add edge
                for (int j = 1; j < 5; j++)
                {
                    if (i - j >= 0 && sequence_array[i] == sequence_array[i - j])
                    {
                        Vector3d euler_conncected = Utility::R2ypr(q_array[i - j].toRotationMatrix());
                        Vector3d relative_t(t_array[i][0] - t_array[i - j][0], t_array[i][1] - t_array[i - j][1], t_array[i][2] - t_array[i - j][2]);
                        relative_t = q_array[i - j].inverse() * relative_t;
                        double relative_yaw = euler_array[i][0] - euler_array[i - j][0];
                        double relative_pitch = euler_array[i][1] - euler_array[i - j][1];
                        double relative_roll = euler_array[i][2] - euler_array[i - j][2];
                        ceres::CostFunction *cost_function = SevenDOFError::Create(relative_t.x(), relative_t.y(), relative_t.z(),
                                                                                  relative_yaw, relative_pitch, relative_roll,
                                                                                  0);
                        problem.AddResidualBlock(cost_function,
                                                 NULL,
                                                 euler_array[i - j],
                                                 t_array[i - j],
                                                 scale_array + i - j,
                                                 euler_array[i],
                                                 t_array[i],
                                                 scale_array + i);
                    }
                }

                //add loop edge
                KeyFrame *kf = GetKeyFrame((*it)->index);
                if (kf && kf->has_loop)
                {
                    assert(kf->loop_index >= first_looped_index);
                    int connected_index = (*it)->local_index;
                    Vector3d euler_conncected = Utility::R2ypr(q_array[connected_index].toRotationMatrix());
                    Vector3d relative_t;
                    relative_t = (*it)->GetLoopRelativeT();
                    Eigen::Quaterniond relative_Q = (*it)->GetLoopRelativeQ();
                    Vector3d relative_ypr = Utility::R2ypr(relative_Q.matrix());
                    ceres::CostFunction *cost_function = RelativePoseWeightError::Create(relative_t.x(), relative_t.y(), relative_t.z(),
                                                                                     relative_ypr.x(), relative_ypr.y(), relative_ypr.z(),
                                                                                     0);
                    problem.AddResidualBlock(cost_function,
                                             loss_function,
                                             euler_array[connected_index],
                                             t_array[connected_index],
                                             scale_array + connected_index,
                                             euler_array[i],
                                             t_array[i],
                                             scale_array + i);
                }
#else
                // construct pose graph by covisibility graph
                (*it)->mutexPoseRel.lock();
                for(auto rel : (*it)->poseRel)
                {
                    if(!rel.second.isLoop)
                    {
                        //add edge
                        if (rel.first->index >= first_looped_index && (*it)->sequence == rel.first->sequence)
                        {
                            int ref_id = rel.first->index - first_looped_index;
                            int cur_id = (*it)->index - first_looped_index;
                            assert(cur_id == i);
                            ceres::CostFunction *cost_function = RelativePoseError::Create(rel.second.Trc);
                            problem.AddResidualBlock(cost_function,
                                                     NULL,
                                                     euler_array[ref_id],
                                                     t_array[ref_id],
                                                     scale_array + ref_id,
                                                     euler_array[i],
                                                     t_array[i],
                                                     scale_array + i);
                        }
                    }
                    else
                    {
                        //add loop edge
                        KeyFrame *kf = GetKeyFrame((*it)->index);
                        assert(kf && kf->has_loop);
                        assert(kf->loop_index >= first_looped_index);
                        int connected_index = (*it)->local_index;
                        ceres::CostFunction *cost_function = RelativePoseWeightError::Create(rel.second.Trc);
                        (*it)->mutexPoseRel.unlock();
                        problem.AddResidualBlock(cost_function,
                                                 loss_function,
                                                 euler_array[connected_index],
                                                 t_array[connected_index],
                                                 scale_array + connected_index,
                                                 euler_array[i],
                                                 t_array[i],
                                                 scale_array + i);
                    }
                }
                (*it)->mutexPoseRel.unlock();
#endif

                if ((*it)->index == cur_index)
                    break;
                i++;
            }
            m_keyframelist.unlock();

            ceres::Solve(options, &problem, &summary);
            //std::cout << summary.BriefReport() << "\n";

            //printf("pose optimization time: %f \n", tmp_t.toc());
            //for (int j = 0 ; j < i; j++)
            //{
            //    printf("optimize i: %d p: %f, %f, %f\n", j, t_array[j][0], t_array[j][1], t_array[j][2]);
            //}
            
            // update local loop pose
            m_keyframelist.lock();
            i = 0;
            for (it = (*allKeyFramesHistory).begin(); it != (*allKeyFramesHistory).end(); it++)
            {
                if ((*it)->index < first_looped_index)
                    continue;
                Quaterniond tmp_q;
                tmp_q = Utility::ypr2R(Vector3d(euler_array[i][0], euler_array[i][1], euler_array[i][2]));
                Vector3d tmp_t = Vector3d(t_array[i][0], t_array[i][1], t_array[i][2]);
                Matrix3d tmp_r = tmp_q.toRotationMatrix();
                KeyFrame *kf = GetKeyFrame((*it)->index);
                if (kf)
                {
                    kf->UpdatePose(tmp_t, tmp_r);
                    kf->UpdateScale(scale_array[i]);
                }
                (*it)->poseMutex.lock();
                Matrix4d TwcOpti_new;
                TwcOpti_new.setIdentity();
                TwcOpti_new.block(0, 0, 3, 3) = tmp_t;
                TwcOpti_new.block(0, 3, 3, 1) = scale_array[i] * tmp_r;
                (*it)->TwcOpti = Sim3(TwcOpti_new);
                (*it)->poseMutex.unlock();

                if ((*it)->index == cur_index)
                    break;
                i++;
            }

            // calculate drift of current loop frame
            Vector3d cur_t, vio_t;
            Matrix3d cur_r, vio_r;
            cur_kf->GetPose(cur_t, cur_r);
            cur_kf->GetVoPose(vio_t, vio_r);
            m_drift.lock();
            ypr_drift = Utility::R2ypr(cur_r) - Utility::R2ypr(vio_r);
            r_drift = Utility::ypr2R(ypr_drift);
            t_drift = cur_t - r_drift * vio_t;
            scale_drift = cur_kf->scale - cur_kf->vo_scale;
            t_drift = (1 + scale_drift) * t_drift;
            m_drift.unlock();
            //cout << "t_drift " << t_drift.transpose() << endl;
            //cout << "r_drift " << Utility::R2ypr(r_drift).transpose() << endl;
            //cout << "yaw drift " << yaw_drift << endl;

            // update later pose of current loop frame
            it++;
            for (; it != (*allKeyFramesHistory).end(); it++) 
            {
                Vector3d P;
                Matrix3d R;
                KeyFrame *kf = GetKeyFrame((*it)->index);
                if (kf)
                {
                    kf->GetVoPose(P, R);
                    P = r_drift * P + t_drift;
                    R = r_drift * R;
                    kf->UpdatePose(P, R);
                }
                P = (*it)->TwcOpti.translation();
                R = (*it)->TwcOpti.rotationMatrix();
                P = r_drift * P + t_drift;
                P = (1 + scale_drift) * P;
                R = r_drift * R;
                Matrix4d TwcOpti_new;
                TwcOpti_new.setIdentity();
                TwcOpti_new.block(0, 0, 3, 3) = P;
                TwcOpti_new.block(0, 3, 3, 1) = R;
                (*it)->TwcOpti = Sim3(TwcOpti_new);
            }
            m_keyframelist.unlock();
            UpdatePath();
        }

        std::chrono::milliseconds dura(2000);
        std::this_thread::sleep_for(dura);
    }
}

void PoseGraph::UpdatePath()
{
    m_keyframelist.lock();
    //todo
    m_keyframelist.unlock();
}

void PoseGraph::SavePoseGraph()
{
    m_keyframelist.lock();
    TicToc tmp_t;
    FILE *pFile;
    printf("pose graph path: %s\n",POSE_GRAPH_SAVE_PATH.c_str());
    printf("pose graph saving... \n");
    string file_path = POSE_GRAPH_SAVE_PATH + "pose_graph.txt";
    pFile = fopen (file_path.c_str(),"w");
    //fprintf(pFile, "index time_stamp Tx Ty Tz Qw Qx Qy Qz loop_index loop_info\n");
    list<KeyFrame*>::iterator it;
    for (it = keyframelist.begin(); it != keyframelist.end(); it++)
    {
        std::string image_path, descriptor_path, brief_path, keypoints_path;
        if (DEBUG_IMAGE)
        {
            image_path = POSE_GRAPH_SAVE_PATH + to_string((*it)->index) + "_image.png";
            imwrite(image_path.c_str(), (*it)->image);
        }
        Quaterniond VIO_tmp_Q{(*it)->vo_R_w_i};
        Quaterniond PG_tmp_Q{(*it)->R_w_i};
        Vector3d VIO_tmp_T = (*it)->vo_T_w_i;
        Vector3d PG_tmp_T = (*it)->T_w_i;

        fprintf (pFile, " %d %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %d %f %f %f %f %f %f %f %f %d\n",(*it)->index, (*it)->time_stamp, 
                                    VIO_tmp_T.x(), VIO_tmp_T.y(), VIO_tmp_T.z(), 
                                    PG_tmp_T.x(), PG_tmp_T.y(), PG_tmp_T.z(), 
                                    VIO_tmp_Q.w(), VIO_tmp_Q.x(), VIO_tmp_Q.y(), VIO_tmp_Q.z(), 
                                    PG_tmp_Q.w(), PG_tmp_Q.x(), PG_tmp_Q.y(), PG_tmp_Q.z(), 
                                    (*it)->loop_index, 
                                    (*it)->loop_info(0), (*it)->loop_info(1), (*it)->loop_info(2), (*it)->loop_info(3),
                                    (*it)->loop_info(4), (*it)->loop_info(5), (*it)->loop_info(6), (*it)->loop_info(7),
                                    (int)(*it)->keypoints.size());

        // write keypoints, brief_descriptors   vector<cv::KeyPoint> keypoints vector<BRIEF::bitset> brief_descriptors;
        assert((*it)->keypoints.size() == (*it)->brief_descriptors.size());
        brief_path = POSE_GRAPH_SAVE_PATH + to_string((*it)->index) + "_briefdes.dat";
        std::ofstream brief_file(brief_path, std::ios::binary);
        keypoints_path = POSE_GRAPH_SAVE_PATH + to_string((*it)->index) + "_keypoints.txt";
        FILE *keypoints_file;
        keypoints_file = fopen(keypoints_path.c_str(), "w");
        for (int i = 0; i < (int)(*it)->keypoints.size(); i++)
        {
            brief_file << (*it)->brief_descriptors[i] << endl;
            fprintf(keypoints_file, "%f %f %f %f\n", (*it)->keypoints[i].pt.x, (*it)->keypoints[i].pt.y, 
                                                     (*it)->keypoints_norm[i].pt.x, (*it)->keypoints_norm[i].pt.y);
        }
        brief_file.close();
        fclose(keypoints_file);
    }
    fclose(pFile);

    printf("save pose graph time: %f s\n", tmp_t.toc() / 1000);
    m_keyframelist.unlock();
}

void PoseGraph::LoadPoseGraph()
{
    TicToc tmp_t;
    FILE * pFile;
    string file_path = POSE_GRAPH_SAVE_PATH + "pose_graph.txt";
    printf("lode pose graph from: %s \n", file_path.c_str());
    printf("pose graph loading...\n");
    pFile = fopen (file_path.c_str(),"r");
    if (pFile == NULL)
    {
        printf("lode previous pose graph error: wrong previous pose graph path or no previous pose graph \n the system will start with new pose graph \n");
        return;
    }
    int index;
    double time_stamp;
    double VIO_Tx, VIO_Ty, VIO_Tz;
    double PG_Tx, PG_Ty, PG_Tz;
    double VIO_Qw, VIO_Qx, VIO_Qy, VIO_Qz;
    double PG_Qw, PG_Qx, PG_Qy, PG_Qz;
    double loop_info_0, loop_info_1, loop_info_2, loop_info_3;
    double loop_info_4, loop_info_5, loop_info_6, loop_info_7;
    int loop_index;
    int keypoints_num;
    Eigen::Matrix<double, 8, 1 > loop_info;
    int cnt = 0;
    while (fscanf(pFile,"%d %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %d %lf %lf %lf %lf %lf %lf %lf %lf %d", &index, &time_stamp, 
                                    &VIO_Tx, &VIO_Ty, &VIO_Tz, 
                                    &PG_Tx, &PG_Ty, &PG_Tz, 
                                    &VIO_Qw, &VIO_Qx, &VIO_Qy, &VIO_Qz, 
                                    &PG_Qw, &PG_Qx, &PG_Qy, &PG_Qz, 
                                    &loop_index,
                                    &loop_info_0, &loop_info_1, &loop_info_2, &loop_info_3, 
                                    &loop_info_4, &loop_info_5, &loop_info_6, &loop_info_7,
                                    &keypoints_num) != EOF) 
    {
        /*
        printf("I read: %d %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %d %lf %lf %lf %lf %lf %lf %lf %lf %d\n", index, time_stamp, 
                                    VIO_Tx, VIO_Ty, VIO_Tz, 
                                    PG_Tx, PG_Ty, PG_Tz, 
                                    VIO_Qw, VIO_Qx, VIO_Qy, VIO_Qz, 
                                    PG_Qw, PG_Qx, PG_Qy, PG_Qz, 
                                    loop_index,
                                    loop_info_0, loop_info_1, loop_info_2, loop_info_3, 
                                    loop_info_4, loop_info_5, loop_info_6, loop_info_7,
                                    keypoints_num);
        */
        cv::Mat image;
        std::string image_path, descriptor_path;
        if (DEBUG_IMAGE)
        {
            image_path = POSE_GRAPH_SAVE_PATH + to_string(index) + "_image.png";
            image = cv::imread(image_path.c_str(), 0);
        }

        Vector3d VIO_T(VIO_Tx, VIO_Ty, VIO_Tz);
        Vector3d PG_T(PG_Tx, PG_Ty, PG_Tz);
        Quaterniond VIO_Q;
        VIO_Q.w() = VIO_Qw;
        VIO_Q.x() = VIO_Qx;
        VIO_Q.y() = VIO_Qy;
        VIO_Q.z() = VIO_Qz;
        Quaterniond PG_Q;
        PG_Q.w() = PG_Qw;
        PG_Q.x() = PG_Qx;
        PG_Q.y() = PG_Qy;
        PG_Q.z() = PG_Qz;
        Matrix3d VIO_R, PG_R;
        VIO_R = VIO_Q.toRotationMatrix();
        PG_R = PG_Q.toRotationMatrix();
        Eigen::Matrix<double, 8, 1 > loop_info;
        loop_info << loop_info_0, loop_info_1, loop_info_2, loop_info_3, loop_info_4, loop_info_5, loop_info_6, loop_info_7;

        if (loop_index != -1)
        {
            if (earliest_loop_index > loop_index || earliest_loop_index == -1)
            {
                earliest_loop_index = loop_index;
            }
        }

        // load keypoints, brief_descriptors   
        string brief_path = POSE_GRAPH_SAVE_PATH + to_string(index) + "_briefdes.dat";
        std::ifstream brief_file(brief_path, std::ios::binary);
        string keypoints_path = POSE_GRAPH_SAVE_PATH + to_string(index) + "_keypoints.txt";
        FILE *keypoints_file;
        keypoints_file = fopen(keypoints_path.c_str(), "r");
        vector<cv::KeyPoint> keypoints;
        vector<cv::KeyPoint> keypoints_norm;
        vector<BRIEF::bitset> brief_descriptors;
        for (int i = 0; i < keypoints_num; i++)
        {
            BRIEF::bitset tmp_des;
            brief_file >> tmp_des;
            brief_descriptors.push_back(tmp_des);
            cv::KeyPoint tmp_keypoint;
            cv::KeyPoint tmp_keypoint_norm;
            double p_x, p_y, p_x_norm, p_y_norm;
            if(!fscanf(keypoints_file,"%lf %lf %lf %lf", &p_x, &p_y, &p_x_norm, &p_y_norm))
                printf(" fail to load pose graph \n");
            tmp_keypoint.pt.x = p_x;
            tmp_keypoint.pt.y = p_y;
            tmp_keypoint_norm.pt.x = p_x_norm;
            tmp_keypoint_norm.pt.y = p_y_norm;
            keypoints.push_back(tmp_keypoint);
            keypoints_norm.push_back(tmp_keypoint_norm);
        }
        brief_file.close();
        fclose(keypoints_file);

        //KeyFrame* keyframe = new KeyFrame(time_stamp, index, VIO_T, VIO_R, PG_T, PG_R, image, loop_index, loop_info, keypoints, keypoints_norm, brief_descriptors);
        //loadKeyFrame(keyframe, 0);
        if (cnt % 20 == 0)
        {
            //ublish();
        }
        cnt++;
    }
    fclose (pFile);
    printf("load pose graph time: %f s\n", tmp_t.toc()/1000);
    base_sequence = 0;
}

}