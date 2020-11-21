#include "LoopClosing/keyframe.hpp"
#include "brief_extractor.hpp"
#include "orb_extractor.hpp"
#include "Utils/global_calib.h"
#include "Utils/utility.hpp"


using namespace std;
using namespace Eigen;
using namespace DVision;


#define MIN_LOOP_NUM 25
#define DEBUG_IMAGE 0


namespace ds_slam
{

template <typename Derived>
static void ReduceVector(vector<Derived> &v, vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}

KeyFrame::KeyFrame(Frame *frame_) : frame(frame_)
{
    time_stamp = frame->timestamp;
    index = frame->index;
    vo_T_w_i = frame->TwcOpti.translation();
    vo_R_w_i = frame->TwcOpti.rotationMatrix();
    vo_scale = frame->TwcOpti.scale();
    assert(vo_scale == 1.0);

    T_w_i = vo_T_w_i;
    R_w_i = vo_R_w_i;
    origin_vo_T = vo_T_w_i;
    origin_vo_R = vo_R_w_i;
    scale = vo_scale;
    image = frame->image.clone();
    cv::resize(image, thumbnail, cv::Size(80, 60));

    Eigen::Vector3d vio_T_i_w = frame->TwcOpti.inverse().translation();
    Eigen::Matrix3d vio_R_i_w = frame->TwcOpti.inverse().rotationMatrix();
    for(unsigned int i = 0; i < frame->window_PC.size(); i++)
    {
        if(frame->window_PC[i]->status == Point::PointStatus::INVALID) // remove outlier points
            continue;

        point_id.push_back(frame->window_PC[i]->id);

        frame->window_PC[i]->ComputeWorldPos();
        cv::Point3f p_3d;
        p_3d.x = frame->window_PC[i]->mWorldPos[0];
        p_3d.y = frame->window_PC[i]->mWorldPos[1];
        p_3d.z = frame->window_PC[i]->mWorldPos[2];
        point_3d.push_back(p_3d);

        cv::Point2f p_2d_norm, p_2d_uv;
        Vec3 camP = vio_R_i_w * frame->window_PC[i]->mWorldPos + vio_T_i_w;
        Vec2 camP_norm;
        camP_norm << camP[0] / camP[2],
            camP[1] / camP[2];
        p_2d_norm.x = camP_norm[0];
        p_2d_norm.y = camP_norm[1];
        Vec2 imgP;
#if 0
        imgP << fxG[0] * camP_norm[0] + cxG[0],
            fyG[0] * camP_norm[1] + cyG[0];
#else
        imgP << frame->fx * camP_norm[0] + frame->cx,
            frame->fy * camP_norm[1] + frame->cy;
#endif
        p_2d_uv.x = imgP[0];
        p_2d_uv.y = imgP[1];
        point_2d_norm.push_back(p_2d_norm);
        point_2d_uv.push_back(p_2d_uv);
    }

    has_loop = false;
    loop_index = -1;
    loop_info << 0, 0, 0, 0, 0, 0, 0, 0;
    sequence = frame->sequence;
    ComputeWindowPointDescriptor();
    ComputePointDescriptor();
    if (!DEBUG_IMAGE)
        image.release();

    qic.setIdentity();
    tic.setZero();
}

void KeyFrame::ComputeWindowPointDescriptor()
{
    for (int i = 0; i < (int)point_2d_uv.size(); i++)
    {
        cv::KeyPoint key;
        key.pt = point_2d_uv[i];
        window_keypoints.push_back(key);
    }
    if (extractor_type == DescriptorType::BRIEF)
    {
        //BriefExtractor extractor(brief_pattern_file);
        BriefExtractor extractor;
        extractor(image, window_keypoints, window_brief_descriptors);
    }
    else if (extractor_type == DescriptorType::ORB)
    {
        ORBExtractor extractor;
        extractor(image, window_keypoints, window_orb_descriptors);
    }
    else
    {
        //todo
    }
}

void KeyFrame::ComputePointDescriptor()
{
    const int fast_th = 20; // corner detector response threshold
    if (1)
        cv::FAST(image, keypoints, fast_th, true);
    else
    {
        vector<cv::Point2f> tmp_pts;
        cv::goodFeaturesToTrack(image, tmp_pts, 500, 0.01, 10);
        for (int i = 0; i < (int)tmp_pts.size(); i++)
        {
            cv::KeyPoint key;
            key.pt = tmp_pts[i];
            keypoints.push_back(key);
        }
    }

    if (extractor_type == DescriptorType::BRIEF)
    {
        //BriefExtractor extractor(brief_pattern_file);
        BriefExtractor extractor;
        extractor(image, keypoints, brief_descriptors);
    }
    else if (extractor_type == DescriptorType::ORB)
    {
        ORBExtractor extractor;
        extractor(image, keypoints, orb_descriptors);
    }
    else
    {
        //todo
    }

    for (int i = 0; i < (int)keypoints.size(); i++)
    {
        Eigen::Vector3d tmp_p;
        //m_camera->liftProjective(Eigen::Vector2d(keypoints[i].pt.x, keypoints[i].pt.y), tmp_p);
        tmp_p = Vec3(fxiG[0] * keypoints[i].pt.x + cxiG[0], fyiG[0] * keypoints[i].pt.y + cyiG[0], 1);
        cv::KeyPoint tmp_norm;
        tmp_norm.pt = cv::Point2f(tmp_p.x() / tmp_p.z(), tmp_p.y() / tmp_p.z());
        keypoints_norm.push_back(tmp_norm);
    }
}

bool KeyFrame::SearchInAera(const BRIEF::bitset window_descriptor,
                            const std::vector<BRIEF::bitset> &descriptors_old,
                            const std::vector<cv::KeyPoint> &keypoints_old,
                            const std::vector<cv::KeyPoint> &keypoints_old_norm,
                            cv::Point2f &best_match,
                            cv::Point2f &best_match_norm)
{
    cv::Point2f best_pt;
    int bestDist = 128;
    int bestIndex = -1;
    for (int i = 0; i < (int)descriptors_old.size(); i++)
    {
        int dis = HammingDis(window_descriptor, descriptors_old[i]);
        if (dis < bestDist)
        {
            bestDist = dis;
            bestIndex = i;
        }
    }
    //printf("best dist %d", bestDist);
    if (bestIndex != -1 && bestDist < 80)
    {
        best_match = keypoints_old[bestIndex].pt;
        best_match_norm = keypoints_old_norm[bestIndex].pt;
        return true;
    }
    else
        return false;
}

void KeyFrame::SearchByBRIEFDes(std::vector<cv::Point2f> &matched_2d_old,
                                std::vector<cv::Point2f> &matched_2d_old_norm,
                                std::vector<uchar> &status,
                                const std::vector<BRIEF::bitset> &descriptors_old,
                                const std::vector<cv::KeyPoint> &keypoints_old,
                                const std::vector<cv::KeyPoint> &keypoints_old_norm)
{
    for (int i = 0; i < (int)window_brief_descriptors.size(); i++)
    {
        cv::Point2f pt(0.f, 0.f);
        cv::Point2f pt_norm(0.f, 0.f);
        if (SearchInAera(window_brief_descriptors[i], descriptors_old, keypoints_old, keypoints_old_norm, pt, pt_norm))
            status.push_back(1);
        else
            status.push_back(0);
        matched_2d_old.push_back(pt);
        matched_2d_old_norm.push_back(pt_norm);
    }
}

void KeyFrame::FundmantalMatrixRANSAC(const std::vector<cv::Point2f> &matched_2d_cur_norm,
                                      const std::vector<cv::Point2f> &matched_2d_old_norm,
                                      vector<uchar> &status)
{
    int n = (int)matched_2d_cur_norm.size();
    for (int i = 0; i < n; i++)
        status.push_back(0);
    if (n >= 8)
    {
        vector<cv::Point2f> tmp_cur(n), tmp_old(n);
        for (int i = 0; i < (int)matched_2d_cur_norm.size(); i++)
        {
#if 0
            double FOCAL_LENGTH = 460.0;
            int COL = 640;
            int ROW = 480;
            double tmp_x, tmp_y;
            m_camera->liftProjective(Eigen::Vector2d(cur_pts[i].x, cur_pts[i].y), tmp_p);
            tmp_x = FOCAL_LENGTH * matched_2d_cur_norm[i].x + COL / 2.0;
            tmp_y = FOCAL_LENGTH * matched_2d_cur_norm[i].y + ROW / 2.0;
            tmp_cur[i] = cv::Point2f(tmp_x, tmp_y);

            m_camera->liftProjective(Eigen::Vector2d(forw_pts[i].x, forw_pts[i].y), tmp_p);
            tmp_x = FOCAL_LENGTH * matched_2d_old_norm[i].x + COL / 2.0;
            tmp_y = FOCAL_LENGTH * matched_2d_old_norm[i].y + ROW / 2.0;
            tmp_old[i] = cv::Point2f(tmp_x, tmp_y);
#else
            double tmp_x, tmp_y;
            tmp_x = fxG[0] * matched_2d_cur_norm[i].x + cxG[0];
            tmp_y = fyG[0] * matched_2d_cur_norm[i].y + cyG[0];
            tmp_cur[i] = cv::Point2f(tmp_x, tmp_y);

            tmp_x = fxG[0] * matched_2d_old_norm[i].x + cxG[0];
            tmp_y = fyG[0] * matched_2d_old_norm[i].y + cyG[0];
            tmp_old[i] = cv::Point2f(tmp_x, tmp_y);
#endif
        }
        cv::findFundamentalMat(tmp_cur, tmp_old, cv::FM_RANSAC, 3.0, 0.9, status);
    }
}

void KeyFrame::PnPRANSAC(const vector<cv::Point2f> &matched_2d_old_norm,
                         const std::vector<cv::Point3f> &matched_3d,
                         std::vector<uchar> &status,
                         Eigen::Vector3d &PnP_T_old, Eigen::Matrix3d &PnP_R_old)
{
    //for (int i = 0; i < matched_3d.size(); i++)
    //	printf("3d x: %f, y: %f, z: %f\n",matched_3d[i].x, matched_3d[i].y, matched_3d[i].z );
    //printf("match size %d \n", matched_3d.size());
    cv::Mat r, rvec, t, D, tmp_r;
    cv::Mat K = (cv::Mat_<double>(3, 3) << 1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0);
    Matrix3d R_inital;
    Vector3d P_inital;
    Matrix3d R_w_c = origin_vo_R * qic;
    Vector3d T_w_c = origin_vo_T + origin_vo_R * tic;

    R_inital = R_w_c.inverse();
    P_inital = -(R_inital * T_w_c);

    cv::eigen2cv(R_inital, tmp_r);
    cv::Rodrigues(tmp_r, rvec);
    cv::eigen2cv(P_inital, t);

    cv::Mat inliers;

    if (CV_MAJOR_VERSION < 3)
        solvePnPRansac(matched_3d, matched_2d_old_norm, K, D, rvec, t, true, 100, 10.0 / 460.0, 100, inliers);
    else
    {
        if (CV_MINOR_VERSION < 2)
            solvePnPRansac(matched_3d, matched_2d_old_norm, K, D, rvec, t, true, 100, sqrt(10.0 / 460.0), 0.99, inliers);
        else
            solvePnPRansac(matched_3d, matched_2d_old_norm, K, D, rvec, t, true, 100, 10.0 / 460.0, 0.99, inliers);
    }

    for (int i = 0; i < (int)matched_2d_old_norm.size(); i++)
        status.push_back(0);

    for (int i = 0; i < inliers.rows; i++)
    {
        int n = inliers.at<int>(i);
        status[n] = 1;
    }

    cv::Rodrigues(rvec, r);
    Matrix3d R_pnp, R_w_c_old;
    cv::cv2eigen(r, R_pnp);
    R_w_c_old = R_pnp.transpose();
    Vector3d T_pnp, T_w_c_old;
    cv::cv2eigen(t, T_pnp);
    T_w_c_old = R_w_c_old * (-T_pnp);

    PnP_R_old = R_w_c_old * qic.transpose();
    PnP_T_old = T_w_c_old - PnP_R_old * tic;
}

bool KeyFrame::FindConnection(KeyFrame *old_kf)
{
    //printf("find Connection\n");
    vector<cv::Point2f> matched_2d_cur, matched_2d_old;
    vector<cv::Point2f> matched_2d_cur_norm, matched_2d_old_norm;
    vector<cv::Point3f> matched_3d;
    vector<double> matched_id;
    vector<uchar> status;

    matched_3d = point_3d;
    matched_2d_cur = point_2d_uv;
    matched_2d_cur_norm = point_2d_norm;
    matched_id = point_id;

#if 0
    if (DEBUG_IMAGE)
    {
        cv::Mat gray_img, loop_match_img;
        cv::Mat old_img = old_kf->image;
        cv::hconcat(image, old_img, gray_img);
        cvtColor(gray_img, loop_match_img, CV_GRAY2RGB);
        for (int i = 0; i < (int)point_2d_uv.size(); i++)
        {
            cv::Point2f cur_pt = point_2d_uv[i];
            cv::circle(loop_match_img, cur_pt, 5, cv::Scalar(0, 255, 0));
        }
        for (int i = 0; i < (int)old_kf->keypoints.size(); i++)
        {
            cv::Point2f old_pt = old_kf->keypoints[i].pt;
            old_pt.x += COL;
            cv::circle(loop_match_img, old_pt, 5, cv::Scalar(0, 255, 0));
        }
        ostringstream path;
        path << "/home/tony-ws1/raw_data/loop_image/"
             << index << "-"
             << old_kf->index << "-"
             << "0raw_point.jpg";
        cv::imwrite(path.str().c_str(), loop_match_img);
    }
#endif
    //printf("search by des\n");
    SearchByBRIEFDes(matched_2d_old, matched_2d_old_norm, status, old_kf->brief_descriptors, old_kf->keypoints, old_kf->keypoints_norm);
    ReduceVector(matched_2d_cur, status);
    ReduceVector(matched_2d_old, status);
    ReduceVector(matched_2d_cur_norm, status);
    ReduceVector(matched_2d_old_norm, status);
    ReduceVector(matched_3d, status);
    ReduceVector(matched_id, status);
    //printf("search by des finish\n");

#if 0
    if (DEBUG_IMAGE)
    {
        int gap = 10;
        cv::Mat gap_image(ROW, gap, CV_8UC1, cv::Scalar(255, 255, 255));
        cv::Mat gray_img, loop_match_img;
        cv::Mat old_img = old_kf->image;
        cv::hconcat(image, gap_image, gap_image);
        cv::hconcat(gap_image, old_img, gray_img);
        cvtColor(gray_img, loop_match_img, CV_GRAY2RGB);
        for (int i = 0; i < (int)matched_2d_cur.size(); i++)
        {
            cv::Point2f cur_pt = matched_2d_cur[i];
            cv::circle(loop_match_img, cur_pt, 5, cv::Scalar(0, 255, 0));
        }
        for (int i = 0; i < (int)matched_2d_old.size(); i++)
        {
            cv::Point2f old_pt = matched_2d_old[i];
            old_pt.x += (COL + gap);
            cv::circle(loop_match_img, old_pt, 5, cv::Scalar(0, 255, 0));
        }
        for (int i = 0; i < (int)matched_2d_cur.size(); i++)
        {
            cv::Point2f old_pt = matched_2d_old[i];
            old_pt.x += (COL + gap);
            cv::line(loop_match_img, matched_2d_cur[i], old_pt, cv::Scalar(0, 255, 0), 1, 8, 0);
        }

        //ostringstream path, path1, path2;
        //path << "/home/tony-ws1/raw_data/loop_image/"
        //     << index << "-"
        //     << old_kf->index << "-"
        //     << "1descriptor_match.jpg";
        //cv::imwrite(path.str().c_str(), loop_match_img);
        //path1 << "/home/tony-ws1/raw_data/loop_image/"
        //      << index << "-"
        //      << old_kf->index << "-"
        //      << "1descriptor_match_1.jpg";
        //cv::imwrite(path1.str().c_str(), image);
        //path2 << "/home/tony-ws1/raw_data/loop_image/"
        //      << index << "-"
        //      << old_kf->index << "-"
        //      << "1descriptor_match_2.jpg";
        //cv::imwrite(path2.str().c_str(), old_img);
    }
#endif
    status.clear();

    //FundmantalMatrixRANSAC(matched_2d_cur_norm, matched_2d_old_norm, status);
    //ReduceVector(matched_2d_cur, status);
    //ReduceVector(matched_2d_old, status);
    //ReduceVector(matched_2d_cur_norm, status);
    //ReduceVector(matched_2d_old_norm, status);
    //ReduceVector(matched_3d, status);
    //ReduceVector(matched_id, status);

#if 0
    if (DEBUG_IMAGE)
    {
        int gap = 10;
        cv::Mat gap_image(ROW, gap, CV_8UC1, cv::Scalar(255, 255, 255));
        cv::Mat gray_img, loop_match_img;
        cv::Mat old_img = old_kf->image;
        cv::hconcat(image, gap_image, gap_image);
        cv::hconcat(gap_image, old_img, gray_img);
        cvtColor(gray_img, loop_match_img, CV_GRAY2RGB);
        for (int i = 0; i < (int)matched_2d_cur.size(); i++)
        {
            cv::Point2f cur_pt = matched_2d_cur[i];
            cv::circle(loop_match_img, cur_pt, 5, cv::Scalar(0, 255, 0));
        }
        for (int i = 0; i < (int)matched_2d_old.size(); i++)
        {
            cv::Point2f old_pt = matched_2d_old[i];
            old_pt.x += (COL + gap);
            cv::circle(loop_match_img, old_pt, 5, cv::Scalar(0, 255, 0));
        }
        for (int i = 0; i < (int)matched_2d_cur.size(); i++)
        {
            cv::Point2f old_pt = matched_2d_old[i];
            old_pt.x += (COL + gap);
            cv::line(loop_match_img, matched_2d_cur[i], old_pt, cv::Scalar(0, 255, 0), 1, 8, 0);
        }

        ostringstream path;
        path << "/home/tony-ws1/raw_data/loop_image/"
             << index << "-"
             << old_kf->index << "-"
             << "2fundamental_match.jpg";
        cv::imwrite(path.str().c_str(), loop_match_img);
    }
#endif
    Eigen::Vector3d PnP_T_old;
    Eigen::Matrix3d PnP_R_old;
    Eigen::Vector3d relative_t;
    Quaterniond relative_q;
    double relative_yaw;
    if ((int)matched_2d_cur.size() > MIN_LOOP_NUM)
    {
        status.clear();
        PnPRANSAC(matched_2d_old_norm, matched_3d, status, PnP_T_old, PnP_R_old);
        ReduceVector(matched_2d_cur, status);
        ReduceVector(matched_2d_old, status);
        ReduceVector(matched_2d_cur_norm, status);
        ReduceVector(matched_2d_old_norm, status);
        ReduceVector(matched_3d, status);
        ReduceVector(matched_id, status);
#if 0
        if (DEBUG_IMAGE)
        {
            int gap = 10;
            cv::Mat gap_image(ROW, gap, CV_8UC1, cv::Scalar(255, 255, 255));
            cv::Mat gray_img, loop_match_img;
            cv::Mat old_img = old_kf->image;
            cv::hconcat(image, gap_image, gap_image);
            cv::hconcat(gap_image, old_img, gray_img);
            cvtColor(gray_img, loop_match_img, CV_GRAY2RGB);
            for (int i = 0; i < (int)matched_2d_cur.size(); i++)
            {
                cv::Point2f cur_pt = matched_2d_cur[i];
                cv::circle(loop_match_img, cur_pt, 5, cv::Scalar(0, 255, 0));
            }
            for (int i = 0; i < (int)matched_2d_old.size(); i++)
            {
                cv::Point2f old_pt = matched_2d_old[i];
                old_pt.x += (COL + gap);
                cv::circle(loop_match_img, old_pt, 5, cv::Scalar(0, 255, 0));
            }
            for (int i = 0; i < (int)matched_2d_cur.size(); i++)
            {
                cv::Point2f old_pt = matched_2d_old[i];
                old_pt.x += (COL + gap);
                cv::line(loop_match_img, matched_2d_cur[i], old_pt, cv::Scalar(0, 255, 0), 2, 8, 0);
            }
            cv::Mat notation(50, COL + gap + COL, CV_8UC3, cv::Scalar(255, 255, 255));
            putText(notation, "current frame: " + to_string(index) + "  sequence: " + to_string(sequence), cv::Point2f(20, 30), CV_FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255), 3);

            putText(notation, "previous frame: " + to_string(old_kf->index) + "  sequence: " + to_string(old_kf->sequence), cv::Point2f(20 + COL + gap, 30), CV_FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255), 3);
            cv::vconcat(notation, loop_match_img, loop_match_img);

            //ostringstream path;
            //path << "/home/tony-ws1/raw_data/loop_image/"
            //     << index << "-"
            //     << old_kf->index << "-"
            //     << "3pnp_match.jpg";
            //cv::imwrite(path.str().c_str(), loop_match_img);

            if ((int)matched_2d_cur.size() > MIN_LOOP_NUM)
            {
                //cv::imshow("loop connection", loop_match_img);
                //cv::waitKey(10);

                cv::Mat thumbimage;
                cv::resize(loop_match_img, thumbimage, cv::Size(loop_match_img.cols / 2, loop_match_img.rows / 2));
                sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", thumbimage).toImageMsg();
                msg->header.stamp = ros::Time(time_stamp);
                pub_match_img.publish(msg);
            }
        }
#endif
    }

    if ((int)matched_2d_cur.size() > MIN_LOOP_NUM)
    {
        relative_t = PnP_R_old.transpose() * (origin_vo_T - PnP_T_old);
        relative_q = PnP_R_old.transpose() * origin_vo_R;
        relative_yaw = Utility::normalizeAngle(Utility::R2ypr(origin_vo_R).x() - Utility::R2ypr(PnP_R_old).x());
        //printf("PNP relative\n");
        //cout << "pnp relative_t " << relative_t.transpose() << endl;
        //cout << "pnp relative_yaw " << relative_yaw << endl;
        if (abs(relative_yaw) < 30.0 && relative_t.norm() < 20.0)
        {
            // flag loop
            has_loop = true;
            loop_index = old_kf->index;
            loop_info << relative_t.x(), relative_t.y(), relative_t.z(),
                relative_q.w(), relative_q.x(), relative_q.y(), relative_q.z(),
                relative_yaw;
            return true;
        }
    }
    //printf("loop final use num %d %lf--------------- \n", (int)matched_2d_cur.size(), t_match.toc());
    return false;
}

int KeyFrame::HammingDis(const BRIEF::bitset &a, const BRIEF::bitset &b)
{
    BRIEF::bitset xor_of_bitset = a ^ b;
    int dis = xor_of_bitset.count();
    return dis;
}

void KeyFrame::GetVoPose(Eigen::Vector3d &_T_w_i, Eigen::Matrix3d &_R_w_i)
{
    _T_w_i = vo_T_w_i;
    _R_w_i = vo_R_w_i;
}

void KeyFrame::GetPose(Eigen::Vector3d &_T_w_i, Eigen::Matrix3d &_R_w_i)
{
    _T_w_i = T_w_i;
    _R_w_i = R_w_i;
}

void KeyFrame::UpdatePose(const Eigen::Vector3d &_T_w_i, const Eigen::Matrix3d &_R_w_i)
{
    T_w_i = _T_w_i;
    R_w_i = _R_w_i;
}

void KeyFrame::UpdateVoPose(const Eigen::Vector3d &_T_w_i, const Eigen::Matrix3d &_R_w_i)
{
    vo_T_w_i = _T_w_i;
    vo_R_w_i = _R_w_i;
    T_w_i = vo_T_w_i;
    R_w_i = vo_R_w_i;
}

void KeyFrame::GetScale(double &scale_)
{
    scale_ = scale;
}

void KeyFrame::UpdateScale(double &scale_)
{
    scale = scale_;
}

Eigen::Vector3d KeyFrame::GetLoopRelativeT()
{
    return Eigen::Vector3d(loop_info(0), loop_info(1), loop_info(2));
}

Eigen::Quaterniond KeyFrame::GetLoopRelativeQ()
{
    return Eigen::Quaterniond(loop_info(3), loop_info(4), loop_info(5), loop_info(6));
}

double KeyFrame::GetLoopRelativeYaw()
{
    return loop_info(7);
}

void KeyFrame::UpdateLoop(Eigen::Matrix<double, 8, 1> &_loop_info)
{
    if (abs(_loop_info(7)) < 30.0 && Vector3d(_loop_info(0), _loop_info(1), _loop_info(2)).norm() < 20.0)
    {
        //printf("update loop info\n");
        loop_info = _loop_info;
    }
}

}
