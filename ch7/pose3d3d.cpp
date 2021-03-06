#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <vector>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/SVD>
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <chrono>
// #include <sophus/se3.hpp>

using namespace std;
using namespace cv;

/// vertex and edges used in g2o ba
// class VertexPose : public g2o::BaseVertex<6, Sophus::SE3d> {
// public:
//   EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

//   virtual void setToOriginImpl() override {
//     _estimate = Sophus::SE3d();
//   }

//   /// left multiplication on SE3
//   virtual void oplusImpl(const double *update) override {
//     Eigen::Matrix<double, 6, 1> update_eigen;
//     update_eigen << update[0], update[1], update[2], update[3], update[4], update[5];
//     _estimate = Sophus::SE3d::exp(update_eigen) * _estimate;
//   }

//   virtual bool read(istream &in) override {}

//   virtual bool write(ostream &out) const override {}
// };

/// g2o edge
// class EdgeProjectXYZRGBDPoseOnly : public g2o::BaseUnaryEdge<3, Eigen::Vector3d, VertexPose> {
// public:
//   EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

//   EdgeProjectXYZRGBDPoseOnly(const Eigen::Vector3d &point) : _point(point) {}

//   virtual void computeError() override {
//     const VertexPose *pose = static_cast<const VertexPose *> ( _vertices[0] );
//     _error = _measurement - pose->estimate() * _point;
//   }

//   virtual void linearizeOplus() override {
//     VertexPose *pose = static_cast<VertexPose *>(_vertices[0]);
//     Sophus::SE3d T = pose->estimate();
//     Eigen::Vector3d xyz_trans = T * _point;
//     _jacobianOplusXi.block<3, 3>(0, 0) = -Eigen::Matrix3d::Identity();
//     _jacobianOplusXi.block<3, 3>(0, 3) = Sophus::SO3d::hat(xyz_trans);
//   }

//   bool read(istream &in) {}

//   bool write(ostream &out) const {}

// protected:
//   Eigen::Vector3d _point;
// };

void find_feature_matches(
    const Mat &img_1, const Mat &img_2,
    std::vector<KeyPoint> &keypoints_1,
    std::vector<KeyPoint> &keypoints_2,
    std::vector<DMatch> &matches)
{
    //-- ?????????
    Mat descriptors_1, descriptors_2;
    Ptr<SIFT> detector = SIFT::create();
    detector->detectAndCompute(img_1, noArray(), keypoints_1, descriptors_1);
    detector->detectAndCompute(img_2, noArray(), keypoints_2, descriptors_2);

    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
    vector<vector<DMatch>> knn_matches;
    matcher->knnMatch(descriptors_1, descriptors_2, knn_matches, 2);

    const float ratio_thresh = 0.25f;
    for (size_t i = 0; i < knn_matches.size(); i++)
        if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
            matches.push_back(knn_matches[i][0]);

    Mat img_matches;
    drawMatches(img_1, keypoints_1, img_2, keypoints_2, matches, img_matches);
    namedWindow("Matches", WINDOW_NORMAL);
    imshow("Matches", img_matches );
    waitKey();
}

// ????????????????????????????????????
Point2d pixel2cam(const Point2d &p, const Mat &K)
{
    return Point2d(
        (p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
        (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1));
}

void pose_estimation_3d3d(
    const vector<Point3f> &pts1,
    const vector<Point3f> &pts2,
    Mat &R, Mat &t)
{
    Point3f p1, p2; // center of mass
    int N = pts1.size();
    for (int i = 0; i < N; i++)
    {
        p1 += pts1[i];
        p2 += pts2[i];
    }
    p1 = Point3f(Vec3f(p1) / N);
    p2 = Point3f(Vec3f(p2) / N);
    vector<Point3f> q1(N), q2(N); // remove the center
    for (int i = 0; i < N; i++)
    {
        q1[i] = pts1[i] - p1;
        q2[i] = pts2[i] - p2;
    }

    // compute q1*q2^T
    Eigen::Matrix3d W = Eigen::Matrix3d::Zero();
    for (int i = 0; i < N; i++)
    {
        W += Eigen::Vector3d(q1[i].x, q1[i].y, q1[i].z) * Eigen::Vector3d(q2[i].x, q2[i].y, q2[i].z).transpose();
    }
    // cout << "W=" << W << endl;

    // SVD on W
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(W, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d V = svd.matrixV();

    if (U.determinant() * V.determinant() < 0)
    {
        for (int x = 0; x < 3; ++x)
        {
            U(x, 2) *= -1;
        }
    }

    // cout << "U=" << U << endl;
    // cout << "V=" << V << endl;

    Eigen::Matrix3d R_ = U * (V.transpose());
    Eigen::Vector3d t_ = Eigen::Vector3d(p1.x, p1.y, p1.z) - R_ * Eigen::Vector3d(p2.x, p2.y, p2.z);

    // convert to cv::Mat
    R = (Mat_<double>(3, 3) << R_(0, 0), R_(0, 1), R_(0, 2),
         R_(1, 0), R_(1, 1), R_(1, 2),
         R_(2, 0), R_(2, 1), R_(2, 2));
    t = (Mat_<double>(3, 1) << t_(0, 0), t_(1, 0), t_(2, 0));
}

// void bundleAdjustment(
//     const vector<Point3f> &pts1,
//     const vector<Point3f> &pts2,
//     Mat &R, Mat &t)
// {
//     // ???????????????????????????g2o
//     typedef g2o::BlockSolverX BlockSolverType;
//     typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType> LinearSolverType; // ?????????????????????
//     // ??????????????????????????????GN, LM, DogLeg ??????
//     auto solver = new g2o::OptimizationAlgorithmLevenberg(
//         g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
//     g2o::SparseOptimizer optimizer; // ?????????
//     optimizer.setAlgorithm(solver); // ???????????????
//     optimizer.setVerbose(true);     // ??????????????????

//     // vertex
//     VertexPose *pose = new VertexPose(); // camera pose
//     pose->setId(0);
//     pose->setEstimate(Sophus::SE3d());
//     optimizer.addVertex(pose);

//     // edges
//     for (size_t i = 0; i < pts1.size(); i++)
//     {
//         EdgeProjectXYZRGBDPoseOnly *edge = new EdgeProjectXYZRGBDPoseOnly(
//             Eigen::Vector3d(pts2[i].x, pts2[i].y, pts2[i].z));
//         edge->setVertex(0, pose);
//         edge->setMeasurement(Eigen::Vector3d(
//             pts1[i].x, pts1[i].y, pts1[i].z));
//         edge->setInformation(Eigen::Matrix3d::Identity());
//         optimizer.addEdge(edge);
//     }

//     chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
//     optimizer.initializeOptimization();
//     optimizer.optimize(10);
//     chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
//     chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
//     cout << "optimization costs time: " << time_used.count() << " seconds." << endl;

//     cout << endl
//          << "after optimization:" << endl;
//     cout << "T=\n"
//          << pose->estimate().matrix() << endl;

//     // convert to cv::Mat
//     Eigen::Matrix3d R_ = pose->estimate().rotationMatrix();
//     Eigen::Vector3d t_ = pose->estimate().translation();
//     R = (Mat_<double>(3, 3) << R_(0, 0), R_(0, 1), R_(0, 2),
//          R_(1, 0), R_(1, 1), R_(1, 2),
//          R_(2, 0), R_(2, 1), R_(2, 2));
//     t = (Mat_<double>(3, 1) << t_(0, 0), t_(1, 0), t_(2, 0));
// }

int main(int argc, char **argv)
{
    //-- ??????IR??????
    Mat img_1 = imread("../1.png", IMREAD_COLOR);
    Mat img_2 = imread("../2.png", IMREAD_COLOR);

    vector<KeyPoint> keypoints_1, keypoints_2;
    vector<DMatch> matches;
    find_feature_matches(img_1, img_2, keypoints_1, keypoints_2, matches);
    cout << "???????????????" << matches.size() << "????????????" << endl;

    // ??????3D???
    Mat depth1 = imread("../1_depth.png", IMREAD_UNCHANGED); // ????????????16?????????????????????????????????
    Mat depth2 = imread("../2_depth.png", IMREAD_UNCHANGED); // ????????????16?????????????????????????????????
    Mat K = (Mat_<double>(3, 3) << 509, 0, 317, 0, 509.0, 248, 0, 0, 1);
    vector<Point3f> pts1, pts2;

    for (DMatch m : matches)
    {
        ushort d1 = depth1.ptr<unsigned short>(int(keypoints_1[m.queryIdx].pt.y))[int(keypoints_1[m.queryIdx].pt.x)];
        ushort d2 = depth2.ptr<unsigned short>(int(keypoints_2[m.trainIdx].pt.y))[int(keypoints_2[m.trainIdx].pt.x)];
        if (d1 == 0 || d2 == 0) // bad depth
            continue;
        Point2d p1 = pixel2cam(keypoints_1[m.queryIdx].pt, K);
        Point2d p2 = pixel2cam(keypoints_2[m.trainIdx].pt, K);
        float dd1 = float(d1) / 1000.0;
        float dd2 = float(d2) / 1000.0;
        pts1.push_back(Point3f(p1.x * dd1, p1.y * dd1, dd1));
        pts2.push_back(Point3f(p2.x * dd2, p2.y * dd2, dd2));
    }

    cout << "3d-3d pairs: " << pts1.size() << endl;
    Mat R, t;
    pose_estimation_3d3d(pts1, pts2, R, t);
    cout << "ICP via SVD results: " << endl;
    cout << "R = " << R << endl;
    cout << "t = " << t << endl;
    cout << "R_inv = " << R.t() << endl;
    cout << "t_inv = " << -R.t() * t << endl;

    // bundleAdjustment(pts1, pts2, R, t);

    // // verify p1 = R*p2 + t
    // for (int i = 0; i < MIN(5,pts1.size()); i++)
    // {
    //     cout << i << endl;
    //     cout << "p1 = " << pts1[i] << endl;
    //     cout << "p2 = " << pts2[i] << endl;
    //     cout << "(R*p2+t) = " << R * (Mat_<double>(3, 1) << pts2[i].x, pts2[i].y, pts2[i].z) + t
    //          << endl;
    // }
}
