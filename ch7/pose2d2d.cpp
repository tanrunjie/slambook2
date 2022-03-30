#include <iostream>
#include <opencv4/opencv2/core/core.hpp>
#include <opencv4/opencv2/features2d/features2d.hpp>
#include <opencv4/opencv2/highgui/highgui.hpp>
#include <opencv4/opencv2/calib3d/calib3d.hpp>

using namespace std;
using namespace cv;


Mat K = (Mat_<double>(3,3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);

void pose_estimate_2d2d(vector<KeyPoint> keypoint_1,
vector<KeyPoint> keypoint_2, vector<DMatch> matches, Mat &R, Mat &t)
{

    vector<Point2f> points1;
    vector<Point2f> points2;

    for(int i=0; i<(int)matches.size(); i++)
    {
        points1.push_back(keypoint_1[matches[i].queryIdx].pt);
        points2.push_back(keypoint_2[matches[i].trainIdx].pt);
    }

    Mat fundamental_matrix;
    // 8点法
    fundamental_matrix = findFundamentalMat(points1, points2, FM_8POINT); 
    cout <<"fundamental_m is " << endl << fundamental_matrix << endl;

    Point2d principal_point(325.1, 249.7);
    double focal_length = 521;
    Mat essential_matrix;
    essential_matrix = findEssentialMat(points1, points2, focal_length, principal_point);
    cout <<"essential_m is " << endl << essential_matrix << endl;

    Mat homography_matrix;
    homography_matrix = findHomography(points1, points2, RANSAC, 3);
    cout <<"homography_m is " << endl << homography_matrix << endl;

    recoverPose(essential_matrix, points1, points2, R, t, focal_length, principal_point);
    cout <<"R:" << R << endl;
    cout <<"t:" << t << endl;
}

Point2d pixel2cam(const Point2d &p, const Mat &K)
{
    return Point2d
    (
        (p.x - K.at<double>(0,2)) / K.at<double>(0,0),
        (p.y - K.at<double>(1,2)) / K.at<double>(1,1)
    );
}

int main(int argc, char **argv)
{
    Mat img_1 = imread("../1.png",IMREAD_COLOR);
    Mat img_2 = imread("../2.png",IMREAD_COLOR);

    vector<KeyPoint> keypoints_1, keypoints_2;
    Mat descriptors_1, descriptors_2;

    Ptr<SIFT> detector = SIFT::create();
    detector->detectAndCompute(img_1, noArray(), keypoints_1, descriptors_1);
    detector->detectAndCompute(img_2, noArray(), keypoints_2, descriptors_2);

    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
    vector<vector<DMatch>> knn_matches;
    matcher->knnMatch(descriptors_1, descriptors_2, knn_matches, 2);

    const float ratio_thresh = 0.2f;
    vector<DMatch> good_matches;

    for(size_t i=0; i<knn_matches.size(); i++)
    {
        if(knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
        {
            good_matches.push_back(knn_matches[i][0]);
        }
    }

    cout <<"found: " << good_matches.size() << " matches" << endl;

    Mat R, t;
    pose_estimate_2d2d(keypoints_1, keypoints_2, good_matches, R, t);

    Mat t_x = (Mat_<double>(3,3) <<
        0, -t.at<double>(2,0), t.at<double>(1,0),
        t.at<double>(2,0), 0, -t.at<double>(0,0),
        -t.at<double>(1,0), t.at<double>(0,0), 0);
    cout <<"t^R=" << endl << t_x * R << endl;

    // 验证对极约束误差
    for(auto m: good_matches)
    {
        Point2d pt1 = pixel2cam(keypoints_1[m.queryIdx].pt, K);
        Mat y1 = (Mat_<double>(3,1) << pt1.x, pt1.y, 1);

        Point2d pt2 = pixel2cam(keypoints_2[m.trainIdx].pt, K);
        Mat y2 = (Mat_<double>(3,1) << pt2.x, pt2.y, 1);
        Mat d = y2.t() * t_x * R * y1;
        cout <<"epipolar constrain = " << d << endl;
    }
    
    return 0;
}