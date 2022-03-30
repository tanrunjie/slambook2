#include <iostream>
#include <opencv4/opencv2/opencv.hpp>

using namespace std;
using namespace cv;

Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);

void pose_estimate_2d2d(vector<KeyPoint> keypoint_1,
                        vector<KeyPoint> keypoint_2, vector<DMatch> matches, Mat &R, Mat &t)
{
    vector<Point2f> points1;
    vector<Point2f> points2;

    for (int i = 0; i < (int)matches.size(); i++)
    {
        points1.push_back(keypoint_1[matches[i].queryIdx].pt);
        points2.push_back(keypoint_2[matches[i].trainIdx].pt);
    }
    Mat fundamental_matrix;
    // 8点法
    fundamental_matrix = findFundamentalMat(points1, points2, FM_8POINT);
    cout << "fundamental_m is " << endl
         << fundamental_matrix << endl;

    Point2d principal_point(325.1, 249.7);
    double focal_length = 521;
    Mat essential_matrix;
    essential_matrix = findEssentialMat(points1, points2, focal_length, principal_point);
    cout << "essential_m is " << endl
         << essential_matrix << endl;

    Mat homography_matrix;
    homography_matrix = findHomography(points1, points2, RANSAC, 3);
    cout << "homography_m is " << endl
         << homography_matrix << endl;

    recoverPose(essential_matrix, points1, points2, R, t, focal_length, principal_point);
    cout << "R:" << R << endl;
    cout << "t:" << t << endl;
}

Point2d pixel2cam(const Point2d &p, const Mat &K)
{
    return Point2d(
        (p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
        (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1));
}

void find_feature_matches(const Mat &img_1, const Mat &img_2,
                          vector<KeyPoint> &keypoints_1, vector<KeyPoint> &keypoints_2, vector<DMatch> &good_matches)
{
    Mat descriptors_1, descriptors_2;

    Ptr<SIFT> detector = SIFT::create();
    detector->detectAndCompute(img_1, noArray(), keypoints_1, descriptors_1);
    detector->detectAndCompute(img_2, noArray(), keypoints_2, descriptors_2);

    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
    vector<vector<DMatch>> knn_matches;
    matcher->knnMatch(descriptors_1, descriptors_2, knn_matches, 2);

    const float ratio_thresh = 0.2f;
    for (size_t i = 0; i < knn_matches.size(); i++)
    {
        if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
        {
            good_matches.push_back(knn_matches[i][0]);
        }
    }
}

inline Scalar get_color(float depth)
{
    float up_th = 50, low_th = 10, th_range = up_th - low_th;
    if (depth > up_th)
        depth = up_th;
    if (depth < low_th)
        depth = low_th;
    return Scalar(255 * depth / th_range, 0, 255 * (1 - depth / th_range));
}

void triangulation(const vector<KeyPoint> &keypoint_1,
                   const vector<KeyPoint> &keypoint_2, const vector<DMatch> &matches,
                   const Mat &R, const Mat &t, vector<Point3d> &points)
{
    Mat T1 = (Mat_<float>(3,4) <<
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0);
    Mat T2 = (Mat_<float>(3,4) <<
        R.at<double>(0,0), R.at<double>(0,1), R.at<double>(0,2), t.at<double>(0,0),
        R.at<double>(1,0), R.at<double>(1,1), R.at<double>(1,2), t.at<double>(1,0),
        R.at<double>(2,0), R.at<double>(2,1), R.at<double>(2,2), t.at<double>(2,0));

    vector<Point2f> pts_1, pts_2;
    for(DMatch m: matches)
    {
        pts_1.push_back(pixel2cam(keypoint_1[m.queryIdx].pt, K));
        pts_2.push_back(pixel2cam(keypoint_2[m.trainIdx].pt, K));
    }
    Mat pts_4d;
    triangulatePoints(T1, T2, pts_1, pts_2, pts_4d);

    for(int i=0; i<pts_4d.cols;i++)
    {
        Mat x = pts_4d.col(i);
        x /= x.at<float>(3,0);
        Point3d p(x.at<float>(0,0), x.at<float>(1,0), x.at<float>(2,0));
        points.push_back(p);
    }

}

int main(int argc, char **argv)
{
    Mat img_1 = imread("../1.png", IMREAD_COLOR);
    Mat img_2 = imread("../2.png", IMREAD_COLOR);

    vector<KeyPoint> keypoints_1, keypoints_2;
    vector<DMatch> matches;
    find_feature_matches(img_1, img_2, keypoints_1, keypoints_2, matches);

    Mat R, t;
    pose_estimate_2d2d(keypoints_1, keypoints_2, matches, R, t);

    // triangulation
    vector<Point3d> points;
    triangulation(keypoints_1, keypoints_2, matches, R, t, points);

    Mat img1_plot = img_1.clone();
    Mat img2_plot = img_2.clone();

    for(int i=0; i < matches.size(); i++)
    {
        float depth1 = points[i].z;
        cout <<"depth: " << depth1 << endl;
        Point2d pt1_cam = pixel2cam(keypoints_1[matches[i].queryIdx].pt, K);
        circle(img1_plot, keypoints_1[matches[i].queryIdx].pt, 2, get_color(depth1), 2);
   
        Mat pt2_trans = R * (Mat_<double>(3,1) <<points[i].x, points[i].y, points[i].z) + t ;
        float depth2 = pt2_trans.at<double>(2, 0);
        circle(img2_plot, keypoints_2[matches[i].trainIdx].pt, 2, get_color(depth2), 2);
    }
    imshow("img1", img1_plot);
    imshow("img2", img2_plot);

    waitKey();

    return 0;
}