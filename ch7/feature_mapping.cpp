#include <iostream>
#include <opencv4/opencv2/core/core.hpp>
#include <opencv4/opencv2/features2d/features2d.hpp>
#include <opencv4/opencv2/highgui/highgui.hpp>
#include <chrono>
#include <vector>

using namespace std;
using namespace cv;

int main(int argc, char **argv)
{
    Mat img_1 = imread("../1.png", IMREAD_COLOR);
    Mat img_2 = imread("../2.png", IMREAD_COLOR);

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

    Mat img_matches;
    drawMatches(img_1, keypoints_1, img_2, keypoints_2, good_matches, img_matches);
    imshow("matches", img_matches);
    waitKey();
    

    return 0;
}