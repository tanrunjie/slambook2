#include <iostream>
#include <chrono>


using namespace std;

#include <opencv4/opencv2/core/core.hpp>
#include <opencv4/opencv2/highgui/highgui.hpp>

int main(int argc, char ** argv)
{
    cv::Mat image;
    image = cv::imread(argv[1]);

    if(image.data == nullptr)
    {
        cerr << "file:" << argv[1] << "does not exist." << endl;
        return 1;
    }

    cout <<"width:" << image.cols << " height:" << image.rows 
    << " channel:" << image.channels() << endl;

    cv::imshow("image", image);
    cv::waitKey(0);

    if(image.type()!=CV_8UC1 && image.type() != CV_8UC3)
    {
        cout <<" invalid image type" << endl;
        return 1;
    }
 
    return 0;
}