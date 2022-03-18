#include <iostream>
#include <opencv4/opencv2/core/core.hpp>
#include <ceres/ceres.h>
#include <chrono>

using namespace std;


// cost func
struct CURVE_FITTING_COST{
    CURVE_FITTING_COST(double x, double y):_x(x), _y(y){}
    // residual
    template<typename T>
    bool operator()(
        const T * const abc,
        T *residual) const 
        {
            // y - exp(ax^2 + bx + c)
            residual[0] = T(_y) - ceres::exp(abc[0] * T(_x)* T(_x)
            + abc[1] * T(_x) + abc[2]);

            return true;
        }

    const double _x, _y;
};

int main(int argc, char ** argv)
{
    double ar = 1.0, br = 2.0, cr = 1.0;   // real
    double ae = 2.0, be = -1.0, ce = 5.0;  // estimated
    int N = 100;
    double w_sigma = 1.0;   //noise
    double inv_sigma = 1.0 / w_sigma;
    cv::RNG rng;

    vector<double> x_data, y_data;
    for(int i=0; i<N ; i++)
    {
        double x = i/100.0;
        x_data.push_back(x);
        y_data.push_back(exp(ar*x*x + br*x + cr) + rng.gaussian(w_sigma * w_sigma));
    }

    double abc[3] = {ae, be, ce};
    // ceres construct problem
    ceres::Problem problem;
    for(int i=0;i<N ;i++)
    {
        problem.AddResidualBlock( // 加误差项
            // 自动求导，模板参数： 误差类型，输出维度，输入维度，要与struct一致
            new ceres::AutoDiffCostFunction<CURVE_FITTING_COST, 1, 3>(
                new CURVE_FITTING_COST(x_data[i], y_data[i])
            ), 
            nullptr ,  // kernel func, here is empty
            abc        // solve parameters
            );   
    }

    // sovler
    ceres::Solver::Options options; 
    options.linear_solver_type = ceres::DENSE_QR;  // 增量如何求解
    options.minimizer_progress_to_stdout = true;    // 输出到cout

    ceres::Solver::Summary summary;     // 优化信息
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    ceres::Solve(options, &problem, &summary);
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2-t1);
    cout <<"solve time cost = " << time_used.count() << " seconds. " << endl;

    cout <<summary.BriefReport() << endl;
    cout <<"estimated a,b,c = ";
    for(auto a:abc) cout << a << " ";
    cout << endl;

    return 0;
}