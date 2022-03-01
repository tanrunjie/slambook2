// eigen block 3*3 left-topper

#include <iostream>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>

using namespace std;
using namespace Eigen;

#define MATRIX_SIZE 30

int main()
{
    cout.precision(3);
    Matrix<double, MATRIX_SIZE, MATRIX_SIZE> matrix_NN = MatrixXd::Random(MATRIX_SIZE, MATRIX_SIZE);
    Matrix<double, 3, 3> matrix_3d1 = MatrixXd::Random(3, 3);
    Matrix3d matrix_3d = Matrix3d::Random();

    // method 1
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            matrix_3d(i, j) = matrix_NN(i, j);
            cout << matrix_3d(i, j) << " ";
        }
        cout << endl;
    }
    matrix_3d = Matrix3d::Identity();

    cout << matrix_3d << endl;

    cout << matrix_NN.block(0,0,3,3) << endl; // block 函数
    return 0;
}