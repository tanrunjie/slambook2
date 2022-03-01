#include <iostream>
#include <ctime>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry> // rotate/translate

using namespace std;
using namespace Eigen;

int main(int argc, char **argv)
{
    Quaterniond q1 = {0.55, 0.3, 0.2, 0.2}; // {x,y,z,w}
    cout << "q1 = \n"
         << q1.coeffs() << endl;

    q1.normalize(); // 归一化
    cout << "normalize q1 = \n"
         << q1.coeffs() << endl;

    Isometry3d T1 = Isometry3d::Identity();
    T1.rotate(q1);
    T1.pretranslate(Vector3d(0.7, 1.1, 0.2));
    cout << T1.matrix() << endl;

    Quaterniond q2 = {-0.1, 0.3, -0.7, 0.2}; // {x,y,z,w}
    q2.normalize();
    Isometry3d T2 = Isometry3d::Identity();
    T2.rotate(q2);
    T2.pretranslate(Vector3d(-0.1, 0.4, 0.8));
    cout << T2.matrix() << endl;

    Vector4d p1 = {0.5, -0.1, 0.2, 1};
    Vector4d pw;
    pw = T1.matrix().colPivHouseholderQr().solve(p1);  // 世界坐标
    cout << pw.transpose() << endl;

    Vector4d p2;
    p2 = T2*pw;
    cout << p2.transpose() << endl;

    return 0;
}