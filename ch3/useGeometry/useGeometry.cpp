#include <iostream>
#include <cmath>

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>

using namespace std;
using namespace Eigen;

int main(int argc, char** argv)
{
    Matrix3d rotation_matrix = Matrix3d::Identity();
    AngleAxisd rotation_vector(M_PI/4, Vector3d(0, 0, 1)); // z:45 degree

    cout.precision(3);
    cout <<"rotation matrix = \n" << rotation_vector.matrix() << endl;

    rotation_matrix = rotation_vector.toRotationMatrix();

    Vector3d v(1,0,0);
    Vector3d v_rotated = rotation_vector * v;
    cout <<"(1,0,0) after rotation =" << v_rotated.transpose() << endl;

    v_rotated = rotation_matrix * v;
    cout <<"(1,0,0) after rotation =" << v_rotated.transpose() << endl;

    // Rotate Matrix to Euler angle
    Vector3d euler_angles = rotation_matrix.eulerAngles(2,1,0); //ZYX: yaw pich roll

    cout <<"yaw pitch roll = " << euler_angles.transpose() << endl;

    Isometry3d T = Isometry3d::Identity(); // 4*4
    T.rotate(rotation_vector);
    T.pretranslate(Vector3d(1,3,4));
    cout <<"Transform matrix = \n" << T.matrix() << endl;

    Vector3d v_transformed = T*v;
    cout <<"v transformed = " << v_transformed.transpose() << endl;
    
    // quaternion
    Quaterniond q = Quaterniond(rotation_vector);
    cout <<"quaternion = \n" << q.coeffs() << endl; // (x,y,z,w)

    q = Quaterniond(rotation_matrix);
    cout <<"quaternion = \n" << q.coeffs() << endl;

    v_rotated = q * v; // qvq^{-1}
    cout <<"(1,0,0) after rotation = " << v_rotated.transpose() << endl;

    return 0;
}