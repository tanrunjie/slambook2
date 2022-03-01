// solve Ax=b

#include <iostream>
#include <ctime>
#include <cmath>
#include <complex>

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>
#include <eigen3/Eigen/Eigenvalues>

#define MATRIX_SIZE 3   // equations nums  
#define MATRIX_SIZE_ 3  // val nums


using namespace std;
using namespace Eigen;

typedef Matrix<double, MATRIX_SIZE, MATRIX_SIZE_> Mat_A;
typedef Matrix<double, MATRIX_SIZE, 1> Mat_B;


double Jacobi_sum(Mat_A & A, Mat_B &x_k, int i)
{
    double sum;
    for(int j=0; j<MATRIX_SIZE_; j++)
        sum += A(i,j) *x_k(j);
    return sum;
}

Mat_B Jacobi(Mat_A &A, Mat_B &b, int &iteration_num, double &accuracy)
{
    Mat_B x_k = MatrixXd::Zero(MATRIX_SIZE_, 1); // initial
    Mat_B x_k1; 
    int k,i;
    double temp;
    double R = 0;
    int isFlag = 0;

    // is Jacobi converge?
    Mat_A D;
    Mat_A L_U;
    Mat_A temp2 = A;
    Mat_A B;
    MatrixXcd EV;  // eigen value of matrix
    double maxev = 0.0;   // max eigen value
    int flag = 0;    // isconverge?

    cout << endl << "In Jacobi Iterative Algorithm" << endl;

    // Decompose A and determine if Jacobi is converge
    for(int l=0; l<MATRIX_SIZE; l++){
        D(l,l) = A(l,l);
        temp2(l,l) = 0;
        if(D(l,l) == 0){
            cout <<"iterative matrix can not be solved" << endl;
            flag = 1;
            break;
        }
    }
    L_U = -temp2;
    B = D.inverse() * L_U;

    EigenSolver<Mat_A> es(B);
    EV = es.eigenvalues();

    for(int index =0; index < MATRIX_SIZE; index++)
        maxev = (maxev > __complex_abs(EV(index))) ? 
        maxev :(__complex_abs(EV(index)));
    cout <<"Jacobi迭代矩阵的谱半径为：" << maxev << endl;

    if(maxev >= 1)
    {
        cout <<"Jacobi 不收敛" << endl;
        flag = 1;
    }

    if(flag == 0){
        cout << "Jacobi迭代算法谱半径小于1，该算法可收敛" << endl;
        cout <<"Jacobi迭代次数和精度：" << endl << iteration_num << " " << accuracy <<endl;
    
        for(k=0; k<iteration_num;k++){
            for(i=0;i<MATRIX_SIZE_;i++)
            {
                x_k1(i) = x_k(i) +(b(i) - Jacobi_sum(A, x_k, i ))/A(i,i);
                temp = fabs(x_k1(i) - x_k(i));
                if(fabs(x_k1(i) - x_k(i)) > R)
                    R = temp;
            }

            if(R<accuracy)
            {
                cout <<"Jacobi iterate " << k <<" times acceed accuracy." << endl;
                isFlag = 1;
                break;
            }

            R = 0;
            x_k = x_k1;
        }
        if(!isFlag)
            cout << endl << "iterate:" << iteration_num << "still not acceed accuracy." << endl;
        return x_k1;


    }
    return x_k;
}

int main(int argc, char ** argv)
{
    cout.precision(3);

    Mat_A matrix_NN = MatrixXd::Random(MATRIX_SIZE, MATRIX_SIZE_);
    Mat_B v_Nd = MatrixXd::Random(MATRIX_SIZE, 1);

    // TEST CASE
    matrix_NN << 10, 3, 1, 2, -10, 3, 1, 3, 10;
    v_Nd << 14, -5, 14;

    Matrix<double, MATRIX_SIZE_, 1> x;
    clock_t tim_stt = clock();

// 1. inverse: 可能没解，当且仅当方阵
#if(MATRIX_SIZE == MATRIX_SIZE_)
    x = matrix_NN.inverse() * v_Nd;
    cout << "direct method use: " << 1000*(clock()-tim_stt)/(double)CLOCKS_PER_SEC
    <<" ms" << endl << x.transpose() << endl;   
#else
    cout <<" can't solve in direct method" << endl;
#endif

// 2. QR: 适合非方阵，若有解为真解，若无解得出近似解
    tim_stt = clock();
    x = matrix_NN.colPivHouseholderQr().solve(v_Nd);
       cout << "QR method use: " << 1000*(clock()-tim_stt)/(double)CLOCKS_PER_SEC
    <<" ms " << endl << x.transpose() << endl;

// 3. 最小二乘：适合非方阵，有解为真解，否则为最小二乘解
    tim_stt = clock();
    x = (matrix_NN.transpose() * matrix_NN).inverse() *
    (matrix_NN.transpose() * v_Nd);
 cout << "Minimal method use: " << 1000*(clock()-tim_stt)/(double)CLOCKS_PER_SEC
    <<" ms " << endl << x.transpose() << endl;


// 4. LU: 仅方阵
#if(MATRIX_SIZE == MATRIX_SIZE_)
    tim_stt = clock();
    x = matrix_NN.lu().solve(v_Nd);
    cout << "LU method use: " << 1000*(clock()-tim_stt)/(double)CLOCKS_PER_SEC
    <<" ms" << endl << x.transpose() << endl;   
#else
    cout <<" can't solve in LU method" << endl;
#endif

// 5. Cholesky: 仅方阵
#if(MATRIX_SIZE == MATRIX_SIZE_)
    tim_stt = clock();
    x = matrix_NN.llt().solve(v_Nd);
    cout << "Cholesky method use: " << 1000*(clock()-tim_stt)/(double)CLOCKS_PER_SEC
    <<" ms" << endl << x.transpose() << endl;   
#else
    cout <<" can't solve in Cholesky method" << endl;
#endif

// 6. Jacobi iterative
#if(MATRIX_SIZE == MATRIX_SIZE_)
    int Iteration_num = 10;
    double Accuracy = 0.01;
    tim_stt = clock();
    x = Jacobi(matrix_NN, v_Nd, Iteration_num, Accuracy);
    cout << "jacobi method use: " << 1000*(clock()-tim_stt)/(double)CLOCKS_PER_SEC
    <<" ms" << endl << x.transpose() << endl;   
#else
    cout <<" can't solve in jacobi method" << endl;
#endif

    return 0;

}