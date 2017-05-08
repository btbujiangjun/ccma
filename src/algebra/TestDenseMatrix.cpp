#include <stdio.h>
#include "BaseMatrix.h"

int main(int argc, char** argv){
    float d[4] =  {1, 3, 2, 5};
    auto dm = new ccma::algebra::DenseMatrixT<float>(d, 2, 2);
    printf("[%d][%d]\n", dm->get_rows(), dm->get_cols());
    dm->display();

    dm->add(3);

    dm->display();

    auto d1 = new ccma::algebra::DenseMatrixT<float>(d, 2, 2);
    dm->clone(d1);

    d1->display();
    d1->set_row_data(d1, 1);
    d1->display();
    d1->extend(d1);
    d1->display();
    delete d1, dm;

    ccma::algebra::BaseMatrixT<int>* mn = new ccma::algebra::DenseMatrixMNT<int>(2, 3, 5);
    mn->display();
    delete mn;

    ccma::algebra::BaseMatrixT<int>* row_mat = new ccma::algebra::DenseRowMatrixT<int>(3, 5);
    row_mat->display();
    delete row_mat;

    ccma::algebra::BaseMatrixT<int>* col_mat = new ccma::algebra::DenseColMatrixT<int>(3, 5);
    col_mat->display();
    delete col_mat;

    ccma::algebra::BaseMatrixT<int>* zero_mat = new ccma::algebra::DenseZeroMatrixT<int>(3);
    zero_mat->display();
    delete zero_mat;

    ccma::algebra::BaseMatrixT<int>* one_mat = new ccma::algebra::DenseOneMatrixT<int>(3);
    one_mat->display();
    delete one_mat;

    ccma::algebra::BaseMatrixT<int>* eye_mat = new ccma::algebra::DenseEyeMatrixT<int>(3);
    eye_mat->display();
    delete eye_mat;

    float data1[9] = {1.0f, 2, 3, 4, 5, 6, 7, 8, 9};
    float label1[3] = {0, 1, 0};
    ccma::algebra::BaseMatrixT<float>* l_mat = new ccma::algebra::LabeledDenseMatrixT<float>(data1, label1, 3, 3);
    l_mat->display();
    delete l_mat;

    float data2[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    float data3[9] = {1, 4, 7, 2, 5, 8, 3, 6, 9};

    ccma::algebra::BaseMatrixT<float>* mat2 = new ccma::algebra::DenseMatrixT<float>(data2, 3, 3);
    ccma::algebra::BaseMatrixT<float>* mat3 = new ccma::algebra::DenseMatrixT<float>(data3, 3, 3);
    mat2->display();
    mat2->dot(mat3);
    mat2->display();

    delete mat2, mat3;

    const uint size = 200000000;
    real* add_data1 = new real[size];
    real* add_data2 = new real[size];
    for(int i = 0; i != size; i++){
        add_data1[i] = i;
        add_data2[i] = i;
    }

    auto add_mat1 = new ccma::algebra::DenseMatrixT<real>();
    add_mat1->set_shallow_data(add_data1, 40000, 5000);
    auto add_mat2 = new ccma::algebra::DenseMatrixT<real>();
    add_mat2->set_shallow_data(add_data2, 40000, 5000);

    auto now = []{return std::chrono::system_clock::now();};
    auto s = now();
    add_mat1->add(add_mat2);
    add_mat1->add(add_mat2);
    add_mat1->add(add_mat2);
    printf("run time:[%lld]\n", std::chrono::duration_cast<std::chrono::milliseconds>(now() - s).count());
    delete add_mat1;
    delete add_mat2;
}
