/***********************************************
 * Author: Jun Jiang - jiangjun4@sina.com
 * Create: 2017-07-07 16:21
 * Last modified : 2017-07-07 16:21
 * Filename      : TestModelLoader.cpp
 * Description   : 
 **********************************************/
#include <string>
#include "utils/ModelLoader.h"

int main(int argc, char** argv){
<<<<<<< HEAD
    ccma::algebra::DenseMatrixT<real> m1;
    const uint mat_size = 1000;
    real* d1 = new real[mat_size];
    for(uint i = 0; i != mat_size; i++){
        d1[i] = i;
    }
    m1.set_shallow_data(d1, 10, 100);

    ccma::utils::ModelLoader<real> loader;
=======
    ccma::algebra::DenseRandomMatrixT<real> m1(10, 10, 0.0, 1);

    ccma::utils::ModelLoader loader;
>>>>>>> 02fcdd16d09ed4c679fb06730b15f8bb59601fc6
    const std::string path = "data/test.model";
    loader.write<real>(&m1, path);
    loader.write<real>(&m1, path, true);


<<<<<<< HEAD
    std::vector<ccma::algebra::BaseMatrixT<real>*> ms;
    loader.read(path, &ms);
    for(auto&& m : ms){
        m->display();
        delete m;
=======
    std::vector<ccma::algebra::BaseMatrixT<real>*> m2;
    loader.read<real>(path, &m2);
    printf("mat_size[%ld]\n", m2.size());
    if(m2.size()){
        m2[0]->display("|");
        delete m2[0];
>>>>>>> 02fcdd16d09ed4c679fb06730b15f8bb59601fc6
    }
}
