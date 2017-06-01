/*********************************************
* Author: Jun Jiang - jiangjun4@sina.com
* Created: 2017-05-31 17:22
* Last modified: 2017-05-31 17:22
* Filename: CNN.cpp
* Description: Convolution Neural Network 
**********************************************/

#include "algorithm/cnn/CNN.h"

namespace ccma{
namespace algorithm{
namespace cnn{
bool CNN::add_layer(Layer* layer){
    /*
     * the first layer must be DataLayer
     */
    if(_layers.size() == 0){
        if(typeid(*layer) != typeid(DataLayer)){
            printf("The first layer must be DataLayer.\n");
            return false;
        }else{
            layer->initialize(nullptr);
            _layers.push_back(layer);
            return true;
        }
    }

    if(layer->initialize(_layers[_layers.size() - 1])){
        _layers.push_back(layer);
        return true;
    }else{
        return false;
    }
}

void CNN::train(ccma::algebra::BaseMatrixT<real>* train_data,
                ccma::algebra::BaseMatrixT<real>* train_label,
                uint epoch,
                ccma::algebra::BaseMatrixT<real>* test_data,
                ccma::algebra::BaseMatrixT<real>* test_label){
    uint num_train_data = train_data->get_rows();
    uint num_test_data = 0;
    if(test_data != nullptr){
        num_test_data = test_data->get_rows();
    }

    //check cnn structure and data dim
    if(!check(train_data->get_cols())){
        return;
    }

    auto mini_batch_data = new ccma::algebra::DenseMatrixT<real>();
    auto mini_batch_label = new ccma::algebra::DenseMatrixT<real>();

    auto now = []{return std::chrono::system_clock::now();};
    for(uint i = 0; i != epoch; i++){

        auto start_time = now();

        for(uint j = 0; j != num_train_data; j++){
            train_data->get_row_data(j, mini_batch_data);
            train_label->get_row_data(j, mini_batch_label);
            int layer_size = _layers.size();
            for(uint k = 0; k < layer_size; k++){
                auto layer = _layers[k];
                Layer* pre_layer = nullptr;
                if(k == 0){
                    mini_batch_data->reshape(layer->get_rows(), layer->get_cols());
                    ((DataLayer*)layer)->set_x(mini_batch_data);
                }
                if(k == _layers.size() - 1){
                    mini_batch_label->transpose();
                    ((FullConnectionLayer*)_layers[k])->set_y(mini_batch_label);
                }
                pre_layer = _layers[k - 1];
                layer->feed_forward(pre_layer);
            }//end feed_forward

            for(int k = layer_size - 1; k >= 0; k--){
                auto layer = _layers[k];
                Layer* pre_layer = nullptr;
                Layer* back_layer = nullptr;
                if(k > 0){
                    pre_layer = _layers[k - 1];
                }
                if(k < layer_size -1){
                    back_layer = _layers[k + 1];
                }
                layer->back_propagation(pre_layer, back_layer);
            }//end back_propagation

            if(j % 100 == 0){
                printf("Epoch[%d][%d/%d]training...\r", i, j, num_train_data);
            }
        }//end per epoch

        auto training_time = now();
        printf("Epoch %d training run time: %lld ms\n", i, std::chrono::duration_cast<std::chrono::milliseconds>(training_time - start_time).count());

        if(num_test_data > 0){
        }
    }//end all epoch

    delete mini_batch_data;
    delete mini_batch_label;
}

void CNN::feed_forward(ccma::algebra::BaseMatrixT<real>* mat){
}

bool CNN::check(uint size){
    if(_layers.size() <= 2){
        printf("convolution neural network layer must bemore than 2.\n");
        return false;
    }
    if(typeid(*_layers[_layers.size() -1]) != typeid(FullConnectionLayer)){
        printf("Convolution neural network last layer must be FullConnectionLayer\n");
        return false;
    }
    if(size != _layers[0]->get_rows() * _layers[0]->get_cols()){
        printf("Data dim size error.");
        return false;
    }
    return true;
}

}//namespace cnn
}//namespace algorithm
}//namespace ccma
