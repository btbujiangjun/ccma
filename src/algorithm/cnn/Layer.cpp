/*********************************************
* Author: Jun Jiang - jiangjun4@sina.com
* Created: 2017-05-22 16:47
* Last modified: 2017-05-22 16:47
* Filename: CNN.cpp
* Description: convolutional network 
**********************************************/

#include "algorithm/cnn/Layer.h"

namespace ccma{
namespace algorithm{
namespace cnn{

bool Layers::add_layer(Layer* layer){
    /*
     * the first layer must be DataLayer
     */
    if(_layer.size() == 0){
        if(typeid(*layer) != typeid(DataLayer)){
            printf("The first layer must be DataLayer.\n");
            return false;
        }else{
            _layers.push_back(layer);
            return true;
        }
    }

    if(layer->initialize(_layers[_layers.size() -1])){
        _layers.push_back(layer);
        return true;
    }else{
        return false;
    }
}

bool DataLayer::initialize(Layer* pre_layer){
    return true;
}
void DataLayer::feed_back(Layer* pre_layer){
}

bool SubSamplingLayer::initialize(Layer* pre_layer){
    uint pre_rows = pre_layer->get_rows();
    uint pre_cols = pre_layer->get_cols();

    if(pre_rows%_scale != 0 || pre_cols%_scale != 0){
        printf("SubSampling Layer scale error.\n");
        return false;
    }else{
        this->_rows = pre_rows/_scale;
        this->_cols = pre_cols/_scale;
        this->_in_map_size = this->_out_map_size = pre_layer->get_out_map_size();
        /*
         * pre_layer, each feature map share a bias.
         */
        set_bias(new ccma::algebra::ColMatrixT<real>(pre_layer->get_out_map_size(), 0.0));
        return true;
    }
}
void SubSamplingLayer::feed_back(Layer* pre_layer){
    uint pooling_size = _scale * scale;

    for(uint i = 0; i != this->_out_map_size; i++){

        auto a = pre_layer->get_activations()[i];
        real* data = new real[this->_rows * this->_cols];

        for(uint j = 0; j != this->_rows; j++){
            for(uint k = 0; k != this->_cols; k++){
                //mean pooling
                real pooling_value = 0;
                for(uint m = 0; m != _scale; m++){
                    for(uint n = 0; n != _scale; n++){
                        pooling_value += a->get_data(j * _scale + m, k * _scale + n);
                    }
                }
                data[j * this->_cols + k] = pooling_value / pooling_size;
            }
        }

        auto activation = new ccma::algebra::BaseMatrixT<real>();
        activation->set_shallow_data(data, this->_rows, this->_cols);
        this->get_activations().push_back(activation);
    }
}

bool ConvolutionLayer::initialize(Layer* pre_layer){
    uint pre_rows = pre_layer->get_rows();
    uint pre_cols = pre_layer->get_cols();

    this->_rows = (pre_rows - _kernal_size) % _stride == 0 ? (pre_rows - _kernal_size) / _stride + 1 : (pre_rows - kernal_size) / _stride + 2;
    this->_cols = (pre_cols - _kernal_size) % _stride == 0 ? (pre_cols - _kernal_size) / _stride + 1 : (pre_cols - kernal_size) / _stride + 2;

    this->_in_map_size = pre_layer->get_out_map_size();

    auto weights = new ccam::algebra::BaseMatrixT<real>(_kernal_size, _kernal_size)[this->_in_map_size * this->_out_map_size];
    for(auto weight : weights){
        //todo random initialize
        this->get_weights().push_back(weight);
    }
    /*
     * feature map shared the same bias of current layer.
     */
    this->set_bias(new ccma::algebra::ColMatrixT<real>(this->_out_map_size, 0.0));

    return true;
}

void ConvolutionLayer::feed_ward(Layer* pre_layer){

    for(uint i = 0; i != this->_out_map_size; i++){
        auto z = new ccma::algebra::BaseMatrixT<real>(this->_rows, this->_cols);
        auto a = new ccma::algebra::BaseMatrixT<real>();
        for(uint j = 0; j != this->_in_map_size; j++){
            pre_layer->get_activations()[j]->clone(a);
            convalute(a, this->get_weights()[i * this->_in_map_size + j]);
            //sum all feature maps of pre_layer.
            z->add(a);
        }
        //add shared bias of feature map in current layer.
        z->add(this->_bias->get_data(i, 1));
        //if sigmoid activative function.
        z->sigmoid();

        this->get_activations().push_back(z);
    }
}

void ConvolutionLayer::back_prapagation(Layer* back_layer){
    for(uint i = 0; i != this->_in_map_size; i++){
    }
}

void ConvolutionLayer::convolute(ccma::algebra::BaseMatrixT<real>* mat, ccma::algebra::BaseMatrixT<real>* shared_weight){
    real* data   = new real[this->_rows * this->_cols];
    uint mat_row = mat->get_rows();
    uint mat_col = mat->get_cols();

    for(uint i = 0; i != this->_rows; i++){
        for(uint j = 0; j != this->_cols; j++){

            real sum = 0.0;
            for(uint k_i = 0; k_i != _kernal_size; k_i++){
                for(uint k_j = 0; k_j != _kernal_size; k_j++){
                    uint row = _stride * i + k_i;
                    uint col = _stride * j + k_j;
                    //fill 0, so ingore out of mat range
                    if(row < mat_row && col < mat_col){
                        sum += mat->get_data(row, col) * shared_weight->get_data(k_i, k_j);
                    }
                }
            }
            data[i * this->_cols + j] = sum;

        }
    }

    mat->set_shallow_data(data, this->_rows, this->_col);
}


bool FullConnectionLayer::initialize(Layer* pre_layer){
    _cols = pre_layer->get_rows() * pre_layer->get_cols() * pre_layer->get_out_map_size();
    this->_in_map_size = pre_layer->get_out_map_size();

    this->set_bias(new ccma::algebra::ColMatrixT<real>(_rows, 0.0));
    auto weight = new ccma::algebra::DenseMatrixT<real>(this->_rows, this->_cols);
    //todo initialize weight
    this->get_weights().push_back(weight);
    return true;
}

void FullConnectionLayer::feed_back(Layer* pre_layer){
    auto activation = new ccma::algebra::DenseMatrixT<real>();

    auto a = new ccma::algebra::BaseMatrixT<real>();
    for(uint i = 0; i != this->_in_map_size; i++){
        auto a = new ccma::algebra::BaseMatrixT<real>();
        pre_layer->get_activations()[i]->clone(a);
        a->reshape(1, this->_cols);
        activation->extend(a);
    }
    delete a;

    auto z = new ccma::algebra::DenseMatrixT<real>();
    this->get_weights()[0]->clone(z);
    z->dot(activation);
    z->add(this->get_bias());
    delete activation;
    //if sigmoid activative function
    z->sigmoid();
    this->get_activations()->push_back(z);
}

void FullConnectionLayer::back_propagation(Layer* back_layer){
}

}//namespace cnn
}//namespace algorithm
}//namespace ccma
