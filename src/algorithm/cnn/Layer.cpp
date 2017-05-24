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
     * each pre_layer's feature map share the same bias.
     */
    this->set_bias(new ccma::algebra::ColMatrixT<real>(this->_in_map_size, 0.0));

    return true;
}

void ConvolutionLayer::feed_ward(Layer* pre_layer){

    for(uint i = 0; i != this->_out_map_size; i++){
        auto z = new ccma::algebra::BaseMatrixT<real>(this->_rows, this->_cols);
        /*
         * sum all of feature maps of pre_layer.
         */
        for(uint j = 0; j != this->_in_map_size; j++){
        }
    }
}

}//namespace cnn
}//namespace algorithm
}//namespace ccma
