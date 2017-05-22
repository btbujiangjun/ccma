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

bool Layers::add(Layer* layer){
    if(_layer.size() == 0){
        if(typeid(*layer) != typeid(DataLayer)){
            printf("The first layer must be DataLayer.\n");
            return false;
        }else{
            _layers.push_back(layer);
            return true;
        }
    }

    Layer* pre_layer = _layers[_layers.size()];

    if(typeid(*layer) == typeid(SubSamplingLayer)){
        if(pre_layer->get_rows() % layer->get_scale() != = 0 || pre_layer->get_cols() % layer->get_scale() != 0){
            printf("SubSampling Layer scale error.\n");
            return false;
        }else{
            layer->set_rows(pre_layer->get_rows() / layer->get_scale());
            layer->set_cols(pre_layer->get_cols() / layer->get_scale());

            auto bias = new ccma::algebra::BaseMatrixT<real>();
            real* data = new real[pre_layer->get_in_map_size()];
            bias->set_shallow_data(data, pre_lay->get_in_map_size());
            layer->set_bias(bias);
        }
    }
}

}//namespace cnn
}//namespace algorithm
}//namespace ccma
