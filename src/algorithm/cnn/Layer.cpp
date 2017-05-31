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

/*
bool Layers::add_layer(Layer* layer){
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
*/

bool DataLayer::initialize(Layer* pre_layer){
    return true;
}
void DataLayer::feed_forward(Layer* pre_layer){
}
void DataLayer::back_propagation(Layer* pre_layer, Layer* back_layer){
}

bool SubSamplingLayer::initialize(Layer* pre_layer){
    uint pre_rows = pre_layer->get_rows();
    uint pre_cols = pre_layer->get_cols();

    if(pre_rows % _scale != 0 || pre_cols % _scale != 0){
        printf("SubSampling Layer scale error.\n");
        return false;
    }else{
        this->_rows = pre_rows / _scale;
        this->_cols = pre_cols / _scale;
        //can't change feature map size.
        this->_in_map_size = this->_out_map_size = pre_layer->get_out_map_size();
        /*
         * pre_layer, each feature map share a bias and initialize value is zero.
         * no pooling weight.
         */
        set_bias(new ccma::algebra::ColMatrixT<real>(this->_in_map_size, 0.0));

        return true;
    }
}
void SubSamplingLayer::feed_forward(Layer* pre_layer){

    uint pooling_size = _scale * _scale;

    // in_map size equal out_map size to subsampling layer.
    for(uint i = 0; i != this->_in_map_size; i++){

        auto a = pre_layer->get_activation(i);
        real* data = new real[this->_rows * this->_cols];

        for(uint j = 0; j != this->_rows; j++){
            for(uint k = 0; k != this->_cols; k++){
                //average pooling
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
        this->set_activation(i, activation);
    }
}
void SubSamplingLayer::back_propagation(Layer* pre_layer, Layer* back_layer){
    if(typeid(*back_layer) == typeid(ConvolutionLayer)){
        uint stride = ((ConvolutionLayer*)back_layer)->get_stride();
        for(uint i = 0 ; i != this->_out_map_size; i++){
            auto z = new ccma::algebra::DenseMatrixT<real>(this->_rows, this->_cols);
            for(uint j = 0; j != back_layer->get_in_map_size(); j++){
                auto d = new ccma::algebra::DenseMatrixT<real>();
                back_layer->get_deltas(j)->clone(d);
                auto w = new ccma::algebra::DenseMatrixT<real>();
                back_layer->get_weight(i, j)->clone(w);
                w->flip180();

                d->convn(w, stride, "full");
                z->add(d);
                delete d;
                delete w;
            }
            this->set_delta(i, z);
        }
    }else if(typeid(*back_layer) == typeid(FullConnectionLayer)){
        auto av = back_layer->get_av();
        if(this->_out_map_size * this->_rows * this->_cols == av->get_cols()){
            real* data = av->get_data();
            for(int i = 0; i != this->_out_map_size){
                auto delta = new ccma::algebra::DenseMatrixT<real>();
                real* d = new real[_rows * _cols];
                memset(d, &data[i * _rows * _cols], sizeof(real) * _rows * _cols);
                delta->set_shallow_data(d, _rows, _cols);
                this->set_delta(delta, i);
            }
        }else{
            printf("SubSamplingLayer back_propagation Size Erorr.\n");
        }
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
    }

    this->_weights = weights;
    /*
     * feature map shared the same bias of current layer.
     */
    this->set_bias(new ccma::algebra::ColMatrixT<real>(this->_out_map_size, 0.0));

    return true;
}

void ConvolutionLayer::feed_forward(Layer* pre_layer){

    //foreach output feature map
    for(uint i = 0; i != this->_out_map_size; i++){
        auto z = new ccma::algebra::BaseMatrixT<real>(this->_rows, this->_cols);
        auto a = new ccma::algebra::BaseMatrixT<real>();
        for(uint j = 0; j != this->_in_map_size; j++){
            pre_layer->get_activation(j)->clone(a);
            a->convn(this->get_weight(i, j), _stride, "valid");
            //sum all feature maps of pre_layer.
            z->add(a);
        }
        //add shared bias of feature map in current layer.
        z->add(this->_bias->get_data(i, 1));
        //if sigmoid activative function.
        z->sigmoid();

        this->set_activation(i, z);
    }
}

void ConvolutionLayer::back_propagation(Layer* pre_layer, Layer* back_layer){
    if(typeid(*back_layer) == typeid(SubSamplingLayer)){

        SubSamplingLayer* sub_layer = (SubSamplingLayer*)back_layer;
        uint scale = sub_layer->get_scale();

        //todo alpha
        real alpha = 0.1;
        real* derivate_bias_data = new real[this->_out_map_size];

        for(uint i = 0; i != this->_out_map_size; i++){
            /*
             * derivative_sigmoid: 
             *  sigmoid(z)*(1-sigmoid(z))
             *  a = sigmoid(z)
             */
            auto a1 = new ccma::algebra::DenseMatrixT<real>();
            auto a2 = new ccma::algebra::DenseMatrixT<real>();
            back_layer->get_activition(i)->clone(a1);
            a1->clone(a2);

            a2->multiply(-1);
            a2->add(1);
            a1->multiply(a2);
            delete a2;

            auto delta = new ccma::algebra::DenseMatrixT<real>();
            back_layer->get_delta(i)->clone(delta);

            /*
             * subsampling layer reduced matrix dim, so recover it by expand function
             */
            delta->expand(scale, scale);
            /*
             * back layer error sharing
             */
            delta->division(scale*scale);
            /*
             * delta_l = derivative_sigmoid * delta_l+1(recover dim)
             */
            a1->multiply(delta);
            delete delta;

            this->set_delta(i, a1);

            /*
             * calc grad and update weight/bias
             * only for online learning, if batch learning
             * need to average weight and bias
             */
            auto derivate_kernal = ccma::algebra::DenseMatrixT<real>();
            for(uint j = 0; j != this->_in_map_size; j++){
                pre_layer->get_activation(j)->clone(derivate_kernal);
                derivate_kernal->flip180();
                derivate_kernal->convn(_deltas[i], _stride, "valid");
                /*
                 * update grad: w -= alpha * new_w
                 */
                derivate_kernal->multiply(alpha);
                this->get_weight(i, j)->subtract(derivate_kernal);
            }
            delete derivate_kernal;
            //update bias
            derivate_bias_data[i] = _deltas[i]->sum();
        }
        auto derivate_bias = new ccma::algebra::DenseMatrixT<real>();
        derivate_bias->set_shallow_data(derivate_bias_data, this->_in_map_size, i);
        derivate_bias->multiply(alpha);
        this->_bias->subtract(derivate_bias);
        delete derivate_bias;
    }else if(typeid(*back_layer) == typeid(FullConnectionLayer)){
        auto av = back_layer->get_av();
        if(this->_out_map_size * this->_rows * this->_cols == av->get_cols()){
            real* data = av->get_data();
            for(int i = 0; i != this->_out_map_size){
                auto delta = new ccma::algebra::DenseMatrixT<real>();
                real* d = new real[_rows * _cols];
                memset(d, &data[i * _rows * _cols], sizeof(real) * _rows * _cols);
                delta->set_shallow_data(d, _rows, _cols);
                this->set_delta(delta, i);
            }
        }else{
            printf("ConvoluationLayer back_propagation Size Erorr.\n");
        }
    }
}

bool FullConnectionLayer::initialize(Layer* pre_layer){
    _cols = pre_layer->get_rows() * pre_layer->get_cols() * pre_layer->get_out_map_size();
    this->_in_map_size = pre_layer->get_out_map_size();

    this->set_bias(new ccma::algebra::ColMatrixT<real>(_rows, 0.0));
    auto weight = new ccma::algebra::DenseMatrixT<real>(this->_rows, this->_cols);
    //todo initialize weight
    this->set_weight(0, weight);
    return true;
}
void FullConnectionLayer::feed_forward(Layer* pre_layer){
    /*
     * concatenate pre_layer's all feature map mat into vector
     */
    auto a = new ccma::algebra::DenseMatrixT<real>();
    auto av = new ccma::algebra::DenseMatrixT<real>();
    for(uint i = 0; i != this->_in_map_size; i++){
        pre_layer->get_activation(i)->clone(a);
        a->reshape(1, this->_cols);
        av->extend(a);
    }
    delete a;
    if(_av != nullptr){
        delete _av;
    }
    _av = av;

    auto activation = new ccma::algebra::DenseMatrixT<real>();
    this->get_weight(0)->clone(activation);
    activation->dot(av);
    activation->add(this->get_bias());
    //if sigmoid activative function
    activation->sigmoid();
    this->set_activation(0, activation);
}

void FullConnectionLayer::back_propagation(Layer* pre_layer, Layer* back_layer){
    if(_error == nullptr){
        _error = new ccma::algebra::DenseMatrixT<real>();
    }
    _activations[0]->clone(_error);
    _error->subtract(_y);

    /* loss function, mse mean square error
     * 1/2 sum(error*error)/size
     * the size is 1 right here
     */
    auto mse_mat = new ccma::algebra::DenseMatrixT<real>();
    _error->clone(mse_mat);
    mse_mat->multiply(mse_mat);
    _loss = mse_mat->sum()/2;
    delete mse_mat;

    /*
     * error * derivate_of_output
     * derivate_of_output is activation * (1-activation)
     */
    auto derivate_output = ccma::algebra::DenseMatrixT<real>();
    _activations[0]->clone(derivate_output);

    auto derivate_output_b = ccma::algebra::DenseMatrixT<real>();
    derivate_output->clone(derivate_output_b);
    derivate_output_b->multiply(-1);
    derivate_output_b->add(1);

    derivate_output->multiply(derivate_output_b);
    derivate_output->multiply(_error);

    delete derivate_output_b;

    /*
     * calc delta: weight.T * derivate_output
     */
    auto delta = ccma::algebra::DenseMatrixT<real>();
    this->_weights[0]->clone(delta);
    delta->transpose();
    delta->dot(derivate_output);
    this->set_delta(delta, 0);

    //if pre_layer is ConvolutionLayer, has sigmoid function
    if(typeid(*pre_layer) == typeid(ConvolutionLayer)){
        auto av1 = ccma::algebra::DenseMatrixT<real>();
        auto av2 = ccma::algebra::DenseMatrixT<real>();
        _av->clone(av1);
        _av->clone(av2);
        /*
         * derivate_sigmoid: z * (1-z)
         */
        av2->multiply(-1);
        av2->add(1);
        av1->multiply(av2);
        this->get_delta->multiply(av1);

        delete av1;
        delete av2;
    }

    /*
     * derivate_weight = derivate_output * av.T
     * derivate_bias = derviate_output
     */
    auto derivate_weight = ccma::algebra::DenseMatrixT<real>();
    auto derivate_bias   = ccma::algebra::DenseMatrixT<real>();
    auto avt             = ccma::algebra::DenseMatrixT<real>();
    derivate_output->clone(derivate_weight);
    derivate_output->clone(derivate_bias);
    avt->clone(_av);

    avt->transpose();
    derivate_weight->dot(avt);
    delete avt;

    /*
     * update weight & bias
     */
    //todo set alpha
    real alpha = 0.1;
    derivate_weight->multiply(alpha);
    derivate_bias->multiply(alpha);
    this->_weights[0]->subtract(derivate_weight);
    this->_bias->subtract(derivate_bias);
    delete derivate_weight;
    delete derivate_bias;
}

}//namespace cnn
}//namespace algorithm
}//namespace ccma
