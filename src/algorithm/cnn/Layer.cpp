/*********************************************
* Author: Jun Jiang - jiangjun4@sina.com
* Created: 2017-05-22 16:47
* Last modified: 2017-05-22 16:47
* Filename: CNN.cpp
* Description: convolutional network 
**********************************************/
#include <typeinfo>
#include "algorithm/cnn/Layer.h"

namespace ccma{
namespace algorithm{
namespace cnn{

bool DataLayer::initialize(Layer* pre_layer){
    return true;
}
void DataLayer::feed_forward(Layer* pre_layer){
    auto activation = new ccma::algebra::DenseMatrixT<real>();
    _x->clone(activation);
    this->set_activation(0, activation);
}
void DataLayer::back_propagation(Layer* pre_layer, Layer* back_layer){
}

bool SubSamplingLayer::initialize(Layer* pre_layer){
    uint pre_rows = pre_layer->get_rows();
    uint pre_cols = pre_layer->get_cols();

    if(pre_rows % _scale != 0 || pre_cols % _scale != 0){
        printf("SubSampling Layer scale error.\n");
        return false;
    }
    this->_rows = pre_rows / _scale;
    this->_cols = pre_cols / _scale;
    //can't change out feature map size.
    this->_in_map_size = this->_out_map_size = pre_layer->get_out_map_size();
    /*
     * pre_layer, each feature map share a bias and initialize value is zero.
     * no pooling weight.
     */
    set_bias(new ccma::algebra::DenseColMatrixT<real>(this->_in_map_size, 0.0));
    return true;
}
void SubSamplingLayer::feed_forward(Layer* pre_layer){

    uint pooling_size = _scale * _scale;

    // in_map size equal out_map size to subsampling layer.
    for(uint i = 0; i != this->_in_map_size; i++){
        auto a = pre_layer->get_activation(i);
        real* data = new real[this->_rows * this->_cols];
        for(uint j = 0; j != this->_rows; j++){
            for(uint k = 0; k != this->_cols; k++){
                real pooling_value = 0;

                //mean pooling
                pooling_value = 0;
                for(uint m = 0; m != _scale; m++){
                    for(uint n = 0; n != _scale; n++){
                        pooling_value += a->get_data(j * _scale + m, k * _scale + n);
                    }
                }
                data[j * this->_cols + k] = pooling_value / pooling_size;

                /*
                //max pooling
                pooling_value = 0;
                for(uint m = 0; m != _scale; m++){
                    for(uint n = 0; n != _scale; n++){
                        real value = a->get_data(j * _scale + m, k * _scale + n);
                        if(m * n == 0 || value > pooling_value){
                            pooling_value = value;
                        }
                    }
                }
                data[j * this->_cols + k] = pooling_value;
                */

            }
        }
        auto activation = new ccma::algebra::DenseMatrixT<real>();
        activation->set_shallow_data(data, this->_rows, this->_cols);
        this->set_activation(i, activation);
    }
}
void SubSamplingLayer::back_propagation(Layer* pre_layer, Layer* back_layer){
    if(typeid(*back_layer) == typeid(ConvolutionLayer)){
        uint stride = ((ConvolutionLayer*)back_layer)->get_stride();
        auto d = new ccma::algebra::DenseMatrixT<real>();
        auto w = new ccma::algebra::DenseMatrixT<real>();
        for(uint i = 0 ; i != this->_out_map_size; i++){
            auto z = new ccma::algebra::DenseMatrixT<real>(this->_rows, this->_cols);
            for(uint j = 0; j != back_layer->get_in_map_size(); j++){
                back_layer->get_delta(j)->clone(d);
                back_layer->get_weight(i, j)->clone(w);
                w->flip180();

                d->convn(w, stride, "full");
                z->add(d);
            }
            this->set_delta(i, z);
        }
        delete d;
        delete w;
    }else if(typeid(*back_layer) == typeid(FullConnectionLayer)){
        auto av = ((FullConnectionLayer*)back_layer)->get_av();
        if(this->_out_map_size * this->_rows * this->_cols == av->get_rows()){
            real* data = av->get_data();
            for(int i = 0; i != this->_out_map_size; i++){
                auto delta = new ccma::algebra::DenseMatrixT<real>();
                real* d = new real[_rows * _cols];
                memcpy(d, &data[i * this->_rows * this->_cols], sizeof(real) * this->_rows * this->_cols);
                delta->set_shallow_data(d, this->_rows, this->_cols);
                this->set_delta(i, delta);
            }
        }else{
            printf("SubSamplingLayer back_propagation Size Erorr.\n");
        }
    }
}

bool ConvolutionLayer::initialize(Layer* pre_layer){
    uint pre_rows = pre_layer->get_rows();
    uint pre_cols = pre_layer->get_cols();

    if(pre_rows < _kernal_size || pre_cols < _kernal_size){
        printf("ConvolutionLayer Size Erorr: pre_rows less than kernal_size.\n");
        return false;
    }

    this->_rows = (pre_rows - _kernal_size) % _stride == 0 ? (pre_rows - _kernal_size) / _stride + 1 : (pre_rows - _kernal_size) / _stride + 2;
    this->_cols = (pre_cols - _kernal_size) % _stride == 0 ? (pre_cols - _kernal_size) / _stride + 1 : (pre_cols - _kernal_size) / _stride + 2;

    this->_in_map_size = pre_layer->get_out_map_size();

    for(uint i = 0; i != this->_out_map_size; i++){
        for(uint j = 0; j != this->_in_map_size; j++){
            auto weight = new ccma::algebra::DenseRandomMatrixT<real>(_kernal_size, _kernal_size, 0.0, 1.0);
            this->set_weight(i, j, weight);
        }
    }

    /*
     * feature map shared the same bias of current layer.
     */
    this->set_bias(new ccma::algebra::DenseColMatrixT<real>(this->_out_map_size, 0.0));

    return true;
}

void ConvolutionLayer::feed_forward(Layer* pre_layer){

    //foreach output feature map
    for(uint i = 0; i != this->_out_map_size; i++){
        auto z = new ccma::algebra::DenseMatrixT<real>(this->_rows, this->_cols);
        auto a = new ccma::algebra::DenseMatrixT<real>();
        for(uint j = 0; j != this->_in_map_size; j++){
            pre_layer->get_activation(j)->clone(a);
            a->convn(this->get_weight(i, j), _stride, "valid");
            //sum all feature maps of pre_layer.
            z->add(a);
        }
        delete a;
        //add shared bias of feature map in current layer.
        z->add(this->get_bias()->get_data(i, 1));
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
        real alpha = 0.5;
        real* derivate_bias_data = new real[this->_out_map_size];

        for(uint i = 0; i != this->_out_map_size; i++){
            /*
             * derivative_sigmoid: 
             *  sigmoid(z)*(1-sigmoid(z))
             *  a = sigmoid(z)
             */
            auto a1 = new ccma::algebra::DenseMatrixT<real>();
            auto a2 = new ccma::algebra::DenseMatrixT<real>();
            this->get_activation(i)->clone(a1);
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
            auto derivate_kernal = new ccma::algebra::DenseMatrixT<real>();
            for(uint j = 0; j != this->_in_map_size; j++){
                pre_layer->get_activation(j)->clone(derivate_kernal);
                derivate_kernal->flip180();
                derivate_kernal->convn(this->get_delta(i), _stride, "valid");
                /*
                 * update grad: w -= alpha * new_w
                 */
                derivate_kernal->multiply(alpha);
                this->get_weight(i, j)->subtract(derivate_kernal);
            }
            delete derivate_kernal;
            //update bias
            derivate_bias_data[i] = this->get_delta(i)->sum();
        }
        auto derivate_bias = new ccma::algebra::DenseMatrixT<real>();
        derivate_bias->set_shallow_data(derivate_bias_data, this->_out_map_size, 1);
        derivate_bias->multiply(alpha);
        this->get_bias()->subtract(derivate_bias);
//        derivate_bias->display();
        delete derivate_bias;

    }else if(typeid(*back_layer) == typeid(FullConnectionLayer)){
        auto av = ((FullConnectionLayer*)back_layer)->get_av();
        if(this->_out_map_size * this->_rows * this->_cols == av->get_rows()){
            real* data = av->get_data();
            for(int i = 0; i != this->_out_map_size; i++){
                auto delta = new ccma::algebra::DenseMatrixT<real>();
                real* d = new real[_rows * _cols];
                memcpy(d, &data[i * this->_rows * this->_cols], sizeof(real) * this->_rows * this->_cols);
                delta->set_shallow_data(d, _rows, _cols);
                this->set_delta(i, delta);
            }
        }else{
            printf("ConvolutionLayer back_propagation Size Erorr.\n");
        }
    }
}

bool FullConnectionLayer::initialize(Layer* pre_layer){
    _cols = pre_layer->get_rows() * pre_layer->get_cols() * pre_layer->get_out_map_size();
    this->_in_map_size = pre_layer->get_out_map_size();

    this->set_bias(new ccma::algebra::DenseColMatrixT<real>(_rows, 0.0));
    auto weight = new ccma::algebra::DenseRandomMatrixT<real>(this->_rows, this->_cols, 0.0, 1.0);
    this->set_weight(0, 0, weight);
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
        a->reshape(a->get_rows() * a->get_cols(), 1);
        av->extend(a, false);
    }
    delete a;
    if(_av != nullptr){
        delete _av;
    }
    _av = av;

    auto activation = new ccma::algebra::DenseMatrixT<real>();
    this->get_weight(0, 0)->clone(activation);
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
    this->get_activation(0)->clone(_error);
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
    auto derivate_output = new ccma::algebra::DenseMatrixT<real>();
    this->get_activation(0)->clone(derivate_output);

    auto derivate_output_b = new ccma::algebra::DenseMatrixT<real>();
    derivate_output->clone(derivate_output_b);
    derivate_output_b->multiply(-1);
    derivate_output_b->add(1);

    derivate_output->multiply(derivate_output_b);
    derivate_output->multiply(_error);

    delete derivate_output_b;

    /*
     * calc delta: weight.T * derivate_output
     */
    auto delta = new ccma::algebra::DenseMatrixT<real>();
    this->get_weight(0, 0)->clone(delta);
    delta->transpose();
    delta->dot(derivate_output);
    this->set_delta(0, delta);

    //if pre_layer is ConvolutionLayer, has sigmoid function
    if(typeid(*pre_layer) == typeid(ConvolutionLayer)){
        auto av1 = new ccma::algebra::DenseMatrixT<real>();
        auto av2 = new ccma::algebra::DenseMatrixT<real>();
        _av->clone(av1);
        _av->clone(av2);
        /*
         * derivate_sigmoid: z * (1-z)
         */
        av2->multiply(-1);
        av2->add(1);
        av1->multiply(av2);
        this->get_delta(0)->multiply(av1);

        delete av1;
        delete av2;
    }

    /*
     * derivate_weight = derivate_output * av.T
     * derivate_bias = derviate_output
     */
    auto derivate_weight = new ccma::algebra::DenseMatrixT<real>();
    auto derivate_bias   = new ccma::algebra::DenseMatrixT<real>();
    auto avt             = new ccma::algebra::DenseMatrixT<real>();
    derivate_output->clone(derivate_weight);
    derivate_output->clone(derivate_bias);
    delete derivate_output;
    _av->clone(avt);

    avt->transpose();
    derivate_weight->dot(avt);
    delete avt;

    /*
     * update weight & bias
     */
    //todo set alpha
    real alpha = 0.5;
    derivate_weight->multiply(alpha);
    derivate_bias->multiply(alpha);
    this->get_weight(0, 0)->subtract(derivate_weight);
    this->get_bias()->subtract(derivate_bias);
    delete derivate_weight;
    delete derivate_bias;
}

}//namespace cnn
}//namespace algorithm
}//namespace ccma
