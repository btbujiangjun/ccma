/*********************************************
* Author: Jun Jiang - jiangjun4@sina.com
* Created: 2017-05-22 16:47
* Last modified: 2017-05-22 16:47
* Filename: CNN.cpp
* Description: convolutional network 
**********************************************/
#include <typeinfo>
#include <math.h>
#include "algorithm/cnn/Layer.h"

namespace ccma{
namespace algorithm{
namespace cnn{

bool DataLayer::initialize(Layer* pre_layer){
    return true;
}
void DataLayer::feed_forward(Layer* pre_layer, bool debug){
    auto activation = new ccma::algebra::DenseMatrixT<real>();
    _x->clone(activation);
    this->set_activation(0, activation);

	//activation->display("|");
}
void DataLayer::back_propagation(Layer* pre_layer, Layer* back_layer){
}

bool SubSamplingLayer::initialize(Layer* pre_layer){
    uint pre_rows = pre_layer->get_rows();
    uint pre_cols = pre_layer->get_cols();

    if(pre_rows % _scale != 0 || pre_cols % _scale != 0){
        printf("SubSampling Layer scale error:pre_rows[%d]pre_cols[%d]scale[%d].\n", pre_rows, pre_cols, _scale);
        return false;
    }
    this->_rows = pre_rows / _scale;
    this->_cols = pre_cols / _scale;
    //can't change out channel size.
    this->_in_channel_size = this->_out_channel_size = pre_layer->get_out_channel_size();
    /*
     * pre_layer, each channel share a bias and initialize value is zero.
     * no pooling weight.
     */
    set_bias(new ccma::algebra::DenseColMatrixT<real>(this->_out_channel_size, 0.0));
    return true;
}
void SubSamplingLayer::feed_forward(Layer* pre_layer, bool debug){
    // in_channel size equal out_channel size to subsampling layer.
    for(uint i = 0; i != this->_in_channel_size; i++){
        auto a = pre_layer->get_activation(i);
        auto activation = new ccma::algebra::DenseMatrixT<real>();
        _pooling->pool(a, this->_rows, this->_cols, this->_scale, activation);
        this->set_activation(i, activation);
		//activation->display("|");
	 }
}
void SubSamplingLayer::back_propagation(Layer* pre_layer, Layer* back_layer){
    if(back_layer->get_is_last_layer()){
        real* data = ((FullConnectionLayer*)back_layer)->get_av()->get_data();
        for(uint i = 0; i != this->_out_channel_size; i++){
            auto delta = new ccma::algebra::DenseMatrixT<real>();
            real* d = new real[_rows * _cols];
            memcpy(d, &data[i * this->_rows * this->_cols], sizeof(real) * this->_rows * this->_cols);
            delta->set_shallow_data(d, this->_rows, this->_cols);
            this->set_delta(i, delta);

			//printf("sub back-full[%d]", i);
			//delta->display();

        }
    }else if(typeid(*back_layer) == typeid(ConvolutionLayer)){
        uint stride = ((ConvolutionLayer*)back_layer)->get_stride();
        auto d = new ccma::algebra::DenseMatrixT<real>();
        auto w = new ccma::algebra::DenseMatrixT<real>();
        for(uint i = 0 ; i != this->_out_channel_size; i++){
            auto delta = new ccma::algebra::DenseMatrixT<real>();
            for(uint j = 0; j != back_layer->get_out_channel_size(); j++){
                back_layer->get_delta(j)->clone(d);
                back_layer->get_weight(i, j)->clone(w);
                d->convn(w, stride, "full");
                delta->add(d);
            }
            this->set_delta(i, delta);

			//printf("sub back-conv[%d]", i);
			//delta->display();

        }
        delete d;
        delete w;
    }
}

bool ConvolutionLayer::initialize(Layer* pre_layer){
    uint pre_rows = pre_layer->get_rows();
    uint pre_cols = pre_layer->get_cols();

    if(pre_rows < _kernal_size || pre_cols < _kernal_size){
        printf("ConvolutionLayer Size Erorr: pre_rows[%d] pre_cols[%d] less than kernal_size[%d].\n", pre_rows, pre_cols, _kernal_size);
        return false;
    }

    this->_rows = (pre_rows - _kernal_size) % _stride == 0 ? (pre_rows - _kernal_size) / _stride + 1 : (pre_rows - _kernal_size) / _stride + 2;
    this->_cols = (pre_cols - _kernal_size) % _stride == 0 ? (pre_cols - _kernal_size) / _stride + 1 : (pre_cols - _kernal_size) / _stride + 2;

    this->_in_channel_size = pre_layer->get_out_channel_size();

    for(uint i = 0; i != this->_in_channel_size; i++){
        for(uint j = 0; j != this->_out_channel_size; j++){
            auto weight = new ccma::algebra::DenseRandomMatrixT<real>(_kernal_size, _kernal_size, 0.0, 1.0);
            this->set_weight(i, j, weight);
        }
    }
    for(uint i = 0; i != this->_in_channel_size; i++){
        for(uint j = 0; j != this->_out_channel_size; j++){
            this->get_weight(i, j)->display();
        }
    }

    /*
     * channel shared the same bias of current layer.
     */
    this->set_bias(new ccma::algebra::DenseColMatrixT<real>(this->_out_channel_size, 0.0));

    return true;
}

void ConvolutionLayer::feed_forward(Layer* pre_layer, bool debug){
    auto a = new ccma::algebra::DenseMatrixT<real>();
    //foreach output channel
    for(uint i = 0; i != this->_out_channel_size; i++){
        auto activation = new ccma::algebra::DenseMatrixT<real>();
        for(uint j = 0; j != pre_layer->get_out_channel_size(); j++){
            pre_layer->get_activation(j)->clone(a);
			//printf("conv pre activation");
			//a->display("|");
			//this->get_weight(j, i)->display("|");
            a->convn(this->get_weight(j, i), _stride, "valid");
            //sum all channels of pre_layer.
            activation->add(a);
        }
        //add shared bias of channel in current layer.
        activation->add(this->get_bias()->get_data(i, 0));
        //if sigmoid activative function.
        activation->sigmoid();
        this->set_activation(i, activation);
		//activation->display("|");
    }
    delete a;
}

void ConvolutionLayer::back_propagation(Layer* pre_layer, Layer* back_layer){
    //todo alpha
    real alpha = 5;
    if(back_layer->get_is_last_layer()){
        real* data = ((FullConnectionLayer*)back_layer)->get_av()->get_data();
        for(uint i = 0; i != this->_out_channel_size; i++){
            auto delta = new ccma::algebra::DenseMatrixT<real>();
            real* d = new real[this->_rows * this->_cols];
            memcpy(d, &data[i * this->_rows * this->_cols], sizeof(real) * this->_rows * this->_cols);
            delta->set_shallow_data(d, this->_rows, this->_cols);
            this->set_delta(i, delta);
        }
    }else if(typeid(*back_layer) == typeid(SubSamplingLayer)){

        SubSamplingLayer* sub_layer = (SubSamplingLayer*)back_layer;
        uint scale = sub_layer->get_scale();

        for(uint i = 0; i != this->_out_channel_size; i++){
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

		}
    }
    /*
     * calc grad and update weight/bias
     * only for online learning, if batch learning
     * need to average weight and bias
     */
    auto derivate_weight = new ccma::algebra::DenseMatrixT<real>();
    real* derivate_bias_data = new real[this->_out_channel_size];
    for(uint i = 0; i != this->_out_channel_size; i++){
		for(uint j = 0; j != pre_layer->get_out_channel_size(); j++){
	    	pre_layer->get_activation(j)->clone(derivate_weight);
		
            derivate_weight->convn(this->get_delta(i), _stride, "valid");
	    	/*
             * update grad: w -= alpha * derivate_weight
	     	*/
	    	derivate_weight->multiply(alpha);
        	this->get_weight(j, i)->subtract(derivate_weight);

			//printf("conv back weight[%d][%d]", j , i);
			//derivate_weight->display();

		}
		//update bias
        derivate_bias_data[i] = this->get_delta(i)->sum();
    }
    delete derivate_weight;

    auto derivate_bias = new ccma::algebra::DenseMatrixT<real>();
    derivate_bias->set_shallow_data(derivate_bias_data, this->_out_channel_size, 1);
    derivate_bias->multiply(alpha);
    this->get_bias()->subtract(derivate_bias);
   
	//derivate_bias->display();

	delete derivate_bias;
}

bool FullConnectionLayer::initialize(Layer* pre_layer){
    _cols = pre_layer->get_rows() * pre_layer->get_cols() * pre_layer->get_out_channel_size();
    this->_in_channel_size = pre_layer->get_out_channel_size();

    this->set_bias(new ccma::algebra::DenseColMatrixT<real>(_rows, 0.0));
    auto weight = new ccma::algebra::DenseRandomMatrixT<real>(this->_rows, this->_cols, 0.0, 1.0);
    this->set_weight(0, 0, weight);
    return true;
}
void FullConnectionLayer::feed_forward(Layer* pre_layer, bool debug){
    /*
     * concatenate pre_layer's all channel mat into vector
     */
    auto a = new ccma::algebra::DenseMatrixT<real>();
    auto av = new ccma::algebra::DenseMatrixT<real>();
    for(uint i = 0; i != this->_in_channel_size; i++){
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
    activation->dot(_av);
    activation->add(this->get_bias());
    //if sigmoid activative function
    activation->sigmoid();
    this->set_activation(0, activation);
	
	//activation->display();
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
    real alpha = 5;
    derivate_weight->multiply(alpha);
    derivate_bias->multiply(alpha);
    this->get_weight(0, 0)->subtract(derivate_weight);
    this->get_bias()->subtract(derivate_bias);

	//printf("full back");
	//derivate_weight->display();
	//derivate_bias->display();

    delete derivate_weight;
    delete derivate_bias;
}

void MeanPooling::pool(ccma::algebra::BaseMatrixT<real>* mat,
                       uint rows,
                       uint cols,
                       uint scale,
                       ccma::algebra::BaseMatrixT<real>* pooling_mat){
    real* data = new real[rows * cols];
    uint pooling_size = scale * scale;
    for(uint j = 0; j != rows; j++){
        for(uint k = 0; k != cols; k++){
            real pooling_value = 0;
            for(uint m = 0; m != scale; m++){
                for(uint n = 0; n != scale; n++){
                    pooling_value += mat->get_data(j * scale + m, k * scale + n);
                }
            }
            data[j * cols + k] = pooling_value / pooling_size;
        }
    }
    pooling_mat->set_shallow_data(data, rows, cols);
}
void MaxPooling::pool(ccma::algebra::BaseMatrixT<real>* mat,
                      uint rows,
                      uint cols,
                      uint scale,
                      ccma::algebra::BaseMatrixT<real>* pooling_mat){
    real* data = new real[rows * cols];
    for(uint j = 0; j != rows; j++){
        for(uint k = 0; k != cols; k++){
            real pooling_value = 0;
            for(uint m = 0; m != scale; m++){
                for(uint n = 0; n != scale; n++){
                    real value = mat->get_data(j * scale + m, k * scale + n);
                    if((m == 0 && n == 0) || value > pooling_value){
                        pooling_value = value;
                    }
                }
            }
            data[j * cols + k] = pooling_value;
        }
    }
    pooling_mat->set_shallow_data(data, rows, cols);
}
void L2Pooling::pool(ccma::algebra::BaseMatrixT<real>* mat,
                     uint rows,
                     uint cols,
                     uint scale,
                     ccma::algebra::BaseMatrixT<real>* pooling_mat){
    real* data = new real[rows * cols];
    for(uint j = 0; j != rows; j++){
        for(uint k = 0; k != cols; k++){
            real pooling_value = 0;
            for(uint m = 0; m != scale; m++){
                for(uint n = 0; n != scale; n++){
                    real value = mat->get_data(j * scale + m, k * scale + n);
                    if(value != 0){
                        pooling_value += (value * value);
                    }
                }
            }
            data[j * cols + k] = sqrt(pooling_value);
        }
    }
    pooling_mat->set_shallow_data(data, rows, cols);
}

}//namespace cnn
}//namespace algorithm
}//namespace ccma
