/*********************************************
* Author: Jun Jiang - jiangjun4@sina.com
* Created: 2017-05-22 15:03
* Last modified: 2017-05-22 15:03
* Filename: Layer.h
* Description: CNN network layer
**********************************************/

#ifndef _CCMA_ALGORITHM_CNN_LAYER_H_
#define _CCMA_ALGORITHM_CNN_LAYER_H_

#include <vector>
#include "algebra/BaseMatrix.h"

namespace ccma{
namespace algorithm{
namespace cnn{

class Layer{
public:
    Layer(uint rows,
          uint cols,
          uint in_map_size,
          uint out_map_size) : _rows(rows), 
        _cols(cols), 
        _in_map_size(in_map_size),
        _out_map_size(out_map_size){}

    ~Layer(){
        if(_bias != nullptr){
            delete _bias;
            _bias = nullptr;
        }

        clear_vector_matrix(&_weights);
        clear_vector_matrix(&_activations);
        clear_vector_matrix(&_deltas);
    }

    virtual bool initialize(Layer* pre_layer = nullptr) = 0;
    virtual void feed_forward(Layer* pre_layer = nullptr) = 0;
    virtual void back_propagation(Layer* pre_layer, Layer* back_layer = nullptr) = 0;

    inline void set_rows(uint rows){_rows = rows;}
    inline uint get_rows(){return _rows;}

    inline void set_cols(uint cols){_cols = cols;}
    inline uint get_cols(){return _cols;}

    /*
     * pre_layer feature map size
     */
    inline void set_in_map_size(uint in_map_size){_in_map_size = in_map_size;}
    inline uint get_in_map_size(){return _in_map_size;}

    /*
     * current_layer feature map size
     */
    inline void set_out_map_size(uint out_map_size){_out_map_size = out_map_size;}
    inline uint get_out_map_size(){return _out_map_size;}

    //activation size equal out_map_size
    inline void set_activation(uint out_map_id, ccma::algebra::BaseMatrixT<real>* activation){
        if(_activations.size() > out_map_id){
            auto a = _activations[out_map_id];
            delete a;
            a = activation;
        }else{
            _activations.push_back(activation);
        }
    }
    inline ccma::algebra::BaseMatrixT<real>* get_activation(uint out_map_id){
        return _activations[out_map_id];
    }

    //delta size equal out_map_size
    inline void set_delta(uint out_map_id, ccma::algebra::BaseMatrixT<real>* delta){
        if(_deltas.size() > out_map_id){
            auto d = _deltas[out_map_id];
            delete d;
            d = delta;
        }else{
            _deltas.push_back(delta);
        }
    }
    inline ccma::algebra::BaseMatrixT<real>* get_delta(uint out_map_id){
        return _deltas[out_map_id];
    }

    //weight size equal out_map_size * in_map_size
    inline void set_weight(uint out_map_id,
                           uint in_map_id,
                           ccma::algebra::BaseMatrixT<real>* weight){
        if(_weights.size() > out_map_id * this->_in_map_size + in_map_id){
            auto w = _weights[out_map_id * this->_in_map_size + in_map_id];
            delete w;
            w = weight;
        }else{
            _weights.push_back(weight);
        }
    }
    inline ccma::algebra::BaseMatrixT<real>* get_weight(uint out_map_id, uint in_map_id){
        return _weights[out_map_id * this->_in_map_size + in_map_id];
    }

    inline void set_bias(ccma::algebra::BaseMatrixT<real>* bias){
        if(_bias != nullptr){
            delete _bias;
        }
        _bias = bias;
    }
    inline ccma::algebra::BaseMatrixT<real>* get_bias(){ return _bias;}

private:
    inline void clear_vector_matrix(std::vector<ccma::algebra::BaseMatrixT<real>*>* vec_mat){
        for(auto mat : *vec_mat){
            delete mat;
            mat = nullptr;
        }
        vec_mat->clear();
    }

protected:
    uint _rows;
    uint _cols;
    /*pre_layer feature map size*/
    uint _in_map_size;
    /* current_layer feature map size*/
    uint _out_map_size;

private:
    std::vector<ccma::algebra::BaseMatrixT<real>*> _activations;
    std::vector<ccma::algebra::BaseMatrixT<real>*> _deltas;
protected:
    std::vector<ccma::algebra::BaseMatrixT<real>*> _weights;
    ccma::algebra::BaseMatrixT<real>* _bias;
};//class Layer

class DataLayer:public Layer{
public:
    DataLayer(uint rows, uint cols):Layer(rows, cols, 1, 1){}
    bool initialize(Layer* pre_layer = nullptr);
    void feed_forward(Layer* pre_layer = nullptr);
    void back_propagation(Layer* pre_layer, Layer* back_layer = nullptr);

    bool set_x(ccma::algebra::BaseMatrixT<real>* x){
        if(x->get_rows() == this->_rows && x->get_cols() == this->_cols){
            _x = x;
            return true;
        }
        printf("DataLayer mat dim error\n");
        return false;
    }
private:
    ccma::algebra::BaseMatrixT<real>* _x;
};//class DataLayer

class SubSamplingLayer:public Layer{
public:
    SubSamplingLayer(uint scale):Layer(0, 0, 0, 0){
        _scale = scale;
    }
    bool initialize(Layer* pre_layer = nullptr);
    void feed_forward(Layer* pre_layer = nullptr);
    void back_propagation(Layer* pre_layer, Layer* back_layer = nullptr);

    uint get_scale(){return _scale;}

protected:
    uint _scale;
};//class SubsamplingLayer

class ConvolutionLayer:public Layer{
public:
     ConvolutionLayer(uint kernal_size, uint stride, uint out_map_size):Layer(0, 0, 1, out_map_size){
        _kernal_size = kernal_size;
        _stride = stride;
    }
    inline uint get_stride()const {return _stride;}

    bool initialize(Layer* pre_layer = nullptr);
    void feed_forward(Layer* pre_layer = nullptr);
    void back_propagation(Layer* pre_layer, Layer* back_layer = nullptr);

protected:
    uint _stride;
    uint _kernal_size;
};//class ConvolutionLayer

class FullConnectionLayer:public Layer{
public:
    FullConnectionLayer(uint rows):Layer(rows, 0, 0, 1){}
    ~FullConnectionLayer(){
        if(_av != nullptr){
            delete _av;
            _av = nullptr;
        }
        if(_error != nullptr){
            delete _error;
            _error = nullptr;
        }
    }
    bool initialize(Layer* pre_layer = nullptr);
    void feed_forward(Layer* pre_layer = nullptr);
    void back_propagation(Layer* pre_layer, Layer* back_layer = nullptr);

    bool set_y(ccma::algebra::BaseMatrixT<real>* y){
        if(y->get_rows() == _rows){
            _y = y;
            return true;
        }
        return false;
    }

    ccma::algebra::BaseMatrixT<real>* get_av(){
        return _av;
    }

    real get_loss() const{
        return _loss;
    }
private:
    ccma::algebra::BaseMatrixT<real>* _y;
    ccma::algebra::BaseMatrixT<real>* _av;//pre_layer activations' vector
    ccma::algebra::BaseMatrixT<real>* _error;
    real _loss;
};//class FullConnectionLayer


}//namespace cnn
}//namespace algorithm
}//namespace ccma


#endif
