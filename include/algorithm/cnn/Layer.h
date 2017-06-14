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

class Pooling{
public:
    virtual ~Pooling(){}
    virtual void pool(ccma::algebra::BaseMatrixT<real>* mat,
                      uint rows,
                      uint cols,
                      uint scale,
                      ccma::algebra::BaseMatrixT<real>* pooling_mat) = 0;
};//class Pooling
class MeanPooling : public Pooling{
public:
    void pool(ccma::algebra::BaseMatrixT<real>* mat,
              uint rows,
              uint cols,
              uint scale,
              ccma::algebra::BaseMatrixT<real>* pooling_mat);
};//class MeanPooling
class MaxPooling : public Pooling{
public:
    void pool(ccma::algebra::BaseMatrixT<real>* mat,
              uint rows,
              uint cols,
              uint scale,
              ccma::algebra::BaseMatrixT<real>* pooling_mat);
};//class MaxPooling
class L2Pooling : public Pooling{
public:
    void pool(ccma::algebra::BaseMatrixT<real>* mat,
              uint rows,
              uint cols,
              uint scale,
              ccma::algebra::BaseMatrixT<real>* pooling_mat);
};//class L2Pooling

class Layer{
public:
    Layer(uint rows,
          uint cols,
          uint in_channel_size,
          uint out_channel_size) : _rows(rows),
        _cols(cols),
        _in_channel_size(in_channel_size),
        _out_channel_size(out_channel_size),
	_is_last_layer(true){}

    virtual ~Layer(){
        if(_bias != nullptr){
            delete _bias;
            _bias = nullptr;
        }

        clear_vector_matrix(&_weights);
        clear_vector_matrix(&_activations);
        clear_vector_matrix(&_deltas);
    }

    virtual bool initialize(Layer* pre_layer = nullptr) = 0;
    virtual void feed_forward(Layer* pre_layer = nullptr, bool debug = false) = 0;
    virtual void back_propagation(Layer* pre_layer, Layer* back_layer = nullptr, bool debug = false) = 0;

    inline void set_rows(uint rows){_rows = rows;}
    inline uint get_rows(){return _rows;}

    inline void set_cols(uint cols){_cols = cols;}
    inline uint get_cols(){return _cols;}

    inline void set_is_last_layer(){_is_last_layer = false;}
    inline bool get_is_last_layer(){return _is_last_layer;}
    /*
     * pre_layer feature channel size
     */
    inline void set_in_channel_size(uint in_channel_size){_in_channel_size = in_channel_size;}
    inline uint get_in_channel_size(){return _in_channel_size;}

    /*
     * current_layer feature channel size
     */
    inline void set_out_channel_size(uint out_channel_size){_out_channel_size = out_channel_size;}
    inline uint get_out_channel_size(){return _out_channel_size;}

    //activation size equal out_channel_size
    inline void set_activation(uint out_channel_id, ccma::algebra::BaseMatrixT<real>* activation){
        set_vec_mat(&_activations, out_channel_id, activation);
    }
    inline ccma::algebra::BaseMatrixT<real>* get_activation(uint out_channel_id){
        return _activations[out_channel_id];
    }

    //delta size equal out_channel_size
    inline void set_delta(uint out_channel_id, ccma::algebra::BaseMatrixT<real>* delta){
        set_vec_mat(&_deltas, out_channel_id, delta);
    }
    inline ccma::algebra::BaseMatrixT<real>* get_delta(uint out_channel_id){
        return _deltas[out_channel_id];
    }

    //weight size equal in_channel_size * out_channel_size
    inline void set_weight(uint in_channel_id,
                           uint out_channel_id,
                           ccma::algebra::BaseMatrixT<real>* weight){
        set_vec_mat(&_weights, in_channel_id * this->_out_channel_size + out_channel_id, weight);
    }
    inline ccma::algebra::BaseMatrixT<real>* get_weight(uint in_channel_id, uint out_channel_id){
        return _weights[in_channel_id * this->_out_channel_size + out_channel_id];
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

    inline void set_vec_mat(std::vector<ccma::algebra::BaseMatrixT<real>*>* vec_mat, uint idx, ccma::algebra::BaseMatrixT<real>* mat){
        uint size = vec_mat->size();
        if(size > idx){
            clear_vector_matrix(vec_mat);
        }
        if(vec_mat->size() == idx){
            vec_mat->push_back(mat);
        }else{
            printf("set_vec_mat error:[%d/%d]\n", size, idx);
        }
    }

protected:
    uint _rows;
    uint _cols;
    /*pre_layer feature channel size*/
    uint _in_channel_size;
    /* current_layer feature channel size*/
    uint _out_channel_size;
    std::vector<ccma::algebra::BaseMatrixT<real>*> _weights;
    ccma::algebra::BaseMatrixT<real>* _bias;

    real _alpha = 0.1;
private:
    std::vector<ccma::algebra::BaseMatrixT<real>*> _activations;
    std::vector<ccma::algebra::BaseMatrixT<real>*> _deltas;

    bool _is_last_layer = true;
};//class Layer

class DataLayer:public Layer{
public:
    DataLayer(uint rows, uint cols):Layer(rows, cols, 1, 1){}
    ~DataLayer(){
        _x = nullptr; //out pointer,no delete
    }
    bool initialize(Layer* pre_layer = nullptr);
    void feed_forward(Layer* pre_layer = nullptr, bool debug = false);
    void back_propagation(Layer* pre_layer, Layer* back_layer = nullptr, bool debug = false);

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
    SubSamplingLayer(uint scale, Pooling* pooling):Layer(0, 0, 0, 0){
        _scale = scale;
        _pooling = pooling;
    }
    ~SubSamplingLayer(){
        delete _pooling;
    }
    bool initialize(Layer* pre_layer = nullptr);
    void feed_forward(Layer* pre_layer = nullptr, bool debug = false);
    void back_propagation(Layer* pre_layer, Layer* back_layer = nullptr, bool debug = false);

    uint get_scale(){return _scale;}

protected:
    uint _scale;
private:
    Pooling* _pooling;
};//class SubsamplingLayer


class ConvolutionLayer:public Layer{
public:
     ConvolutionLayer(uint kernal_size, uint stride, uint out_channel_size):Layer(0, 0, 1, out_channel_size){
        _kernal_size = kernal_size;
        _stride = stride;
    }
    inline uint get_stride()const {return _stride;}

    bool initialize(Layer* pre_layer = nullptr);
    void feed_forward(Layer* pre_layer = nullptr, bool debug = false);
    void back_propagation(Layer* pre_layer, Layer* back_layer = nullptr, bool debug = false);
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

        _y = nullptr;//out pointer, not delete data.
    }
    bool initialize(Layer* pre_layer = nullptr);
    void feed_forward(Layer* pre_layer = nullptr, bool debug = false);
    void back_propagation(Layer* pre_layer, Layer* back_layer = nullptr, bool debug = false);

    bool set_y(ccma::algebra::BaseMatrixT<real>* y){
        if(y->get_rows() == _rows){
            _y = y;
            return true;
        }
        return false;
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
