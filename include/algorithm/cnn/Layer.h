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

class Layers{
public:
    bool add_layer(Layer* layer);
    bool init_network();
    void feed_forward(ccma::algebra::BaseMatrixT<real>* data);
    void back_propagation();

private:
    std::vector<Layer*> _layers;
}; //class Layers


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
    virtual void back_propagation(Layer* back_layer = nullptr) = 0;

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
     * back_layer feature map size
     */
    inline void set_out_map_size(uint out_map_size){_out_map_size = out_map_size;}
    inline uint get_out_map_size(){return _out_map_size;}

    inline std::vector<ccma::algebra::BaseMatrixT<real>*> get_activations(){return _activations;}

    inline std::vector<ccma::algebra::BaseMatrixT<real>*> get_deltas(){return _deltas;}

    inline std::vector<ccma::algebra::BaseMatrixT<real>*> get_weights(){return _weights;}

    inline void set_bias(ccma::algebra::BaseMatrixT<real>* bias){
        if(_bias != nullptr){
            delete _bias;
        }
        _bias = bias;
    }
    inline ccma::algebra::BaseMatrixT<real> get_bias(){ return _bias;}

private:
    void clear_vector_matrix(std::vector<ccma::algebra::BaseMatrixT<real>*>* vec_mat){
        for(auto mat : vec_mat){
            delete mat;
            mat = nullptr;
        }
        vec_mat->clear();
    }

protected:
    uint _rows;
    uint _cols;
    uint _in_map_size;/*pre_layer feature map size*/
    uint _out_map_size;/* cur_layer feature map size*/

private:
    std::vector<ccma::algebra::BaseMatrixT<real>*> _activations;
    std::vector<ccma::algebra::BaseMatrixT<real>*> _deltas;
    std::vector<ccma::algebra::BaseMatrixT<real>*> _weights;
    ccma::algebra::BaseMatrixT<real>* _bias;
};//class Layer

class DataLayer:public Layer{
public:
    DataLayer(uint rows, uint cols):Layer(rows, cols, 1, 1){}
    ~DataLayer(){
        Layer::~Layer();
        if(_x != nullptr){
            delete _x;
            _x = nullptr;
        }
    }

    bool set_x(ccma::algebra::BaseMatrixT<real>* x){
        if(x->get_rows() == this->_rows && x->get_cols()){
            if(_x != nullptr){
                delete _x;
            }
            _x = x;
            return true;
        }
        return false;
    }
private:
    ccma::algebra::BaseMatrixT<real>* x;
};//class DataLayer

class SubSamplingLayer:public Layer{
public:
    SubSamplingLayer(uint scale):Layer(0, 0, 0, 0){
        _scale = scale;
    }

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
private:
     void convolute(ccma::algebra::BaseMatrixT<real>* mat, ccma::algebra::BaseMatrixT<real>* shared_weight);

protected:
    uint _stride;
    uint _kernal_size;
};//class ConvolutionLayer

class FullConnectionLayer:public Layer{
public:
    FullConnectionLayer(uint rows):(rows, 0, 0, 1){}
    ~FullConnectionLayer(){
        Layer::~Layer();
        if(_y != nullptr){
            delete _y;
            _y = nullptr;
        }
        if(_error != nullptr){
            delete _error;
            _error = nullptr;
        }
    }

    bool set_y(ccma::algebra::BaseMatrixT<real>* y){
        if(y->get_rows() == _rows){
            if(_y != nullptr){
                delete _y;
            }
            _y = y;
            return true;
        }
        return false;
    }
private:
    ccma::algebra::BaseMatrixT<real>* _y;
    ccma::algebra::BaseMatrixT<real>* _error;
};//class FullConnectionLayer


}//namespace cnn
}//namespace algorithm
}//namespace ccma


#endif
