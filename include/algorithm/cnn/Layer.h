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

public://inner class
    class Layer{
    public:
        Layer(uint rows,
              uint cols,
              uint stride,
              uint scale,
              uint kernal_size,
              uint in_map_size,
              uint out_map_size) : _rows(rows), 
        _cols(cols), 
        _stride(stride), 
        _scale(scale), 
        _kernal_size(kernal_size), 
        _in_map_size(in_map_size),
        _out_map_size(out_map_size){}

        virtual void feed_forward() = 0;
        virtual void back_propagation() = 0;

        virtual void set_activations(real* activations) = 0;
        virtual real* get_activations() = 0;

        inline void set_rows(uint rows){_rows = rows;}
        inline uint get_rows(){return _rows;}
        inline void set_cols(uint cols){_cols = cols;}
        inline uint get_cols(){return _cols;}

        inline uint get_stride(){return _stride;}
        inline uint get_scale(){return _scale;}
        inline uint get_kernal_size(){return _kernal_size;}

        inline void set_in_map_size(uint in_map_size){_in_map_size = in_map_size;}
        inline uint get_in_map_size(){return _in_map_size;}

        inline uint get_out_map_size(){return _out_map_size;}

        inline void set_bias(ccma::algebra::BaseMatrixT<real>* bias){_bias = bias;}

    protected:
        uint _rows;
        uint _cols;
        uint _stride;
        uint _scale;
        uint _kernal_size;
        uint _in_map_size;
        uint _out_map_size;

    private:
        ccma::algebra::BaseMatrixT<real>* _weight;
        ccma::algebra::BaseMatrixT<real>* _bias;
    };//class Layer

    class DataLayer:public Layer{
    public:
        DataLayer(uint rows, uint cols):Layer(rows, cols, 0, 0, 0, 1, 0){}
    };//class DataLayer

    class SubSamplingLayer:public Layer{
    public:
        SubSamplingLayer(uint scale):Layer(0, 0, 0, scale, 0, 0, 0){}
    };//class SubsamplingLayer

    class ConvLayer:public Layer{
    public:
        ConvLayer(uint kernal_size, uint stride, uint in_map_size):Layer(0, 0, stride, 0, kernal_size, in_map_size, 0){}
    };//class ConvLayer

    class FullConnectionLayer:public Layer{
    public:
        FullConnectionLayer(uint rows, uint cols):(rows, cols, 0, 0, 0, 1, 0){}
    };//class FullConnectionLayer

};//class Layers


}//namespace cnn
}//namespace algorithm
}//namespace ccma


#endif
