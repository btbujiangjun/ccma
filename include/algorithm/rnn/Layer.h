/***********************************************
 * Author: Jun Jiang - jiangjun4@sina.com
 * Create: 2017-06-20 16:23
 * Last modified : 2017-06-20 16:23
 * Filename      : Layer.h
 * Description   : RNN network Layer 
 **********************************************/

#ifndef _CCMA_ALGORITHM_RNN_LAYER_H
#define _CCMA_ALGORITHM_RNN_LAYER_H_

#include "algebra/BaseMatrix.h"

namespace ccma{
namespace algorithm{
namespace rnn{

class Layer{
public:
	Layer(uint dim){
		_dim = dim;	
	}

	~Layer(){
		if(_store != nullptr){
			delete _store;
		}
		if(_activation != nullptr){
			delete _activation;
		}
		if(_weight != nullptr){
			delete _weight;
		}
		if(_pre_weight != nullptr){
			delete _pre_weight;
		}
	}

	inline uint get_dim(){return _dim;}
	inline ccma::algebra::BaseMatrix<real>* get_store(){return _store;}


	virtual bool initialize(Layer* pre_layer = nullptr) = 0;
	virtual void feed_farward(Layer* pre_layer = nullptr, debug = false) = 0;
	virtual void back_propagation(Layer* pre_layer,
								  Layer* back_layer = nullptr, 
								  bool debug = false);

private:
	uint _dim;

	ccma::algebra::BaseMatrix<real>* _store;
	ccma::algebra::BaseMatrix<real>* _activation;

	ccma::algebra::BaseMatrix<real>* _weight;
	ccma::algebra::BaseMatrix<real>* _pre_weight;
};//class Layer


class DataLayer:public Layer{
public:
	DataLayer(uint dim):Layer(dim){}

	void set_x(ccma::algebra::BaseMatrix<real>* x){_x = x;}

	bool initialize(Layer* pre_layer = nullptr);
	void feed_farward(Layer* pre_layer = nullptr, bool debug = false);
	void back_propagation(Layer* pre_layer,
						  Layer* back_layer = nullptr,
						  bool debug = false);

private:
	ccma::algebra::BaseMatrix<real>* _x;
};//class DataLayer

class RNNLayer : public Layer{
public:
	RNNLayer(uint dim):Layer(dim){}
	bool initialize(Layer* pre_layer = nullptr);
	void feed_farward(Layer* pre_layer = nullptr, bool debug = false);
	void back_propagation(Layer* pre_layer,
						  Layer* back_layer = nullptr,
						  bool debug = false);
};//class RNNLayer

class FullConnetionLayer:public Layer{
public:
	FullConnetionLayer(dim dim):Layer(dim){}

	void set_y(ccma::algebra::BaseMatrix<real>* y){_y = y;}


	bool initialize(Layer* pre_layer = nullptr);
	void feed_farward(Layer* pre_layer = nullptr, bool debug = false);
	void back_propagation(Layer* pre_layer,
						  Layer* back_layer = nullptr,
						  bool debug = false);

private:
	ccma::algebra::BaseMatrix<real>* _y;
};//class FullConnetionLayer

}//namespace rnn
}//namespace algorithm
}//namespace ccma

#endif
