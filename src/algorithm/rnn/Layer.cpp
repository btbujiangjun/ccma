/***********************************************
 * Author: Jun Jiang - jiangjun4@sina.com
 * Create: 2017-06-20 17:05
 * Last modified : 2017-06-20 17:05
 * Filename      : Layer.cpp
 * Description   : RNN network Layer 
 **********************************************/

#include "algorithm/rnn/Layer.h"

namespace ccma{
namespace algorithm{
namespace rnn{

bool DataLayer::initialize(){
	return true;
}

void DataLayer::feed_farword(Layer* pre_layer, bool debug){
	if(_store == nullptr){
		_store = new ccma::algebra::BaseMatrix<real>();
	}
}

}//namespace rnn
}//namespace algorithm
}//namespace ccma
