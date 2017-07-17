/***********************************************
 * Author: Jun Jiang - jiangjun4@sina.com
 * Create: 2017-06-20 16:08
 * Last modified : 2017-06-20 16:08
 * Filename      : RNN.h
 * Description   : Recurrent Neural Network 
 **********************************************/

#ifndef _CCMA_ALGORITHM_RNN_RNN_H_
#define _CCMA_ALGORITHM_RNN_RNN_H_

#include <vector>
#include <thread>
#include "algorithm/rnn/Layer.h"
#include "utils/ModelLoader.h"

namespace ccma{
namespace algorithm{
namespace rnn{

class RNN{
public:
    RNN(uint feature_dim,
        uint hidden_dim,
        std::string path = "",
        uint bptt_truncate = 4){

        _feature_dim    = feature_dim;
        _hidden_dim     = hidden_dim;
        _bptt_truncate  = bptt_truncate;
        _path           = path;

        _U = new ccma::algebra::DenseRandomMatrixT<real>(hidden_dim, feature_dim, 0, 1, -std::sqrt(1.0/feature_dim), std::sqrt(1.0/feature_dim));
        _W = new ccma::algebra::DenseRandomMatrixT<real>(hidden_dim, hidden_dim, 0, 1, -std::sqrt(1.0/hidden_dim), std::sqrt(1.0/hidden_dim));
        _V = new ccma::algebra::DenseRandomMatrixT<real>(feature_dim, hidden_dim, 0, 1, -std::sqrt(1.0/hidden_dim), std::sqrt(1.0/hidden_dim));

        _layer = new ccma::algorithm::rnn::Layer(hidden_dim, bptt_truncate);
    }

    RNN(const std::string& path, uint bptt_truncate = 4){
        if(load_model(path)){
            _feature_dim    = _U->get_cols();
            _hidden_dim     = _U->get_rows();
            _bptt_truncate  = bptt_truncate;
            _path           = path;
            _layer          = new ccma::algorithm::rnn::Layer(_hidden_dim, _bptt_truncate);
        }

    }
    ~RNN(){
        if(_U != nullptr){
            delete _U;
        }
        if(_W != nullptr){
            delete _W;
        }
        if(_V != nullptr){
            delete _V;
        }
        delete _layer;
    }

    void sgd(std::vector<ccma::algebra::BaseMatrixT<real>*>* train_seq_data,
             std::vector<ccma::algebra::BaseMatrixT<real>*>* train_seq_label, 
             const uint epoch = 5,
             const uint mini_batch_size = 1,
             const real alpha = 0.1);

    bool load_model(const std::string& path);
    bool write_model(const std::string& path);

private:
    void mini_batch_update(std::vector<ccma::algebra::BaseMatrixT<real>*> train_seq_data,
                           std::vector<ccma::algebra::BaseMatrixT<real>*> train_seq_label, 
			  	           const real alpha,
                           const bool debug,
				           const int j);

    real loss(std::vector<ccma::algebra::BaseMatrixT<real>*>* train_seq_data,
              std::vector<ccma::algebra::BaseMatrixT<real>*>* train_seq_label);

    real total_loss(std::vector<ccma::algebra::BaseMatrixT<real>*>* train_seq_data,
                    std::vector<ccma::algebra::BaseMatrixT<real>*>* train_seq_label);
    
	bool check_data(std::vector<ccma::algebra::BaseMatrixT<real>*>* train_seq_data,
             		std::vector<ccma::algebra::BaseMatrixT<real>*>* train_seq_label); 
private:
    uint _feature_dim;
    uint _hidden_dim;
    uint _bptt_truncate; 

    ccma::utils::ModelLoader loader;
    std::string _path;

    ccma::algebra::BaseMatrixT<real>* _U;
    ccma::algebra::BaseMatrixT<real>* _W;
    ccma::algebra::BaseMatrixT<real>* _V;

    Layer* _layer;
    const uint _num_hardware_concurrency = std::thread::hardware_concurrency() == 0 ? 1 : std::thread::hardware_concurrency();
};//class RNN 

}//namespace rnn
}//namespace algorithm
}//namespace ccma

#endif
