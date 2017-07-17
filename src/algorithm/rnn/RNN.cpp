/***********************************************
 * Author: Jun Jiang - jiangjun4@sina.com
 * Create: 2017-06-23 14:32
 * Last modified : 2017-06-23 14:32
 * Filename      : RNN.cpp
 * Description   : 
 **********************************************/

#include "algorithm/rnn/RNN.h"

namespace ccma{
namespace algorithm{
namespace rnn{

void RNN::sgd(std::vector<ccma::algebra::BaseMatrixT<real>*>* train_seq_data,
              std::vector<ccma::algebra::BaseMatrixT<real>*>* train_seq_label, 
              const uint epoch,
              const uint mini_batch_size,
              const real alpha){

	if(!check_data(train_seq_data, train_seq_label)){
        printf("RNN::sgd Data dim Error.\n");
		return ;
	}

    auto seq_data  = new ccma::algebra::DenseMatrixT<real>();
	auto seq_label = new ccma::algebra::DenseMatrixT<real>();
	
	uint num_train_data = train_seq_data->size();
	bool debug = (num_train_data <= 5);
    
	auto now = []{return std::chrono::system_clock::now();};

	for(uint i = 0; i != epoch; i++){

		auto start_time = now();
        std::vector<ccma::algebra::BaseMatrixT<real>*> mini_batch_data;
        std::vector<ccma::algebra::BaseMatrixT<real>*> mini_batch_label;

		for(uint j = 0; j != num_train_data; j++){

            if(j % mini_batch_size == 0){
                mini_batch_data.clear();
                mini_batch_label.clear();
            }
            
            mini_batch_data.push_back(train_seq_data->at(j));
            mini_batch_label.push_back(train_seq_label->at(j));

            if( j % mini_batch_size == (mini_batch_size - 1) || j == (num_train_data - 1)){
                mini_batch_update(mini_batch_data, mini_batch_label, alpha, debug, j);

                mini_batch_data.clear();
                mini_batch_label.clear();
            }
		}

        if(_path != ""){
            write_model(_path);
        }

        auto training_time = now();
        printf("Epoch[%d] training run time: %lld ms, loss[%f] base loss[%f]\n", i, (long long int)std::chrono::duration_cast<std::chrono::milliseconds>(training_time - start_time).count(), loss(train_seq_data, train_seq_label), std::log(_feature_dim));
	}

	printf("training finished.\n");

    delete seq_data;
	delete seq_label;
}

void RNN::mini_batch_update(std::vector<ccma::algebra::BaseMatrixT<real>*> train_seq_data,
                            std::vector<ccma::algebra::BaseMatrixT<real>*> train_seq_label, 
		         	        const real alpha,
                            const bool debug,
                            const int j){

    const uint num_train_data   = train_seq_data.size();
    ccma::algebra::BaseMatrixT<real>* derivate_weight       = new ccma::algebra::DenseMatrixT<real>[num_train_data];
    ccma::algebra::BaseMatrixT<real>* derivate_pre_weight   = new ccma::algebra::DenseMatrixT<real>[num_train_data];
    ccma::algebra::BaseMatrixT<real>* derivate_act_weight   = new ccma::algebra::DenseMatrixT<real>[num_train_data];

    const uint num_thread = std::min(num_train_data, _num_hardware_concurrency);

    std::vector<std::thread> threads(num_thread);
    for(uint i = 0; i != num_train_data; i++){
        threads[i % num_thread] = std::thread(std::mem_fun_ref(_layer::back_propagation), this, train_seq_data[i], train_seq_label[i], _U, _W, _V, &derivate_weight[i], &derivate_pre_weight[i], &derivate_act_weight[i], debug);
        if(i == (num_train_data - 1) || i % num_thread == (num_thread - 1) ){
            for(uint j = i / num_thread * num_thread; j != i; j++){
                threads[j % num_thread].join();
            }
        }
    }

    for(uint i = 1; i != num_train_data; i++){
        derivate_weight[0].add(&derivate_weight[i]);
        derivate_pre_weight[0].add(&derivate_pre_weight[i]);
        derivate_act_weight[0].add(&derivate_act_weight[i]);
    }

    derivate_weight[0].multiply(alpha / num_train_data);
    derivate_pre_weight[0].multiply(alpha / num_train_data);
    derivate_act_weight[0].multiply(alpha / num_train_data);

    _U->subtract(&derivate_weight[0]);
    _W->subtract(&derivate_pre_weight[0]);
    _V->subtract(&derivate_act_weight[0]);

    delete[] derivate_weight;
    delete[] derivate_pre_weight;
    delete[] derivate_act_weight;
}

/*
void RNN::sgd_step(ccma::algebra::BaseMatrixT<real>* train_seq_data,
                   ccma::algebra::BaseMatrixT<real>* train_seq_label, 
			       real alpha,
                   bool debug){

    auto derivate_weight     = new ccma::algebra::DenseMatrixT<real>();
    auto derivate_pre_weight = new ccma::algebra::DenseMatrixT<real>();
    auto derivate_act_weight = new ccma::algebra::DenseMatrixT<real>();

	_layer->back_propagation(train_seq_data, train_seq_label, _U, _W, _V, derivate_weight, derivate_pre_weight, derivate_act_weight, debug);

	derivate_weight->multiply(alpha);
	_U->subtract(derivate_weight);

	derivate_pre_weight->multiply(alpha);
	_W->subtract(derivate_pre_weight);

	derivate_act_weight->multiply(alpha);
	_V->subtract(derivate_act_weight);

    delete derivate_weight;
    delete derivate_pre_weight;
    delete derivate_act_weight;
}
*/

real RNN::total_loss(std::vector<ccma::algebra::BaseMatrixT<real>*>* train_seq_data,
                     std::vector<ccma::algebra::BaseMatrixT<real>*>* train_seq_label){

	real loss_value = 0;
	
	auto seq_data   = new ccma::algebra::DenseMatrixT<real>();
	auto seq_label  = new ccma::algebra::DenseMatrixT<real>();
	
    auto state      = new ccma::algebra::DenseMatrixT<real>();
	auto activation = new ccma::algebra::DenseMatrixT<real>();
	
    uint num_train_data = train_seq_data->size();
	for(uint j = 0; j != num_train_data; j++){

		train_seq_data->at(j)->clone(seq_data);
		train_seq_label->at(j)->clone(seq_label);

		_layer->feed_farward(seq_data, _U, _W, _V, state, activation, false);

		auto mat_label = seq_label->argmax(0);
        uint rows = mat_label->get_rows();
        for(uint row = 0; row != rows; row++){
            loss_value -= std::log(activation->get_data(row, mat_label->get_data(row, 0)));
        }
        delete mat_label;
	}

    delete seq_data;
	delete seq_label;

    delete state;
    delete activation;

	return loss_value;
}

real RNN::loss(std::vector<ccma::algebra::BaseMatrixT<real>*>* train_seq_data,
               std::vector<ccma::algebra::BaseMatrixT<real>*>* train_seq_label){

    //L(y, o) = - (1/N)(Sum y_n*log(o_n))

    real loss_value = total_loss(train_seq_data, train_seq_label);
    uint N = 0;
    
    uint size = train_seq_label->size();
    for(uint i = 0; i != size; i++){
        N += train_seq_label->at(i)->get_rows();
    }

    return loss_value / N;
}

bool RNN::check_data(std::vector<ccma::algebra::BaseMatrixT<real>*>* train_seq_data,
              		 std::vector<ccma::algebra::BaseMatrixT<real>*>* train_seq_label){ 
	uint size = train_seq_data->size();

	for(uint i = 0; i != size; i++){
		if(train_seq_data->at(i)->get_rows() != train_seq_label->at(i)->get_rows() ||
			train_seq_data->at(i)->get_cols() != train_seq_label->at(i)->get_cols()){
			printf("train_data[%d] do not match[%d-%d][%d-%d]\n", i, train_seq_data->at(i)->get_rows(), train_seq_label->at(i)->get_rows(), train_seq_data->at(i)->get_cols(), train_seq_label->at(i)->get_cols());
			
			return false;
		}
	}
	return true;
}			  

bool RNN::load_model(const std::string& path){
    std::vector<ccma::algebra::BaseMatrixT<real>*> models;
    if(!loader.read<real>(path, &models,"RNNMODEL") || models.size() != 3){
        for(auto&& model : models){
            delete model;
        }
        return false;
    }
    
    if(_U != nullptr){
        delete _U;
    }
    _U = models[0];
    if(_W != nullptr){
        delete _W;
    }
    _W = models[1];
    if(_V != nullptr){
        delete _V;
    }
    _U = models[2];

    _feature_dim    = _U->get_cols();
    _hidden_dim     = _U->get_rows();
    _path           = path;

    if(_layer != nullptr){
        delete _layer;
    }

    _layer = new ccma::algorithm::rnn::Layer(_hidden_dim, _bptt_truncate);

    return true;
}

bool RNN::write_model(const std::string& path){
    std::vector<ccma::algebra::BaseMatrixT<real>*> models;
    models.push_back(_U);
    models.push_back(_W);
    models.push_back(_V);

    return loader.write<real>(models, path, false, "RNNMODEL");
}

}//namespace rnn
}//namespace algorithm
}//namespace ccma
