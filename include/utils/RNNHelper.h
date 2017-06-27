
/***********************************************
 * Author: Jun Jiang - jiangjun4@sina.com
 * Create: 2017-06-23 15:40
 * Last modified : 2017-06-23 15:40
 * Filename      : RNNHelper.h
 * Description   : Read & generate one-hot 
                   word vector 
 **********************************************/

#ifndef _CCMA_UTILS_RNNHELPLER_H_
#define _CCMA_UTILS_RNNHELPLER_H_


#include <map>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <stdlib.h>
#include "algebra/BaseMatrix.h"
#include "utils/StringHelper.h"

namespace ccma{
namespace utils{

class RNNHelper{
public:
	RNNHelper(uint feature_dim){
		_feature_dim = feature_dim;
	}

	void read_word2index(const std::string& data_file, std::map<std::string, uint> dict);

	void read_seqdata(const std::string& data_file,
					  std::vector<ccma::algebra::BaseMatrixT<real>*>* train_seq_data,
					  const std::string& label_file,
					  std::vector<ccma::algebra::BaseMatrixT<real>*>* train_seq_label,
                      int limit = -1);
private:
	void read_data(const std::string& data_file,
                   std::vector<ccma::algebra::BaseMatrixT<real>*>* mat,
                   int limit = -1);

private:
	uint _feature_dim;

};//class RNNHelper

	
void RNNHelper::read_word2index(const std::string& data_file, std::map<std::string, uint> dict){
	std::ifstream in_file(data_file.data());;
	if(!in_file.is_open()){
		printf("Open file:%s failed.\n", data_file.c_str());
	}else{
        ccma::utils::StringHelper helper;
		std::string data;
		while(getline(in_file, data)){
            auto line_data = helper.split(data, '\t');
            if(line_data.size() == 2){
				uint value = helper.str2int(line_data[1]);
				if(value < _feature_dim){
					dict.insert(std::map<std::string, uint>::value_type(line_data[0], value));
				}
            }
		}
		in_file.close();
	}
}

void RNNHelper::read_seqdata(const std::string& data_file,
				   		     std::vector<ccma::algebra::BaseMatrixT<real>*>* train_seq_data,
					  		 const std::string& label_file,
					  		 std::vector<ccma::algebra::BaseMatrixT<real>*>* train_seq_label,
                             int limit){
    read_data(data_file, train_seq_data, limit);
    read_data(label_file, train_seq_label, limit);
}

void RNNHelper::read_data(const std::string& data_file,
                          std::vector<ccma::algebra::BaseMatrixT<real>*>* mat,
                          int limit){
	std::ifstream in_file(data_file.c_str());
	if(!in_file.is_open()){
		printf("Open file:%s failed.\n", data_file.c_str());
	}else{
		std::string data;
        ccma::utils::StringHelper helper;
		while(getline(in_file, data)){
            auto line_data = helper.split(data, '\t');
            uint rows = line_data.size();
            auto mat_data = new ccma::algebra::DenseMatrixT<real>(rows, _feature_dim);
            for(uint i = 0; i != rows ; i++){
                mat_data->set_data(1, i, helper.str2int(line_data[i]));
            }
            mat->push_back(mat_data);
            if(limit > 0 && (int)mat->size() >= limit){
                break;
            }
		}
		in_file.close();
	}
}


}//namespace
}//namespace ccma

#endif
