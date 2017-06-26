
#ifndef _CCMA_UTILS_RNNHELPLER_H_
#define _CCMA_UTILS_RNNHELPLER_H_


#include <map>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <stdlib.h>
#include "algebra/BaseMatrix.h"

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
					  std::vector<ccma::algebra::BaseMatrixT<real>*>* train_seq_label)
private:
	void read_data(const std::string& data_file, std::vector<ccma::algebra::BaseMatrixT<real>*>* mat)

private:
	uint _feature_dim;

};//class RNNHelper

	
void RNNHelper::read_word2index(const std::string& data_file, std::map<std::string, uint> dict){
	std::ifstream in_file(data_file.data());;
	//in_file.open(data_file.data());
	if(!in_file.is_open()){
		printf("Open file:%s failed.\n", data_file.c_str());
	}else{
		std::string data;
		while(getline(in_file, data)){
			std::size_t idx = data.find_last_of('\t');
			if(idx != std::string::npos){
				uint value = atoi(data.substr(idx + 1).c_str());
				if(value < _feature_dim){
					dict.insert(std::map<std::string, uint>::value_type(data.substr(0, idx), value));
				}
			}
		}
		in_file.close();
	}
}

void RNNHelper::read_seqdata(const std::string& data_file,
				   		     std::vector<ccma::algebra::BaseMatrixT<real>*>* train_seq_data,
					  		 const std::string& label_file,
					  		 std::vector<ccma::algebra::BaseMatrixT<real>*>* train_seq_label){
	std::ifstream in_file(data_file.c_str());
	if(!in_file.is_open()){
		printf("Open file:%s failed.\n", data_file.c_str());
	}else{
		std::string data;
		while(getline(in_file, data)){
		
		}
	}
}
void RNNHelper::read_data(const std::string& data_file, std::vector<ccma::algebra::BaseMatrixT<real>*>* mat){
	std::ifstream in_file(data_file.c_str());
	if(!in_file.is_open()){
		printf("Open file:%s failed.\n", data_file.c_str());
	}else{
		std::string data;
		uint rows = 0;
		while(getline(in_file, data)){
			rows++;
		}
	
		in_file.seekg(0, std::ios::beg);
		while(getline(in_file, data){
			auto mat = new 
		}

		in_file.close();
	}

}
}//namespace
}//namespace ccma

#endif
