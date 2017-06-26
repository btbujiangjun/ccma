/*********************************************
* Author: Jun Jiang - jiangjun4@sina.com
* Created: 2017-06-01 11:22
* Last modified: 2017-06-01 11:22
* Filename: TestRNN.cpp
* Description: 
**********************************************/

#include "utils/RNNHelper.h"

int main(int argc, char** argv){
	std::map<std::string, uint> map;
	ccma::utils::RNNHelper helper(8000);
	helper.read_word2index("data/word_to_index", map);
}
