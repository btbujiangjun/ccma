/***********************************************
 * Author: Jun Jiang - jiangjun4@sina.com
 * Create: 2017-07-07 11:10
 * Last modified : 2017-07-07 11:10
 * Filename      : ModelLoader.h
 * Description   : 
 **********************************************/

#ifndef _CCMA_UTILS_MODELLOADER_H_
#define _CCMA_UTILS_MODELLOADER_H_

#include <fstream>
#include <vector>
#include "algebra/BaseMatrix.h"

namespace ccma{
namespace utils{

class ModelInfo{
public:
    uint rows;
    uint cols;
};//class ModelInfo

class ModelLoader{
public:
	template<class T>
    void write(std::vector<ccma::algebra::BaseMatrixT<T>*> models,
               const std::string& path,
               bool is_append = false);
	template<class T>
    void write(ccma::algebra::BaseMatrixT<T>* model,
               const std::string& path,
               bool is_append = false);
	template<class T>
    bool read(const std::string& path,
              std::vector<ccma::algebra::BaseMatrixT<T>*>* models);
private:
	template<class T>
    void generate_header(std::vector<ccma::algebra::BaseMatrixT<T>*> models,
                         std::vector<ModelInfo>* infos);
};//class ModelLoader

template<class T>
void ModelLoader::write(std::vector<ccma::algebra::BaseMatrixT<T>*> models,
                        const std::string& path,
                        bool is_append){
    std::vector<ccma::algebra::BaseMatrixT<T>*> old_models;
    if(is_append){
        read(path, &old_models);
    }
    
    uint num_models = models.size() + old_models.size();
    std::vector<ModelInfo> infos;

    generate_header(old_models, &infos);
    generate_header(models, &infos);
     
    std::ofstream out_file(path, std::ios::binary);

    out_file.write((char*)&num_models, sizeof(uint));
    for(auto&& info : infos){
        out_file.write((char*)&info.rows, sizeof(uint));
        out_file.write((char*)&info.cols, sizeof(uint));
    }

    uint num_old_models = old_models.size();
    for(uint i = 0; i != num_old_models; i++){
        out_file.write((char*)old_models[i]->get_data(), sizeof(T) * infos[i].rows * infos[i].cols);
    }

    num_models = models.size();
    for(uint i = 0; i != num_models; i++){
        ModelInfo info = infos[i+num_old_models];
        out_file.write((char*)models[i]->get_data(), sizeof(T) * info.rows * info.cols);
    }

    out_file.close();

}
template<class T>
void ModelLoader::write(ccma::algebra::BaseMatrixT<T>* model,
                        const std::string& path,
                        bool is_append){
    std::vector<ccma::algebra::BaseMatrixT<T>*> models;
    models.push_back(model);
    write(models, path, is_append);
}

template<class T>
bool ModelLoader::read(const std::string& path,
                       std::vector<ccma::algebra::BaseMatrixT<T>*>* models){
    std::ifstream in_file(path, std::ios::binary);
    if(!in_file){
        printf("Can't open Filename:%s\n", path.c_str());
        return false;
    }

    uint num_models = 9;
    in_file.read((char*)(&num_models), sizeof(uint));

    std::vector<ModelInfo> infos;
    for(uint i = 0; i != num_models; i++){
        ModelInfo info;
        in_file.read((char*)(&info.rows), sizeof(uint));
        in_file.read((char*)(&info.cols), sizeof(uint));
        infos.push_back(info);
    }

    models->clear();
    for(uint i = 0; i != num_models; i++){
        ModelInfo info = infos[i];
        uint size      = info.rows * info.cols;
        
        T* data = new T[size];
        in_file.read((char*)data, sizeof(T) * size);

        auto mat = new ccma::algebra::DenseMatrixT<T>();
        mat->set_shallow_data(data, info.rows, info.cols);
        models->push_back(mat);
    }

    in_file.close();
    return true;
}

template<class T>
void ModelLoader::generate_header(std::vector<ccma::algebra::BaseMatrixT<T>*> models,
                                     std::vector<ModelInfo>* infos){
    for(auto&& model : models){
        ModelInfo info;
        info.rows = model->get_rows();
        info.cols = model->get_cols();
        infos->push_back(info);
    }
}

}//namespace utils
}//namespace ccma

#endif
