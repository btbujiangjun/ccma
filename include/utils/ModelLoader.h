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
    char type;
    uint rows;
    uint cols;
};//class ModelInfo

class ModelLoader{
public:
    template<class T>
    bool write(std::vector<ccma::algebra::BaseMatrixT<T>*> models,
               const std::string& path,
               bool is_append = false,
               const std::string& signature = "");
    template<class T>
    bool write(ccma::algebra::BaseMatrixT<T>* model,
               const std::string& path,
               bool is_append = false,
               const std::string& signature = "");
    template<class T>
    bool read(const std::string& path,
              std::vector<ccma::algebra::BaseMatrixT<T>*>* models,
              const std::string& signature = "");
private:
    template<class T>
    bool generate_header(std::vector<ccma::algebra::BaseMatrixT<T>*> models,
                         std::vector<ModelInfo>* infos);
};//class ModelLoader

template<class T>
bool ModelLoader::write(std::vector<ccma::algebra::BaseMatrixT<T>*> models,
                        const std::string& path,
                        bool is_append,
                        const std::string& signature){
    std::vector<ccma::algebra::BaseMatrixT<T>*> old_models;
    if(is_append){
        read<T>(path, &old_models, signature);
    }
    
    uint num_models = models.size() + old_models.size();
    std::vector<ModelInfo> infos;

    if(!generate_header(old_models, &infos) || !generate_header(models, &infos)){
        return false;
    }
     
    std::ofstream out_file(path, std::ios::binary);

    out_file.write(signature.c_str(), sizeof(char)*signature.size());
    out_file.write((char*)&num_models, sizeof(uint));
    for(auto&& info : infos){
        out_file.write(&info.type, sizeof(char));
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

    return true;
}
template<class T>
bool ModelLoader::write(ccma::algebra::BaseMatrixT<T>* model,
                        const std::string& path,
                        bool is_append,
                        const std::string& signature){
    std::vector<ccma::algebra::BaseMatrixT<T>*> models;
    models.push_back(model);
    return write(models, path, is_append, signature);
}

template<class T>
bool ModelLoader::read(const std::string& path,
                       std::vector<ccma::algebra::BaseMatrixT<T>*>* models,
                       const std::string& signature){
    std::ifstream in_file(path, std::ios::binary);
    if(!in_file){
        printf("Can't open Filename:%s\n", path.c_str());
        return false;
    }

    uint num_models = 0;
    std::string read_signature(signature.size(), ' ');
    in_file.read(&read_signature[0], sizeof(char) * signature.size());
    if(read_signature != signature){
        printf("ModelLoader read model error:[signature error]\n");
        in_file.close();
        return false;
    }

    in_file.read((char*)(&num_models), sizeof(uint));

    std::vector<ModelInfo> infos;
    for(uint i = 0; i != num_models; i++){
        ModelInfo info;
        in_file.read(&info.type, sizeof(char));
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
bool ModelLoader::generate_header(std::vector<ccma::algebra::BaseMatrixT<T>*> models,
                                  std::vector<ModelInfo>* infos){
    for(auto&& model : models){
        ModelInfo info;
        info.rows = model->get_rows();
        info.cols = model->get_cols();
        if(typeid(T) == typeid(int)){
            info.type = 'i';
        }else if(typeid(T) == typeid(float)){
            info.type = 'f';
        }else if(typeid(T) == typeid(double)){
            info.type = 'd';
        }else{
            printf("ModelLoader not support data type:[%s]\n", typeid(T).name());
            return false;
        }
        infos->push_back(info);
    }
    return true;
}

}//namespace utils
}//namespace ccma

#endif
