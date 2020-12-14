//
// Created by etri on 20. 1. 16..
//

#ifndef GLOW_CCODEGENERATOR_H
#define GLOW_CCODEGENERATOR_H


#include "Loader.h"
#include "glow/Graph/Graph.h"

#include <glog/logging.h>

namespace glow {
    class CCodeGenerator {

    private:
      Node* firstNode = nullptr;
      Node* lastNode = nullptr;
    protected:
        std::set<std::string> varSet_;
        std::set<std::string> allocVarList_;
        std::set<std::string> freeVarList_;
        //Loader* loader_;

    public:
        //CCodeGenerator(Loader& loader) {loader_ = &loader;}
        void generateCodeFromModels(Loader &loader);
        void generateDeclaration(Loader &loader, std::string& writeFileC, Node& n);
        void generateInitialization(Loader &loader, std::string& writeFileC, std::ofstream* writeFileBin, Node& n);
        void generateInference(Loader &loader, std::string inputName, std::string& writeFileC, std::string inAryStr, Node& n);
        void generateFree(Loader &loader, std::string inputName, std::string& writeFileC, std::string inAryStr, Node& n);
//        void generateDeclaration(Loader &loader, std::ofstream* writeFileC, Node& n);
//        void generateInitialization(Loader &loader, std::ofstream* writeFileC, std::ofstream* writeFileBin, Node& n);
//        void generateInference(Loader &loader, std::ofstream* writeFileC, std::string inAryStr, Node& n);
//        void generateFree(Loader &loader, std::ofstream* writeFileC, std::string inAryStr, Node& n);

        void declareConstVar(std::string& wfilec, Constant* node);
        void allocConstVar(std::string& wfilec, std::ofstream* wfileb, Constant* node);
        std::string getDataCountStr(const TypeRef value);
        std::string getTotalSizeStr(const NodeValue& value);
        std::string getDataArrayStr(const Type* value);
        std::string replaceAll(std::string &str, std::string pattern, std::string replace);




    };
}

#endif //GLOW_CCODEGENERATOR_H
