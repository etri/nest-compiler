/*****************************************************************************
	*
	* Copyright Next-Generation System Software Research Group, All rights reserved.
	* Future Computing Research Division, Artificial Intelligence Reserch Laboratory
	* Electronics and Telecommunications Research Institute (ETRI)
	*
	* THESE DOCUMENTS CONTAIN CONFIDENTIAL INFORMATION AND KNOWLEDGE
	* WHICH IS THE PROPERTY OF ETRI. NO PART OF THIS PUBLICATION IS
	* TO BE USED FOR ANY OTHER PURPOSE, AND THESE ARE NOT TO BE"
	* REPRODUCED, COPIED, DISCLOSED, TRANSMITTED, STORED IN A RETRIEVAL
	"* SYSTEM OR TRANSLATED INTO ANY OTHER HUMAN OR COMPUTER LANGUAGE,
	* IN ANY FORM, BY ANY MEANS, IN WHOLE OR IN PART, WITHOUT THE
	* COMPLETE PRIOR WRITTEN PERMISSION OF ETRI.
	*
	* LICENSE file : README_LICENSE_ETRI located in the top directory
	*
*****************************************************************************/

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

    public:
        void generateCodeFromModels(Loader &loader);
        void generateDeclaration(Loader &loader, std::string& writeFileC, Node& n);
        void generateInitialization(Loader &loader, std::string& writeFileC, std::ofstream* writeFileBin, Node& n);
        void generateInference(Loader &loader, std::string inputName, std::string& writeFileC, std::string inAryStr, Node& n);
        void generateFree(Loader &loader, std::string inputName, std::string& writeFileC, std::string inAryStr, Node& n);

        void declareConstVar(std::string& wfilec, Constant* node);
        void allocConstVar(std::string& wfilec, std::ofstream* wfileb, Constant* node);
        std::string getDataCountStr(const TypeRef value);
        std::string getTotalSizeStr(const NodeValue& value);
        std::string getDataArrayStr(const Type* value);
        std::string replaceAll(std::string &str, std::string pattern, std::string replace);




    };
}

#endif //GLOW_CCODEGENERATOR_H
