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
	* LICENSE file : LICENSE_ETRI located in the top directory
	*
*****************************************************************************/

#include "library.h"


vector<size_t> NCHW2NHWC = { 0u, 2u, 3u, 1u };
vector<size_t> NHWC2NCHW = { 0u, 3u, 1u, 2u };

#ifdef __APPLE__
dispatch_semaphore_t    semaphore;
#else
sem_t  semaphore;
#endif

#ifdef _DEBUG
int conv_idx = 0;
#endif

void read_vector(vector<size_t>* vec, ifstream* readFileBin)
{
//	cout << "read_vector" << endl;
  int len = 0;
  readFileBin->read((char *)&len, sizeof(int));
  for(int i = 0; i < len; i++) {
    size_t val;
    readFileBin->read((char *)&val, sizeof(size_t));
    vec->push_back(val);
  }
}