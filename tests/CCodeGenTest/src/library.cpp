/*
 * library_def.cpp
 *
 *  Created on: 2019. 9. 3.
 *      Author: etri
 */

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