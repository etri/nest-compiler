add_library(Support
              Debug.cpp
              Error.cpp
              Random.cpp
              ${NESTC_ROOT_DIR}/lib/Support/Support.cpp
              ThreadPool.cpp
              ZipUtils.cpp)
target_link_libraries(Support
                      PUBLIC
                        Flags
                        LLVMSupport
                        Miniz
                        glog::glog
                        folly_jemalloc
                        )

add_subdirectory(TensorPool)
