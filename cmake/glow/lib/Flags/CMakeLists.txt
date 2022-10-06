add_library(Flags
              ${NESTC_ROOT_DIR}/lib/Flags/Flags.cpp)
target_link_libraries(Flags
                      PUBLIC
                        gflags
                        glog::glog
                        LLVMSupport)
