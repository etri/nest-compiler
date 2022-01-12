INCLUDE_DIRECTORIES(include ../vtalib/3rdparty/dlpack/include)
INCLUDE_DIRECTORIES(../vtalib/3rdparty/dmlc-core/include ${vtalib_SOURCE_DIR}/3rdparty/dlpack/include)

if(NESTC_USE_VTASIM)
    INCLUDE_DIRECTORIES(include ../vtalib/include/simulator)
    if(NESTC_USE_PRECOMPILED_EVTA_LIBRARY)
        set(BUNDLE_LINK_LIB vta_runtime png LLVMSupport pthread glog gflags
                Base
                CodeGen
                Graph
                IR
                LLVMCore
                IROptimizer
                GraphOptimizerPipeline
                QuantizationBase
                Runtime)
    else()
        set(BUNDLE_LINK_LIB png)
    endif()
    set(BUNDLE_DEPENDS VTABundle)
else()
    INCLUDE_DIRECTORIES(include ../vtalib/include/zcu102)
    if(NESTC_USE_PRECOMPILED_EVTA_LIBRARY)
        set(BUNDLE_LINK_LIB vta_runtime png LLVMSupport pthread glog gflags
                vta_runtime arm_compute arm_compute_core arm_compute_graph gomp
                cma)
    else()
        set(BUNDLE_LINK_LIB png)
    endif()
    set(BUNDLE_DEPENDS VTABundle)
endif()


INCLUDE_DIRECTORIES(include ../vtalib/include/Bundle)
FILE(GLOB subdirs RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/*)
foreach(object ${subdirs})
    if(IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${object})
        message(STATUS "Adding ${object} Bundles.")
        add_subdirectory(${object})
    endif()
endforeach()