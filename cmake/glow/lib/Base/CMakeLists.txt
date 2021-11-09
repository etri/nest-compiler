add_library(Base
              ${NESTC_ROOT_DIR}/lib/Base/Tensor.cpp
              TensorSerialization.cpp
              ${NESTC_ROOT_DIR}/lib/Base/Type.cpp
              IO.cpp
              TaggedList.cpp)

target_include_directories(Base PUBLIC BEFORE ${NESTC_ROOT_DIR}/include/glow/Base/)

target_link_libraries(Base
                      PUBLIC
                        Support
                        glog::glog)

if(PNG_FOUND)
  target_sources(Base PRIVATE Image.cpp)
  target_include_directories(Base
                             PRIVATE
                               ${PNG_INCLUDE_DIR})
  target_link_libraries(Base
                        PRIVATE
                          ${PNG_LIBRARY})
  # Link z.lib if exists. For some targets (MSVC) it might not
  # be required because the PNG library already contains z.lib.
  find_library(Z_LIB z)
  if(Z_LIB_FOUND)
    target_link_libraries(Base
                          PRIVATE
                            ${Z_LIB})
  endif()
endif()

add_dependencies(Base AutoGen)
