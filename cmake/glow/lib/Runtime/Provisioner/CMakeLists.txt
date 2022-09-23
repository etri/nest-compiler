add_library(Provisioner
              ${NESTC_ROOT_DIR}/lib/Runtime/Provisioner/Provisioner.cpp)

target_link_libraries(Provisioner
                      PRIVATE
                        Backend
                        Backends
                        Flags
                        Graph
                        Runtime)
