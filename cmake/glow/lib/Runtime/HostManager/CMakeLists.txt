add_library(HostManager
        ${NESTC_ROOT_DIR}/lib/Runtime/HostManager/HostManager.cpp)

target_link_libraries(HostManager
                      PRIVATE
                        Backends
                        Base
                        Executor
                        Exporter
                        Importer
                        Graph
                        GraphOptimizer
                        Partitioner
                        Provisioner
                        Runtime)
