add_library(Partitioner
              ${NESTC_ROOT_DIR}/lib/Partitioner/PartitionerBase.cpp
              ${NESTC_ROOT_DIR}/lib/Partitioner/PartitionerUtils.cpp
              PartitionerOptimizer.cpp
              PartitionerValidation.cpp
              Partitioner.cpp
              ${NESTC_ROOT_DIR}/lib/Partitioner/NestPartitionerSchedule.cpp
              ${NESTC_ROOT_DIR}/lib/Partitioner/NestPartitioner.cpp)

target_link_libraries(Partitioner
                      PRIVATE
                        Backends
                        Flags
                        Graph
                        GraphOptimizer)

add_library(PartitionerNestos
              ${NESTC_ROOT_DIR}/lib/Partitioner/PartitionerBase.cpp
              ${NESTC_ROOT_DIR}/lib/Partitioner/PartitionerUtils.cpp
              PartitionerOptimizer.cpp
              PartitionerValidation.cpp
              Partitioner.cpp
              # ${NESTC_ROOT_DIR}/lib/Partitioner/NestPartitionerSchedule.cpp
              ${NESTC_ROOT_DIR}/lib/Partitioner/OsPartitionerSchedule.cpp
              ${NESTC_ROOT_DIR}/lib/Partitioner/NestPartitioner.cpp)

target_link_libraries(PartitionerNestos
                      PRIVATE
                        Backends
                        Flags
                        Graph
                        GraphOptimizer)
