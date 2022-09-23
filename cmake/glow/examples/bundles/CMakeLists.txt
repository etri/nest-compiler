add_custom_command(
        OUTPUT
        ${NESTC_BINARY_DIR}/models/mxnet_exported_resnet18.onnx
        COMMAND
        mkdir -p ${NESTC_BINARY_DIR}/models
        COMMAND
        aws s3 cp s3://nestc-data-pub/VTABundleTests/Resnet18/mxnet_exported_resnet18.onnx ${NESTC_BINARY_DIR}/models/mxnet_exported_resnet18.onnx --no-sign-request
)

add_custom_target(MxnetResnet18ONNXDownload
        DEPENDS
        ${NESTC_BINARY_DIR}/models/mxnet_exported_resnet18.onnx
        )

if (GLOW_WITH_BUNDLES)
  add_subdirectory(lenet_mnist)
  add_subdirectory(resnet50)
  add_subdirectory(bundle_with_multiple_entries)
  add_subdirectory(bundle_with_extra_objects)
  add_subdirectory(bundle_instrument)
  add_subdirectory(bundle_tflite_custom)
  FILE(GLOB subdirs RELATIVE ${NESTC_ROOT_DIR}/examples/bundles/ ${NESTC_ROOT_DIR}/examples/bundles/*)
  foreach(object ${subdirs})
    if(IS_DIRECTORY ${NESTC_ROOT_DIR}/examples/bundles/${object})
      message(STATUS "Add a bundle example, ${object}")
      add_subdirectory(${NESTC_ROOT_DIR}/examples/bundles/${object} ${NESTC_BINARY_DIR}/examples/bundles/${object})
    endif()
  endforeach()
endif()
