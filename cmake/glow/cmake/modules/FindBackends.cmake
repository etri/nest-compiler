set(GLOW_BACKENDS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/lib/Backends")
file(GLOB subdirs RELATIVE "${GLOW_BACKENDS_DIR}" "${GLOW_BACKENDS_DIR}/*")
foreach(object ${subdirs})
  if(IS_DIRECTORY "${GLOW_BACKENDS_DIR}/${object}")
    set(backendEnabled "GLOW_WITH_${object}")
    string(TOUPPER "${backendEnabled}" backendEnabled)
    if(NOT DEFINED ${backendEnabled} OR ${backendEnabled})
      list(APPEND GLOW_BACKENDS "${object}")
    endif()
  endif()
endforeach()

set(NEST_BACKENDS_DIR "${NESTC_ROOT_DIR}/lib/Backends")
file(GLOB subdirs RELATIVE "${NEST_BACKENDS_DIR}" "${NEST_BACKENDS_DIR}/*")
foreach(object ${subdirs})
  message(find_backend)
  message(${object})
  if(IS_DIRECTORY "${NEST_BACKENDS_DIR}/${object}")
    set(backendEnabled NESTC_WITH_${object})
    string(TOUPPER "${backendEnabled}" backendEnabled)
    if(NOT DEFINED ${backendEnabled} OR ${backendEnabled})
      list(APPEND GLOW_BACKENDS "${object}")
    endif()
  endif()
endforeach()