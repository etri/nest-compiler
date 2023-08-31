function(add_nestc_test)
    set(options NEWTON ZCU102 ONLY_ZCU102 MULTIEVTA)
    set(sh_cmd )
    set(oneValueArgs NAME)
    set(multiValueArgs COMMAND DEPENDS USE_SH PARAMS USE_DIFF DIFF_TARGET)
    cmake_parse_arguments(ARG "${options}" "${oneValueArgs}"
            "${multiValueArgs}" ${ARGN})

    if (NOT ARG_NAME)
        list(GET ARG_UNPARSED_ARGUMENTS 0 ARG_NAME)
        list(REMOVE_AT ARG_UNPARSED_ARGUMENTS 0)
    endif()

    if (NOT ARG_NAME)
        message(FATAL_ERROR "Name mandatory")
    endif()

    if (NOT ARG_COMMAND)
        set(ARG_COMMAND ${ARG_UNPARSED_ARGUMENTS})
    endif()

    if (NOT ARG_COMMAND)
        message(FATAL_ERROR "Command mandatory")
    endif()

    list(GET ARG_COMMAND 0 TEST_EXEC)
    list(APPEND ARG_DEPENDS ${TEST_EXEC})

    set_property(GLOBAL APPEND PROPERTY NESTC_TEST_DEPENDS ${ARG_DEPENDS})
    # Produce the specific test rule using the default built-in.
    if(NOT ARG_USE_SH)
        add_test(NAME ${ARG_NAME} COMMAND ${ARG_COMMAND})
    else()
        add_test(NAME ${ARG_NAME} COMMAND sh -c "${ARG_COMMAND} ${ARG_PARAMS}")
        if(ARG_USE_DIFF)
            add_test(NAME ${ARG_NAME}_compare COMMAND
                    diff output.bin ${ARG_DIFF_TARGET})
        endif()
    endif()

    if(NOT ARG_ONLY_ZCU102)
        set_property(TEST ${ARG_NAME}  PROPERTY LABELS NESTC)
    endif()
    if(ARG_USE_DIFF)
        if(NOT ARG_ONLY_ZCU102)
            set_property(TEST ${ARG_NAME}_compare  PROPERTY LABELS NESTC)
            set_property(TEST ${ARG_NAME}_compare  APPEND PROPERTY DEPENDS ${ARG_NAME})
        endif()
    endif()
    if(ARG_ZCU102 OR ARG_ONLY_ZCU102)
        set_property(TEST ${ARG_NAME} APPEND PROPERTY LABELS ZCU102)
        set_property(GLOBAL APPEND PROPERTY ZCU102_TEST_DEPENDS ${ARG_DEPENDS})
        if(ARG_USE_DIFF)
            set_property(TEST ${ARG_NAME}_compare  APPEND PROPERTY LABELS ZCU102)
            set_property(TEST ${ARG_NAME}_compare  APPEND PROPERTY DEPENDS ${ARG_NAME})
        endif()
    endif()
    if(ARG_NEWTON)
        set_property(TEST ${ARG_NAME} APPEND PROPERTY LABELS NEWTON)
        set_property(GLOBAL APPEND PROPERTY NEWTON_TEST_DEPENDS ${ARG_DEPENDS})
        if(ARG_USE_DIFF)
            set_property(TEST ${ARG_NAME}_compare  APPEND PROPERTY LABELS NEWTON)
            set_property(TEST ${ARG_NAME}_compare  APPEND PROPERTY DEPENDS ${ARG_NAME})
        endif()
    endif()
    if(ARG_MULTIEVTA)
        set_property(TEST ${ARG_NAME} APPEND PROPERTY LABELS MULTIEVTA)
        set_property(GLOBAL APPEND PROPERTY MULTIEVTA_TEST_DEPENDS ${ARG_DEPENDS})
    endif()
    if(ARG_ENLIGHT)
        set_property(TEST ${ARG_NAME} APPEND PROPERTY LABELS ENLIGHT)
        set_property(GLOBAL APPEND PROPERTY ENLIGHT_TEST_DEPENDS ${ARG_DEPENDS})
        if(ARG_USE_DIFF)
            set_property(TEST ${ARG_NAME}_compare  APPEND PROPERTY LABELS ENLIGHT)
            set_property(TEST ${ARG_NAME}_compare  APPEND PROPERTY DEPENDS ${ARG_NAME})
        endif()
    endif()
endfunction()
