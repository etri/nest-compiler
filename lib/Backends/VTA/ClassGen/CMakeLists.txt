set(VTA_VERIF_FILENAME VTASpecificNodesVerification.h)
configure_file(${VTA_VERIF_FILENAME}
        ${GLOW_BINARY_DIR}/glow/${VTA_VERIF_FILENAME} COPYONLY)

set(VERIF_FILENAME VTASpecificInstrsVerification.h)
configure_file(${VERIF_FILENAME}
        ${GLOW_BINARY_DIR}/glow/${VERIF_FILENAME} COPYONLY)
