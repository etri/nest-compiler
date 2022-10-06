set(VERIF_FILENAME NMPSpecificNodesVerification.h)
configure_file(${VERIF_FILENAME}
               ${GLOW_BINARY_DIR}/glow/${VERIF_FILENAME} COPYONLY)

set(VERIF_FILENAME NMPSpecificInstrsVerification.h)
configure_file(${VERIF_FILENAME}
               ${GLOW_BINARY_DIR}/glow/${VERIF_FILENAME} COPYONLY)
