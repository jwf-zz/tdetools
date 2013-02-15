# Find ANN

# This macro must define:
#  ANN_FOUND             < Conditional
#  ANN_INCLUDE_DIR       < Paths
#  ANN_LIBRARIES         < Paths as well

find_path(ANN_DIR
  NAMES  lib/libann.a
  HINTS  ${ANN_INCLUDE_DIR}
         ${ISISROOT}/3rdParty
)

find_PATH(ANN_INCLUDE_DIR
  NAMES  ANN/ANN.h
         ANN/ANNx.h
  HINTS  ${ANN_DIR}
  PATH_SUFFIXES inc include
)

# Deciding if ANN was found
set(ANN_INCLUDE_DIRS ${ANN_INCLUDE_DIR})

if(ANN_INCLUDE_DIR)
  set(ANN_FOUND TRUE)
else(ANN_INCLUDE_DIR)
  set(ANN_FOUND FALSE)
endif(ANN_INCLUDE_DIR)

FIND_LIBRARY(ANN_LIBRARY
  NAMES ann
  HINTS ${ANN_DIR}
  PATH_SUFFIXES lib lib64
)

set(ANN_LIBRARIES ${ANN_LIBRARY})
