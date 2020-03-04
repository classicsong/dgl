# MKL Module

################################################################################################
# Config mkl compilation.
# Usage:
#   dgl_config_mkl(<dgl_mkl_src>)
macro(dgl_config_mkl)
  if(NOT MKL_FOUND)
    message(FATAL_ERROR "Cannot find MKL.")
  endif()
  # always set the includedir when mkl is available
  # avoid global retrigger of cmake
	include_directories(${MKL_INCLUDE_DIR})

  add_definitions(-DDGL_USE_MKL)

  list(APPEND DGL_LINKER_LIBS ${MKL_LIBRARIES})
endmacro()