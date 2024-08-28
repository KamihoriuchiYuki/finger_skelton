#----------------------------------------------------------------
# Generated CMake target import file.
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "ade" for configuration ""
set_property(TARGET ade APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(ade PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_NOCONFIG "CXX"
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib/libade.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS ade )
list(APPEND _IMPORT_CHECK_FILES_FOR_ade "${_IMPORT_PREFIX}/lib/libade.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
