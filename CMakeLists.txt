cmake_minimum_required(VERSION 3.5.1)
project( testing )
find_package( Boost )
find_package( GSL )

# since 3.1
set (CMAKE_CXX_STANDARD 11)

include_directories( ${Boost_INCLUDE_DIRS})
include_directories( ${CMAKE_SOURCE_DIR}/include )
include_directories( ${GSL_INCLUDE_DIRS} )
LIST(APPEND SOURCES
  src/operator_overloading.cpp
  src/main.cpp)

add_executable( testing.out ${SOURCES} )
target_link_libraries( testing.out ${Boost_LIBRARIES} ${GSL_LIBRARIES} GSL::gsl GSL::gslcblas )


# # sets RPATH for *install*ed files
# set(CMAKE_INSTALL_RPATH "\$ORIGIN/lib") # Note CMake escaping aroung '$' sign.
# # *built* files will also used RPATH which has been set before
# set(CMAKE_BUILD_WITH_INSTALL_RPATH on)

# # Use 'find_library' calls. They return absolute paths needs for *link* stage.
# find_library(...)
# ...
# set(VFC_LIBS ...)
# # Create executable and link it using variables filled by 'find_library'
# add_executable(flowers ...)
# target_link_libraries(flowers ${VFC_LIBS})
