cmake_minimum_required(VERSION 3.5.1)

project(testing LANGUAGES CUDA CXX)

if(MSVC)
  add_compile_options(/W3 /WX)
else()
  #add_compile_options(-W -Wall -Werror -fno-omit-frame-pointer)
  #add_compile_options(-W -Wall)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -W -Wall -Werror -fno-omit-frame-pointer")
endif()
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CUDA_STANDARD 11)

#list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake_modules/tensorflow-cmake")

# tensorflow dependency
# find_package( TensorFlow REQUIRED )
find_package( Eigen3 REQUIRED )
# find_package( Protobuf REQUIRED )

find_package( Eigen3 3.3 REQUIRED NO_MODULE )
# find_package( Vc 1.4 REQUIRED )
find_package( GTest REQUIRED )
find_package( benchmark REQUIRED )
find_package( OpenMP REQUIRED )

# use -DCMAKE_MODULE_PATH to help cmake locate cmake modules (FindXXXX)
find_package( SFML 2 REQUIRED network audio graphics window system)

# find_package( ortools CONFIG REQUIRED )

#add_subdirectory(abseil-cpp)

add_executable ( testing.out test.cpp )
target_link_libraries ( testing.out PRIVATE Eigen3::Eigen ${OpenMP_CXX_FLAGS} benchmark::benchmark ${Vc_LIBRARIES}) # ortools::ortools  )
target_compile_options ( testing.out PRIVATE ${OpenMP_CXX_FLAGS} ${Vc_ALL_FLAGS} "-O3" )
target_include_directories ( testing.out PRIVATE ${Vc_INCLUDE_DIR} )

#message ( STATUS ${Vc_LIB_DIR} )
#message ( STATUS ${Vc_INCLUDE_DIR} )
# message ( STATUS ${Vc_DEFINITIONS} )
# message ( STATUS ${Vc_COMPILER_FLAGS} )
# message ( STATUS ${Vc_ARCHITECTURE_FLAGS} )
# message ( STATUS ${Vc_ALL_FLnnAGS} )

# add_executable ( testing.out test.cpp )
# target_include_directories(testing.out PRIVATE
#   ${Protobuf_INCLUDE_DIR} ${EIGEN3_INCLUDE_DIR} ${TensorFlow_INCLUDE_DIR} "${CMAKE_CURRENT_LIST_DIR}/include")
# target_link_libraries(testing.out PRIVATE
#   ${TensorFlow_LIBRARY} ${GTEST_LIBRARIES} ${GTEST_MAIN_LIBRARIES})

# add_executable(hw.out hw.cpp)
# target_include_directories(hw.out PRIVATE ${GTEST_INCLUDE_DIR})
# target_link_libraries(hw.out PRIVATE ${GTEST_LIBRARIES} ${GTEST_MAIN_LIBRARIES} benchmark::benchmark)

# add_executable(hw2.out hw2.cpp)
# target_include_directories(hw2.out PRIVATE ${GTEST_INCLUDE_DIR})
# target_link_libraries(hw2.out PRIVATE ${GTEST_LIBRARIES} benchmark::benchmark) #${GTEST_MAIN_LIBRARIES})

add_executable(draw_window.out draw_window.cpp)
target_link_libraries(draw_window.out PRIVATE ${SFML_LIBRARIES} ${SFML_DEPENDENCIES})

add_executable(test_scan.out test/test_scan.cu)
target_link_libraries(test_scan.out PRIVATE ${GTEST_LIBRARIES} ${GTEST_MAIN_LIBRARIES} benchmark::benchmark)
target_include_directories(test_scan.out PRIVATE cuda)
#enable_language(C ASM_NASM)
#set_source_files_properties(src/avx.asm PROPERTIES COMPILE_FLAGS "-x assembler-with-cpp")

#add_executable(avx.out src/AVX.cpp src/avx.asm)
#add_executable(avx.out src/AVX.cpp)
























# project( testing )
# find_package( Boost )
# find_package( GSL )
# find_package( OpenMP )
# #find_package( OpenCV )

# # since 3.1
# set (CMAKE_CXX_STANDARD 11)

# if (OPENMP_FOUND)
#     set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
#     set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
#     set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
# endif()


# include_directories( ${Boost_INCLUDE_DIRS})
# include_directories( ${CMAKE_SOURCE_DIR}/include )
# include_directories( ${GSL_INCLUDE_DIRS} )
# include_directories( ${OpenCV_INCLUDE_DIRS} )

# list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}")

# add_executable( testing.out test.cpp )
# #LIST(APPEND SOURCES
# #  src/operator_overloading.cpp)

# #add_executable( testing.out ${SOURCES} )
# #target_link_libraries( testing.out ${Boost_LIBRARIES} ${GSL_LIBRARIES} GSL::gsl GSL::gslcblas )

# add_executable( testing.out src/main.cpp)

# #add_executable( testing.out src/load_data.cpp )
# #target_link_libraries( testing.out ${OpenCV_LIBS} )

# # # sets RPATH for *install*ed files
# # set(CMAKE_INSTALL_RPATH "\$ORIGIN/lib") # Note CMake escaping aroung '$' sign.
# # # *built* files will also used RPATH which has been set before
# # set(CMAKE_BUILD_WITH_INSTALL_RPATH on)

# # # Use 'find_library' calls. They return absolute paths needs for *link* stage.
# # find_library(...)
# # ...
# # set(VFC_LIBS ...)
# # # Create executable and link it using variables filled by 'find_library'
# # add_executable(flowers ...)
# # target_link_libraries(flowers ${VFC_LIBS})
