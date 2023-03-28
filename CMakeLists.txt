cmake_minimum_required(VERSION 3.1)

# manual perferences
set(CMAKE_VERBOSE_MAKEFILE ON)
set(DNDS_SUPRESS_INSERT_CHECK ON CACHE BOOL "close the inserted outputs")
set(DNDS_USE_NDEBUG_MACRO OFF CACHE BOOL "use DEBUG macro for eigen output and cassert etc") #performance impact is small so off by default(without -DNDEBUG)

if(UNIX OR MINGW)
    set(CMAKE_CXX_COMPILER mpicxx CACHE FILEPATH "compiler with your MPI wrapping")
endif()

set(CMAKE_INSTALL_PREFIX build/bin CACHE FILEPATH "install place")

set(CMAKE_BUILD_TYPE Release CACHE STRING "Release or Debug")
set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    "Release" "Debug" "RelWithDebInfo" "MinSizeRel")

# set(MPIEXEC_EXECUTABLE "mpiexec.mpich" CACHE FILEPATH "exe path to mpi")
# set(MPI_C_COMPILER "mpicc.mpich" CACHE FILEPATH "mpicc")
# set(MPI_CXX_COMPILER "mpicxx.mpich" CACHE FILEPATH "mpicxx")
# ###########################################################################################
project(DNDS LANGUAGES CXX)

# ###########################################################################################
if(MSVC)
    # # MPI
    find_package(MPI REQUIRED)

    if(${MPI_FOUND})
        message(STATUS "MPI Found: ${MPI_CXX_HEADER_DIR}")
    else()
        message(SEND_ERROR "MPI Not Found")
    endif()
endif()

# set(CMAKE_CXX_COMPILER ${MPI_CXX_COMPILER} CACHE FILEPATH "cxx")
# set(CMAKE_CXX_COMPILER ${MPI_CXX_COMPILER})
#
#
#
#
# ###############
# Determine directories
# ###############
# # libraries
find_library(EXTERNAL_LIB_METIS metis PATHS .)
find_library(EXTERNAL_LIB_BLAS NAMES openblas blas)
find_library(EXTERNAL_LIB_LAPACKE lapacke)
set(EXTERNAL_LIBS \\
    ${EXTERNAL_LIB_METIS}\\
    ${EXTERNAL_LIB_BLAS}\\
    ${EXTERNAL_LIB_LAPACKE})
message(STATUS "External libs ${EXTERNAL_LIBS}")

if(MSVC)
    set(EXTERNAL_LIBS ${EXTERNAL_LIBS}\\
        ${MPI_LIBRARIES}\\
    )
endif()

link_libraries(${EXTERNAL_LIBS})

# # includes
set(EXTERNAL_INCLUDES \\
    "")

if(MSVC)
    set(EXTERNAL_INCLUDES ${EXTERNAL_INCLUDES}\\
        ${MPI_INCLUDE_PATH}\\
    )
endif()

include_directories(${EXTERNAL_INCLUDES})

aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR} ROOT_SOURCE)

# ###############
# Compiling Flags
# ###############
if(UNIX OR MINGW)
    set(COMPILE_FLAG_BASIC -std=c++17 -Wall)

    set(COMPILE_FLAG_ADDITIONAL "")

    if(DNDS_SUPRESS_INSERT_CHECK)
        set(COMPILE_FLAG_ADDITIONAL ${COMPILE_FLAG_ADDITIONAL} -DNINSERT)
    endif()

    if(NOT DNDS_USE_NDEBUG_MACRO)
        set(COMPILE_FLAG_ADDITIONAL ${COMPILE_FLAG_ADDITIONAL} -UNDEBUG)
    endif()
else()
    message(FATAL_ERROR "NOT YET IMPLEMENTED HERE")
endif()

set(CXX_WARNING_IGNORE -Wno-comment -Wno-unused-variable -Wno-sign-compare -Wno-unused-but-set-variable -Wno-class-memaccess)

# ###############
# Actual Targets
# ###############
add_library(ROOT_OBJ_LIBRARY STATIC ${ROOT_SOURCE})
target_compile_options(ROOT_OBJ_LIBRARY PUBLIC ${COMPILE_FLAG_ADDITIONAL} ${COMPILE_FLAG_BASIC})
target_compile_options(ROOT_OBJ_LIBRARY PUBLIC ${CXX_WARNING_IGNORE})

set(DNDS_PRIMARY_EXES euler2D euler eulerSA)

foreach(EXENAME ${DNDS_PRIMARY_EXES})
    add_executable(${EXENAME} test/${EXENAME})
    target_link_libraries(${EXENAME} ROOT_OBJ_LIBRARY)

    # target_compile_features(${EXENAME} PUBLIC cxx_std_17)
    target_compile_options(${EXENAME} PUBLIC ${COMPILE_FLAG_ADDITIONAL} ${COMPILE_FLAG_BASIC})
    target_compile_options(${EXENAME} PUBLIC ${CXX_WARNING_IGNORE})
endforeach(EXENAME ${DNDS_PRIMARY_EXES})
