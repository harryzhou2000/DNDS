-include Makefile.arc.in

ifneq (${NATIVE_ARCH},TH2B)
export OMPI_CXX=clang++
# export OMPI_CXX=g++
export MPICH_CXX=clang++
# export MPICH_CXX=g++
endif

first: what

CPC=mpicxx.openmpi
# CPC=mpicxx.mpich
# CPC=/home/harry/tools/openmpi-4.1.4/BUILD_GCC/bin/mpicxx
arch:
ifeq (${NATIVE_ARCH},TH2B)
	echo "Arch Type TH2B"
endif

ifeq (${NATIVE_ARCH},TH2B)
CPC=mpicxx
endif


# MPIINC=-I "C:\Program Files (x86)\Microsoft SDKs\MPI\Include"
# MPILIB=-L "C:\Program Files (x86)\Microsoft SDKs\MPI\Lib\x64" -lmsmpi
# CGNSINC=-I "E:\projects\tools\CGNS-4.2.0\build\include"
# CGNSLIB=-L "E:\projects\tools\CGNS-4.2.0\build\lib" -lcgns
# CPC=g++


MPIINC=
MPILIB=
CGNSINC=
CGNSLIB=

PYTHON_CFLAGS=$(/usr/bin/python3-config  --cflags)
PYTHON_LDFLAGS=$(/usr/bin/python3-config  --ldflags)


INCLUDE=${MPIINC} ${CGNSINC} ${PYTHON_CFLAGS}
LINK   =${MPILIB} ${CGNSLIB} ${PYTHON_LDFLAGS} -lmetis -llapacke -lopenblas

# for TH2B pp089
ifeq (${NATIVE_ARCH},TH2B)
INCLUDE=${MPIINC} ${CGNSINC} ${PYTHON_CFLAGS} -I/PARA/pp089/BIGDATA-2/apps/include
LINK   =${MPILIB} ${CGNSLIB} ${PYTHON_LDFLAGS} -L/PARA/pp089/BIGDATA-2/apps/lib -L/PARA/pp089/BIGDATA-2/apps/lib64 -lmetis -llapacke -lblas
endif

CXX_COMPILE_FLAGS=${INCLUDE} -std=c++17 -Wall \
 -Wno-comment -Wno-unused-variable -Wno-sign-compare -Wno-unused-but-set-variable -Wno-class-memaccess
CXX_LINK_FLAGS=${LINK}

SINGLE_TARGETS=test/mpitest.exe test/test.exe test/cgnstest.exe test/elemtest.exe\
 test/meshtest.exe test/staticReconstructionTest.exe\
 test/gmrestest.exe test/adtest.exe test/testGas.exe\
test/eikonal.exe test/staticReconstructionTestJR.exe test/euler.exe test/eulerSA.exe


PREBUILD=DNDS_Defines.o DNDS_Elements.o DNDS_MPI.o DNDS_FV_VR.o DNDS_FV_CR.o DNDS_FV_EulerEvaluator.o DNDS_Scripting.o
PREBUILD_DEP:=$(PREBUILD:.o=.d)

PREBUILD_FAST= DNDS_HardEigen.o DNDS_Mesh.o DNDS_Profiling.o
PREBUILD_FAST_DEP:=$(PREBUILD_FAST:.o=.d)

HEADERS=$(wildcard *.hpp *.h)



FLAGS=-g
# FLAGS=-Os
# FLAGS=-O2
# FLAGS=-Og -g
# FLAGS=-O3 
FLAGS=-O3 -DNINSERT
# FLAGS=-O3 -DNDEBUG  -DNINSERT


# FLAGS_FAST=-g
# FLAGS_FAST=-Os
# FLAGS_FAST=-g -O3
FLAGS_FAST=-O3
# FLAGS_FAST=-O3 -DNDEBUG


-include $(PREBUILD_FAST_DEP)
-include $(PREBUILD_DEP)

$(PREBUILD):%.o: %.cpp 
# mind that only first input is compiled for other dependencies are included files
# mind that -MMD instead of -MM to actually compile it
	$(CPC) $< -c -o $@ $(FLAGS) $(CXX_COMPILE_FLAGS) -MMD 

$(PREBUILD_FAST):%.o: %.cpp 
# mind that only first input is compiled for other dependencies are included files
# mind that -MMD instead of -MM to actually compile it
	$(CPC) $< -c -o $@ $(FLAGS_FAST)  $(CXX_COMPILE_FLAGS)  -MMD 

all: ${SINGLE_TARGETS}

what:
	echo `mpicxx.openmpi --showme`
	echo ${SINGLE_TARGETS}

.PRECIOUS: %.o ## don't rm the immediate .o s!!



VPATH:=test

$(SINGLE_TARGETS):%.exe: %.cpp ${HEADERS} ${PREBUILD} ${PREBUILD_FAST}
	${CPC} -o $@ $(filter %.cpp , $^) ${PREBUILD} ${PREBUILD_FAST} $(FLAGS) $(CXX_COMPILE_FLAGS) $(CXX_LINK_FLAGS)

test/testTensor.exe: test/testTensor.cpp SmallTensor.hpp
	g++ $< -o $@ -std=c++14


.PHONY: clean first arch

clean:
	rm -f *.exe *.o *.d
	rm -f test/*.exe