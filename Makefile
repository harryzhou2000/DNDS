# export OMPI_CXX=clang++
export OMPI_CXX=g++

first: what


CPC=mpicxx
# MPIINC=-I "C:\Program Files (x86)\Microsoft SDKs\MPI\Include"
# MPILIB=-L "C:\Program Files (x86)\Microsoft SDKs\MPI\Lib\x64" -lmsmpi
# CGNSINC=-I "E:\projects\tools\CGNS-4.2.0\build\include"
# CGNSLIB=-L "E:\projects\tools\CGNS-4.2.0\build\lib" -lcgns
# CPC=g++


MPIINC=
MPILIB=
CGNSINC=
CGNSLIB=-lcgns

PYTHON_CFLAGS=$(/usr/bin/python3-config  --cflags)
PYTHON_LDFLAGS=$(/usr/bin/python3-config  --ldflags)


INCLUDE=${MPIINC} ${CGNSINC} ${PYTHON_CFLAGS}
LINK   =${MPILIB} ${CGNSLIB} ${PYTHON_LDFLAGS} -lmetis -llapacke -lopenblas

CXX_COMPILE_FLAGS=${INCLUDE} -std=c++14 -Wall -Wno-comment -Wno-unused-variable -Wno-sign-compare -Wno-unused-but-set-variable
CXX_LINK_FLAGS=${LINK}

SINGLE_TARGETS=test/mpitest.exe test/test.exe test/cgnstest.exe test/elemtest.exe test/meshtest.exe test/staticReconstructionTest.exe\
test/eikonal.exe


PREBUILD=DNDS_Defines.o DNDS_Elements.o DNDS_MPI.o DNDS_FV_VR.o DNDS_FV_CR.o
PREBUILD_DEP:=$(PREBUILD:.o=.d)

PREBUILD_FAST=DNDS_Mesh.o DNDS_HardEigen.o
PREBUILD_FAST_DEP:=$(PREBUILD_FAST:.o=.d)

HEADERS=$(wildcard *.hpp *.h)



# FLAGS=-g
# FLAGS=-Os
# FLAGS=-O2
# FLAGS=-g -O2

FLAGS=-O3
# FLAGS=-O3 -DNDEBUG


# FLAGS_FAST=-g
# FLAGS_FAST=-Os
# FLAGS_FAST=-g -O3
FLAGS_FAST=-O3 -DNDEBUG


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
	echo `mpicxx --showme`
	echo ${SINGLE_TARGETS}

.PRECIOUS: %.o ## don't rm the immediate .o s!!



VPATH:=test
%.exe: %.cpp ${HEADERS} ${PREBUILD} ${PREBUILD_FAST}
	${CPC} -o $@ $(filter %.cpp , $^) ${PREBUILD} ${PREBUILD_FAST} $(FLAGS) $(CXX_COMPILE_FLAGS) $(CXX_LINK_FLAGS)

.PHONY: clean first

clean:
	rm -f *.exe *.o *.d
	rm -f test/*.exe