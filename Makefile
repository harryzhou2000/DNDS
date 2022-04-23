
# MPIINC=-I "C:\Program Files (x86)\Microsoft SDKs\MPI\Include"
# MPILIB=-L "C:\Program Files (x86)\Microsoft SDKs\MPI\Lib\x64" -lmsmpi
# CGNSINC=-I "E:\projects\tools\CGNS-4.2.0\build\include"
# CGNSLIB=-L "E:\projects\tools\CGNS-4.2.0\build\lib" -lcgns
# CPC=g++


MPIINC=
MPILIB=
CGNSINC=
CGNSLIB=-lcgns
CPC=mpicxx

INCLUDE=${MPIINC} ${CGNSINC}
LINK   =${MPILIB} ${CGNSLIB} -lmetis

SINGLE_TARGETS=test/mpitest.exe test/test.exe test/cgnstest.exe test/elemtest.exe test/meshtest.exe test/staticReconstructionTest.exe\
test/eikonal.exe


PREBUILD=DNDS_Defines.o DNDS_Elements.o DNDS_MPI.o
PREBUILD_DEP:=$(PREBUILD:.o=.d)

PREBUILD_FAST=DNDS_Mesh.o DNDS_HardEigen.o DNDS_FV_VR.o DNDS_FV_CR.o
PREBUILD_FAST_DEP:=$(PREBUILD_FAST:.o=.d)

HEADERS=$(wildcard *.hpp *.h)

FLAGS=-g
# FLAGS=-O2
FLAGS=-O3 -DNDEBUG

FLAGS_FAST=-g
FLAGS_FAST=-O3 -DNDEBUG


-include $(PREBUILD_FAST_DEP)
-include $(PREBUILD_DEP)

$(PREBUILD):%.o: %.cpp 
# mind that only first input is compiled for other dependencies are included files
# mind that -MMD instead of -MM to actually compile it
	$(CPC) $< -c -o $@ $(FLAGS) -MMD 

$(PREBUILD_FAST):%.o: %.cpp 
# mind that only first input is compiled for other dependencies are included files
# mind that -MMD instead of -MM to actually compile it
	$(CPC) $< -c -o $@ $(FLAGS_FAST)  $(FPFLAGS) -MMD 

.PRECIOUS: %.o ## don't rm the immediate .o s!!


all: ${SINGLE_TARGETS}

VPATH:=test
%.exe: %.cpp ${HEADERS} ${PREBUILD} ${PREBUILD_FAST}
	${CPC} -o $@ $(filter %.cpp , $^) ${PREBUILD} ${PREBUILD_FAST}  ${INCLUDE} ${LINK} ${FLAGS}


.PHONY: clean

clean:
	rm -f *.exe *.o *.d
	rm -f test/*.exe