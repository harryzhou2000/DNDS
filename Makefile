
MPIINC=-I "C:\Program Files (x86)\Microsoft SDKs\MPI\Include"
MPILIB=-L "C:\Program Files (x86)\Microsoft SDKs\MPI\Lib\x64" -lmsmpi

INCLUDE=${MPIINC}
LINK   =${MPILIB}

SINGLE_TARGETS=test/mpitest.exe test/test.exe

HEADERS=DNDS_Array.hpp DNDS_BasicTypes.hpp DNDS_Defines.h DNDS_IndexMapping.hpp DNDS_MPI.hpp

VPATH:=test



FLAGS=-g
# FLAGS=-O3 -DNDEBUG



all: ${SINGLE_TARGETS}


%.exe: %.cpp
	g++ -o $@ $^  ${INCLUDE} ${LINK} ${FLAGS}


.PHONY: clean

clean:
	rm -f *.exe
	rm -f test/*.exe