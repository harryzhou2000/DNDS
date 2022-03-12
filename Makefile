
MPIINC=-I "C:\Program Files (x86)\Microsoft SDKs\MPI\Include"
MPILIB=-L "C:\Program Files (x86)\Microsoft SDKs\MPI\Lib\x64" -lmsmpi
CGNSINC=-I "E:\projects\tools\CGNS-4.2.0\build\include"
CGNSLIB=-L "E:\projects\tools\CGNS-4.2.0\build\lib" -lcgns

INCLUDE=${MPIINC} ${CGNSINC}
LINK   =${MPILIB} ${CGNSLIB}

SINGLE_TARGETS=test/mpitest.exe test/test.exe test/cgnstest.exe

HEADERS=$(wildcard *.hpp *.h)

VPATH:=test



FLAGS=-g
# FLAGS=-O3 -DNDEBUG


all: ${SINGLE_TARGETS}


%.exe: %.cpp ${HEADERS}
	g++ -o $@ $(filter %.cpp , $^)  ${INCLUDE} ${LINK} ${FLAGS}


.PHONY: clean

clean:
	rm -f *.exe
	rm -f test/*.exe