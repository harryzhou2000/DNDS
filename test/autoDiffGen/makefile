
genRoe.exe: genRoe.cpp ../../DNDS_AutoDiff.hpp
	clang++ genRoe.cpp -o genRoe.exe -g -lopenblas -llapacke


genRoeTest.exe: genRoeTest.cpp ../../DNDS_AutoDiff.hpp
	clang++ genRoeTest.cpp -o genRoeTest.exe -O3  -ffp-contract=fast -funsafe-math-optimizations -lopenblas -llapacke -std=c++14

# -funsafe-math-optimizations -freciprocal-math

genVis.exe: genVis.cpp ../../DNDS_AutoDiff.hpp
	clang++ genVis.cpp -o genVis.exe -O3 -lopenblas -llapacke

genVisTest.exe: genVisTest.cpp ../../DNDS_AutoDiff.hpp
	clang++ genVisTest.cpp -o genVisTest.exe -O3  -ffp-contract=fast -lopenblas -llapacke -std=c++14
