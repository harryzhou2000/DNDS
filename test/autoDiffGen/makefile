
genRoe.exe: genRoe.cpp ../../DNDS_AutoDiff.hpp
	clang++ genRoe.cpp -o genRoe.exe -g -lopenblas -llapacke


genRoeTest.exe: genRoeTest.cpp ../../DNDS_AutoDiff.hpp
	clang++ genRoeTest.cpp -o genRoeTest.exe -O3  -ffp-contract=fast -funsafe-math-optimizations -lopenblas -llapacke -std=c++14

# -funsafe-math-optimizations -freciprocal-math