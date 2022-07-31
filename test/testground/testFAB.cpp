#include <iostream>

extern void *getA();
extern void *getB();

// mpicxx.openmpi -o testFAB.exe testFAB.cpp fA.cpp fB.cpp ../../DNDS_Profiling.cpp -I../..

int main()
{
    std::cout << getA() << ", " << getB() << std::endl;
    return 0;
}