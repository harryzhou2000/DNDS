#include <iostream>
#include "../../DNDS_Defines.h"
#include <cstdlib>

int main(int argc, char *argv[])
{
    assert(argc == 3);

    int a = std::stoi(argv[1]);
    int b = std::stoi(argv[2]);

    std::cout << DNDS::mod(a, b) << std::endl;

    return 0;
}