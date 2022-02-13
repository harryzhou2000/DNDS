
#include "../DNDS_Array.hpp"
#include "../DNDS_BasicTypes.hpp"

#include <iostream>

void testType();

int main(int argc, char *argv[])
{
    int ierr;
    ierr = MPI_Init(&argc, &argv);

    testType();

    ierr = MPI_Finalize();
    return 0;
}

void testType()
{
    std::cout << DNDS::Array<DNDS::RealBatch<2>>::PrintTypes() << std::endl;

    std::shared_ptr<int> A;
    A = std::make_shared<int>(10);
    A.reset();
    std::cout << A.use_count() << std::endl;

    DNDS::MPIInfo mpi;
    mpi.setWorld();

    long long testSiz = 1024 * 1024 * 3;
    const uint32_t bsize = 20;
    DNDS::Array<DNDS::RealBatch<bsize>> ArrayRB(std::make_shared<DNDS::RealBatch<bsize>::Context>(DNDS::RealBatch<bsize>::Context(testSiz)), mpi);
    std::cout << ArrayRB.pIndexer->LengthByte() << std::endl
              << int(ArrayRB.data[160]) << std::endl
              << ArrayRB.pIndexer->Length << std::endl;
    std::cout << ArrayRB[9] << std::endl;

    double t0, t1;

    t0 = MPI_Wtime();
    for (int i = 0; i < ArrayRB.pIndexer->Length; i++)
    {
        auto instance = ArrayRB[i];
        instance[0] = 1.0;
        ArrayRB[i][1] = 2.0;
    }
    t1 = MPI_Wtime();
    std::cout << "Perf: " << t1 - t0 << std::endl;

    std::vector<double> parray(testSiz * bsize);

    t0 = MPI_Wtime();
    for (int i = 0; i < testSiz; i++)
    {
        parray[i * bsize + 0] = 1.0;
        parray[i * bsize + 1] = 2.0;
    }
    t1 = MPI_Wtime();

    std::cout << "Perf: " << t1 - t0 << std::endl;

    std::cout << ArrayRB[9];
    std::cout << ArrayRB[testSiz - 1] << std::endl;
}