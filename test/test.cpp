
#include "../DNDS_Array.hpp"
#include "../DNDS_BasicTypes.hpp"
#include "../DNDS_MPI.hpp"
#include "../DNDS_DerivedTypes.hpp"
#include "../DNDS_Mesh.hpp"

#include <iostream>
#include <stdlib.h>


void testGhostLarge_Cascade();
void testPoint();
void testAdj();

int main(int argc, char *argv[])
{
    int ierr;
    ierr = MPI_Init(&argc, &argv);

    testGhostLarge_Cascade();
    // testPoint();
    // testAdj();

    ierr = MPI_Finalize();
    return 0;
}


void testGhostLarge_Cascade()
{
    DNDS::MPIInfo mpi;
    mpi.setWorld();

    int dsize, dstart, demandSize, demandStart, dmax;

    dsize = 1024 * 1024 * 1;
    dstart = 1024 * 1024 * 1 * mpi.rank;
    dmax = 1024 * 1024 * 1 * mpi.size;

    DNDS::Array<DNDS::VReal> ArrayA(
        DNDS::VReal::Context([](DNDS::index i) -> DNDS::rowsize
                             //  { return (i % 3 + 1); },
                             { return (1); },
                             dsize),
        mpi);
    for (int i = 0; i < ArrayA.size(); i++)
    {
        auto instance = ArrayA[i];
        for (int j = 0; j < instance.size(); j++)
            instance[j] = dstart + i;
    }

    demandSize = 1024;
    srand(mpi.rank);
    DNDS::tIndexVec PullDemand(demandSize);
    for (int i = 0; i < PullDemand.size(); i++)
        // PullDemand[i] = std::abs(RAND_MAX * rand() + rand()) % dmax;
        PullDemand[i] = i;
    // if(mpi.rank == 1)
    //     PullDemand.clear();

    decltype(ArrayA) ArrayAGhost(&ArrayA);

    ArrayAGhost.createGlobalMapping();
    ArrayAGhost.createGhostMapping(PullDemand);
    ArrayAGhost.createMPITypes();

    ArrayAGhost.initPersistentPull();

    int N = 100;
    MPI_Barrier(mpi.comm);
    double t0 = MPI_Wtime();
    for (int i = 0; i < N; i++)
    {
        ArrayAGhost.startPersistentPull();
        ArrayAGhost.waitPersistentPull();
    }
    double t1 = MPI_Wtime();
    MPI_Barrier(mpi.comm);
    if (mpi.rank == 0)
        std::cout << "Time Per Pull " << (t1 - t0) / N << std::endl;

    if (mpi.rank == 0)
    {
        std::cout << "Rank " << mpi.rank << " ";
        for (int i = 0; i < ArrayA.sizeGhost(); i++)
        {
            auto instance = ArrayA.indexGhostData(i);
            std::cout << "PD=" << ArrayAGhost.pLGhostMapping->ghostIndex[i] << "\t";
            for (int j = 0; j < instance.size(); j++)
                std::cout << instance[j] << "\t";
            std::cout << '\n';
        }
    }
    std::cout << std::endl;
}

void testPoint()
{
    DNDS::MPIInfo mpi;
    mpi.setWorld();
    DNDS::Array<DNDS::Vec3DBatch> ArrayP(DNDS::Vec3DBatch::Context(150), mpi);
    for (int i = 0; i < 150; i++)
        ArrayP[i].p() << 1, 2, i;
    for (int i = 0; i < 150; i++)
        std::cout << ArrayP[i].p().transpose() << std::endl;

    DNDS::Array<DNDS::SmallMatricesBatch> ArrayM(
        DNDS::SmallMatricesBatch::Context(
            [&](DNDS::index i) -> DNDS::rowsize
            {
                int nmats = i % 3 + 1;
                std::vector<int> matSizes(nmats * 2);
                for (int i = 0; i < nmats; i++)
                    matSizes[i * 2 + 0] = matSizes[i * 2 + 1] = i + 12;

                return DNDS::SmallMatricesBatch::predictSize(nmats, matSizes);
            },
            [&](uint8_t *data, DNDS::index siz, DNDS::index i)
            {
                int nmats = i % 3 + 1;
                std::vector<int> matSizes(nmats * 2);
                for (int i = 0; i < nmats; i++)
                    matSizes[i * 2 + 0] = matSizes[i * 2 + 1] = i + 12;

                DNDS::SmallMatricesBatch::initializeData(data, nmats, matSizes);
            },
            150),
        mpi);

    // DNDS::Array<DNDS::SmallMatricesBatch> Array1(
    //     DNDS::SmallMatricesBatch::Context(
    //         [&](DNDS::index i) -> DNDS::rowsize
    //         {
    //             int nmats = i % 3 + 1;
    //             std::vector<int> matSizes(nmats * 2);
    //             for (int i = 0; i < nmats; i++)
    //                 matSizes[i * 2 + 0] = matSizes[i * 2 + 1] = i + 12;

    //             return DNDS::SmallMatricesBatch::predictSize(nmats, matSizes);
    //         },
    //         150),
    //     mpi); // can't call
    
    for (int i = 0; i < 150; i++)
    {
        auto b = ArrayM[i];
        ///
        int nmats = i % 3 + 1;
        std::vector<int> matSizes(nmats * 2);
        for (int i = 0; i < nmats; i++)
            matSizes[i * 2 + 0] = matSizes[i * 2 + 1] = i + 12;
        ///
        // b.Initialize(nmats, matSizes);
        for (int im = 0; im < b.getNMat(); im++)
            b.m(im).setIdentity();
    }
    for (int i = 0; i < 150; i++)
    {
        auto b = ArrayM[i];
        for (int im = 0; im < b.getNMat(); im++)
            std::cout << "M\n"
                      << b.m(im) << std::endl;
    }
}

void testAdj()
{
    DNDS::MPIInfo mpi;
    mpi.setWorld();
    auto p1 = std::make_shared<DNDS::tAdjStatic2Array>(DNDS::tAdjStatic2Array::tContext(4), mpi);
    std::shared_ptr<DNDS::tAdjStatic2Array> p2;
    DNDS::forEachInArray(*p1, [&](DNDS::tAdjStatic2Array::tComponent &e, DNDS::index i)
                         { e[0] = 8 * mpi.rank + i * 2 + 0, e[1] = 8 * mpi.rank + i * 2 + 1; });
    if (mpi.rank == 0)
        (*p1)[0][0] = DNDS::FACE_2_VOL_EMPTY;
    std::vector<int> partI, partJ;
    if (mpi.rank % 2 == 0)
    {
        partI = std::vector<int>{0, 1, 0, 1};

        partJ = std::vector<int>{0, 0, 0, 0, 1, 1, 1, 1};
    }
    else
    {
        partI = std::vector<int>{0, 0, 1, 1};
        partJ = std::vector<int>{0, 0, 1, 1, 1, 1, 0, 0};
    }
    std::vector<DNDS::index> partIPush, partIPushStart;
    std::vector<DNDS::index> partJS2G;

    DNDS::Partition2LocalIdx(partI, partIPush, partIPushStart, mpi);
    DNDS::Partition2Serial2Global(partJ, partJS2G, mpi, mpi.size);
    DNDS::ConvertAdjSerial2Global(p1, partJS2G, mpi);
    DNDS::DistributeByPushLocal(p1, p2, partIPush, partIPushStart);

    auto p3 = std::make_shared<DNDS::tAdjStatic2Array>(p2.get()); // p2->p3
    p3->createGlobalMapping();
    // InsertCheck(mpi);
    // p2->LogStatus();
    // exit(0);
    MPISerialDo(mpi,
                [&]
                {
                    std::cout << std::endl;
                    std::cout << "Rank = " << mpi.rank << std::endl;
                    std::cout << "Ref count: " << p1.use_count() << p2.use_count() << std::endl;
                    DNDS::forEachInArray(*p2, [&](DNDS::tAdjStatic2Array::tComponent &e, DNDS::index i)
                                         { 
                                             //std::cout<<i <<std::endl;
                                             std::cout << p3->pLGlobalMapping->operator()(mpi.rank, i) << "===" << e[0] << ", " << e[1] << std::endl; });
                });
    // !note that global mapping of son is for father!!!!

    // InsertCheck(mpi);
}
