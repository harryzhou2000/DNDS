
#include "../DNDS_Array.hpp"
#include "../DNDS_BasicTypes.hpp"
#include "../DNDS_MPI.hpp"

#include <iostream>
#include <stdlib.h>

void testType();
void testMPI();
void testGhost();
void testGhostLarge();

int main(int argc, char *argv[])
{
    int ierr;
    ierr = MPI_Init(&argc, &argv);

    // testType();
    // testMPI();
    // testGhost();
    testGhostLarge();

    ierr = MPI_Finalize();
    return 0;
}

void testType()
{
    std::cout << DNDS::Array<DNDS::Batch<double, 2>>::PrintTypes() << std::endl;

    // std::vector<int> siztest(1024 * 1024 * 128, -1);
    // std::cout << siztest[siztest.size() - 1] << std::endl;
    // for (auto &i : siztest)
    //     i = 1;
    // siztest.resize(1024*1024*256*4,3);
    // std::string a;
    // std::cin >> a;

    1024LL * 1024 * 1024 * 4;

    std::shared_ptr<int> A;
    A = std::make_shared<int>(10);
    A.reset();
    std::cout << A.use_count() << std::endl;

    DNDS::MPIInfo mpi;
    mpi.setWorld();

    long long testSiz = 1024 * 1024 * 3;
    const uint32_t bsize = 20;
    DNDS::Array<DNDS::Batch<double, bsize>> ArrayRB(DNDS::Batch<double, bsize>::Context(testSiz), mpi);
    std::cout << ArrayRB.indexer.LengthByte() << std::endl
              //   << int(ArrayRB.data[160]) << std::endl
              << ArrayRB.indexer.Length << std::endl;
    std::cout << ArrayRB[9] << std::endl;

    double t0, t1;

    t0 = MPI_Wtime();
    for (int i = 0; i < ArrayRB.indexer.Length; i++)
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

    DNDS::tpIndexVec rowB(new DNDS::tIndexVec);
    DNDS::tRowsizeVec rowBsiz(12, 3);
    rowBsiz[4] = 4, rowBsiz[11] = 5;
    DNDS::AccumulateRowSize(rowBsiz, *rowB);

    DNDS::Array<DNDS::VarBatch<double>> ArrayRVB(DNDS::VarBatch<double>::Context(0, rowB), mpi);

    std::cout << "ArrayRVB bytes: " << ArrayRVB.sizeByte() << std::endl;

    double acc = 0.0;
    for (int i = 0; i < ArrayRVB.size(); i++)
    {
        auto instance = ArrayRVB[i];
        for (int j = 0; j < instance.size(); j++)
            instance[j] = acc++;
        std::cout << instance << std::endl;
    }
}

void testMPI()
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (size != 2)
    {
        std::cout << "Size not 2 \n";
        return;
    }
    std::vector<int> buf(1024 * 1024, 0);
    for (int i = 0; i < buf.size(); i++)
        buf[i] = i;

    if (rank == 0)
    {
        std::vector<int> blklen(10, 3);
        std::vector<MPI_Aint> blkPlc(10, 0);
        blklen[9] = 2;
        blklen[4] = 4;
        blkPlc[0] = 0, blkPlc[1] = 4, blkPlc[2] = 8, blkPlc[3] = 11, blkPlc[4] = 15, blkPlc[5] = 123,
        blkPlc[6] = 126, blkPlc[7] = 129, blkPlc[8] = 1000123, blkPlc[9] = 1000125;
        for (auto &d : blkPlc)
            d *= sizeof(int);
        MPI_Datatype stype;
        MPI_Type_create_hindexed(10, blklen.data(), blkPlc.data(), MPI_INT, &stype);
        MPI_Type_commit(&stype);
        int packedSiz;
        MPI_Pack_size(1, stype, MPI_COMM_WORLD, &packedSiz);
        std::cout << "Packed siz: " << packedSiz << std::endl;

        std::vector<uint_fast8_t> packBuf(packedSiz);
        int packBufPos = 0;
        MPI_Pack(buf.data(), 1, stype, packBuf.data(), packedSiz, &packBufPos, MPI_COMM_WORLD);
        blklen.resize(0);
        MPI_Send(buf.data(), 1, stype, 1, 123, MPI_COMM_WORLD);
        MPI_Type_free(&stype);
    }
    else
    {
        MPI_Status recvStat;
        MPI_Recv(buf.data(), 120, MPI_BYTE, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &recvStat);
        for (int i = 0; i < 30; i++)
            std::cout << buf[i] << '\t';
        std::cout << std::endl;
    }
}

void testGhost()
{
    DNDS::MPIInfo mpi;
    mpi.setWorld();
    if (mpi.size != 4)
    {
        std::cout << "Size not 4\n";
        return;
    }

    int dsize, dstart, demandSize, demandStart;

    switch (mpi.rank)
    {
    case 0:
        dsize = 4, dstart = 0, demandSize = 2, demandStart = 7;
        break;
    case 1:
        dsize = 5, dstart = 4, demandSize = 3, demandStart = 9;
        break;
    case 2:
        dsize = 4, dstart = 9, demandSize = 1, demandStart = 15;
        break;
    case 3:
        dsize = 5, dstart = 13, demandSize = 2, demandStart = 1;
        break;
    default:
        return;
        break;
    }

    DNDS::Array<DNDS::VReal> ArrayA(DNDS::VReal::Context([](DNDS::index i) -> DNDS::rowsize
                                                         { return (i + 1); },
                                                         dsize),
                                    mpi);
    for (int i = 0; i < ArrayA.size(); i++)
    {
        auto instance = ArrayA[i];
        for (int j = 0; j < instance.size(); j++)
            instance[j] = dstart + i;
    }
    DNDS::tIndexVec PullDemand(demandSize);
    for (int i = 0; i < PullDemand.size(); i++)
        PullDemand[i] = demandStart + i;
    if (mpi.rank == 3)
        PullDemand[1] = 14;

    ArrayA.createGhostMapping(PullDemand);
    ArrayA.createMPITypes();

    ArrayA.initPersistentPull();
    ArrayA.startPersistentPull();
    ArrayA.waitPersistentPull();

    std::cout << "Rank " << mpi.rank << " ";
    std::cout << "Size Ghost " << ArrayA.sizeGhost() << " ::\t";
    for (int i = 0; i < ArrayA.sizeGhost(); i++)
    {
        auto instance = ArrayA.indexGhostData(i);
        std::cout << "Pack Size " << instance.size() << "::\t";
        for (int j = 0; j < instance.size(); j++)
        {
            std::cout << instance[j] << "\t";
            instance[j] *= -1;
        }
    }
    std::cout << std::endl;

    ArrayA.initPersistentPush();
    ArrayA.startPersistentPush();
    ArrayA.waitPersistentPush();

    std::cout << "Rank " << mpi.rank << " ";
    for (int i = 0; i < ArrayA.size(); i++)
    {
        auto instance = ArrayA[i];
        for (int j = 0; j < instance.size(); j++)
            std::cout << instance[j] << "\t";
    }
    std::cout << std::endl;
}

void testGhostLarge()
{
    DNDS::MPIInfo mpi;
    mpi.setWorld();

    int dsize, dstart, demandSize, demandStart, dmax;

    dsize = 1024 * 1024;
    dstart = 1024 * 1024 * mpi.rank;
    dmax = 1024 * 1024 * mpi.size;

    DNDS::Array<DNDS::VReal> ArrayA(DNDS::VReal::Context([](DNDS::index i) -> DNDS::rowsize
                                                         { return (i % 3 + 1); },
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
        PullDemand[i] = (RAND_MAX * rand() + rand()) % dmax;
    std::sort(PullDemand.begin(), PullDemand.end());

    ArrayA.createGhostMapping(PullDemand);
    ArrayA.createMPITypes();

    ArrayA.initPersistentPull();

    int N = 100;
    MPI_Barrier(mpi.comm);
    double t0 = MPI_Wtime();
    for (int i = 0; i < N; i++)
    {
        ArrayA.startPersistentPull();
        ArrayA.waitPersistentPull();
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
            std::cout << "PD=" << ArrayA.pLGhostMapping->ghostIndex[i] << "\t";
            for (int j = 0; j < instance.size(); j++)
                std::cout << instance[j] << "\t";
            std::cout << '\n';
        }
    }
    std::cout << std::endl;
}