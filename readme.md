## Naming Convention:

coded class: Hungarian
std::shared_ptr<T> type short-form-typedef: tp + T
std::vector<T> type short-from-typedef: t + T + Vec
STL container wrapper class: Hungarian

t... instances: ...
std::shared_ptr or pointer instances: p...
std::vector or vector wrapper instances: ...Vec



## Array

Array is a distributed data structure holding contiguously arrayed data.

It has template args:

**T**, a class which DO NOT hold data, but can be built onto a pointer and a size


It holds member data:

**data**, private, contiguous set of bytes holding actual data;

**dataGhost**, private, contiguous set of bytes holding data referencing from other processes;

**mpi**, private, mpi communicator info

**context**, T::Context instance, indexer building reference

**indexer** & **ghostIndexer**, T::Indexer instance, holds data to index the array

**pLGlobalMapping**, shared pointer to a GlobalOffsetsMapping instance, records how array's local indexing maps to global indexing, and vise versa.

**pLGhostMapping**, shared pointer to a OffsetAscendIndexMapping instance, records how array's ghost part indexing maps to global indexing and vise versa.

**pPushTypeVec**, shared pointer to MPITypePairHolder instance (which automatically destroys MPI_Datatypes), (\*pPushTypeVec)[i].first is comm rank to push to, (\*pPushTypeVec)[i].second is the MPI type on the location of data corresponding to the rank. i.e. the data type sends data when pulling, and receives data when pushing, on data.

**pPullTypeVec**, shared pointer to MPITypePairHolder instance (\*pPullTypeVec)[i].second is the MPI type on the location of dataGhost corresponding to the rank. i.e. the data type sends data when pushing and receives data when pulling, on dataGhost.

