# Notes for DNDS's development

# 2022-4-24
The library currently runs on eikonal, at least, without much problem. The reason why both reference code and this blows up without increment limitation is yet to be checked.

Currently, this library tries to provide a rapid development environment for FEM/DG/High-Order-FVM/FR methods, providing:

- low-level data structure: distributed 2-d array, allowing multiple levels/directions of persistent communication
- mesh manager, read/write mesh/data, and manage mesh distribution
- element manager, in-place unified operation for various mesh elements
- solver classes, here fv vfv etc. which defines the numeric algorithms
- utilities

DNDS is currently entirely based on MPI infrastructures, and the communication models is based on the assumption of using spacial partition on the calculating space.

possible problems to solve: 
- 3-d space, some code is fixed to 3-d operations, could improve for 2-d case? could support higher spacial dimensions? (but higher dim unstructured mesh is a new topology problem, and is rarely needed...)
- specialize some data arrays to achieve better performance or syntax, e.g. fixed width 2-d arrays on indexing and communications, 1-width array returning methods
- local CSR array? or just use distributed version?
- adding concept constraints? check mainstream super computers' c++ std!
- some forEach iteration is better with for iteration... adding iterator-based forEach-es?
- is preloaded -Pair object type necessary? is data-connecting necessary?

future features:
- python and json utilities for scripting
- PETSc and so on support
- CGNS io support
- tecplot binary io support
- in-time visualization?...

# 2022-4-27
of the errors/incompatibilities:
    could be in VR/CR implementation (losing items, mixing u-rec and u-mean, matrix misuse)
    could be in weights difference or RHS implementation
    one was in hardeigen, which is Singular value filtering ???(do without?)

# 2022 4-28
of the heterogeneous systems:
    beware that currently DNDS do not handle endian problems!!!

# performance

| machine | cpu | memory | notes | 16Gw |
|--|--|--|--|--|
| HZZ7 | 9750H 6c12t 12M-L3 | 16G 2666 x2 | on WSL|  5.1 |
| GPU704 | 6326x2 16c32tx2 24M-L3x2 | 32G 3200 x2||  8.7 |
| TC C6 64| 8372cx2 16c32tx2 43M-L3x2 | 16G x16 (should be 3200)||7.3|
| TC SA3 64| EPYC 7K83 32c64t ~128M-L3? | 16G x16 (should be 3200)||6.3|

|mesh | memory size|
|---|---|
|H3(14280)| 793M |
|H300(145600)| 7924M |
|500k(509600)| 27.0G |


| machine | mesh          | np  | 10 step time |
| --- | --- | ---         | --- | ---          |
|HZZ7 | H3(14280)         | 1   | 3.3          |
|HZZ7 | H3(14280)         | 2   | 1.85         |
|HZZ7 | H3(14280)         | 4   | 1.3          |
|HZZ7 | H3(14280)         | 6   | 1.2          |
|HZZ7 | H300(145600)      | 1   | 35           |
|HZZ7 | H300(145600)      | 2   | 19.5         |
|HZZ7 | H300(145600)      | 4   | 13           |
|HZZ7 | H300(145600)      | 6   | 12           |
|GPU704 | H3(14280)       | 1   | 2.9          |
|GPU704 | H3(14280)       | 2   | 1.65         |
|GPU704 | H3(14280)       | 4   | 0.83         |
|GPU704 | H3(14280)       | 8   | 0.65         |
|GPU704 | H3(14280)       | 16  | 0.62         |
|GPU704 | H3(14280)       | 32  | 0.61         |
|GPU704 | H300(145600)    | 1   | 30.5         |
|GPU704 | H300(145600)    | 2   | 17.5         |
|GPU704 | H300(145600)    | 4   | 9            |
|GPU704 | H300(145600)    | 8   | 6.9          |
|GPU704 | H300(145600)    | 16  | 6.5          |
|GPU704 | H300(145600)    | 32  | 6.4          |
|GPU704 | 500k(509600)    | 8   | 24           |
|GPU704 | 500k(509600)    | 16  | 22.9         |
|GPU704 | 500k(509600)    | 32  | 22.3         |
|TC C6 64 | H3(14280)     | 1   | 3            |
|TC C6 64 | H3(14280)     | 2   | 1.52         |
|TC C6 64 | H3(14280)     | 4   | 0.76         |
|TC C6 64 | H3(14280)     | 8   | 0.41         |
|TC C6 64 | H3(14280)     | 16  | 0.23         |
|TC C6 64 | H3(14280)     | 32  | 0.14         |
|TC C6 64 | H300(145600)  | 1   | 31.6         |
|TC C6 64 | H300(145600)  | 2   | 16.3         |
|TC C6 64 | H300(145600)  | 4   | 8.2          |
|TC C6 64 | H300(145600)  | 8   | 4.3          |
|TC C6 64 | H300(145600)  | 16  | 2.4          |
|TC C6 64 | H300(145600)  | 32  | 1.45         |
|TC C6 64 | 500k(509600)  | 8   | 15.2         |
|TC C6 64 | 500k(509600)  | 16  | 8.4          |
|TC C6 64 | 500k(509600)  | 32  | 5.1          |
|TC SA3 64 | H3(14280)    | 1   | 3.4          |
|TC SA3 64 | H3(14280)    | 2   | 1.7          |
|TC SA3 64 | H3(14280)    | 4   | 0.78         |
|TC SA3 64 | H3(14280)    | 8   | 0.41         |
|TC SA3 64 | H3(14280)    | 16  | 0.24         |
|TC SA3 64 | H3(14280)    | 32  | 0.18         |
|TC SA3 64 | H300(145600) | 1   | 35           |
|TC SA3 64 | H300(145600) | 2   | 17.7         |
|TC SA3 64 | H300(145600) | 4   | 8.9          |
|TC SA3 64 | H300(145600) | 8   | 4.7          |
|TC SA3 64 | H300(145600) | 16  | 2.9          |
|TC SA3 64 | H300(145600) | 32  | 2.2          |
|TC SA3 64 | 500k(509600) | 8   | 16.6         |
|TC SA3 64 | 500k(509600) | 16  | 10.1         |
|TC SA3 64 | 500k(509600) | 32  | 7.7          |

|test case   | np  | GPU704 | HZZ7 | TC C6 64 | TC SA3 64|
|---         |---  | ---    | ---  |  ---     | ---      |
|H3(14280)   | 1   | 2.9    | 3.3  |   3      |  3.4     |
|H3(14280)   | 2   | 1.65   | 1.85 |   1.52   |  1.7     |
|H3(14280)   | 4   | 0.83   | 1.3  |   0.76   |  0.78    |
|H3(14280)   | 8   | 0.65   |      |   0.41   |  0.41    |
|H3(14280)   | 16  | 0.62   |      |   0.23   |  0.24    |
|H3(14280)   | 32  | 0.61   |      |   0.14   |  0.18    |
|H300(145600)| 1   | 30.5   | 35   |   31.6   |  35      |
|H300(145600)| 2   | 17.5   | 19.5 |   16.3   |  17.7    |
|H300(145600)| 4   | 9      | 13   |   8.2    |  8.9     |
|H300(145600)| 8   | 6.9    |      |   4.3    |  4.7     |
|H300(145600)| 16  | 6.5    |      |   2.4    |  2.9     |
|H300(145600)| 32  | 6.4    |      |   1.45   |  2.2     |
|500k(509600)| 8   | 24     |      |   15.2   |  16.6    |
|500k(509600)| 16  | 22.9   |      |   8.4    |  10.1    |
|500k(509600)| 32  | 22.3   |      |   5.1    |  7.7     |


# super large obj files:
not due to -O3 unrolling or inlining, but -g symbols... normally should be 1.5M  (eikonal.exe)

# about integration:
in this scheme for eikonal, downgrading face int order is bad, downgrading volume int order is ok