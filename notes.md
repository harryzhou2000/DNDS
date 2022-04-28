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
of the hetrogenious systems:
    beware that currently DNDS do not handle endian problems!!!





