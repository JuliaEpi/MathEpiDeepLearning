---
sidebar_position: 1
---
# Descriptions

- Tensor and matrix computation
- Linear Solvers
- Nonlinear Solvers
- Matrix Equations
- HPC


# MKL Labpack

# PETSc and SLEPc

[PETSc 3.17 — PETSc 3.17.3 documentation](https://petsc.org/release/)

Scalable Library for Eigenvalue Problem Computations[SLEPc / slepc · GitLab](https://gitlab.com/slepc/slepc)

# suitesparse
[suitesparse : a suite of sparse matrix software](https://people.engr.tamu.edu/davis/suitesparse.html)

# Pardiso and MKLPardiso


# HSL
[HSL Mathematical Software Library](https://www.hsl.rl.ac.uk/)


# MUMPS
[MUMPS : a parallel sparse direct solver](http://mumps.enseeiht.fr/)

# cuSolvers
[cuSOLVER :: CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/cusolver/index.html)


### <span id="head17">3.1.3. Matrix and Tensor computation</span>

Matrix organization

[JuliaArrays](https://github.com/JuliaArrays)

- [JuliaArrays/StaticArrays.jl: Statically sized arrays for Julia](https://github.com/JuliaArrays/StaticArrays.jl)

- [JuliaArrays/ArrayInterface.jl: Designs for new Base array interface primitives, used widely through scientific machine learning (SciML) and other organizations](https://github.com/JuliaArrays/ArrayInterface.jl)

- [JuliaArrays/StructArrays.jl: Efficient implementation of struct arrays in Julia](https://github.com/JuliaArrays/StructArrays.jl)

- [JuliaArrays/LazyArrays.jl: Lazy arrays and linear algebra in Julia](https://github.com/JuliaArrays/LazyArrays.jl)
- [JuliaArrays/AxisArrays.jl: Performant arrays where each dimension can have a named axis with values](https://github.com/JuliaArrays/AxisArrays.jl)
- [JuliaArrays/OffsetArrays.jl: Fortran-like arrays with arbitrary, zero or negative starting indices.](https://github.com/JuliaArrays/OffsetArrays.jl)
- [JuliaArrays/BlockArrays.jl: BlockArrays for Julia](https://github.com/JuliaArrays/BlockArrays.jl)
- [JuliaArrays/ArraysOfArrays.jl: Efficient storage and handling of nested arrays in Julia](https://github.com/JuliaArrays/ArraysOfArrays.jl)
- [JuliaArrays/InfiniteArrays.jl: A Julia package for representing infinite-dimensional arrays](https://github.com/JuliaArrays/InfiniteArrays.jl)
- [JuliaArrays/FillArrays.jl: Julia package for lazily representing matrices filled with a single entry](https://github.com/JuliaArrays/FillArrays.jl)

[JuliaMatrices](https://github.com/JuliaMatrices)

- [JuliaMatrices/BandedMatrices.jl: A Julia package for representing banded matrices](https://github.com/JuliaMatrices/BandedMatrices.jl)

- [JuliaMatrices/BlockBandedMatrices.jl: A Julia package for representing block-banded matrices and banded-block-banded matrices](https://github.com/JuliaMatrices/BlockBandedMatrices.jl)
- [JuliaMatrices/SpecialMatrices.jl: Julia package for working with special matrix types.](https://github.com/JuliaMatrices/SpecialMatrices.jl)
- [JuliaMatrices/InfiniteLinearAlgebra.jl: A Julia repository for linear algebra with infinite matrices](https://github.com/JuliaMatrices/InfiniteLinearAlgebra.jl)

[RalphAS](https://github.com/RalphAS)

Good[JuliaLinearAlgebra](https://github.com/JuliaLinearAlgebra)

[JuliaSparse](https://github.com/JuliaSparse)

[JuliaLang/SparseArrays.jl: SparseArrays.jl is a Julia stdlib](https://github.com/JuliaLang/SparseArrays.jl)

[SciML/LabelledArrays.jl: Arrays which also have a label for each element for easy scientific machine learning (SciML)](https://github.com/SciML/LabelledArrays.jl)

[SciML/RecursiveArrayTools.jl: Tools for easily handling objects like arrays of arrays and deeper nestings in scientific machine learning (SciML) and other applications](https://github.com/SciML/RecursiveArrayTools.jl)

Python:

numpy

numba

[scikit-hep/awkward-1.0: Manipulate JSON-like data with NumPy-like idioms.](https://github.com/scikit-hep/awkward-1.0)

#### <span id="head18">Special Matrix and Arrays</span>

[JuliaMatrices/SpecialMatrices.jl: Julia package for working with special matrix types.](https://github.com/JuliaMatrices/SpecialMatrices.jl)

[SciML/LabelledArrays.jl: Arrays which also have a label for each element for easy scientific machine learning (SciML)](https://github.com/SciML/LabelledArrays.jl)

#### <span id="head80"> Computation </span>

BLAS and LAPACK[JuliaLinearAlgebra/MKL.jl: Intel MKL linear algebra backend for Julia](https://github.com/JuliaLinearAlgebra/MKL.jl)

[mcabbott/Tullio.jl: ⅀](https://github.com/mcabbott/Tullio.jl)

[JuliaLinearAlgebra/Octavian.jl: Multi-threaded BLAS-like library that provides pure Julia matrix multiplication](https://github.com/JuliaLinearAlgebra/Octavian.jl)

[JuliaGPU/GemmKernels.jl: Flexible and performant GEMM kernels in Julia](https://github.com/JuliaGPU/GemmKernels.jl)

[MasonProtter/Gaius.jl: Divide and Conquer Linear Algebra](https://github.com/MasonProtter/Gaius.jl)

#### <span id="head19"> Eigenvalues and Solvers </span>

Eig[nep-pack/NonlinearEigenproblems.jl: Nonlinear eigenvalue problems in Julia: Iterative methods and benchmarks](https://github.com/nep-pack/NonlinearEigenproblems.jl)

Solver[SciML/LinearSolve.jl: LinearSolve.jl: High-Performance Unified Linear Solvers](https://github.com/SciML/LinearSolve.jl)

Julia:

Eig:
[JuliaLinearAlgebra/Arpack.jl: Julia Wrappers for the arpack-ng Fortran library](https://github.com/JuliaLinearAlgebra/Arpack.jl)

[dgleich/GenericArpack.jl: A pure Julia translation of the Arpack library for eigenvalues and eigenvectors but for any numeric types. (Symmetric only right now)](https://github.com/dgleich/GenericArpack.jl)

[JuliaLinearAlgebra/ArnoldiMethod.jl: Implicitly Restarted Arnoldi Method, natively in Julia](https://github.com/JuliaLinearAlgebra/ArnoldiMethod.jl)

[Jutho/KrylovKit.jl: Krylov methods for linear problems, eigenvalues, singular values and matrix functions](https://github.com/Jutho/KrylovKit.jl)

[pablosanjose/QuadEig.jl: Julia implementation of the `quadeig` algorithm for the solution of quadratic matrix pencils](https://github.com/pablosanjose/QuadEig.jl)

[JuliaApproximation/SpectralMeasures.jl: Julia package for finding the spectral measure of structured self adjoint operators](https://github.com/JuliaApproximation/SpectralMeasures.jl)

[dgleich/GenericArpack.jl: A pure Julia translation of the Arpack library for eigenvalues and eigenvectors but for any numeric types. (Symmetric only right now)](https://github.com/dgleich/GenericArpack.jl)

Solver:

[JuliaInv/KrylovMethods.jl: Simple and fast Julia implementation of Krylov subspace methods for linear systems.](https://github.com/JuliaInv/KrylovMethods.jl)

[JuliaSmoothOptimizers/Krylov.jl: A Julia Basket of Hand-Picked Krylov Methods](https://github.com/JuliaSmoothOptimizers/Krylov.jl)

Eig Too[JuliaLinearAlgebra/IterativeSolvers.jl: Iterative algorithms for solving linear systems, eigensystems, and singular value problems](https://github.com/JuliaLinearAlgebra/IterativeSolvers.jl)

[tjdiamandis/RandomizedPreconditioners.jl](https://github.com/tjdiamandis/RandomizedPreconditioners.jl)

[JuliaLinearAlgebra/RecursiveFactorization.jl](https://github.com/JuliaLinearAlgebra/RecursiveFactorization.jl)

Spectral methods

[JuliaApproximation/SpectralMeasures.jl: Julia package for finding the spectral measure of structured self adjoint operators](https://github.com/JuliaApproximation/SpectralMeasures.jl)

[tpapp/SpectralKit.jl: Building blocks of spectral methods for Julia.](https://github.com/tpapp/SpectralKit.jl)

[markmbaum/BasicInterpolators.jl: Basic (+chebyshev) interpolation recipes in Julia](https://github.com/markmbaum/BasicInterpolators.jl)

Spasrse Slover

Sparse[JuliaSparse/Pardiso.jl: Calling the PARDISO library from Julia](https://github.com/JuliaSparse/Pardiso.jl)

Sparse[JuliaSparse/MKLSparse.jl: Make available to Julia the sparse functionality in MKL](https://github.com/JuliaSparse/MKLSparse.jl)

Sparse[JuliaLang/SuiteSparse.jl: Development of SuiteSparse.jl, which ships as part of the Julia standard library.](https://github.com/JuliaLang/SuiteSparse.jl)

Python:

[scipy.sparse.linalg.eigs — SciPy v1.7.1 Manual](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.eigs.html?highlight=scipy%20sparse%20linalg%20eigs#scipy.sparse.linalg.eigs)

#### <span id="head20">Maps and Operators</span>

[Jutho/LinearMaps.jl: A Julia package for defining and working with linear maps, also known as linear transformations or linear operators acting on vectors. The only requirement for a LinearMap is that it can act on a vector (by multiplication) efficiently.](https://github.com/Jutho/LinearMaps.jl)

[emmt/LazyAlgebra.jl: A Julia package to extend the notion of vectors and matrices](https://github.com/emmt/LazyAlgebra.jl)

[JuliaSmoothOptimizers/LinearOperators.jl: Linear Operators for Julia](https://github.com/JuliaSmoothOptimizers/LinearOperators.jl)

[kul-optec/AbstractOperators.jl: Abstract operators for large scale optimization in Julia](https://github.com/kul-optec/AbstractOperators.jl)

[matthieugomez/InfinitesimalGenerators.jl: A set of tools to work with Markov Processes](https://github.com/matthieugomez/InfinitesimalGenerators.jl)

[ranocha/SummationByPartsOperators.jl: A Julia library of summation-by-parts (SBP) operators used in finite difference, Fourier pseudospectral, continuous Galerkin, and discontinuous Galerkin methods to get provably stable semidiscretizations, paying special attention to boundary conditions.](https://github.com/ranocha/SummationByPartsOperators.jl)

[hakkelt/FunctionOperators.jl: Julia package that allows writing code close to mathematical notation memory-efficiently.](https://github.com/hakkelt/FunctionOperators.jl)

[JuliaApproximation/ApproxFun.jl: Julia package for function approximation](https://github.com/JuliaApproximation/ApproxFun.jl)

#### <span id="head21">Matrxi Equations</span>

[andreasvarga/MatrixEquations.jl: Solution of Lyapunov, Sylvester and Riccati matrix equations using Julia](https://github.com/andreasvarga/MatrixEquations.jl)

#### <span id="head22">Kronecker-based algebra</span>

[MichielStock/Kronecker.jl: A general-purpose toolbox for efficient Kronecker-based algebra.](https://github.com/MichielStock/Kronecker.jl)
