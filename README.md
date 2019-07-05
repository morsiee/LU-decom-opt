# LU-decom-opt
LU decomposition using four different ways

row-based parallelization, column-based parallelization, row-based parallelization with aid of BLAS L1 to scale and multiply vectors, and finally a reference implementation using LAPACK. Our result are as the following (local runs - Core i7 machine ):

    Solution 1
         9.45 +- 1.28 GFLOP/s
    Solution 2
        12.51 +- 1.82 GFLOP/s
    Solution 3
        48.42 +- 1.23 GFLOP/s
    LAPACK
        30.84 +- 0.96 GFLOP/s
