#include <math.h>
#include <stdio.h>
#include <omp.h>
#include <cassert>
#include <mkl.h>

void LU_decomp_LAPACKE(const int n, const int lda, double* const A) {
	// LU decomposition with mkl call
	int *ipiv = (int *) _mm_malloc(sizeof(int) * (n + 1), 64);
	LAPACKE_dgetrf(LAPACK_ROW_MAJOR, n, n, A, lda, ipiv);
}

void LU_decomp(const int n, const int lda, double* const A) {

	for (int k = 0; k < n - 1; ++k) {
		int i = k + 1;
		int j = i;
		int cnt = n - j;
		cblas_dscal((n - i), 1 / A[k * lda + k], &A[i * lda + k],
				lda);

#pragma omp parallel for shared(A)
		for (; i < n; i++) {
			cblas_daxpy(cnt, -1 * A[i * lda + k], &A[k * lda + j], 1,
					&A[i * lda + j], 1);

		}
	}

}


void LU_decomp_2(const int n, const int lda, double* const A) {
	for (int k = 0; k < n; k++) {
		int i = k + 1;
		double kk = A[k * lda + k];
#pragma omp parallel for shared(A) /*default(none) schedule(static, 8) */
		for (; i < n; i++) {
			A[i * lda + k] = A[i * lda + k] / kk;
			double ik = A[i * lda + k];
#pragma simd
#pragma ivdep
#pragma unroll(16)
			for (int j = k + 1; j < n; j++) {
				A[i * lda + j] -= ik * A[k * lda + j];
			}
		}
	}
}

void LU_decomp_1(const int n, const int lda, double* const A) {
	for (int k = 0; k < n - 1; k++) {
#pragma simd
#pragma ivdep
		for (int i = k + 1; i < n; i++) {
			A[i * lda + k] /= A[k * lda + k];
		}

#pragma omp parallel for shared(A,k) schedule(static, 8)
		for (int i = k + 1; i < n; i++) {

#pragma simd
#pragma ivdep
			for (int j = k + 1; j < n; j++) {
				A[i * lda + j] -= A[i * lda + k] * A[k * lda + j];
			}
		}
	}

}

void VerifyResult(const int n, const int lda, double* LU, double* refA) {

	// Verifying that A=LU
	double* A = (double*) _mm_malloc(sizeof(double) * n * lda, 64);
	double* L = (double*) _mm_malloc(sizeof(double) * n * lda, 64);
	double* U = (double*) _mm_malloc(sizeof(double) * n * lda, 64);
	A[0:lda] = 0.0f;
	L[0:lda] = 0.0f;
	U[0:lda] = 0.0f;
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < i; j++)
			L[i * lda + j] = LU[i * lda + j];
		L[i * lda + i] = 1.0f;
		for (int j = i; j < n; j++)
			U[i * lda + j] = LU[i * lda + j];
	}
	for (int i = 0; i < n; i++)
		for (int j = 0; j < n; j++)
			for (int k = 0; k < n; k++)
				A[i * lda + j] += L[i * lda + k] * U[k * lda + j];

	double deviation1 = 0.0;
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			deviation1 += (refA[i * lda + j] - A[i * lda + j])
					* (refA[i * lda + j] - A[i * lda + j]);
		}
	}
	deviation1 /= (double) (n * lda);
	if (isnan(deviation1) || (deviation1 > 1.0e-2)) {
		printf("ERROR: LU is not equal to A (deviation1=%e)!\n", deviation1);
		exit(1);
	}

#ifdef VERBOSE
	printf("\n(L-D)+U:\n");
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++)
		printf("%10.3e", LU[i*lda+j]);
		printf("\n");
	}

	printf("\nL:\n");
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++)
		printf("%10.3e", L[i*lda+j]);
		printf("\n");
	}

	printf("\nU:\n");
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++)
		printf("%10.3e", U[i*lda+j]);
		printf("\n");
	}

	printf("\nLU:\n");
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++)
		printf("%10.3e", A[i*lda+j]);
		printf("\n");
	}

	printf("\nA:\n");
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++)
		printf("%10.3e", refA[i*lda+j]);
		printf("\n");
	}

	printf("deviation1=%e\n", deviation1);
#endif

	_mm_free(A);
	_mm_free(L);
	_mm_free(U);

}

int main(const int argc, const char** argv) {

	// Problem size and other parameters
	const int n = 512;
	const int lda = 528;
	const int nMatrices = 100;
	const double HztoPerf = 1e-9 * 2.0 / 3.0 * double(n * n * lda) * nMatrices;

	const size_t containerSize = sizeof(double) * n * lda + 64;
	double* A = (double*) _mm_malloc(containerSize, 64);
	double* referenceMatrix = (double*) _mm_malloc(containerSize, 64);

	// Initialize matrix
#pragma omp parallel for
	for (int i = 0; i < n; i++) {
		double sum = 0.0f;
		for (int j = 0; j < n; j++) {
			A[i * lda + j] = (double) (i * n + j);
			sum += A[i * lda + j];
		}
		sum -= A[i * lda + i];
		A[i * lda + i] = 2.0f * sum;
	}
	A[(n - 1) * lda + n] = 0.0f; // Touch just in case
	referenceMatrix[0:n*lda] = A[0:n*lda];

	// Perform benchmark
	printf("LU decomposition of %d matrices of size %dx%d on %s...\n\n",
			nMatrices, n, n,
#ifndef __MIC__
			"CPU"
#else
			"MIC"
#endif
)	;

	double rate = 0, dRate = 0; // Benchmarking data
	const int nTrials = 10;
	const int skipTrials = 3; // First step is warm-up on Xeon Phi coprocessor
	printf("\033[1m%5s %10s %8s\033[0m\n", "Trial", "Time, s", "GFLOP/s");

	// Verify result
	LU_decomp(n, lda, A);
	VerifyResult(n, lda, A, referenceMatrix);

	// Measure performance
	for (int trial = 1; trial <= nTrials; trial++) {

		const double tStart = omp_get_wtime(); // Start timing
		// Benchmarking multiple decompositions to improve statistics
		for (int m = 0; m < nMatrices; m++)
			LU_decomp(n, lda, A);
		const double tEnd = omp_get_wtime(); // End timing

		if (trial > skipTrials) { // Collect statistics
			rate += HztoPerf / (tEnd - tStart);
			dRate += HztoPerf * HztoPerf / ((tEnd - tStart) * (tEnd - tStart));
		}

		printf("%5d %10.3e %8.2f %s\n", trial, (tEnd - tStart),
				HztoPerf / (tEnd - tStart), (trial <= skipTrials ? "*" : ""));
		fflush(stdout);
	}
	rate /= (double) (nTrials - skipTrials);
	dRate = sqrt(dRate / (double) (nTrials - skipTrials) - rate * rate);
	printf("-----------------------------------------------------\n");
	printf("\033[1m%s %4s \033[42m%10.2f +- %.2f GFLOP/s\033[0m\n",
			"Average performance:", "", rate, dRate);
	printf("-----------------------------------------------------\n");
	printf("* - warm-up, not included in average\n\n");

	_mm_free(A);
	_mm_free(referenceMatrix);

}
