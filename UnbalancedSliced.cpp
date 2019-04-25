#include "UnbalancedSliced.h"

#ifdef _WIN32
#include <malloc.h>
#endif


// cost function in 1-d ; simply used the quadratic cost
double cost(double x, double y) {
	const double z = x - y;
	return z*z;
}
float cost(float x, float y) {
	//return std::abs(x - y);
	const float z = x - y;
	return z*z;
}
__m256d cost(const __m256d &x, const __m256d &y) {
	__m256d diff = _mm256_sub_pd(x, y);
	return _mm256_mul_pd(diff, diff);
}
__m256 cost(const __m256 &x, const __m256 &y) {
	__m256 diff = _mm256_sub_ps(x, y);
	return _mm256_mul_ps(diff, diff);
}


// sum the cost of consecutively matching the first n values in h1 starting from start1 to the first n consecutive values of h2 starting at start2
// uses AVX : supposes h1 is aligned to 256 bits ; h2 need not be aligned
double sumCosts(const double* h1, int start1, const double* h2, int start2, int n) {
	if (n < 32) {
		double s = 0;
		const double* h1p = &h1[start1];
		const double* h2p = &h2[start2];
		for (int j = 0; j <n; j++) {
			s += cost(*h1p, *h2p); h1p++;  h2p++;
		}
		return s;
	}
	// AVX ; supposes h1 aligned
	double s = 0;
	while (start1 % 4 != 0) {
		s += cost(h1[start1], h2[start2]);
		start1++;
		start2++;
		n--;
	}
	const double* h1p = &h1[start1];
	const double* h2p = &h2[start2];
	__m256d s2 = _mm256_setzero_pd();
	for (int j = 0; j < n-3; j+=4) {
		__m256d hm1 = _mm256_load_pd(h1p);
		__m256d hm2 = _mm256_loadu_pd(h2p);
		s2 = _mm256_add_pd(s2, cost(hm1, hm2)); h1p+=4;  h2p+=4;
	}
	s2 = _mm256_hadd_pd(s2, s2);
	s += ((double*)&s2)[0] + ((double*)&s2)[2];
	for (int j = n-3; j < n; j++) {
		s += cost(*h1p, *h2p); h1p++;  h2p++;
	}
	return s;
}


float sum8(__m256 x) {
	// hiQuad = ( x7, x6, x5, x4 )
	const __m128 hiQuad = _mm256_extractf128_ps(x, 1);
	// loQuad = ( x3, x2, x1, x0 )
	const __m128 loQuad = _mm256_castps256_ps128(x);
	// sumQuad = ( x3 + x7, x2 + x6, x1 + x5, x0 + x4 )
	const __m128 sumQuad = _mm_add_ps(loQuad, hiQuad);
	// loDual = ( -, -, x1 + x5, x0 + x4 )
	const __m128 loDual = sumQuad;
	// hiDual = ( -, -, x3 + x7, x2 + x6 )
	const __m128 hiDual = _mm_movehl_ps(sumQuad, sumQuad);
	// sumDual = ( -, -, x1 + x3 + x5 + x7, x0 + x2 + x4 + x6 )
	const __m128 sumDual = _mm_add_ps(loDual, hiDual);
	// lo = ( -, -, -, x0 + x2 + x4 + x6 )
	const __m128 lo = sumDual;
	// hi = ( -, -, -, x1 + x3 + x5 + x7 )
	const __m128 hi = _mm_shuffle_ps(sumDual, sumDual, 0x1);
	// sum = ( -, -, -, x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7 )
	const __m128 sum = _mm_add_ss(lo, hi);
	return _mm_cvtss_f32(sum);
}

// sum the cost of consecutively matching the first n values in h1 starting from start1 to the first n consecutive values of h2 starting at start2
// uses AVX : supposes h1 is aligned to 256 bits ; h2 need not be aligned
float sumCosts(const float* h1, int start1, const float* h2, int start2, int n) {
	float s2b = -1;
	if (n < 32) {
		float s = 0;
		const float* h1p = &h1[start1];
		const float* h2p = &h2[start2];
		for (int j = 0; j < n; j++) {
			s += cost(*h1p, *h2p); h1p++;  h2p++;
		}
		return s;
	}
	// AVX ; supposes h1 aligned
	float s = 0;
	while (start1 % 8 != 0) {
		s += cost(h1[start1], h2[start2]);
		start1++;
		start2++;
		n--;
	}
	const float* h1p = &h1[start1];
	const float* h2p = &h2[start2];
	__m256 s2 = _mm256_setzero_ps();
	int j;
	for (j = 0; j < n - 7; j += 8) {
		__m256 hm1 = _mm256_load_ps(h1p);
		__m256 hm2 = _mm256_loadu_ps(h2p);
		s2 = _mm256_add_ps(s2, cost(hm1, hm2)); h1p += 8;  h2p += 8;
	}
	s += sum8(s2);
	for (int k = j; k < n; k++) {
		s += cost(*h1p, *h2p); h1p++;  h2p++;
	}
	return s;
}

void * malloc_simd(const size_t size, const size_t alignment) {
#if defined(WIN32) || defined(_MSC_VER)           // WIN32
    return _aligned_malloc(size, alignment);
#elif defined __linux__     // Linux
    return memalign(alignment, size);
#elif defined __MACH__      // Mac OS X
    return malloc(size);
#else                       // other (use valloc for page-aligned memory)
   // return valloc(size);
	return _aligned_malloc(size, alignment);
#endif
}

void free_simd(void* mem) {
#if defined(WIN32) || defined(_MSC_VER)           // WIN32
    return _aligned_free(mem);
#elif defined __linux__     // Linux
    free(mem);
#elif defined __MACH__      // Mac OS X
    free(mem);
#else                       // other (use valloc for page-aligned memory)
    free(mem);
#endif
}
