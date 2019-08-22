#pragma once
/*
  Copyright (c) 2019 CNRS
  Nicolas Bonneel <nicolas.bonneel@liris.cnrs.fr>
  David Coeurjolly <david.coeurjolly@liris.cnrs.fr>

  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are met:

  1. Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

  2. Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIEDi
  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
  DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
  ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <iostream>
#include <vector>
#include <algorithm>
#include <utility>
#include <chrono>
#include <ctime>
#include <cstring>
#include <cmath>
#include <omp.h>
#include <list>
#include <random>
#define cimg_display 0
#include "CImg.h"
#include "Point.h"

#ifdef _MSC_VER
  #include <intrin.h>
#else
  #include <immintrin.h>
  #if __linux__
    #include <malloc.h>
  #endif
#endif


#ifndef M_PI
#define M_PI 3.14159265358979323856
#endif

float cost(float x, float y);
double cost(double x, double y);
__m256 cost(const __m256 &x, const __m256 &y);
__m256d cost(const __m256d &x, const __m256d &y);
double sumCosts(const double* h1, int start1, const double* h2, int start2, int n);
float sumCosts(const float* h1, int start1, const float* h2, int start2, int n);
void * malloc_simd(const size_t size, const size_t alignment);
void free_simd(void* mem);

struct params {
	params() {};
	params(int d0, int f0, int d1, int f1, int d) :start0(d0), end0(f0), start1(d1), end1(f1) {};
	int start0, end0, start1, end1;
};


#ifdef __APPLE__
static std::default_random_engine engine(10); // 10 = random seed
static std::uniform_real_distribution<double> uniform(0, 1);
#else
static thread_local std::default_random_engine engine(10); // 10 = random seed
static thread_local std::uniform_real_distribution<double> uniform(0, 1);
#endif

template<typename T>
Point<2, T> BoxMuller() {
	double r1 = uniform(engine);
	double r2 = uniform(engine);

	Point<2, T> p;
	double f = sqrt(-2 * log(std::max(1E-12, std::min(1. - 1E-12, r1))));
	p[0] = f * cos(2 * M_PI*r2);
	p[1] = f * sin(2 * M_PI*r2);
	return p;
}


template<int DIM, typename T>
struct Projector {
	Projector(const Point<DIM, T> &dir) : dir(dir) {};
	double proj(const Point<DIM, T> &p) {
		double proj = 0;
		for (int i = 0; i < DIM; i++) {
			proj += p[i] * dir[i];
		}
		return proj;
	}
	Point<DIM, T> dir;
};



class UnbalancedSliced {
public:


	// nearest neighbors in 1d
	template<typename T>
	void nearest_neighbor_match(const T *hist1, const T* hist2, const params &p, std::vector<int> &assignment) {

		int cursor = p.start1;
		for (int i = p.start0; i < p.end0; i++) {
			T mind = std::numeric_limits<T>::max();
			int minj = -1;
			for (int j = std::max(p.start1, cursor); j < p.end1; j++) {
				T d = cost(hist1[i], hist2[j]);
				cursor = j - 1;
				if (d <= mind) {
					mind = d;
					minj = j;
				} else
					if (d > mind+std::numeric_limits<T>::epsilon()) {
						break;
					}

			}
			assignment[i] = minj;
		}
	}

	// handles the case where the first sequence starts before or ends after the second sequence, or where the NN of the first (resp. last) elements of hist1 are the first (resp. last) elements of hist2
	// also restricts problem size based on the number of non-injective values, but that won't be super useful
	// returns 1 if hist1 entirely consumed ; 0 otherwise
	template<typename T>
	int reduce_range(const T *hist1, const T* hist2, std::vector<int> &assignment, params &inparam, T& emd, int* assNN, int nbbij) {
		params p0 = inparam;

		/// hist1 (partly) at the left of hist2 : can match the outside of hist1 to the begining of hist2
		int cursor1 = inparam.start1;
		int min0 = inparam.start0;
		T localchange = 0;
		for (int i = inparam.start0; i < inparam.end0; i++) {
			if (hist1[i] <= hist2[cursor1]) {
				assignment[i] = cursor1;
				localchange += cost(hist1[i], hist2[cursor1]);
				cursor1++;
				min0 = i + 1;
			} else break;
		}
		inparam.start0 = min0;
		inparam.start1 = cursor1;

		if (inparam.end0 == inparam.start0) {
#pragma omp atomic
			emd += localchange;
			return 1;
		}

		/// hist1 (partly) at the right of hist2 : can match the outside of hist1 to the end of hist2
		int cursor1b = inparam.end1 - 1;
		int max0 = inparam.end0 - 1;
		for (int i = inparam.end0 - 1; i >= inparam.start0; i--) {
			if (hist1[i] >= hist2[cursor1b]) {
				assignment[i] = cursor1b;
				localchange += cost(hist1[i], hist2[cursor1b]);
				cursor1b--;
				max0 = i - 1;
			} else break;
		}

		inparam.end0 = max0 + 1;
		inparam.end1 = cursor1b + 1;

		if (inparam.end0 == inparam.start0) {
#pragma omp atomic
			emd += localchange;
			return 1;
		}


		// non-injective matches not super useful anymore.

		// We don't need to match to the entire range of hist2 : only the range of the nearest neighbors of the bounds of hist1, +/- the number of non-injective values
		int M = inparam.end0 - inparam.start0;
		inparam.start1 = std::max(inparam.start1, assNN[inparam.start0] - nbbij);
		inparam.end1 = std::min(inparam.end1, assNN[inparam.end0 - 1] + nbbij + 1); // +1 since bound is excluded


		//if the NN of the begining of hist1 are matched to the first values of hist2, then they should be matched
		int cursor = inparam.start1;
		int i;

		for (i = inparam.start0; i < inparam.end0; i++) {
			if (assNN[i] == cursor && (i == inparam.end0 - 1 || assNN[i + 1] != assNN[i])) {
				assignment[i] = cursor;
				localchange += cost(hist1[i], hist2[assignment[i]]);
				cursor++;
			} else break;
		}
		inparam.start0 = i;
		inparam.start1 = cursor;

		if (inparam.start0 == inparam.end0) {
#pragma omp atomic
			emd += localchange;
			return 1;
		}

		int prevfin = inparam.end0;
		int prevend1 = inparam.end1;
		//if the NN of the end of hist1 are matched to the last values of hist2, then they should be matched
		cursor = inparam.end1 - 1;
		for (i = inparam.end0 - 1; i >= inparam.start0; i--) {
			if (assNN[i] == cursor && (i == inparam.start0 || assNN[i - 1] != assNN[i])) {
				assignment[i] = cursor;
				localchange += cost(hist1[i], hist2[assignment[i]]);
				cursor--;
			} else break;
		}
		inparam.end0 = i + 1;
		inparam.end1 = cursor + 1;

		if (inparam.start0 == inparam.end0) {
#pragma omp atomic
			emd += localchange;
			return 1;
		}


#pragma omp atomic
		emd += localchange;
		return 0;
	}



	// handles trivial cases: M==N, M==N-1, M==1, or nearest neighbor map is injective
	// return 1 = subproblem solved
	// return 0 = problem not solved

	template<typename T>
	int handle_simple_cases(const params &p, const T* hist1, const T* hist2, int* assignment, int* assNN, T &value) {
		int start0 = p.start0;
		int start1 = p.start1;
		int end0 = p.end0;
		int end1 = p.end1;
		int M = end0 - start0;
		int N = end1 - start1;
		if (M == 0) return 1;
		if (M == N) {
			T d = 0;
			for (int i = 0; i < M; i++) {
				assignment[start0 + i] = i + start1;
				d += cost(hist1[start0 + i], hist2[start1 + i]);
			}
#pragma omp atomic
			value += d;
			return 1;
		}
		if (M == N - 1) {
			T d1 = 0;
			T d2 = 0;
			for (int i = 0; i < M; i++) {
				d2 += cost(hist1[start0 + i], hist2[start1 + i + 1]);
			}
			T d = 0;
			T b = d2;
			T best_s = d2; // this is actually optional: this is a constant.
			int besti = -1;
			for (int i = 0; i < M; i++) {
				d1 += cost(hist1[start0 + i], hist2[start1 + i]); // forward cost
				b -= cost(hist1[start0 + i], hist2[start1 + i + 1]); // backward cost
				T s = b + d1;
				if (s < best_s) {
					best_s = s;
					besti = i;
				}
			}
			for (int i = 0; i < M; i++) {
				if (i <= besti) {
					assignment[start0 + i] = i + start1;
				} else {
					assignment[start0 + i] = i + 1 + start1;
				}
			}
#pragma omp atomic
			value += best_s;
			return 1;
		}
		if (M == 1) {
			assignment[start0] = assNN[start0];
			T c = cost(hist1[start0], hist2[assNN[start0]]);

#pragma omp atomic
			value += c;
			return 1;
		}

// checks if NN is injective
		{
			int curId = 0;
			T sumMin = 0;
			bool valid = true;
			for (int i = 0; i < M; i++) {
				int ass;
				T h1 = hist1[start0 + i];
				T mini = std::numeric_limits<T>::max();
				for (int j = curId; j < N; j++) {
					T v = cost(h1, hist2[start1 + j]);
					curId = j;
					if (v < mini) {
						mini = v;
						ass = j + start1;
					}
					if (j < N - 1) {
						T vnext = cost(h1, hist2[start1 + j + 1]);
						if (vnext > v) break;
					}
				}
				if (mini == std::numeric_limits<T>::max()) {
					valid = false;
					break;
				}
				if (i > 0 && ass == assignment[start0 + i - 1]) {
					valid = false;
					break;
				}
				sumMin += mini;
				assignment[start0 + i] = ass;
			}
			if (valid) {
#pragma omp atomic
				value += sumMin;
				return 1;
			}
		}

		return 0;
	}





// decompose a problem into subproblems in (quasi) linear time.
	template<typename T>
	bool linear_time_decomposition(const params &p, const T* hist1, const T* hist2, int* assNN, std::vector<params>& newp) {

		if (p.end0 - p.start0 < 20) { // not worth splitting already tiny problems
			return false;
		}
		int N = p.end1 - p.start1;
		std::vector<int> taken(N, -1);
		std::vector<int> ninj(N, 0);
		taken[assNN[p.start0] - p.start1] = p.start0;
		ninj[assNN[p.start0] - p.start1]++;

		std::vector<int> prev_free(p.end1 - p.start1);
		std::vector<int> next_free(p.end1 - p.start1);
		for (int i = 0; i < prev_free.size(); i++) {
			prev_free[i] = i;
			next_free[i] = i;
		}

		int first_right = assNN[p.start0 + 1] - p.start1, last_left = assNN[p.start0 + 1] - p.start1;
		for (int i = p.start0 + 1; i < p.end0; i++) {
			int ass = assNN[i];
			int assOffset = ass - p.start1;
			ninj[assOffset]++;
			if ((taken[assOffset]) < 0) {
				taken[assOffset] = i;
				first_right = assOffset;
				last_left = assOffset;
			} else {

				if (ninj[assOffset] > 1) {
					int cur = last_left-1;
					while (cur >=0) {
						if (taken[cur] < 0 || cur == 0) {
							taken[cur] = i;
							prev_free[assOffset] = cur;
							next_free[cur] = next_free[assOffset];
							break;
						} else {
							if (prev_free[cur] == cur) cur--; else
							cur = prev_free[cur];
						}
					}

					last_left = std::max(0, cur);
				} else {
					prev_free[assOffset] = prev_free[prev_free[assOffset]];
				}
				if (first_right < N - 1) {
					first_right++;
				}
				taken[first_right] = i;
				prev_free[first_right] = last_left;
				next_free[assOffset] = first_right;
				next_free[last_left] = first_right;

			}
		}

		int lastStart = p.start0;
		for (int i = p.start1; i < p.end1; i++) {

			int assOffset = i - p.start1;

			int maxival = taken[assOffset];

			if (taken[assOffset] >= 0) {
				params curp;
				if (next_free[assOffset] == assOffset) {
					curp.start1 = p.start1 + assOffset;
					curp.start0 = lastStart;
					lastStart++;
					curp.end0 = curp.start0 + 1;
					curp.end1 = curp.start1 + 1;
					newp.push_back(curp);
				} else {
					int right = next_free[assOffset];
					while (right<N-1 && next_free[right] != right) {
						right = next_free[right];
					}
					for (int j = assOffset; j <= right; j++) {
						maxival = std::max(maxival, taken[j]);
					}
					curp.start0 = lastStart;
					curp.end0 = maxival + 1;
					lastStart = curp.end0;
					curp.start1 = p.start1 + assOffset;
					curp.end1 = p.start1 + right + 1;
					newp.push_back(curp);
					i = p.start1 + right;
				}
			}

		}

		return true;
	}


	template<typename T>
	void simple_solve(const params &p, const T* hist1, const T* hist2, int* assignment, int* assNN, T &value) {

		int N = p.end1 - p.start1;
		std::vector<int> taken(N, -1);
		std::vector<int> ninj(N, 0);
		taken[assNN[p.start0] - p.start1] = p.start0;
		ninj[assNN[p.start0] - p.start1]++;

		std::vector<int> prev_free(p.end1 - p.start1);
		std::vector<int> next_free(p.end1 - p.start1);
		std::vector<T> cost_dontMove(p.end1 - p.start1, 0);
		std::vector<T> cost_moveLeft(p.end1 - p.start1, 0);
		int ass0 = assNN[p.start0];
		bool lastok = true;
		cost_dontMove[assNN[p.start0] - p.start1] = cost(hist1[p.start0], hist2[ass0]);
		cost_moveLeft[assNN[p.start0] - p.start1] = (ass0==0)?std::numeric_limits<T>::max():cost(hist1[p.start0], hist2[ass0 -1]);

		for (int i = 0; i < prev_free.size(); i++) {
			prev_free[i] = i;
			next_free[i] = i;
		}

		int first_right = assNN[p.start0 + 1] - p.start1, last_left = assNN[p.start0 + 1] - p.start1;
		for (int i = p.start0 + 1; i < p.end0; i++) {
			int ass = assNN[i];
			int assOffset = ass - p.start1;
			ninj[assOffset]++;
			if ((taken[assOffset]) < 0) {
				taken[assOffset] = i;
				first_right = assOffset;
				last_left = assOffset;
				cost_dontMove[assOffset] = cost(hist1[i], hist2[ass]);
				cost_moveLeft[assOffset] = (ass == 0) ? std::numeric_limits<T>::max() : cost(hist1[i], hist2[ass-1]);
				lastok = true;
			} else {

				T sumDontMove = 0;
				T sumMoveLeft = 0;
					int cur = prev_free[first_right] - 1;
					bool isok = true;
					while (cur >= 0) {
						sumDontMove += cost_dontMove[cur + 1];
						if (cost_moveLeft[cur + 1] < 0) isok = false;
						sumMoveLeft += cost_moveLeft[cur + 1];
						if (taken[cur] < 0 ) {
							break;
						} else {
							if (prev_free[cur] == cur) {
								cur--;
							}	else {
								cur = prev_free[cur]-1;
							}
						}
					}

					T cdM;
					T cmL;
					if (first_right >= N - 1)
						cdM = std::numeric_limits<T>::max();
					else
						cdM = sumDontMove + cost(hist1[i], hist2[p.start1 + first_right + 1]);
					if (cur < 0) {
						cmL = std::numeric_limits<T>::max();
					} else {
						if (isok)
							cmL = sumMoveLeft + cost(hist1[i], hist2[p.start1 + first_right]);
						else {
							cmL = 0;
							if (cur >= 0 && first_right < N - 1) {
								cmL = sumCosts(hist1, i - (first_right - cur), hist2, p.start1 + cur + 1 - 1, first_right-cur+1);
							}
						}

					}

					if (cmL < cdM || (first_right >= N - 1)) {
						last_left = std::max(0, cur);
						taken[last_left] = i;
						prev_free[assOffset] = prev_free[last_left];
						prev_free[first_right] = prev_free[last_left];
						next_free[last_left] = next_free[first_right];
						lastok = false;
						cost_dontMove[last_left] = cmL;
						cost_moveLeft[last_left] = -1; // we invalidate this value
					}
					else {
						first_right++;
						taken[first_right] = i;
						prev_free[first_right] = prev_free[cur+1];
						prev_free[assOffset] = prev_free[cur + 1];
						next_free[assOffset] = next_free[first_right];
						next_free[cur + 1] = next_free[first_right];
						cost_dontMove[cur + 1] = cdM;
						cost_moveLeft[cur + 1] = cmL;
						lastok = true;
					}
			}
		}

		int lastStart = p.start0;
		for (int i = p.start1; i < p.end1; i++) {
			int assOffset = i - p.start1;

			int maxival = taken[assOffset];

			if (taken[assOffset] >= 0) {
				params curp;
				if (next_free[assOffset] == assOffset) {
					curp.start1 = p.start1 + assOffset;
					curp.start0 = lastStart;
					lastStart++;
					curp.end0 = curp.start0 + 1;
					curp.end1 = curp.start1 + 1;
					for (int j = 0; j < curp.end0 - curp.start0; j++) {
						assignment[curp.start0 + j] = curp.start1 + j;
						value += cost(hist1[curp.start0 + j], hist2[curp.start1 + j]);
					}
				}
				else {
					int right = next_free[assOffset];
					while (right < N - 1 && next_free[right] != right) {
						right = next_free[right];
					}
					for (int j = assOffset; j <= right; j++) {
						maxival = std::max(maxival, taken[j]);
					}
					curp.start0 = lastStart;
					curp.end0 = maxival + 1;
					lastStart = curp.end0;
					curp.start1 = p.start1 + assOffset;
					curp.end1 = p.start1 + right + 1;
					for (int j = 0; j < curp.end0 - curp.start0; j++) {
						assignment[curp.start0 + j] = curp.start1 + j;
						value += cost(hist1[curp.start0 + j], hist2[curp.start1 + j]);
					}
					i = p.start1 + right;
				}
			}

		}
	}


	template<typename T>
	T transport1d(const T *hist1, const T* hist2, int M0, int N0, std::vector<int> &assignment, double* timingSplits = NULL) {

		assignment.resize(M0);
		params initp(0, M0, 0, N0, 0);
		T value = 0;

		// starts computing nearest neighbor match
		std::vector<int> assNN(M0);
		nearest_neighbor_match(hist1, hist2, initp, assNN);

		// computes the number of non-injective matches in an interval
		int nbbij = 0;
		for (int i = initp.start0 + 1; i < initp.end0; i++) {
			if (assNN[i] == assNN[i - 1]) nbbij++;
		}

		int ret1 = reduce_range(hist1, hist2, assignment, initp, value, &assNN[0], nbbij);
		if (ret1 == 1) return value;

		nearest_neighbor_match(hist1, hist2, initp, assNN); // since the bounds of the problem have changed, the NN maps has changed as well
		std::vector<params> splits;

		bool res = linear_time_decomposition(initp, hist1, hist2, &assNN[0], splits);

		std::vector<params > todo;
		if (res) {
			todo.reserve(splits.size());
			for (int i = 0; i < splits.size(); i++) {
				if (splits[i].end0 == splits[i].start0 + 1) { // we directly handle problems of size 1 here
					assignment[splits[i].start0] = assNN[splits[i].start0];
					value += cost(hist1[splits[i].start0], hist2[assNN[splits[i].start0]]);
				}
				else
					todo.push_back(splits[i]);
			}
		}
		else {
			todo.push_back(initp);
		}

#pragma omp parallel for schedule(dynamic)
		for (int i = 0; i < todo.size(); i++) {

			params p = todo[i];

			nearest_neighbor_match(hist1, hist2, p, assNN); // since the bounds of the problem have changed, the NN maps has changed as well
			int ret = handle_simple_cases(p, hist1, hist2, &assignment[0], &assNN[0], value);
			if (ret == 1) continue;

			int nbbij = 0;
			for (int i = p.start0 + 1; i < p.end0; i++) {
				if (assNN[i] == assNN[i - 1]) nbbij++;
			}

			ret = reduce_range(hist1, hist2, assignment, p, value, &assNN[0], nbbij);
			if (ret == 1) continue;


			ret = handle_simple_cases(p, hist1, hist2, &assignment[0], &assNN[0], value);
			if (ret == 1) continue;

			nearest_neighbor_match(hist1, hist2, p, assNN); // since the bounds of the problem have changed, the NN maps has changed as well
			simple_solve(p, hist1, hist2, &assignment[0], &assNN[0], value);
		}


		return value;
	}



	// advect = true : used for matching one distrib to another such as in our FIST algorithm : this function will advect cloud1 to cloud2 along a sliced wasserstein flow
	// advect = false : used to compute barycenters or sliced EMD (we don't perform any stochastic gradient descent then, this will merely compute the sliced wasserstein distance)

	template<int DIM, typename T>
	double correspondencesNd(std::vector<Point<DIM, T> > &cloud1, const std::vector<Point<DIM, T> > &cloud2, int niter, bool advect = false) {


		// we won't use the indices here for the moment nor the assignment, so if memory is an issue, we can remove the variables below
		std::vector<std::pair<T, int > > cloud1Idx(cloud1.size());
		std::vector<std::pair<T, int > > cloud2Idx(cloud2.size());


		Point<DIM, T> dir;
		std::vector<Point<DIM, T> > deltas(cloud1.size(), Point<DIM, T>());

		T* projHist1 = (T*)malloc_simd(cloud1.size() * sizeof(T), 32);
		T* projHist2 = (T*)malloc_simd(cloud2.size() * sizeof(T), 32);

		engine.seed(10);

		std::vector<int> corr1d;
		double d = 0;
		for (int iter = 0; iter < niter; iter++) { // number of random slices

			// random directions
			double n = 0;
			for (int i = 0; i < DIM; i+=2) {
				Point<2, double> randGauss = BoxMuller<double>();
				dir[i] = randGauss[0];
				n += dir[i] * dir[i];
				if (i < DIM-1) {
					dir[i+1] = randGauss[1];
					n += dir[i+1] * dir[i+1];
				}
			}
			n = std::sqrt(n);
			for (int i = 0; i < DIM; i++) {
				dir[i] /= n;
			}


			// sort according to projection on direction
			Projector<DIM, T> proj(dir);

			for (int i = 0; i < cloud1.size(); i++) {
				cloud1Idx[i] = std::make_pair(proj.proj(cloud1[i]), i);
			}

			for (int i = 0; i < cloud2.size(); i++) {
				cloud2Idx[i] = std::make_pair(proj.proj(cloud2[i]), i);
			}


			std::sort(cloud1Idx.begin(), cloud1Idx.end());
			std::sort(cloud2Idx.begin(), cloud2Idx.end());

			for (int i = 0; i < cloud1.size(); i++) {
				projHist1[i] = cloud1Idx[i].first;
			}
			for (int i = 0; i < cloud2.size(); i++) {
				projHist2[i] = cloud2Idx[i].first;
			}


			T emd = transport1d(projHist1, projHist2, cloud1.size(), cloud2.size(), corr1d);

			d += emd;


			if (advect) {
				for (int i = 0; i < cloud1Idx.size(); i++) {
					for (int j = 0; j < DIM; j++) {
						cloud1[cloud1Idx[i].second][j] += (projHist2[corr1d[i]] - projHist1[i])*dir[j];
					}
				}
			}
		}

		free_simd(projHist1);
		free_simd(projHist2);

		return d*2.0/niter;
	}


	template<int DIM, typename T>  // Mbary should be less than min_i(bary[i].size())
	void unbalanced_barycenter(int Mbary, int niters, int nslices, const std::vector<T> &weights, const std::vector< std::vector<Point<DIM, T> > > &points, std::vector<Point<DIM, T> > &barycenter) {


		auto start = std::chrono::system_clock::now();

		barycenter.resize(Mbary);

		for (int i = 0; i < Mbary; i++) {
			for (int j = 0; j < DIM; j++) {
				barycenter[i][j] = points[0][i][j]; //(rand() / (double)RAND_MAX)* 512.0; // //(rand() / (double)RAND_MAX)* 512.0;
			}
		}

		// a fixed set of slice directions across iterations (might be rotated)
		// random direction for DIM>2, else equispaced
		srand(10);
		engine.seed(10);
		std::vector<Point<DIM, T> > dirs(nslices);
		for (int slice = 0; slice < nslices; slice++) {
			if (DIM == 2) {
				double theta = slice*M_PI / nslices;
				dirs[slice][0] = cos(theta);
				dirs[slice][1] = sin(theta);
			} else {
				double n = 0;
				for (int i = 0; i < DIM; i+=2) {
					Point<2, double> randGauss = BoxMuller<double>();
					dirs[slice][i] = randGauss[0];
					n += dirs[slice][i] * dirs[slice][i];
					if (i < DIM - 1) {
						dirs[slice][i + 1] = randGauss[1];
						n += dirs[slice][i + 1] * dirs[slice][i + 1];
					}
				}
				n = std::sqrt(n);
				for (int i = 0; i < DIM; i++) {
					dirs[slice][i] /= n;
				}
			}
		}

		std::vector<T*> projHist1(omp_get_max_threads());
		for (int i=0; i< omp_get_max_threads() ; i++)
			projHist1[i] = (T*)malloc_simd(barycenter.size() * sizeof(T), 32);

		std::vector<std::vector<std::pair<T, int > > > cloud1Idx(omp_get_max_threads(), std::vector<std::pair<T, int > >(Mbary));
		std::vector<std::vector<std::pair<T, int > > > cloud2Idx(omp_get_max_threads());

		for (int iter = 0; iter < niters; iter++) {

			double d = 0;

			std::vector<Point<DIM, T> > offset(barycenter.size());
			std::vector<Point<DIM, T> > newbary = barycenter;
			for (int cloud = 0; cloud < points.size(); cloud++) {

#pragma omp parallel
				{
					int thread_num = omp_get_thread_num();
					cloud2Idx[thread_num].resize(points[cloud].size());
					T* projHist2 = (T*)malloc_simd(points[cloud].size() * sizeof(T), 32);
					std::vector<int> corr1d;
					double local_d = 0;

#pragma omp for schedule(dynamic)
					for (int slice = 0; slice < nslices; slice++) { // number of random slices

						Point<DIM, T> dir = dirs[slice];

						// sort according to projection on direction
						Projector<DIM, T> proj(dir);
						for (int i = 0; i < Mbary; i++) {
							cloud1Idx[thread_num][i] = std::make_pair(proj.proj(barycenter[i]), i);
						}
						for (int i = 0; i < points[cloud].size(); i++) {
							cloud2Idx[thread_num][i] = std::make_pair(proj.proj(points[cloud][i]), i);
						}
						std::sort(cloud1Idx[thread_num].begin(), cloud1Idx[thread_num].end());

						for (int i = 0; i < Mbary; i++) {
							projHist1[thread_num][i] = cloud1Idx[thread_num][i].first;
						}

						std::sort(cloud2Idx[thread_num].begin(), cloud2Idx[thread_num].end());

						for (int i = 0; i < points[cloud].size(); i++) {
							projHist2[i] = cloud2Idx[thread_num][i].first;
						}

						transport1d(projHist1[thread_num], projHist2, Mbary, points[cloud].size(), corr1d);


						for (int i = 0; i < corr1d.size(); i++) {
							local_d += weights[cloud] * cost(projHist1[thread_num][i], projHist2[corr1d[i]]);
						}
#pragma omp critical
						{
							for (int i = 0; i < cloud1Idx[thread_num].size(); i++) {
								int perm = cloud1Idx[thread_num][i].second;
								for (int j = 0; j < DIM; j++) {
									newbary[perm][j] += DIM * (weights[cloud] * (projHist2[corr1d[i]] - projHist1[thread_num][i])*dir[j]) / nslices;
								}
							}
						}
					}
					free_simd(projHist2);
#pragma omp atomic
					d += local_d;
				}

			}
			barycenter = newbary;
		}


		for (int i=0; i<omp_get_max_threads(); i++)
			free_simd(projHist1[i]);


	}


// transport-based ICP, using either a rigid transform (scaling = false) or similarity transform (scaling = true)
	template<int DIM, typename T>
	void fast_iterative_sliced_transport(int niters, int nslices, std::vector<Point<DIM, T> > &pointsSrc, const std::vector<Point<DIM, T> > &pointsDst, std::vector<double> &rot, std::vector<double> &trans, bool useScaling, double &scaling) {

		rot.resize(DIM*DIM);
		trans.resize(DIM);
		scaling = 1;
		std::fill(rot.begin(), rot.end(), 0);
		for (int i = 0; i < DIM; i++)
			rot[i*DIM + i] = 1;
		std::fill(trans.begin(), trans.end(), 0);

		for (int iter = 0; iter < niters; iter++) {
			std::vector<Point<DIM, T> > pointsSrcCopy(pointsSrc);
			correspondencesNd(pointsSrcCopy, pointsDst, nslices, true);

			Point<DIM, T> center1, center2;
			for (int i = 0; i < pointsSrc.size(); i++) {
				center1 += pointsSrc[i];
				center2 += pointsSrcCopy[i];  // pointsSrcCopy and pointsSrc have the same size
			}
			center1 *= (1.0 / pointsSrc.size());
			center2 *= (1.0 / pointsSrc.size());


			double cov[DIM*DIM];
			memset(cov, 0, DIM*DIM * sizeof(cov[0]));
			for (int i = 0; i < pointsSrc.size(); i++) {
				Point<DIM, T> p = pointsSrc[i] - center1;
				Point<DIM, T> q = pointsSrcCopy[i] - center2;
				for (int j = 0; j < DIM; j++) {
					for (int k = 0; k < DIM; k++) {
						cov[j * DIM + k] += q[j] * p[k];
					}
				}
			}
			cimg_library::CImg<double> mat(cov, DIM, DIM), S(1,DIM), U(DIM, DIM), V(DIM, DIM), orth(DIM,DIM), diag(DIM, DIM,1,1,0.0), rotM(DIM, DIM);
			mat.SVD(U, S, V, true, 100, 0.0);
			orth = U * V.get_transpose();
			double d = orth.det();
			for (int i = 0; i < DIM - 1; i++) {
				diag(i, i) = 1;
			}
			diag(DIM - 1, DIM - 1) = d;

			double scal = 1;
			if (useScaling) {
				double std = 0;
				for (int i = 0; i < pointsSrc.size(); i++) {
					std += (pointsSrc[i]-center1).norm2();
				}
				double s2 = 0;
				for (int i = 0; i < DIM; i++) {
					s2 += std::abs(S(0,i));
				}
				scal = s2 / std;
				scaling *= scal;
			}

			rotM = U * diag*V.get_transpose();

			cimg_library::CImg<double> rotG(const_cast<double*>(&rot[0]), DIM, DIM,1,1,true), transG(const_cast<double*>(&trans[0]), 1,DIM,1,1,true), C1(&center1[0], 1, DIM), C2(&center2[0], 1, DIM);
			rotG = rotM*rotG;
			transG = transG + C2 - C1;

			for (int i = 0; i < pointsSrc.size(); i++) {
				cimg_library::CImg<T> P(const_cast<T*>(&pointsSrc[i][0]), 1,DIM,1,1,true);
				P = scal*(rotM * (P - C1)) + C2;
			}
		}
		cimg_library::CImg<double> rotG(const_cast<double*>(&rot[0]), DIM, DIM, 1, 1, true), transG(const_cast<double*>(&trans[0]), 1, DIM, 1, 1, true);
		if (useScaling)
			transG = scaling*rotG * transG;
		else
			transG = rotG * transG;
	}



}; // end class
