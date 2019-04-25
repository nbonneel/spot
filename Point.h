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

template<int DIM, typename T>
class Point {
public:
	Point<DIM, T>() {
		memset(coords, 0, DIM * sizeof(T));
	}
	T operator[](int i) const { return coords[i]; };
	T& operator[](int i) { return coords[i]; };

	void operator+=(const Point<DIM, T>& rhs) {
		for (int i = 0; i < DIM; i++) {
			coords[i] += rhs[i];
		}
	}
	void operator*=(T rhs) {
		for (int i = 0; i < DIM; i++) {
			coords[i] *= rhs;
		}
	}
	void operator-=(const Point<DIM, T>& rhs) {
		for (int i = 0; i < DIM; i++) {
			coords[i] -= rhs[i];
		}
	}
	bool operator==(const Point<DIM, T>& rhs) {		
		for (int i = 0; i < DIM; i++) {
			if (coords[i] != rhs[i]) return false;
		}
		return true;
	}
	T norm2() {
		T s = 0;
		for (int i = 0; i < DIM; i++) {
			s += coords[i] * coords[i];
		}
		return s;
	}
	T coords[DIM];
};


template<int DIM, typename T>
T dot(const Point<DIM, T> &p, const Point<DIM, T> &q) {
	T s = 0;
	for (int i = 0; i < DIM; i++)
		s += p[i] * q[i];
	return s;
}

template<int DIM, typename T>
Point<DIM, T> operator*(const Point<DIM, T> &p, T f) {
	Point<DIM, T> r;
	for (int i = 0; i < DIM; i++) {
		r[i] = p[i] * f;
	}
	return r;
}

template<int DIM, typename T>
Point<DIM, T> operator+(const Point<DIM, T> &p, const Point<DIM, T> &q) {
	Point<DIM, T> r;
	for (int i = 0; i < DIM; i++) {
		r[i] = p[i] + q[i];
	}
	return r;
}
template<int DIM, typename T>
Point<DIM, T> operator-(const Point<DIM, T> &p, const Point<DIM, T> &q) {
	Point<DIM, T> r;
	for (int i = 0; i < DIM; i++) {
		r[i] = p[i] - q[i];
	}
	return r;
}
