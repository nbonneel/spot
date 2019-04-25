#pragma once

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
