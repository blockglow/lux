use core::{mem, ops};
use core::f32::consts::PI;
use core::ops::{Mul, Neg};

use crate::prelude::*;

pub struct Projection {
	pub a: f32,
	pub fov: f32,
	pub near: f32,
	pub far: f32,
}

impl Default for Projection {
	fn default() -> Self {
		Self {
			a: 1.0,
			fov: PI / 2.0,
			near: 1.0,
			far: 2.0,
		}
	}
}

impl From<Projection> for Matrix<4, 4> {
	fn from(value: Projection) -> Self {
		let Projection { a, fov: theta, near: n, far: f } = value;
		let tan_half_theta = (theta / 2.0).tan();
		let x_x = a.inverse() / tan_half_theta;
		let y_y = 1.0 / tan_half_theta;
		let z_z = f / (f - n);
		let z_w = - (n * f) / (f - n);
		let w_z = 1.0;
		let x = Vector([x_x, 0.0, 0.0, 0.0]);
		let y = Vector([0.0, y_y, 0.0, 0.0]);
		let z = Vector([0.0, 0.0, z_z, w_z]);
		let w = Vector([0.0, 0.0, z_w, 0.0]);
		Matrix::<4, 4>::from_axes((x,y,z,w))
	}
}

pub trait Math {
	fn add(self, rhs: Self) -> Self;
	fn mul(self, rhs: Self) -> Self;
}

pub trait Number: Math + Copy + PartialEq {
	fn zero() -> Self where Self: Sized;
	fn one() -> Self where Self: Sized;
	fn whole(num: usize) -> Self where Self: Sized;
}

pub trait Epsilon {
	fn epsilon() -> Self where Self: Sized;
}

impl Epsilon for f32 {
	fn epsilon() -> Self
	where
		Self: Sized
	{
		f32::EPSILON
	}
}

pub trait Power<Rhs> {
	fn pow(self, rhs: Rhs) -> Self where Self: Sized;
}

impl Power<f32> for f32 {
	fn pow(self, rhs: Self) -> Self
	where
		Self: Sized
	{
		unsafe { libm::powf(self, rhs) }
	}
}

pub trait Abs {
	fn abs(self) -> Self where Self: Sized;
}

impl Abs for f32 {
	fn abs(self) -> Self
	where
		Self: Sized
	{
		unsafe { libm::fabsf(self) }
	}
}

pub trait Sqrt {
	fn sqrt(self) -> Self where Self: Sized;
}

impl Sqrt for f32 {
	fn sqrt(self) -> Self
	where
		Self: Sized
	{
		unsafe { libm::sqrtf(self) }
	}
}

pub trait Trig {
	fn sin(self) -> Self where Self: Sized;
	fn cos(self) -> Self where Self: Sized;
	fn tan(self) -> Self where Self: Sized;
}

impl Trig for f32 {
	fn sin(self) -> Self
	where
		Self: Sized
	{
		unsafe { libm::sinf(self) }
	}
	fn cos(self) -> Self
	where
		Self: Sized
	{
		unsafe { libm::cosf(self) }
	}
	fn tan(self) -> Self
	where
		Self: Sized
	{
		unsafe { libm::tanf(self) }
	}
}

pub trait Negate {
	fn negate(self) -> Self;
}

impl Math for f32 {
	fn add(self, rhs: Self) -> Self {
		self + rhs
	}

	fn mul(self, rhs: Self) -> Self {
		self * rhs
	}
}

impl Negate for f32 {
	fn negate(self) -> Self {
		-self
	}
}

pub trait Inverse {
	fn inverse(self) -> Self;
}

impl Inverse for f32 {
	fn inverse(self) -> Self {
		1.0 / self
	}
}

impl Number for f32 {
	fn zero() -> Self
	where
		Self: Sized
	{
		0.0
	}

	fn one() -> Self
	where
		Self: Sized
	{
		1.0
	}

	fn whole(num: usize) -> Self {
		num as f32
	}
}

pub trait VectorMath<const N: usize, T: Number>: Math {
	fn scale(self, amount: T) -> Self;
}

impl<T: Number, const N: usize> Math for [T; N] {
	fn add(self, rhs: Self) -> Self {
		let mut data = unsafe { mem::zeroed::<Self>() };
		for (i, x) in self.into_iter().zip(rhs).map(|(x, y)| x.add(y)).enumerate() {
			data[i] = x;
		}
		data
	}

	fn mul(self, rhs: Self) -> Self {
		let mut data = unsafe { mem::zeroed::<Self>() };
		for (i, x) in self.into_iter().zip(rhs).map(|(x, y)| x.mul(y)).enumerate() {
			data[i] = x;
		}
		data
	}
}

impl<T: Number + Negate, const N: usize> Negate for [T; N] {
	fn negate(mut self) -> Self {
		for x in &mut self {
			*x = x.negate();
		}
		self
	}
}

impl<T: Number + Inverse, const N: usize> Inverse for [T; N] {
	fn inverse(mut self) -> Self {
		for x in &mut self {
			*x = x.inverse();
		}
		self
	}
}

impl<const N: usize, T: Number> VectorMath<N, T> for [T; N] {
	fn scale(mut self, amount: T) -> Self {
		for x in &mut self {
			*x = x.mul(amount);
		}
		self
	}
}

#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct Vector<const N: usize, T = f32>(pub [T; N]);

impl<const N: usize, T: Number + Negate> Neg for Vector<N, T> {
	type Output = Self;

	fn neg(mut self) -> Self::Output {
		self.0.iter_mut().for_each(|x| *x = x.negate());
		self
	}
}

impl<const N: usize, T: Number + Power<f32>> Power<f32> for Vector<N, T> {
	fn pow(mut self, rhs: f32) -> Self
	where
		Self: Sized
	{
		for x in self.0.iter_mut() {
			*x = x.pow(rhs);
		}
		self
	}
}

impl Vector<3, f32> {
	pub const X: Self = Self([1.0, 0.0, 0.0]);
	pub const Y: Self = Self([0.0, 1.0, 0.0]);
	pub const Z: Self = Self([0.0, 0.0, 1.0]);
}

impl<const N: usize, T: Number> Vector<N, T> {
	fn identity() -> Self {
		Self([T::zero(); N])
	}
}

impl<const N: usize, T: Number + Inverse + Sqrt> Vector<N, T> {
	pub fn magnitude(self) -> T {
		let mut x = T::zero();
		for (i, y) in self.0.into_iter().enumerate() {
			x = x.add(y.mul(y));
		}
		x.sqrt()
	}
	pub fn normalize(mut self) -> Option<Self> {
		let mag = self.magnitude();
		if mag == T::zero() {
			None?
		}
		let inv_mag = self.magnitude().inverse();
		self.0.iter_mut().for_each(|x| *x = x.mul(inv_mag));
		Some(self)
	}
}
impl<const N: usize, T: Number + Epsilon + Inverse + Sqrt + Abs + PartialOrd> Vector<N, T> {
	pub fn deadzone(self) -> Self {
		let deadzone = T::whole(5).inverse();
		if self.magnitude() < deadzone {
			Self::identity()
		} else {
			self.normalize().unwrap_or(Self::identity())
		}
	}
}
impl<T: Number + Negate> Vector<3, T> {
	pub fn dot(self, rhs: Self) -> T {
		self.0[0].mul(rhs.0[0]).add(self.0[1].mul(rhs.0[1])).add(self.0[2].mul(rhs.0[2]))
	}
	pub fn cross(self, rhs: Self) -> Self {
		Self(
			[self.0[1].mul( rhs.0[2]).add( rhs.0[1].mul( self.0[2]).negate()),
				self.0[2].mul(rhs.0[0]).add( rhs.0[2].mul( self.0[0]).negate()),
				self.0[0].mul(rhs.0[1]).add( rhs.0[0].mul( self.0[1]).negate())])

	}
}


impl<T: Number, const N: usize> Mul<T> for Vector<N, T> {
	type Output = Self;

	fn mul(mut self, rhs: T) -> Self::Output {
		self.0.iter_mut().for_each(|x| *x=x.mul(rhs));
		self
	}
}

impl<const N: usize, T: Copy> IntoIterator for Vector<N, T> {
	type Item = T;
	type IntoIter = VectorIter<N, T>;

	fn into_iter(self) -> Self::IntoIter {
		VectorIter {
			cursor: 0,
			vector: self,
		}
	}
}

pub struct VectorIter<const N: usize, T> {
	cursor: usize,
	vector: Vector<N, T>
}

impl<const N: usize, T: Copy> Iterator for VectorIter<N, T> {
	type Item = T;

	fn next(&mut self) -> Option<Self::Item> {
		if self.cursor >= N {
			None?
		}
		let res = Some(self.vector.0[self.cursor]);
		self.cursor += 1;
		res
	}
}

impl<const N: usize, T: Default + Copy> Default for Vector<N, T> {
	fn default() -> Self {
		Self([T::default(); N])
	}
}

impl<const N: usize, T: Number> core::ops::Add for Vector<N, T> {
	type Output = Self;

	fn add(self, rhs: Self) -> Self::Output {
		Self(self.0.add(rhs.0))
	}
}

impl<const N: usize, T: Number> core::ops::AddAssign for Vector<N, T> {
	fn add_assign(&mut self, rhs: Self) {
		*self = *self + rhs;
	}
}

impl<const N: usize, T: Number + Negate> core::ops::Sub for Vector<N, T> {
	type Output = Self;

	fn sub(self, rhs: Self) -> Self::Output {
		Self(self.0.add(rhs.0.negate()))
	}
}

impl<const N: usize, T: Number + Negate> core::ops::SubAssign for Vector<N, T> {
	fn sub_assign(&mut self, rhs: Self) {
		*self = *self - rhs;
	}
}

impl<const N: usize, T: Number> core::ops::Mul for Vector<N, T> {
	type Output = Self;

	fn mul(self, rhs: Self) -> Self::Output {
		Self(self.0.mul(rhs.0))
	}
}

impl<const N: usize, T: Number> core::ops::MulAssign for Vector<N, T> {
	fn mul_assign(&mut self, rhs: Self) {
		*self = *self * rhs;
	}
}

impl<const N: usize, T: Number + Inverse> core::ops::Div for Vector<N, T> {
	type Output = Self;

	fn div(self, rhs: Self) -> Self::Output {
		Self(self.0.mul(rhs.0.inverse()))
	}
}

impl<const N: usize, T: Number + Inverse> core::ops::DivAssign for Vector<N, T> {
	fn div_assign(&mut self, rhs: Self) {
		*self = *self / rhs;
	}
}

impl<const N: usize, const M: usize, T> MatrixMath<N, M> for [T; N * M] {

}

pub trait Axes<M> {
	fn to_matrix(self) -> M;
}

impl<T: Number> Axes<Matrix<4, 4, T>> for (Vector<4, T>, Vector<4, T>, Vector<4, T>, Vector<4, T>) {
	fn to_matrix(self) -> Matrix<4, 4, T> {
		let (x,y,z,w) = self;
		let mut m = Matrix::default();
		iter::chain(x, y)
			.chain(z)
			.chain(w)
			.enumerate()
			.for_each(|(i, x)| m.0[i] = x);
		m
	}
}

pub trait MatrixMath<const N: usize, const M: usize> {
}

#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct Matrix<const N: usize, const M: usize, T = f32>([T; N * M])
where
	[T; N * M]: MatrixMath<N, M>;


impl<const N: usize, const M: usize, const P: usize, T: Number + Default>
core::ops::Mul<Matrix<M, P, T>> for Matrix<N, M, T>
where
	[T; N * M]: MatrixMath<N, M>,
	[T; M * P]: MatrixMath<M, P>,
	[T; N * P]: MatrixMath<N, P>,
{
	type Output = Matrix<N, P, T>;

	fn mul(self, rhs: Matrix<M, P, T>) -> Self::Output {
		let mut result = Matrix::<N, P, T>::default();

		for i in 0..N {
			for j in 0..P {
				let mut sum = T::zero();
				for k in 0..M {
					// Convert 2D indexing to 1D for both matrices
					let self_idx = i * M + k;
					let rhs_idx = k * P + j;
					sum = sum.add(self.0[self_idx].mul(rhs.0[rhs_idx]));
				}
				result.0[i * P + j] = sum;
			}
		}
		result
	}
}

// Implement MulAssign for square matrices
impl<const N: usize, T: Number + Default + Copy>
core::ops::MulAssign for Matrix<N, N, T> where [T; N * N]: MatrixMath<N, N> {
	fn mul_assign(&mut self, rhs: Self) {
		*self = *self * rhs;
	}
}

impl<const N: usize, const M: usize, T: Default + Copy> Matrix<N, M, T> where [T; N*M]: MatrixMath<N, M>
{
	fn from_axes<A: Sized + Axes<Self>>(axes: A) -> Self
	where
		Self: Sized {
		axes.to_matrix()
	}
}

impl<const N: usize, const M: usize, T: Number + Copy> Default for Matrix<N, M, T> where [T; N*M]: MatrixMath<N, M> {
	fn default() -> Self {
		Self([T::zero(); N * M])
	}
}

impl<T: Number + Inverse + Negate + Sqrt + fmt::Debug> Matrix<4, 4, T> {
	pub fn normalize(self) -> Self {
		let mut result = [T::zero(); 16];

		// Process each column
		for col in 0..4 {
			// Calculate column length (magnitude)
			let mut length = T::zero();
			for row in 0..4 {
				length = length.add(self.0[col * 4 + row].mul(self.0[col * 4 + row]));
			}
			length = length.sqrt();

			// Normalize column if length is not zero
			if length != T::zero() {
				let inv_length = length.inverse();
				for row in 0..4 {
					result[col * 4 + row] = self.0[col * 4 + row].mul(inv_length);
				}
			} else {
				// Copy column unchanged if length is zero
				for row in 0..4 {
					result[col * 4 + row] = self.0[col * 4 + row];
				}
			}
		}

		Self(result)
	}
	fn identity() -> Self {
		let mut x = Self::default();
		for i in (0..4).map(|i| 4*i + i) {
			x.0[i] = T::one();
		}
		x
	}
	pub fn from_translation(translation: Vector<3, T>) -> Self {
		let mut w = [T::one(); 4];
		translation.into_iter().enumerate().for_each(|(i,x)| w[i] = x);
		let mut m = Self::identity();
		w.into_iter().enumerate().map(|(i, x)| (i + 12, x)).for_each(|(i,x)| m.0[i] = x);
		m
	}
	pub fn inverse(self) -> Option<Self> {
		// Create augmented 4x4 matrix and identity matrix
		let mut m = self.0;
		let mut inv = [T::zero(); 16];
		for i in 0..4 {
			inv[i * 4 + i] = T::one();
		}

		// For each column
		for j in 0..4 {
			// Find pivot
			let mut pivot_val = m[j * 4 + j];
			let mut pivot_row = j;

			// Look for largest pivot in current column
			for i in j..4 {
				let val = m[j * 4 + i];
				if val != T::zero() {
					pivot_val = val;
					pivot_row = i;
					break;
				}
			}

			if pivot_val == T::zero() {
				return None;
			}

			// Swap rows if needed
			if pivot_row != j {
				for k in 0..4 {
					// Swap elements in original matrix
					let tmp = m[k * 4 + j];
					m[k * 4 + j] = m[k * 4 + pivot_row];
					m[k * 4 + pivot_row] = tmp;

					// Swap elements in inverse matrix
					let tmp = inv[k * 4 + j];
					inv[k * 4 + j] = inv[k * 4 + pivot_row];
					inv[k * 4 + pivot_row] = tmp;
				}
			}

			// Scale pivot row
			let pivot_inv = m[j * 4 + j].inverse();
			for k in 0..4 {
				m[k * 4 + j] = m[k * 4 + j].mul(pivot_inv);
				inv[k * 4 + j] = inv[k * 4 + j].mul(pivot_inv);
			}

			// Eliminate column
			for i in 0..4 {
				if i != j {
					let factor = m[j * 4 + i];
					for k in 0..4 {
						m[k * 4 + i] = m[k * 4 + i].add(m[k * 4 + j].mul(factor).negate());
						inv[k * 4 + i] = inv[k * 4 + i].add(inv[k * 4 + j].mul(factor).negate());
					}
				}
			}
		}

		Some(Self(inv))
	}
	pub fn transpose(&self) -> Self {

		let mut result = [T::zero(); 16];
		for row in 0..4 {
			for col in 0..4 {
				result[col * 4 + row] = self.0[row * 4 + col];
			}
		}
		Self(result)
	}
}

#[derive(Clone, Copy, Debug)]
pub struct Quaternion {
	imaginary: Vector<3>,
	real: f32,
}

impl Default for Quaternion {
	fn default() -> Self {
		Self {
			imaginary: default(),
			real: 1.0
		}
	}
}

impl ops::Add for Quaternion {
	type Output = Self;

	fn add(self, rhs: Self) -> Self::Output {
		Self {
			imaginary: self.imaginary + rhs.imaginary,
			real: self.real + rhs.real
		}
	}
}

impl ops::AddAssign for Quaternion {
	fn add_assign(&mut self, rhs: Self) {
		*self = *self + rhs;
	}
}

impl Quaternion {
	pub fn to_matrix(self) -> Matrix<4, 4> {
		let Self {
			real: w,
			imaginary: Vector([x, y, z]),
		} = self.normalize().unwrap();
		let x2 = x + x;
		let y2 = y + y;
		let z2 = z + z;
		let xx = x * x2;
		let xy = x * y2;
		let xz = x * z2;
		let yy = y * y2;
		let yz = y * z2;
		let zz = z * z2;
		let wx = w * x2;
		let wy = w * y2;
		let wz = w * z2;

		let _x = Vector([1.0 - (yy + zz), xy + wz, xz - wy, 0.0]);
		let _y = Vector([xy - wz, 1.0 - (xx + zz), yz + wx, 0.0]);
		let _z = Vector([xz + wy, yz - wx, 1.0 - (xx + yy), 0.0]);
		let _w = Vector([0.,0.,0.,1.]);
		Matrix::from_axes((_x,_y,_z,_w)).normalize()
	}
	pub fn normalize(self) -> Option<Self> {
		// Calculate magnitude (norm) of quaternion
		let norm = unsafe {
			(
				self.real * self.real +
					self.imaginary.magnitude()
			).sqrt()
		};

		// Check if magnitude is zero (or very close to zero)
		if norm == 0.0 {
			return None;
		}

		// Calculate inverse of norm
		let inv_norm = 1.0 / norm;

		// Scale all components by inverse norm
		Some(Self {
			real: self.real * inv_norm,
			imaginary: self.imaginary * inv_norm,
		})
	}

	pub fn from_angle_axis(angle: f32, axis: Vector<3>) -> Option<Self> {
		let axis = axis.normalize()?;

		let half_angle = angle / 2.0;
		let sin_half = half_angle.sin();

		Some(Self {
			imaginary: axis * sin_half,
			real: half_angle.cos()
		}).map(Quaternion::normalize).flatten()
	}

	pub fn pure() -> Self {
		Self { real: 0., imaginary: default() }
	}

	pub fn from_euler_angles(angles: Vector<3>) -> Option<Self> {
		let roll = angles.0[0];
		let pitch = angles.0[1];
		let yaw = angles.0[2];
		// Define the basis vectors for each axis
		let x_axis = Vector([1.0, 0.0, 0.0]);
		let y_axis = Vector([0.0, 1.0, 0.0]);
		let z_axis = Vector([0.0, 0.0, 1.0]);

		// Create quaternions for each axis rotation
		let roll_quat = Self::from_angle_axis(roll, x_axis)?;   // Roll around X
		let pitch_quat = Self::from_angle_axis(pitch, y_axis)?; // Pitch around Y
		let yaw_quat = Self::from_angle_axis(yaw, z_axis)?;     // Yaw around Z

		// Combine rotations: first yaw, then pitch, then roll
		// Note: Quaternion multiplication is not commutative
		Some(roll_quat * pitch_quat * yaw_quat)
	}
	pub fn identity() -> Self {
		Self::default()
	}
}

// Implement the Mul trait for more ergonomic usage
impl core::ops::Mul for Quaternion {
	type Output = Self;

	fn mul(self, rhs: Self) -> Self::Output {
		let w1 = self.real;
		let w2 = rhs.real;
		let v1 = self.imaginary;
		let v2 = rhs.imaginary;

		// Real part: w1*w2 - v1·v2
		let real = w1 * w2 - v1.dot(v2);

		// Imaginary part: w1*v2 + w2*v1 + v1×v2
		let imaginary = v2 * w1 + v1 * w2 + v1.cross(v2);


		Self::normalize(Self { real, imaginary }).unwrap_or(Self::identity())
	}
}

impl core::ops::Mul<Vector<3>> for Quaternion {
	type Output = Vector<3>;

	fn mul(self, v: Vector<3>) -> Self::Output {
		// Extract quaternion components
		let w = self.real;
		let q = self.imaginary;

		// Calculate vector rotation using the quaternion formula:
		// v' = v + 2w(q × v) + 2(q × (q × v))

		// First cross product: q × v
		let qcv = q.cross(v);

		// Second cross product: q × (q × v)
		let qcqcv = q.cross(qcv);

		// Combine terms: v + 2w(q × v) + 2(q × (q × v))
		v + qcv * w * 2.0
			+ qcqcv * 2.0
	}
}// Implement MulAssign for in-place multiplication
impl core::ops::MulAssign for Quaternion {
	fn mul_assign(&mut self, rhs: Self) {
		*self = self.mul(rhs);
	}
}