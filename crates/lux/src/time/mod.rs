#[cfg(target_os = "linux")]
pub use linux::*;

pub trait Time {
	fn now() -> Self;
}

pub trait Span {
	fn as_secs(self) -> f32;
}

#[cfg(target_os = "linux")]
mod linux {
	use crate::prelude::*;

	use super::*;

	const SEC_TO_NANO: u128 = 1e+9 as u128;

	#[derive(Clone, Copy)]
	pub struct Instant(u128);

	impl Time for Instant {
		fn now() -> Self {
			let mut timespec = unsafe { mem::zeroed::<libc::timespec>() };
			unsafe { libc::clock_gettime(libc::CLOCK_REALTIME, &mut timespec) };
			Self(timespec.tv_sec as u128 * SEC_TO_NANO + timespec.tv_nsec as u128)
		}
	}

	impl ops::Sub for Instant {
		type Output = Duration;

		fn sub(self, rhs: Self) -> Self::Output {
			Duration(self.0.saturating_sub(rhs.0))
		}
	}

	impl ops::Add<Duration> for Instant {
		type Output = Self;

		fn add(mut self, rhs: Duration) -> Self::Output {
			self.0 += rhs.0;
			self
		}
	}

	#[derive(Clone, Copy)]
	pub struct Duration(u128);

	impl Span for Duration {
		fn as_secs(self) -> f32 {
			self.0 as f32 / SEC_TO_NANO as f32
		}
	}
}