use crate::prelude::*;

pub trait Controller {
	fn connect(num: usize) -> Option<Self> where Self: Sized;
	fn poll(&mut self);
	fn get_axis(&self, num: Axis) -> f32;
	fn get_button(&self, num: Button) -> bool;
	fn get_button_axis(&self, pos: Button, neg: Button) -> f32 {
		let mut input = 0.0;

		if self.get_button(pos) {
			input += 1.0;
		}

		if self.get_button(neg) {
			input -= 1.0;
		}

		input
	}
}

pub fn get_platform_controller(num: usize) -> Box<dyn Controller> {
	#[cfg(target_os = "linux")]
	Box::new(linux::Controller::connect(num).unwrap())
}

#[derive(Clone, Copy)]
pub struct Button(u8);
#[derive(Clone, Copy)]
pub struct Axis(u8);

pub struct Xbox;

impl Xbox {
	pub const L_STICK_X: Axis = Axis(0);
	pub const L_STICK_Y: Axis = Axis(1);
	pub const L_TRIGGER: Axis = Axis(2);
	pub const R_STICK_X: Axis = Axis(3);
	pub const R_STICK_Y: Axis = Axis(4);
	pub const R_TRIGGER: Axis = Axis(5);
	pub const D_PAD_X: Axis = Axis(6);
	pub const D_PAD_Y: Axis = Axis(7);

	pub const A: Button = Button(0);
	pub const B: Button = Button(1);
	pub const X: Button = Button(2);
	pub const Y: Button = Button(3);
	
	pub const L_BUMPER: Button = Button(4);
	pub const R_BUMPER: Button = Button(5);

	pub const MENU_A: Button = Button(6);
	pub const MENU_B: Button = Button(7);
	pub const L_STICK: Button = Button(9);
	pub const R_STICK: Button = Button(10);
}

#[cfg(target_os = "linux")]
mod linux {
	use core::str::FromStr;

	use libc::{O_NONBLOCK, O_RDONLY};

	use crate::prelude::*;

	use super::{Axis, Button};
	use super::Controller as BaseController;

	pub const JS_EVENT_BUTTON: u8 = 0x01;
	pub const JS_EVENT_AXIS: u8 = 0x02;
	pub const JS_EVENT_INIT: u8 = 0x80;

	pub type JsAxis = u8;
	pub type Value = i16;

	#[repr(C)]
	pub struct JsEvent {
		pub time: u32,
		pub value: i16,
		pub type_: u8,
		pub number: u8,
	}

	pub struct Controller {
		fd: c_int,
		axis: BTreeMap<JsAxis, Value>,
		button: BTreeMap<JsAxis, Value>,
	}

	impl BaseController for Controller {
		fn connect(num: usize) -> Option<Self>
		where
			Self: Sized
		{
			let devname = CString::from_str(&format!("/dev/input/js{num}")).unwrap();
			let fd = unsafe { libc::open(devname.as_ptr() as *const _, O_RDONLY | O_NONBLOCK) };
			Some(Controller { fd, axis: BTreeMap::new(), button: BTreeMap::new() })
		}

		fn poll(&mut self) {
			let mut ev = unsafe { mem::zeroed::<JsEvent>() };
			while unsafe { libc::read(self.fd, &mut ev as *mut _ as *mut _, mem::size_of::<JsEvent>())} > 0 {
				ev.type_ &= !JS_EVENT_INIT;
				dbg!(ev.number);
				if ev.type_ & JS_EVENT_AXIS != 0 {
					self.axis.insert(ev.number, ev.value);
				}
				if ev.type_ & JS_EVENT_BUTTON != 0 {
					self.button.insert(ev.number, ev.value);
				}
			}
		}

		fn get_axis(&self, num: Axis) -> f32 {
			(self.axis.get(&num.0).copied().unwrap_or(0) as f32 / i16::MAX as f32)
		}
		
		fn get_button(&self, num: Button) -> bool {
			self.button.get(&num.0).copied().unwrap_or(0) != 0
		}
	}
}