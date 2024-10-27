use core::ffi::c_void;
use core::ptr;

use libc::c_char;

use crate::gfx::{Surface, SurfaceHandle};

pub struct Window {
    display: *const c_void,
    window: *const c_void,
}

impl Surface for Window {
    fn open() -> Self
    where
        Self: Sized,
    {
        let (display, window) = unsafe {
            let display = XOpenDisplay(ptr::null());

            let screen = XDefaultScreen(display);

            let window = XCreateSimpleWindow(
                display,
                XRootWindow(display, screen),
                10,
                10,
                1920,
                1080,
                1,
                XBlackPixel(display, screen),
                XWhitePixel(display, screen),
            );
            XMapWindow(display, window);

            (display, window)
        };
        Self { display, window }
    }

    fn update(&self) {
        let mut buf = crate::core_alloc::vec![0u8; 65535];
        unsafe {
            XNextEvent(self.display, buf.as_mut_ptr() as *mut _);
        }
    }

    fn size(&self) -> (u32, u32) {
        unsafe {
            let screen = XScreenOfDisplay(self.display, 0);
            (
                XWidthOfScreen(screen) as u32,
                XHeightOfScreen(screen) as u32,
            )
        }
    }

    fn handle(&self) -> SurfaceHandle {
        SurfaceHandle::Linux {
            display: self.display,
            window: self.window,
        }
    }
}

extern "C" {
    fn XScreenOfDisplay(display: *const c_void, num: u32) -> *const c_void;
    fn XWidthOfScreen(screen: *const c_void) -> i32;
    fn XHeightOfScreen(screen: *const c_void) -> i32;
    fn XOpenDisplay(display_name: *const c_char) -> *const c_void;
    fn XBlackPixel(display: *const c_void, screen: i32) -> u64;
    fn XWhitePixel(display: *const c_void, screen: i32) -> u64;
    fn XDefaultScreen(display: *const c_void) -> i32;
    fn XRootWindow(display: *const c_void, screen: i32) -> *const c_void;
    fn XMapWindow(display: *const c_void, window: *const c_void);
    fn XCreateSimpleWindow(
        display: *const c_void,
        root: *const c_void,
        x: i32,
        y: i32,
        width: u32,
        height: u32,
        border_width: u32,
        border: u64,
        background: u64,
    ) -> *const c_void;
    fn XNextEvent(display: *const c_void, event: *mut c_void);
}
