pub fn main() {
        println!("cargo::rustc-link-lib=X11");
        println!("cargo::rustc-link-lib=c");
        println!("cargo::rustc-link-lib=vulkan");
}
