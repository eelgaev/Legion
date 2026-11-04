pub fn build() {
    cc::Build::new()
        .file("src/vision/stb_image.c")
        .include("third_party/stb")
        .compile("legion_stb_image");
    println!("cargo:rerun-if-changed=src/vision/stb_image.c");
    println!("cargo:rerun-if-changed=third_party/stb/stb_image.h");
}
