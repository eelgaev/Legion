# stb_image

`stb_image.h` v2.30 is copied unchanged from
`llama.cpp/vendor/stb/stb_image.h` at llama.cpp commit
`f3363593f457169cdfbdfe22ea399b1cc139ba18`.
It is dual licensed under MIT or public domain (Unlicense); both license
texts are retained at the end of the header. Legion uses the MIT option.

The implementation is compiled by `build/stb.rs` from `src/vision/stb_image.c`.
PNG, JPEG, GIF and BMP use the same three-channel decoder as mtmd, preserving
its exact RGB bytes. WebP is decoded separately with the `image` crate;
llama.cpp parity is not required for WebP.
