# [Triton](https://github.com/triton-lang/triton) fork for Windows support

See `windows-fix` branch for the code.

Based on [wkpark](https://github.com/wkpark/triton/tree/windows-fix), [mantaionut](https://github.com/mantaionut/triton/tree/windows_support), [eaplatanios](https://github.com/eaplatanios/triton/tree/windows-fix), and more development in the community. Thank you all!

## Why?

Free software should run on non-free platforms, as per Richard Stallman.

## Progress

* Forked from the official main branch as of today
* Can build the package locally and jit Python functions
* When I run Flux or CogVideoX in ComfyUI on Windows, it's almost as fast as on WSL on the same machine (although the memory usage is hard to profile in WSL)
* Only MSVC is supported, from my experience it's much more stable than GCC and Clang when working with CUDA on Windows
* Only CUDA is supported, welcome help to support AMD
* TODO: Fix all tests
* TODO: Auto find the paths of MSVC, Windows SDK, and CUDA when jitting
* TODO: Build wheels
