add_triton_library(TritonToTritonGPU
    RelayoutTritonGPU.cpp
    TritonGPUConversion.cpp
    TritonToTritonGPUPass.cpp

    DEPENDS
    TritonConversionPassIncGen

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRPass
    MLIRTransforms
    TritonIR
    TritonGPUIR
)

if (TRITON_BUILD_PROTON)
    target_link_libraries(TritonToTritonGPU PUBLIC ProtonIR)
endif()