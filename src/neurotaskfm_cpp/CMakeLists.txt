find_package(Torch REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgcodecs imgproc videoio)
find_package(HDF5 REQUIRED COMPONENTS C CXX)
find_package(yaml-cpp REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(Boost 1.74 REQUIRED COMPONENTS system)
find_package(Threads REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(NIFTI REQUIRED niftiio)
separate_arguments(TORCH_CXX_FLAGS_LIST NATIVE_COMMAND "${TORCH_CXX_FLAGS}")

set(NTFM_CPP_HEADERS
  include/neurotaskfm/cli.h
  include/neurotaskfm/config.h
  include/neurotaskfm/data.h
  include/neurotaskfm/imaging_tools.h
  include/neurotaskfm/losses.h
  include/neurotaskfm/model.h
  include/neurotaskfm/runtime.h
  include/neurotaskfm/tools.h
)

set(NTFM_CPP_SOURCES
  src/cli.cpp
  src/config.cpp
  src/data.cpp
  src/imaging_tools.cpp
  src/kspace_cuda.cu
  src/losses.cpp
  src/model.cpp
  src/model_encoders.cpp
  src/model_heads.cpp
  src/model_primitives.cpp
  src/runtime.cpp
  src/inference.cpp
  src/analysis.cpp
  src/tools.cpp
  src/web_server.cpp
)

add_library(neurotaskfm STATIC ${NTFM_CPP_HEADERS} ${NTFM_CPP_SOURCES})
target_include_directories(neurotaskfm PUBLIC include ${HDF5_INCLUDE_DIRS} ${NIFTI_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
target_link_libraries(neurotaskfm
  PUBLIC
    ${TORCH_LIBRARIES}
    CUDA::cufft
    ${OpenCV_LIBS}
    ${HDF5_CXX_LIBRARIES}
    ${HDF5_LIBRARIES}
    ${NIFTI_LIBRARIES}
    yaml-cpp
    nlohmann_json::nlohmann_json
    Boost::system
    Threads::Threads
    neurocompiler
)
target_compile_options(neurotaskfm PRIVATE
  ${TORCH_CXX_FLAGS_LIST}
  $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic;-Wconversion>
)
target_compile_definitions(neurotaskfm PUBLIC NTFM_CUDA_ONLY=1 TORCH_CUDA=1)

function(ntfm_executable target source)
  add_executable(${target} ${source})
  target_link_libraries(${target} PRIVATE neurotaskfm)
  set_target_properties(${target} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
  install(TARGETS ${target} RUNTIME DESTINATION bin)
endfunction()

ntfm_executable(ntfm-train src/main_train.cpp)
ntfm_executable(ntfm-infer src/main_infer.cpp)
ntfm_executable(ntfm-personalize src/main_personalize.cpp)
ntfm_executable(ntfm-tool src/main_tools.cpp)
ntfm_executable(neurotask-web src/main_web.cpp)

ntfm_executable(ntfm-probe src/main_analysis.cpp)
target_compile_definitions(ntfm-probe PRIVATE NTFM_ANALYSIS_COMMAND="probe")
ntfm_executable(ntfm-evaluate-cross-state src/main_analysis.cpp)
target_compile_definitions(ntfm-evaluate-cross-state PRIVATE NTFM_ANALYSIS_COMMAND="cross-state")
ntfm_executable(ntfm-evaluate src/main_analysis.cpp)
target_compile_definitions(ntfm-evaluate PRIVATE NTFM_ANALYSIS_COMMAND="evaluate")

set_property(TARGET neurotaskfm PROPERTY CXX_STANDARD 20)
set_property(TARGET neurotaskfm PROPERTY CUDA_STANDARD 20)
set_property(TARGET neurotaskfm PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET neurotaskfm PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
install(TARGETS neurotaskfm ARCHIVE DESTINATION lib)
install(DIRECTORY include/ DESTINATION include)
