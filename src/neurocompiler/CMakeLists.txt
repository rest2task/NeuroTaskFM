find_package(CUDAToolkit REQUIRED)
find_package(HDF5 REQUIRED COMPONENTS C CXX)
find_package(DCMTK REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(NIFTI REQUIRED niftiio)

set(NTFM_HEADERS
  include/neurotask/cuda_check.h
  include/neurotask/device_buffer.h
  include/neurotask/types.h
  include/neurotask/geometry.h
  include/neurotask/nifti_reader.h
  include/neurotask/dicom_tokenizer.h
  include/neurotask/compiler_artifacts.h
  include/neurotask/feature_pack.h
  include/neurotask/operator_graph.h
  include/neurotask/operators.h
)

set(NTFM_SOURCES
  src/nifti_reader.cpp
  src/geometry.cpp
  src/dicom_tokenizer.cpp
  src/compiler_artifacts.cpp
  src/feature_pack.cpp
  src/operator_graph.cpp
  src/operators.cpp
  src/cuda/normalize.cu
  src/cuda/resample.cu
  src/cuda/registration.cu
  src/cuda/temporal.cu
  src/cuda/features.cu
  src/cuda/slices.cu
)

add_library(neurocompiler STATIC ${NTFM_HEADERS} ${NTFM_SOURCES})
target_include_directories(neurocompiler PUBLIC include ${HDF5_INCLUDE_DIRS} ${DCMTK_INCLUDE_DIRS} ${NIFTI_INCLUDE_DIRS})
target_link_libraries(neurocompiler PUBLIC CUDA::cudart CUDA::cublas CUDA::cufft CUDA::cusolver CUDA::cusparse ${HDF5_CXX_LIBRARIES} ${HDF5_LIBRARIES} ${DCMTK_LIBRARIES} ${NIFTI_LIBRARIES} yaml-cpp nlohmann_json::nlohmann_json)
target_compile_definitions(neurocompiler PUBLIC NTFM_CUDA_ONLY=1)
set_target_properties(neurocompiler PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(neurocompile src/main_neurocompile.cpp)
target_link_libraries(neurocompile PRIVATE neurocompiler)

install(TARGETS neurocompiler neurocompile ARCHIVE DESTINATION lib RUNTIME DESTINATION bin)
install(DIRECTORY include/ DESTINATION include)

set_target_properties(neurocompiler PROPERTIES ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set_target_properties(neurocompile PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
