#include "src/gemm_test_lib_internal.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "src/gemm_test_lib.h"
#include "src/multi_gemm_lib.h"

namespace platforms_gpus::gemm_test::internal {
namespace {

TEST(SelectGemmInterfaceTest, ModernInterface) {
  auto result = SelectGemmInterface("half", 7.0);
  EXPECT_NE(nullptr, dynamic_cast<CudaCublasInterface*>(result.get()));
}

TEST(SelectGemmInterfaceTest, LegacySinglePrecision) {
  auto result = SelectGemmInterface("single", 3.0);
  EXPECT_NE(nullptr,
            dynamic_cast<LegacyCudaCublasInterface<float>*>(result.get()));
}

TEST(SelectGemmInterfaceTest, LegacyDoublePrecision) {
  auto result = SelectGemmInterface("double", 3.0);
  EXPECT_NE(nullptr,
            dynamic_cast<LegacyCudaCublasInterface<double>*>(result.get()));
}

TEST(SelectGemmInterfaceTest, Failure) {
  EXPECT_EQ(nullptr, SelectGemmInterface("half", 3.0));
}

TEST(GpuDataHandlerTest, DestructsSafelyWithoutAllocation) {
  GpuDataHandler<float, float> data_handler;
}
// TODO: When classes are templated on compute type, add them
// here and remove the ugly StringRep conversions below.
struct HalfInHalfOut {
  using Input = half_float::half;
  using Output = half_float::half;
  static constexpr absl::string_view Compute() { return "half"; }
};

struct HalfInFloatOut {
  using Input = half_float::half;
  using Output = float;
  static constexpr absl::string_view Compute() { return "single"; }
};

struct FloatInFloatOut {
  using Input = float;
  using Output = float;
  static constexpr absl::string_view Compute() { return "single"; }
};

struct DoubleInDoubleOut {
  using Input = double;
  using Output = double;
  static constexpr absl::string_view Compute() { return "double"; }
};

struct IntInIntOut {
  using Input = int8_t;
  using Output = int32_t;
  static constexpr absl::string_view Compute() { return "int32"; }
};

struct IntInFloatOut {
  using Input = int8_t;
  using Output = float;
  static constexpr absl::string_view Compute() { return "single"; }
};

template <typename T>
class GpuDataHandlerAllocationTest : public ::testing::Test {
 public:
  GpuDataHandlerAllocationTest() { CUDA_CHECK(cudaStreamCreate(&stream_)); }
  ~GpuDataHandlerAllocationTest() override {
    CUDA_CHECK(cudaStreamDestroy(stream_));
  }

 protected:
  cudaStream_t stream_;
};

TYPED_TEST_SUITE_P(GpuDataHandlerAllocationTest);

TYPED_TEST_P(GpuDataHandlerAllocationTest, SetupAndCleanup) {
  using InputPrecision = typename TypeParam::Input;
  using OutputPrecision = typename TypeParam::Output;
  cudaSetDevice(0);

  const int array_size = 2048;

  RandomMatrix<InputPrecision> input_a(array_size, array_size);
  RandomMatrix<InputPrecision> input_b(array_size, array_size);
  absl::BitGen bitgen;
  input_a.Initialize(&bitgen, 1, false);
  input_b.Initialize(&bitgen, 1, false);

  GpuDataHandler<InputPrecision, OutputPrecision> data_handler;
  data_handler.SetGpuId(0);
  data_handler.SetComputeType(TypeParam::Compute());
  data_handler.Initialize(&input_a, &input_b, this->stream_);
  EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(this->stream_));
}

REGISTER_TYPED_TEST_SUITE_P(GpuDataHandlerAllocationTest, SetupAndCleanup);

using MyTypes = ::testing::Types<HalfInHalfOut, HalfInFloatOut, FloatInFloatOut,
                                 DoubleInDoubleOut, IntInIntOut, IntInFloatOut>;

INSTANTIATE_TYPED_TEST_SUITE_P(MixedPrecisionGpuDataHandler,
                               GpuDataHandlerAllocationTest, MyTypes);

// Gets string representation for types, for context options population
// in typed tests.
template <typename T>
std::string StringRep();
template <>
std::string StringRep<float>() {
  return "single";
}
template <>
std::string StringRep<double>() {
  return "double";
}
template <>
std::string StringRep<half_float::half>() {
  return "half";
}
template <>
std::string StringRep<int8_t>() {
  return "int8";
}
template <>
std::string StringRep<int32_t>() {
  return "int32";
}

// Initializes and copies over host data through the handler. Note that host
// data is not retained beyond this function, but that's fine since we operate
// only on the GPU data.
template <typename InputPrecision, typename OutputPrecision>
void InitializeGPUDataForGEMM(
    const ContextOption& options, const cudaStream_t& stream,
    GpuDataHandler<InputPrecision, OutputPrecision>* data_handler) {
  const int array_size = options.dim_size_m;
  RandomMatrix<InputPrecision> input_a(array_size, array_size);
  RandomMatrix<InputPrecision> input_b(array_size, array_size);
  absl::BitGen bitgen;
  input_a.Initialize(&bitgen, 1, false);
  input_b.Initialize(&bitgen, 1, false);

  data_handler->SetComputeType(options.compute_type);
  data_handler->SetGpuId(0);
  data_handler->Initialize(&input_a, &input_b, stream);
  // Wait for data transfer to complete.
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

template <typename T>
class LegacyCublasTest : public ::testing::Test {
 public:
  LegacyCublasTest() {
    CUDA_CHECK(cudaStreamCreate(&stream_));
    CUBLAS_CHECK(cublasCreate(&cublas_handle_));
    CUBLAS_CHECK(cublasSetStream(cublas_handle_, stream_));
    CUBLAS_CHECK(
        cublasSetPointerMode(cublas_handle_, CUBLAS_POINTER_MODE_DEVICE));
    options_.dim_size_m = 2048;
    options_.dim_size_n = 2048;
    options_.dim_size_k = 2048;
    options_.transa = false;
    options_.transb = false;
    options_.algorithm = "gemm_algo_0";
  }

  ~LegacyCublasTest() override {
    CUDA_CHECK(cudaStreamDestroy(stream_));
    CUBLAS_CHECK(cublasDestroy(cublas_handle_));
  };

 protected:
  ContextOption options_;
  cudaStream_t stream_;
  cublasHandle_t cublas_handle_;
};

TYPED_TEST_SUITE_P(LegacyCublasTest);

TYPED_TEST_P(LegacyCublasTest, LegacyCublas) {
  using InputPrecision = typename TypeParam::Input;
  using OutputPrecision = typename TypeParam::Output;

  this->options_.data_type_in = StringRep<InputPrecision>();
  this->options_.data_type_out = StringRep<OutputPrecision>();
  this->options_.compute_type = StringRep<OutputPrecision>();

  GpuDataHandler<InputPrecision, OutputPrecision> data_handler;
  // Set up some random data on the GPU for us to compute on, we dont' care
  // about the actual contents.
  InitializeGPUDataForGEMM<InputPrecision, OutputPrecision>(
      this->options_, this->stream_, &data_handler);

  LegacyCudaCublasInterface<InputPrecision> cublas;
  CUBLAS_CHECK(cublas.MatrixMultiComputation(
      this->options_, this->cublas_handle_, data_handler.Alpha(),
      data_handler.InputA(), data_handler.InputB(), data_handler.Beta(),
      data_handler.Output()));
  CUDA_CHECK(cudaStreamSynchronize(this->stream_));
}

REGISTER_TYPED_TEST_SUITE_P(LegacyCublasTest, LegacyCublas);

using LegacyTypes = ::testing::Types<FloatInFloatOut, DoubleInDoubleOut>;
INSTANTIATE_TYPED_TEST_SUITE_P(LegacyCublasPrecisionsTest, LegacyCublasTest,
                               LegacyTypes);

// This class has the same logic as the Legacy cuBLAS test, but inheritance
// doesn't play well with typed tests.
template <typename T>
class ModernCublasTest : public ::testing::Test {
 public:
  ModernCublasTest() {
    CUDA_CHECK(cudaStreamCreate(&stream_));
    CUBLAS_CHECK(cublasCreate(&cublas_handle_));
    CUBLAS_CHECK(cublasSetStream(cublas_handle_, stream_));
    CUBLAS_CHECK(
        cublasSetPointerMode(cublas_handle_, CUBLAS_POINTER_MODE_DEVICE));
    options_.dim_size_m = 2048;
    options_.dim_size_n = 2048;
    options_.dim_size_k = 2048;
    options_.transa = false;
    options_.transb = false;
    options_.algorithm = "gemm_tensor_algo_default";
  }

  ~ModernCublasTest() override {
    CUDA_CHECK(cudaStreamDestroy(stream_));
    CUBLAS_CHECK(cublasDestroy(cublas_handle_));
  };

 protected:
  ContextOption options_;
  cudaStream_t stream_;
  cublasHandle_t cublas_handle_;
};

TYPED_TEST_SUITE_P(ModernCublasTest);

TYPED_TEST_P(ModernCublasTest, ModernCublas) {
  using InputPrecision = typename TypeParam::Input;
  using OutputPrecision = typename TypeParam::Output;

  this->options_.data_type_in = StringRep<InputPrecision>();
  this->options_.data_type_out = StringRep<OutputPrecision>();
  this->options_.compute_type = StringRep<OutputPrecision>();

  GpuDataHandler<InputPrecision, OutputPrecision> data_handler;
  // Set up some random data on the GPU for us to compute on, we dont' care
  // about the actual contents.
  InitializeGPUDataForGEMM<InputPrecision, OutputPrecision>(
      this->options_, this->stream_, &data_handler);

  CudaCublasInterface cublas;
  CUBLAS_CHECK(cublas.MatrixMultiComputation(
      this->options_, this->cublas_handle_, data_handler.Alpha(),
      data_handler.InputA(), data_handler.InputB(), data_handler.Beta(),
      data_handler.Output()));
  CUDA_CHECK(cudaStreamSynchronize(this->stream_));
}

REGISTER_TYPED_TEST_SUITE_P(ModernCublasTest, ModernCublas);

using ModernTypes =
    ::testing::Types<HalfInHalfOut, HalfInFloatOut, FloatInFloatOut,
                     DoubleInDoubleOut, IntInIntOut, IntInFloatOut>;

INSTANTIATE_TYPED_TEST_SUITE_P(ModernCublasPrecisionsTest, ModernCublasTest,
                               ModernTypes);

}  // namespace
}  // namespace platforms_gpus::gemm_test::internal
