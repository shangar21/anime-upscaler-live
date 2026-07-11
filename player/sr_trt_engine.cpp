#include "sr_trt_engine.hpp"
#include <fstream>
#include <vector>
#include <iostream>

using namespace nvinfer1;

namespace {
class Logger : public ILogger {
    void log(Severity severity, const char *msg) noexcept override {
        if (severity <= Severity::kWARNING) std::cerr << "[TRT] " << msg << "\n";
    }
};
Logger gLogger;
}

struct SrTrtEngine::Impl {
    IRuntime *runtime = nullptr;
    ICudaEngine *engine = nullptr;
    IExecutionContext *context = nullptr;
    std::string inputName, outputName;
};

bool SrTrtEngine::load(const std::string &enginePath) {
    impl_ = new Impl();

    std::ifstream f(enginePath, std::ios::binary);
    if (!f) {
        std::cerr << "SrTrtEngine: could not open " << enginePath << "\n";
        return false;
    }
    std::vector<char> data((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());

    impl_->runtime = createInferRuntime(gLogger);
    impl_->engine  = impl_->runtime->deserializeCudaEngine(data.data(), data.size());
    if (!impl_->engine) {
        std::cerr << "SrTrtEngine: deserialization failed for " << enginePath << "\n";
        return false;
    }
    impl_->context = impl_->engine->createExecutionContext();
    if (!impl_->context) return false;

    // Find input and output tensor names.
    int nIO = impl_->engine->getNbIOTensors();
    for (int i = 0; i < nIO; ++i) {
        const char *name = impl_->engine->getIOTensorName(i);
        if (impl_->engine->getTensorIOMode(name) == TensorIOMode::kINPUT)
            impl_->inputName = name;
        else
            impl_->outputName = name;
    }
    if (impl_->inputName.empty() || impl_->outputName.empty()) {
        std::cerr << "SrTrtEngine: could not locate input/output tensors\n";
        return false;
    }

    // Derive geometry from the optimization profile (input) and engine
    // binding (output). getProfileShape is input-only in TRT 10.x -- the
    // output shape must be read from the engine's tensor shape directly.
    auto inProfileOpt = impl_->engine->getProfileShape(
        impl_->inputName.c_str(), 0, OptProfileSelector::kOPT);
    auto inProfileMax = impl_->engine->getProfileShape(
        impl_->inputName.c_str(), 0, OptProfileSelector::kMAX);

    // Set the input shape on the context so we can query the resulting
    // output shape -- TRT computes output dims from the set input shape.
    impl_->context->setInputShape(
        impl_->inputName.c_str(),
        Dims4{inProfileOpt.d[0], inProfileOpt.d[1],
              inProfileOpt.d[2], inProfileOpt.d[3]});

    auto outShape = impl_->context->getTensorShape(impl_->outputName.c_str());

    kTileLR_   = inProfileOpt.d[2];   // LR spatial dim at opt shape
    kTileHR_   = outShape.d[2];       // HR spatial dim from actual binding
    kMaxBatch_ = inProfileMax.d[0];   // max batch from profile

    if (kTileLR_ <= 0 || kTileHR_ <= 0 || kTileHR_ % kTileLR_ != 0) {
        std::cerr << "SrTrtEngine: unexpected tile dims LR=" << kTileLR_
                  << " HR=" << kTileHR_ << " — not a valid SR engine?\n";
        return false;
    }
    kScale_ = kTileHR_ / kTileLR_;

    std::cerr << "SrTrtEngine: loaded " << enginePath << "\n"
              << "  scale=" << kScale_ << "  LR tile=" << kTileLR_
              << "  HR tile=" << kTileHR_ << "  maxBatch=" << kMaxBatch_ << "\n";

    cudaError_t e1 = cudaMalloc(&d_input_,
        size_t(kMaxBatch_) * 3 * kTileLR_ * kTileLR_ * sizeof(float));
    cudaError_t e2 = cudaMalloc(&d_output_,
        size_t(kMaxBatch_) * 3 * kTileHR_ * kTileHR_ * sizeof(float));
    if (e1 != cudaSuccess || e2 != cudaSuccess) {
        std::cerr << "SrTrtEngine: device buffer allocation failed\n";
        return false;
    }

    impl_->context->setTensorAddress(impl_->inputName.c_str(),  d_input_);
    impl_->context->setTensorAddress(impl_->outputName.c_str(), d_output_);
    return true;
}

void SrTrtEngine::infer(int batchSize, cudaStream_t stream) {
    impl_->context->setInputShape(
        impl_->inputName.c_str(), Dims4{batchSize, 3, kTileLR_, kTileLR_});
    impl_->context->enqueueV3(stream);
}

SrTrtEngine::~SrTrtEngine() {
    if (d_input_)  cudaFree(d_input_);
    if (d_output_) cudaFree(d_output_);
    if (impl_) {
        delete impl_->context;
        delete impl_->engine;
        delete impl_->runtime;
        delete impl_;
    }
}
