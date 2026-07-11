/*
 * Custom nvdsvideotemplate library running the APISR TensorRT upscaler.
 *
 * IMPORTANT: Processing is fully synchronous inside ProcessBuffer — no
 * background thread, no queue, no cross-thread gst_pad_push. This avoids
 * a confirmed deadlock between the GStreamer streaming thread and a
 * separate OutputThread, caused by gst_pad_push requiring the streaming
 * thread (which is blocked upstream waiting for a V4L2 capture buffer
 * that OutputThread hasn't released yet).
 *
 * At 8-15ms/frame GPU processing time, synchronous processing on the
 * streaming thread is easily fast enough for real-time 30fps playback.
 */
#include <iostream>
#include <chrono>
#include <cuda_runtime.h>

#include "nvbufsurface.h"
#include "nvbufsurftransform.h"
#include "gstnvdsmeta.h"

#include "nvdscustomlib_base.hpp"
#include "sr_trt_engine.hpp"
#include "sr_pipeline.hpp"

#define GST_CAPS_FEATURE_MEMORY_NVMM "memory:NVMM"

static const char *kDefaultEnginePath = "model/trt_engines/apisr_fp16.trt";

class ApisrUpscaleAlgorithm : public DSCustomLibraryBase
{
public:
    ApisrUpscaleAlgorithm() = default;
    ~ApisrUpscaleAlgorithm();

    bool SetInitParams(DSCustom_CreateParams *params) override;
    bool SetProperty(Property &prop) override;
    bool HandleEvent(GstEvent *event) override;
    char *QueryProperties() override;
    GstCaps *GetCompatibleCaps(GstPadDirection direction,
                                GstCaps *in_caps, GstCaps *othercaps) override;
    BufferResult ProcessBuffer(GstBuffer *inbuf) override;

private:
    gdouble m_scaleFactor = 1;
    std::string m_enginePath = kDefaultEnginePath;
    int m_overlapHr = 16;

    SrTrtEngine m_engine;
    SrPipeline  m_pipeline;
    bool m_engineReady = false;

    GstBufferPool *m_dsBufferPool = nullptr;
    cudaStream_t m_cudaStream = nullptr;
    int m_frameCount = 0;
};

extern "C" IDSCustomLibrary *CreateCustomAlgoCtx(DSCustom_CreateParams *params);
extern "C" IDSCustomLibrary *CreateCustomAlgoCtx(DSCustom_CreateParams *params) {
    return new ApisrUpscaleAlgorithm();
}

bool ApisrUpscaleAlgorithm::SetProperty(Property &prop) {
    if (prop.key == "engine-path") {
        m_enginePath = prop.value;
        // Pre-load engine so m_scaleFactor is correct BEFORE GetCompatibleCaps
        // fires during caps negotiation. Without this, the output buffer pool
        // gets allocated at input resolution (scale=1) instead of upscaled.
        if (!m_engine.isLoaded()) {
            if (m_engine.load(m_enginePath)) {
                m_scaleFactor = m_engine.scale();
                std::cerr << "ApisrUpscale: pre-loaded engine, scale=" << m_scaleFactor << "\n";
            } else {
                std::cerr << "ApisrUpscale: WARNING engine pre-load failed\n";
            }
        }
    } else if (prop.key == "overlap") {
        try { m_overlapHr = std::stoi(prop.value); }
        catch (...) { return false; }
    } else if (prop.key == "scale-factor") {
        try { m_scaleFactor = std::stod(prop.value); }
        catch (...) { return false; }
    }
    return true;
}

bool ApisrUpscaleAlgorithm::SetInitParams(DSCustom_CreateParams *params) {
    DSCustomLibraryBase::SetInitParams(params);
    m_cudaStream = params->m_cudaStream;

    GstStructure *s1 = gst_caps_get_structure(m_inCaps, 0);
    BufferPoolConfig pool_config = {0};
    pool_config.cuda_mem_type = NVBUF_MEM_DEFAULT;
    pool_config.gpu_id = params->m_gpuId;
    pool_config.max_buffers = 4;
    gst_structure_get_int(s1, "batch-size", &pool_config.batch_size);
    if (pool_config.batch_size == 0) pool_config.batch_size = 1;

    m_dsBufferPool = CreateBufferPool(&pool_config, m_outCaps);
    if (!m_dsBufferPool) {
        GST_ERROR_OBJECT(m_element, "ApisrUpscale: output buffer pool creation failed");
        return false;
    }

    cudaSetDevice(params->m_gpuId);
    if (!m_engine.isLoaded()) {
        if (!m_engine.load(m_enginePath)) {
            GST_ERROR_OBJECT(m_element, "ApisrUpscale: failed to load engine: %s",
                              m_enginePath.c_str());
            return false;
        }
    }
    m_scaleFactor = m_engine.scale();

    if (!m_pipeline.init(params->m_gpuId, m_cudaStream)) {
        GST_ERROR_OBJECT(m_element, "ApisrUpscale: pipeline init failed");
        return false;
    }
    m_engineReady = true;
    return true;
}

GstCaps *ApisrUpscaleAlgorithm::GetCompatibleCaps(
    GstPadDirection direction, GstCaps *in_caps, GstCaps *othercaps)
{
    othercaps = gst_caps_truncate(othercaps);
    othercaps = gst_caps_make_writable(othercaps);

    GstStructure *s1 = gst_caps_get_structure(in_caps, 0);
    GstStructure *s2 = gst_caps_get_structure(othercaps, 0);

    gint width = 0, height = 0, num = 0, denom = 0;
    gst_structure_get_int(s1, "width", &width);
    gst_structure_get_int(s1, "height", &height);

    gst_structure_set(s2,
        "width",  G_TYPE_INT, (gint)(m_scaleFactor * width),
        "height", G_TYPE_INT, (gint)(m_scaleFactor * height),
        NULL);
    if (gst_structure_get_fraction(s1, "framerate", &num, &denom))
        gst_structure_fixate_field_nearest_fraction(s2, "framerate", num, denom);

    const gchar *inputFmt = gst_structure_get_string(s1, "format");
    if (!gst_structure_get_string(s2, "format"))
        gst_structure_set(s2, "format", G_TYPE_STRING, inputFmt, NULL);

    gst_structure_set(s2, "pixel-aspect-ratio", GST_TYPE_FRACTION, 1, 1, NULL);

    GstCapsFeatures *features = gst_caps_features_new(GST_CAPS_FEATURE_MEMORY_NVMM, NULL);
    gst_caps_set_features(othercaps, 0, features);

    return gst_caps_ref(othercaps);
}

char *ApisrUpscaleAlgorithm::QueryProperties() {
    char *str = new char[512];
    snprintf(str, 512,
        "APISR upscaler customlib properties:\n"
        "  engine-path:/abs/path/to/apisr_fp16.trt\n"
        "  overlap:16  (HR pixels)\n");
    return str;
}

bool ApisrUpscaleAlgorithm::HandleEvent(GstEvent *event) {
    return true;
}

BufferResult ApisrUpscaleAlgorithm::ProcessBuffer(GstBuffer *inbuf) {
    if (!m_engineReady) return BufferResult::Buffer_Error;

    NvBufSurface *in_surf = getNvBufSurface(inbuf);
    if (!in_surf) return BufferResult::Buffer_Error;

    // Acquire output buffer from our pool.
    GstBuffer *outBuffer = nullptr;
    if (gst_buffer_pool_acquire_buffer(m_dsBufferPool, &outBuffer, NULL) != GST_FLOW_OK) {
        GST_ERROR_OBJECT(m_element, "ApisrUpscale: output buffer acquire failed");
        return BufferResult::Buffer_Error;
    }

    gst_buffer_copy_into(outBuffer, inbuf, GST_BUFFER_COPY_META, 0, -1);
    NvBufSurface *out_surf = getNvBufSurface(outBuffer);
    if (!out_surf) {
        gst_buffer_unref(outBuffer);
        return BufferResult::Buffer_Error;
    }

    auto tStart = std::chrono::steady_clock::now();

    for (guint i = 0; i < in_surf->numFilled; ++i) {
        NvBufSurfaceParams &inP  = in_surf->surfaceList[i];
        NvBufSurfaceParams &outP = out_surf->surfaceList[i];

        m_pipeline.processFrame(
            (const uint8_t *)inP.dataPtr,  inP.planeParams.pitch[0],
            inP.width, inP.height,
            (uint8_t *)outP.dataPtr,       outP.planeParams.pitch[0],
            m_overlapHr, m_engine, m_cudaStream);
    }
    cudaStreamSynchronize(m_cudaStream);

    if (++m_frameCount <= 10 || m_frameCount % 60 == 0) {
        auto tEnd = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(tEnd - tStart).count();
        std::cerr << "ApisrUpscale: frame " << m_frameCount
                  << " GPU ms=" << ms << " FPS=" << (1000.0 / ms) << "\n";
    }

    out_surf->numFilled = in_surf->numFilled;
    GST_BUFFER_PTS(outBuffer)      = GST_BUFFER_PTS(inbuf);
    GST_BUFFER_DURATION(outBuffer) = GST_BUFFER_DURATION(inbuf);

    // Push output downstream on THIS thread (the streaming thread).
    // No cross-thread gst_pad_push — avoids the deadlock.
    GstFlowReturn flow = gst_pad_push(
        GST_BASE_TRANSFORM_SRC_PAD(m_element), outBuffer);
    if (flow != GST_FLOW_OK) {
        std::cerr << "ApisrUpscale: gst_pad_push returned "
                  << gst_flow_get_name(flow) << "\n";
    }

    // Buffer_Drop: we already pushed our own output; tell the template
    // to drop the original input buffer (don't push it downstream).
    return BufferResult::Buffer_Drop;
}

ApisrUpscaleAlgorithm::~ApisrUpscaleAlgorithm() {
    if (m_dsBufferPool) {
        gst_buffer_pool_set_active(m_dsBufferPool, FALSE);
        gst_object_unref(m_dsBufferPool);
    }
}
