#include "module_c.h"
#include <MNN/expr/Module.hpp>
#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/Executor.hpp>
#include <vector>
#include <string>
#include <cstring>
#include <cstdio>

// Use namespace aliases to avoid conflicts
namespace MNNExpress = MNN::Express;

// Internal structure to hold module and its info
// Use MnnModule to avoid name conflict with MNN::Express::Module
struct MnnModuleWrapper {
    std::shared_ptr<MNNExpress::Module> module;
    std::vector<std::string> inputNames;
    std::vector<std::string> outputNames;
    std::vector<const char*> inputNamePtrs;
    std::vector<const char*> outputNamePtrs;
    ModuleInfo info;
};

// Internal VARP wrapper
struct MnnVarpWrapper {
    MNNExpress::VARP var;
    std::vector<int> dims;
};

// Cast helpers
static MnnModuleWrapper* toWrapper(Module* m) {
    return reinterpret_cast<MnnModuleWrapper*>(m);
}

static const MnnModuleWrapper* toWrapper(const Module* m) {
    return reinterpret_cast<const MnnModuleWrapper*>(m);
}

static MnnVarpWrapper* toVarpWrapper(VARP* v) {
    return reinterpret_cast<MnnVarpWrapper*>(v);
}

static const MnnVarpWrapper* toVarpWrapper(const VARP* v) {
    return reinterpret_cast<const MnnVarpWrapper*>(v);
}

Module* Module_loadFromFile(const char* fileName,
                            const char** inputNames, size_t inputCount,
                            const char** outputNames, size_t outputCount,
                            const ModuleConfig* config) {
    if (!fileName) {
        return nullptr;
    }

    std::vector<std::string> inputs;
    std::vector<std::string> outputs;

    // Allow empty input/output names - MNN will auto-detect from model
    if (inputNames) {
        for (size_t i = 0; i < inputCount; i++) {
            if (inputNames[i]) {
                inputs.push_back(inputNames[i]);
            }
        }
    }

    if (outputNames) {
        for (size_t i = 0; i < outputCount; i++) {
            if (outputNames[i]) {
                outputs.push_back(outputNames[i]);
            }
        }
    }

    fprintf(stderr, "[MNN Module] Loading from file: %s\n", fileName);
    fprintf(stderr, "[MNN Module] Requested inputs: %zu, outputs: %zu\n", inputs.size(), outputs.size());

    MNNExpress::Module::Config moduleConfig;
    if (config) {
        moduleConfig.dynamic = config->dynamic != 0;
        moduleConfig.shapeMutable = config->shapeMutable != 0;
        moduleConfig.rearrange = config->rearrange != 0;
    }

    auto mnnModule = MNNExpress::Module::load(inputs, outputs, fileName, &moduleConfig);
    if (!mnnModule) {
        fprintf(stderr, "[MNN Module] Failed to load module\n");
        return nullptr;
    }

    MnnModuleWrapper* wrapper = new MnnModuleWrapper();
    wrapper->module.reset(mnnModule);

    // Get actual input/output names from the loaded module (may be auto-detected)
    auto moduleInfo = mnnModule->getInfo();
    if (moduleInfo) {
        wrapper->inputNames = moduleInfo->inputNames;
        wrapper->outputNames = moduleInfo->outputNames;
        fprintf(stderr, "[MNN Module] Model detected inputs: %zu, outputs: %zu\n",
                wrapper->inputNames.size(), wrapper->outputNames.size());
        fprintf(stderr, "[MNN Module] Default format: %s\n",
                moduleInfo->defaultFormat == MNNExpress::NCHW ? "NCHW" : "NHWC");

        // Print detailed input info
        for (size_t i = 0; i < moduleInfo->inputs.size() && i < wrapper->inputNames.size(); i++) {
            auto& info = moduleInfo->inputs[i];
            fprintf(stderr, "  input[%zu]: %s, dims=[", i, wrapper->inputNames[i].c_str());
            for (size_t d = 0; d < info.dim.size(); d++) {
                fprintf(stderr, "%d%s", info.dim[d], d + 1 < info.dim.size() ? "," : "");
            }
            fprintf(stderr, "], type.code=%d, type.bits=%d\n",
                    info.type.code, info.type.bits);
        }
        for (size_t i = 0; i < wrapper->outputNames.size(); i++) {
            fprintf(stderr, "  output[%zu]: %s\n", i, wrapper->outputNames[i].c_str());
        }
    } else {
        // Fallback to provided names
        wrapper->inputNames = inputs;
        wrapper->outputNames = outputs;
        fprintf(stderr, "[MNN Module] Warning: no module info available, using provided names\n");
    }

    // Setup name pointers for C API
    for (const auto& name : wrapper->inputNames) {
        wrapper->inputNamePtrs.push_back(name.c_str());
    }
    for (const auto& name : wrapper->outputNames) {
        wrapper->outputNamePtrs.push_back(name.c_str());
    }

    wrapper->info.inputCount = wrapper->inputNames.size();
    wrapper->info.outputCount = wrapper->outputNames.size();
    wrapper->info.inputNames = wrapper->inputNamePtrs.data();
    wrapper->info.outputNames = wrapper->outputNamePtrs.data();

    return reinterpret_cast<Module*>(wrapper);
}

Module* Module_loadFromBuffer(const void* buffer, size_t length,
                              const char** inputNames, size_t inputCount,
                              const char** outputNames, size_t outputCount,
                              const ModuleConfig* config) {
    if (!buffer || length == 0) {
        return nullptr;
    }

    std::vector<std::string> inputs;
    std::vector<std::string> outputs;

    // Allow empty input/output names - MNN will auto-detect from model
    if (inputNames) {
        for (size_t i = 0; i < inputCount; i++) {
            if (inputNames[i]) {
                inputs.push_back(inputNames[i]);
            }
        }
    }

    if (outputNames) {
        for (size_t i = 0; i < outputCount; i++) {
            if (outputNames[i]) {
                outputs.push_back(outputNames[i]);
            }
        }
    }

    fprintf(stderr, "[MNN Module] Loading from buffer (%zu bytes)\n", length);
    fprintf(stderr, "[MNN Module] Requested inputs: %zu, outputs: %zu\n", inputs.size(), outputs.size());

    MNNExpress::Module::Config moduleConfig;
    if (config) {
        moduleConfig.dynamic = config->dynamic != 0;
        moduleConfig.shapeMutable = config->shapeMutable != 0;
        moduleConfig.rearrange = config->rearrange != 0;
    }

    auto mnnModule = MNNExpress::Module::load(inputs, outputs,
                                              static_cast<const uint8_t*>(buffer),
                                              length, &moduleConfig);
    if (!mnnModule) {
        fprintf(stderr, "[MNN Module] Failed to load module from buffer\n");
        return nullptr;
    }

    MnnModuleWrapper* wrapper = new MnnModuleWrapper();
    wrapper->module.reset(mnnModule);

    // Get actual input/output names from the loaded module (may be auto-detected)
    auto moduleInfo = mnnModule->getInfo();
    if (moduleInfo) {
        wrapper->inputNames = moduleInfo->inputNames;
        wrapper->outputNames = moduleInfo->outputNames;
        fprintf(stderr, "[MNN Module] Model detected inputs: %zu, outputs: %zu\n",
                wrapper->inputNames.size(), wrapper->outputNames.size());
    } else {
        // Fallback to provided names
        wrapper->inputNames = inputs;
        wrapper->outputNames = outputs;
        fprintf(stderr, "[MNN Module] Warning: no module info available, using provided names\n");
    }

    // Setup name pointers for C API
    for (const auto& name : wrapper->inputNames) {
        wrapper->inputNamePtrs.push_back(name.c_str());
    }
    for (const auto& name : wrapper->outputNames) {
        wrapper->outputNamePtrs.push_back(name.c_str());
    }

    wrapper->info.inputCount = wrapper->inputNames.size();
    wrapper->info.outputCount = wrapper->outputNames.size();
    wrapper->info.inputNames = wrapper->inputNamePtrs.data();
    wrapper->info.outputNames = wrapper->outputNamePtrs.data();

    return reinterpret_cast<Module*>(wrapper);
}

void Module_destroy(Module* module) {
    if (module) {
        delete toWrapper(module);
    }
}

const ModuleInfo* Module_getInfo(const Module* module) {
    if (!module) {
        return nullptr;
    }
    return &toWrapper(module)->info;
}

VARP* VARP_create(const int* dims, size_t dimCount, int dataType) {
    if (!dims || dimCount == 0) {
        return nullptr;
    }

    std::vector<int> shape(dims, dims + dimCount);

    halide_type_t type;
    switch (dataType) {
        case 0: // float32
            type = halide_type_of<float>();
            break;
        case 1: // int32
            type = halide_type_of<int32_t>();
            break;
        case 2: // int64
            type = halide_type_of<int64_t>();
            break;
        case 3: // uint8
            type = halide_type_of<uint8_t>();
            break;
        default:
            type = halide_type_of<float>();
            break;
    }

    MnnVarpWrapper* wrapper = new MnnVarpWrapper();
    wrapper->var = MNNExpress::_Input(shape, MNNExpress::NCHW, type);
    wrapper->dims = shape;

    return reinterpret_cast<VARP*>(wrapper);
}

void VARP_destroy(VARP* varp) {
    if (varp) {
        delete toVarpWrapper(varp);
    }
}

int VARP_getInfo(const VARP* varp, VARPInfo* info) {
    if (!varp || !info) {
        return -1;
    }

    auto wrapper = toVarpWrapper(varp);
    auto varInfo = wrapper->var->getInfo();
    if (!varInfo) {
        return -1;
    }

    info->dimCount = varInfo->dim.size();
    info->dims = varInfo->dim.data();
    info->elementType = varInfo->type.code;
    info->elementSize = varInfo->type.bytes();

    return 0;
}

int VARP_writeFloat(VARP* varp, const float* data, size_t count) {
    if (!varp || !data) {
        return -1;
    }

    auto wrapper = toVarpWrapper(varp);
    auto ptr = wrapper->var->writeMap<float>();
    if (!ptr) {
        return -1;
    }

    memcpy(ptr, data, count * sizeof(float));
    return 0;
}

int VARP_writeInt32(VARP* varp, const int32_t* data, size_t count) {
    if (!varp || !data) {
        return -1;
    }

    auto wrapper = toVarpWrapper(varp);
    auto ptr = wrapper->var->writeMap<int32_t>();
    if (!ptr) {
        return -1;
    }

    memcpy(ptr, data, count * sizeof(int32_t));
    return 0;
}

int VARP_writeInt64(VARP* varp, const int64_t* data, size_t count) {
    if (!varp || !data) {
        return -1;
    }

    auto wrapper = toVarpWrapper(varp);
    auto ptr = wrapper->var->writeMap<int64_t>();
    if (!ptr) {
        return -1;
    }

    memcpy(ptr, data, count * sizeof(int64_t));
    return 0;
}

int VARP_readFloat(const VARP* varp, float* data, size_t count) {
    if (!varp || !data) {
        return -1;
    }

    auto wrapper = toVarpWrapper(varp);
    auto ptr = wrapper->var->readMap<float>();
    if (!ptr) {
        return -1;
    }

    memcpy(data, ptr, count * sizeof(float));
    return 0;
}

int VARP_readInt32(const VARP* varp, int32_t* data, size_t count) {
    if (!varp || !data) {
        return -1;
    }

    auto wrapper = toVarpWrapper(varp);
    auto ptr = wrapper->var->readMap<int32_t>();
    if (!ptr) {
        return -1;
    }

    memcpy(data, ptr, count * sizeof(int32_t));
    return 0;
}

int Module_forward(Module* module,
                   VARP* const* inputs, size_t inputCount,
                   VARP** outputs, size_t outputCount) {
    if (!module || !inputs || !outputs) {
        fprintf(stderr, "[MNN Module] Error: null parameter\n");
        return -1;
    }

    auto wrapper = toWrapper(module);

    std::vector<MNNExpress::VARP> inputVars;
    fprintf(stderr, "[MNN Module] Preparing %zu inputs...\n", inputCount);
    for (size_t i = 0; i < inputCount; i++) {
        if (inputs[i]) {
            auto vw = toVarpWrapper(inputs[i]);
            auto info = vw->var->getInfo();
            if (info) {
                const char* typeName = "unknown";
                if (info->type.code == halide_type_int && info->type.bits == 64) typeName = "int64";
                else if (info->type.code == halide_type_int && info->type.bits == 32) typeName = "int32";
                else if (info->type.code == halide_type_float && info->type.bits == 32) typeName = "float32";
                else if (info->type.code == halide_type_uint && info->type.bits == 8) typeName = "uint8";

                fprintf(stderr, "  input[%zu]: dims=[", i);
                for (size_t d = 0; d < info->dim.size(); d++) {
                    fprintf(stderr, "%d%s", info->dim[d], d + 1 < info->dim.size() ? "," : "");
                }
                fprintf(stderr, "], size=%d, type=%s (code=%d,bits=%d)\n",
                        info->size, typeName, info->type.code, info->type.bits);
            } else {
                fprintf(stderr, "  input[%zu]: (no info available)\n", i);
            }
            inputVars.push_back(vw->var);
        }
    }

    fprintf(stderr, "[MNN Module] Running forward with %zu inputs...\n", inputVars.size());

    auto outputVars = wrapper->module->onForward(inputVars);
    fprintf(stderr, "[MNN Module] Forward returned %zu outputs\n", outputVars.size());

    // Print output details if available
    for (size_t i = 0; i < outputVars.size(); i++) {
        auto info = outputVars[i]->getInfo();
        if (info) {
            fprintf(stderr, "  output[%zu]: dims=[", i);
            for (size_t d = 0; d < info->dim.size(); d++) {
                fprintf(stderr, "%d%s", info->dim[d], d + 1 < info->dim.size() ? "," : "");
            }
            fprintf(stderr, "], size=%d\n", info->size);
        }
    }

    // Return actual output count as negative if mismatch (for debugging)
    // -2 means no outputs, -3 means too few, -4 means too many
    if (outputVars.empty()) {
        fprintf(stderr, "[MNN Module] Error: no outputs!\n");
        return -2;  // No outputs - forward failed
    }
    if (outputVars.size() < outputCount) {
        fprintf(stderr, "[MNN Module] Error: expected %zu outputs, got %zu\n", outputCount, outputVars.size());
        return -3;  // Fewer outputs than expected
    }
    if (outputVars.size() > outputCount) {
        fprintf(stderr, "[MNN Module] Warning: expected %zu outputs, got %zu (using first %zu)\n",
                outputCount, outputVars.size(), outputCount);
        // Continue anyway, just use the first outputCount outputs
    }

    for (size_t i = 0; i < outputCount; i++) {
        MnnVarpWrapper* outWrapper = new MnnVarpWrapper();
        outWrapper->var = outputVars[i];

        auto info = outputVars[i]->getInfo();
        if (info) {
            outWrapper->dims = info->dim;
        }

        outputs[i] = reinterpret_cast<VARP*>(outWrapper);
    }

    return 0;
}

size_t VARP_getElementCount(const VARP* varp) {
    if (!varp) {
        return 0;
    }

    auto wrapper = toVarpWrapper(varp);
    auto info = wrapper->var->getInfo();
    if (!info) {
        return 0;
    }

    return info->size;
}
