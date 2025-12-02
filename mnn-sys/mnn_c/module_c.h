#ifndef MODULE_C_H
#define MODULE_C_H

#include "error_code_c.h"
#include "tensor_c.h"
#include <MNN/MNNForwardType.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque pointer to MNN::Express::Module
typedef struct Module Module;

// Opaque pointer to MNN::Express::VARP (Variable)
typedef struct VARP VARP;

// Module configuration
typedef struct {
    // Load module as dynamic, default static (0 = static, 1 = dynamic)
    int dynamic;
    // For static mode, if the shape is mutable (1 = true, 0 = false)
    int shapeMutable;
    // Pre-rearrange weights or not (1 = true, 0 = false)
    int rearrange;
    // Backend type (MNNForwardType)
    MNNForwardType backendType;
    // Number of threads (for CPU backend)
    int numThreads;
} ModuleConfig;

// Module info structure
typedef struct {
    // Number of inputs
    size_t inputCount;
    // Number of outputs
    size_t outputCount;
    // Input names (array of strings)
    const char** inputNames;
    // Output names (array of strings)
    const char** outputNames;
} ModuleInfo;

// VARP info structure (for input/output tensors)
typedef struct {
    // Dimension count
    size_t dimCount;
    // Dimensions
    const int* dims;
    // Element type (halide_type_t equivalent)
    int elementType;
    // Element size in bytes
    size_t elementSize;
} VARPInfo;

/**
 * @brief Load a module from file
 * @param fileName Path to the MNN model file
 * @param inputNames Array of input tensor names (NULL-terminated or use inputCount)
 * @param inputCount Number of input names
 * @param outputNames Array of output tensor names (NULL-terminated or use outputCount)
 * @param outputCount Number of output names
 * @param config Module configuration (can be NULL for defaults)
 * @return Module pointer if success, NULL otherwise
 */
Module* Module_loadFromFile(const char* fileName,
                            const char** inputNames, size_t inputCount,
                            const char** outputNames, size_t outputCount,
                            const ModuleConfig* config);

/**
 * @brief Load a module from buffer
 * @param buffer Model data buffer
 * @param length Buffer length
 * @param inputNames Array of input tensor names
 * @param inputCount Number of input names
 * @param outputNames Array of output tensor names
 * @param outputCount Number of output names
 * @param config Module configuration (can be NULL for defaults)
 * @return Module pointer if success, NULL otherwise
 */
Module* Module_loadFromBuffer(const void* buffer, size_t length,
                              const char** inputNames, size_t inputCount,
                              const char** outputNames, size_t outputCount,
                              const ModuleConfig* config);

/**
 * @brief Destroy a module
 * @param module Module to destroy
 */
void Module_destroy(Module* module);

/**
 * @brief Get module info
 * @param module Module pointer
 * @return ModuleInfo pointer (owned by module, do not free)
 */
const ModuleInfo* Module_getInfo(const Module* module);

/**
 * @brief Create a VARP for input
 * @param dims Dimension array
 * @param dimCount Number of dimensions
 * @param dataType Data type (0 = float32, 1 = int32, 2 = int64, etc.)
 * @return VARP pointer
 */
VARP* VARP_create(const int* dims, size_t dimCount, int dataType);

/**
 * @brief Destroy a VARP
 * @param varp VARP to destroy
 */
void VARP_destroy(VARP* varp);

/**
 * @brief Get VARP info
 * @param varp VARP pointer
 * @param info Output info structure
 * @return 0 on success, non-zero on error
 */
int VARP_getInfo(const VARP* varp, VARPInfo* info);

/**
 * @brief Write float data to VARP
 * @param varp VARP pointer
 * @param data Float data array
 * @param count Number of elements
 * @return 0 on success, non-zero on error
 */
int VARP_writeFloat(VARP* varp, const float* data, size_t count);

/**
 * @brief Write int32 data to VARP
 * @param varp VARP pointer
 * @param data Int32 data array
 * @param count Number of elements
 * @return 0 on success, non-zero on error
 */
int VARP_writeInt32(VARP* varp, const int32_t* data, size_t count);

/**
 * @brief Write int64 data to VARP
 * @param varp VARP pointer
 * @param data Int64 data array
 * @param count Number of elements
 * @return 0 on success, non-zero on error
 */
int VARP_writeInt64(VARP* varp, const int64_t* data, size_t count);

/**
 * @brief Read float data from VARP
 * @param varp VARP pointer
 * @param data Output float data array
 * @param count Number of elements to read
 * @return 0 on success, non-zero on error
 */
int VARP_readFloat(const VARP* varp, float* data, size_t count);

/**
 * @brief Read int32 data from VARP
 * @param varp VARP pointer
 * @param data Output int32 data array
 * @param count Number of elements to read
 * @return 0 on success, non-zero on error
 */
int VARP_readInt32(const VARP* varp, int32_t* data, size_t count);

/**
 * @brief Run module forward pass
 * @param module Module pointer
 * @param inputs Array of input VARPs
 * @param inputCount Number of inputs
 * @param outputs Array to store output VARPs (caller must allocate)
 * @param outputCount Expected number of outputs
 * @return 0 on success, non-zero on error
 */
int Module_forward(Module* module,
                   VARP* const* inputs, size_t inputCount,
                   VARP** outputs, size_t outputCount);

/**
 * @brief Get the number of elements in a VARP
 * @param varp VARP pointer
 * @return Number of elements
 */
size_t VARP_getElementCount(const VARP* varp);

#ifdef __cplusplus
}
#endif

#endif // MODULE_C_H
