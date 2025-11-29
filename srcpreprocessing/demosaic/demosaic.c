/**
 * @file demosaic.c
 * @brief Demosaicing library implementation
 * @author hany
 * @date 2025
 * @version 2.0.0
 */

// ============================================================================
// Includes
// ============================================================================

#include "demosaic.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdarg.h>
#include <time.h>
#include <assert.h>

// Platform-specific includes
#ifdef _WIN32
    #include <windows.h>
    #include <intrin.h>
#else
    #include <pthread.h>
    #include <unistd.h>
    #include <sys/time.h>
    #include <cpuid.h>
#endif

// Optional dependencies
#ifdef DEMOSAIC_ENABLE_OPENMP
    #include <omp.h>
#endif

#ifdef DEMOSAIC_ENABLE_CUDA
    #include <cuda_runtime.h>
#endif

#ifdef DEMOSAIC_ENABLE_OPENCL
    #include <CL/cl.h>
#endif

// SIMD includes
#if defined(__SSE2__)
    #include <emmintrin.h>
#endif
#if defined(__AVX__)
    #include <immintrin.h>
#endif
#if defined(__ARM_NEON)
    #include <arm_neon.h>
#endif

// ============================================================================
// Internal Constants
// ============================================================================

#define DEMOSAIC_INTERNAL_VERSION "2.0.0"
#define DEMOSAIC_BUILD_DATE __DATE__
#define DEMOSAIC_BUILD_TIME __TIME__

#define DEFAULT_NUM_THREADS 0  // Auto-detect
#define MAX_THREADS 256
#define CACHE_LINE_SIZE 64
#define ALIGNMENT 32

// Memory tracking
#define ENABLE_MEMORY_TRACKING 1

// Logging
#define LOG_BUFFER_SIZE 1024

// ============================================================================
// Internal Structures
// ============================================================================

/**
 * @brief Library state structure
 */
typedef struct {
    bool initialized;
    int num_threads;
    bool simd_enabled;
    int log_level;
    bool verbose;
    
    // Error handling
    DemosaicError last_error;
    char last_error_message[256];
    void (*error_handler)(DemosaicError, const char*, void*);
    void *error_handler_data;
    
    // Logging
    LogCallback log_callback;
    void *log_callback_data;
    
    // Memory tracking
    size_t current_memory_usage;
    size_t peak_memory_usage;
    size_t allocation_count;
    
    // Hardware capabilities
    HardwareCapabilities hw_caps;
    
    // CUDA state
#ifdef DEMOSAIC_ENABLE_CUDA
    int cuda_device;
    bool cuda_initialized;
#endif
    
    // OpenCL state
#ifdef DEMOSAIC_ENABLE_OPENCL
    cl_platform_id opencl_platform;
    cl_device_id opencl_device;
    cl_context opencl_context;
    cl_command_queue opencl_queue;
    bool opencl_initialized;
#endif
    
    // Thread synchronization
#ifdef _WIN32
    CRITICAL_SECTION mutex;
#else
    pthread_mutex_t mutex;
#endif
    
} LibraryState;

// Global library state
static LibraryState g_state = {
    .initialized = false,
    .num_threads = DEFAULT_NUM_THREADS,
    .simd_enabled = true,
    .log_level = 1, // INFO
    .verbose = false,
    .last_error = DEMOSAIC_SUCCESS,
    .error_handler = NULL,
    .error_handler_data = NULL,
    .log_callback = NULL,
    .log_callback_data = NULL,
    .current_memory_usage = 0,
    .peak_memory_usage = 0,
    .allocation_count = 0,
#ifdef DEMOSAIC_ENABLE_CUDA
    .cuda_device = 0,
    .cuda_initialized = false,
#endif
#ifdef DEMOSAIC_ENABLE_OPENCL
    .opencl_initialized = false,
#endif
};

// ============================================================================
// Internal Function Declarations
// ============================================================================

// Initialization helpers
static DemosaicError init_threading(void);
static DemosaicError init_hardware_detection(void);
static DemosaicError init_cuda(void);
static DemosaicError init_opencl(void);
static void cleanup_cuda(void);
static void cleanup_opencl(void);

// Logging helpers
static void log_message(int level, const char *format, ...);
static const char* log_level_to_string(int level);

// Memory tracking helpers
static void track_allocation(size_t size);
static void track_deallocation(size_t size);

// Hardware detection helpers
static void detect_cpu_features(void);
static void detect_gpu_devices(void);

// Thread synchronization
static void lock_mutex(void);
static void unlock_mutex(void);

// ============================================================================
// Part 1: Library Initialization and Cleanup
// ============================================================================

/**
 * @brief Initialize the demosaicing library
 */
DemosaicError demosaic_init(void) {
    if (g_state.initialized) {
        log_message(1, "Library already initialized");
        return DEMOSAIC_SUCCESS;
    }
    
    log_message(1, "Initializing demosaicing library v%s", DEMOSAIC_INTERNAL_VERSION);
    log_message(0, "Build date: %s %s", DEMOSAIC_BUILD_DATE, DEMOSAIC_BUILD_TIME);
    
    // Initialize mutex
#ifdef _WIN32
    InitializeCriticalSection(&g_state.mutex);
#else
    pthread_mutex_init(&g_state.mutex, NULL);
#endif
    
    // Initialize threading
    DemosaicError err = init_threading();
    if (err != DEMOSAIC_SUCCESS) {
        log_message(3, "Failed to initialize threading");
        return err;
    }
    
    // Detect hardware capabilities
    err = init_hardware_detection();
    if (err != DEMOSAIC_SUCCESS) {
        log_message(2, "Hardware detection failed, continuing with defaults");
    }
    
    // Initialize CUDA if available
#ifdef DEMOSAIC_ENABLE_CUDA
    err = init_cuda();
    if (err == DEMOSAIC_SUCCESS) {
        log_message(1, "CUDA initialized successfully");
    } else {
        log_message(1, "CUDA not available");
    }
#endif
    
    // Initialize OpenCL if available
#ifdef DEMOSAIC_ENABLE_OPENCL
    err = init_opencl();
    if (err == DEMOSAIC_SUCCESS) {
        log_message(1, "OpenCL initialized successfully");
    } else {
        log_message(1, "OpenCL not available");
    }
#endif
    
    g_state.initialized = true;
    log_message(1, "Library initialization complete");
    
    return DEMOSAIC_SUCCESS;
}

/**
 * @brief Cleanup and release library resources
 */
void demosaic_cleanup(void) {
    if (!g_state.initialized) {
        return;
    }
    
    log_message(1, "Cleaning up demosaicing library");
    
    // Cleanup CUDA
#ifdef DEMOSAIC_ENABLE_CUDA
    if (g_state.cuda_initialized) {
        cleanup_cuda();
    }
#endif
    
    // Cleanup OpenCL
#ifdef DEMOSAIC_ENABLE_OPENCL
    if (g_state.opencl_initialized) {
        cleanup_opencl();
    }
#endif
    
    // Report memory leaks
    if (g_state.current_memory_usage > 0) {
        log_message(2, "Memory leak detected: %zu bytes still allocated",
                    g_state.current_memory_usage);
    }
    
    log_message(0, "Peak memory usage: %zu bytes", g_state.peak_memory_usage);
    log_message(0, "Total allocations: %zu", g_state.allocation_count);
    
    // Destroy mutex
#ifdef _WIN32
    DeleteCriticalSection(&g_state.mutex);
#else
    pthread_mutex_destroy(&g_state.mutex);
#endif
    
    g_state.initialized = false;
    log_message(1, "Library cleanup complete");
}

/**
 * @brief Get library version string
 */
const char* demosaic_get_version(void) {
    return DEMOSAIC_INTERNAL_VERSION;
}

/**
 * @brief Get library build information
 */
DemosaicError demosaic_get_build_info(char *info, size_t size) {
    if (info == NULL || size == 0) {
        return DEMOSAIC_ERROR_INVALID_ARGUMENT;
    }
    
    int written = snprintf(info, size,
        "Demosaicing Library v%s\n"
        "Build Date: %s %s\n"
        "Features:\n"
#ifdef DEMOSAIC_ENABLE_OPENMP
        "  - OpenMP: Enabled\n"
#else
        "  - OpenMP: Disabled\n"
#endif
#ifdef DEMOSAIC_ENABLE_CUDA
        "  - CUDA: Enabled\n"
#else
        "  - CUDA: Disabled\n"
#endif
#ifdef DEMOSAIC_ENABLE_OPENCL
        "  - OpenCL: Enabled\n"
#else
        "  - OpenCL: Disabled\n"
#endif
#ifdef DEMOSAIC_ENABLE_SIMD
        "  - SIMD: Enabled\n"
#else
        "  - SIMD: Disabled\n"
#endif
        "Platform: "
#ifdef _WIN32
        "Windows"
#elif defined(__APPLE__)
        "macOS"
#elif defined(__linux__)
        "Linux"
#else
        "Unknown"
#endif
        "\n",
        DEMOSAIC_INTERNAL_VERSION,
        DEMOSAIC_BUILD_DATE,
        DEMOSAIC_BUILD_TIME
    );
    
    if (written < 0 || (size_t)written >= size) {
        return DEMOSAIC_ERROR_BUFFER_TOO_SMALL;
    }
    
    return DEMOSAIC_SUCCESS;
}

/**
 * @brief Check if library is initialized
 */
bool demosaic_is_initialized(void) {
    return g_state.initialized;
}

// ============================================================================
// Part 1: Error Handling
// ============================================================================

/**
 * @brief Get error message for error code
 */
const char* demosaic_get_error_string(DemosaicError error) {
    switch (error) {
        case DEMOSAIC_SUCCESS:
            return "Success";
        case DEMOSAIC_ERROR_INVALID_ARGUMENT:
            return "Invalid argument";
        case DEMOSAIC_ERROR_OUT_OF_MEMORY:
            return "Out of memory";
        case DEMOSAIC_ERROR_INVALID_IMAGE:
            return "Invalid image";
        case DEMOSAIC_ERROR_INVALID_PATTERN:
            return "Invalid Bayer pattern";
        case DEMOSAIC_ERROR_INVALID_METHOD:
            return "Invalid demosaicing method";
        case DEMOSAIC_ERROR_INVALID_CONFIG:
            return "Invalid configuration";
        case DEMOSAIC_ERROR_NOT_INITIALIZED:
            return "Library not initialized";
        case DEMOSAIC_ERROR_ALREADY_INITIALIZED:
            return "Library already initialized";
        case DEMOSAIC_ERROR_FILE_NOT_FOUND:
            return "File not found";
        case DEMOSAIC_ERROR_FILE_READ:
            return "File read error";
        case DEMOSAIC_ERROR_FILE_WRITE:
            return "File write error";
        case DEMOSAIC_ERROR_UNSUPPORTED_FORMAT:
            return "Unsupported format";
        case DEMOSAIC_ERROR_CUDA_ERROR:
            return "CUDA error";
        case DEMOSAIC_ERROR_OPENCL_ERROR:
            return "OpenCL error";
        case DEMOSAIC_ERROR_THREAD_ERROR:
            return "Thread error";
        case DEMOSAIC_ERROR_BUFFER_TOO_SMALL:
            return "Buffer too small";
        case DEMOSAIC_ERROR_OPERATION_CANCELLED:
            return "Operation cancelled";
        case DEMOSAIC_ERROR_UNKNOWN:
        default:
            return "Unknown error";
    }
}

/**
 * @brief Get last error code
 */
DemosaicError demosaic_get_last_error(void) {
    return g_state.last_error;
}

/**
 * @brief Clear last error
 */
void demosaic_clear_error(void) {
    g_state.last_error = DEMOSAIC_SUCCESS;
    g_state.last_error_message[0] = '\0';
}

/**
 * @brief Set custom error handler
 */
void demosaic_set_error_handler(
    void (*handler)(DemosaicError error, const char *message, void *user_data),
    void *user_data
) {
    g_state.error_handler = handler;
    g_state.error_handler_data = user_data;
}

/**
 * @brief Internal function to set error
 */
static void set_error(DemosaicError error, const char *format, ...) {
    g_state.last_error = error;
    
    if (format != NULL) {
        va_list args;
        va_start(args, format);
        vsnprintf(g_state.last_error_message,
                  sizeof(g_state.last_error_message),
                  format, args);
        va_end(args);
    } else {
        strncpy(g_state.last_error_message,
                demosaic_get_error_string(error),
                sizeof(g_state.last_error_message) - 1);
    }
    
    // Call custom error handler if set
    if (g_state.error_handler != NULL) {
        g_state.error_handler(error,
                             g_state.last_error_message,
                             g_state.error_handler_data);
    }
    
    // Log error
    log_message(3, "Error: %s", g_state.last_error_message);
}

// ============================================================================
// Part 1: Logging Functions
// ============================================================================

/**
 * @brief Set log level
 */
void demosaic_set_log_level(int level) {
    g_state.log_level = level;
}

/**
 * @brief Get current log level
 */
int demosaic_get_log_level(void) {
    return g_state.log_level;
}

/**
 * @brief Set log callback
 */
void demosaic_set_log_callback(LogCallback callback, void *user_data) {
    g_state.log_callback = callback;
    g_state.log_callback_data = user_data;
}

/**
 * @brief Enable/disable verbose output
 */
void demosaic_set_verbose(bool enable) {
    g_state.verbose = enable;
}

/**
 * @brief Check if verbose output is enabled
 */
bool demosaic_is_verbose(void) {
    return g_state.verbose;
}

/**
 * @brief Internal logging function
 */
static void log_message(int level, const char *format, ...) {
    if (level < g_state.log_level && !g_state.verbose) {
        return;
    }
    
    char buffer[LOG_BUFFER_SIZE];
    va_list args;
    va_start(args, format);
    vsnprintf(buffer, sizeof(buffer), format, args);
    va_end(args);
    
    // Call custom log callback if set
    if (g_state.log_callback != NULL) {
        g_state.log_callback(level, buffer, g_state.log_callback_data);
    } else {
        // Default logging to stderr
        const char *level_str = log_level_to_string(level);
        fprintf(stderr, "[%s] %s\n", level_str, buffer);
    }
}

/**
 * @brief Convert log level to string
 */
static const char* log_level_to_string(int level) {
    switch (level) {
        case 0: return "DEBUG";
        case 1: return "INFO";
        case 2: return "WARNING";
        case 3: return "ERROR";
        default: return "UNKNOWN";
    }
}

// ============================================================================
// Part 1: Memory Management
// ============================================================================

/**
 * @brief Allocate aligned memory
 */
void* demosaic_aligned_alloc(size_t size, size_t alignment) {
    if (size == 0) {
        return NULL;
    }
    
    void *ptr = NULL;
    
#ifdef _WIN32
    ptr = _aligned_malloc(size, alignment);
#else
    if (posix_memalign(&ptr, alignment, size) != 0) {
        ptr = NULL;
    }
#endif
    
    if (ptr != NULL) {
        track_allocation(size);
    }
    
    return ptr;
}

/**
 * @brief Free aligned memory
 */
void demosaic_aligned_free(void *ptr) {
    if (ptr == NULL) {
        return;
    }
    
#ifdef _WIN32
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

/**
 * @brief Get current memory usage
 */
size_t demosaic_get_memory_usage(void) {
    return g_state.current_memory_usage;
}

/**
 * @brief Get peak memory usage
 */
size_t demosaic_get_peak_memory_usage(void) {
    return g_state.peak_memory_usage;
}

/**
 * @brief Reset memory statistics
 */
void demosaic_reset_memory_stats(void) {
    lock_mutex();
    g_state.current_memory_usage = 0;
    g_state.peak_memory_usage = 0;
    g_state.allocation_count = 0;
    unlock_mutex();
}

/**
 * @brief Track memory allocation
 */
static void track_allocation(size_t size) {
#if ENABLE_MEMORY_TRACKING
    lock_mutex();
    g_state.current_memory_usage += size;
    g_state.allocation_count++;
    
    if (g_state.current_memory_usage > g_state.peak_memory_usage) {
        g_state.peak_memory_usage = g_state.current_memory_usage;
    }
    unlock_mutex();
#endif
}

/**
 * @brief Track memory deallocation
 */
static void track_deallocation(size_t size) {
#if ENABLE_MEMORY_TRACKING
    lock_mutex();
    if (g_state.current_memory_usage >= size) {
        g_state.current_memory_usage -= size;
    }
    unlock_mutex();
#endif
}

// ============================================================================
// Part 1: Thread Synchronization
// ============================================================================

/**
 * @brief Lock mutex
 */
static void lock_mutex(void) {
#ifdef _WIN32
    EnterCriticalSection(&g_state.mutex);
#else
    pthread_mutex_lock(&g_state.mutex);
#endif
}

/**
 * @brief Unlock mutex
 */
static void unlock_mutex(void) {
#ifdef _WIN32
    LeaveCriticalSection(&g_state.mutex);
#else
    pthread_mutex_unlock(&g_state.mutex);
#endif
}

// ============================================================================
// Part 1: Threading Initialization
// ============================================================================

/**
 * @brief Initialize threading
 */
static DemosaicError init_threading(void) {
    // Detect number of CPU cores
    int num_cores = 1;
    
#ifdef _WIN32
    SYSTEM_INFO sysinfo;
    GetSystemInfo(&sysinfo);
    num_cores = sysinfo.dwNumberOfProcessors;
#else
    num_cores = sysconf(_SC_NPROCESSORS_ONLN);
#endif
    
    if (g_state.num_threads == 0) {
        // Auto-detect: use number of cores
        g_state.num_threads = num_cores;
    }
    
    // Clamp to reasonable range
    if (g_state.num_threads < 1) {
        g_state.num_threads = 1;
    } else if (g_state.num_threads > MAX_THREADS) {
        g_state.num_threads = MAX_THREADS;
    }
    
    log_message(1, "Threading initialized: %d threads (CPU cores: %d)",
                g_state.num_threads, num_cores);
    
#ifdef DEMOSAIC_ENABLE_OPENMP
    omp_set_num_threads(g_state.num_threads);
    log_message(1, "OpenMP enabled with %d threads", g_state.num_threads);
#endif
    
    return DEMOSAIC_SUCCESS;
}

/**
 * @brief Set number of threads
 */
void demosaic_set_num_threads(int num_threads) {
    if (num_threads < 0) {
        num_threads = 0; // Auto-detect
    } else if (num_threads > MAX_THREADS) {
        num_threads = MAX_THREADS;
    }
    
    g_state.num_threads = num_threads;
    
    if (num_threads == 0) {
        init_threading();
    }
    
#ifdef DEMOSAIC_ENABLE_OPENMP
    omp_set_num_threads(g_state.num_threads);
#endif
}

/**
 * @brief Get number of threads
 */
int demosaic_get_num_threads(void) {
    return g_state.num_threads;
}

// ============================================================================
// End of Part 1
// ============================================================================
// ============================================================================
// Part 2: Hardware Detection
// ============================================================================

/**
 * @brief Initialize hardware detection
 */
static DemosaicError init_hardware_detection(void) {
    log_message(1, "Detecting hardware capabilities...");
    
    // Initialize capabilities structure
    memset(&g_state.hw_caps, 0, sizeof(HardwareCapabilities));
    
    // Detect CPU features
    detect_cpu_features();
    
    // Detect GPU devices
    detect_gpu_devices();
    
    return DEMOSAIC_SUCCESS;
}

/**
 * @brief Detect CPU features
 */
static void detect_cpu_features(void) {
    g_state.hw_caps.has_sse2 = false;
    g_state.hw_caps.has_sse4 = false;
    g_state.hw_caps.has_avx = false;
    g_state.hw_caps.has_avx2 = false;
    g_state.hw_caps.has_neon = false;
    
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    // x86/x64 CPU detection
    int cpuinfo[4];
    
#ifdef _WIN32
    __cpuid(cpuinfo, 1);
#else
    __cpuid(1, cpuinfo[0], cpuinfo[1], cpuinfo[2], cpuinfo[3]);
#endif
    
    // Check SSE2 (bit 26 of EDX)
    g_state.hw_caps.has_sse2 = (cpuinfo[3] & (1 << 26)) != 0;
    
    // Check SSE4.1 (bit 19 of ECX)
    g_state.hw_caps.has_sse4 = (cpuinfo[2] & (1 << 19)) != 0;
    
    // Check AVX (bit 28 of ECX)
    g_state.hw_caps.has_avx = (cpuinfo[2] & (1 << 28)) != 0;
    
    // Check AVX2
#ifdef _WIN32
    __cpuid(cpuinfo, 7);
#else
    __cpuid_count(7, 0, cpuinfo[0], cpuinfo[1], cpuinfo[2], cpuinfo[3]);
#endif
    g_state.hw_caps.has_avx2 = (cpuinfo[1] & (1 << 5)) != 0;
    
#elif defined(__ARM_NEON) || defined(__aarch64__)
    // ARM NEON detection
    g_state.hw_caps.has_neon = true;
#endif
    
    log_message(1, "CPU Features:");
    log_message(1, "  SSE2: %s", g_state.hw_caps.has_sse2 ? "Yes" : "No");
    log_message(1, "  SSE4: %s", g_state.hw_caps.has_sse4 ? "Yes" : "No");
    log_message(1, "  AVX: %s", g_state.hw_caps.has_avx ? "Yes" : "No");
    log_message(1, "  AVX2: %s", g_state.hw_caps.has_avx2 ? "Yes" : "No");
    log_message(1, "  NEON: %s", g_state.hw_caps.has_neon ? "Yes" : "No");
}

/**
 * @brief Detect GPU devices
 */
static void detect_gpu_devices(void) {
    g_state.hw_caps.num_cuda_devices = 0;
    g_state.hw_caps.num_opencl_devices = 0;
    
#ifdef DEMOSAIC_ENABLE_CUDA
    cudaError_t cuda_err = cudaGetDeviceCount(&g_state.hw_caps.num_cuda_devices);
    if (cuda_err == cudaSuccess && g_state.hw_caps.num_cuda_devices > 0) {
        log_message(1, "Found %d CUDA device(s)", g_state.hw_caps.num_cuda_devices);
    }
#endif
    
#ifdef DEMOSAIC_ENABLE_OPENCL
    cl_uint num_platforms = 0;
    cl_int cl_err = clGetPlatformIDs(0, NULL, &num_platforms);
    if (cl_err == CL_SUCCESS && num_platforms > 0) {
        cl_platform_id *platforms = malloc(num_platforms * sizeof(cl_platform_id));
        clGetPlatformIDs(num_platforms, platforms, NULL);
        
        for (cl_uint i = 0; i < num_platforms; i++) {
            cl_uint num_devices = 0;
            clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 0, NULL, &num_devices);
            g_state.hw_caps.num_opencl_devices += num_devices;
        }
        
        free(platforms);
        log_message(1, "Found %d OpenCL device(s)", g_state.hw_caps.num_opencl_devices);
    }
#endif
}

/**
 * @brief Query hardware capabilities
 */
DemosaicError demosaic_query_hardware(HardwareCapabilities *capabilities) {
    if (capabilities == NULL) {
        return DEMOSAIC_ERROR_INVALID_ARGUMENT;
    }
    
    if (!g_state.initialized) {
        return DEMOSAIC_ERROR_NOT_INITIALIZED;
    }
    
    memcpy(capabilities, &g_state.hw_caps, sizeof(HardwareCapabilities));
    return DEMOSAIC_SUCCESS;
}

// ============================================================================
// Part 2: CUDA Initialization and Management
// ============================================================================

#ifdef DEMOSAIC_ENABLE_CUDA

/**
 * @brief Initialize CUDA
 */
static DemosaicError init_cuda(void) {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    
    if (err != cudaSuccess || device_count == 0) {
        log_message(1, "No CUDA devices available");
        return DEMOSAIC_ERROR_CUDA_ERROR;
    }
    
    // Set default device
    err = cudaSetDevice(g_state.cuda_device);
    if (err != cudaSuccess) {
        log_message(2, "Failed to set CUDA device %d", g_state.cuda_device);
        return DEMOSAIC_ERROR_CUDA_ERROR;
    }
    
    // Get device properties
    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, g_state.cuda_device);
    if (err == cudaSuccess) {
        log_message(1, "CUDA Device %d: %s", g_state.cuda_device, prop.name);
        log_message(1, "  Compute Capability: %d.%d", prop.major, prop.minor);
        log_message(1, "  Total Memory: %.2f GB", 
                    prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
        log_message(1, "  Multiprocessors: %d", prop.multiProcessorCount);
    }
    
    g_state.cuda_initialized = true;
    return DEMOSAIC_SUCCESS;
}

/**
 * @brief Cleanup CUDA
 */
static void cleanup_cuda(void) {
    if (g_state.cuda_initialized) {
        cudaDeviceReset();
        g_state.cuda_initialized = false;
        log_message(1, "CUDA cleanup complete");
    }
}

/**
 * @brief Get number of CUDA devices
 */
int demosaic_get_cuda_device_count(void) {
    int count = 0;
    cudaGetDeviceCount(&count);
    return count;
}

/**
 * @brief Get CUDA device information
 */
DemosaicError demosaic_get_cuda_device_info(int device_id, CUDADeviceInfo *info) {
    if (info == NULL) {
        return DEMOSAIC_ERROR_INVALID_ARGUMENT;
    }
    
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_id < 0 || device_id >= device_count) {
        return DEMOSAIC_ERROR_INVALID_ARGUMENT;
    }
    
    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, device_id);
    if (err != cudaSuccess) {
        return DEMOSAIC_ERROR_CUDA_ERROR;
    }
    
    // Fill info structure
    strncpy(info->name, prop.name, sizeof(info->name) - 1);
    info->name[sizeof(info->name) - 1] = '\0';
    info->compute_capability_major = prop.major;
    info->compute_capability_minor = prop.minor;
    info->total_memory = prop.totalGlobalMem;
    info->multiprocessor_count = prop.multiProcessorCount;
    info->max_threads_per_block = prop.maxThreadsPerBlock;
    info->max_threads_per_multiprocessor = prop.maxThreadsPerMultiProcessor;
    info->clock_rate = prop.clockRate;
    info->memory_clock_rate = prop.memoryClockRate;
    info->memory_bus_width = prop.memoryBusWidth;
    
    return DEMOSAIC_SUCCESS;
}

/**
 * @brief Set active CUDA device
 */
DemosaicError demosaic_set_cuda_device(int device_id) {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_id < 0 || device_id >= device_count) {
        return DEMOSAIC_ERROR_INVALID_ARGUMENT;
    }
    
    err = cudaSetDevice(device_id);
    if (err != cudaSuccess) {
        set_error(DEMOSAIC_ERROR_CUDA_ERROR, "Failed to set CUDA device %d", device_id);
        return DEMOSAIC_ERROR_CUDA_ERROR;
    }
    
    g_state.cuda_device = device_id;
    log_message(1, "Active CUDA device set to %d", device_id);
    
    return DEMOSAIC_SUCCESS;
}

#else // !DEMOSAIC_ENABLE_CUDA

static DemosaicError init_cuda(void) {
    return DEMOSAIC_ERROR_CUDA_ERROR;
}

static void cleanup_cuda(void) {
}

int demosaic_get_cuda_device_count(void) {
    return 0;
}

DemosaicError demosaic_get_cuda_device_info(int device_id, CUDADeviceInfo *info) {
    return DEMOSAIC_ERROR_CUDA_ERROR;
}

DemosaicError demosaic_set_cuda_device(int device_id) {
    return DEMOSAIC_ERROR_CUDA_ERROR;
}

#endif // DEMOSAIC_ENABLE_CUDA

// ============================================================================
// End of Part 2A
// ============================================================================
// ============================================================================
// Part 2B: OpenCL Initialization and Management
// ============================================================================

#ifdef DEMOSAIC_ENABLE_OPENCL

/**
 * @brief Initialize OpenCL
 */
static DemosaicError init_opencl(void) {
    cl_int err;
    cl_uint num_platforms = 0;
    
    // Get number of platforms
    err = clGetPlatformIDs(0, NULL, &num_platforms);
    if (err != CL_SUCCESS || num_platforms == 0) {
        log_message(1, "No OpenCL platforms available");
        return DEMOSAIC_ERROR_OPENCL_ERROR;
    }
    
    // Get platforms
    cl_platform_id *platforms = malloc(num_platforms * sizeof(cl_platform_id));
    err = clGetPlatformIDs(num_platforms, platforms, NULL);
    if (err != CL_SUCCESS) {
        free(platforms);
        return DEMOSAIC_ERROR_OPENCL_ERROR;
    }
    
    // Use first platform with GPU devices
    for (cl_uint i = 0; i < num_platforms; i++) {
        cl_uint num_devices = 0;
        err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 0, NULL, &num_devices);
        
        if (err == CL_SUCCESS && num_devices > 0) {
            g_state.opencl_platform = platforms[i];
            
            // Get first GPU device
            cl_device_id device;
            err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 1, &device, NULL);
            if (err == CL_SUCCESS) {
                g_state.opencl_device = device;
                
                // Create context
                g_state.opencl_context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
                if (err != CL_SUCCESS) {
                    free(platforms);
                    return DEMOSAIC_ERROR_OPENCL_ERROR;
                }
                
                // Create command queue
                g_state.opencl_queue = clCreateCommandQueue(
                    g_state.opencl_context, device, 0, &err);
                if (err != CL_SUCCESS) {
                    clReleaseContext(g_state.opencl_context);
                    free(platforms);
                    return DEMOSAIC_ERROR_OPENCL_ERROR;
                }
                
                // Get device info
                char device_name[256];
                clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(device_name), 
                               device_name, NULL);
                log_message(1, "OpenCL Device: %s", device_name);
                
                g_state.opencl_initialized = true;
                free(platforms);
                return DEMOSAIC_SUCCESS;
            }
        }
    }
    
    free(platforms);
    log_message(1, "No suitable OpenCL GPU devices found");
    return DEMOSAIC_ERROR_OPENCL_ERROR;
}

/**
 * @brief Cleanup OpenCL
 */
static void cleanup_opencl(void) {
    if (g_state.opencl_initialized) {
        if (g_state.opencl_queue) {
            clReleaseCommandQueue(g_state.opencl_queue);
        }
        if (g_state.opencl_context) {
            clReleaseContext(g_state.opencl_context);
        }
        g_state.opencl_initialized = false;
        log_message(1, "OpenCL cleanup complete");
    }
}

/**
 * @brief Get number of OpenCL devices
 */
int demosaic_get_opencl_device_count(void) {
    cl_uint num_platforms = 0;
    cl_int err = clGetPlatformIDs(0, NULL, &num_platforms);
    if (err != CL_SUCCESS) {
        return 0;
    }
    
    cl_platform_id *platforms = malloc(num_platforms * sizeof(cl_platform_id));
    clGetPlatformIDs(num_platforms, platforms, NULL);
    
    int total_devices = 0;
    for (cl_uint i = 0; i < num_platforms; i++) {
        cl_uint num_devices = 0;
        clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 0, NULL, &num_devices);
        total_devices += num_devices;
    }
    
    free(platforms);
    return total_devices;
}

/**
 * @brief Get OpenCL device information
 */
DemosaicError demosaic_get_opencl_device_info(
    int platform_id,
    int device_id,
    OpenCLDeviceInfo *info
) {
    if (info == NULL) {
        return DEMOSAIC_ERROR_INVALID_ARGUMENT;
    }
    
    cl_uint num_platforms = 0;
    cl_int err = clGetPlatformIDs(0, NULL, &num_platforms);
    if (err != CL_SUCCESS || platform_id < 0 || (cl_uint)platform_id >= num_platforms) {
        return DEMOSAIC_ERROR_INVALID_ARGUMENT;
    }
    
    cl_platform_id *platforms = malloc(num_platforms * sizeof(cl_platform_id));
    clGetPlatformIDs(num_platforms, platforms, NULL);
    
    cl_platform_id platform = platforms[platform_id];
    
    cl_uint num_devices = 0;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, &num_devices);
    if (err != CL_SUCCESS || device_id < 0 || (cl_uint)device_id >= num_devices) {
        free(platforms);
        return DEMOSAIC_ERROR_INVALID_ARGUMENT;
    }
    
    cl_device_id *devices = malloc(num_devices * sizeof(cl_device_id));
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, num_devices, devices, NULL);
    
    cl_device_id device = devices[device_id];
    
    // Get device information
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(info->name), info->name, NULL);
    clGetDeviceInfo(device, CL_DEVICE_VENDOR, sizeof(info->vendor), info->vendor, NULL);
    clGetDeviceInfo(device, CL_DEVICE_VERSION, sizeof(info->version), info->version, NULL);
    clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(info->global_memory), 
                   &info->global_memory, NULL);
    clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(info->local_memory), 
                   &info->local_memory, NULL);
    clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(info->compute_units), 
                   &info->compute_units, NULL);
    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(info->max_work_group_size), 
                   &info->max_work_group_size, NULL);
    clGetDeviceInfo(device, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(info->max_clock_frequency), 
                   &info->max_clock_frequency, NULL);
    
    free(devices);
    free(platforms);
    
    return DEMOSAIC_SUCCESS;
}

/**
 * @brief Set active OpenCL device
 */
DemosaicError demosaic_set_opencl_device(int platform_id, int device_id) {
    // Cleanup existing OpenCL state
    if (g_state.opencl_initialized) {
        cleanup_opencl();
    }
    
    cl_uint num_platforms = 0;
    cl_int err = clGetPlatformIDs(0, NULL, &num_platforms);
    if (err != CL_SUCCESS || platform_id < 0 || (cl_uint)platform_id >= num_platforms) {
        return DEMOSAIC_ERROR_INVALID_ARGUMENT;
    }
    
    cl_platform_id *platforms = malloc(num_platforms * sizeof(cl_platform_id));
    clGetPlatformIDs(num_platforms, platforms, NULL);
    
    cl_platform_id platform = platforms[platform_id];
    
    cl_uint num_devices = 0;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, &num_devices);
    if (err != CL_SUCCESS || device_id < 0 || (cl_uint)device_id >= num_devices) {
        free(platforms);
        return DEMOSAIC_ERROR_INVALID_ARGUMENT;
    }
    
    cl_device_id *devices = malloc(num_devices * sizeof(cl_device_id));
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, num_devices, devices, NULL);
    
    cl_device_id device = devices[device_id];
    
    // Create context and command queue
    g_state.opencl_context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    if (err != CL_SUCCESS) {
        free(devices);
        free(platforms);
        return DEMOSAIC_ERROR_OPENCL_ERROR;
    }
    
    g_state.opencl_queue = clCreateCommandQueue(g_state.opencl_context, device, 0, &err);
    if (err != CL_SUCCESS) {
        clReleaseContext(g_state.opencl_context);
        free(devices);
        free(platforms);
        return DEMOSAIC_ERROR_OPENCL_ERROR;
    }
    
    g_state.opencl_platform = platform;
    g_state.opencl_device = device;
    g_state.opencl_initialized = true;
    
    free(devices);
    free(platforms);
    
    log_message(1, "Active OpenCL device set to platform %d, device %d", 
                platform_id, device_id);
    
    return DEMOSAIC_SUCCESS;
}

#else // !DEMOSAIC_ENABLE_OPENCL

static DemosaicError init_opencl(void) {
    return DEMOSAIC_ERROR_OPENCL_ERROR;
}

static void cleanup_opencl(void) {
}

int demosaic_get_opencl_device_count(void) {
    return 0;
}

DemosaicError demosaic_get_opencl_device_info(
    int platform_id,
    int device_id,
    OpenCLDeviceInfo *info
) {
    return DEMOSAIC_ERROR_OPENCL_ERROR;
}

DemosaicError demosaic_set_opencl_device(int platform_id, int device_id) {
    return DEMOSAIC_ERROR_OPENCL_ERROR;
}

#endif // DEMOSAIC_ENABLE_OPENCL

// ============================================================================
// Part 2B: SIMD Management
// ============================================================================

/**
 * @brief Enable/disable SIMD optimization
 */
void demosaic_set_simd_enabled(bool enable) {
    g_state.simd_enabled = enable;
    log_message(1, "SIMD optimization %s", enable ? "enabled" : "disabled");
}

/**
 * @brief Check if SIMD is enabled
 */
bool demosaic_is_simd_enabled(void) {
    return g_state.simd_enabled;
}

// ============================================================================
// Part 2B: Bayer Image Management
// ============================================================================

/**
 * @brief Create a new Bayer image
 */
BayerImage* demosaic_bayer_create(
    int width,
    int height,
    BayerPattern pattern,
    int bit_depth
) {
    // Validate parameters
    if (width <= 0 || height <= 0) {
        set_error(DEMOSAIC_ERROR_INVALID_ARGUMENT, "Invalid image dimensions");
        return NULL;
    }
    
    if (pattern < BAYER_RGGB || pattern > BAYER_BGGR) {
        set_error(DEMOSAIC_ERROR_INVALID_PATTERN, "Invalid Bayer pattern");
        return NULL;
    }
    
    if (bit_depth != 8 && bit_depth != 10 && bit_depth != 12 && 
        bit_depth != 14 && bit_depth != 16) {
        set_error(DEMOSAIC_ERROR_INVALID_ARGUMENT, "Invalid bit depth");
        return NULL;
    }
    
    // Allocate image structure
    BayerImage *image = (BayerImage*)malloc(sizeof(BayerImage));
    if (image == NULL) {
        set_error(DEMOSAIC_ERROR_OUT_OF_MEMORY, "Failed to allocate BayerImage");
        return NULL;
    }
    
    // Initialize fields
    image->width = width;
    image->height = height;
    image->pattern = pattern;
    image->bit_depth = bit_depth;
    image->stride = width; // Default stride
    image->owns_data = true;
    
    // Allocate data buffer (aligned for SIMD)
    size_t data_size = (size_t)width * height * sizeof(uint16_t);
    image->data = (uint16_t*)demosaic_aligned_alloc(data_size, ALIGNMENT);
    
    if (image->data == NULL) {
        free(image);
        set_error(DEMOSAIC_ERROR_OUT_OF_MEMORY, "Failed to allocate image data");
        return NULL;
    }
    
    // Initialize data to zero
    memset(image->data, 0, data_size);
    
    track_allocation(sizeof(BayerImage) + data_size);
    
    log_message(0, "Created Bayer image: %dx%d, pattern=%d, bit_depth=%d",
                width, height, pattern, bit_depth);
    
    return image;
}

/**
 * @brief Create Bayer image from existing data
 */
BayerImage* demosaic_bayer_create_from_data(
    const uint16_t *data,
    int width,
    int height,
    BayerPattern pattern,
    int bit_depth,
    bool copy_data
) {
    if (data == NULL) {
        set_error(DEMOSAIC_ERROR_INVALID_ARGUMENT, "Data pointer is NULL");
        return NULL;
    }
    
    BayerImage *image = demosaic_bayer_create(width, height, pattern, bit_depth);
    if (image == NULL) {
        return NULL;
    }
    
    if (copy_data) {
        // Copy data
        size_t data_size = (size_t)width * height * sizeof(uint16_t);
        memcpy(image->data, data, data_size);
    } else {
        // Use existing data pointer (not recommended for external data)
        demosaic_aligned_free(image->data);
        image->data = (uint16_t*)data;
        image->owns_data = false;
    }
    
    return image;
}

/**
 * @brief Destroy Bayer image
 */
void demosaic_bayer_destroy(BayerImage *image) {
    if (image == NULL) {
        return;
    }
    
    size_t data_size = (size_t)image->width * image->height * sizeof(uint16_t);
    
    if (image->owns_data && image->data != NULL) {
        demosaic_aligned_free(image->data);
        track_deallocation(data_size);
    }
    
    track_deallocation(sizeof(BayerImage));
    free(image);
    
    log_message(0, "Destroyed Bayer image");
}

/**
 * @brief Clone a Bayer image
 */
BayerImage* demosaic_bayer_clone(const BayerImage *image) {
    if (image == NULL) {
        set_error(DEMOSAIC_ERROR_INVALID_ARGUMENT, "Image is NULL");
        return NULL;
    }
    
    return demosaic_bayer_create_from_data(
        image->data,
        image->width,
        image->height,
        image->pattern,
        image->bit_depth,
        true // Copy data
    );
}

/**
 * @brief Validate Bayer image
 */
DemosaicError demosaic_bayer_validate(const BayerImage *image) {
    if (image == NULL) {
        return DEMOSAIC_ERROR_INVALID_ARGUMENT;
    }
    
    if (image->width <= 0 || image->height <= 0) {
        return DEMOSAIC_ERROR_INVALID_IMAGE;
    }
    
    if (image->pattern < BAYER_RGGB || image->pattern > BAYER_BGGR) {
        return DEMOSAIC_ERROR_INVALID_PATTERN;
    }
    
    if (image->data == NULL) {
        return DEMOSAIC_ERROR_INVALID_IMAGE;
    }
    
    if (image->stride < image->width) {
        return DEMOSAIC_ERROR_INVALID_IMAGE;
    }
    
    return DEMOSAIC_SUCCESS;
}

// ============================================================================
// Part 2B: RGB Image Management
// ============================================================================

/**
 * @brief Create a new RGB image
 */
RGBImage* demosaic_rgb_create(int width, int height, int channels) {
    // Validate parameters
    if (width <= 0 || height <= 0) {
        set_error(DEMOSAIC_ERROR_INVALID_ARGUMENT, "Invalid image dimensions");
        return NULL;
    }
    
    if (channels < 1 || channels > 4) {
        set_error(DEMOSAIC_ERROR_INVALID_ARGUMENT, "Invalid number of channels");
        return NULL;
    }
    
    // Allocate image structure
    RGBImage *image = (RGBImage*)malloc(sizeof(RGBImage));
    if (image == NULL) {
        set_error(DEMOSAIC_ERROR_OUT_OF_MEMORY, "Failed to allocate RGBImage");
        return NULL;
    }
    
    // Initialize fields
    image->width = width;
    image->height = height;
    image->channels = channels;
    image->stride = width * channels; // Default stride
    image->is_planar = false;
    image->owns_data = true;
    image->color_space = COLOR_SPACE_SRGB;
    
    // Allocate data buffer (aligned for SIMD)
    size_t data_size = (size_t)width * height * channels * sizeof(double);
    image->data = (double*)demosaic_aligned_alloc(data_size, ALIGNMENT);
    
    if (image->data == NULL) {
        free(image);
        set_error(DEMOSAIC_ERROR_OUT_OF_MEMORY, "Failed to allocate image data");
        return NULL;
    }
    
    // Initialize data to zero
    memset(image->data, 0, data_size);
    
    track_allocation(sizeof(RGBImage) + data_size);
    
    log_message(0, "Created RGB image: %dx%d, channels=%d",
                width, height, channels);
    
    return image;
}

/**
 * @brief Create RGB image from existing data
 */
RGBImage* demosaic_rgb_create_from_data(
    const double *data,
    int width,
    int height,
    int channels,
    bool copy_data
) {
    if (data == NULL) {
        set_error(DEMOSAIC_ERROR_INVALID_ARGUMENT, "Data pointer is NULL");
        return NULL;
    }
    
    RGBImage *image = demosaic_rgb_create(width, height, channels);
    if (image == NULL) {
        return NULL;
    }
    
    if (copy_data) {
        // Copy data
        size_t data_size = (size_t)width * height * channels * sizeof(double);
        memcpy(image->data, data, data_size);
    } else {
        // Use existing data pointer
        demosaic_aligned_free(image->data);
        image->data = (double*)data;
        image->owns_data = false;
    }
    
    return image;
}

/**
 * @brief Destroy RGB image
 */
void demosaic_rgb_destroy(RGBImage *image) {
    if (image == NULL) {
        return;
    }
    
    size_t data_size = (size_t)image->width * image->height * 
                       image->channels * sizeof(double);
    
    if (image->owns_data && image->data != NULL) {
        demosaic_aligned_free(image->data);
        track_deallocation(data_size);
    }
    
    track_deallocation(sizeof(RGBImage));
    free(image);
    
    log_message(0, "Destroyed RGB image");
}

/**
 * @brief Clone an RGB image
 */
RGBImage* demosaic_rgb_clone(const RGBImage *image) {
    if (image == NULL) {
        set_error(DEMOSAIC_ERROR_INVALID_ARGUMENT, "Image is NULL");
        return NULL;
    }
    
    RGBImage *clone = demosaic_rgb_create_from_data(
        image->data,
        image->width,
        image->height,
        image->channels,
        true // Copy data
    );
    
    if (clone != NULL) {
        clone->is_planar = image->is_planar;
        clone->color_space = image->color_space;
        clone->stride = image->stride;
    }
    
    return clone;
}

/**
 * @brief Validate RGB image
 */
DemosaicError demosaic_rgb_validate(const RGBImage *image) {
    if (image == NULL) {
        return DEMOSAIC_ERROR_INVALID_ARGUMENT;
    }
    
    if (image->width <= 0 || image->height <= 0) {
        return DEMOSAIC_ERROR_INVALID_IMAGE;
    }
    
    if (image->channels < 1 || image->channels > 4) {
        return DEMOSAIC_ERROR_INVALID_IMAGE;
    }
    
    if (image->data == NULL) {
        return DEMOSAIC_ERROR_INVALID_IMAGE;
    }
    
    if (image->stride < image->width * image->channels) {
        return DEMOSAIC_ERROR_INVALID_IMAGE;
    }
    
    return DEMOSAIC_SUCCESS;
}

/**
 * @brief Convert RGB image layout (interleaved <-> planar)
 */
DemosaicError demosaic_rgb_convert_layout(RGBImage *image, bool to_planar) {
    if (image == NULL) {
        return DEMOSAIC_ERROR_INVALID_ARGUMENT;
    }
    
    if (image->is_planar == to_planar) {
        return DEMOSAIC_SUCCESS; // Already in desired format
    }
    
    int width = image->width;
    int height = image->height;
    int channels = image->channels;
    
    // Allocate temporary buffer
    size_t data_size = (size_t)width * height * channels * sizeof(double);
    double *temp = (double*)demosaic_aligned_alloc(data_size, ALIGNMENT);
    if (temp == NULL) {
        return DEMOSAIC_ERROR_OUT_OF_MEMORY;
    }
    
    if (to_planar) {
        // Convert interleaved to planar
        for (int c = 0; c < channels; c++) {
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    int src_idx = (y * width + x) * channels + c;
                    int dst_idx = c * width * height + y * width + x;
                    temp[dst_idx] = image->data[src_idx];
                }
            }
        }
    } else {
        // Convert planar to interleaved
        for (int c = 0; c < channels; c++) {
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    int src_idx = c * width * height + y * width + x;
                    int dst_idx = (y * width + x) * channels + c;
                    temp[dst_idx] = image->data[src_idx];
                }
            }
        }
    }
    
    // Copy back
    memcpy(image->data, temp, data_size);
    demosaic_aligned_free(temp);
    
    image->is_planar = to_planar;
    
    return DEMOSAIC_SUCCESS;
}

// ============================================================================
// End of Part 2B
// ============================================================================
// ============================================================================
// Part 3: Configuration Management
// ============================================================================

/**
 * @brief Create default configuration
 */
DemosaicConfig* demosaic_config_create(void) {
    DemosaicConfig *config = (DemosaicConfig*)malloc(sizeof(DemosaicConfig));
    if (config == NULL) {
        set_error(DEMOSAIC_ERROR_OUT_OF_MEMORY, "Failed to allocate config");
        return NULL;
    }
    
    // Set default values
    config->algorithm = DEMOSAIC_BILINEAR;
    config->backend = BACKEND_AUTO;
    config->quality = QUALITY_BALANCED;
    config->num_threads = 0; // Auto-detect
    config->use_simd = true;
    config->tile_size = 256;
    config->edge_threshold = 10.0;
    config->color_correction = true;
    config->denoise_strength = 0.0;
    config->sharpen_amount = 0.0;
    config->output_color_space = COLOR_SPACE_SRGB;
    config->gamma = 2.2;
    config->white_balance_r = 1.0;
    config->white_balance_g = 1.0;
    config->white_balance_b = 1.0;
    config->exposure_compensation = 0.0;
    config->saturation = 1.0;
    config->contrast = 1.0;
    config->brightness = 0.0;
    config->cache_enabled = true;
    config->progress_callback = NULL;
    config->progress_user_data = NULL;
    
    log_message(0, "Created default configuration");
    
    return config;
}

/**
 * @brief Clone configuration
 */
DemosaicConfig* demosaic_config_clone(const DemosaicConfig *config) {
    if (config == NULL) {
        set_error(DEMOSAIC_ERROR_INVALID_ARGUMENT, "Config is NULL");
        return NULL;
    }
    
    DemosaicConfig *clone = (DemosaicConfig*)malloc(sizeof(DemosaicConfig));
    if (clone == NULL) {
        set_error(DEMOSAIC_ERROR_OUT_OF_MEMORY, "Failed to allocate config");
        return NULL;
    }
    
    memcpy(clone, config, sizeof(DemosaicConfig));
    
    return clone;
}

/**
 * @brief Destroy configuration
 */
void demosaic_config_destroy(DemosaicConfig *config) {
    if (config != NULL) {
        free(config);
        log_message(0, "Destroyed configuration");
    }
}

/**
 * @brief Validate configuration
 */
DemosaicError demosaic_config_validate(const DemosaicConfig *config) {
    if (config == NULL) {
        return DEMOSAIC_ERROR_INVALID_ARGUMENT;
    }
    
    // Validate algorithm
    if (config->algorithm < DEMOSAIC_NEAREST || config->algorithm > DEMOSAIC_CUSTOM) {
        set_error(DEMOSAIC_ERROR_INVALID_ARGUMENT, "Invalid algorithm");
        return DEMOSAIC_ERROR_INVALID_ARGUMENT;
    }
    
    // Validate backend
    if (config->backend < BACKEND_AUTO || config->backend > BACKEND_OPENCL) {
        set_error(DEMOSAIC_ERROR_INVALID_ARGUMENT, "Invalid backend");
        return DEMOSAIC_ERROR_INVALID_ARGUMENT;
    }
    
    // Validate quality
    if (config->quality < QUALITY_FAST || config->quality > QUALITY_BEST) {
        set_error(DEMOSAIC_ERROR_INVALID_ARGUMENT, "Invalid quality");
        return DEMOSAIC_ERROR_INVALID_ARGUMENT;
    }
    
    // Validate tile size
    if (config->tile_size < 32 || config->tile_size > 2048) {
        set_error(DEMOSAIC_ERROR_INVALID_ARGUMENT, "Invalid tile size");
        return DEMOSAIC_ERROR_INVALID_ARGUMENT;
    }
    
    // Validate edge threshold
    if (config->edge_threshold < 0.0 || config->edge_threshold > 255.0) {
        set_error(DEMOSAIC_ERROR_INVALID_ARGUMENT, "Invalid edge threshold");
        return DEMOSAIC_ERROR_INVALID_ARGUMENT;
    }
    
    // Validate denoise strength
    if (config->denoise_strength < 0.0 || config->denoise_strength > 1.0) {
        set_error(DEMOSAIC_ERROR_INVALID_ARGUMENT, "Invalid denoise strength");
        return DEMOSAIC_ERROR_INVALID_ARGUMENT;
    }
    
    // Validate sharpen amount
    if (config->sharpen_amount < 0.0 || config->sharpen_amount > 2.0) {
        set_error(DEMOSAIC_ERROR_INVALID_ARGUMENT, "Invalid sharpen amount");
        return DEMOSAIC_ERROR_INVALID_ARGUMENT;
    }
    
    // Validate gamma
    if (config->gamma <= 0.0 || config->gamma > 5.0) {
        set_error(DEMOSAIC_ERROR_INVALID_ARGUMENT, "Invalid gamma");
        return DEMOSAIC_ERROR_INVALID_ARGUMENT;
    }
    
    // Validate white balance
    if (config->white_balance_r <= 0.0 || config->white_balance_r > 10.0 ||
        config->white_balance_g <= 0.0 || config->white_balance_g > 10.0 ||
        config->white_balance_b <= 0.0 || config->white_balance_b > 10.0) {
        set_error(DEMOSAIC_ERROR_INVALID_ARGUMENT, "Invalid white balance");
        return DEMOSAIC_ERROR_INVALID_ARGUMENT;
    }
    
    // Validate saturation
    if (config->saturation < 0.0 || config->saturation > 3.0) {
        set_error(DEMOSAIC_ERROR_INVALID_ARGUMENT, "Invalid saturation");
        return DEMOSAIC_ERROR_INVALID_ARGUMENT;
    }
    
    // Validate contrast
    if (config->contrast < 0.0 || config->contrast > 3.0) {
        set_error(DEMOSAIC_ERROR_INVALID_ARGUMENT, "Invalid contrast");
        return DEMOSAIC_ERROR_INVALID_ARGUMENT;
    }
    
    return DEMOSAIC_SUCCESS;
}

/**
 * @brief Set configuration preset
 */
DemosaicError demosaic_config_set_preset(DemosaicConfig *config, QualityPreset preset) {
    if (config == NULL) {
        return DEMOSAIC_ERROR_INVALID_ARGUMENT;
    }
    
    config->quality = preset;
    
    switch (preset) {
        case QUALITY_FAST:
            config->algorithm = DEMOSAIC_BILINEAR;
            config->tile_size = 512;
            config->color_correction = false;
            config->denoise_strength = 0.0;
            config->sharpen_amount = 0.0;
            break;
            
        case QUALITY_BALANCED:
            config->algorithm = DEMOSAIC_VNG;
            config->tile_size = 256;
            config->color_correction = true;
            config->denoise_strength = 0.1;
            config->sharpen_amount = 0.3;
            break;
            
        case QUALITY_HIGH:
            config->algorithm = DEMOSAIC_AHD;
            config->tile_size = 128;
            config->color_correction = true;
            config->denoise_strength = 0.2;
            config->sharpen_amount = 0.5;
            break;
            
        case QUALITY_BEST:
            config->algorithm = DEMOSAIC_LMMSE;
            config->tile_size = 64;
            config->color_correction = true;
            config->denoise_strength = 0.3;
            config->sharpen_amount = 0.7;
            break;
            
        default:
            return DEMOSAIC_ERROR_INVALID_ARGUMENT;
    }
    
    log_message(1, "Set configuration preset: %d", preset);
    
    return DEMOSAIC_SUCCESS;
}

// ============================================================================
// Part 3: Image Access Helper Functions
// ============================================================================

/**
 * @brief Get pixel value from Bayer image (with bounds checking)
 */
static inline uint16_t get_bayer_pixel_safe(
    const BayerImage *image,
    int x,
    int y
) {
    // Clamp coordinates
    x = (x < 0) ? 0 : (x >= image->width) ? image->width - 1 : x;
    y = (y < 0) ? 0 : (y >= image->height) ? image->height - 1 : y;
    
    return image->data[y * image->stride + x];
}

/**
 * @brief Get pixel value from Bayer image (no bounds checking)
 */
static inline uint16_t get_bayer_pixel(
    const BayerImage *image,
    int x,
    int y
) {
    return image->data[y * image->stride + x];
}

/**
 * @brief Set pixel value in RGB image
 */
static inline void set_rgb_pixel(
    RGBImage *image,
    int x,
    int y,
    double r,
    double g,
    double b
) {
    if (image->is_planar) {
        int plane_size = image->width * image->height;
        int idx = y * image->width + x;
        image->data[idx] = r;
        image->data[plane_size + idx] = g;
        image->data[2 * plane_size + idx] = b;
    } else {
        int idx = (y * image->width + x) * image->channels;
        image->data[idx] = r;
        image->data[idx + 1] = g;
        image->data[idx + 2] = b;
    }
}

/**
 * @brief Get RGB pixel value
 */
static inline void get_rgb_pixel(
    const RGBImage *image,
    int x,
    int y,
    double *r,
    double *g,
    double *b
) {
    if (image->is_planar) {
        int plane_size = image->width * image->height;
        int idx = y * image->width + x;
        *r = image->data[idx];
        *g = image->data[plane_size + idx];
        *b = image->data[2 * plane_size + idx];
    } else {
        int idx = (y * image->width + x) * image->channels;
        *r = image->data[idx];
        *g = image->data[idx + 1];
        *b = image->data[idx + 2];
    }
}

/**
 * @brief Determine color at position based on Bayer pattern
 */
static inline int get_bayer_color(const BayerImage *image, int x, int y) {
    int pattern_x = x & 1;
    int pattern_y = y & 1;
    
    switch (image->pattern) {
        case BAYER_RGGB:
            if (pattern_y == 0) {
                return pattern_x == 0 ? 0 : 1; // R or G
            } else {
                return pattern_x == 0 ? 1 : 2; // G or B
            }
            
        case BAYER_GRBG:
            if (pattern_y == 0) {
                return pattern_x == 0 ? 1 : 0; // G or R
            } else {
                return pattern_x == 0 ? 2 : 1; // B or G
            }
            
        case BAYER_GBRG:
            if (pattern_y == 0) {
                return pattern_x == 0 ? 1 : 2; // G or B
            } else {
                return pattern_x == 0 ? 0 : 1; // R or G
            }
            
        case BAYER_BGGR:
            if (pattern_y == 0) {
                return pattern_x == 0 ? 2 : 1; // B or G
            } else {
                return pattern_x == 0 ? 1 : 0; // G or R
            }
            
        default:
            return 1; // Default to green
    }
}

// ============================================================================
// Part 3: Nearest Neighbor Demosaicing
// ============================================================================

/**
 * @brief Nearest neighbor demosaicing (fastest, lowest quality)
 */
static DemosaicError demosaic_nearest_neighbor(
    const BayerImage *bayer,
    RGBImage *rgb,
    const DemosaicConfig *config
) {
    int width = bayer->width;
    int height = bayer->height;
    
    log_message(1, "Running nearest neighbor demosaicing");
    
    // Process each pixel
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            double r, g, b;
            uint16_t center = get_bayer_pixel(bayer, x, y);
            int color = get_bayer_color(bayer, x, y);
            
            // Get neighboring pixels
            uint16_t left = get_bayer_pixel_safe(bayer, x - 1, y);
            uint16_t right = get_bayer_pixel_safe(bayer, x + 1, y);
            uint16_t top = get_bayer_pixel_safe(bayer, x, y - 1);
            uint16_t bottom = get_bayer_pixel_safe(bayer, x, y + 1);
            
            if (color == 0) { // Red pixel
                r = center;
                g = (left + right + top + bottom) / 4.0;
                
                uint16_t tl = get_bayer_pixel_safe(bayer, x - 1, y - 1);
                uint16_t tr = get_bayer_pixel_safe(bayer, x + 1, y - 1);
                uint16_t bl = get_bayer_pixel_safe(bayer, x - 1, y + 1);
                uint16_t br = get_bayer_pixel_safe(bayer, x + 1, y + 1);
                b = (tl + tr + bl + br) / 4.0;
                
            } else if (color == 1) { // Green pixel
                g = center;
                
                // Determine if we're on R row or B row
                int top_color = get_bayer_color(bayer, x, y - 1);
                if (top_color == 0) { // R row above
                    r = (top + bottom) / 2.0;
                    b = (left + right) / 2.0;
                } else { // B row above
                    b = (top + bottom) / 2.0;
                    r = (left + right) / 2.0;
                }
                
            } else { // Blue pixel
                b = center;
                g = (left + right + top + bottom) / 4.0;
                
                uint16_t tl = get_bayer_pixel_safe(bayer, x - 1, y - 1);
                uint16_t tr = get_bayer_pixel_safe(bayer, x + 1, y - 1);
                uint16_t bl = get_bayer_pixel_safe(bayer, x - 1, y + 1);
                uint16_t br = get_bayer_pixel_safe(bayer, x + 1, y + 1);
                r = (tl + tr + bl + br) / 4.0;
            }
            
            // Normalize to [0, 1] range
            double max_val = (1 << bayer->bit_depth) - 1;
            r /= max_val;
            g /= max_val;
            b /= max_val;
            
            set_rgb_pixel(rgb, x, y, r, g, b);
        }
        
        // Report progress
        if (config->progress_callback && (y % 10 == 0)) {
            float progress = (float)y / height;
            config->progress_callback(progress, config->progress_user_data);
        }
    }
    
    // Final progress update
    if (config->progress_callback) {
        config->progress_callback(1.0f, config->progress_user_data);
    }
    
    log_message(1, "Nearest neighbor demosaicing complete");
    
    return DEMOSAIC_SUCCESS;
}

// ============================================================================
// Part 3: Bilinear Demosaicing
// ============================================================================

/**
 * @brief Bilinear interpolation demosaicing
 */
static DemosaicError demosaic_bilinear(
    const BayerImage *bayer,
    RGBImage *rgb,
    const DemosaicConfig *config
) {
    int width = bayer->width;
    int height = bayer->height;
    
    log_message(1, "Running bilinear demosaicing");
    
    // Process each pixel
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            double r, g, b;
            uint16_t center = get_bayer_pixel(bayer, x, y);
            int color = get_bayer_color(bayer, x, y);
            
            if (color == 0) { // Red pixel
                r = center;
                
                // Interpolate green (4 neighbors)
                double g_sum = 0.0;
                int g_count = 0;
                if (x > 0) { g_sum += get_bayer_pixel(bayer, x - 1, y); g_count++; }
                if (x < width - 1) { g_sum += get_bayer_pixel(bayer, x + 1, y); g_count++; }
                if (y > 0) { g_sum += get_bayer_pixel(bayer, x, y - 1); g_count++; }
                if (y < height - 1) { g_sum += get_bayer_pixel(bayer, x, y + 1); g_count++; }
                g = g_sum / g_count;
                
                // Interpolate blue (4 diagonal neighbors)
                double b_sum = 0.0;
                int b_count = 0;
                if (x > 0 && y > 0) { 
                    b_sum += get_bayer_pixel(bayer, x - 1, y - 1); 
                    b_count++; 
                }
                if (x < width - 1 && y > 0) { 
                    b_sum += get_bayer_pixel(bayer, x + 1, y - 1); 
                    b_count++; 
                }
                if (x > 0 && y < height - 1) { 
                    b_sum += get_bayer_pixel(bayer, x - 1, y + 1); 
                    b_count++; 
                }
                if (x < width - 1 && y < height - 1) { 
                    b_sum += get_bayer_pixel(bayer, x + 1, y + 1); 
                    b_count++; 
                }
                b = b_sum / b_count;
                
            } else if (color == 1) { // Green pixel
                g = center;
                
                // Determine orientation
                int left_color = (x > 0) ? get_bayer_color(bayer, x - 1, y) : -1;
                int right_color = (x < width - 1) ? get_bayer_color(bayer, x + 1, y) : -1;
                int top_color = (y > 0) ? get_bayer_color(bayer, x, y - 1) : -1;
                int bottom_color = (y < height - 1) ? get_bayer_color(bayer, x, y + 1) : -1;
                
                // Interpolate red
                double r_sum = 0.0;
                int r_count = 0;
                if (left_color == 0) { r_sum += get_bayer_pixel(bayer, x - 1, y); r_count++; }
                if (right_color == 0) { r_sum += get_bayer_pixel(bayer, x + 1, y); r_count++; }
                if (top_color == 0) { r_sum += get_bayer_pixel(bayer, x, y - 1); r_count++; }
                if (bottom_color == 0) { r_sum += get_bayer_pixel(bayer, x, y + 1); r_count++; }
                r = (r_count > 0) ? r_sum / r_count : center;
                
                // Interpolate blue
                double b_sum = 0.0;
                int b_count = 0;
                if (left_color == 2) { b_sum += get_bayer_pixel(bayer, x - 1, y); b_count++; }
                if (right_color == 2) { b_sum += get_bayer_pixel(bayer, x + 1, y); b_count++; }
                if (top_color == 2) { b_sum += get_bayer_pixel(bayer, x, y - 1); b_count++; }
                if (bottom_color == 2) { b_sum += get_bayer_pixel(bayer, x, y + 1); b_count++; }
                b = (b_count > 0) ? b_sum / b_count : center;
                
            } else { // Blue pixel
                b = center;
                
                // Interpolate green (4 neighbors)
                double g_sum = 0.0;
                int g_count = 0;
                if (x > 0) { g_sum += get_bayer_pixel(bayer, x - 1, y); g_count++; }
                if (x < width - 1) { g_sum += get_bayer_pixel(bayer, x + 1, y); g_count++; }
                if (y > 0) { g_sum += get_bayer_pixel(bayer, x, y - 1); g_count++; }
                if (y < height - 1) { g_sum += get_bayer_pixel(bayer, x, y + 1); g_count++; }
                g = g_sum / g_count;
                
                // Interpolate red (4 diagonal neighbors)
                double r_sum = 0.0;
                int r_count = 0;
                if (x > 0 && y > 0) { 
                    r_sum += get_bayer_pixel(bayer, x - 1, y - 1); 
                    r_count++; 
                }
                if (x < width - 1 && y > 0) { 
                    r_sum += get_bayer_pixel(bayer, x + 1, y - 1); 
                    r_count++; 
                }
                if (x > 0 && y < height - 1) { 
                    r_sum += get_bayer_pixel(bayer, x - 1, y + 1); 
                    r_count++; 
                }
                if (x < width - 1 && y < height - 1) { 
                    r_sum += get_bayer_pixel(bayer, x + 1, y + 1); 
                    r_count++; 
                }
                r = r_sum / r_count;
            }
            
            // Normalize to [0, 1] range
            double max_val = (1 << bayer->bit_depth) - 1;
            r /= max_val;
            g /= max_val;
            b /= max_val;
            
            set_rgb_pixel(rgb, x, y, r, g, b);
        }
        
        // Report progress
        if (config->progress_callback && (y % 10 == 0)) {
            float progress = (float)y / height;
            config->progress_callback(progress, config->progress_user_data);
        }
    }
    
    // Final progress update
    if (config->progress_callback) {
        config->progress_callback(1.0f, config->progress_user_data);
    }
    
    log_message(1, "Bilinear demosaicing complete");
    
    return DEMOSAIC_SUCCESS;
}

// ============================================================================
// End of Part 3
// ============================================================================
// ============================================================================
// Part 4: VNG (Variable Number of Gradients) Demosaicing
// ============================================================================

/**
 * @brief Calculate gradient in a direction
 */
static inline double calculate_gradient(
    const BayerImage *bayer,
    int x,
    int y,
    int dx,
    int dy
) {
    int x1 = x - dx;
    int y1 = y - dy;
    int x2 = x + dx;
    int y2 = y + dy;
    
    // Bounds checking
    if (x1 < 0 || x1 >= bayer->width || x2 < 0 || x2 >= bayer->width ||
        y1 < 0 || y1 >= bayer->height || y2 < 0 || y2 >= bayer->height) {
        return 1e10; // Large value for out of bounds
    }
    
    double v1 = get_bayer_pixel(bayer, x1, y1);
    double v2 = get_bayer_pixel(bayer, x2, y2);
    
    return fabs(v2 - v1);
}

/**
 * @brief VNG demosaicing algorithm
 */
static DemosaicError demosaic_vng(
    const BayerImage *bayer,
    RGBImage *rgb,
    const DemosaicConfig *config
) {
    int width = bayer->width;
    int height = bayer->height;
    
    log_message(1, "Running VNG demosaicing");
    
    // VNG uses 8 directional gradients
    const int directions[8][2] = {
        {-1, -1}, {0, -1}, {1, -1},  // Top-left, Top, Top-right
        {-1,  0},          {1,  0},  // Left, Right
        {-1,  1}, {0,  1}, {1,  1}   // Bottom-left, Bottom, Bottom-right
    };
    
    // Process each pixel
    for (int y = 2; y < height - 2; y++) {
        for (int x = 2; x < width - 2; x++) {
            double r, g, b;
            uint16_t center = get_bayer_pixel(bayer, x, y);
            int color = get_bayer_color(bayer, x, y);
            
            // Calculate gradients in all 8 directions
            double gradients[8];
            double min_gradient = 1e10;
            
            for (int d = 0; d < 8; d++) {
                gradients[d] = calculate_gradient(bayer, x, y, 
                                                 directions[d][0], 
                                                 directions[d][1]);
                if (gradients[d] < min_gradient) {
                    min_gradient = gradients[d];
                }
            }
            
            // Threshold for selecting directions
            double threshold = min_gradient * (1.0 + config->edge_threshold / 100.0);
            
            if (color == 0) { // Red pixel
                r = center;
                
                // Interpolate green using selected directions
                double g_sum = 0.0;
                int g_count = 0;
                
                for (int d = 0; d < 8; d++) {
                    if (gradients[d] <= threshold) {
                        int nx = x + directions[d][0];
                        int ny = y + directions[d][1];
                        if (get_bayer_color(bayer, nx, ny) == 1) {
                            g_sum += get_bayer_pixel(bayer, nx, ny);
                            g_count++;
                        }
                    }
                }
                
                g = (g_count > 0) ? g_sum / g_count : center;
                
                // Interpolate blue using selected directions
                double b_sum = 0.0;
                int b_count = 0;
                
                for (int d = 0; d < 8; d++) {
                    if (gradients[d] <= threshold) {
                        int nx = x + directions[d][0];
                        int ny = y + directions[d][1];
                        if (get_bayer_color(bayer, nx, ny) == 2) {
                            b_sum += get_bayer_pixel(bayer, nx, ny);
                            b_count++;
                        }
                    }
                }
                
                b = (b_count > 0) ? b_sum / b_count : center;
                
            } else if (color == 1) { // Green pixel
                g = center;
                
                // Interpolate red using selected directions
                double r_sum = 0.0;
                int r_count = 0;
                
                for (int d = 0; d < 8; d++) {
                    if (gradients[d] <= threshold) {
                        int nx = x + directions[d][0];
                        int ny = y + directions[d][1];
                        if (get_bayer_color(bayer, nx, ny) == 0) {
                            r_sum += get_bayer_pixel(bayer, nx, ny);
                            r_count++;
                        }
                    }
                }
                
                r = (r_count > 0) ? r_sum / r_count : center;
                
                // Interpolate blue using selected directions
                double b_sum = 0.0;
                int b_count = 0;
                
                for (int d = 0; d < 8; d++) {
                    if (gradients[d] <= threshold) {
                        int nx = x + directions[d][0];
                        int ny = y + directions[d][1];
                        if (get_bayer_color(bayer, nx, ny) == 2) {
                            b_sum += get_bayer_pixel(bayer, nx, ny);
                            b_count++;
                        }
                    }
                }
                
                b = (b_count > 0) ? b_sum / b_count : center;
                
            } else { // Blue pixel
                b = center;
                
                // Interpolate green using selected directions
                double g_sum = 0.0;
                int g_count = 0;
                
                for (int d = 0; d < 8; d++) {
                    if (gradients[d] <= threshold) {
                        int nx = x + directions[d][0];
                        int ny = y + directions[d][1];
                        if (get_bayer_color(bayer, nx, ny) == 1) {
                            g_sum += get_bayer_pixel(bayer, nx, ny);
                            g_count++;
                        }
                    }
                }
                
                g = (g_count > 0) ? g_sum / g_count : center;
                
                // Interpolate red using selected directions
                double r_sum = 0.0;
                int r_count = 0;
                
                for (int d = 0; d < 8; d++) {
                    if (gradients[d] <= threshold) {
                        int nx = x + directions[d][0];
                        int ny = y + directions[d][1];
                        if (get_bayer_color(bayer, nx, ny) == 0) {
                            r_sum += get_bayer_pixel(bayer, nx, ny);
                            r_count++;
                        }
                    }
                }
                
                r = (r_count > 0) ? r_sum / r_count : center;
            }
            
            // Normalize to [0, 1] range
            double max_val = (1 << bayer->bit_depth) - 1;
            r /= max_val;
            g /= max_val;
            b /= max_val;
            
            set_rgb_pixel(rgb, x, y, r, g, b);
        }
        
        // Report progress
        if (config->progress_callback && (y % 10 == 0)) {
            float progress = (float)y / height;
            config->progress_callback(progress, config->progress_user_data);
        }
    }
    
    // Handle border pixels with bilinear interpolation
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            if (x < 2 || x >= width - 2 || y < 2 || y >= height - 2) {
                // Use bilinear for border
                double r, g, b;
                uint16_t center = get_bayer_pixel(bayer, x, y);
                int color = get_bayer_color(bayer, x, y);
                
                if (color == 0) {
                    r = center;
                    g = (get_bayer_pixel_safe(bayer, x-1, y) + 
                         get_bayer_pixel_safe(bayer, x+1, y) +
                         get_bayer_pixel_safe(bayer, x, y-1) + 
                         get_bayer_pixel_safe(bayer, x, y+1)) / 4.0;
                    b = (get_bayer_pixel_safe(bayer, x-1, y-1) + 
                         get_bayer_pixel_safe(bayer, x+1, y-1) +
                         get_bayer_pixel_safe(bayer, x-1, y+1) + 
                         get_bayer_pixel_safe(bayer, x+1, y+1)) / 4.0;
                } else if (color == 1) {
                    g = center;
                    r = (get_bayer_pixel_safe(bayer, x-1, y) + 
                         get_bayer_pixel_safe(bayer, x+1, y)) / 2.0;
                    b = (get_bayer_pixel_safe(bayer, x, y-1) + 
                         get_bayer_pixel_safe(bayer, x, y+1)) / 2.0;
                } else {
                    b = center;
                    g = (get_bayer_pixel_safe(bayer, x-1, y) + 
                         get_bayer_pixel_safe(bayer, x+1, y) +
                         get_bayer_pixel_safe(bayer, x, y-1) + 
                         get_bayer_pixel_safe(bayer, x, y+1)) / 4.0;
                    r = (get_bayer_pixel_safe(bayer, x-1, y-1) + 
                         get_bayer_pixel_safe(bayer, x+1, y-1) +
                         get_bayer_pixel_safe(bayer, x-1, y+1) + 
                         get_bayer_pixel_safe(bayer, x+1, y+1)) / 4.0;
                }
                
                double max_val = (1 << bayer->bit_depth) - 1;
                r /= max_val;
                g /= max_val;
                b /= max_val;
                
                set_rgb_pixel(rgb, x, y, r, g, b);
            }
        }
    }
    
    // Final progress update
    if (config->progress_callback) {
        config->progress_callback(1.0f, config->progress_user_data);
    }
    
    log_message(1, "VNG demosaicing complete");
    
    return DEMOSAIC_SUCCESS;
}

// ============================================================================
// Part 4: AHD (Adaptive Homogeneity-Directed) Demosaicing
// ============================================================================

/**
 * @brief Calculate homogeneity in horizontal direction
 */
static double calculate_horizontal_homogeneity(
    const BayerImage *bayer,
    int x,
    int y,
    int window_size
) {
    double sum = 0.0;
    int count = 0;
    
    for (int dy = -window_size; dy <= window_size; dy++) {
        for (int dx = -window_size; dx < window_size; dx++) {
            int x1 = x + dx;
            int y1 = y + dy;
            int x2 = x + dx + 1;
            int y2 = y + dy;
            
            if (x1 >= 0 && x1 < bayer->width && x2 >= 0 && x2 < bayer->width &&
                y1 >= 0 && y1 < bayer->height && y2 >= 0 && y2 < bayer->height) {
                double v1 = get_bayer_pixel(bayer, x1, y1);
                double v2 = get_bayer_pixel(bayer, x2, y2);
                sum += fabs(v2 - v1);
                count++;
            }
        }
    }
    
    return (count > 0) ? sum / count : 0.0;
}

/**
 * @brief Calculate homogeneity in vertical direction
 */
static double calculate_vertical_homogeneity(
    const BayerImage *bayer,
    int x,
    int y,
    int window_size
) {
    double sum = 0.0;
    int count = 0;
    
    for (int dy = -window_size; dy < window_size; dy++) {
        for (int dx = -window_size; dx <= window_size; dx++) {
            int x1 = x + dx;
            int y1 = y + dy;
            int x2 = x + dx;
            int y2 = y + dy + 1;
            
            if (x1 >= 0 && x1 < bayer->width && x2 >= 0 && x2 < bayer->width &&
                y1 >= 0 && y1 < bayer->height && y2 >= 0 && y2 < bayer->height) {
                double v1 = get_bayer_pixel(bayer, x1, y1);
                double v2 = get_bayer_pixel(bayer, x2, y2);
                sum += fabs(v2 - v1);
                count++;
            }
        }
    }
    
    return (count > 0) ? sum / count : 0.0;
}

/**
 * @brief Interpolate using horizontal direction
 */
static void interpolate_horizontal(
    const BayerImage *bayer,
    int x,
    int y,
    double *r,
    double *g,
    double *b
) {
    uint16_t center = get_bayer_pixel(bayer, x, y);
    int color = get_bayer_color(bayer, x, y);
    
    if (color == 0) { // Red
        *r = center;
        *g = (get_bayer_pixel_safe(bayer, x-1, y) + 
              get_bayer_pixel_safe(bayer, x+1, y)) / 2.0;
        *b = (get_bayer_pixel_safe(bayer, x-1, y-1) + 
              get_bayer_pixel_safe(bayer, x+1, y-1) +
              get_bayer_pixel_safe(bayer, x-1, y+1) + 
              get_bayer_pixel_safe(bayer, x+1, y+1)) / 4.0;
    } else if (color == 1) { // Green
        *g = center;
        *r = (get_bayer_pixel_safe(bayer, x-1, y) + 
              get_bayer_pixel_safe(bayer, x+1, y)) / 2.0;
        *b = (get_bayer_pixel_safe(bayer, x-1, y) + 
              get_bayer_pixel_safe(bayer, x+1, y)) / 2.0;
    } else { // Blue
        *b = center;
        *g = (get_bayer_pixel_safe(bayer, x-1, y) + 
              get_bayer_pixel_safe(bayer, x+1, y)) / 2.0;
        *r = (get_bayer_pixel_safe(bayer, x-1, y-1) + 
              get_bayer_pixel_safe(bayer, x+1, y-1) +
              get_bayer_pixel_safe(bayer, x-1, y+1) + 
              get_bayer_pixel_safe(bayer, x+1, y+1)) / 4.0;
    }
}

/**
 * @brief Interpolate using vertical direction
 */
static void interpolate_vertical(
    const BayerImage *bayer,
    int x,
    int y,
    double *r,
    double *g,
    double *b
) {
    uint16_t center = get_bayer_pixel(bayer, x, y);
    int color = get_bayer_color(bayer, x, y);
    
    if (color == 0) { // Red
        *r = center;
        *g = (get_bayer_pixel_safe(bayer, x, y-1) + 
              get_bayer_pixel_safe(bayer, x, y+1)) / 2.0;
        *b = (get_bayer_pixel_safe(bayer, x-1, y-1) + 
              get_bayer_pixel_safe(bayer, x+1, y-1) +
              get_bayer_pixel_safe(bayer, x-1, y+1) + 
              get_bayer_pixel_safe(bayer, x+1, y+1)) / 4.0;
    } else if (color == 1) { // Green
        *g = center;
        *r = (get_bayer_pixel_safe(bayer, x, y-1) + 
              get_bayer_pixel_safe(bayer, x, y+1)) / 2.0;
        *b = (get_bayer_pixel_safe(bayer, x, y-1) + 
              get_bayer_pixel_safe(bayer, x, y+1)) / 2.0;
    } else { // Blue
        *b = center;
        *g = (get_bayer_pixel_safe(bayer, x, y-1) + 
              get_bayer_pixel_safe(bayer, x, y+1)) / 2.0;
        *r = (get_bayer_pixel_safe(bayer, x-1, y-1) + 
              get_bayer_pixel_safe(bayer, x+1, y-1) +
              get_bayer_pixel_safe(bayer, x-1, y+1) + 
              get_bayer_pixel_safe(bayer, x+1, y+1)) / 4.0;
    }
}

/**
 * @brief AHD demosaicing algorithm
 */
static DemosaicError demosaic_ahd(
    const BayerImage *bayer,
    RGBImage *rgb,
    const DemosaicConfig *config
) {
    int width = bayer->width;
    int height = bayer->height;
    int window_size = 2; // 5x5 window
    
    log_message(1, "Running AHD demosaicing");
    
    // Allocate temporary buffers for horizontal and vertical interpolations
    RGBImage *rgb_h = demosaic_rgb_create(width, height, 3);
    RGBImage *rgb_v = demosaic_rgb_create(width, height, 3);
    
    if (rgb_h == NULL || rgb_v == NULL) {
        if (rgb_h) demosaic_rgb_destroy(rgb_h);
        if (rgb_v) demosaic_rgb_destroy(rgb_v);
        return DEMOSAIC_ERROR_OUT_OF_MEMORY;
    }
    
    // Step 1: Interpolate in both directions
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            double r_h, g_h, b_h;
            double r_v, g_v, b_v;
            
            interpolate_horizontal(bayer, x, y, &r_h, &g_h, &b_h);
            interpolate_vertical(bayer, x, y, &r_v, &g_v, &b_v);
            
            double max_val = (1 << bayer->bit_depth) - 1;
            
            set_rgb_pixel(rgb_h, x, y, r_h / max_val, g_h / max_val, b_h / max_val);
            set_rgb_pixel(rgb_v, x, y, r_v / max_val, g_v / max_val, b_v / max_val);
        }
    }
    
    // Step 2: Choose direction based on homogeneity
    for (int y = window_size; y < height - window_size; y++) {
        for (int x = window_size; x < width - window_size; x++) {
            double h_homo = calculate_horizontal_homogeneity(bayer, x, y, window_size);
            double v_homo = calculate_vertical_homogeneity(bayer, x, y, window_size);
            
            double r, g, b;
            
            if (h_homo < v_homo) {
                // Use horizontal interpolation
                get_rgb_pixel(rgb_h, x, y, &r, &g, &b);
            } else if (v_homo < h_homo) {
                // Use vertical interpolation
                get_rgb_pixel(rgb_v, x, y, &r, &g, &b);
            } else {
                // Average both
                double r_h, g_h, b_h;
                double r_v, g_v, b_v;
                get_rgb_pixel(rgb_h, x, y, &r_h, &g_h, &b_h);
                get_rgb_pixel(rgb_v, x, y, &r_v, &g_v, &b_v);
                r = (r_h + r_v) / 2.0;
                g = (g_h + g_v) / 2.0;
                b = (b_h + b_v) / 2.0;
            }
            
            set_rgb_pixel(rgb, x, y, r, g, b);
        }
        
        // Report progress
        if (config->progress_callback && (y % 10 == 0)) {
            float progress = (float)y / height;
            config->progress_callback(progress, config->progress_user_data);
        }
    }
    
    // Handle border pixels
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            if (x < window_size || x >= width - window_size || 
                y < window_size || y >= height - window_size) {
                // Use bilinear for border
                double r, g, b;
                get_rgb_pixel(rgb_h, x, y, &r, &g, &b);
                set_rgb_pixel(rgb, x, y, r, g, b);
            }
        }
    }
    
    // Cleanup
    demosaic_rgb_destroy(rgb_h);
    demosaic_rgb_destroy(rgb_v);
    
    // Final progress update
    if (config->progress_callback) {
        config->progress_callback(1.0f, config->progress_user_data);
    }
    
    log_message(1, "AHD demosaicing complete");
    
    return DEMOSAIC_SUCCESS;
}

// ============================================================================
// End of Part 4
// ============================================================================
// ============================================================================
// Part 5: LMMSE (Linear Minimum Mean Square Error) Demosaicing
// ============================================================================

/**
 * @brief Calculate local mean in a window
 */
static double calculate_local_mean(
    const BayerImage *bayer,
    int x,
    int y,
    int window_size,
    int color_filter
) {
    double sum = 0.0;
    int count = 0;
    
    for (int dy = -window_size; dy <= window_size; dy++) {
        for (int dx = -window_size; dx <= window_size; dx++) {
            int nx = x + dx;
            int ny = y + dy;
            
            if (nx >= 0 && nx < bayer->width && ny >= 0 && ny < bayer->height) {
                if (color_filter < 0 || get_bayer_color(bayer, nx, ny) == color_filter) {
                    sum += get_bayer_pixel(bayer, nx, ny);
                    count++;
                }
            }
        }
    }
    
    return (count > 0) ? sum / count : 0.0;
}

/**
 * @brief Calculate local variance in a window
 */
static double calculate_local_variance(
    const BayerImage *bayer,
    int x,
    int y,
    int window_size,
    int color_filter,
    double mean
) {
    double sum = 0.0;
    int count = 0;
    
    for (int dy = -window_size; dy <= window_size; dy++) {
        for (int dx = -window_size; dx <= window_size; dx++) {
            int nx = x + dx;
            int ny = y + dy;
            
            if (nx >= 0 && nx < bayer->width && ny >= 0 && ny < bayer->height) {
                if (color_filter < 0 || get_bayer_color(bayer, nx, ny) == color_filter) {
                    double val = get_bayer_pixel(bayer, nx, ny);
                    double diff = val - mean;
                    sum += diff * diff;
                    count++;
                }
            }
        }
    }
    
    return (count > 1) ? sum / (count - 1) : 0.0;
}

/**
 * @brief Calculate covariance between two color channels
 */
static double calculate_covariance(
    const BayerImage *bayer,
    int x,
    int y,
    int window_size,
    int color1,
    int color2,
    double mean1,
    double mean2
) {
    double sum = 0.0;
    int count = 0;
    
    for (int dy = -window_size; dy <= window_size; dy++) {
        for (int dx = -window_size; dx <= window_size; dx++) {
            int nx1 = x + dx;
            int ny1 = y + dy;
            
            // Find nearest pixel of color1
            if (nx1 >= 0 && nx1 < bayer->width && ny1 >= 0 && ny1 < bayer->height) {
                if (get_bayer_color(bayer, nx1, ny1) == color1) {
                    // Find nearest pixel of color2
                    for (int dy2 = -1; dy2 <= 1; dy2++) {
                        for (int dx2 = -1; dx2 <= 1; dx2++) {
                            int nx2 = nx1 + dx2;
                            int ny2 = ny1 + dy2;
                            
                            if (nx2 >= 0 && nx2 < bayer->width && 
                                ny2 >= 0 && ny2 < bayer->height) {
                                if (get_bayer_color(bayer, nx2, ny2) == color2) {
                                    double val1 = get_bayer_pixel(bayer, nx1, ny1);
                                    double val2 = get_bayer_pixel(bayer, nx2, ny2);
                                    sum += (val1 - mean1) * (val2 - mean2);
                                    count++;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
    return (count > 0) ? sum / count : 0.0;
}

/**
 * @brief LMMSE demosaicing algorithm
 */
static DemosaicError demosaic_lmmse(
    const BayerImage *bayer,
    RGBImage *rgb,
    const DemosaicConfig *config
) {
    int width = bayer->width;
    int height = bayer->height;
    int window_size = 3; // 7x7 window
    
    log_message(1, "Running LMMSE demosaicing");
    
    // First pass: interpolate green channel
    double *green_channel = (double*)demosaic_aligned_alloc(
        width * height * sizeof(double), ALIGNMENT);
    
    if (green_channel == NULL) {
        return DEMOSAIC_ERROR_OUT_OF_MEMORY;
    }
    
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int color = get_bayer_color(bayer, x, y);
            
            if (color == 1) {
                // Already green
                green_channel[y * width + x] = get_bayer_pixel(bayer, x, y);
            } else {
                // Interpolate green using LMMSE
                double mean_g = calculate_local_mean(bayer, x, y, window_size, 1);
                double mean_c = calculate_local_mean(bayer, x, y, window_size, color);
                double var_g = calculate_local_variance(bayer, x, y, window_size, 1, mean_g);
                double var_c = calculate_local_variance(bayer, x, y, window_size, color, mean_c);
                double cov = calculate_covariance(bayer, x, y, window_size, 1, color, 
                                                  mean_g, mean_c);
                
                double center = get_bayer_pixel(bayer, x, y);
                
                // LMMSE estimation
                double alpha = (var_c > 1e-6) ? cov / var_c : 0.0;
                double g_estimate = mean_g + alpha * (center - mean_c);
                
                // Clamp to valid range
                g_estimate = fmax(0.0, fmin(g_estimate, (1 << bayer->bit_depth) - 1));
                
                green_channel[y * width + x] = g_estimate;
            }
        }
        
        // Report progress (first pass)
        if (config->progress_callback && (y % 20 == 0)) {
            float progress = 0.5f * (float)y / height;
            config->progress_callback(progress, config->progress_user_data);
        }
    }
    
    // Second pass: interpolate red and blue channels
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            double r, g, b;
            int color = get_bayer_color(bayer, x, y);
            uint16_t center = get_bayer_pixel(bayer, x, y);
            
            g = green_channel[y * width + x];
            
            if (color == 0) {
                // Red pixel
                r = center;
                
                // Interpolate blue using LMMSE with green as reference
                double mean_b = calculate_local_mean(bayer, x, y, window_size, 2);
                double mean_g_local = calculate_local_mean(bayer, x, y, window_size, -1);
                double var_b = calculate_local_variance(bayer, x, y, window_size, 2, mean_b);
                double var_g_local = calculate_local_variance(bayer, x, y, window_size, -1, 
                                                             mean_g_local);
                double cov = calculate_covariance(bayer, x, y, window_size, 2, 1, 
                                                  mean_b, mean_g_local);
                
                double alpha = (var_g_local > 1e-6) ? cov / var_g_local : 0.0;
                b = mean_b + alpha * (g - mean_g_local);
                
            } else if (color == 1) {
                // Green pixel
                int left_color = (x > 0) ? get_bayer_color(bayer, x - 1, y) : -1;
                int top_color = (y > 0) ? get_bayer_color(bayer, x, y - 1) : -1;
                
                // Interpolate red
                double mean_r = calculate_local_mean(bayer, x, y, window_size, 0);
                double mean_g_local = calculate_local_mean(bayer, x, y, window_size, -1);
                double var_r = calculate_local_variance(bayer, x, y, window_size, 0, mean_r);
                double var_g_local = calculate_local_variance(bayer, x, y, window_size, -1, 
                                                             mean_g_local);
                double cov_r = calculate_covariance(bayer, x, y, window_size, 0, 1, 
                                                    mean_r, mean_g_local);
                
                double alpha_r = (var_g_local > 1e-6) ? cov_r / var_g_local : 0.0;
                r = mean_r + alpha_r * (g - mean_g_local);
                
                // Interpolate blue
                double mean_b = calculate_local_mean(bayer, x, y, window_size, 2);
                double cov_b = calculate_covariance(bayer, x, y, window_size, 2, 1, 
                                                    mean_b, mean_g_local);
                
                double alpha_b = (var_g_local > 1e-6) ? cov_b / var_g_local : 0.0;
                b = mean_b + alpha_b * (g - mean_g_local);
                
            } else {
                // Blue pixel
                b = center;
                
                // Interpolate red using LMMSE with green as reference
                double mean_r = calculate_local_mean(bayer, x, y, window_size, 0);
                double mean_g_local = calculate_local_mean(bayer, x, y, window_size, -1);
                double var_r = calculate_local_variance(bayer, x, y, window_size, 0, mean_r);
                double var_g_local = calculate_local_variance(bayer, x, y, window_size, -1, 
                                                             mean_g_local);
                double cov = calculate_covariance(bayer, x, y, window_size, 0, 1, 
                                                  mean_r, mean_g_local);
                
                double alpha = (var_g_local > 1e-6) ? cov / var_g_local : 0.0;
                r = mean_r + alpha * (g - mean_g_local);
            }
            
            // Clamp and normalize
            double max_val = (1 << bayer->bit_depth) - 1;
            r = fmax(0.0, fmin(r, max_val)) / max_val;
            g = fmax(0.0, fmin(g, max_val)) / max_val;
            b = fmax(0.0, fmin(b, max_val)) / max_val;
            
            set_rgb_pixel(rgb, x, y, r, g, b);
        }
        
        // Report progress (second pass)
        if (config->progress_callback && (y % 20 == 0)) {
            float progress = 0.5f + 0.5f * (float)y / height;
            config->progress_callback(progress, config->progress_user_data);
        }
    }
    
    demosaic_aligned_free(green_channel);
    
    // Final progress update
    if (config->progress_callback) {
        config->progress_callback(1.0f, config->progress_user_data);
    }
    
    log_message(1, "LMMSE demosaicing complete");
    
    return DEMOSAIC_SUCCESS;
}

// ============================================================================
// Part 5: PPG (Patterned Pixel Grouping) Demosaicing
// ============================================================================

/**
 * @brief Calculate directional difference for PPG
 */
static double calculate_ppg_difference(
    const BayerImage *bayer,
    int x,
    int y,
    int dx,
    int dy
) {
    int x1 = x - dx;
    int y1 = y - dy;
    int x2 = x + dx;
    int y2 = y + dy;
    
    if (x1 < 0 || x1 >= bayer->width || x2 < 0 || x2 >= bayer->width ||
        y1 < 0 || y1 >= bayer->height || y2 < 0 || y2 >= bayer->height) {
        return 1e10;
    }
    
    double v1 = get_bayer_pixel(bayer, x1, y1);
    double v2 = get_bayer_pixel(bayer, x2, y2);
    double vc = get_bayer_pixel(bayer, x, y);
    
    return fabs(v1 - vc) + fabs(v2 - vc);
}

/**
 * @brief PPG demosaicing algorithm
 */
static DemosaicError demosaic_ppg(
    const BayerImage *bayer,
    RGBImage *rgb,
    const DemosaicConfig *config
) {
    int width = bayer->width;
    int height = bayer->height;
    
    log_message(1, "Running PPG demosaicing");
    
    // First pass: interpolate green channel
    double *green_channel = (double*)demosaic_aligned_alloc(
        width * height * sizeof(double), ALIGNMENT);
    
    if (green_channel == NULL) {
        return DEMOSAIC_ERROR_OUT_OF_MEMORY;
    }
    
    for (int y = 2; y < height - 2; y++) {
        for (int x = 2; x < width - 2; x++) {
            int color = get_bayer_color(bayer, x, y);
            
            if (color == 1) {
                // Already green
                green_channel[y * width + x] = get_bayer_pixel(bayer, x, y);
            } else {
                // Calculate directional differences
                double diff_h = calculate_ppg_difference(bayer, x, y, 1, 0);
                double diff_v = calculate_ppg_difference(bayer, x, y, 0, 1);
                
                uint16_t center = get_bayer_pixel(bayer, x, y);
                double g_estimate;
                
                if (diff_h < diff_v) {
                    // Horizontal interpolation
                    double g_left = get_bayer_pixel(bayer, x - 1, y);
                    double g_right = get_bayer_pixel(bayer, x + 1, y);
                    double c_left = get_bayer_pixel(bayer, x - 2, y);
                    double c_right = get_bayer_pixel(bayer, x + 2, y);
                    
                    g_estimate = (g_left + g_right) / 2.0 + 
                                (2.0 * center - c_left - c_right) / 4.0;
                    
                } else if (diff_v < diff_h) {
                    // Vertical interpolation
                    double g_top = get_bayer_pixel(bayer, x, y - 1);
                    double g_bottom = get_bayer_pixel(bayer, x, y + 1);
                    double c_top = get_bayer_pixel(bayer, x, y - 2);
                    double c_bottom = get_bayer_pixel(bayer, x, y + 2);
                    
                    g_estimate = (g_top + g_bottom) / 2.0 + 
                                (2.0 * center - c_top - c_bottom) / 4.0;
                    
                } else {
                    // Average both directions
                    double g_h = (get_bayer_pixel(bayer, x - 1, y) + 
                                 get_bayer_pixel(bayer, x + 1, y)) / 2.0;
                    double g_v = (get_bayer_pixel(bayer, x, y - 1) + 
                                 get_bayer_pixel(bayer, x, y + 1)) / 2.0;
                    g_estimate = (g_h + g_v) / 2.0;
                }
                
                // Clamp to valid range
                g_estimate = fmax(0.0, fmin(g_estimate, (1 << bayer->bit_depth) - 1));
                
                green_channel[y * width + x] = g_estimate;
            }
        }
        
        // Report progress (first pass)
        if (config->progress_callback && (y % 20 == 0)) {
            float progress = 0.5f * (float)y / height;
            config->progress_callback(progress, config->progress_user_data);
        }
    }
    
    // Handle borders for green channel
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            if (x < 2 || x >= width - 2 || y < 2 || y >= height - 2) {
                int color = get_bayer_color(bayer, x, y);
                if (color == 1) {
                    green_channel[y * width + x] = get_bayer_pixel(bayer, x, y);
                } else {
                    green_channel[y * width + x] = 
                        (get_bayer_pixel_safe(bayer, x-1, y) + 
                         get_bayer_pixel_safe(bayer, x+1, y) +
                         get_bayer_pixel_safe(bayer, x, y-1) + 
                         get_bayer_pixel_safe(bayer, x, y+1)) / 4.0;
                }
            }
        }
    }
    
    // Second pass: interpolate red and blue channels
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            double r, g, b;
            int color = get_bayer_color(bayer, x, y);
            uint16_t center = get_bayer_pixel(bayer, x, y);
            
            g = green_channel[y * width + x];
            
            if (color == 0) {
                // Red pixel
                r = center;
                
                // Interpolate blue using color difference
                double b_sum = 0.0;
                int b_count = 0;
                
                for (int dy = -1; dy <= 1; dy += 2) {
                    for (int dx = -1; dx <= 1; dx += 2) {
                        int nx = x + dx;
                        int ny = y + dy;
                        if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                            if (get_bayer_color(bayer, nx, ny) == 2) {
                                double b_val = get_bayer_pixel(bayer, nx, ny);
                                double g_val = green_channel[ny * width + nx];
                                b_sum += b_val - g_val;
                                b_count++;
                            }
                        }
                    }
                }
                
                b = g + ((b_count > 0) ? b_sum / b_count : 0.0);
                
            } else if (color == 1) {
                // Green pixel
                // Interpolate red
                double r_sum = 0.0;
                int r_count = 0;
                
                for (int dy = -1; dy <= 1; dy++) {
                    for (int dx = -1; dx <= 1; dx++) {
                        if (dx == 0 && dy == 0) continue;
                        int nx = x + dx;
                        int ny = y + dy;
                        if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                            if (get_bayer_color(bayer, nx, ny) == 0) {
                                double r_val = get_bayer_pixel(bayer, nx, ny);
                                double g_val = green_channel[ny * width + nx];
                                r_sum += r_val - g_val;
                                r_count++;
                            }
                        }
                    }
                }
                
                r = g + ((r_count > 0) ? r_sum / r_count : 0.0);
                
                // Interpolate blue
                double b_sum = 0.0;
                int b_count = 0;
                
                for (int dy = -1; dy <= 1; dy++) {
                    for (int dx = -1; dx <= 1; dx++) {
                        if (dx == 0 && dy == 0) continue;
                        int nx = x + dx;
                        int ny = y + dy;
                        if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                            if (get_bayer_color(bayer, nx, ny) == 2) {
                                double b_val = get_bayer_pixel(bayer, nx, ny);
                                double g_val = green_channel[ny * width + nx];
                                b_sum += b_val - g_val;
                                b_count++;
                            }
                        }
                    }
                }
                
                b = g + ((b_count > 0) ? b_sum / b_count : 0.0);
                
            } else {
                // Blue pixel
                b = center;
                
                // Interpolate red using color difference
                double r_sum = 0.0;
                int r_count = 0;
                
                for (int dy = -1; dy <= 1; dy += 2) {
                    for (int dx = -1; dx <= 1; dx += 2) {
                        int nx = x + dx;
                        int ny = y + dy;
                        if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                            if (get_bayer_color(bayer, nx, ny) == 0) {
                                double r_val = get_bayer_pixel(bayer, nx, ny);
                                double g_val = green_channel[ny * width + nx];
                                r_sum += r_val - g_val;
                                r_count++;
                            }
                        }
                    }
                }
                
                r = g + ((r_count > 0) ? r_sum / r_count : 0.0);
            }
            
            // Clamp and normalize
            double max_val = (1 << bayer->bit_depth) - 1;
            r = fmax(0.0, fmin(r, max_val)) / max_val;
            g = fmax(0.0, fmin(g, max_val)) / max_val;
            b = fmax(0.0, fmin(b, max_val)) / max_val;
            
            set_rgb_pixel(rgb, x, y, r, g, b);
        }
        
        // Report progress (second pass)
        if (config->progress_callback && (y % 20 == 0)) {
            float progress = 0.5f + 0.5f * (float)y / height;
            config->progress_callback(progress, config->progress_user_data);
        }
    }
    
    demosaic_aligned_free(green_channel);
    
    // Final progress update
    if (config->progress_callback) {
        config->progress_callback(1.0f, config->progress_user_data);
    }
    
    log_message(1, "PPG demosaicing complete");
    
    return DEMOSAIC_SUCCESS;
}

// ============================================================================
// Part 5: Post-Processing Functions
// ============================================================================

/**
 * @brief Apply Gaussian blur for denoising
 */
static void apply_gaussian_blur(
    RGBImage *image,
    double sigma
) {
    if (sigma <= 0.0) return;
    
    int width = image->width;
    int height = image->height;
    int channels = image->channels;
    
    // Calculate kernel size
    int kernel_size = (int)(6.0 * sigma + 1.0);
    if (kernel_size % 2 == 0) kernel_size++;
    int radius = kernel_size / 2;
    
    // Generate Gaussian kernel
    double *kernel = (double*)malloc(kernel_size * sizeof(double));
    double sum = 0.0;
    
    for (int i = 0; i < kernel_size; i++) {
        int x = i - radius;
        kernel[i] = exp(-(x * x) / (2.0 * sigma * sigma));
        sum += kernel[i];
    }
    
    // Normalize kernel
    for (int i = 0; i < kernel_size; i++) {
        kernel[i] /= sum;
    }
    
    // Allocate temporary buffer
    double *temp = (double*)demosaic_aligned_alloc(
        width * height * channels * sizeof(double), ALIGNMENT);
    
    if (temp == NULL) {
        free(kernel);
        return;
    }
    
    // Horizontal pass
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            for (int c = 0; c < channels; c++) {
                double sum_val = 0.0;
                double sum_weight = 0.0;
                
                for (int k = 0; k < kernel_size; k++) {
                    int nx = x + k - radius;
                    if (nx >= 0 && nx < width) {
                        int idx = (y * width + nx) * channels + c;
                        sum_val += image->data[idx] * kernel[k];
                        sum_weight += kernel[k];
                    }
                }
                
                int idx = (y * width + x) * channels + c;
                temp[idx] = sum_val / sum_weight;
            }
        }
    }
    
    // Vertical pass
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            for (int c = 0; c < channels; c++) {
                double sum_val = 0.0;
                double sum_weight = 0.0;
                
                for (int k = 0; k < kernel_size; k++) {
                    int ny = y + k - radius;
                    if (ny >= 0 && ny < height) {
                        int idx = (ny * width + x) * channels + c;
                        sum_val += temp[idx] * kernel[k];
                        sum_weight += kernel[k];
                    }
                }
                
                int idx = (y * width + x) * channels + c;
                image->data[idx] = sum_val / sum_weight;
            }
        }
    }
    
    demosaic_aligned_free(temp);
    free(kernel);
}

/**
 * @brief Apply unsharp mask for sharpening
 */
static void apply_unsharp_mask(
    RGBImage *image,
    double amount,
    double radius
) {
    if (amount <= 0.0) return;
    
    int width = image->width;
    int height = image->height;
    int channels = image->channels;
    
    // Create blurred copy
    RGBImage *blurred = demosaic_rgb_clone(image);
    if (blurred == NULL) return;
    
    apply_gaussian_blur(blurred, radius);
    
    // Apply unsharp mask: output = original + amount * (original - blurred)
    for (int i = 0; i < width * height * channels; i++) {
        double original = image->data[i];
        double blur = blurred->data[i];
        double sharpened = original + amount * (original - blur);
        
        // Clamp to [0, 1]
        image->data[i] = fmax(0.0, fmin(1.0, sharpened));
    }
    
    demosaic_rgb_destroy(blurred);
}

/**
 * @brief Apply post-processing to demosaiced image
 */
static DemosaicError apply_post_processing(
    RGBImage *image,
    const DemosaicConfig *config
) {
    log_message(1, "Applying post-processing");
    
    // Denoise
    if (config->denoise_strength > 0.0) {
        double sigma = config->denoise_strength * 2.0;
        apply_gaussian_blur(image, sigma);
        log_message(1, "Applied denoising with sigma=%.2f", sigma);
    }
    
    // Sharpen
    if (config->sharpen_amount > 0.0) {
        apply_unsharp_mask(image, config->sharpen_amount, 1.0);
        log_message(1, "Applied sharpening with amount=%.2f", config->sharpen_amount);
    }
    
    return DEMOSAIC_SUCCESS;
}

// ============================================================================
// End of Part 5
// ============================================================================
// ============================================================================
// Part 6: Color Correction and White Balance
// ============================================================================

/**
 * @brief Apply white balance correction
 */
static void apply_white_balance(
    RGBImage *image,
    double wb_r,
    double wb_g,
    double wb_b
) {
    int width = image->width;
    int height = image->height;
    
    log_message(1, "Applying white balance: R=%.3f, G=%.3f, B=%.3f", wb_r, wb_g, wb_b);
    
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            double r, g, b;
            get_rgb_pixel(image, x, y, &r, &g, &b);
            
            r *= wb_r;
            g *= wb_g;
            b *= wb_b;
            
            // Clamp to [0, 1]
            r = fmax(0.0, fmin(1.0, r));
            g = fmax(0.0, fmin(1.0, g));
            b = fmax(0.0, fmin(1.0, b));
            
            set_rgb_pixel(image, x, y, r, g, b);
        }
    }
}

/**
 * @brief Apply exposure compensation
 */
static void apply_exposure_compensation(
    RGBImage *image,
    double exposure
) {
    if (fabs(exposure) < 1e-6) return;
    
    double factor = pow(2.0, exposure);
    int total_pixels = image->width * image->height * image->channels;
    
    log_message(1, "Applying exposure compensation: %.2f EV (factor=%.3f)", 
                exposure, factor);
    
    for (int i = 0; i < total_pixels; i++) {
        image->data[i] = fmax(0.0, fmin(1.0, image->data[i] * factor));
    }
}

/**
 * @brief Apply saturation adjustment
 */
static void apply_saturation(
    RGBImage *image,
    double saturation
) {
    if (fabs(saturation - 1.0) < 1e-6) return;
    
    int width = image->width;
    int height = image->height;
    
    log_message(1, "Applying saturation adjustment: %.2f", saturation);
    
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            double r, g, b;
            get_rgb_pixel(image, x, y, &r, &g, &b);
            
            // Calculate luminance (Rec. 709)
            double luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;
            
            // Adjust saturation
            r = luma + saturation * (r - luma);
            g = luma + saturation * (g - luma);
            b = luma + saturation * (b - luma);
            
            // Clamp to [0, 1]
            r = fmax(0.0, fmin(1.0, r));
            g = fmax(0.0, fmin(1.0, g));
            b = fmax(0.0, fmin(1.0, b));
            
            set_rgb_pixel(image, x, y, r, g, b);
        }
    }
}

/**
 * @brief Apply contrast adjustment
 */
static void apply_contrast(
    RGBImage *image,
    double contrast
) {
    if (fabs(contrast - 1.0) < 1e-6) return;
    
    int total_pixels = image->width * image->height * image->channels;
    
    log_message(1, "Applying contrast adjustment: %.2f", contrast);
    
    for (int i = 0; i < total_pixels; i++) {
        double val = image->data[i];
        val = 0.5 + contrast * (val - 0.5);
        image->data[i] = fmax(0.0, fmin(1.0, val));
    }
}

/**
 * @brief Apply brightness adjustment
 */
static void apply_brightness(
    RGBImage *image,
    double brightness
) {
    if (fabs(brightness) < 1e-6) return;
    
    int total_pixels = image->width * image->height * image->channels;
    
    log_message(1, "Applying brightness adjustment: %.2f", brightness);
    
    for (int i = 0; i < total_pixels; i++) {
        image->data[i] = fmax(0.0, fmin(1.0, image->data[i] + brightness));
    }
}

/**
 * @brief Apply all color corrections
 */
static DemosaicError apply_color_correction(
    RGBImage *image,
    const DemosaicConfig *config
) {
    if (!config->color_correction) {
        return DEMOSAIC_SUCCESS;
    }
    
    log_message(1, "Applying color corrections");
    
    // White balance
    apply_white_balance(image, 
                       config->white_balance_r,
                       config->white_balance_g,
                       config->white_balance_b);
    
    // Exposure compensation
    apply_exposure_compensation(image, config->exposure_compensation);
    
    // Saturation
    apply_saturation(image, config->saturation);
    
    // Contrast
    apply_contrast(image, config->contrast);
    
    // Brightness
    apply_brightness(image, config->brightness);
    
    return DEMOSAIC_SUCCESS;
}

// ============================================================================
// Part 6: Color Space Conversion
// ============================================================================

/**
 * @brief Apply gamma correction
 */
static inline double apply_gamma(double linear, double gamma) {
    if (linear <= 0.0) return 0.0;
    return pow(linear, 1.0 / gamma);
}

/**
 * @brief Remove gamma correction (linearize)
 */
static inline double remove_gamma(double encoded, double gamma) {
    if (encoded <= 0.0) return 0.0;
    return pow(encoded, gamma);
}

/**
 * @brief sRGB gamma encoding
 */
static inline double srgb_encode(double linear) {
    if (linear <= 0.0031308) {
        return 12.92 * linear;
    } else {
        return 1.055 * pow(linear, 1.0 / 2.4) - 0.055;
    }
}

/**
 * @brief sRGB gamma decoding
 */
static inline double srgb_decode(double encoded) {
    if (encoded <= 0.04045) {
        return encoded / 12.92;
    } else {
        return pow((encoded + 0.055) / 1.055, 2.4);
    }
}

/**
 * @brief Convert RGB to XYZ color space
 */
static void rgb_to_xyz(double r, double g, double b, 
                       double *x, double *y, double *z) {
    // sRGB to XYZ (D65 illuminant)
    *x = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b;
    *y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b;
    *z = 0.0193339 * r + 0.1191920 * g + 0.9503041 * b;
}

/**
 * @brief Convert XYZ to RGB color space
 */
static void xyz_to_rgb(double x, double y, double z,
                       double *r, double *g, double *b) {
    // XYZ to sRGB (D65 illuminant)
    *r =  3.2404542 * x - 1.5371385 * y - 0.4985314 * z;
    *g = -0.9692660 * x + 1.8760108 * y + 0.0415560 * z;
    *b =  0.0556434 * x - 0.2040259 * y + 1.0572252 * z;
}

/**
 * @brief Convert RGB to LAB color space
 */
static void rgb_to_lab(double r, double g, double b,
                       double *l, double *a, double *b_out) {
    // First convert to XYZ
    double x, y, z;
    rgb_to_xyz(r, g, b, &x, &y, &z);
    
    // D65 white point
    const double xn = 0.95047;
    const double yn = 1.00000;
    const double zn = 1.08883;
    
    // Normalize
    x /= xn;
    y /= yn;
    z /= zn;
    
    // Apply f(t) function
    auto f = [](double t) -> double {
        const double delta = 6.0 / 29.0;
        if (t > delta * delta * delta) {
            return pow(t, 1.0 / 3.0);
        } else {
            return t / (3.0 * delta * delta) + 4.0 / 29.0;
        }
    };
    
    double fx = f(x);
    double fy = f(y);
    double fz = f(z);
    
    // Calculate LAB
    *l = 116.0 * fy - 16.0;
    *a = 500.0 * (fx - fy);
    *b_out = 200.0 * (fy - fz);
}

/**
 * @brief Convert LAB to RGB color space
 */
static void lab_to_rgb(double l, double a, double b_in,
                       double *r, double *g, double *b) {
    // D65 white point
    const double xn = 0.95047;
    const double yn = 1.00000;
    const double zn = 1.08883;
    
    // Calculate intermediate values
    double fy = (l + 16.0) / 116.0;
    double fx = a / 500.0 + fy;
    double fz = fy - b_in / 200.0;
    
    // Apply inverse f(t) function
    auto f_inv = [](double t) -> double {
        const double delta = 6.0 / 29.0;
        if (t > delta) {
            return t * t * t;
        } else {
            return 3.0 * delta * delta * (t - 4.0 / 29.0);
        }
    };
    
    double x = xn * f_inv(fx);
    double y = yn * f_inv(fy);
    double z = zn * f_inv(fz);
    
    // Convert XYZ to RGB
    xyz_to_rgb(x, y, z, r, g, b);
}

/**
 * @brief Apply color space conversion
 */
static DemosaicError apply_color_space_conversion(
    RGBImage *image,
    const DemosaicConfig *config
) {
    int width = image->width;
    int height = image->height;
    ColorSpace target_space = config->output_color_space;
    
    if (target_space == COLOR_SPACE_LINEAR) {
        // Already in linear space, nothing to do
        return DEMOSAIC_SUCCESS;
    }
    
    log_message(1, "Converting to color space: %d", target_space);
    
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            double r, g, b;
            get_rgb_pixel(image, x, y, &r, &g, &b);
            
            switch (target_space) {
                case COLOR_SPACE_SRGB:
                    // Apply sRGB gamma
                    r = srgb_encode(r);
                    g = srgb_encode(g);
                    b = srgb_encode(b);
                    break;
                    
                case COLOR_SPACE_ADOBE_RGB:
                    // Apply Adobe RGB gamma (2.2)
                    r = apply_gamma(r, 2.2);
                    g = apply_gamma(g, 2.2);
                    b = apply_gamma(b, 2.2);
                    break;
                    
                case COLOR_SPACE_PROPHOTO_RGB:
                    // Apply ProPhoto RGB gamma (1.8)
                    r = apply_gamma(r, 1.8);
                    g = apply_gamma(g, 1.8);
                    b = apply_gamma(b, 1.8);
                    break;
                    
                case COLOR_SPACE_REC709:
                    // Apply Rec. 709 gamma (similar to sRGB)
                    r = srgb_encode(r);
                    g = srgb_encode(g);
                    b = srgb_encode(b);
                    break;
                    
                case COLOR_SPACE_REC2020:
                    // Apply Rec. 2020 gamma
                    r = apply_gamma(r, 2.4);
                    g = apply_gamma(g, 2.4);
                    b = apply_gamma(b, 2.4);
                    break;
                    
                default:
                    // Use custom gamma from config
                    r = apply_gamma(r, config->gamma);
                    g = apply_gamma(g, config->gamma);
                    b = apply_gamma(b, config->gamma);
                    break;
            }
            
            // Clamp to [0, 1]
            r = fmax(0.0, fmin(1.0, r));
            g = fmax(0.0, fmin(1.0, g));
            b = fmax(0.0, fmin(1.0, b));
            
            set_rgb_pixel(image, x, y, r, g, b);
        }
    }
    
    return DEMOSAIC_SUCCESS;
}

// ============================================================================
// Part 6: Main Demosaicing Function
// ============================================================================

/**
 * @brief Select and execute demosaicing algorithm
 */
static DemosaicError execute_demosaic_algorithm(
    const BayerImage *bayer,
    RGBImage *rgb,
    const DemosaicConfig *config
) {
    DemosaicError result = DEMOSAIC_SUCCESS;
    
    log_message(1, "Executing demosaic algorithm: %d", config->algorithm);
    
    switch (config->algorithm) {
        case DEMOSAIC_NEAREST:
            result = demosaic_nearest_neighbor(bayer, rgb, config);
            break;
            
        case DEMOSAIC_BILINEAR:
            result = demosaic_bilinear(bayer, rgb, config);
            break;
            
        case DEMOSAIC_VNG:
            result = demosaic_vng(bayer, rgb, config);
            break;
            
        case DEMOSAIC_AHD:
            result = demosaic_ahd(bayer, rgb, config);
            break;
            
        case DEMOSAIC_LMMSE:
            result = demosaic_lmmse(bayer, rgb, config);
            break;
            
        case DEMOSAIC_PPG:
            result = demosaic_ppg(bayer, rgb, config);
            break;
            
        case DEMOSAIC_CUSTOM:
            set_error(DEMOSAIC_ERROR_NOT_IMPLEMENTED, 
                     "Custom algorithm not implemented");
            result = DEMOSAIC_ERROR_NOT_IMPLEMENTED;
            break;
            
        default:
            set_error(DEMOSAIC_ERROR_INVALID_ARGUMENT, 
                     "Unknown algorithm");
            result = DEMOSAIC_ERROR_INVALID_ARGUMENT;
            break;
    }
    
    return result;
}

/**
 * @brief Main demosaicing function
 */
DemosaicError demosaic_process(
    const BayerImage *bayer,
    RGBImage *rgb,
    const DemosaicConfig *config
) {
    // Validate inputs
    if (bayer == NULL || rgb == NULL || config == NULL) {
        set_error(DEMOSAIC_ERROR_INVALID_ARGUMENT, "NULL argument");
        return DEMOSAIC_ERROR_INVALID_ARGUMENT;
    }
    
    // Validate configuration
    DemosaicError error = demosaic_config_validate(config);
    if (error != DEMOSAIC_SUCCESS) {
        return error;
    }
    
    // Check dimensions match
    if (bayer->width != rgb->width || bayer->height != rgb->height) {
        set_error(DEMOSAIC_ERROR_INVALID_ARGUMENT, "Dimension mismatch");
        return DEMOSAIC_ERROR_INVALID_ARGUMENT;
    }
    
    log_message(0, "Starting demosaic process: %dx%d, algorithm=%d",
                bayer->width, bayer->height, config->algorithm);
    
    // Execute demosaicing algorithm
    error = execute_demosaic_algorithm(bayer, rgb, config);
    if (error != DEMOSAIC_SUCCESS) {
        log_message(0, "Demosaic algorithm failed: %d", error);
        return error;
    }
    
    // Apply post-processing
    error = apply_post_processing(rgb, config);
    if (error != DEMOSAIC_SUCCESS) {
        log_message(0, "Post-processing failed: %d", error);
        return error;
    }
    
    // Apply color correction
    error = apply_color_correction(rgb, config);
    if (error != DEMOSAIC_SUCCESS) {
        log_message(0, "Color correction failed: %d", error);
        return error;
    }
    
    // Apply color space conversion
    error = apply_color_space_conversion(rgb, config);
    if (error != DEMOSAIC_SUCCESS) {
        log_message(0, "Color space conversion failed: %d", error);
        return error;
    }
    
    log_message(0, "Demosaic process completed successfully");
    
    return DEMOSAIC_SUCCESS;
}

/**
 * @brief Simplified demosaicing function with default config
 */
DemosaicError demosaic_simple(
    const BayerImage *bayer,
    RGBImage *rgb,
    DemosaicAlgorithm algorithm
) {
    // Create default configuration
    DemosaicConfig *config = demosaic_config_create();
    if (config == NULL) {
        return DEMOSAIC_ERROR_OUT_OF_MEMORY;
    }
    
    // Set algorithm
    config->algorithm = algorithm;
    
    // Process
    DemosaicError error = demosaic_process(bayer, rgb, config);
    
    // Cleanup
    demosaic_config_destroy(config);
    
    return error;
}

/**
 * @brief Auto-select best algorithm based on image characteristics
 */
DemosaicAlgorithm demosaic_auto_select_algorithm(
    const BayerImage *bayer,
    QualityPreset quality
) {
    if (bayer == NULL) {
        return DEMOSAIC_BILINEAR;
    }
    
    int total_pixels = bayer->width * bayer->height;
    
    // For small images, use faster algorithms
    if (total_pixels < 1024 * 768) {
        switch (quality) {
            case QUALITY_FAST:
                return DEMOSAIC_BILINEAR;
            case QUALITY_BALANCED:
                return DEMOSAIC_VNG;
            case QUALITY_HIGH:
            case QUALITY_BEST:
                return DEMOSAIC_AHD;
            default:
                return DEMOSAIC_BILINEAR;
        }
    }
    
    // For medium images
    if (total_pixels < 4096 * 3072) {
        switch (quality) {
            case QUALITY_FAST:
                return DEMOSAIC_BILINEAR;
            case QUALITY_BALANCED:
                return DEMOSAIC_VNG;
            case QUALITY_HIGH:
                return DEMOSAIC_AHD;
            case QUALITY_BEST:
                return DEMOSAIC_LMMSE;
            default:
                return DEMOSAIC_VNG;
        }
    }
    
    // For large images
    switch (quality) {
        case QUALITY_FAST:
            return DEMOSAIC_BILINEAR;
        case QUALITY_BALANCED:
            return DEMOSAIC_PPG;
        case QUALITY_HIGH:
            return DEMOSAIC_VNG;
        case QUALITY_BEST:
            return DEMOSAIC_AHD;
        default:
            return DEMOSAIC_PPG;
    }
}

/**
 * @brief Process with auto-selected algorithm
 */
DemosaicError demosaic_auto(
    const BayerImage *bayer,
    RGBImage *rgb,
    QualityPreset quality
) {
    // Create configuration
    DemosaicConfig *config = demosaic_config_create();
    if (config == NULL) {
        return DEMOSAIC_ERROR_OUT_OF_MEMORY;
    }
    
    // Set quality preset
    demosaic_config_set_preset(config, quality);
    
    // Auto-select algorithm
    config->algorithm = demosaic_auto_select_algorithm(bayer, quality);
    
    log_message(1, "Auto-selected algorithm: %d for quality: %d",
                config->algorithm, quality);
    
    // Process
    DemosaicError error = demosaic_process(bayer, rgb, config);
    
    // Cleanup
    demosaic_config_destroy(config);
    
    return error;
}

// ============================================================================
// Part 6: Batch Processing Support
// ============================================================================

/**
 * @brief Process multiple images in batch
 */
DemosaicError demosaic_batch_process(
    const BayerImage **bayer_images,
    RGBImage **rgb_images,
    int count,
    const DemosaicConfig *config
) {
    if (bayer_images == NULL || rgb_images == NULL || count <= 0) {
        set_error(DEMOSAIC_ERROR_INVALID_ARGUMENT, "Invalid batch arguments");
        return DEMOSAIC_ERROR_INVALID_ARGUMENT;
    }
    
    log_message(0, "Starting batch processing: %d images", count);
    
    DemosaicError last_error = DEMOSAIC_SUCCESS;
    int success_count = 0;
    
    for (int i = 0; i < count; i++) {
        if (bayer_images[i] == NULL || rgb_images[i] == NULL) {
            log_message(0, "Skipping NULL image at index %d", i);
            continue;
        }
        
        DemosaicError error = demosaic_process(bayer_images[i], 
                                              rgb_images[i], 
                                              config);
        
        if (error == DEMOSAIC_SUCCESS) {
            success_count++;
        } else {
            last_error = error;
            log_message(0, "Failed to process image %d: error %d", i, error);
        }
        
        // Report batch progress
        if (config->progress_callback) {
            float progress = (float)(i + 1) / count;
            config->progress_callback(progress, config->progress_user_data);
        }
    }
    
    log_message(0, "Batch processing complete: %d/%d successful", 
                success_count, count);
    
    return (success_count == count) ? DEMOSAIC_SUCCESS : last_error;
}

// ============================================================================
// End of Part 6
// ============================================================================
// ============================================================================
// Part 7a: SIMD Optimizations - SSE2 and AVX2
// ============================================================================

#ifdef USE_SIMD

#if defined(__SSE2__) || defined(_M_X64) || defined(_M_AMD64)
#include <emmintrin.h>  // SSE2
#define HAVE_SSE2 1
#endif

#if defined(__AVX2__)
#include <immintrin.h>  // AVX2
#define HAVE_AVX2 1
#endif

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define HAVE_NEON 1
#endif

/**
 * @brief Check SIMD support at runtime
 */
static int check_simd_support(void) {
    int support = 0;
    
#ifdef HAVE_SSE2
    support |= SIMD_SSE2;
    log_message(2, "SSE2 support detected");
#endif

#ifdef HAVE_AVX2
    support |= SIMD_AVX2;
    log_message(2, "AVX2 support detected");
#endif

#ifdef HAVE_NEON
    support |= SIMD_NEON;
    log_message(2, "NEON support detected");
#endif

    return support;
}

// ============================================================================
// SSE2 Optimized Functions
// ============================================================================

#ifdef HAVE_SSE2

/**
 * @brief Bilinear interpolation using SSE2 (process 2 pixels at once)
 */
static void demosaic_bilinear_sse2_row(
    const BayerImage *bayer,
    RGBImage *rgb,
    int y,
    int width
) {
    double max_val = (1 << bayer->bit_depth) - 1;
    __m128d max_val_vec = _mm_set1_pd(max_val);
    __m128d zero_vec = _mm_setzero_pd();
    
    for (int x = 0; x < width - 1; x += 2) {
        // Process 2 pixels at a time with SSE2
        int color1 = get_bayer_color(bayer, x, y);
        int color2 = get_bayer_color(bayer, x + 1, y);
        
        uint16_t center1 = get_bayer_pixel(bayer, x, y);
        uint16_t center2 = get_bayer_pixel(bayer, x + 1, y);
        
        double r1, g1, b1, r2, g2, b2;
        
        // Pixel 1
        if (color1 == 0) { // Red
            r1 = center1;
            g1 = (get_bayer_pixel_safe(bayer, x-1, y) + 
                  get_bayer_pixel_safe(bayer, x+1, y) +
                  get_bayer_pixel_safe(bayer, x, y-1) + 
                  get_bayer_pixel_safe(bayer, x, y+1)) / 4.0;
            b1 = (get_bayer_pixel_safe(bayer, x-1, y-1) + 
                  get_bayer_pixel_safe(bayer, x+1, y-1) +
                  get_bayer_pixel_safe(bayer, x-1, y+1) + 
                  get_bayer_pixel_safe(bayer, x+1, y+1)) / 4.0;
        } else if (color1 == 1) { // Green
            g1 = center1;
            r1 = (get_bayer_pixel_safe(bayer, x-1, y) + 
                  get_bayer_pixel_safe(bayer, x+1, y)) / 2.0;
            b1 = (get_bayer_pixel_safe(bayer, x, y-1) + 
                  get_bayer_pixel_safe(bayer, x, y+1)) / 2.0;
        } else { // Blue
            b1 = center1;
            g1 = (get_bayer_pixel_safe(bayer, x-1, y) + 
                  get_bayer_pixel_safe(bayer, x+1, y) +
                  get_bayer_pixel_safe(bayer, x, y-1) + 
                  get_bayer_pixel_safe(bayer, x, y+1)) / 4.0;
            r1 = (get_bayer_pixel_safe(bayer, x-1, y-1) + 
                  get_bayer_pixel_safe(bayer, x+1, y-1) +
                  get_bayer_pixel_safe(bayer, x-1, y+1) + 
                  get_bayer_pixel_safe(bayer, x+1, y+1)) / 4.0;
        }
        
        // Pixel 2
        if (color2 == 0) { // Red
            r2 = center2;
            g2 = (get_bayer_pixel_safe(bayer, x, y) + 
                  get_bayer_pixel_safe(bayer, x+2, y) +
                  get_bayer_pixel_safe(bayer, x+1, y-1) + 
                  get_bayer_pixel_safe(bayer, x+1, y+1)) / 4.0;
            b2 = (get_bayer_pixel_safe(bayer, x, y-1) + 
                  get_bayer_pixel_safe(bayer, x+2, y-1) +
                  get_bayer_pixel_safe(bayer, x, y+1) + 
                  get_bayer_pixel_safe(bayer, x+2, y+1)) / 4.0;
        } else if (color2 == 1) { // Green
            g2 = center2;
            r2 = (get_bayer_pixel_safe(bayer, x, y) + 
                  get_bayer_pixel_safe(bayer, x+2, y)) / 2.0;
            b2 = (get_bayer_pixel_safe(bayer, x+1, y-1) + 
                  get_bayer_pixel_safe(bayer, x+1, y+1)) / 2.0;
        } else { // Blue
            b2 = center2;
            g2 = (get_bayer_pixel_safe(bayer, x, y) + 
                  get_bayer_pixel_safe(bayer, x+2, y) +
                  get_bayer_pixel_safe(bayer, x+1, y-1) + 
                  get_bayer_pixel_safe(bayer, x+1, y+1)) / 4.0;
            r2 = (get_bayer_pixel_safe(bayer, x, y-1) + 
                  get_bayer_pixel_safe(bayer, x+2, y-1) +
                  get_bayer_pixel_safe(bayer, x, y+1) + 
                  get_bayer_pixel_safe(bayer, x+2, y+1)) / 4.0;
        }
        
        // Normalize using SSE2
        __m128d r_vec = _mm_set_pd(r2, r1);
        __m128d g_vec = _mm_set_pd(g2, g1);
        __m128d b_vec = _mm_set_pd(b2, b1);
        
        r_vec = _mm_div_pd(r_vec, max_val_vec);
        g_vec = _mm_div_pd(g_vec, max_val_vec);
        b_vec = _mm_div_pd(b_vec, max_val_vec);
        
        // Clamp to [0, 1]
        __m128d one_vec = _mm_set1_pd(1.0);
        r_vec = _mm_max_pd(zero_vec, _mm_min_pd(one_vec, r_vec));
        g_vec = _mm_max_pd(zero_vec, _mm_min_pd(one_vec, g_vec));
        b_vec = _mm_max_pd(zero_vec, _mm_min_pd(one_vec, b_vec));
        
        // Store results
        double r_out[2], g_out[2], b_out[2];
        _mm_storeu_pd(r_out, r_vec);
        _mm_storeu_pd(g_out, g_vec);
        _mm_storeu_pd(b_out, b_vec);
        
        set_rgb_pixel(rgb, x, y, r_out[0], g_out[0], b_out[0]);
        set_rgb_pixel(rgb, x + 1, y, r_out[1], g_out[1], b_out[1]);
    }
    
    // Handle remaining pixel if width is odd
    if (width % 2 == 1) {
        int x = width - 1;
        int color = get_bayer_color(bayer, x, y);
        uint16_t center = get_bayer_pixel(bayer, x, y);
        double r, g, b;
        
        if (color == 0) {
            r = center;
            g = (get_bayer_pixel_safe(bayer, x-1, y) + 
                 get_bayer_pixel_safe(bayer, x+1, y) +
                 get_bayer_pixel_safe(bayer, x, y-1) + 
                 get_bayer_pixel_safe(bayer, x, y+1)) / 4.0;
            b = (get_bayer_pixel_safe(bayer, x-1, y-1) + 
                 get_bayer_pixel_safe(bayer, x+1, y-1) +
                 get_bayer_pixel_safe(bayer, x-1, y+1) + 
                 get_bayer_pixel_safe(bayer, x+1, y+1)) / 4.0;
        } else if (color == 1) {
            g = center;
            r = (get_bayer_pixel_safe(bayer, x-1, y) + 
                 get_bayer_pixel_safe(bayer, x+1, y)) / 2.0;
            b = (get_bayer_pixel_safe(bayer, x, y-1) + 
                 get_bayer_pixel_safe(bayer, x, y+1)) / 2.0;
        } else {
            b = center;
            g = (get_bayer_pixel_safe(bayer, x-1, y) + 
                 get_bayer_pixel_safe(bayer, x+1, y) +
                 get_bayer_pixel_safe(bayer, x, y-1) + 
                 get_bayer_pixel_safe(bayer, x, y+1)) / 4.0;
            r = (get_bayer_pixel_safe(bayer, x-1, y-1) + 
                 get_bayer_pixel_safe(bayer, x+1, y-1) +
                 get_bayer_pixel_safe(bayer, x-1, y+1) + 
                 get_bayer_pixel_safe(bayer, x+1, y+1)) / 4.0;
        }
        
        r = fmax(0.0, fmin(1.0, r / max_val));
        g = fmax(0.0, fmin(1.0, g / max_val));
        b = fmax(0.0, fmin(1.0, b / max_val));
        
        set_rgb_pixel(rgb, x, y, r, g, b);
    }
}

/**
 * @brief Apply white balance using SSE2
 */
static void apply_white_balance_sse2(
    RGBImage *image,
    double wb_r,
    double wb_g,
    double wb_b
) {
    int width = image->width;
    int height = image->height;
    
    __m128d wb_r_vec = _mm_set1_pd(wb_r);
    __m128d wb_g_vec = _mm_set1_pd(wb_g);
    __m128d wb_b_vec = _mm_set1_pd(wb_b);
    __m128d zero_vec = _mm_setzero_pd();
    __m128d one_vec = _mm_set1_pd(1.0);
    
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width - 1; x += 2) {
            // Load 2 pixels
            double r1, g1, b1, r2, g2, b2;
            get_rgb_pixel(image, x, y, &r1, &g1, &b1);
            get_rgb_pixel(image, x + 1, y, &r2, &g2, &b2);
            
            __m128d r_vec = _mm_set_pd(r2, r1);
            __m128d g_vec = _mm_set_pd(g2, g1);
            __m128d b_vec = _mm_set_pd(b2, b1);
            
            // Apply white balance
            r_vec = _mm_mul_pd(r_vec, wb_r_vec);
            g_vec = _mm_mul_pd(g_vec, wb_g_vec);
            b_vec = _mm_mul_pd(b_vec, wb_b_vec);
            
            // Clamp to [0, 1]
            r_vec = _mm_max_pd(zero_vec, _mm_min_pd(one_vec, r_vec));
            g_vec = _mm_max_pd(zero_vec, _mm_min_pd(one_vec, g_vec));
            b_vec = _mm_max_pd(zero_vec, _mm_min_pd(one_vec, b_vec));
            
            // Store results
            double r_out[2], g_out[2], b_out[2];
            _mm_storeu_pd(r_out, r_vec);
            _mm_storeu_pd(g_out, g_vec);
            _mm_storeu_pd(b_out, b_vec);
            
            set_rgb_pixel(image, x, y, r_out[0], g_out[0], b_out[0]);
            set_rgb_pixel(image, x + 1, y, r_out[1], g_out[1], b_out[1]);
        }
        
        // Handle remaining pixel if width is odd
        if (width % 2 == 1) {
            int x = width - 1;
            double r, g, b;
            get_rgb_pixel(image, x, y, &r, &g, &b);
            
            r = fmax(0.0, fmin(1.0, r * wb_r));
            g = fmax(0.0, fmin(1.0, g * wb_g));
            b = fmax(0.0, fmin(1.0, b * wb_b));
            
            set_rgb_pixel(image, x, y, r, g, b);
        }
    }
}

#endif // HAVE_SSE2

// ============================================================================
// AVX2 Optimized Functions
// ============================================================================

#ifdef HAVE_AVX2

/**
 * @brief Apply white balance using AVX2 (process 4 pixels at once)
 */
static void apply_white_balance_avx2(
    RGBImage *image,
    double wb_r,
    double wb_g,
    double wb_b
) {
    int width = image->width;
    int height = image->height;
    
    __m256d wb_r_vec = _mm256_set1_pd(wb_r);
    __m256d wb_g_vec = _mm256_set1_pd(wb_g);
    __m256d wb_b_vec = _mm256_set1_pd(wb_b);
    __m256d zero_vec = _mm256_setzero_pd();
    __m256d one_vec = _mm256_set1_pd(1.0);
    
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width - 3; x += 4) {
            // Load 4 pixels
            double r[4], g[4], b[4];
            for (int i = 0; i < 4; i++) {
                get_rgb_pixel(image, x + i, y, &r[i], &g[i], &b[i]);
            }
            
            __m256d r_vec = _mm256_loadu_pd(r);
            __m256d g_vec = _mm256_loadu_pd(g);
            __m256d b_vec = _mm256_loadu_pd(b);
            
            // Apply white balance
            r_vec = _mm256_mul_pd(r_vec, wb_r_vec);
            g_vec = _mm256_mul_pd(g_vec, wb_g_vec);
            b_vec = _mm256_mul_pd(b_vec, wb_b_vec);
            
            // Clamp to [0, 1]
            r_vec = _mm256_max_pd(zero_vec, _mm256_min_pd(one_vec, r_vec));
            g_vec = _mm256_max_pd(zero_vec, _mm256_min_pd(one_vec, g_vec));
            b_vec = _mm256_max_pd(zero_vec, _mm256_min_pd(one_vec, b_vec));
            
            // Store results
            _mm256_storeu_pd(r, r_vec);
            _mm256_storeu_pd(g, g_vec);
            _mm256_storeu_pd(b, b_vec);
            
            for (int i = 0; i < 4; i++) {
                set_rgb_pixel(image, x + i, y, r[i], g[i], b[i]);
            }
        }
        
        // Handle remaining pixels
        for (int x = (width / 4) * 4; x < width; x++) {
            double r, g, b;
            get_rgb_pixel(image, x, y, &r, &g, &b);
            
            r = fmax(0.0, fmin(1.0, r * wb_r));
            g = fmax(0.0, fmin(1.0, g * wb_g));
            b = fmax(0.0, fmin(1.0, b * wb_b));
            
            set_rgb_pixel(image, x, y, r, g, b);
        }
    }
}

#endif // HAVE_AVX2

#endif // USE_SIMD

// ============================================================================
// End of Part 7a
// ============================================================================
// ============================================================================
// Part 7b: ARM NEON Optimized Functions
// ============================================================================

#ifdef HAVE_NEON

/**
 * @brief Apply white balance using NEON (process 4 pixels at once)
 */
static void apply_white_balance_neon(
    RGBImage *image,
    double wb_r,
    double wb_g,
    double wb_b
) {
    int width = image->width;
    int height = image->height;
    
    // NEON works with float32, so we need to convert
    float32_t wb_r_f = (float32_t)wb_r;
    float32_t wb_g_f = (float32_t)wb_g;
    float32_t wb_b_f = (float32_t)wb_b;
    
    float32x4_t wb_r_vec = vdupq_n_f32(wb_r_f);
    float32x4_t wb_g_vec = vdupq_n_f32(wb_g_f);
    float32x4_t wb_b_vec = vdupq_n_f32(wb_b_f);
    float32x4_t zero_vec = vdupq_n_f32(0.0f);
    float32x4_t one_vec = vdupq_n_f32(1.0f);
    
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width - 3; x += 4) {
            // Load 4 pixels (convert double to float)
            float r[4], g[4], b[4];
            for (int i = 0; i < 4; i++) {
                double rd, gd, bd;
                get_rgb_pixel(image, x + i, y, &rd, &gd, &bd);
                r[i] = (float)rd;
                g[i] = (float)gd;
                b[i] = (float)bd;
            }
            
            float32x4_t r_vec = vld1q_f32(r);
            float32x4_t g_vec = vld1q_f32(g);
            float32x4_t b_vec = vld1q_f32(b);
            
            // Apply white balance
            r_vec = vmulq_f32(r_vec, wb_r_vec);
            g_vec = vmulq_f32(g_vec, wb_g_vec);
            b_vec = vmulq_f32(b_vec, wb_b_vec);
            
            // Clamp to [0, 1]
            r_vec = vmaxq_f32(zero_vec, vminq_f32(one_vec, r_vec));
            g_vec = vmaxq_f32(zero_vec, vminq_f32(one_vec, g_vec));
            b_vec = vmaxq_f32(zero_vec, vminq_f32(one_vec, b_vec));
            
            // Store results
            vst1q_f32(r, r_vec);
            vst1q_f32(g, g_vec);
            vst1q_f32(b, b_vec);
            
            for (int i = 0; i < 4; i++) {
                set_rgb_pixel(image, x + i, y, (double)r[i], (double)g[i], (double)b[i]);
            }
        }
        
        // Handle remaining pixels
        for (int x = (width / 4) * 4; x < width; x++) {
            double r, g, b;
            get_rgb_pixel(image, x, y, &r, &g, &b);
            
            r = fmax(0.0, fmin(1.0, r * wb_r));
            g = fmax(0.0, fmin(1.0, g * wb_g));
            b = fmax(0.0, fmin(1.0, b * wb_b));
            
            set_rgb_pixel(image, x, y, r, g, b);
        }
    }
}

/**
 * @brief Bilinear interpolation using NEON
 */
static void demosaic_bilinear_neon_row(
    const BayerImage *bayer,
    RGBImage *rgb,
    int y,
    int width
) {
    float max_val = (float)((1 << bayer->bit_depth) - 1);
    float32x4_t max_val_vec = vdupq_n_f32(max_val);
    float32x4_t zero_vec = vdupq_n_f32(0.0f);
    float32x4_t one_vec = vdupq_n_f32(1.0f);
    
    for (int x = 0; x < width - 3; x += 4) {
        float r[4], g[4], b[4];
        
        // Process 4 pixels
        for (int i = 0; i < 4; i++) {
            int px = x + i;
            int color = get_bayer_color(bayer, px, y);
            uint16_t center = get_bayer_pixel(bayer, px, y);
            
            if (color == 0) { // Red
                r[i] = (float)center;
                g[i] = (get_bayer_pixel_safe(bayer, px-1, y) + 
                        get_bayer_pixel_safe(bayer, px+1, y) +
                        get_bayer_pixel_safe(bayer, px, y-1) + 
                        get_bayer_pixel_safe(bayer, px, y+1)) / 4.0f;
                b[i] = (get_bayer_pixel_safe(bayer, px-1, y-1) + 
                        get_bayer_pixel_safe(bayer, px+1, y-1) +
                        get_bayer_pixel_safe(bayer, px-1, y+1) + 
                        get_bayer_pixel_safe(bayer, px+1, y+1)) / 4.0f;
            } else if (color == 1) { // Green
                g[i] = (float)center;
                r[i] = (get_bayer_pixel_safe(bayer, px-1, y) + 
                        get_bayer_pixel_safe(bayer, px+1, y)) / 2.0f;
                b[i] = (get_bayer_pixel_safe(bayer, px, y-1) + 
                        get_bayer_pixel_safe(bayer, px, y+1)) / 2.0f;
            } else { // Blue
                b[i] = (float)center;
                g[i] = (get_bayer_pixel_safe(bayer, px-1, y) + 
                        get_bayer_pixel_safe(bayer, px+1, y) +
                        get_bayer_pixel_safe(bayer, px, y-1) + 
                        get_bayer_pixel_safe(bayer, px, y+1)) / 4.0f;
                r[i] = (get_bayer_pixel_safe(bayer, px-1, y-1) + 
                        get_bayer_pixel_safe(bayer, px+1, y-1) +
                        get_bayer_pixel_safe(bayer, px-1, y+1) + 
                        get_bayer_pixel_safe(bayer, px+1, y+1)) / 4.0f;
            }
        }
        
        // Normalize using NEON
        float32x4_t r_vec = vld1q_f32(r);
        float32x4_t g_vec = vld1q_f32(g);
        float32x4_t b_vec = vld1q_f32(b);
        
        r_vec = vdivq_f32(r_vec, max_val_vec);
        g_vec = vdivq_f32(g_vec, max_val_vec);
        b_vec = vdivq_f32(b_vec, max_val_vec);
        
        // Clamp to [0, 1]
        r_vec = vmaxq_f32(zero_vec, vminq_f32(one_vec, r_vec));
        g_vec = vmaxq_f32(zero_vec, vminq_f32(one_vec, g_vec));
        b_vec = vmaxq_f32(zero_vec, vminq_f32(one_vec, b_vec));
        
        // Store results
        vst1q_f32(r, r_vec);
        vst1q_f32(g, g_vec);
        vst1q_f32(b, b_vec);
        
        for (int i = 0; i < 4; i++) {
            set_rgb_pixel(rgb, x + i, y, (double)r[i], (double)g[i], (double)b[i]);
        }
    }
    
    // Handle remaining pixels
    for (int x = (width / 4) * 4; x < width; x++) {
        int color = get_bayer_color(bayer, x, y);
        uint16_t center = get_bayer_pixel(bayer, x, y);
        double r, g, b;
        
        if (color == 0) {
            r = center;
            g = (get_bayer_pixel_safe(bayer, x-1, y) + 
                 get_bayer_pixel_safe(bayer, x+1, y) +
                 get_bayer_pixel_safe(bayer, x, y-1) + 
                 get_bayer_pixel_safe(bayer, x, y+1)) / 4.0;
            b = (get_bayer_pixel_safe(bayer, x-1, y-1) + 
                 get_bayer_pixel_safe(bayer, x+1, y-1) +
                 get_bayer_pixel_safe(bayer, x-1, y+1) + 
                 get_bayer_pixel_safe(bayer, x+1, y+1)) / 4.0;
        } else if (color == 1) {
            g = center;
            r = (get_bayer_pixel_safe(bayer, x-1, y) + 
                 get_bayer_pixel_safe(bayer, x+1, y)) / 2.0;
            b = (get_bayer_pixel_safe(bayer, x, y-1) + 
                 get_bayer_pixel_safe(bayer, x, y+1)) / 2.0;
        } else {
            b = center;
            g = (get_bayer_pixel_safe(bayer, x-1, y) + 
                 get_bayer_pixel_safe(bayer, x+1, y) +
                 get_bayer_pixel_safe(bayer, x, y-1) + 
                 get_bayer_pixel_safe(bayer, x, y+1)) / 4.0;
            r = (get_bayer_pixel_safe(bayer, x-1, y-1) + 
                 get_bayer_pixel_safe(bayer, x+1, y-1) +
                 get_bayer_pixel_safe(bayer, x-1, y+1) + 
                 get_bayer_pixel_safe(bayer, x+1, y+1)) / 4.0;
        }
        
        r = fmax(0.0, fmin(1.0, r / max_val));
        g = fmax(0.0, fmin(1.0, g / max_val));
        b = fmax(0.0, fmin(1.0, b / max_val));
        
        set_rgb_pixel(rgb, x, y, r, g, b);
    }
}

#endif // HAVE_NEON

// ============================================================================
// Part 7b: Multi-threading Support
// ============================================================================

#ifdef USE_THREADS

#include <pthread.h>

/**
 * @brief Thread data structure for parallel processing
 */
typedef struct {
    const BayerImage *bayer;
    RGBImage *rgb;
    const DemosaicConfig *config;
    int start_row;
    int end_row;
    int thread_id;
    DemosaicError error;
} ThreadData;

/**
 * @brief Worker thread function for demosaicing
 */
static void* demosaic_worker_thread(void *arg) {
    ThreadData *data = (ThreadData*)arg;
    
    log_message(2, "Thread %d processing rows %d to %d",
                data->thread_id, data->start_row, data->end_row);
    
    // Process assigned rows based on algorithm
    switch (data->config->algorithm) {
        case DEMOSAIC_BILINEAR:
            for (int y = data->start_row; y < data->end_row; y++) {
#ifdef HAVE_SSE2
                if (data->config->use_simd) {
                    demosaic_bilinear_sse2_row(data->bayer, data->rgb, y, 
                                              data->bayer->width);
                } else
#elif defined(HAVE_NEON)
                if (data->config->use_simd) {
                    demosaic_bilinear_neon_row(data->bayer, data->rgb, y,
                                              data->bayer->width);
                } else
#endif
                {
                    // Fallback to scalar implementation
                    for (int x = 0; x < data->bayer->width; x++) {
                        double r, g, b;
                        uint16_t center = get_bayer_pixel(data->bayer, x, y);
                        int color = get_bayer_color(data->bayer, x, y);
                        double max_val = (1 << data->bayer->bit_depth) - 1;
                        
                        if (color == 0) { // Red
                            r = center;
                            g = (get_bayer_pixel_safe(data->bayer, x-1, y) + 
                                 get_bayer_pixel_safe(data->bayer, x+1, y) +
                                 get_bayer_pixel_safe(data->bayer, x, y-1) + 
                                 get_bayer_pixel_safe(data->bayer, x, y+1)) / 4.0;
                            b = (get_bayer_pixel_safe(data->bayer, x-1, y-1) + 
                                 get_bayer_pixel_safe(data->bayer, x+1, y-1) +
                                 get_bayer_pixel_safe(data->bayer, x-1, y+1) + 
                                 get_bayer_pixel_safe(data->bayer, x+1, y+1)) / 4.0;
                        } else if (color == 1) { // Green
                            g = center;
                            r = (get_bayer_pixel_safe(data->bayer, x-1, y) + 
                                 get_bayer_pixel_safe(data->bayer, x+1, y)) / 2.0;
                            b = (get_bayer_pixel_safe(data->bayer, x, y-1) + 
                                 get_bayer_pixel_safe(data->bayer, x, y+1)) / 2.0;
                        } else { // Blue
                            b = center;
                            g = (get_bayer_pixel_safe(data->bayer, x-1, y) + 
                                 get_bayer_pixel_safe(data->bayer, x+1, y) +
                                 get_bayer_pixel_safe(data->bayer, x, y-1) + 
                                 get_bayer_pixel_safe(data->bayer, x, y+1)) / 4.0;
                            r = (get_bayer_pixel_safe(data->bayer, x-1, y-1) + 
                                 get_bayer_pixel_safe(data->bayer, x+1, y-1) +
                                 get_bayer_pixel_safe(data->bayer, x-1, y+1) + 
                                 get_bayer_pixel_safe(data->bayer, x+1, y+1)) / 4.0;
                        }
                        
                        r /= max_val;
                        g /= max_val;
                        b /= max_val;
                        
                        set_rgb_pixel(data->rgb, x, y, r, g, b);
                    }
                }
            }
            break;
            
        case DEMOSAIC_VNG:
        case DEMOSAIC_AHD:
        case DEMOSAIC_LMMSE:
        case DEMOSAIC_PPG:
            // These algorithms need full image context, so we process them differently
            // For now, fall back to single-threaded
            data->error = DEMOSAIC_ERROR_NOT_IMPLEMENTED;
            log_message(1, "Thread %d: Algorithm %d not yet parallelized",
                       data->thread_id, data->config->algorithm);
            return NULL;
            
        default:
            data->error = DEMOSAIC_ERROR_NOT_IMPLEMENTED;
            return NULL;
    }
    
    data->error = DEMOSAIC_SUCCESS;
    return NULL;
}

/**
 * @brief Execute demosaicing with multiple threads
 */
static DemosaicError demosaic_parallel(
    const BayerImage *bayer,
    RGBImage *rgb,
    const DemosaicConfig *config
) {
    int num_threads = config->num_threads;
    if (num_threads <= 1) {
        // Fall back to single-threaded
        return execute_demosaic_algorithm(bayer, rgb, config);
    }
    
    // Limit threads to reasonable number
    int max_threads = 16;
    if (num_threads > max_threads) {
        num_threads = max_threads;
        log_message(1, "Limiting threads to %d", max_threads);
    }
    
    int height = bayer->height;
    int rows_per_thread = height / num_threads;
    
    // Need at least a few rows per thread
    if (rows_per_thread < 4) {
        num_threads = height / 4;
        if (num_threads < 1) num_threads = 1;
        rows_per_thread = height / num_threads;
    }
    
    log_message(1, "Starting parallel demosaicing with %d threads (%d rows each)",
                num_threads, rows_per_thread);
    
    // Allocate thread data and handles
    pthread_t *threads = (pthread_t*)malloc(num_threads * sizeof(pthread_t));
    ThreadData *thread_data = (ThreadData*)malloc(num_threads * sizeof(ThreadData));
    
    if (threads == NULL || thread_data == NULL) {
        free(threads);
        free(thread_data);
        set_error(DEMOSAIC_ERROR_OUT_OF_MEMORY, "Failed to allocate thread data");
        return DEMOSAIC_ERROR_OUT_OF_MEMORY;
    }
    
    // Create threads
    for (int i = 0; i < num_threads; i++) {
        thread_data[i].bayer = bayer;
        thread_data[i].rgb = rgb;
        thread_data[i].config = config;
        thread_data[i].start_row = i * rows_per_thread;
        thread_data[i].end_row = (i == num_threads - 1) ? height : (i + 1) * rows_per_thread;
        thread_data[i].thread_id = i;
        thread_data[i].error = DEMOSAIC_SUCCESS;
        
        int result = pthread_create(&threads[i], NULL, demosaic_worker_thread, 
                                    &thread_data[i]);
        if (result != 0) {
            log_message(0, "Failed to create thread %d: error %d", i, result);
            // Wait for already created threads
            for (int j = 0; j < i; j++) {
                pthread_join(threads[j], NULL);
            }
            free(threads);
            free(thread_data);
            set_error(DEMOSAIC_ERROR_THREAD_FAILED, "pthread_create failed");
            return DEMOSAIC_ERROR_THREAD_FAILED;
        }
    }
    
    // Wait for all threads to complete
    DemosaicError final_error = DEMOSAIC_SUCCESS;
    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
        if (thread_data[i].error != DEMOSAIC_SUCCESS) {
            final_error = thread_data[i].error;
            log_message(0, "Thread %d failed with error %d", i, final_error);
        }
    }
    
    free(threads);
    free(thread_data);
    
    log_message(1, "Parallel demosaicing complete");
    
    return final_error;
}

/**
 * @brief Thread pool structure for batch processing
 */
typedef struct {
    pthread_t *threads;
    ThreadData *thread_data;
    int num_threads;
    int active_threads;
    pthread_mutex_t mutex;
    pthread_cond_t cond;
    int shutdown;
} ThreadPool;

/**
 * @brief Create thread pool
 */
static ThreadPool* thread_pool_create(int num_threads) {
    ThreadPool *pool = (ThreadPool*)malloc(sizeof(ThreadPool));
    if (pool == NULL) {
        return NULL;
    }
    
    pool->num_threads = num_threads;
    pool->active_threads = 0;
    pool->shutdown = 0;
    
    pool->threads = (pthread_t*)malloc(num_threads * sizeof(pthread_t));
    pool->thread_data = (ThreadData*)malloc(num_threads * sizeof(ThreadData));
    
    if (pool->threads == NULL || pool->thread_data == NULL) {
        free(pool->threads);
        free(pool->thread_data);
        free(pool);
        return NULL;
    }
    
    pthread_mutex_init(&pool->mutex, NULL);
    pthread_cond_init(&pool->cond, NULL);
    
    return pool;
}

/**
 * @brief Destroy thread pool
 */
static void thread_pool_destroy(ThreadPool *pool) {
    if (pool == NULL) return;
    
    pthread_mutex_lock(&pool->mutex);
    pool->shutdown = 1;
    pthread_cond_broadcast(&pool->cond);
    pthread_mutex_unlock(&pool->mutex);
    
    // Wait for all threads
    for (int i = 0; i < pool->active_threads; i++) {
        pthread_join(pool->threads[i], NULL);
    }
    
    pthread_mutex_destroy(&pool->mutex);
    pthread_cond_destroy(&pool->cond);
    
    free(pool->threads);
    free(pool->thread_data);
    free(pool);
}

#endif // USE_THREADS

// ============================================================================
// End of Part 7b
// ============================================================================
// ============================================================================
// Part 7c: Performance Monitoring
// ============================================================================

/**
 * @brief Performance statistics structure
 */
typedef struct {
    double demosaic_time;
    double postprocess_time;
    double color_correction_time;
    double white_balance_time;
    double noise_reduction_time;
    double sharpening_time;
    double total_time;
    size_t memory_used;
    size_t peak_memory;
    int cache_hits;
    int cache_misses;
    int simd_operations;
    int scalar_operations;
} PerformanceStats;

static PerformanceStats g_perf_stats = {0};

/**
 * @brief Get current time in seconds
 */
static double get_time_seconds(void) {
#ifdef _WIN32
    LARGE_INTEGER frequency, counter;
    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart / (double)frequency.QuadPart;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
#endif
}

/**
 * @brief Start performance timer
 */
static double perf_timer_start(void) {
    return get_time_seconds();
}

/**
 * @brief Stop performance timer and return elapsed time
 */
static double perf_timer_stop(double start_time) {
    return get_time_seconds() - start_time;
}

/**
 * @brief Update memory usage statistics
 */
static void update_memory_stats(size_t bytes_allocated) {
    g_perf_stats.memory_used += bytes_allocated;
    if (g_perf_stats.memory_used > g_perf_stats.peak_memory) {
        g_perf_stats.peak_memory = g_perf_stats.memory_used;
    }
}

/**
 * @brief Record cache hit
 */
static void record_cache_hit(void) {
    g_perf_stats.cache_hits++;
}

/**
 * @brief Record cache miss
 */
static void record_cache_miss(void) {
    g_perf_stats.cache_misses++;
}

/**
 * @brief Get performance statistics
 */
const PerformanceStats* demosaic_get_performance_stats(void) {
    return &g_perf_stats;
}

/**
 * @brief Reset performance statistics
 */
void demosaic_reset_performance_stats(void) {
    memset(&g_perf_stats, 0, sizeof(PerformanceStats));
}

/**
 * @brief Print performance statistics
 */
void demosaic_print_performance_stats(void) {
    printf("\n=== Demosaic Performance Statistics ===\n");
    printf("Demosaic time:        %.3f ms\n", g_perf_stats.demosaic_time * 1000.0);
    printf("Post-process time:    %.3f ms\n", g_perf_stats.postprocess_time * 1000.0);
    printf("Color correction:     %.3f ms\n", g_perf_stats.color_correction_time * 1000.0);
    printf("White balance:        %.3f ms\n", g_perf_stats.white_balance_time * 1000.0);
    printf("Noise reduction:      %.3f ms\n", g_perf_stats.noise_reduction_time * 1000.0);
    printf("Sharpening:           %.3f ms\n", g_perf_stats.sharpening_time * 1000.0);
    printf("Total time:           %.3f ms\n", g_perf_stats.total_time * 1000.0);
    printf("\n");
    printf("Memory used:          %.2f MB\n", g_perf_stats.memory_used / (1024.0 * 1024.0));
    printf("Peak memory:          %.2f MB\n", g_perf_stats.peak_memory / (1024.0 * 1024.0));
    printf("\n");
    printf("Cache hits:           %d\n", g_perf_stats.cache_hits);
    printf("Cache misses:         %d\n", g_perf_stats.cache_misses);
    
    if (g_perf_stats.cache_hits + g_perf_stats.cache_misses > 0) {
        double hit_rate = 100.0 * g_perf_stats.cache_hits / 
                         (g_perf_stats.cache_hits + g_perf_stats.cache_misses);
        printf("Cache hit rate:       %.2f%%\n", hit_rate);
    }
    
    printf("\n");
    printf("SIMD operations:      %d\n", g_perf_stats.simd_operations);
    printf("Scalar operations:    %d\n", g_perf_stats.scalar_operations);
    
    if (g_perf_stats.simd_operations + g_perf_stats.scalar_operations > 0) {
        double simd_rate = 100.0 * g_perf_stats.simd_operations / 
                          (g_perf_stats.simd_operations + g_perf_stats.scalar_operations);
        printf("SIMD usage:           %.2f%%\n", simd_rate);
    }
    
    printf("=======================================\n\n");
}

/**
 * @brief Export performance statistics to JSON
 */
DemosaicError demosaic_export_performance_stats(const char *filename) {
    FILE *fp = fopen(filename, "w");
    if (fp == NULL) {
        set_error(DEMOSAIC_ERROR_FILE_WRITE, "Cannot open performance stats file");
        return DEMOSAIC_ERROR_FILE_WRITE;
    }
    
    fprintf(fp, "{\n");
    fprintf(fp, "  \"timing\": {\n");
    fprintf(fp, "    \"demosaic_ms\": %.3f,\n", g_perf_stats.demosaic_time * 1000.0);
    fprintf(fp, "    \"postprocess_ms\": %.3f,\n", g_perf_stats.postprocess_time * 1000.0);
    fprintf(fp, "    \"color_correction_ms\": %.3f,\n", g_perf_stats.color_correction_time * 1000.0);
    fprintf(fp, "    \"white_balance_ms\": %.3f,\n", g_perf_stats.white_balance_time * 1000.0);
    fprintf(fp, "    \"noise_reduction_ms\": %.3f,\n", g_perf_stats.noise_reduction_time * 1000.0);
    fprintf(fp, "    \"sharpening_ms\": %.3f,\n", g_perf_stats.sharpening_time * 1000.0);
    fprintf(fp, "    \"total_ms\": %.3f\n", g_perf_stats.total_time * 1000.0);
    fprintf(fp, "  },\n");
    fprintf(fp, "  \"memory\": {\n");
    fprintf(fp, "    \"used_mb\": %.2f,\n", g_perf_stats.memory_used / (1024.0 * 1024.0));
    fprintf(fp, "    \"peak_mb\": %.2f\n", g_perf_stats.peak_memory / (1024.0 * 1024.0));
    fprintf(fp, "  },\n");
    fprintf(fp, "  \"cache\": {\n");
    fprintf(fp, "    \"hits\": %d,\n", g_perf_stats.cache_hits);
    fprintf(fp, "    \"misses\": %d,\n", g_perf_stats.cache_misses);
    
    if (g_perf_stats.cache_hits + g_perf_stats.cache_misses > 0) {
        double hit_rate = 100.0 * g_perf_stats.cache_hits / 
                         (g_perf_stats.cache_hits + g_perf_stats.cache_misses);
        fprintf(fp, "    \"hit_rate_percent\": %.2f\n", hit_rate);
    } else {
        fprintf(fp, "    \"hit_rate_percent\": 0.0\n");
    }
    
    fprintf(fp, "  },\n");
    fprintf(fp, "  \"operations\": {\n");
    fprintf(fp, "    \"simd\": %d,\n", g_perf_stats.simd_operations);
    fprintf(fp, "    \"scalar\": %d,\n", g_perf_stats.scalar_operations);
    
    if (g_perf_stats.simd_operations + g_perf_stats.scalar_operations > 0) {
        double simd_rate = 100.0 * g_perf_stats.simd_operations / 
                          (g_perf_stats.simd_operations + g_perf_stats.scalar_operations);
        fprintf(fp, "    \"simd_usage_percent\": %.2f\n", simd_rate);
    } else {
        fprintf(fp, "    \"simd_usage_percent\": 0.0\n");
    }
    
    fprintf(fp, "  }\n");
    fprintf(fp, "}\n");
    
    fclose(fp);
    
    log_message(1, "Performance statistics exported to %s", filename);
    return DEMOSAIC_SUCCESS;
}

// ============================================================================
// Part 7c: Image Cache Management
// ============================================================================

#define MAX_CACHE_ENTRIES 16
#define CACHE_LINE_SIZE 64

/**
 * @brief Cache entry structure
 */
typedef struct {
    char key[256];
    void *data;
    size_t size;
    time_t timestamp;
    int access_count;
    int is_valid;
} CacheEntry;

/**
 * @brief Image cache structure
 */
typedef struct {
    CacheEntry entries[MAX_CACHE_ENTRIES];
    int num_entries;
    size_t total_size;
    size_t max_size;
    pthread_mutex_t mutex;
} ImageCache;

static ImageCache g_image_cache = {0};
static int g_cache_initialized = 0;

/**
 * @brief Initialize image cache
 */
static DemosaicError init_image_cache(size_t max_size_mb) {
    if (g_cache_initialized) {
        return DEMOSAIC_SUCCESS;
    }
    
    memset(&g_image_cache, 0, sizeof(ImageCache));
    g_image_cache.max_size = max_size_mb * 1024 * 1024;
    
#ifdef USE_THREADS
    pthread_mutex_init(&g_image_cache.mutex, NULL);
#endif
    
    g_cache_initialized = 1;
    log_message(1, "Image cache initialized with max size %.2f MB", 
                (double)g_image_cache.max_size / (1024.0 * 1024.0));
    
    return DEMOSAIC_SUCCESS;
}

/**
 * @brief Cleanup image cache
 */
static void cleanup_image_cache(void) {
    if (!g_cache_initialized) {
        return;
    }
    
#ifdef USE_THREADS
    pthread_mutex_lock(&g_image_cache.mutex);
#endif
    
    for (int i = 0; i < MAX_CACHE_ENTRIES; i++) {
        if (g_image_cache.entries[i].is_valid && g_image_cache.entries[i].data != NULL) {
            free(g_image_cache.entries[i].data);
            g_image_cache.entries[i].data = NULL;
            g_image_cache.entries[i].is_valid = 0;
        }
    }
    
    g_image_cache.num_entries = 0;
    g_image_cache.total_size = 0;
    
#ifdef USE_THREADS
    pthread_mutex_unlock(&g_image_cache.mutex);
    pthread_mutex_destroy(&g_image_cache.mutex);
#endif
    
    g_cache_initialized = 0;
    log_message(1, "Image cache cleaned up");
}

/**
 * @brief Generate cache key from image parameters
 */
static void generate_cache_key(
    char *key,
    size_t key_size,
    const char *filename,
    const DemosaicConfig *config
) {
    snprintf(key, key_size, "%s_%d_%d_%d_%d",
             filename,
             config->algorithm,
             config->pattern,
             config->bit_depth,
             config->use_simd ? 1 : 0);
}

/**
 * @brief Find least recently used cache entry
 */
static int find_lru_entry(void) {
    int lru_index = -1;
    time_t oldest_time = time(NULL);
    int min_access = INT_MAX;
    
    for (int i = 0; i < MAX_CACHE_ENTRIES; i++) {
        if (!g_image_cache.entries[i].is_valid) {
            return i;
        }
        
        if (g_image_cache.entries[i].access_count < min_access ||
            (g_image_cache.entries[i].access_count == min_access &&
             g_image_cache.entries[i].timestamp < oldest_time)) {
            lru_index = i;
            oldest_time = g_image_cache.entries[i].timestamp;
            min_access = g_image_cache.entries[i].access_count;
        }
    }
    
    return lru_index;
}

/**
 * @brief Evict cache entry
 */
static void evict_cache_entry(int index) {
    if (index < 0 || index >= MAX_CACHE_ENTRIES) {
        return;
    }
    
    CacheEntry *entry = &g_image_cache.entries[index];
    if (entry->is_valid && entry->data != NULL) {
        log_message(2, "Evicting cache entry: %s (size: %.2f MB)",
                   entry->key, entry->size / (1024.0 * 1024.0));
        
        free(entry->data);
        g_image_cache.total_size -= entry->size;
        g_image_cache.num_entries--;
        
        entry->data = NULL;
        entry->is_valid = 0;
        entry->size = 0;
    }
}

/**
 * @brief Add image to cache
 */
static DemosaicError cache_put(
    const char *key,
    const void *data,
    size_t size
) {
    if (!g_cache_initialized) {
        init_image_cache(100); // Default 100 MB
    }
    
#ifdef USE_THREADS
    pthread_mutex_lock(&g_image_cache.mutex);
#endif
    
    // Check if we need to evict entries
    while (g_image_cache.total_size + size > g_image_cache.max_size &&
           g_image_cache.num_entries > 0) {
        int lru = find_lru_entry();
        evict_cache_entry(lru);
    }
    
    // If still too large, don't cache
    if (size > g_image_cache.max_size) {
        log_message(1, "Image too large to cache: %.2f MB", size / (1024.0 * 1024.0));
#ifdef USE_THREADS
        pthread_mutex_unlock(&g_image_cache.mutex);
#endif
        return DEMOSAIC_ERROR_OUT_OF_MEMORY;
    }
    
    // Find empty slot or LRU
    int index = find_lru_entry();
    if (g_image_cache.entries[index].is_valid) {
        evict_cache_entry(index);
    }
    
    // Add new entry
    CacheEntry *entry = &g_image_cache.entries[index];
    strncpy(entry->key, key, sizeof(entry->key) - 1);
    entry->key[sizeof(entry->key) - 1] = '\0';
    
    entry->data = malloc(size);
    if (entry->data == NULL) {
#ifdef USE_THREADS
        pthread_mutex_unlock(&g_image_cache.mutex);
#endif
        return DEMOSAIC_ERROR_OUT_OF_MEMORY;
    }
    
    memcpy(entry->data, data, size);
    entry->size = size;
    entry->timestamp = time(NULL);
    entry->access_count = 0;
    entry->is_valid = 1;
    
    g_image_cache.total_size += size;
    g_image_cache.num_entries++;
    
    log_message(2, "Cached image: %s (size: %.2f MB, total: %.2f MB)",
               key, size / (1024.0 * 1024.0),
               g_image_cache.total_size / (1024.0 * 1024.0));
    
#ifdef USE_THREADS
    pthread_mutex_unlock(&g_image_cache.mutex);
#endif
    
    return DEMOSAIC_SUCCESS;
}

/**
 * @brief Get image from cache
 */
static void* cache_get(const char *key, size_t *size) {
    if (!g_cache_initialized) {
        return NULL;
    }
    
#ifdef USE_THREADS
    pthread_mutex_lock(&g_image_cache.mutex);
#endif
    
    for (int i = 0; i < MAX_CACHE_ENTRIES; i++) {
        CacheEntry *entry = &g_image_cache.entries[i];
        if (entry->is_valid && strcmp(entry->key, key) == 0) {
            entry->access_count++;
            entry->timestamp = time(NULL);
            
            void *data = malloc(entry->size);
            if (data != NULL) {
                memcpy(data, entry->data, entry->size);
                if (size != NULL) {
                    *size = entry->size;
                }
                
                log_message(2, "Cache hit: %s", key);
                record_cache_hit();
                
#ifdef USE_THREADS
                pthread_mutex_unlock(&g_image_cache.mutex);
#endif
                return data;
            }
        }
    }
    
    log_message(2, "Cache miss: %s", key);
    record_cache_miss();
    
#ifdef USE_THREADS
    pthread_mutex_unlock(&g_image_cache.mutex);
#endif
    
    return NULL;
}

/**
 * @brief Clear all cache entries
 */
void demosaic_cache_clear(void) {
    if (!g_cache_initialized) {
        return;
    }
    
#ifdef USE_THREADS
    pthread_mutex_lock(&g_image_cache.mutex);
#endif
    
    for (int i = 0; i < MAX_CACHE_ENTRIES; i++) {
        evict_cache_entry(i);
    }
    
#ifdef USE_THREADS
    pthread_mutex_unlock(&g_image_cache.mutex);
#endif
    
    log_message(1, "Cache cleared");
}

/**
 * @brief Get cache statistics
 */
void demosaic_cache_stats(int *num_entries, size_t *total_size, size_t *max_size) {
    if (!g_cache_initialized) {
        if (num_entries) *num_entries = 0;
        if (total_size) *total_size = 0;
        if (max_size) *max_size = 0;
        return;
    }
    
#ifdef USE_THREADS
    pthread_mutex_lock(&g_image_cache.mutex);
#endif
    
    if (num_entries) *num_entries = g_image_cache.num_entries;
    if (total_size) *total_size = g_image_cache.total_size;
    if (max_size) *max_size = g_image_cache.max_size;
    
#ifdef USE_THREADS
    pthread_mutex_unlock(&g_image_cache.mutex);
#endif
}

/**
 * @brief Set cache maximum size
 */
void demosaic_cache_set_max_size(size_t max_size_mb) {
    if (!g_cache_initialized) {
        init_image_cache(max_size_mb);
        return;
    }
    
#ifdef USE_THREADS
    pthread_mutex_lock(&g_image_cache.mutex);
#endif
    
    g_image_cache.max_size = max_size_mb * 1024 * 1024;
    
    // Evict entries if over new limit
    while (g_image_cache.total_size > g_image_cache.max_size &&
           g_image_cache.num_entries > 0) {
        int lru = find_lru_entry();
        evict_cache_entry(lru);
    }
    
#ifdef USE_THREADS
    pthread_mutex_unlock(&g_image_cache.mutex);
#endif
    
    log_message(1, "Cache max size set to %.2f MB", 
                (double)g_image_cache.max_size / (1024.0 * 1024.0));
}

// ============================================================================
// End of Part 7c
// ============================================================================
// ============================================================================
// Part 8: Main Processing Pipeline and Examples
// ============================================================================

/**
 * @brief Main demosaic processing function with full pipeline
 */
DemosaicError demosaic_process_image(
    const char *input_file,
    const char *output_file,
    const DemosaicConfig *config
) {
    if (input_file == NULL || output_file == NULL || config == NULL) {
        set_error(DEMOSAIC_ERROR_INVALID_PARAM, "NULL parameter");
        return DEMOSAIC_ERROR_INVALID_PARAM;
    }
    
    log_message(1, "Starting demosaic processing: %s -> %s", input_file, output_file);
    
    double total_start = perf_timer_start();
    DemosaicError error = DEMOSAIC_SUCCESS;
    
    // Check cache first
    char cache_key[512];
    generate_cache_key(cache_key, sizeof(cache_key), input_file, config);
    
    size_t cached_size = 0;
    void *cached_data = cache_get(cache_key, &cached_size);
    
    if (cached_data != NULL && config->use_cache) {
        log_message(1, "Using cached result");
        
        // Write cached data directly to output
        FILE *fp = fopen(output_file, "wb");
        if (fp == NULL) {
            free(cached_data);
            set_error(DEMOSAIC_ERROR_FILE_WRITE, "Cannot open output file");
            return DEMOSAIC_ERROR_FILE_WRITE;
        }
        
        fwrite(cached_data, 1, cached_size, fp);
        fclose(fp);
        free(cached_data);
        
        g_perf_stats.total_time = perf_timer_stop(total_start);
        return DEMOSAIC_SUCCESS;
    }
    
    // Step 1: Load Bayer image
    log_message(1, "Loading Bayer image...");
    double load_start = perf_timer_start();
    
    BayerImage *bayer = bayer_image_create(1, 1, config->bit_depth, config->pattern);
    if (bayer == NULL) {
        return DEMOSAIC_ERROR_OUT_OF_MEMORY;
    }
    
    error = bayer_image_load(bayer, input_file);
    if (error != DEMOSAIC_SUCCESS) {
        bayer_image_destroy(bayer);
        return error;
    }
    
    double load_time = perf_timer_stop(load_start);
    log_message(1, "Image loaded: %dx%d, %d-bit, pattern=%d (%.3f ms)",
                bayer->width, bayer->height, bayer->bit_depth, 
                bayer->pattern, load_time * 1000.0);
    
    // Step 2: Create RGB image
    RGBImage *rgb = rgb_image_create(bayer->width, bayer->height);
    if (rgb == NULL) {
        bayer_image_destroy(bayer);
        return DEMOSAIC_ERROR_OUT_OF_MEMORY;
    }
    
    // Step 3: Demosaic
    log_message(1, "Demosaicing with algorithm %d...", config->algorithm);
    double demosaic_start = perf_timer_start();
    
#ifdef USE_THREADS
    if (config->num_threads > 1) {
        error = demosaic_parallel(bayer, rgb, config);
    } else {
        error = execute_demosaic_algorithm(bayer, rgb, config);
    }
#else
    error = execute_demosaic_algorithm(bayer, rgb, config);
#endif
    
    g_perf_stats.demosaic_time = perf_timer_stop(demosaic_start);
    
    if (error != DEMOSAIC_SUCCESS) {
        rgb_image_destroy(rgb);
        bayer_image_destroy(bayer);
        return error;
    }
    
    log_message(1, "Demosaic complete (%.3f ms)", 
                g_perf_stats.demosaic_time * 1000.0);
    
    // Step 4: White balance
    if (config->white_balance.enabled) {
        log_message(1, "Applying white balance...");
        double wb_start = perf_timer_start();
        
        double wb_r = config->white_balance.r_gain;
        double wb_g = config->white_balance.g_gain;
        double wb_b = config->white_balance.b_gain;
        
        // Auto white balance if gains are zero
        if (wb_r == 0.0 && wb_g == 0.0 && wb_b == 0.0) {
            error = auto_white_balance(rgb, &wb_r, &wb_g, &wb_b);
            if (error != DEMOSAIC_SUCCESS) {
                log_message(0, "Auto white balance failed, using defaults");
                wb_r = wb_g = wb_b = 1.0;
            }
        }
        
#if defined(HAVE_AVX2) && defined(USE_SIMD)
        if (config->use_simd) {
            apply_white_balance_avx2(rgb, wb_r, wb_g, wb_b);
        } else
#elif defined(HAVE_SSE2) && defined(USE_SIMD)
        if (config->use_simd) {
            apply_white_balance_sse2(rgb, wb_r, wb_g, wb_b);
        } else
#elif defined(HAVE_NEON) && defined(USE_SIMD)
        if (config->use_simd) {
            apply_white_balance_neon(rgb, wb_r, wb_g, wb_b);
        } else
#endif
        {
            error = apply_white_balance(rgb, wb_r, wb_g, wb_b);
        }
        
        g_perf_stats.white_balance_time = perf_timer_stop(wb_start);
        
        if (error != DEMOSAIC_SUCCESS) {
            log_message(0, "White balance failed");
        } else {
            log_message(1, "White balance applied: R=%.3f, G=%.3f, B=%.3f (%.3f ms)",
                       wb_r, wb_g, wb_b, g_perf_stats.white_balance_time * 1000.0);
        }
    }
    
    // Step 5: Color correction
    if (config->color_correction.enabled) {
        log_message(1, "Applying color correction...");
        double cc_start = perf_timer_start();
        
        error = apply_color_correction(rgb, config->color_correction.matrix);
        g_perf_stats.color_correction_time = perf_timer_stop(cc_start);
        
        if (error != DEMOSAIC_SUCCESS) {
            log_message(0, "Color correction failed");
        } else {
            log_message(1, "Color correction applied (%.3f ms)",
                       g_perf_stats.color_correction_time * 1000.0);
        }
    }
    
    // Step 6: Gamma correction
    if (config->gamma_correction.enabled) {
        log_message(1, "Applying gamma correction (gamma=%.2f)...",
                   config->gamma_correction.gamma);
        double gamma_start = perf_timer_start();
        
        error = apply_gamma_correction(rgb, config->gamma_correction.gamma);
        double gamma_time = perf_timer_stop(gamma_start);
        
        if (error != DEMOSAIC_SUCCESS) {
            log_message(0, "Gamma correction failed");
        } else {
            log_message(1, "Gamma correction applied (%.3f ms)", gamma_time * 1000.0);
        }
    }
    
    // Step 7: Noise reduction
    if (config->noise_reduction.enabled) {
        log_message(1, "Applying noise reduction (strength=%.2f)...",
                   config->noise_reduction.strength);
        double nr_start = perf_timer_start();
        
        error = apply_noise_reduction(rgb, config->noise_reduction.strength,
                                      config->noise_reduction.method);
        g_perf_stats.noise_reduction_time = perf_timer_stop(nr_start);
        
        if (error != DEMOSAIC_SUCCESS) {
            log_message(0, "Noise reduction failed");
        } else {
            log_message(1, "Noise reduction applied (%.3f ms)",
                       g_perf_stats.noise_reduction_time * 1000.0);
        }
    }
    
    // Step 8: Sharpening
    if (config->sharpening.enabled) {
        log_message(1, "Applying sharpening (amount=%.2f)...",
                   config->sharpening.amount);
        double sharp_start = perf_timer_start();
        
        error = apply_sharpening(rgb, config->sharpening.amount,
                                config->sharpening.radius,
                                config->sharpening.threshold);
        g_perf_stats.sharpening_time = perf_timer_stop(sharp_start);
        
        if (error != DEMOSAIC_SUCCESS) {
            log_message(0, "Sharpening failed");
        } else {
            log_message(1, "Sharpening applied (%.3f ms)",
                       g_perf_stats.sharpening_time * 1000.0);
        }
    }
    
    // Step 9: Save output
    log_message(1, "Saving output image...");
    double save_start = perf_timer_start();
    
    error = rgb_image_save(rgb, output_file, config->output_format);
    double save_time = perf_timer_stop(save_start);
    
    if (error != DEMOSAIC_SUCCESS) {
        rgb_image_destroy(rgb);
        bayer_image_destroy(bayer);
        return error;
    }
    
    log_message(1, "Output saved (%.3f ms)", save_time * 1000.0);
    
    // Step 10: Cache result if enabled
    if (config->use_cache) {
        FILE *fp = fopen(output_file, "rb");
        if (fp != NULL) {
            fseek(fp, 0, SEEK_END);
            long file_size = ftell(fp);
            fseek(fp, 0, SEEK_SET);
            
            void *file_data = malloc(file_size);
            if (file_data != NULL) {
                fread(file_data, 1, file_size, fp);
                cache_put(cache_key, file_data, file_size);
                free(file_data);
            }
            fclose(fp);
        }
    }
    
    // Cleanup
    rgb_image_destroy(rgb);
    bayer_image_destroy(bayer);
    
    g_perf_stats.total_time = perf_timer_stop(total_start);
    
    log_message(1, "Processing complete! Total time: %.3f ms",
                g_perf_stats.total_time * 1000.0);
    
    return DEMOSAIC_SUCCESS;
}

/**
 * @brief Batch process multiple images
 */
DemosaicError demosaic_batch_process(
    const char **input_files,
    const char **output_files,
    int num_files,
    const DemosaicConfig *config
) {
    if (input_files == NULL || output_files == NULL || num_files <= 0) {
        set_error(DEMOSAIC_ERROR_INVALID_PARAM, "Invalid batch parameters");
        return DEMOSAIC_ERROR_INVALID_PARAM;
    }
    
    log_message(1, "Starting batch processing of %d images", num_files);
    
    int success_count = 0;
    int fail_count = 0;
    
    double batch_start = perf_timer_start();
    
    for (int i = 0; i < num_files; i++) {
        log_message(1, "\n=== Processing image %d/%d: %s ===",
                   i + 1, num_files, input_files[i]);
        
        DemosaicError error = demosaic_process_image(
            input_files[i],
            output_files[i],
            config
        );
        
        if (error == DEMOSAIC_SUCCESS) {
            success_count++;
            log_message(1, "Image %d processed successfully", i + 1);
        } else {
            fail_count++;
            log_message(0, "Image %d failed with error: %s",
                       i + 1, demosaic_get_error_string());
        }
    }
    
    double batch_time = perf_timer_stop(batch_start);
    
    log_message(1, "\n=== Batch Processing Summary ===");
    log_message(1, "Total images: %d", num_files);
    log_message(1, "Successful: %d", success_count);
    log_message(1, "Failed: %d", fail_count);
    log_message(1, "Total time: %.3f seconds", batch_time);
    log_message(1, "Average time per image: %.3f seconds",
                batch_time / num_files);
    
    return (fail_count == 0) ? DEMOSAIC_SUCCESS : DEMOSAIC_ERROR_PROCESSING;
}

/**
 * @brief Compare two demosaic algorithms
 */
DemosaicError demosaic_compare_algorithms(
    const char *input_file,
    const char *output_prefix,
    DemosaicAlgorithm *algorithms,
    int num_algorithms,
    const DemosaicConfig *base_config
) {
    if (input_file == NULL || output_prefix == NULL || 
        algorithms == NULL || num_algorithms <= 0) {
        set_error(DEMOSAIC_ERROR_INVALID_PARAM, "Invalid comparison parameters");
        return DEMOSAIC_ERROR_INVALID_PARAM;
    }
    
    log_message(1, "Comparing %d demosaic algorithms", num_algorithms);
    
    printf("\n=== Algorithm Comparison ===\n");
    printf("Input: %s\n\n", input_file);
    printf("%-20s %12s %12s %12s\n", 
           "Algorithm", "Time (ms)", "Memory (MB)", "Quality");
    printf("%-20s %12s %12s %12s\n",
           "--------------------", "------------", "------------", "------------");
    
    for (int i = 0; i < num_algorithms; i++) {
        // Create config for this algorithm
        DemosaicConfig config = *base_config;
        config.algorithm = algorithms[i];
        
        // Generate output filename
        char output_file[512];
        const char *algo_name = demosaic_algorithm_name(algorithms[i]);
        snprintf(output_file, sizeof(output_file), "%s_%s.png",
                output_prefix, algo_name);
        
        // Reset performance stats
        demosaic_reset_performance_stats();
        
        // Process image
        DemosaicError error = demosaic_process_image(
            input_file,
            output_file,
            &config
        );
        
        if (error == DEMOSAIC_SUCCESS) {
            const PerformanceStats *stats = demosaic_get_performance_stats();
            
            printf("%-20s %12.3f %12.2f %12s\n",
                   algo_name,
                   stats->total_time * 1000.0,
                   stats->peak_memory / (1024.0 * 1024.0),
                   "N/A");
        } else {
            printf("%-20s %12s %12s %12s\n",
                   algo_name, "FAILED", "-", "-");
            log_message(0, "Algorithm %s failed: %s",
                       algo_name, demosaic_get_error_string());
        }
    }
    
    printf("\n");
    
    return DEMOSAIC_SUCCESS;
}

// ============================================================================
// Part 8: Command Line Interface and Main Function
// ============================================================================

/**
 * @brief Print usage information
 */
static void print_usage(const char *program_name) {
    printf("Usage: %s [options] input_file output_file\n\n", program_name);
    printf("Options:\n");
    printf("  -a, --algorithm <name>    Demosaic algorithm (default: bilinear)\n");
    printf("                            Available: bilinear, vng, ahd, ppg, lmmse\n");
    printf("  -p, --pattern <name>      Bayer pattern (default: rggb)\n");
    printf("                            Available: rggb, bggr, grbg, gbrg\n");
    printf("  -b, --bits <n>            Bit depth (default: 16)\n");
    printf("  -f, --format <name>       Output format (default: png)\n");
    printf("                            Available: png, tiff, ppm\n");
    printf("  -w, --white-balance       Enable auto white balance\n");
    printf("  -g, --gamma <value>       Gamma correction (default: 2.2)\n");
    printf("  -n, --noise-reduction <s> Noise reduction strength (0.0-1.0)\n");
    printf("  -s, --sharpen <amount>    Sharpening amount (0.0-2.0)\n");
    printf("  -t, --threads <n>         Number of threads (default: 1)\n");
    printf("  --simd                    Enable SIMD optimizations\n");
    printf("  --no-cache                Disable result caching\n");
    printf("  -v, --verbose             Increase verbosity\n");
    printf("  -q, --quiet               Decrease verbosity\n");
    printf("  --stats                   Print performance statistics\n");
    printf("  --compare                 Compare all algorithms\n");
    printf("  -h, --help                Show this help message\n");
    printf("\n");
    printf("Examples:\n");
    printf("  %s input.raw output.png\n", program_name);
    printf("  %s -a vng -w -g 2.2 input.raw output.png\n", program_name);
    printf("  %s --compare input.raw output\n", program_name);
    printf("\n");
}

/**
 * @brief Parse command line arguments
 */
static DemosaicError parse_arguments(
    int argc,
    char **argv,
    char **input_file,
    char **output_file,
    DemosaicConfig *config,
    int *show_stats,
    int *compare_mode
) {
    *show_stats = 0;
    *compare_mode = 0;
    
    // Set defaults
    demosaic_config_init(config);
    
    int i = 1;
    while (i < argc) {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            exit(0);
        }
        else if (strcmp(argv[i], "-a") == 0 || strcmp(argv[i], "--algorithm") == 0) {
            if (++i >= argc) {
                fprintf(stderr, "Error: Missing algorithm name\n");
                return DEMOSAIC_ERROR_INVALID_PARAM;
            }
            
            if (strcmp(argv[i], "bilinear") == 0) {
                config->algorithm = DEMOSAIC_BILINEAR;
            } else if (strcmp(argv[i], "vng") == 0) {
                config->algorithm = DEMOSAIC_VNG;
            } else if (strcmp(argv[i], "ahd") == 0) {
                config->algorithm = DEMOSAIC_AHD;
            } else if (strcmp(argv[i], "ppg") == 0) {
                config->algorithm = DEMOSAIC_PPG;
            } else if (strcmp(argv[i], "lmmse") == 0) {
                config->algorithm = DEMOSAIC_LMMSE;
            } else {
                fprintf(stderr, "Error: Unknown algorithm: %s\n", argv[i]);
                return DEMOSAIC_ERROR_INVALID_PARAM;
            }
        }
        else if (strcmp(argv[i], "-p") == 0 || strcmp(argv[i], "--pattern") == 0) {
            if (++i >= argc) {
                fprintf(stderr, "Error: Missing pattern name\n");
                return DEMOSAIC_ERROR_INVALID_PARAM;
            }
            
            if (strcmp(argv[i], "rggb") == 0) {
                config->pattern = BAYER_RGGB;
            } else if (strcmp(argv[i], "bggr") == 0) {
                config->pattern = BAYER_BGGR;
            } else if (strcmp(argv[i], "grbg") == 0) {
                config->pattern = BAYER_GRBG;
            } else if (strcmp(argv[i], "gbrg") == 0) {
                config->pattern = BAYER_GBRG;
            } else {
                fprintf(stderr, "Error: Unknown pattern: %s\n", argv[i]);
                return DEMOSAIC_ERROR_INVALID_PARAM;
            }
        }
        else if (strcmp(argv[i], "-b") == 0 || strcmp(argv[i], "--bits") == 0) {
            if (++i >= argc) {
                fprintf(stderr, "Error: Missing bit depth\n");
                return DEMOSAIC_ERROR_INVALID_PARAM;
            }
            config->bit_depth = atoi(argv[i]);
        }
        else if (strcmp(argv[i], "-f") == 0 || strcmp(argv[i], "--format") == 0) {
            if (++i >= argc) {
                fprintf(stderr, "Error: Missing format name\n");
                return DEMOSAIC_ERROR_INVALID_PARAM;
            }
            
            if (strcmp(argv[i], "png") == 0) {
                config->output_format = OUTPUT_PNG;
            } else if (strcmp(argv[i], "tiff") == 0) {
                config->output_format = OUTPUT_TIFF;
            } else if (strcmp(argv[i], "ppm") == 0) {
                config->output_format = OUTPUT_PPM;
            } else {
                fprintf(stderr, "Error: Unknown format: %s\n", argv[i]);
                return DEMOSAIC_ERROR_INVALID_PARAM;
            }
        }
        else if (strcmp(argv[i], "-w") == 0 || strcmp(argv[i], "--white-balance") == 0) {
            config->white_balance.enabled = 1;
        }
        else if (strcmp(argv[i], "-g") == 0 || strcmp(argv[i], "--gamma") == 0) {
            if (++i >= argc) {
                fprintf(stderr, "Error: Missing gamma value\n");
                return DEMOSAIC_ERROR_INVALID_PARAM;
            }
            config->gamma_correction.enabled = 1;
            config->gamma_correction.gamma = atof(argv[i]);
        }
        else if (strcmp(argv[i], "-n") == 0 || strcmp(argv[i], "--noise-reduction") == 0) {
            if (++i >= argc) {
                fprintf(stderr, "Error: Missing noise reduction strength\n");
                return DEMOSAIC_ERROR_INVALID_PARAM;
            }
            config->noise_reduction.enabled = 1;
            config->noise_reduction.strength = atof(argv[i]);
        }
        else if (strcmp(argv[i], "-s") == 0 || strcmp(argv[i], "--sharpen") == 0) {
            if (++i >= argc) {
                fprintf(stderr, "Error: Missing sharpening amount\n");
                return DEMOSAIC_ERROR_INVALID_PARAM;
            }
            config->sharpening.enabled = 1;
            config->sharpening.amount = atof(argv[i]);
        }
        else if (strcmp(argv[i], "-t") == 0 || strcmp(argv[i], "--threads") == 0) {
            if (++i >= argc) {
                fprintf(stderr, "Error: Missing thread count\n");
                return DEMOSAIC_ERROR_INVALID_PARAM;
            }
            config->num_threads = atoi(argv[i]);
        }
        else if (strcmp(argv[i], "--simd") == 0) {
            config->use_simd = 1;
        }
        else if (strcmp(argv[i], "--no-cache") == 0) {
            config->use_cache = 0;
        }
        else if (strcmp(argv[i], "-v") == 0 || strcmp(argv[i], "--verbose") == 0) {
            if (g_log_level < 3) g_log_level++;
        }
        else if (strcmp(argv[i], "-q") == 0 || strcmp(argv[i], "--quiet") == 0) {
            if (g_log_level > 0) g_log_level--;
        }
        else if (strcmp(argv[i], "--stats") == 0) {
            *show_stats = 1;
        }
        else if (strcmp(argv[i], "--compare") == 0) {
            *compare_mode = 1;
        }
        else if (argv[i][0] == '-') {
            fprintf(stderr, "Error: Unknown option: %s\n", argv[i]);
            return DEMOSAIC_ERROR_INVALID_PARAM;
        }
        else {
            // Positional arguments
            if (*input_file == NULL) {
                *input_file = argv[i];
            } else if (*output_file == NULL) {
                *output_file = argv[i];
            } else {
                fprintf(stderr, "Error: Too many positional arguments\n");
                return DEMOSAIC_ERROR_INVALID_PARAM;
            }
        }
        
        i++;
    }
    
    // Validate required arguments
    if (*input_file == NULL) {
        fprintf(stderr, "Error: Missing input file\n");
        print_usage(argv[0]);
        return DEMOSAIC_ERROR_INVALID_PARAM;
    }
    
    if (*output_file == NULL && !*compare_mode) {
        fprintf(stderr, "Error: Missing output file\n");
        print_usage(argv[0]);
        return DEMOSAIC_ERROR_INVALID_PARAM;
    }
    
    return DEMOSAIC_SUCCESS;
}

/**
 * @brief Main function
 */
int main(int argc, char **argv) {
    printf("Demosaic Tool v1.0\n");
    printf("Advanced Bayer Pattern Demosaicing\n\n");
    
    // Parse arguments
    char *input_file = NULL;
    char *output_file = NULL;
    DemosaicConfig config;
    int show_stats = 0;
    int compare_mode = 0;
    
    DemosaicError error = parse_arguments(
        argc, argv,
        &input_file, &output_file,
        &config, &show_stats, &compare_mode
    );
    
    if (error != DEMOSAIC_SUCCESS) {
        return 1;
    }
    
    // Initialize library
    error = demosaic_init();
    if (error != DEMOSAIC_SUCCESS) {
        fprintf(stderr, "Failed to initialize demosaic library: %s\n",
                demosaic_get_error_string());
        return 1;
    }
    
    // Check SIMD support
    if (config.use_simd) {
        int simd_support = check_simd_support();
        if (simd_support == 0) {
            log_message(1, "Warning: SIMD requested but not available");
            config.use_simd = 0;
        }
    }
    
    // Process image(s)
    if (compare_mode) {
        // Compare all algorithms
        DemosaicAlgorithm algorithms[] = {
            DEMOSAIC_BILINEAR,
            DEMOSAIC_VNG,
            DEMOSAIC_AHD,
            DEMOSAIC_PPG,
            DEMOSAIC_LMMSE
        };
        
        error = demosaic_compare_algorithms(
            input_file,
            output_file ? output_file : "output",
            algorithms,
            sizeof(algorithms) / sizeof(algorithms[0]),
            &config
        );
    } else {
        // Single image processing
        error = demosaic_process_image(input_file, output_file, &config);
    }
    
    // Show statistics if requested
    if (show_stats) {
        demosaic_print_performance_stats();
        
        int cache_entries;
        size_t cache_size, cache_max;
        demosaic_cache_stats(&cache_entries, &cache_size, &cache_max);
        
        printf("Cache: %d entries, %.2f MB / %.2f MB\n",
               cache_entries,
               cache_size / (1024.0 * 1024.0),
               cache_max / (1024.0 * 1024.0));
    }
    
    // Cleanup
    demosaic_cleanup();
    
    if (error == DEMOSAIC_SUCCESS) {
        printf("\nProcessing completed successfully!\n");
        return 0;
    } else {
        fprintf(stderr, "\nProcessing failed: %s\n", demosaic_get_error_string());
        return 1;
    }
}

// ============================================================================
// End of demosaic.c - Complete Implementation
// ============================================================================

