/**
 * @file demosaic.h
 * @brief Bayer Pattern Demosaicing Library Header - Part 1
 * @author hany
 * @version 1.0.0
 * @date 2025
 * 
 * @description
 * Bayer去马赛克算法库完整头文件 - 第一部分
 * 包含：基础定义、平台检测、错误码、枚举类型
 * 
 * @copyright Copyright (c) 2024
 * @license MIT License
 */

#ifndef DEMOSAIC_H
#define DEMOSAIC_H

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Standard Headers
// ============================================================================

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

// ============================================================================
// Version Information
// ============================================================================

#define DEMOSAIC_VERSION_MAJOR      2
#define DEMOSAIC_VERSION_MINOR      0
#define DEMOSAIC_VERSION_PATCH      0
#define DEMOSAIC_VERSION_STRING     "2.0.0"

// ============================================================================
// Platform Detection
// ============================================================================

// Operating System
#if defined(_WIN32) || defined(_WIN64)
    #define DEMOSAIC_PLATFORM_WINDOWS
#elif defined(__linux__)
    #define DEMOSAIC_PLATFORM_LINUX
#elif defined(__APPLE__)
    #define DEMOSAIC_PLATFORM_MACOS
#endif

// Compiler
#if defined(_MSC_VER)
    #define DEMOSAIC_COMPILER_MSVC
#elif defined(__GNUC__)
    #define DEMOSAIC_COMPILER_GCC
#elif defined(__clang__)
    #define DEMOSAIC_COMPILER_CLANG
#endif

// ============================================================================
// API Macros
// ============================================================================

#ifdef DEMOSAIC_PLATFORM_WINDOWS
    #ifdef DEMOSAIC_BUILD_SHARED
        #define DEMOSAIC_API __declspec(dllexport)
    #else
        #define DEMOSAIC_API __declspec(dllimport)
    #endif
#else
    #define DEMOSAIC_API __attribute__((visibility("default")))
#endif

// Inline
#if defined(DEMOSAIC_COMPILER_MSVC)
    #define DEMOSAIC_INLINE __forceinline
#else
    #define DEMOSAIC_INLINE __attribute__((always_inline)) inline
#endif

// Alignment
#if defined(DEMOSAIC_COMPILER_MSVC)
    #define DEMOSAIC_ALIGN(x) __declspec(align(x))
#else
    #define DEMOSAIC_ALIGN(x) __attribute__((aligned(x)))
#endif

// ============================================================================
// Constants
// ============================================================================

#define DEMOSAIC_MAX_WIDTH          65536
#define DEMOSAIC_MAX_HEIGHT         65536
#define DEMOSAIC_MIN_WIDTH          2
#define DEMOSAIC_MIN_HEIGHT         2

#define DEMOSAIC_MAX_CHANNELS       4
#define DEMOSAIC_MAX_THREADS        256

#define DEMOSAIC_EPSILON            1e-10
#define DEMOSAIC_PI                 3.14159265358979323846

// ============================================================================
// Utility Macros
// ============================================================================

#define DEMOSAIC_MIN(a, b)          ((a) < (b) ? (a) : (b))
#define DEMOSAIC_MAX(a, b)          ((a) > (b) ? (a) : (b))
#define DEMOSAIC_CLAMP(x, min, max) DEMOSAIC_MIN(DEMOSAIC_MAX(x, min), max)
#define DEMOSAIC_ABS(x)             ((x) < 0 ? -(x) : (x))
#define DEMOSAIC_SQR(x)             ((x) * (x))

// ============================================================================
// Error Codes
// ============================================================================

/**
 * @brief Error codes for demosaicing operations
 */
typedef enum {
    DEMOSAIC_SUCCESS = 0,                   ///< Success
    
    // General errors (1-99)
    DEMOSAIC_ERROR_NULL_POINTER = 1,        ///< Null pointer
    DEMOSAIC_ERROR_INVALID_PARAM = 2,       ///< Invalid parameter
    DEMOSAIC_ERROR_MEMORY_ALLOCATION = 3,   ///< Memory allocation failed
    DEMOSAIC_ERROR_OUT_OF_MEMORY = 4,       ///< Out of memory
    
    // Image errors (100-199)
    DEMOSAIC_ERROR_INVALID_IMAGE = 100,     ///< Invalid image
    DEMOSAIC_ERROR_INVALID_WIDTH = 101,     ///< Invalid width
    DEMOSAIC_ERROR_INVALID_HEIGHT = 102,    ///< Invalid height
    DEMOSAIC_ERROR_INVALID_CHANNELS = 103,  ///< Invalid channels
    DEMOSAIC_ERROR_INVALID_BIT_DEPTH = 104, ///< Invalid bit depth
    DEMOSAIC_ERROR_DIMENSION_MISMATCH = 105,///< Dimension mismatch
    
    // Pattern errors (200-299)
    DEMOSAIC_ERROR_INVALID_PATTERN = 200,   ///< Invalid Bayer pattern
    DEMOSAIC_ERROR_PATTERN_DETECTION = 201, ///< Pattern detection failed
    
    // Algorithm errors (300-399)
    DEMOSAIC_ERROR_INVALID_METHOD = 300,    ///< Invalid method
    DEMOSAIC_ERROR_NOT_IMPLEMENTED = 301,   ///< Not implemented
    DEMOSAIC_ERROR_CONVERGENCE = 302,       ///< Convergence failed
    
    // File I/O errors (400-499)
    DEMOSAIC_ERROR_FILE_NOT_FOUND = 400,    ///< File not found
    DEMOSAIC_ERROR_FILE_OPEN = 401,         ///< Cannot open file
    DEMOSAIC_ERROR_FILE_READ = 402,         ///< Read error
    DEMOSAIC_ERROR_FILE_WRITE = 403,        ///< Write error
    DEMOSAIC_ERROR_FILE_FORMAT = 404,       ///< Invalid format
    
    // Hardware errors (500-599)
    DEMOSAIC_ERROR_CUDA = 500,              ///< CUDA error
    DEMOSAIC_ERROR_OPENCL = 510,            ///< OpenCL error
    
    DEMOSAIC_ERROR_UNKNOWN = -1             ///< Unknown error
} DemosaicError;

// ============================================================================
// Bayer Pattern Types
// ============================================================================

/**
 * @brief Bayer pattern types
 * 
 * Pattern layout (2x2):
 * RGGB: R G    BGGR: B G    GRBG: G R    GBRG: G B
 *       G B          G R          B G          R G
 */
typedef enum {
    BAYER_RGGB = 0,     ///< Red-Green-Green-Blue
    BAYER_BGGR = 1,     ///< Blue-Green-Green-Red
    BAYER_GRBG = 2,     ///< Green-Red-Blue-Green
    BAYER_GBRG = 3,     ///< Green-Blue-Red-Green
    BAYER_UNKNOWN = -1  ///< Unknown pattern
} BayerPattern;

// ============================================================================
// Demosaicing Methods
// ============================================================================

/**
 * @brief Demosaicing algorithm methods
 */
typedef enum {
    DEMOSAIC_NEAREST = 0,           ///< Nearest neighbor
    DEMOSAIC_BILINEAR,              ///< Bilinear interpolation
    DEMOSAIC_EDGE_AWARE,            ///< Edge-aware
    DEMOSAIC_GRADIENT_CORRECTED,    ///< Gradient-corrected
    DEMOSAIC_MALVAR,                ///< Malvar-He-Cutler
    DEMOSAIC_AHD,                   ///< Adaptive Homogeneity-Directed
    DEMOSAIC_VNG,                   ///< Variable Number of Gradients
    DEMOSAIC_PPG,                   ///< Pixel Grouping
    DEMOSAIC_LMMSE,                 ///< Linear MMSE
    DEMOSAIC_CNN,                   ///< CNN-based
    DEMOSAIC_AUTO                   ///< Auto selection
} DemosaicMethod;

// ============================================================================
// Color Space Types
// ============================================================================

/**
 * @brief Color space types
 */
typedef enum {
    COLOR_SPACE_RGB = 0,    ///< RGB
    COLOR_SPACE_SRGB,       ///< sRGB
    COLOR_SPACE_LINEAR,     ///< Linear RGB
    COLOR_SPACE_XYZ,        ///< CIE XYZ
    COLOR_SPACE_LAB,        ///< CIE L*a*b*
    COLOR_SPACE_YUV,        ///< YUV
    COLOR_SPACE_HSV,        ///< HSV
    COLOR_SPACE_HSL         ///< HSL
} ColorSpace;

// ============================================================================
// Processing Options
// ============================================================================

/**
 * @brief Processing backend types
 */
typedef enum {
    BACKEND_CPU = 0,        ///< CPU
    BACKEND_OPENMP,         ///< OpenMP
    BACKEND_CUDA,           ///< CUDA
    BACKEND_OPENCL,         ///< OpenCL
    BACKEND_AUTO            ///< Auto
} ProcessingBackend;

/**
 * @brief Quality level presets
 */
typedef enum {
    QUALITY_FAST = 0,       ///< Fast
    QUALITY_BALANCED,       ///< Balanced
    QUALITY_HIGH,           ///< High
    QUALITY_BEST            ///< Best
} QualityLevel;

/**
 * @brief Edge detection methods
 */
typedef enum {
    EDGE_SOBEL = 0,         ///< Sobel
    EDGE_PREWITT,           ///< Prewitt
    EDGE_SCHARR,            ///< Scharr
    EDGE_LAPLACIAN,         ///< Laplacian
    EDGE_CANNY              ///< Canny
} EdgeDetectionMethod;

/**
 * @brief Interpolation methods
 */
typedef enum {
    INTERP_NEAREST = 0,     ///< Nearest
    INTERP_BILINEAR,        ///< Bilinear
    INTERP_BICUBIC,         ///< Bicubic
    INTERP_LANCZOS          ///< Lanczos
} InterpolationMethod;

// ============================================================================
// End of Part 1
// ============================================================================
// ============================================================================
// Part 2: Data Structures
// ============================================================================

// ============================================================================
// Basic Structures
// ============================================================================

/**
 * @brief 2D Point structure
 */
typedef struct {
    int x;                      ///< X coordinate
    int y;                      ///< Y coordinate
} Point2D;

/**
 * @brief 2D Size structure
 */
typedef struct {
    int width;                  ///< Width
    int height;                 ///< Height
} Size2D;

/**
 * @brief Rectangle structure
 */
typedef struct {
    int x;                      ///< X coordinate
    int y;                      ///< Y coordinate
    int width;                  ///< Width
    int height;                 ///< Height
} Rectangle;

/**
 * @brief Color structure (RGB)
 */
typedef struct {
    double r;                   ///< Red channel [0, 1]
    double g;                   ///< Green channel [0, 1]
    double b;                   ///< Blue channel [0, 1]
} Color;

/**
 * @brief Color with alpha
 */
typedef struct {
    double r;                   ///< Red channel [0, 1]
    double g;                   ///< Green channel [0, 1]
    double b;                   ///< Blue channel [0, 1]
    double a;                   ///< Alpha channel [0, 1]
} ColorRGBA;

// ============================================================================
// Image Structures
// ============================================================================

/**
 * @brief Raw Bayer image structure
 */
typedef struct {
    uint16_t *data;             ///< Raw Bayer data (single channel)
    int width;                  ///< Image width in pixels
    int height;                 ///< Image height in pixels
    BayerPattern pattern;       ///< Bayer pattern type
    int bit_depth;              ///< Bit depth (8, 10, 12, 14, 16)
    
    // Level information
    double black_level;         ///< Black level offset
    double white_level;         ///< White level (saturation point)
    
    // Memory management
    bool owns_data;             ///< Whether this structure owns the data
    size_t data_size;           ///< Size of data buffer in bytes
    
    // Metadata
    void *metadata;             ///< Optional metadata pointer
} BayerImage;

/**
 * @brief RGB image structure
 */
typedef struct {
    double *data;               ///< RGB data (interleaved: RGBRGB...)
    int width;                  ///< Image width in pixels
    int height;                 ///< Image height in pixels
    int channels;               ///< Number of channels (typically 3)
    ColorSpace color_space;     ///< Color space
    
    // Planar data (optional, for some algorithms)
    double *r_channel;          ///< Red channel (planar)
    double *g_channel;          ///< Green channel (planar)
    double *b_channel;          ///< Blue channel (planar)
    
    // Memory management
    bool owns_data;             ///< Whether this structure owns the data
    bool is_planar;             ///< Whether data is in planar format
    size_t data_size;           ///< Size of data buffer in bytes
    
    // Metadata
    void *metadata;             ///< Optional metadata pointer
} RGBImage;

/**
 * @brief Image metadata structure
 */
typedef struct {
    // Camera information
    char camera_make[64];       ///< Camera manufacturer
    char camera_model[64];      ///< Camera model
    char lens_model[64];        ///< Lens model
    
    // Capture settings
    double iso;                 ///< ISO sensitivity
    double exposure_time;       ///< Exposure time (seconds)
    double aperture;            ///< Aperture (f-number)
    double focal_length;        ///< Focal length (mm)
    
    // Image properties
    int orientation;            ///< EXIF orientation
    double pixel_pitch;         ///< Pixel pitch (micrometers)
    
    // Timestamp
    char capture_time[32];      ///< Capture timestamp
    
    // Custom data
    void *custom_data;          ///< Custom metadata
    size_t custom_data_size;    ///< Size of custom data
} ImageMetadata;

// ============================================================================
// Color Processing Structures
// ============================================================================

/**
 * @brief White balance coefficients
 */
typedef struct {
    double r_gain;              ///< Red channel gain
    double g_gain;              ///< Green channel gain
    double b_gain;              ///< Blue channel gain
    
    // Optional color temperature info
    double temperature;         ///< Color temperature (Kelvin)
    double tint;                ///< Tint adjustment
    
    // Normalization
    bool normalized;            ///< Whether gains are normalized
} WhiteBalance;

/**
 * @brief Color correction matrix (3x3)
 */
typedef struct {
    double matrix[3][3];        ///< 3x3 transformation matrix
    char name[64];              ///< Matrix name/description
    bool is_identity;           ///< Whether this is identity matrix
} ColorMatrix;

/**
 * @brief Tone curve structure
 */
typedef struct {
    int num_points;             ///< Number of curve points
    double *input;              ///< Input values [0, 1]
    double *output;             ///< Output values [0, 1]
    
    // Curve type
    enum {
        TONE_CURVE_LINEAR,
        TONE_CURVE_GAMMA,
        TONE_CURVE_CUSTOM
    } type;
    
    double gamma;               ///< Gamma value (if type is GAMMA)
} ToneCurve;

/**
 * @brief Color profile structure
 */
typedef struct {
    char name[64];              ///< Profile name
    ColorSpace color_space;     ///< Color space
    WhiteBalance white_balance; ///< White balance
    ColorMatrix color_matrix;   ///< Color correction matrix
    ToneCurve tone_curve;       ///< Tone curve
    double gamma;               ///< Gamma value
} ColorProfile;

// ============================================================================
// Configuration Structures
// ============================================================================

/**
 * @brief Demosaicing configuration
 */
typedef struct {
    // Algorithm settings
    DemosaicMethod method;              ///< Demosaicing method
    QualityLevel quality;               ///< Quality level
    
    // Processing options
    ProcessingBackend backend;          ///< Processing backend
    int num_threads;                    ///< Number of threads (0 = auto)
    bool use_simd;                      ///< Enable SIMD optimization
    
    // Color processing
    bool apply_white_balance;           ///< Apply white balance
    WhiteBalance white_balance;         ///< White balance coefficients
    bool apply_color_correction;        ///< Apply color correction
    ColorMatrix color_matrix;           ///< Color correction matrix
    ColorSpace output_color_space;      ///< Output color space
    
    // Image enhancement
    bool denoise;                       ///< Apply denoising
    double denoise_strength;            ///< Denoising strength [0, 1]
    bool sharpen;                       ///< Apply sharpening
    double sharpen_amount;              ///< Sharpening amount [0, 1]
    double sharpen_radius;              ///< Sharpening radius
    
    // Artifact suppression
    bool suppress_false_color;          ///< Suppress false color
    double false_color_strength;        ///< False color suppression [0, 1]
    bool suppress_zipper;               ///< Suppress zipper artifacts
    double zipper_strength;             ///< Zipper suppression [0, 1]
    bool suppress_moire;                ///< Suppress moiré
    double moire_strength;              ///< Moiré suppression [0, 1]
    
    // Edge detection
    EdgeDetectionMethod edge_method;    ///< Edge detection method
    double edge_threshold;              ///< Edge detection threshold
    
    // Advanced options
    int window_size;                    ///< Processing window size
    double gradient_threshold;          ///< Gradient threshold
    int max_iterations;                 ///< Maximum iterations
    double convergence_threshold;       ///< Convergence threshold
    
    // Interpolation
    InterpolationMethod interp_method;  ///< Interpolation method
    
    // Debug options
    bool verbose;                       ///< Verbose output
    bool save_intermediate;             ///< Save intermediate results
    char output_dir[256];               ///< Output directory
    
    // Custom parameters
    void *custom_params;                ///< Custom parameters
    size_t custom_params_size;          ///< Size of custom parameters
} DemosaicConfig;

/**
 * @brief Processing statistics
 */
typedef struct {
    // Timing information
    double total_time;                  ///< Total processing time (seconds)
    double demosaic_time;               ///< Demosaicing time
    double color_correction_time;       ///< Color correction time
    double enhancement_time;            ///< Enhancement time
    double io_time;                     ///< I/O time
    
    // Quality metrics
    double psnr;                        ///< Peak Signal-to-Noise Ratio (dB)
    double ssim;                        ///< Structural Similarity Index
    double mse;                         ///< Mean Squared Error
    double mae;                         ///< Mean Absolute Error
    
    // Artifact metrics
    double zipper_metric;               ///< Zipper artifact metric
    double false_color_metric;          ///< False color metric
    double moire_metric;                ///< Moiré metric
    
    // Image statistics (per channel)
    double mean_r, mean_g, mean_b;      ///< Mean values
    double std_r, std_g, std_b;         ///< Standard deviations
    double min_r, min_g, min_b;         ///< Minimum values
    double max_r, max_g, max_b;         ///< Maximum values
    
    // Histogram statistics
    double median_r, median_g, median_b;///< Median values
    double q25_r, q25_g, q25_b;         ///< 25th percentile
    double q75_r, q75_g, q75_b;         ///< 75th percentile
    
    // Processing information
    int num_threads_used;               ///< Number of threads used
    ProcessingBackend backend_used;     ///< Backend used
    char method_name[64];               ///< Method name
    
    // Memory usage
    size_t peak_memory_usage;           ///< Peak memory usage (bytes)
    size_t current_memory_usage;        ///< Current memory usage (bytes)
    
    // Error information
    int num_warnings;                   ///< Number of warnings
    int num_errors;                     ///< Number of errors
} DemosaicStats;

// ============================================================================
// Algorithm-Specific Structures
// ============================================================================

/**
 * @brief Edge direction structure
 */
typedef struct {
    double horizontal;                  ///< Horizontal gradient
    double vertical;                    ///< Vertical gradient
    double diagonal_pos;                ///< Positive diagonal (/)
    double diagonal_neg;                ///< Negative diagonal (\)
    double angle;                       ///< Edge angle (radians)
    double strength;                    ///< Edge strength
    int direction;                      ///< Dominant direction (0-3)
} EdgeDirection;

/**
 * @brief Gradient structure
 */
typedef struct {
    double dx;                          ///< X gradient
    double dy;                          ///< Y gradient
    double magnitude;                   ///< Gradient magnitude
    double angle;                       ///< Gradient angle
} Gradient;

/**
 * @brief Homogeneity map (for AHD algorithm)
 */
typedef struct {
    double *h_map;                      ///< Horizontal homogeneity
    double *v_map;                      ///< Vertical homogeneity
    int width;                          ///< Map width
    int height;                         ///< Map height
    bool owns_data;                     ///< Memory ownership
} HomogeneityMap;

/**
 * @brief Directional interpolation weights
 */
typedef struct {
    double north;                       ///< North weight
    double south;                       ///< South weight
    double east;                        ///< East weight
    double west;                        ///< West weight
    double northeast;                   ///< Northeast weight
    double northwest;                   ///< Northwest weight
    double southeast;                   ///< Southeast weight
    double southwest;                   ///< Southwest weight
} DirectionalWeights;

/**
 * @brief VNG (Variable Number of Gradients) data
 */
typedef struct {
    Gradient *gradients;                ///< Gradient array
    int num_gradients;                  ///< Number of gradients
    double threshold;                   ///< Gradient threshold
} VNGData;

/**
 * @brief PPG (Pixel Grouping) data
 */
typedef struct {
    int *groups;                        ///< Pixel group assignments
    int num_groups;                     ///< Number of groups
    double *group_means;                ///< Mean values per group
} PPGData;

// ============================================================================
// Filter Structures
// ============================================================================

/**
 * @brief Convolution kernel
 */
typedef struct {
    double *data;                       ///< Kernel data
    int width;                          ///< Kernel width
    int height;                         ///< Kernel height
    int center_x;                       ///< Center X coordinate
    int center_y;                       ///< Center Y coordinate
    double scale;                       ///< Scale factor
    bool normalized;                    ///< Whether kernel is normalized
} ConvolutionKernel;

/**
 * @brief Bilateral filter parameters
 */
typedef struct {
    double sigma_spatial;               ///< Spatial sigma
    double sigma_range;                 ///< Range sigma
    int window_size;                    ///< Window size
} BilateralFilterParams;

/**
 * @brief Non-local means filter parameters
 */
typedef struct {
    double h;                           ///< Filtering parameter
    int template_window_size;           ///< Template window size
    int search_window_size;             ///< Search window size
} NLMFilterParams;

// ============================================================================
// GPU/Hardware Structures
// ============================================================================

/**
 * @brief CUDA device information
 */
typedef struct {
    int device_id;                      ///< Device ID
    char device_name[256];              ///< Device name
    size_t total_memory;                ///< Total memory (bytes)
    size_t free_memory;                 ///< Free memory (bytes)
    int compute_capability_major;       ///< Compute capability (major)
    int compute_capability_minor;       ///< Compute capability (minor)
    int multiprocessor_count;           ///< Number of SMs
    bool available;                     ///< Device available
} CUDADeviceInfo;

/**
 * @brief OpenCL device information
 */
typedef struct {
    int platform_id;                    ///< Platform ID
    int device_id;                      ///< Device ID
    char platform_name[256];            ///< Platform name
    char device_name[256];              ///< Device name
    size_t global_memory;               ///< Global memory (bytes)
    size_t local_memory;                ///< Local memory (bytes)
    int compute_units;                  ///< Number of compute units
    bool available;                     ///< Device available
} OpenCLDeviceInfo;

/**
 * @brief Hardware capabilities
 */
typedef struct {
    // CPU features
    bool has_sse2;                      ///< SSE2 support
    bool has_avx;                       ///< AVX support
    bool has_avx2;                      ///< AVX2 support
    bool has_neon;                      ///< NEON support
    int num_cpu_cores;                  ///< Number of CPU cores
    
    // GPU features
    bool has_cuda;                      ///< CUDA available
    bool has_opencl;                    ///< OpenCL available
    int num_cuda_devices;               ///< Number of CUDA devices
    int num_opencl_devices;             ///< Number of OpenCL devices
    
    // Memory
    size_t total_system_memory;         ///< Total system memory (bytes)
    size_t available_system_memory;     ///< Available memory (bytes)
} HardwareCapabilities;

// ============================================================================
// Callback Structures
// ============================================================================

/**
 * @brief Progress callback function type
 * @param progress Progress value [0, 1]
 * @param message Progress message
 * @param user_data User data pointer
 * @return true to continue, false to cancel
 */
typedef bool (*ProgressCallback)(double progress, const char *message, void *user_data);

/**
 * @brief Log callback function type
 * @param level Log level (0=debug, 1=info, 2=warning, 3=error)
 * @param message Log message
 * @param user_data User data pointer
 */
typedef void (*LogCallback)(int level, const char *message, void *user_data);

/**
 * @brief Callback configuration
 */
typedef struct {
    ProgressCallback progress_callback; ///< Progress callback
    LogCallback log_callback;           ///< Log callback
    void *progress_user_data;           ///< Progress callback user data
    void *log_user_data;                ///< Log callback user data
} CallbackConfig;

// ============================================================================
// End of Part 2
// ============================================================================
// ============================================================================
// Part 3: Core API Functions
// ============================================================================

// ============================================================================
// Library Initialization and Cleanup
// ============================================================================

/**
 * @brief Initialize the demosaicing library
 * @return Error code
 */
DEMOSAIC_API DemosaicError demosaic_init(void);

/**
 * @brief Cleanup and release library resources
 */
DEMOSAIC_API void demosaic_cleanup(void);

/**
 * @brief Get library version string
 * @return Version string (e.g., "2.0.0")
 */
DEMOSAIC_API const char* demosaic_get_version(void);

/**
 * @brief Get library build information
 * @param info Buffer to store build info
 * @param size Buffer size
 * @return Error code
 */
DEMOSAIC_API DemosaicError demosaic_get_build_info(char *info, size_t size);

/**
 * @brief Check if library is initialized
 * @return true if initialized, false otherwise
 */
DEMOSAIC_API bool demosaic_is_initialized(void);

// ============================================================================
// Error Handling
// ============================================================================

/**
 * @brief Get error message for error code
 * @param error Error code
 * @return Error message string
 */
DEMOSAIC_API const char* demosaic_get_error_string(DemosaicError error);

/**
 * @brief Get last error code
 * @return Last error code
 */
DEMOSAIC_API DemosaicError demosaic_get_last_error(void);

/**
 * @brief Clear last error
 */
DEMOSAIC_API void demosaic_clear_error(void);

/**
 * @brief Set custom error handler
 * @param handler Error handler callback
 * @param user_data User data pointer
 */
DEMOSAIC_API void demosaic_set_error_handler(
    void (*handler)(DemosaicError error, const char *message, void *user_data),
    void *user_data
);

// ============================================================================
// Image Creation and Destruction
// ============================================================================

/**
 * @brief Create a new Bayer image
 * @param width Image width
 * @param height Image height
 * @param pattern Bayer pattern
 * @param bit_depth Bit depth (8, 10, 12, 14, 16)
 * @return Pointer to BayerImage or NULL on error
 */
DEMOSAIC_API BayerImage* demosaic_bayer_create(
    int width,
    int height,
    BayerPattern pattern,
    int bit_depth
);

/**
 * @brief Create Bayer image from existing data
 * @param data Raw Bayer data
 * @param width Image width
 * @param height Image height
 * @param pattern Bayer pattern
 * @param bit_depth Bit depth
 * @param copy_data Whether to copy data (true) or use pointer (false)
 * @return Pointer to BayerImage or NULL on error
 */
DEMOSAIC_API BayerImage* demosaic_bayer_create_from_data(
    const uint16_t *data,
    int width,
    int height,
    BayerPattern pattern,
    int bit_depth,
    bool copy_data
);

/**
 * @brief Destroy Bayer image and free resources
 * @param image Pointer to BayerImage
 */
DEMOSAIC_API void demosaic_bayer_destroy(BayerImage *image);

/**
 * @brief Clone a Bayer image
 * @param image Source image
 * @return Cloned image or NULL on error
 */
DEMOSAIC_API BayerImage* demosaic_bayer_clone(const BayerImage *image);

/**
 * @brief Create a new RGB image
 * @param width Image width
 * @param height Image height
 * @param channels Number of channels (typically 3)
 * @return Pointer to RGBImage or NULL on error
 */
DEMOSAIC_API RGBImage* demosaic_rgb_create(
    int width,
    int height,
    int channels
);

/**
 * @brief Create RGB image from existing data
 * @param data RGB data
 * @param width Image width
 * @param height Image height
 * @param channels Number of channels
 * @param copy_data Whether to copy data
 * @return Pointer to RGBImage or NULL on error
 */
DEMOSAIC_API RGBImage* demosaic_rgb_create_from_data(
    const double *data,
    int width,
    int height,
    int channels,
    bool copy_data
);

/**
 * @brief Destroy RGB image and free resources
 * @param image Pointer to RGBImage
 */
DEMOSAIC_API void demosaic_rgb_destroy(RGBImage *image);

/**
 * @brief Clone an RGB image
 * @param image Source image
 * @return Cloned image or NULL on error
 */
DEMOSAIC_API RGBImage* demosaic_rgb_clone(const RGBImage *image);

// ============================================================================
// Configuration Management
// ============================================================================

/**
 * @brief Create default configuration
 * @return Pointer to DemosaicConfig or NULL on error
 */
DEMOSAIC_API DemosaicConfig* demosaic_config_create_default(void);

/**
 * @brief Create configuration with quality preset
 * @param quality Quality level
 * @return Pointer to DemosaicConfig or NULL on error
 */
DEMOSAIC_API DemosaicConfig* demosaic_config_create_preset(QualityLevel quality);

/**
 * @brief Destroy configuration
 * @param config Pointer to DemosaicConfig
 */
DEMOSAIC_API void demosaic_config_destroy(DemosaicConfig *config);

/**
 * @brief Clone configuration
 * @param config Source configuration
 * @return Cloned configuration or NULL on error
 */
DEMOSAIC_API DemosaicConfig* demosaic_config_clone(const DemosaicConfig *config);

/**
 * @brief Validate configuration
 * @param config Configuration to validate
 * @return Error code
 */
DEMOSAIC_API DemosaicError demosaic_config_validate(const DemosaicConfig *config);

/**
 * @brief Load configuration from file
 * @param filename Configuration file path
 * @return Pointer to DemosaicConfig or NULL on error
 */
DEMOSAIC_API DemosaicConfig* demosaic_config_load(const char *filename);

/**
 * @brief Save configuration to file
 * @param config Configuration to save
 * @param filename Output file path
 * @return Error code
 */
DEMOSAIC_API DemosaicError demosaic_config_save(
    const DemosaicConfig *config,
    const char *filename
);

// ============================================================================
// Core Demosaicing Functions
// ============================================================================

/**
 * @brief Perform demosaicing with default configuration
 * @param bayer_image Input Bayer image
 * @param rgb_image Output RGB image (will be created if NULL)
 * @return Error code
 */
DEMOSAIC_API DemosaicError demosaic_process(
    const BayerImage *bayer_image,
    RGBImage **rgb_image
);

/**
 * @brief Perform demosaicing with custom configuration
 * @param bayer_image Input Bayer image
 * @param rgb_image Output RGB image (will be created if NULL)
 * @param config Demosaicing configuration
 * @return Error code
 */
DEMOSAIC_API DemosaicError demosaic_process_with_config(
    const BayerImage *bayer_image,
    RGBImage **rgb_image,
    const DemosaicConfig *config
);

/**
 * @brief Perform demosaicing with statistics collection
 * @param bayer_image Input Bayer image
 * @param rgb_image Output RGB image
 * @param config Configuration
 * @param stats Statistics output (can be NULL)
 * @return Error code
 */
DEMOSAIC_API DemosaicError demosaic_process_with_stats(
    const BayerImage *bayer_image,
    RGBImage **rgb_image,
    const DemosaicConfig *config,
    DemosaicStats *stats
);

/**
 * @brief Perform demosaicing with progress callback
 * @param bayer_image Input Bayer image
 * @param rgb_image Output RGB image
 * @param config Configuration
 * @param callback_config Callback configuration
 * @return Error code
 */
DEMOSAIC_API DemosaicError demosaic_process_with_callback(
    const BayerImage *bayer_image,
    RGBImage **rgb_image,
    const DemosaicConfig *config,
    const CallbackConfig *callback_config
);

// ============================================================================
// Specific Algorithm Functions
// ============================================================================

/**
 * @brief Nearest neighbor demosaicing
 * @param bayer_image Input Bayer image
 * @param rgb_image Output RGB image
 * @return Error code
 */
DEMOSAIC_API DemosaicError demosaic_nearest(
    const BayerImage *bayer_image,
    RGBImage **rgb_image
);

/**
 * @brief Bilinear interpolation demosaicing
 * @param bayer_image Input Bayer image
 * @param rgb_image Output RGB image
 * @return Error code
 */
DEMOSAIC_API DemosaicError demosaic_bilinear(
    const BayerImage *bayer_image,
    RGBImage **rgb_image
);

/**
 * @brief Edge-aware demosaicing
 * @param bayer_image Input Bayer image
 * @param rgb_image Output RGB image
 * @param edge_threshold Edge detection threshold
 * @return Error code
 */
DEMOSAIC_API DemosaicError demosaic_edge_aware(
    const BayerImage *bayer_image,
    RGBImage **rgb_image,
    double edge_threshold
);

/**
 * @brief Malvar-He-Cutler demosaicing
 * @param bayer_image Input Bayer image
 * @param rgb_image Output RGB image
 * @return Error code
 */
DEMOSAIC_API DemosaicError demosaic_malvar(
    const BayerImage *bayer_image,
    RGBImage **rgb_image
);

/**
 * @brief Adaptive Homogeneity-Directed (AHD) demosaicing
 * @param bayer_image Input Bayer image
 * @param rgb_image Output RGB image
 * @param iterations Number of iterations
 * @return Error code
 */
DEMOSAIC_API DemosaicError demosaic_ahd(
    const BayerImage *bayer_image,
    RGBImage **rgb_image,
    int iterations
);

/**
 * @brief Variable Number of Gradients (VNG) demosaicing
 * @param bayer_image Input Bayer image
 * @param rgb_image Output RGB image
 * @param threshold Gradient threshold
 * @return Error code
 */
DEMOSAIC_API DemosaicError demosaic_vng(
    const BayerImage *bayer_image,
    RGBImage **rgb_image,
    double threshold
);

/**
 * @brief Pixel Grouping (PPG) demosaicing
 * @param bayer_image Input Bayer image
 * @param rgb_image Output RGB image
 * @return Error code
 */
DEMOSAIC_API DemosaicError demosaic_ppg(
    const BayerImage *bayer_image,
    RGBImage **rgb_image
);

/**
 * @brief Linear Minimum Mean Square Error (LMMSE) demosaicing
 * @param bayer_image Input Bayer image
 * @param rgb_image Output RGB image
 * @param window_size Processing window size
 * @return Error code
 */
DEMOSAIC_API DemosaicError demosaic_lmmse(
    const BayerImage *bayer_image,
    RGBImage **rgb_image,
    int window_size
);

// ============================================================================
// Bayer Pattern Functions
// ============================================================================

/**
 * @brief Detect Bayer pattern from image
 * @param image Bayer image
 * @return Detected pattern or BAYER_UNKNOWN
 */
DEMOSAIC_API BayerPattern demosaic_detect_pattern(const BayerImage *image);

/**
 * @brief Get Bayer pattern name
 * @param pattern Bayer pattern
 * @return Pattern name string
 */
DEMOSAIC_API const char* demosaic_pattern_to_string(BayerPattern pattern);

/**
 * @brief Parse Bayer pattern from string
 * @param pattern_str Pattern string (e.g., "RGGB")
 * @return Bayer pattern or BAYER_UNKNOWN
 */
DEMOSAIC_API BayerPattern demosaic_pattern_from_string(const char *pattern_str);

/**
 * @brief Get color at Bayer position
 * @param pattern Bayer pattern
 * @param x X coordinate
 * @param y Y coordinate
 * @return Color channel (0=R, 1=G, 2=B)
 */
DEMOSAIC_API int demosaic_get_bayer_color(BayerPattern pattern, int x, int y);

/**
 * @brief Check if position is green in Bayer pattern
 * @param pattern Bayer pattern
 * @param x X coordinate
 * @param y Y coordinate
 * @return true if green, false otherwise
 */
DEMOSAIC_API bool demosaic_is_green(BayerPattern pattern, int x, int y);

// ============================================================================
// Color Processing Functions
// ============================================================================

/**
 * @brief Apply white balance to RGB image
 * @param image RGB image (modified in-place)
 * @param wb White balance coefficients
 * @return Error code
 */
DEMOSAIC_API DemosaicError demosaic_apply_white_balance(
    RGBImage *image,
    const WhiteBalance *wb
);

/**
 * @brief Auto white balance
 * @param image RGB image
 * @param wb Output white balance coefficients
 * @return Error code
 */
DEMOSAIC_API DemosaicError demosaic_auto_white_balance(
    const RGBImage *image,
    WhiteBalance *wb
);

/**
 * @brief Apply color correction matrix
 * @param image RGB image (modified in-place)
 * @param matrix Color correction matrix
 * @return Error code
 */
DEMOSAIC_API DemosaicError demosaic_apply_color_matrix(
    RGBImage *image,
    const ColorMatrix *matrix
);

/**
 * @brief Convert color space
 * @param image RGB image (modified in-place)
 * @param target_space Target color space
 * @return Error code
 */
DEMOSAIC_API DemosaicError demosaic_convert_color_space(
    RGBImage *image,
    ColorSpace target_space
);

/**
 * @brief Apply gamma correction
 * @param image RGB image (modified in-place)
 * @param gamma Gamma value (typically 2.2 for sRGB)
 * @return Error code
 */
DEMOSAIC_API DemosaicError demosaic_apply_gamma(
    RGBImage *image,
    double gamma
);

/**
 * @brief Apply tone curve
 * @param image RGB image (modified in-place)
 * @param curve Tone curve
 * @return Error code
 */
DEMOSAIC_API DemosaicError demosaic_apply_tone_curve(
    RGBImage *image,
    const ToneCurve *curve
);

/**
 * @brief Apply color profile
 * @param image RGB image (modified in-place)
 * @param profile Color profile
 * @return Error code
 */
DEMOSAIC_API DemosaicError demosaic_apply_color_profile(
    RGBImage *image,
    const ColorProfile *profile
);

// ============================================================================
// Image Enhancement Functions
// ============================================================================

/**
 * @brief Apply denoising
 * @param image RGB image (modified in-place)
 * @param strength Denoising strength [0, 1]
 * @return Error code
 */
DEMOSAIC_API DemosaicError demosaic_denoise(
    RGBImage *image,
    double strength
);

/**
 * @brief Apply sharpening
 * @param image RGB image (modified in-place)
 * @param amount Sharpening amount [0, 1]
 * @param radius Sharpening radius
 * @return Error code
 */
DEMOSAIC_API DemosaicError demosaic_sharpen(
    RGBImage *image,
    double amount,
    double radius
);

/**
 * @brief Suppress false color artifacts
 * @param image RGB image (modified in-place)
 * @param strength Suppression strength [0, 1]
 * @return Error code
 */
DEMOSAIC_API DemosaicError demosaic_suppress_false_color(
    RGBImage *image,
    double strength
);

/**
 * @brief Suppress zipper artifacts
 * @param image RGB image (modified in-place)
 * @param strength Suppression strength [0, 1]
 * @return Error code
 */
DEMOSAIC_API DemosaicError demosaic_suppress_zipper(
    RGBImage *image,
    double strength
);

/**
 * @brief Suppress moiré artifacts
 * @param image RGB image (modified in-place)
 * @param strength Suppression strength [0, 1]
 * @return Error code
 */
DEMOSAIC_API DemosaicError demosaic_suppress_moire(
    RGBImage *image,
    double strength
);

// ============================================================================
// Quality Assessment Functions
// ============================================================================

/**
 * @brief Calculate PSNR between two images
 * @param image1 First RGB image
 * @param image2 Second RGB image
 * @param psnr Output PSNR value (dB)
 * @return Error code
 */
DEMOSAIC_API DemosaicError demosaic_calculate_psnr(
    const RGBImage *image1,
    const RGBImage *image2,
    double *psnr
);

/**
 * @brief Calculate SSIM between two images
 * @param image1 First RGB image
 * @param image2 Second RGB image
 * @param ssim Output SSIM value [0, 1]
 * @return Error code
 */
DEMOSAIC_API DemosaicError demosaic_calculate_ssim(
    const RGBImage *image1,
    const RGBImage *image2,
    double *ssim
);

/**
 * @brief Calculate MSE between two images
 * @param image1 First RGB image
 * @param image2 Second RGB image
 * @param mse Output MSE value
 * @return Error code
 */
DEMOSAIC_API DemosaicError demosaic_calculate_mse(
    const RGBImage *image1,
    const RGBImage *image2,
    double *mse
);

/**
 * @brief Assess demosaicing quality
 * @param image RGB image
 * @param stats Output quality statistics
 * @return Error code
 */
DEMOSAIC_API DemosaicError demosaic_assess_quality(
    const RGBImage *image,
    DemosaicStats *stats
);

/**
 * @brief Detect zipper artifacts
 * @param image RGB image
 * @param metric Output zipper metric
 * @return Error code
 */
DEMOSAIC_API DemosaicError demosaic_detect_zipper(
    const RGBImage *image,
    double *metric
);

/**
 * @brief Detect false color artifacts
 * @param image RGB image
 * @param metric Output false color metric
 * @return Error code
 */
DEMOSAIC_API DemosaicError demosaic_detect_false_color(
    const RGBImage *image,
    double *metric
);

// ============================================================================
// File I/O Functions
// ============================================================================

/**
 * @brief Load Bayer image from RAW file
 * @param filename Input file path
 * @param image Output Bayer image
 * @return Error code
 */
DEMOSAIC_API DemosaicError demosaic_load_raw(
    const char *filename,
    BayerImage **image
);

/**
 * @brief Save Bayer image to RAW file
 * @param image Bayer image
 * @param filename Output file path
 * @return Error code
 */
DEMOSAIC_API DemosaicError demosaic_save_raw(
    const BayerImage *image,
    const char *filename
);

/**
 * @brief Load RGB image from file (PNG, JPEG, TIFF, etc.)
 * @param filename Input file path
 * @param image Output RGB image
 * @return Error code
 */
DEMOSAIC_API DemosaicError demosaic_load_image(
    const char *filename,
    RGBImage **image
);

/**
 * @brief Save RGB image to file
 * @param image RGB image
 * @param filename Output file path
 * @param quality Quality for lossy formats (0-100)
 * @return Error code
 */
DEMOSAIC_API DemosaicError demosaic_save_image(
    const RGBImage *image,
    const char *filename,
    int quality
);

/**
 * @brief Load image metadata
 * @param filename Input file path
 * @param metadata Output metadata
 * @return Error code
 */
DEMOSAIC_API DemosaicError demosaic_load_metadata(
    const char *filename,
    ImageMetadata **metadata
);

// ============================================================================
// End of Part 3
// ============================================================================
// ============================================================================
// Part 4: Utility Functions, Hardware Acceleration, and Inline Functions
// ============================================================================

// ============================================================================
// Image Utility Functions
// ============================================================================

/**
 * @brief Get pixel value from Bayer image
 * @param image Bayer image
 * @param x X coordinate
 * @param y Y coordinate
 * @return Pixel value
 */
DEMOSAIC_API uint16_t demosaic_bayer_get_pixel(
    const BayerImage *image,
    int x,
    int y
);

/**
 * @brief Set pixel value in Bayer image
 * @param image Bayer image
 * @param x X coordinate
 * @param y Y coordinate
 * @param value Pixel value
 */
DEMOSAIC_API void demosaic_bayer_set_pixel(
    BayerImage *image,
    int x,
    int y,
    uint16_t value
);

/**
 * @brief Get RGB pixel from RGB image
 * @param image RGB image
 * @param x X coordinate
 * @param y Y coordinate
 * @param color Output color
 * @return Error code
 */
DEMOSAIC_API DemosaicError demosaic_rgb_get_pixel(
    const RGBImage *image,
    int x,
    int y,
    Color *color
);

/**
 * @brief Set RGB pixel in RGB image
 * @param image RGB image
 * @param x X coordinate
 * @param y Y coordinate
 * @param color Input color
 * @return Error code
 */
DEMOSAIC_API DemosaicError demosaic_rgb_set_pixel(
    RGBImage *image,
    int x,
    int y,
    const Color *color
);

/**
 * @brief Convert RGB image to planar format
 * @param image RGB image (modified in-place)
 * @return Error code
 */
DEMOSAIC_API DemosaicError demosaic_rgb_to_planar(RGBImage *image);

/**
 * @brief Convert RGB image to interleaved format
 * @param image RGB image (modified in-place)
 * @return Error code
 */
DEMOSAIC_API DemosaicError demosaic_rgb_to_interleaved(RGBImage *image);

/**
 * @brief Crop Bayer image
 * @param image Source Bayer image
 * @param rect Crop rectangle
 * @param output Output cropped image
 * @return Error code
 */
DEMOSAIC_API DemosaicError demosaic_bayer_crop(
    const BayerImage *image,
    const Rectangle *rect,
    BayerImage **output
);

/**
 * @brief Crop RGB image
 * @param image Source RGB image
 * @param rect Crop rectangle
 * @param output Output cropped image
 * @return Error code
 */
DEMOSAIC_API DemosaicError demosaic_rgb_crop(
    const RGBImage *image,
    const Rectangle *rect,
    RGBImage **output
);

/**
 * @brief Resize RGB image
 * @param image Source RGB image
 * @param new_width New width
 * @param new_height New height
 * @param method Interpolation method
 * @param output Output resized image
 * @return Error code
 */
DEMOSAIC_API DemosaicError demosaic_rgb_resize(
    const RGBImage *image,
    int new_width,
    int new_height,
    InterpolationMethod method,
    RGBImage **output
);

/**
 * @brief Rotate RGB image
 * @param image Source RGB image
 * @param angle Rotation angle (degrees, clockwise)
 * @param output Output rotated image
 * @return Error code
 */
DEMOSAIC_API DemosaicError demosaic_rgb_rotate(
    const RGBImage *image,
    double angle,
    RGBImage **output
);

/**
 * @brief Flip RGB image
 * @param image RGB image (modified in-place)
 * @param horizontal Flip horizontally
 * @param vertical Flip vertically
 * @return Error code
 */
DEMOSAIC_API DemosaicError demosaic_rgb_flip(
    RGBImage *image,
    bool horizontal,
    bool vertical
);

// ============================================================================
// Statistics Functions
// ============================================================================

/**
 * @brief Calculate image statistics
 * @param image RGB image
 * @param stats Output statistics
 * @return Error code
 */
DEMOSAIC_API DemosaicError demosaic_calculate_statistics(
    const RGBImage *image,
    DemosaicStats *stats
);

/**
 * @brief Calculate histogram
 * @param image RGB image
 * @param channel Channel index (0=R, 1=G, 2=B, -1=luminance)
 * @param bins Number of histogram bins
 * @param histogram Output histogram array (must be pre-allocated)
 * @return Error code
 */
DEMOSAIC_API DemosaicError demosaic_calculate_histogram(
    const RGBImage *image,
    int channel,
    int bins,
    int *histogram
);

/**
 * @brief Calculate cumulative histogram
 * @param histogram Input histogram
 * @param bins Number of bins
 * @param cumulative Output cumulative histogram
 * @return Error code
 */
DEMOSAIC_API DemosaicError demosaic_calculate_cumulative_histogram(
    const int *histogram,
    int bins,
    int *cumulative
);

/**
 * @brief Equalize histogram
 * @param image RGB image (modified in-place)
 * @return Error code
 */
DEMOSAIC_API DemosaicError demosaic_equalize_histogram(RGBImage *image);

/**
 * @brief Calculate image entropy
 * @param image RGB image
 * @param entropy Output entropy value
 * @return Error code
 */
DEMOSAIC_API DemosaicError demosaic_calculate_entropy(
    const RGBImage *image,
    double *entropy
);

// ============================================================================
// Edge Detection Functions
// ============================================================================

/**
 * @brief Detect edges in image
 * @param image RGB image
 * @param method Edge detection method
 * @param threshold Edge threshold
 * @param edges Output edge map (grayscale)
 * @return Error code
 */
DEMOSAIC_API DemosaicError demosaic_detect_edges(
    const RGBImage *image,
    EdgeDetectionMethod method,
    double threshold,
    RGBImage **edges
);

/**
 * @brief Calculate gradient map
 * @param image RGB image
 * @param gradients Output gradient array (must be pre-allocated)
 * @return Error code
 */
DEMOSAIC_API DemosaicError demosaic_calculate_gradients(
    const RGBImage *image,
    Gradient *gradients
);

/**
 * @brief Calculate edge direction map
 * @param image RGB image
 * @param directions Output direction array (must be pre-allocated)
 * @return Error code
 */
DEMOSAIC_API DemosaicError demosaic_calculate_edge_directions(
    const RGBImage *image,
    EdgeDirection *directions
);

// ============================================================================
// Filter Functions
// ============================================================================

/**
 * @brief Create Gaussian kernel
 * @param size Kernel size (must be odd)
 * @param sigma Standard deviation
 * @return Convolution kernel or NULL on error
 */
DEMOSAIC_API ConvolutionKernel* demosaic_create_gaussian_kernel(
    int size,
    double sigma
);

/**
 * @brief Create Laplacian kernel
 * @param size Kernel size (3 or 5)
 * @return Convolution kernel or NULL on error
 */
DEMOSAIC_API ConvolutionKernel* demosaic_create_laplacian_kernel(int size);

/**
 * @brief Create Sobel kernel
 * @param horizontal true for horizontal, false for vertical
 * @return Convolution kernel or NULL on error
 */
DEMOSAIC_API ConvolutionKernel* demosaic_create_sobel_kernel(bool horizontal);

/**
 * @brief Destroy convolution kernel
 * @param kernel Convolution kernel
 */
DEMOSAIC_API void demosaic_destroy_kernel(ConvolutionKernel *kernel);

/**
 * @brief Apply convolution filter
 * @param image RGB image (modified in-place)
 * @param kernel Convolution kernel
 * @return Error code
 */
DEMOSAIC_API DemosaicError demosaic_apply_convolution(
    RGBImage *image,
    const ConvolutionKernel *kernel
);

/**
 * @brief Apply Gaussian blur
 * @param image RGB image (modified in-place)
 * @param sigma Standard deviation
 * @return Error code
 */
DEMOSAIC_API DemosaicError demosaic_gaussian_blur(
    RGBImage *image,
    double sigma
);

/**
 * @brief Apply bilateral filter
 * @param image RGB image (modified in-place)
 * @param params Bilateral filter parameters
 * @return Error code
 */
DEMOSAIC_API DemosaicError demosaic_bilateral_filter(
    RGBImage *image,
    const BilateralFilterParams *params
);

/**
 * @brief Apply non-local means filter
 * @param image RGB image (modified in-place)
 * @param params NLM filter parameters
 * @return Error code
 */
DEMOSAIC_API DemosaicError demosaic_nlm_filter(
    RGBImage *image,
    const NLMFilterParams *params
);

/**
 * @brief Apply median filter
 * @param image RGB image (modified in-place)
 * @param window_size Window size (must be odd)
 * @return Error code
 */
DEMOSAIC_API DemosaicError demosaic_median_filter(
    RGBImage *image,
    int window_size
);

// ============================================================================
// Hardware Acceleration Functions
// ============================================================================

/**
 * @brief Query hardware capabilities
 * @param capabilities Output hardware capabilities
 * @return Error code
 */
DEMOSAIC_API DemosaicError demosaic_query_hardware(
    HardwareCapabilities *capabilities
);

/**
 * @brief Get number of available CUDA devices
 * @return Number of CUDA devices
 */
DEMOSAIC_API int demosaic_get_cuda_device_count(void);

/**
 * @brief Get CUDA device information
 * @param device_id Device ID
 * @param info Output device information
 * @return Error code
 */
DEMOSAIC_API DemosaicError demosaic_get_cuda_device_info(
    int device_id,
    CUDADeviceInfo *info
);

/**
 * @brief Set active CUDA device
 * @param device_id Device ID
 * @return Error code
 */
DEMOSAIC_API DemosaicError demosaic_set_cuda_device(int device_id);

/**
 * @brief Get number of available OpenCL devices
 * @return Number of OpenCL devices
 */
DEMOSAIC_API int demosaic_get_opencl_device_count(void);

/**
 * @brief Get OpenCL device information
 * @param platform_id Platform ID
 * @param device_id Device ID
 * @param info Output device information
 * @return Error code
 */
DEMOSAIC_API DemosaicError demosaic_get_opencl_device_info(
    int platform_id,
    int device_id,
    OpenCLDeviceInfo *info
);

/**
 * @brief Set active OpenCL device
 * @param platform_id Platform ID
 * @param device_id Device ID
 * @return Error code
 */
DEMOSAIC_API DemosaicError demosaic_set_opencl_device(
    int platform_id,
    int device_id
);

/**
 * @brief Enable/disable SIMD optimization
 * @param enable true to enable, false to disable
 */
DEMOSAIC_API void demosaic_set_simd_enabled(bool enable);

/**
 * @brief Check if SIMD is enabled
 * @return true if enabled, false otherwise
 */
DEMOSAIC_API bool demosaic_is_simd_enabled(void);

/**
 * @brief Set number of threads for parallel processing
 * @param num_threads Number of threads (0 = auto)
 */
DEMOSAIC_API void demosaic_set_num_threads(int num_threads);

/**
 * @brief Get number of threads being used
 * @return Number of threads
 */
DEMOSAIC_API int demosaic_get_num_threads(void);

// ============================================================================
// Memory Management Functions
// ============================================================================

/**
 * @brief Allocate aligned memory
 * @param size Size in bytes
 * @param alignment Alignment in bytes (must be power of 2)
 * @return Pointer to allocated memory or NULL on error
 */
DEMOSAIC_API void* demosaic_aligned_alloc(size_t size, size_t alignment);

/**
 * @brief Free aligned memory
 * @param ptr Pointer to memory
 */
DEMOSAIC_API void demosaic_aligned_free(void *ptr);

/**
 * @brief Get current memory usage
 * @return Memory usage in bytes
 */
DEMOSAIC_API size_t demosaic_get_memory_usage(void);

/**
 * @brief Get peak memory usage
 * @return Peak memory usage in bytes
 */
DEMOSAIC_API size_t demosaic_get_peak_memory_usage(void);

/**
 * @brief Reset memory statistics
 */
DEMOSAIC_API void demosaic_reset_memory_stats(void);

// ============================================================================
// Debugging and Logging Functions
// ============================================================================

/**
 * @brief Set log level
 * @param level Log level (0=debug, 1=info, 2=warning, 3=error)
 */
DEMOSAIC_API void demosaic_set_log_level(int level);

/**
 * @brief Get current log level
 * @return Current log level
 */
DEMOSAIC_API int demosaic_get_log_level(void);

/**
 * @brief Set log callback
 * @param callback Log callback function
 * @param user_data User data pointer
 */
DEMOSAIC_API void demosaic_set_log_callback(
    LogCallback callback,
    void *user_data
);

/**
 * @brief Enable/disable verbose output
 * @param enable true to enable, false to disable
 */
DEMOSAIC_API void demosaic_set_verbose(bool enable);

/**
 * @brief Check if verbose output is enabled
 * @return true if enabled, false otherwise
 */
DEMOSAIC_API bool demosaic_is_verbose(void);

/**
 * @brief Save debug image
 * @param image RGB image
 * @param name Debug image name
 * @return Error code
 */
DEMOSAIC_API DemosaicError demosaic_save_debug_image(
    const RGBImage *image,
    const char *name
);

/**
 * @brief Print configuration
 * @param config Configuration to print
 */
DEMOSAIC_API void demosaic_print_config(const DemosaicConfig *config);

/**
 * @brief Print statistics
 * @param stats Statistics to print
 */
DEMOSAIC_API void demosaic_print_stats(const DemosaicStats *stats);

/**
 * @brief Print image information
 * @param image RGB image
 */
DEMOSAIC_API void demosaic_print_image_info(const RGBImage *image);

// ============================================================================
// Benchmark Functions
// ============================================================================

/**
 * @brief Benchmark demosaicing method
 * @param bayer_image Input Bayer image
 * @param method Demosaicing method
 * @param iterations Number of iterations
 * @param avg_time Output average time (seconds)
 * @return Error code
 */
DEMOSAIC_API DemosaicError demosaic_benchmark(
    const BayerImage *bayer_image,
    DemosaicMethod method,
    int iterations,
    double *avg_time
);

/**
 * @brief Compare demosaicing methods
 * @param bayer_image Input Bayer image
 * @param reference Reference RGB image (ground truth)
 * @param methods Array of methods to compare
 * @param num_methods Number of methods
 * @param results Output comparison results
 * @return Error code
 */
DEMOSAIC_API DemosaicError demosaic_compare_methods(
    const BayerImage *bayer_image,
    const RGBImage *reference,
    const DemosaicMethod *methods,
    int num_methods,
    DemosaicStats *results
);

// ============================================================================
// Inline Utility Functions
// ============================================================================

/**
 * @brief Clamp value to range [min, max]
 */
static DEMOSAIC_INLINE double demosaic_clamp_double(
    double value,
    double min,
    double max
) {
    return value < min ? min : (value > max ? max : value);
}

/**
 * @brief Clamp integer value to range [min, max]
 */
static DEMOSAIC_INLINE int demosaic_clamp_int(int value, int min, int max) {
    return value < min ? min : (value > max ? max : value);
}

/**
 * @brief Linear interpolation
 */
static DEMOSAIC_INLINE double demosaic_lerp(double a, double b, double t) {
    return a + (b - a) * t;
}

/**
 * @brief Bilinear interpolation
 */
static DEMOSAIC_INLINE double demosaic_bilerp(
    double v00, double v10,
    double v01, double v11,
    double tx, double ty
) {
    double v0 = demosaic_lerp(v00, v10, tx);
    double v1 = demosaic_lerp(v01, v11, tx);
    return demosaic_lerp(v0, v1, ty);
}

/**
 * @brief Convert RGB to grayscale (luminance)
 */
static DEMOSAIC_INLINE double demosaic_rgb_to_gray(
    double r, double g, double b
) {
    return 0.299 * r + 0.587 * g + 0.114 * b;
}

/**
 * @brief Calculate squared difference
 */
static DEMOSAIC_INLINE double demosaic_squared_diff(double a, double b) {
    double diff = a - b;
    return diff * diff;
}

/**
 * @brief Calculate Euclidean distance
 */
static DEMOSAIC_INLINE double demosaic_euclidean_distance(
    double x1, double y1,
    double x2, double y2
) {
    double dx = x2 - x1;
    double dy = y2 - y1;
    return sqrt(dx * dx + dy * dy);
}

/**
 * @brief Check if coordinates are within image bounds
 */
static DEMOSAIC_INLINE bool demosaic_in_bounds(
    int x, int y,
    int width, int height
) {
    return x >= 0 && x < width && y >= 0 && y < height;
}

/**
 * @brief Calculate 1D array index from 2D coordinates
 */
static DEMOSAIC_INLINE int demosaic_index_2d(
    int x, int y,
    int width
) {
    return y * width + x;
}

/**
 * @brief Calculate 1D array index for RGB data
 */
static DEMOSAIC_INLINE int demosaic_index_rgb(
    int x, int y,
    int channel,
    int width,
    int channels
) {
    return (y * width + x) * channels + channel;
}

/**
 * @brief Mirror coordinate at boundary
 */
static DEMOSAIC_INLINE int demosaic_mirror_coord(int coord, int size) {
    if (coord < 0) {
        return -coord - 1;
    } else if (coord >= size) {
        return 2 * size - coord - 1;
    }
    return coord;
}

/**
 * @brief Wrap coordinate at boundary
 */
static DEMOSAIC_INLINE int demosaic_wrap_coord(int coord, int size) {
    if (coord < 0) {
        return coord + size;
    } else if (coord >= size) {
        return coord - size;
    }
    return coord;
}

/**
 * @brief Convert degrees to radians
 */
static DEMOSAIC_INLINE double demosaic_deg_to_rad(double degrees) {
    return degrees * DEMOSAIC_PI / 180.0;
}

/**
 * @brief Convert radians to degrees
 */
static DEMOSAIC_INLINE double demosaic_rad_to_deg(double radians) {
    return radians * 180.0 / DEMOSAIC_PI;
}

/**
 * @brief Fast power of 2 check
 */
static DEMOSAIC_INLINE bool demosaic_is_power_of_2(int n) {
    return n > 0 && (n & (n - 1)) == 0;
}

/**
 * @brief Round to nearest integer
 */
static DEMOSAIC_INLINE int demosaic_round(double value) {
    return (int)(value + 0.5);
}

/**
 * @brief Calculate maximum of three values
 */
static DEMOSAIC_INLINE double demosaic_max3(double a, double b, double c) {
    return DEMOSAIC_MAX(DEMOSAIC_MAX(a, b), c);
}

/**
 * @brief Calculate minimum of three values
 */
static DEMOSAIC_INLINE double demosaic_min3(double a, double b, double c) {
    return DEMOSAIC_MIN(DEMOSAIC_MIN(a, b), c);
}

/**
 * @brief Swap two integers
 */
static DEMOSAIC_INLINE void demosaic_swap_int(int *a, int *b) {
    int temp = *a;
    *a = *b;
    *b = temp;
}

/**
 * @brief Swap two doubles
 */
static DEMOSAIC_INLINE void demosaic_swap_double(double *a, double *b) {
    double temp = *a;
    *a = *b;
    *b = temp;
}

// ============================================================================
// Color Space Conversion Inline Functions
// ============================================================================

/**
 * @brief Convert linear RGB to sRGB
 */
static DEMOSAIC_INLINE double demosaic_linear_to_srgb(double linear) {
    if (linear <= 0.0031308) {
        return 12.92 * linear;
    } else {
        return 1.055 * pow(linear, 1.0 / 2.4) - 0.055;
    }
}

/**
 * @brief Convert sRGB to linear RGB
 */
static DEMOSAIC_INLINE double demosaic_srgb_to_linear(double srgb) {
    if (srgb <= 0.04045) {
        return srgb / 12.92;
    } else {
        return pow((srgb + 0.055) / 1.055, 2.4);
    }
}

/**
 * @brief Apply gamma correction
 */
static DEMOSAIC_INLINE double demosaic_apply_gamma_inline(
    double value,
    double gamma
) {
    return pow(value, 1.0 / gamma);
}

/**
 * @brief Remove gamma correction
 */
static DEMOSAIC_INLINE double demosaic_remove_gamma_inline(
    double value,
    double gamma
) {
    return pow(value, gamma);
}

// ============================================================================
// Version and Feature Macros
// ============================================================================

/**
 * @brief Check if feature is available
 */
#define DEMOSAIC_HAS_FEATURE(feature) (defined(DEMOSAIC_FEATURE_##feature))

/**
 * @brief Feature flags
 */
#ifdef DEMOSAIC_ENABLE_CUDA
    #define DEMOSAIC_FEATURE_CUDA
#endif

#ifdef DEMOSAIC_ENABLE_OPENCL
    #define DEMOSAIC_FEATURE_OPENCL
#endif

#ifdef DEMOSAIC_ENABLE_OPENMP
    #define DEMOSAIC_FEATURE_OPENMP
#endif

#ifdef DEMOSAIC_ENABLE_SIMD
    #define DEMOSAIC_FEATURE_SIMD
#endif

// ============================================================================
// Closing
// ============================================================================

#ifdef __cplusplus
}
#endif

#endif // DEMOSAIC_H

// ============================================================================
// End of demosaic.h
// ============================================================================

/**
 * @example example_basic.c
 * Basic demosaicing example
 * 
 * @example example_advanced.c
 * Advanced demosaicing with custom configuration
 * 
 * @example example_batch.c
 * Batch processing example
 * 
 * @example example_cuda.c
 * CUDA acceleration example
 * 
 * @example example_quality.c
 * Quality assessment example
 */


