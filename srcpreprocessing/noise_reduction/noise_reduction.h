#ifndef NOISE_REDUCTION_H
#define NOISE_REDUCTION_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Version Information
// ============================================================================

#define NOISE_REDUCTION_VERSION_MAJOR 1
#define NOISE_REDUCTION_VERSION_MINOR 0
#define NOISE_REDUCTION_VERSION_PATCH 0

// ============================================================================
// Error Codes
// ============================================================================

typedef enum {
    NOISE_SUCCESS = 0,
    NOISE_ERROR_INVALID_PARAM = -1,
    NOISE_ERROR_OUT_OF_MEMORY = -2,
    NOISE_ERROR_INVALID_IMAGE = -3,
    NOISE_ERROR_UNSUPPORTED_FORMAT = -4,
    NOISE_ERROR_PROCESSING_FAILED = -5,
    NOISE_ERROR_FILE_IO = -6,
    NOISE_ERROR_DIMENSION_MISMATCH = -7,
    NOISE_ERROR_NOT_IMPLEMENTED = -8
} NoiseError;

// ============================================================================
// Image Structure
// ============================================================================

typedef enum {
    IMAGE_FORMAT_GRAYSCALE,
    IMAGE_FORMAT_RGB,
    IMAGE_FORMAT_RGBA,
    IMAGE_FORMAT_YUV,
    IMAGE_FORMAT_HSV,
    IMAGE_FORMAT_LAB
} ImageFormat;

typedef enum {
    DATA_TYPE_UINT8,
    DATA_TYPE_UINT16,
    DATA_TYPE_FLOAT32,
    DATA_TYPE_FLOAT64
} DataType;

typedef struct {
    void *data;           // Image data buffer
    int width;            // Image width in pixels
    int height;           // Image height in pixels
    int channels;         // Number of channels
    ImageFormat format;   // Image format
    DataType data_type;   // Data type
    int stride;           // Bytes per row
    size_t data_size;     // Total data size in bytes
} Image;

// ============================================================================
// Noise Types
// ============================================================================

typedef enum {
    NOISE_TYPE_GAUSSIAN,
    NOISE_TYPE_SALT_PEPPER,
    NOISE_TYPE_POISSON,
    NOISE_TYPE_SPECKLE,
    NOISE_TYPE_UNIFORM,
    NOISE_TYPE_IMPULSE,
    NOISE_TYPE_PERIODIC,
    NOISE_TYPE_UNKNOWN
} NoiseType;

// ============================================================================
// Basic Filter Parameters
// ============================================================================

typedef struct {
    int kernel_size;      // Filter kernel size (e.g., 3, 5, 7)
    double sigma;         // Standard deviation for Gaussian
    double strength;      // Filter strength (0.0 - 1.0)
    int iterations;       // Number of iterations
    bool preserve_edges;  // Edge preservation flag
} FilterParams;

// ============================================================================
// Image Management Functions
// ============================================================================

/**
 * Create a new image
 * @param image Pointer to image pointer
 * @param width Image width
 * @param height Image height
 * @param channels Number of channels
 * @param format Image format
 * @return Error code
 */
NoiseError image_create(
    Image **image,
    int width,
    int height,
    int channels,
    ImageFormat format);

/**
 * Create image with specific data type
 */
NoiseError image_create_typed(
    Image **image,
    int width,
    int height,
    int channels,
    ImageFormat format,
    DataType data_type);

/**
 * Destroy an image and free memory
 * @param image Image to destroy
 */
void image_destroy(Image *image);

/**
 * Clone an image
 * @param dest Destination image pointer
 * @param src Source image
 * @return Error code
 */
NoiseError image_clone(Image **dest, const Image *src);

/**
 * Copy image data
 * @param dest Destination image
 * @param src Source image
 * @return Error code
 */
NoiseError image_copy(Image *dest, const Image *src);

/**
 * Convert image format
 * @param output Output image
 * @param input Input image
 * @param target_format Target format
 * @return Error code
 */
NoiseError image_convert_format(
    Image *output,
    const Image *input,
    ImageFormat target_format);

/**
 * Convert image data type
 */
NoiseError image_convert_type(
    Image *output,
    const Image *input,
    DataType target_type);

/**
 * Get pixel value at coordinates
 * @param image Source image
 * @param x X coordinate
 * @param y Y coordinate
 * @param pixel Output pixel buffer
 * @return Error code
 */
NoiseError image_get_pixel(
    const Image *image,
    int x,
    int y,
    void *pixel);

/**
 * Set pixel value at coordinates
 * @param image Target image
 * @param x X coordinate
 * @param y Y coordinate
 * @param pixel Input pixel data
 * @return Error code
 */
NoiseError image_set_pixel(
    Image *image,
    int x,
    int y,
    const void *pixel);

/**
 * Get pixel with boundary handling
 */
NoiseError image_get_pixel_safe(
    const Image *image,
    int x,
    int y,
    void *pixel,
    int boundary_mode);

/**
 * Resize image
 */
NoiseError image_resize(
    Image *output,
    const Image *input,
    int new_width,
    int new_height,
    int interpolation);

/**
 * Crop image
 */
NoiseError image_crop(
    Image *output,
    const Image *input,
    int x,
    int y,
    int width,
    int height);

/**
 * Split image into channels
 */
NoiseError image_split_channels(
    Image **channels,
    const Image *input);

/**
 * Merge channels into image
 */
NoiseError image_merge_channels(
    Image *output,
    const Image **channels,
    int num_channels);

// ============================================================================
// Image I/O Functions
// ============================================================================

/**
 * Load image from file
 */
NoiseError image_load(
    Image **image,
    const char *filename);

/**
 * Save image to file
 */
NoiseError image_save(
    const Image *image,
    const char *filename);

/**
 * Load raw image data
 */
NoiseError image_load_raw(
    Image **image,
    const char *filename,
    int width,
    int height,
    int channels,
    ImageFormat format);

/**
 * Save raw image data
 */
NoiseError image_save_raw(
    const Image *image,
    const char *filename);

// ============================================================================
// Image Statistics Functions
// ============================================================================

/**
 * Calculate image mean
 */
NoiseError image_mean(
    double *mean,
    const Image *image,
    int channel);

/**
 * Calculate image variance
 */
NoiseError image_variance(
    double *variance,
    const Image *image,
    int channel);

/**
 * Calculate image standard deviation
 */
NoiseError image_std_dev(
    double *std_dev,
    const Image *image,
    int channel);

/**
 * Calculate image histogram
 */
NoiseError image_histogram(
    int *histogram,
    const Image *image,
    int channel,
    int bins);

/**
 * Calculate image min/max values
 */
NoiseError image_min_max(
    double *min_val,
    double *max_val,
    const Image *image,
    int channel);

/**
 * Calculate image entropy
 */
NoiseError image_entropy(
    double *entropy,
    const Image *image,
    int channel);

// ============================================================================
// Noise Analysis Functions
// ============================================================================

/**
 * Estimate noise level in image
 * @param noise_level Output noise level
 * @param image Input image
 * @return Error code
 */
NoiseError estimate_noise_level(
    double *noise_level,
    const Image *image);

/**
 * Estimate noise level per channel
 */
NoiseError estimate_noise_level_per_channel(
    double *noise_levels,
    const Image *image);

/**
 * Detect noise type
 * @param type Output noise type
 * @param image Input image
 * @return Error code
 */
NoiseError detect_noise_type(
    NoiseType *type,
    const Image *image);

/**
 * Analyze noise characteristics
 */
typedef struct {
    NoiseType type;
    double level;
    double variance;
    bool is_uniform;
    bool is_periodic;
    double frequency;  // For periodic noise
} NoiseCharacteristics;

NoiseError analyze_noise(
    NoiseCharacteristics *characteristics,
    const Image *image);

/**
 * Calculate Signal-to-Noise Ratio (SNR)
 * @param snr Output SNR value
 * @param noisy Noisy image
 * @param clean Clean reference image
 * @return Error code
 */
NoiseError calculate_snr(
    double *snr,
    const Image *noisy,
    const Image *clean);

/**
 * Calculate Peak Signal-to-Noise Ratio (PSNR)
 * @param psnr Output PSNR value
 * @param image1 First image
 * @param image2 Second image
 * @return Error code
 */
NoiseError calculate_psnr(
    double *psnr,
    const Image *image1,
    const Image *image2);

/**
 * Calculate Structural Similarity Index (SSIM)
 * @param ssim Output SSIM value
 * @param image1 First image
 * @param image2 Second image
 * @return Error code
 */
NoiseError calculate_ssim(
    double *ssim,
    const Image *image1,
    const Image *image2);

/**
 * Calculate Mean Squared Error (MSE)
 */
NoiseError calculate_mse(
    double *mse,
    const Image *image1,
    const Image *image2);

/**
 * Calculate Mean Absolute Error (MAE)
 */
NoiseError calculate_mae(
    double *mae,
    const Image *image1,
    const Image *image2);

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Get error message string
 */
const char* noise_error_string(NoiseError error);

/**
 * Get image format name
 */
const char* image_format_name(ImageFormat format);

/**
 * Get noise type name
 */
const char* noise_type_name(NoiseType type);

/**
 * Validate image
 */
bool image_is_valid(const Image *image);

/**
 * Check if two images are compatible
 */
bool images_are_compatible(const Image *image1, const Image *image2);

/**
 * Get bytes per pixel
 */
int get_bytes_per_pixel(const Image *image);

/**
 * Get pixel size for data type
 */
size_t get_data_type_size(DataType type);

#ifdef __cplusplus
}
#endif

#endif // NOISE_REDUCTION_H (Part 1/3)
// ============================================================================
// Part 2/3: Basic and Advanced Filtering Algorithms
// ============================================================================

// ============================================================================
// Basic Filtering Functions
// ============================================================================

/**
 * Apply Gaussian blur filter
 * @param output Output image
 * @param input Input image
 * @param kernel_size Kernel size (odd number)
 * @param sigma Standard deviation
 * @return Error code
 */
NoiseError gaussian_filter(
    Image *output,
    const Image *input,
    int kernel_size,
    double sigma);

/**
 * Apply separable Gaussian filter (faster)
 */
NoiseError gaussian_filter_separable(
    Image *output,
    const Image *input,
    int kernel_size,
    double sigma);

/**
 * Apply median filter
 * @param output Output image
 * @param input Input image
 * @param kernel_size Kernel size
 * @return Error code
 */
NoiseError median_filter(
    Image *output,
    const Image *input,
    int kernel_size);

/**
 * Apply fast median filter
 */
NoiseError fast_median_filter(
    Image *output,
    const Image *input,
    int kernel_size);

/**
 * Apply mean filter (box blur)
 * @param output Output image
 * @param input Input image
 * @param kernel_size Kernel size
 * @return Error code
 */
NoiseError mean_filter(
    Image *output,
    const Image *input,
    int kernel_size);

/**
 * Apply weighted mean filter
 */
NoiseError weighted_mean_filter(
    Image *output,
    const Image *input,
    const double *weights,
    int kernel_size);

// ============================================================================
// Bilateral Filter
// ============================================================================

typedef struct {
    int window_size;      // Bilateral filter window size
    double sigma_color;   // Color space sigma
    double sigma_space;   // Coordinate space sigma
    int border_mode;      // Border handling mode
} BilateralParams;

/**
 * Apply bilateral filter
 * @param output Output image
 * @param input Input image
 * @param params Bilateral filter parameters
 * @return Error code
 */
NoiseError bilateral_filter(
    Image *output,
    const Image *input,
    const BilateralParams *params);

/**
 * Apply fast bilateral filter
 */
NoiseError fast_bilateral_filter(
    Image *output,
    const Image *input,
    const BilateralParams *params);

/**
 * Apply adaptive bilateral filter
 */
NoiseError adaptive_bilateral_filter(
    Image *output,
    const Image *input,
    int window_size);

// ============================================================================
// Guided Filter
// ============================================================================

/**
 * Apply guided filter
 * @param output Output image
 * @param input Input image to filter
 * @param guide Guide image
 * @param radius Filter radius
 * @param epsilon Regularization parameter
 * @return Error code
 */
NoiseError guided_filter(
    Image *output,
    const Image *input,
    const Image *guide,
    int radius,
    double epsilon);

/**
 * Apply fast guided filter
 */
NoiseError fast_guided_filter(
    Image *output,
    const Image *input,
    const Image *guide,
    int radius,
    double epsilon,
    int subsample_ratio);

// ============================================================================
// Non-Local Means (NLM)
// ============================================================================

typedef struct {
    int search_window;    // Search window size
    int patch_size;       // Patch size for comparison
    double h;             // Filtering parameter
    double sigma;         // Noise standard deviation estimate
    int num_threads;      // Number of threads for parallel processing
} NLMeansParams;

/**
 * Non-Local Means denoising
 * @param output Output image
 * @param input Input image
 * @param params NLM parameters
 * @return Error code
 */
NoiseError nlmeans_denoise(
    Image *output,
    const Image *input,
    const NLMeansParams *params);

/**
 * Fast Non-Local Means
 */
NoiseError fast_nlmeans_denoise(
    Image *output,
    const Image *input,
    const NLMeansParams *params);

/**
 * Non-Local Means with automatic parameter estimation
 */
NoiseError nlmeans_auto(
    Image *output,
    const Image *input);

// ============================================================================
// BM3D (Block-Matching and 3D Filtering)
// ============================================================================

typedef struct {
    int search_window;    // Search window size
    int patch_size;       // Block size
    double h_luminance;   // Luminance filtering strength
    double h_color;       // Color filtering strength
    int max_matched_blocks; // Maximum number of similar blocks
    double sigma;         // Noise standard deviation
    bool use_hard_threshold; // Use hard thresholding in first step
} BM3DParams;

/**
 * BM3D denoising
 * @param output Output image
 * @param input Input image
 * @param params BM3D parameters
 * @return Error code
 */
NoiseError bm3d_denoise(
    Image *output,
    const Image *input,
    const BM3DParams *params);

/**
 * BM3D with automatic parameter estimation
 */
NoiseError bm3d_auto(
    Image *output,
    const Image *input);

// ============================================================================
// Total Variation (TV) Denoising
// ============================================================================

typedef struct {
    double lambda;        // Regularization parameter
    int iterations;       // Number of iterations
    double tolerance;     // Convergence tolerance
    bool isotropic;       // Use isotropic TV
    double time_step;     // Time step for gradient descent
} TVDenoiseParams;

/**
 * Total Variation denoising
 * @param output Output image
 * @param input Input image
 * @param params TV parameters
 * @return Error code
 */
NoiseError tv_denoise(
    Image *output,
    const Image *input,
    const TVDenoiseParams *params);

/**
 * Chambolle TV denoising algorithm
 */
NoiseError tv_chambolle(
    Image *output,
    const Image *input,
    double lambda,
    int iterations);

/**
 * Split Bregman TV denoising
 */
NoiseError tv_split_bregman(
    Image *output,
    const Image *input,
    double lambda,
    int iterations);

// ============================================================================
// Wavelet Denoising
// ============================================================================

typedef enum {
    WAVELET_HAAR,
    WAVELET_DB4,
    WAVELET_DB8,
    WAVELET_SYM4,
    WAVELET_COIF4,
    WAVELET_BIOR4_4
} WaveletType;

typedef enum {
    THRESHOLD_SOFT,
    THRESHOLD_HARD,
    THRESHOLD_GARROTE
} ThresholdType;

typedef struct {
    int levels;              // Number of decomposition levels
    double threshold;        // Threshold value
    WaveletType wavelet;     // Wavelet type
    ThresholdType threshold_type; // Thresholding method
    bool use_bayesian;       // Use Bayesian threshold estimation
} WaveletParams;

/**
 * Wavelet denoising
 * @param output Output image
 * @param input Input image
 * @param params Wavelet parameters
 * @return Error code
 */
NoiseError wavelet_denoise(
    Image *output,
    const Image *input,
    const WaveletParams *params);

/**
 * Wavelet denoising with VisuShrink threshold
 */
NoiseError wavelet_denoise_visushrink(
    Image *output,
    const Image *input,
    int levels,
    WaveletType wavelet);

/**
 * Wavelet denoising with BayesShrink threshold
 */
NoiseError wavelet_denoise_bayesshrink(
    Image *output,
    const Image *input,
    int levels,
    WaveletType wavelet);

// ============================================================================
// Anisotropic Diffusion
// ============================================================================

typedef enum {
    DIFFUSION_PERONA_MALIK_1,
    DIFFUSION_PERONA_MALIK_2,
    DIFFUSION_WEICKERT,
    DIFFUSION_TUKEY
} DiffusionType;

/**
 * Anisotropic diffusion (Perona-Malik)
 * @param output Output image
 * @param input Input image
 * @param iterations Number of iterations
 * @param lambda Time step
 * @param kappa Gradient threshold
 * @return Error code
 */
NoiseError anisotropic_diffusion(
    Image *output,
    const Image *input,
    int iterations,
    double lambda,
    double kappa);

/**
 * Anisotropic diffusion with specific type
 */
NoiseError anisotropic_diffusion_typed(
    Image *output,
    const Image *input,
    DiffusionType type,
    int iterations,
    double lambda,
    double kappa);

/**
 * Coherence-enhancing diffusion
 */
NoiseError coherence_enhancing_diffusion(
    Image *output,
    const Image *input,
    int iterations,
    double alpha,
    double sigma);

// ============================================================================
// Wiener Filter
// ============================================================================

/**
 * Wiener filter denoising
 * @param output Output image
 * @param input Input image
 * @param kernel_size Kernel size
 * @param noise_variance Noise variance estimate
 * @return Error code
 */
NoiseError wiener_filter(
    Image *output,
    const Image *input,
    int kernel_size,
    double noise_variance);

/**
 * Adaptive Wiener filter
 */
NoiseError adaptive_wiener_filter(
    Image *output,
    const Image *input,
    int window_size);

/**
 * Wiener filter in frequency domain
 */
NoiseError wiener_filter_frequency(
    Image *output,
    const Image *input,
    double noise_power,
    double signal_power);

// ============================================================================
// Morphological Filters
// ============================================================================

typedef enum {
    MORPH_RECT,
    MORPH_CROSS,
    MORPH_ELLIPSE
} MorphShape;

/**
 * Morphological opening
 * @param output Output image
 * @param input Input image
 * @param kernel_size Kernel size
 * @return Error code
 */
NoiseError morphological_opening(
    Image *output,
    const Image *input,
    int kernel_size);

/**
 * Morphological closing
 * @param output Output image
 * @param input Input image
 * @param kernel_size Kernel size
 * @return Error code
 */
NoiseError morphological_closing(
    Image *output,
    const Image *input,
    int kernel_size);

/**
 * Morphological gradient
 * @param output Output image
 * @param input Input image
 * @param kernel_size Kernel size
 * @return Error code
 */
NoiseError morphological_gradient(
    Image *output,
    const Image *input,
    int kernel_size);

/**
 * Morphological operations with custom structuring element
 */
NoiseError morphological_operation(
    Image *output,
    const Image *input,
    const uint8_t *structuring_element,
    int se_width,
    int se_height,
    int operation); // 0=erode, 1=dilate, 2=open, 3=close

/**
 * Top-hat transform
 */
NoiseError top_hat_transform(
    Image *output,
    const Image *input,
    int kernel_size,
    bool white_tophat);

/**
 * Morphological reconstruction
 */
NoiseError morphological_reconstruction(
    Image *output,
    const Image *marker,
    const Image *mask);

// End of Part 2/3
// ============================================================================
// Part 3/3: Advanced Algorithms and Utility Functions
// ============================================================================

// ============================================================================
// Edge-Preserving Filters
// ============================================================================

/**
 * Edge-preserving smoothing
 * @param output Output image
 * @param input Input image
 * @param sigma_color Color space sigma
 * @param sigma_space Spatial sigma
 * @return Error code
 */
NoiseError edge_preserving_smooth(
    Image *output,
    const Image *input,
    double sigma_color,
    double sigma_space);

/**
 * Domain transform filter
 * @param output Output image
 * @param input Input image
 * @param sigma_color Color sigma
 * @param sigma_space Spatial sigma
 * @param iterations Number of iterations
 * @return Error code
 */
NoiseError domain_transform_filter(
    Image *output,
    const Image *input,
    double sigma_color,
    double sigma_space,
    int iterations);

/**
 * Recursive bilateral filter
 */
NoiseError recursive_bilateral_filter(
    Image *output,
    const Image *input,
    double sigma_color,
    double sigma_space);

/**
 * L0 gradient minimization
 */
NoiseError l0_gradient_minimization(
    Image *output,
    const Image *input,
    double lambda,
    double kappa);

/**
 * Relative total variation filter
 */
NoiseError relative_total_variation(
    Image *output,
    const Image *input,
    double lambda,
    int iterations);

// ============================================================================
// Frequency Domain Filters
// ============================================================================

/**
 * FFT-based noise reduction
 * @param output Output image
 * @param input Input image
 * @param threshold Frequency threshold
 * @return Error code
 */
NoiseError fft_denoise(
    Image *output,
    const Image *input,
    double threshold);

/**
 * Notch filter (remove periodic noise)
 * @param output Output image
 * @param input Input image
 * @param frequencies Array of frequencies to remove
 * @param num_frequencies Number of frequencies
 * @param bandwidth Notch bandwidth
 * @return Error code
 */
NoiseError notch_filter(
    Image *output,
    const Image *input,
    const double *frequencies,
    int num_frequencies,
    double bandwidth);

/**
 * Band-pass filter
 * @param output Output image
 * @param input Input image
 * @param low_freq Low frequency cutoff
 * @param high_freq High frequency cutoff
 * @return Error code
 */
NoiseError bandpass_filter(
    Image *output,
    const Image *input,
    double low_freq,
    double high_freq);

/**
 * Ideal low-pass filter
 */
NoiseError ideal_lowpass_filter(
    Image *output,
    const Image *input,
    double cutoff_frequency);

/**
 * Butterworth filter
 */
NoiseError butterworth_filter(
    Image *output,
    const Image *input,
    double cutoff_frequency,
    int order);

/**
 * Homomorphic filter
 */
NoiseError homomorphic_filter(
    Image *output,
    const Image *input,
    double gamma_low,
    double gamma_high,
    double cutoff);

// ============================================================================
// Specialized Noise Removal
// ============================================================================

/**
 * Remove salt and pepper noise
 * @param output Output image
 * @param input Input image
 * @param kernel_size Kernel size
 * @return Error code
 */
NoiseError remove_salt_pepper_noise(
    Image *output,
    const Image *input,
    int kernel_size);

/**
 * Remove Gaussian noise
 * @param output Output image
 * @param input Input image
 * @param sigma Noise standard deviation
 * @return Error code
 */
NoiseError remove_gaussian_noise(
    Image *output,
    const Image *input,
    double sigma);

/**
 * Remove speckle noise
 * @param output Output image
 * @param input Input image
 * @param window_size Window size
 * @return Error code
 */
NoiseError remove_speckle_noise(
    Image *output,
    const Image *input,
    int window_size);

/**
 * Remove periodic noise
 * @param output Output image
 * @param input Input image
 * @return Error code
 */
NoiseError remove_periodic_noise(
    Image *output,
    const Image *input);

/**
 * Remove impulse noise
 */
NoiseError remove_impulse_noise(
    Image *output,
    const Image *input,
    double threshold);

/**
 * Remove quantization noise
 */
NoiseError remove_quantization_noise(
    Image *output,
    const Image *input);

/**
 * Remove compression artifacts
 */
NoiseError remove_compression_artifacts(
    Image *output,
    const Image *input,
    int quality_estimate);

// ============================================================================
// Adaptive Filters
// ============================================================================

/**
 * Adaptive median filter
 * @param output Output image
 * @param input Input image
 * @param max_window_size Maximum window size
 * @return Error code
 */
NoiseError adaptive_median_filter(
    Image *output,
    const Image *input,
    int max_window_size);

/**
 * Local adaptive filter
 * @param output Output image
 * @param input Input image
 * @param window_size Window size
 * @param noise_variance Noise variance estimate
 * @return Error code
 */
NoiseError local_adaptive_filter(
    Image *output,
    const Image *input,
    int window_size,
    double noise_variance);

/**
 * Adaptive Gaussian filter
 */
NoiseError adaptive_gaussian_filter(
    Image *output,
    const Image *input,
    int max_kernel_size);

/**
 * Sigma filter
 */
NoiseError sigma_filter(
    Image *output,
    const Image *input,
    int window_size,
    double sigma_range);

/**
 * Kuwahara filter
 */
NoiseError kuwahara_filter(
    Image *output,
    const Image *input,
    int window_size);

// ============================================================================
// Multi-scale Processing
// ============================================================================

/**
 * Pyramid-based denoising
 * @param output Output image
 * @param input Input image
 * @param levels Number of pyramid levels
 * @param threshold Threshold value
 * @return Error code
 */
NoiseError pyramid_denoise(
    Image *output,
    const Image *input,
    int levels,
    double threshold);

/**
 * Multi-resolution denoising
 * @param output Output image
 * @param input Input image
 * @param levels Number of levels
 * @param params Filter parameters
 * @return Error code
 */
NoiseError multiresolution_denoise(
    Image *output,
    const Image *input,
    int levels,
    const FilterParams *params);

/**
 * Laplacian pyramid denoising
 */
NoiseError laplacian_pyramid_denoise(
    Image *output,
    const Image *input,
    int levels,
    double threshold);

/**
 * Steerable pyramid denoising
 */
NoiseError steerable_pyramid_denoise(
    Image *output,
    const Image *input,
    int levels,
    int orientations);

// ============================================================================
// Deep Learning Based (Placeholder for integration)
// ============================================================================

/**
 * DnCNN denoising (requires external model)
 * @param output Output image
 * @param input Input image
 * @param model_path Path to model file
 * @return Error code
 */
NoiseError dncnn_denoise(
    Image *output,
    const Image *input,
    const char *model_path);

/**
 * FFDNet denoising (requires external model)
 * @param output Output image
 * @param input Input image
 * @param noise_level Noise level
 * @param model_path Path to model file
 * @return Error code
 */
NoiseError ffdnet_denoise(
    Image *output,
    const Image *input,
    double noise_level,
    const char *model_path);

/**
 * CBDNet denoising
 */
NoiseError cbdnet_denoise(
    Image *output,
    const Image *input,
    const char *model_path);

/**
 * RIDNet denoising
 */
NoiseError ridnet_denoise(
    Image *output,
    const Image *input,
    const char *model_path);

// ============================================================================
// Hybrid and Combined Methods
// ============================================================================

/**
 * Combine multiple denoising results
 */
NoiseError combine_denoise_results(
    Image *output,
    const Image **inputs,
    const double *weights,
    int num_inputs);

/**
 * Sequential denoising (apply multiple filters)
 */
NoiseError sequential_denoise(
    Image *output,
    const Image *input,
    const int *filter_types,
    const void **filter_params,
    int num_filters);

/**
 * Adaptive method selection
 */
NoiseError adaptive_denoise(
    Image *output,
    const Image *input);

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Add synthetic noise to image (for testing)
 * @param output Output image
 * @param input Input image
 * @param type Noise type
 * @param intensity Noise intensity
 * @return Error code
 */
NoiseError add_noise(
    Image *output,
    const Image *input,
    NoiseType type,
    double intensity);

/**
 * Add Gaussian noise
 */
NoiseError add_gaussian_noise(
    Image *output,
    const Image *input,
    double mean,
    double std_dev);

/**
 * Add salt and pepper noise
 */
NoiseError add_salt_pepper_noise(
    Image *output,
    const Image *input,
    double salt_prob,
    double pepper_prob);

/**
 * Add Poisson noise
 */
NoiseError add_poisson_noise(
    Image *output,
    const Image *input);

/**
 * Add speckle noise
 */
NoiseError add_speckle_noise(
    Image *output,
    const Image *input,
    double variance);

/**
 * Compare two images and generate difference map
 * @param output Output difference map
 * @param image1 First image
 * @param image2 Second image
 * @return Error code
 */
NoiseError generate_difference_map(
    Image *output,
    const Image *image1,
    const Image *image2);

/**
 * Generate error map with color coding
 */
NoiseError generate_error_map(
    Image *output,
    const Image *reference,
    const Image *test);

/**
 * Auto-tune denoising parameters
 * @param params Output parameters
 * @param image Input image
 * @return Error code
 */
NoiseError auto_tune_parameters(
    FilterParams *params,
    const Image *image);

/**
 * Get recommended parameters for noise type
 * @param params Output parameters
 * @param noise_type Detected noise type
 * @param noise_level Estimated noise level
 * @return Error code
 */
NoiseError get_recommended_params(
    FilterParams *params,
    NoiseType noise_type,
    double noise_level);

/**
 * Batch process multiple images
 * @param outputs Array of output images
 * @param inputs Array of input images
 * @param count Number of images
 * @param params Filter parameters
 * @return Error code
 */
NoiseError batch_denoise(
    Image **outputs,
    const Image **inputs,
    int count,
    const FilterParams *params);

/**
 * Process image region
 */
NoiseError denoise_region(
    Image *image,
    int x,
    int y,
    int width,
    int height,
    const FilterParams *params);

/**
 * Get algorithm name
 * @param algorithm_id Algorithm identifier
 * @return Algorithm name string
 */
const char* get_algorithm_name(int algorithm_id);

/**
 * List available algorithms
 */
NoiseError list_algorithms(
    char ***algorithm_names,
    int *count);

/**
 * Benchmark algorithm performance
 */
typedef struct {
    double processing_time;
    double psnr;
    double ssim;
    size_t memory_used;
} BenchmarkResult;

NoiseError benchmark_algorithm(
    BenchmarkResult *result,
    const Image *input,
    const Image *reference,
    int algorithm_id,
    const void *params);

// ============================================================================
// Performance and Optimization
// ============================================================================

/**
 * Enable multi-threading
 * @param enable Enable/disable flag
 */
void enable_multithreading(bool enable);

/**
 * Set number of threads
 * @param num_threads Number of threads
 */
void set_num_threads(int num_threads);

/**
 * Get number of threads
 */
int get_num_threads(void);

/**
 * Enable GPU acceleration (if available)
 * @param enable Enable/disable flag
 * @return Error code
 */
NoiseError enable_gpu_acceleration(bool enable);

/**
 * Check if GPU is available
 */
bool is_gpu_available(void);

/**
 * Get GPU device information
 */
NoiseError get_gpu_info(
    char *device_name,
    size_t name_size,
    size_t *memory_size);

/**
 * Enable SIMD optimizations
 */
void enable_simd(bool enable);

/**
 * Set cache size for optimization
 */
void set_cache_size(size_t size);

/**
 * Get processing time statistics
 * @param elapsed_time Output elapsed time
 * @param memory_used Output memory usage
 * @return Error code
 */
NoiseError get_processing_stats(
    double *elapsed_time,
    size_t *memory_used);

/**
 * Reset performance counters
 */
void reset_performance_counters(void);

/**
 * Get detailed performance profile
 */
typedef struct {
    double total_time;
    double preprocessing_time;
    double filtering_time;
    double postprocessing_time;
    size_t peak_memory;
    size_t current_memory;
    int num_operations;
} PerformanceProfile;

NoiseError get_performance_profile(PerformanceProfile *profile);

// ============================================================================
// Configuration and Settings
// ============================================================================

/**
 * Initialize library
 */
NoiseError noise_reduction_init(void);

/**
 * Cleanup library resources
 */
void noise_reduction_cleanup(void);

/**
 * Get library version
 */
void get_version(int *major, int *minor, int *patch);

/**
 * Get version string
 */
const char* get_version_string(void);

/**
 * Set verbosity level
 */
void set_verbosity(int level);

/**
 * Enable/disable logging
 */
void enable_logging(bool enable, const char *log_file);

/**
 * Set memory limit
 */
NoiseError set_memory_limit(size_t limit_bytes);

/**
 * Get memory limit
 */
size_t get_memory_limit(void);

/**
 * Set temporary directory
 */
NoiseError set_temp_directory(const char *path);

// ============================================================================
// Debug and Visualization
// ============================================================================

/**
 * Visualize filter kernel
 */
NoiseError visualize_kernel(
    Image *output,
    const double *kernel,
    int size);

/**
 * Visualize frequency response
 */
NoiseError visualize_frequency_response(
    Image *output,
    const Image *input);

/**
 * Generate noise map
 */
NoiseError generate_noise_map(
    Image *output,
    const Image *input);

/**
 * Export processing pipeline
 */
NoiseError export_pipeline(
    const char *filename,
    const char *format);

#ifdef __cplusplus
}
#endif

#endif // NOISE_REDUCTION_H (Part 3/3 - Complete)

