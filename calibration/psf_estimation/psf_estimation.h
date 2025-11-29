/**
 * @file psf_estimation.h
 * @brief Point Spread Function (PSF) Estimation Library
 * @author hany
 * @version 1.0.0
 * @date 2025
 * 
 * This library provides comprehensive PSF estimation and analysis tools for
 * optical system characterization, including:
 * - Edge-based PSF estimation
 * - Star-based PSF estimation
 * - Blind deconvolution
 * - PSF modeling and fitting
 * - Image quality metrics
 */

#ifndef PSF_ESTIMATION_H
#define PSF_ESTIMATION_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

// ============================================================================
// Version Information
// ============================================================================

#define PSF_ESTIMATION_VERSION_MAJOR 1
#define PSF_ESTIMATION_VERSION_MINOR 0
#define PSF_ESTIMATION_VERSION_PATCH 0
#define PSF_ESTIMATION_VERSION_STRING "1.0.0"

// ============================================================================
// Platform-specific definitions
// ============================================================================

#ifdef _WIN32
    #ifdef PSF_ESTIMATION_EXPORTS
        #define PSF_API __declspec(dllexport)
    #else
        #define PSF_API __declspec(dllimport)
    #endif
#else
    #define PSF_API __attribute__((visibility("default")))
#endif

// ============================================================================
// Error Codes
// ============================================================================

typedef enum {
    PSF_SUCCESS = 0,
    PSF_ERROR_NULL_POINTER = -1,
    PSF_ERROR_INVALID_PARAM = -2,
    PSF_ERROR_MEMORY_ALLOCATION = -3,
    PSF_ERROR_INVALID_IMAGE = -4,
    PSF_ERROR_NO_EDGES_FOUND = -5,
    PSF_ERROR_NO_STARS_FOUND = -6,
    PSF_ERROR_CONVERGENCE_FAILED = -7,
    PSF_ERROR_INVALID_PSF = -8,
    PSF_ERROR_FILE_IO = -9,
    PSF_ERROR_UNSUPPORTED_FORMAT = -10,
    PSF_ERROR_INSUFFICIENT_DATA = -11,
    PSF_ERROR_NUMERICAL_ERROR = -12
} PSFError;

// ============================================================================
// Data Structures
// ============================================================================

/**
 * @brief 2D Point structure
 */
typedef struct {
    double x;
    double y;
} PSFPoint2D;

/**
 * @brief Image structure
 */
typedef struct {
    int width;
    int height;
    int channels;
    double *data;  // Normalized to [0, 1]
} PSFImage;

/**
 * @brief PSF kernel structure
 */
typedef struct {
    int width;
    int height;
    double *data;  // Normalized PSF kernel
    double center_x;
    double center_y;
    double total_energy;
} PSFKernel;

/**
 * @brief Edge detection result
 */
typedef struct {
    PSFPoint2D *points;
    int num_points;
    double angle;  // Edge angle in radians
    double strength;  // Edge strength
    double sharpness;  // Edge sharpness metric
} PSFEdge;

/**
 * @brief Star detection result
 */
typedef struct {
    PSFPoint2D center;
    double brightness;
    double fwhm;  // Full Width at Half Maximum
    double ellipticity;
    double angle;  // Orientation angle
    double snr;  // Signal-to-noise ratio
} PSFStar;

/**
 * @brief PSF model types
 */
typedef enum {
    PSF_MODEL_GAUSSIAN,
    PSF_MODEL_MOFFAT,
    PSF_MODEL_AIRY,
    PSF_MODEL_DOUBLE_GAUSSIAN,
    PSF_MODEL_GENERALIZED_GAUSSIAN,
    PSF_MODEL_CUSTOM
} PSFModelType;

/**
 * @brief Gaussian PSF parameters
 */
typedef struct {
    double amplitude;
    double center_x;
    double center_y;
    double sigma_x;
    double sigma_y;
    double rotation;  // Rotation angle in radians
    double background;
} PSFGaussianParams;

/**
 * @brief Moffat PSF parameters
 */
typedef struct {
    double amplitude;
    double center_x;
    double center_y;
    double alpha;  // Width parameter
    double beta;   // Shape parameter
    double rotation;
    double background;
} PSFMoffatParams;

/**
 * @brief Airy PSF parameters
 */
typedef struct {
    double amplitude;
    double center_x;
    double center_y;
    double radius;  // First zero radius
    double obscuration;  // Central obscuration ratio
    double background;
} PSFAiryParams;

/**
 * @brief PSF model parameters union
 */
typedef union {
    PSFGaussianParams gaussian;
    PSFMoffatParams moffat;
    PSFAiryParams airy;
} PSFModelParams;

/**
 * @brief Complete PSF model
 */
typedef struct {
    PSFModelType type;
    PSFModelParams params;
    double fit_error;
    int num_iterations;
} PSFModel;

/**
 * @brief PSF quality metrics
 */
typedef struct {
    double fwhm_x;  // Full Width at Half Maximum in x
    double fwhm_y;  // Full Width at Half Maximum in y
    double fwhm_avg;  // Average FWHM
    double ellipticity;  // PSF ellipticity
    double strehl_ratio;  // Strehl ratio
    double encircled_energy_50;  // 50% encircled energy radius
    double encircled_energy_80;  // 80% encircled energy radius
    double rms_width;  // RMS width
    double entropy;  // PSF entropy
    double sharpness;  // Sharpness metric
} PSFQualityMetrics;

/**
 * @brief Edge Spread Function (ESF)
 */
typedef struct {
    double *data;
    int length;
    double pixel_spacing;
    double edge_position;
} PSFEdgeSpreadFunction;

/**
 * @brief Line Spread Function (LSF)
 */
typedef struct {
    double *data;
    int length;
    double pixel_spacing;
    double peak_position;
    double fwhm;
} PSFLineSpreadFunction;

/**
 * @brief Modulation Transfer Function (MTF)
 */
typedef struct {
    double *frequencies;  // Cycles per pixel
    double *values;  // MTF values [0, 1]
    int length;
    double mtf50;  // Frequency at 50% MTF
    double mtf20;  // Frequency at 20% MTF
} PSFModulationTransferFunction;

/**
 * @brief PSF estimation configuration
 */
typedef struct {
    // Edge-based estimation
    double edge_threshold;
    int edge_roi_size;
    int edge_oversample_factor;
    
    // Star-based estimation
    double star_detection_threshold;
    int star_min_size;
    int star_max_size;
    double star_roundness_threshold;
    
    // Blind deconvolution
    int blind_max_iterations;
    double blind_convergence_threshold;
    int blind_psf_size;
    
    // General
    int psf_kernel_size;
    bool normalize_psf;
    bool remove_background;
    double noise_level;
} PSFEstimationConfig;

/**
 * @brief Blind deconvolution context
 */
typedef struct {
    PSFImage *current_image;
    PSFKernel *current_psf;
    double *residual;
    int iteration;
    double error;
    bool converged;
} PSFBlindDeconvContext;

// ============================================================================
// Image Operations
// ============================================================================

/**
 * @brief Create image structure
 */
PSF_API PSFError psf_create_image(
    int width,
    int height,
    int channels,
    PSFImage **image);

/**
 * @brief Destroy image structure
 */
PSF_API void psf_destroy_image(PSFImage *image);

/**
 * @brief Copy image
 */
PSF_API PSFError psf_copy_image(
    const PSFImage *src,
    PSFImage **dst);

/**
 * @brief Load image from raw data
 */
PSF_API PSFError psf_load_image_from_data(
    const unsigned char *data,
    int width,
    int height,
    int channels,
    PSFImage **image);

/**
 * @brief Convert image to grayscale
 */
PSF_API PSFError psf_convert_to_grayscale(
    const PSFImage *src,
    PSFImage **dst);

/**
 * @brief Normalize image to [0, 1]
 */
PSF_API PSFError psf_normalize_image(
    PSFImage *image,
    double min_val,
    double max_val);

/**
 * @brief Extract image region
 */
PSF_API PSFError psf_extract_roi(
    const PSFImage *src,
    int x,
    int y,
    int width,
    int height,
    PSFImage **roi);

// ============================================================================
// PSF Kernel Operations
// ============================================================================

/**
 * @brief Create PSF kernel
 */
PSF_API PSFError psf_create_kernel(
    int width,
    int height,
    PSFKernel **kernel);

/**
 * @brief Destroy PSF kernel
 */
PSF_API void psf_destroy_kernel(PSFKernel *kernel);

/**
 * @brief Copy PSF kernel
 */
PSF_API PSFError psf_copy_kernel(
    const PSFKernel *src,
    PSFKernel **dst);

/**
 * @brief Normalize PSF kernel
 */
PSF_API PSFError psf_normalize_kernel(PSFKernel *kernel);

/**
 * @brief Resize PSF kernel
 */
PSF_API PSFError psf_resize_kernel(
    const PSFKernel *src,
    int new_width,
    int new_height,
    PSFKernel **dst);

/**
 * @brief Center PSF kernel
 */
PSF_API PSFError psf_center_kernel(PSFKernel *kernel);

// ============================================================================
// Edge-Based PSF Estimation
// ============================================================================

/**
 * @brief Detect edges in image
 */
PSF_API PSFError psf_detect_edges(
    const PSFImage *image,
    double threshold,
    PSFEdge ***edges,
    int *num_edges);

/**
 * @brief Destroy edge structure
 */
PSF_API void psf_destroy_edge(PSFEdge *edge);

/**
 * @brief Destroy edge array
 */
PSF_API void psf_destroy_edges(PSFEdge **edges, int num_edges);

/**
 * @brief Extract Edge Spread Function from edge
 */
PSF_API PSFError psf_extract_esf(
    const PSFImage *image,
    const PSFEdge *edge,
    int roi_size,
    int oversample_factor,
    PSFEdgeSpreadFunction **esf);

/**
 * @brief Destroy ESF structure
 */
PSF_API void psf_destroy_esf(PSFEdgeSpreadFunction *esf);

/**
 * @brief Compute Line Spread Function from ESF
 */
PSF_API PSFError psf_esf_to_lsf(
    const PSFEdgeSpreadFunction *esf,
    PSFLineSpreadFunction **lsf);

/**
 * @brief Destroy LSF structure
 */
PSF_API void psf_destroy_lsf(PSFLineSpreadFunction *lsf);

/**
 * @brief Compute PSF from LSF
 */
PSF_API PSFError psf_lsf_to_psf(
    const PSFLineSpreadFunction *lsf_x,
    const PSFLineSpreadFunction *lsf_y,
    PSFKernel **psf);

/**
 * @brief Estimate PSF from edges (complete pipeline)
 */
PSF_API PSFError psf_estimate_from_edges(
    const PSFImage *image,
    const PSFEstimationConfig *config,
    PSFKernel **psf);

// ============================================================================
// Star-Based PSF Estimation
// ============================================================================

/**
 * @brief Detect stars in image
 */
PSF_API PSFError psf_detect_stars(
    const PSFImage *image,
    const PSFEstimationConfig *config,
    PSFStar **stars,
    int *num_stars);

/**
 * @brief Destroy star array
 */
PSF_API void psf_destroy_stars(PSFStar *stars);

/**
 * @brief Extract PSF from single star
 */
PSF_API PSFError psf_extract_from_star(
    const PSFImage *image,
    const PSFStar *star,
    int kernel_size,
    PSFKernel **psf);

/**
 * @brief Estimate PSF from multiple stars (average)
 */
PSF_API PSFError psf_estimate_from_stars(
    const PSFImage *image,
    const PSFEstimationConfig *config,
    PSFKernel **psf);

/**
 * @brief Estimate spatially varying PSF from stars
 */
PSF_API PSFError psf_estimate_spatially_varying(
    const PSFImage *image,
    const PSFEstimationConfig *config,
    int grid_width,
    int grid_height,
    PSFKernel ***psf_grid);

// ============================================================================
// Blind Deconvolution
// ============================================================================

/**
 * @brief Create blind deconvolution context
 */
PSF_API PSFError psf_create_blind_deconv_context(
    const PSFImage *image,
    int psf_size,
    PSFBlindDeconvContext **context);

/**
 * @brief Destroy blind deconvolution context
 */
PSF_API void psf_destroy_blind_deconv_context(
    PSFBlindDeconvContext *context);

/**
 * @brief Perform one iteration of blind deconvolution
 */
PSF_API PSFError psf_blind_deconv_iterate(
    PSFBlindDeconvContext *context);

/**
 * @brief Estimate PSF using blind deconvolution
 */
PSF_API PSFError psf_estimate_blind_deconvolution(
    const PSFImage *image,
    const PSFEstimationConfig *config,
    PSFKernel **psf,
    PSFImage **deconvolved);

/**
 * @brief Richardson-Lucy deconvolution
 */
PSF_API PSFError psf_richardson_lucy_deconvolution(
    const PSFImage *image,
    const PSFKernel *psf,
    int num_iterations,
    PSFImage **deconvolved);

/**
 * @brief Wiener deconvolution
 */
PSF_API PSFError psf_wiener_deconvolution(
    const PSFImage *image,
    const PSFKernel *psf,
    double noise_variance,
    PSFImage **deconvolved);

// ============================================================================
// PSF Modeling and Fitting
// ============================================================================

/**
 * @brief Fit Gaussian model to PSF
 */
PSF_API PSFError psf_fit_gaussian(
    const PSFKernel *psf,
    PSFGaussianParams *params);

/**
 * @brief Fit Moffat model to PSF
 */
PSF_API PSFError psf_fit_moffat(
    const PSFKernel *psf,
    PSFMoffatParams *params);

/**
 * @brief Fit Airy model to PSF
 */
PSF_API PSFError psf_fit_airy(
    const PSFKernel *psf,
    PSFAiryParams *params);

/**
 * @brief Fit PSF model (automatic model selection)
 */
PSF_API PSFError psf_fit_model(
    const PSFKernel *psf,
    PSFModelType model_type,
    PSFModel **model);

/**
 * @brief Destroy PSF model
 */
PSF_API void psf_destroy_model(PSFModel *model);

/**
 * @brief Generate PSF from model
 */
PSF_API PSFError psf_generate_from_model(
    const PSFModel *model,
    int width,
    int height,
    PSFKernel **psf);

/**
 * @brief Evaluate model at point
 */
PSF_API double psf_evaluate_model(
    const PSFModel *model,
    double x,
    double y);

// ============================================================================
// PSF Quality Analysis
// ============================================================================

/**
 * @brief Compute PSF quality metrics
 */
PSF_API PSFError psf_compute_quality_metrics(
    const PSFKernel *psf,
    PSFQualityMetrics *metrics);

/**
 * @brief Compute FWHM (Full Width at Half Maximum)
 */
PSF_API PSFError psf_compute_fwhm(
    const PSFKernel *psf,
    double *fwhm_x,
    double *fwhm_y);

/**
 * @brief Compute Strehl ratio
 */
PSF_API PSFError psf_compute_strehl_ratio(
    const PSFKernel *psf,
    const PSFKernel *ideal_psf,
    double *strehl_ratio);

/**
 * @brief Compute encircled energy
 */
PSF_API PSFError psf_compute_encircled_energy(
    const PSFKernel *psf,
    double radius,
    double *energy);

/**
 * @brief Compute encircled energy curve
 */
PSF_API PSFError psf_compute_encircled_energy_curve(
    const PSFKernel *psf,
    int num_points,
    double **radii,
    double **energies);

/**
 * @brief Compute PSF entropy
 */
PSF_API double psf_compute_entropy(const PSFKernel *psf);

/**
 * @brief Compute PSF sharpness
 */
PSF_API double psf_compute_sharpness(const PSFKernel *psf);

// ============================================================================
// MTF Analysis
// ============================================================================

/**
 * @brief Compute MTF from LSF
 */
PSF_API PSFError psf_compute_mtf_from_lsf(
    const PSFLineSpreadFunction *lsf,
    PSFModulationTransferFunction **mtf);

/**
 * @brief Compute MTF from PSF
 */
PSF_API PSFError psf_compute_mtf_from_psf(
    const PSFKernel *psf,
    PSFModulationTransferFunction **mtf_x,
    PSFModulationTransferFunction **mtf_y);

/**
 * @brief Destroy MTF structure
 */
PSF_API void psf_destroy_mtf(PSFModulationTransferFunction *mtf);

/**
 * @brief Evaluate MTF at frequency
 */
PSF_API double psf_evaluate_mtf(
    const PSFModulationTransferFunction *mtf,
    double frequency);

/**
 * @brief Compute MTF50 (frequency at 50% MTF)
 */
PSF_API double psf_compute_mtf50(
    const PSFModulationTransferFunction *mtf);

// ============================================================================
// Image Convolution and Deconvolution
// ============================================================================

/**
 * @brief Convolve image with PSF
 */
PSF_API PSFError psf_convolve_image(
    const PSFImage *image,
    const PSFKernel *psf,
    PSFImage **result);

/**
 * @brief Deconvolve image with PSF
 */
PSF_API PSFError psf_deconvolve_image(
    const PSFImage *image,
    const PSFKernel *psf,
    int num_iterations,
    PSFImage **result);

// ============================================================================
// Configuration and Utilities
// ============================================================================

/**
 * @brief Create default configuration
 */
PSF_API PSFError psf_create_default_config(
    PSFEstimationConfig **config);

/**
 * @brief Destroy configuration
 */
PSF_API void psf_destroy_config(PSFEstimationConfig *config);

/**
 * @brief Save PSF to file
 */
PSF_API PSFError psf_save_kernel(
    const char *filename,
    const PSFKernel *psf);

/**
 * @brief Load PSF from file
 */
PSF_API PSFError psf_load_kernel(
    const char *filename,
    PSFKernel **psf);

/**
 * @brief Save PSF model to file
 */
PSF_API PSFError psf_save_model(
    const char *filename,
    const PSFModel *model);

/**
 * @brief Load PSF model from file
 */
PSF_API PSFError psf_load_model(
    const char *filename,
    PSFModel **model);

/**
 * @brief Get error string
 */
PSF_API const char* psf_get_error_string(PSFError error);

/**
 * @brief Get library version
 */
PSF_API void psf_get_version(int *major, int *minor, int *patch);

/**
 * @brief Get version string
 */
PSF_API const char* psf_get_version_string(void);

/**
 * @brief Print library information
 */
PSF_API void psf_print_info(void);

/**
 * @brief Print PSF kernel information
 */
PSF_API void psf_print_kernel_info(const PSFKernel *psf);

/**
 * @brief Print PSF quality metrics
 */
PSF_API void psf_print_quality_metrics(
    const PSFQualityMetrics *metrics);

/**
 * @brief Print PSF model parameters
 */
PSF_API void psf_print_model_params(const PSFModel *model);

/**
 * @brief Visualize PSF (ASCII art)
 */
PSF_API void psf_visualize_kernel(
    const PSFKernel *psf,
    int console_width,
    int console_height);

/**
 * @brief Generate PSF report
 */
PSF_API PSFError psf_generate_report(
    const char *filename,
    const PSFKernel *psf,
    const PSFModel *model,
    const PSFQualityMetrics *metrics);

#ifdef __cplusplus
}
#endif

#endif // PSF_ESTIMATION_H
