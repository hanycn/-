/**
 * @file psf_estimation.c
 * @brief Point Spread Function (PSF) Estimation Library Implementation
 * @author hany
 * @version 1.0.0
 */

#include "psf_estimation.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <assert.h>

#ifdef _OPENMP
#include <omp.h>
#endif

// ============================================================================
// Constants and Macros
// ============================================================================

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifndef M_E
#define M_E 2.71828182845904523536
#endif

#define PSF_EPSILON 1e-10
#define PSF_MAX_ITERATIONS 1000
#define PSF_DEFAULT_KERNEL_SIZE 15
#define PSF_DEFAULT_OVERSAMPLE 4

// Memory management macros
#define PSF_MALLOC(type, count) ((type*)malloc((count) * sizeof(type)))
#define PSF_CALLOC(type, count) ((type*)calloc((count), sizeof(type)))
#define PSF_REALLOC(ptr, type, count) ((type*)realloc((ptr), (count) * sizeof(type)))
#define PSF_FREE(ptr) do { if(ptr) { free(ptr); (ptr) = NULL; } } while(0)

// Math utilities
#define PSF_MIN(a, b) ((a) < (b) ? (a) : (b))
#define PSF_MAX(a, b) ((a) > (b) ? (a) : (b))
#define PSF_CLAMP(x, min, max) (PSF_MIN(PSF_MAX((x), (min)), (max)))
#define PSF_SQR(x) ((x) * (x))
#define PSF_ABS(x) ((x) < 0 ? -(x) : (x))

// ============================================================================
// Internal Helper Functions Declarations
// ============================================================================

static double gaussian_1d(double x, double sigma);
static double gaussian_2d(double x, double y, double sigma_x, double sigma_y, double rotation);
static double moffat_2d(double x, double y, double alpha, double beta);
static double airy_disk(double r, double radius);
static void compute_gradient(const double *data, int width, int height, 
                            double *grad_x, double *grad_y);
static void compute_histogram(const double *data, int length, 
                              int num_bins, double *histogram);
static double compute_otsu_threshold(const double *histogram, int num_bins);
static void apply_gaussian_filter(const double *input, double *output,
                                  int width, int height, double sigma);
static void sobel_filter(const double *input, double *grad_x, double *grad_y,
                        int width, int height);
static void non_maximum_suppression(const double *grad_mag, const double *grad_dir,
                                   double *output, int width, int height);
static void hysteresis_thresholding(double *edges, int width, int height,
                                   double low_threshold, double high_threshold);

// FFT helper functions
static void fft_1d(const double *input, double *real_out, double *imag_out, int n);
static void ifft_1d(const double *real_in, const double *imag_in, 
                   double *output, int n);
static void fft_2d(const double *input, double *real_out, double *imag_out,
                  int width, int height);
static void ifft_2d(const double *real_in, const double *imag_in,
                   double *output, int width, int height);

// ============================================================================
// Error Handling
// ============================================================================

/**
 * @brief Get error message string
 */
PSF_API const char* psf_get_error_string(PSFError error)
{
    switch (error) {
        case PSF_SUCCESS:
            return "Success";
        case PSF_ERROR_NULL_POINTER:
            return "Null pointer error";
        case PSF_ERROR_INVALID_PARAM:
            return "Invalid parameter";
        case PSF_ERROR_MEMORY_ALLOCATION:
            return "Memory allocation failed";
        case PSF_ERROR_INVALID_IMAGE:
            return "Invalid image data";
        case PSF_ERROR_NO_EDGES_FOUND:
            return "No edges found in image";
        case PSF_ERROR_NO_STARS_FOUND:
            return "No stars found in image";
        case PSF_ERROR_CONVERGENCE_FAILED:
            return "Algorithm failed to converge";
        case PSF_ERROR_INVALID_PSF:
            return "Invalid PSF kernel";
        case PSF_ERROR_FILE_IO:
            return "File I/O error";
        case PSF_ERROR_UNSUPPORTED_FORMAT:
            return "Unsupported format";
        case PSF_ERROR_INSUFFICIENT_DATA:
            return "Insufficient data";
        case PSF_ERROR_NUMERICAL_ERROR:
            return "Numerical error";
        default:
            return "Unknown error";
    }
}

/**
 * @brief Get library version
 */
PSF_API void psf_get_version(int *major, int *minor, int *patch)
{
    if (major) *major = PSF_ESTIMATION_VERSION_MAJOR;
    if (minor) *minor = PSF_ESTIMATION_VERSION_MINOR;
    if (patch) *patch = PSF_ESTIMATION_VERSION_PATCH;
}

/**
 * @brief Get version string
 */
PSF_API const char* psf_get_version_string(void)
{
    return PSF_ESTIMATION_VERSION_STRING;
}

/**
 * @brief Print library information
 */
PSF_API void psf_print_info(void)
{
    printf("PSF Estimation Library\n");
    printf("Version: %s\n", PSF_ESTIMATION_VERSION_STRING);
    printf("Build date: %s %s\n", __DATE__, __TIME__);
    printf("\n");
    printf("Features:\n");
    printf("  - Edge-based PSF estimation\n");
    printf("  - Star-based PSF estimation\n");
    printf("  - Blind deconvolution\n");
    printf("  - PSF modeling (Gaussian, Moffat, Airy)\n");
    printf("  - MTF analysis\n");
    printf("  - Image quality metrics\n");
    
#ifdef _OPENMP
    printf("  - OpenMP acceleration enabled\n");
#endif
}

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
    PSFImage **image)
{
    if (!image) {
        return PSF_ERROR_NULL_POINTER;
    }
    
    if (width <= 0 || height <= 0 || channels <= 0) {
        return PSF_ERROR_INVALID_PARAM;
    }
    
    PSFImage *img = PSF_MALLOC(PSFImage, 1);
    if (!img) {
        return PSF_ERROR_MEMORY_ALLOCATION;
    }
    
    img->width = width;
    img->height = height;
    img->channels = channels;
    
    img->data = PSF_CALLOC(double, width * height * channels);
    if (!img->data) {
        PSF_FREE(img);
        return PSF_ERROR_MEMORY_ALLOCATION;
    }
    
    *image = img;
    return PSF_SUCCESS;
}

/**
 * @brief Destroy image structure
 */
PSF_API void psf_destroy_image(PSFImage *image)
{
    if (!image) return;
    
    PSF_FREE(image->data);
    PSF_FREE(image);
}

/**
 * @brief Copy image
 */
PSF_API PSFError psf_copy_image(
    const PSFImage *src,
    PSFImage **dst)
{
    if (!src || !dst) {
        return PSF_ERROR_NULL_POINTER;
    }
    
    PSFError err = psf_create_image(src->width, src->height, src->channels, dst);
    if (err != PSF_SUCCESS) {
        return err;
    }
    
    size_t data_size = src->width * src->height * src->channels;
    memcpy((*dst)->data, src->data, data_size * sizeof(double));
    
    return PSF_SUCCESS;
}

/**
 * @brief Load image from raw data
 */
PSF_API PSFError psf_load_image_from_data(
    const unsigned char *data,
    int width,
    int height,
    int channels,
    PSFImage **image)
{
    if (!data || !image) {
        return PSF_ERROR_NULL_POINTER;
    }
    
    PSFError err = psf_create_image(width, height, channels, image);
    if (err != PSF_SUCCESS) {
        return err;
    }
    
    // Convert unsigned char [0, 255] to double [0, 1]
    size_t total_pixels = width * height * channels;
    
#ifdef _OPENMP
    #pragma omp parallel for
#endif
    for (size_t i = 0; i < total_pixels; i++) {
        (*image)->data[i] = data[i] / 255.0;
    }
    
    return PSF_SUCCESS;
}

/**
 * @brief Convert image to grayscale
 */
PSF_API PSFError psf_convert_to_grayscale(
    const PSFImage *src,
    PSFImage **dst)
{
    if (!src || !dst) {
        return PSF_ERROR_NULL_POINTER;
    }
    
    if (src->channels == 1) {
        // Already grayscale, just copy
        return psf_copy_image(src, dst);
    }
    
    PSFError err = psf_create_image(src->width, src->height, 1, dst);
    if (err != PSF_SUCCESS) {
        return err;
    }
    
    int num_pixels = src->width * src->height;
    
#ifdef _OPENMP
    #pragma omp parallel for
#endif
    for (int i = 0; i < num_pixels; i++) {
        if (src->channels == 3) {
            // RGB to grayscale using standard weights
            double r = src->data[i * 3 + 0];
            double g = src->data[i * 3 + 1];
            double b = src->data[i * 3 + 2];
            (*dst)->data[i] = 0.299 * r + 0.587 * g + 0.114 * b;
        } else if (src->channels == 4) {
            // RGBA to grayscale
            double r = src->data[i * 4 + 0];
            double g = src->data[i * 4 + 1];
            double b = src->data[i * 4 + 2];
            (*dst)->data[i] = 0.299 * r + 0.587 * g + 0.114 * b;
        } else {
            // Average all channels
            double sum = 0.0;
            for (int c = 0; c < src->channels; c++) {
                sum += src->data[i * src->channels + c];
            }
            (*dst)->data[i] = sum / src->channels;
        }
    }
    
    return PSF_SUCCESS;
}

/**
 * @brief Normalize image to [0, 1]
 */
PSF_API PSFError psf_normalize_image(
    PSFImage *image,
    double min_val,
    double max_val)
{
    if (!image) {
        return PSF_ERROR_NULL_POINTER;
    }
    
    size_t total_pixels = image->width * image->height * image->channels;
    
    // Find current min and max if not provided
    if (min_val >= max_val) {
        min_val = DBL_MAX;
        max_val = -DBL_MAX;
        
        for (size_t i = 0; i < total_pixels; i++) {
            if (image->data[i] < min_val) min_val = image->data[i];
            if (image->data[i] > max_val) max_val = image->data[i];
        }
    }
    
    if (max_val - min_val < PSF_EPSILON) {
        // Constant image
        return PSF_SUCCESS;
    }
    
    double scale = 1.0 / (max_val - min_val);
    
#ifdef _OPENMP
    #pragma omp parallel for
#endif
    for (size_t i = 0; i < total_pixels; i++) {
        image->data[i] = (image->data[i] - min_val) * scale;
        image->data[i] = PSF_CLAMP(image->data[i], 0.0, 1.0);
    }
    
    return PSF_SUCCESS;
}

/**
 * @brief Extract image region
 */
PSF_API PSFError psf_extract_roi(
    const PSFImage *src,
    int x,
    int y,
    int width,
    int height,
    PSFImage **roi)
{
    if (!src || !roi) {
        return PSF_ERROR_NULL_POINTER;
    }
    
    // Validate ROI bounds
    if (x < 0 || y < 0 || width <= 0 || height <= 0 ||
        x + width > src->width || y + height > src->height) {
        return PSF_ERROR_INVALID_PARAM;
    }
    
    PSFError err = psf_create_image(width, height, src->channels, roi);
    if (err != PSF_SUCCESS) {
        return err;
    }
    
    // Copy ROI data
    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            for (int c = 0; c < src->channels; c++) {
                int src_idx = ((y + row) * src->width + (x + col)) * src->channels + c;
                int dst_idx = (row * width + col) * src->channels + c;
                (*roi)->data[dst_idx] = src->data[src_idx];
            }
        }
    }
    
    return PSF_SUCCESS;
}

// ============================================================================
// PSF Kernel Operations
// ============================================================================

/**
 * @brief Create PSF kernel
 */
PSF_API PSFError psf_create_kernel(
    int width,
    int height,
    PSFKernel **kernel)
{
    if (!kernel) {
        return PSF_ERROR_NULL_POINTER;
    }
    
    if (width <= 0 || height <= 0) {
        return PSF_ERROR_INVALID_PARAM;
    }
    
    PSFKernel *k = PSF_MALLOC(PSFKernel, 1);
    if (!k) {
        return PSF_ERROR_MEMORY_ALLOCATION;
    }
    
    k->width = width;
    k->height = height;
    k->center_x = width / 2.0;
    k->center_y = height / 2.0;
    k->total_energy = 0.0;
    
    k->data = PSF_CALLOC(double, width * height);
    if (!k->data) {
        PSF_FREE(k);
        return PSF_ERROR_MEMORY_ALLOCATION;
    }
    
    *kernel = k;
    return PSF_SUCCESS;
}

/**
 * @brief Destroy PSF kernel
 */
PSF_API void psf_destroy_kernel(PSFKernel *kernel)
{
    if (!kernel) return;
    
    PSF_FREE(kernel->data);
    PSF_FREE(kernel);
}

/**
 * @brief Copy PSF kernel
 */
PSF_API PSFError psf_copy_kernel(
    const PSFKernel *src,
    PSFKernel **dst)
{
    if (!src || !dst) {
        return PSF_ERROR_NULL_POINTER;
    }
    
    PSFError err = psf_create_kernel(src->width, src->height, dst);
    if (err != PSF_SUCCESS) {
        return err;
    }
    
    memcpy((*dst)->data, src->data, src->width * src->height * sizeof(double));
    (*dst)->center_x = src->center_x;
    (*dst)->center_y = src->center_y;
    (*dst)->total_energy = src->total_energy;
    
    return PSF_SUCCESS;
}

/**
 * @brief Normalize PSF kernel
 */
PSF_API PSFError psf_normalize_kernel(PSFKernel *kernel)
{
    if (!kernel) {
        return PSF_ERROR_NULL_POINTER;
    }
    
    // Compute total energy
    double sum = 0.0;
    int size = kernel->width * kernel->height;
    
    for (int i = 0; i < size; i++) {
        sum += kernel->data[i];
    }
    
    if (sum < PSF_EPSILON) {
        return PSF_ERROR_INVALID_PSF;
    }
    
    // Normalize
    double scale = 1.0 / sum;
    
#ifdef _OPENMP
    #pragma omp parallel for
#endif
    for (int i = 0; i < size; i++) {
        kernel->data[i] *= scale;
    }
    
    kernel->total_energy = 1.0;
    
    return PSF_SUCCESS;
}

/**
 * @brief Resize PSF kernel using bilinear interpolation
 */
PSF_API PSFError psf_resize_kernel(
    const PSFKernel *src,
    int new_width,
    int new_height,
    PSFKernel **dst)
{
    if (!src || !dst) {
        return PSF_ERROR_NULL_POINTER;
    }
    
    if (new_width <= 0 || new_height <= 0) {
        return PSF_ERROR_INVALID_PARAM;
    }
    
    PSFError err = psf_create_kernel(new_width, new_height, dst);
    if (err != PSF_SUCCESS) {
        return err;
    }
    
    double scale_x = (double)src->width / new_width;
    double scale_y = (double)src->height / new_height;
    
#ifdef _OPENMP
    #pragma omp parallel for collapse(2)
#endif
    for (int y = 0; y < new_height; y++) {
        for (int x = 0; x < new_width; x++) {
            // Map to source coordinates
            double src_x = (x + 0.5) * scale_x - 0.5;
            double src_y = (y + 0.5) * scale_y - 0.5;
            
            // Bilinear interpolation
            int x0 = (int)floor(src_x);
            int y0 = (int)floor(src_y);
            int x1 = x0 + 1;
            int y1 = y0 + 1;
            
            double fx = src_x - x0;
            double fy = src_y - y0;
            
            double value = 0.0;
            
            if (x0 >= 0 && x0 < src->width && y0 >= 0 && y0 < src->height) {
                value += src->data[y0 * src->width + x0] * (1 - fx) * (1 - fy);
            }
            if (x1 >= 0 && x1 < src->width && y0 >= 0 && y0 < src->height) {
                value += src->data[y0 * src->width + x1] * fx * (1 - fy);
            }
            if (x0 >= 0 && x0 < src->width && y1 >= 0 && y1 < src->height) {
                value += src->data[y1 * src->width + x0] * (1 - fx) * fy;
            }
            if (x1 >= 0 && x1 < src->width && y1 >= 0 && y1 < src->height) {
                value += src->data[y1 * src->width + x1] * fx * fy;
            }
            
            (*dst)->data[y * new_width + x] = value;
        }
    }
    
    // Update center position
    (*dst)->center_x = src->center_x * new_width / src->width;
    (*dst)->center_y = src->center_y * new_height / src->height;
    
    // Normalize
    psf_normalize_kernel(*dst);
    
    return PSF_SUCCESS;
}

/**
 * @brief Center PSF kernel (find and set center of mass)
 */
PSF_API PSFError psf_center_kernel(PSFKernel *kernel)
{
    if (!kernel) {
        return PSF_ERROR_NULL_POINTER;
    }
    
    // Compute center of mass
    double sum = 0.0;
    double cx = 0.0;
    double cy = 0.0;
    
    for (int y = 0; y < kernel->height; y++) {
        for (int x = 0; x < kernel->width; x++) {
            double value = kernel->data[y * kernel->width + x];
            sum += value;
            cx += x * value;
            cy += y * value;
        }
    }
    
    if (sum < PSF_EPSILON) {
        return PSF_ERROR_INVALID_PSF;
    }
    
    kernel->center_x = cx / sum;
    kernel->center_y = cy / sum;
    kernel->total_energy = sum;
    
    return PSF_SUCCESS;
}

// ============================================================================
// Internal Helper Functions - Math Utilities
// ============================================================================

/**
 * @brief 1D Gaussian function
 */
static double gaussian_1d(double x, double sigma)
{
    double sigma_sq = sigma * sigma;
    return exp(-0.5 * x * x / sigma_sq) / (sigma * sqrt(2.0 * M_PI));
}

/**
 * @brief 2D Gaussian function with rotation
 */
static double gaussian_2d(double x, double y, double sigma_x, double sigma_y, double rotation)
{
    double cos_r = cos(rotation);
    double sin_r = sin(rotation);
    
    // Rotate coordinates
    double x_rot = x * cos_r + y * sin_r;
    double y_rot = -x * sin_r + y * cos_r;
    
    double sigma_x_sq = sigma_x * sigma_x;
    double sigma_y_sq = sigma_y * sigma_y;
    
    double exponent = -0.5 * (x_rot * x_rot / sigma_x_sq + y_rot * y_rot / sigma_y_sq);
    
    return exp(exponent) / (2.0 * M_PI * sigma_x * sigma_y);
}

/**
 * @brief 2D Moffat function
 */
static double moffat_2d(double x, double y, double alpha, double beta)
{
    double r_sq = x * x + y * y;
    double alpha_sq = alpha * alpha;
    
    return (beta - 1.0) / (M_PI * alpha_sq) * 
           pow(1.0 + r_sq / alpha_sq, -beta);
}

/**
 * @brief Airy disk function
 */
static double airy_disk(double r, double radius)
{
    if (r < PSF_EPSILON) {
        return 1.0;
    }
    
    double x = M_PI * r / radius;
    
    // First order Bessel function J1(x) approximation
    double j1;
    if (x < 3.0) {
        double x2 = x * x;
        j1 = x * (0.5 - x2 * (0.0625 - x2 * 0.00260416666667));
    } else {
        double x_inv = 1.0 / x;
        j1 = sqrt(2.0 / (M_PI * x)) * 
             (sin(x - 0.75 * M_PI) * (1.0 - 0.1875 * x_inv * x_inv) +
              cos(x - 0.75 * M_PI) * 0.125 * x_inv);
    }
    
    double airy = 2.0 * j1 / x;
    return airy * airy;
}

/**
 * @brief Compute image gradient
 */
static void compute_gradient(
    const double *data,
    int width,
    int height,
    double *grad_x,
    double *grad_y)
{
    // Sobel operator
    const int sobel_x[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };
    
    const int sobel_y[3][3] = {
        {-1, -2, -1},
        { 0,  0,  0},
        { 1,  2,  1}
    };
    
#ifdef _OPENMP
    #pragma omp parallel for collapse(2)
#endif
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            double gx = 0.0;
            double gy = 0.0;
            
            for (int ky = -1; ky <= 1; ky++) {
                for (int kx = -1; kx <= 1; kx++) {
                    double pixel = data[(y + ky) * width + (x + kx)];
                    gx += pixel * sobel_x[ky + 1][kx + 1];
                    gy += pixel * sobel_y[ky + 1][kx + 1];
                }
            }
            
            int idx = y * width + x;
            grad_x[idx] = gx / 8.0;
            grad_y[idx] = gy / 8.0;
        }
    }
}

/**
 * @brief Apply Gaussian filter
 */
static void apply_gaussian_filter(
    const double *input,
    double *output,
    int width,
    int height,
    double sigma)
{
    // Create Gaussian kernel
    int kernel_size = (int)(6 * sigma + 1);
    if (kernel_size % 2 == 0) kernel_size++;
    int kernel_radius = kernel_size / 2;
    
    double *kernel = PSF_MALLOC(double, kernel_size);
    double sum = 0.0;
    
    for (int i = 0; i < kernel_size; i++) {
        int x = i - kernel_radius;
        kernel[i] = gaussian_1d(x, sigma);
        sum += kernel[i];
    }
    
    // Normalize kernel
    for (int i = 0; i < kernel_size; i++) {
        kernel[i] /= sum;
    }
    
    // Temporary buffer for horizontal pass
    double *temp = PSF_MALLOC(double, width * height);
    
    // Horizontal pass
#ifdef _OPENMP
    #pragma omp parallel for collapse(2)
#endif
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            double value = 0.0;
            
            for (int k = 0; k < kernel_size; k++) {
                int xx = x + k - kernel_radius;
                xx = PSF_CLAMP(xx, 0, width - 1);
                value += input[y * width + xx] * kernel[k];
            }
            
            temp[y * width + x] = value;
        }
    }
    
    // Vertical pass
#ifdef _OPENMP
    #pragma omp parallel for collapse(2)
#endif
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            double value = 0.0;
            
            for (int k = 0; k < kernel_size; k++) {
                int yy = y + k - kernel_radius;
                yy = PSF_CLAMP(yy, 0, height - 1);
                value += temp[yy * width + x] * kernel[k];
            }
            
            output[y * width + x] = value;
        }
    }
    
    PSF_FREE(kernel);
    PSF_FREE(temp);
}
// ============================================================================
// Edge Detection
// ============================================================================

/**
 * @brief Sobel edge detection filter
 */
static void sobel_filter(
    const double *input,
    double *grad_x,
    double *grad_y,
    int width,
    int height)
{
    const int sobel_x[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };
    
    const int sobel_y[3][3] = {
        {-1, -2, -1},
        { 0,  0,  0},
        { 1,  2,  1}
    };
    
#ifdef _OPENMP
    #pragma omp parallel for collapse(2)
#endif
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            double gx = 0.0;
            double gy = 0.0;
            
            for (int ky = -1; ky <= 1; ky++) {
                for (int kx = -1; kx <= 1; kx++) {
                    double pixel = input[(y + ky) * width + (x + kx)];
                    gx += pixel * sobel_x[ky + 1][kx + 1];
                    gy += pixel * sobel_y[ky + 1][kx + 1];
                }
            }
            
            int idx = y * width + x;
            grad_x[idx] = gx;
            grad_y[idx] = gy;
        }
    }
}

/**
 * @brief Non-maximum suppression for edge detection
 */
static void non_maximum_suppression(
    const double *grad_mag,
    const double *grad_dir,
    double *output,
    int width,
    int height)
{
#ifdef _OPENMP
    #pragma omp parallel for collapse(2)
#endif
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            int idx = y * width + x;
            double magnitude = grad_mag[idx];
            double angle = grad_dir[idx];
            
            // Quantize angle to 0, 45, 90, 135 degrees
            angle = fmod(angle + M_PI, M_PI);
            int direction = (int)(4.0 * angle / M_PI + 0.5) % 4;
            
            double neighbor1 = 0.0, neighbor2 = 0.0;
            
            switch (direction) {
                case 0: // Horizontal (0 degrees)
                    neighbor1 = grad_mag[y * width + (x - 1)];
                    neighbor2 = grad_mag[y * width + (x + 1)];
                    break;
                case 1: // Diagonal (45 degrees)
                    neighbor1 = grad_mag[(y - 1) * width + (x + 1)];
                    neighbor2 = grad_mag[(y + 1) * width + (x - 1)];
                    break;
                case 2: // Vertical (90 degrees)
                    neighbor1 = grad_mag[(y - 1) * width + x];
                    neighbor2 = grad_mag[(y + 1) * width + x];
                    break;
                case 3: // Diagonal (135 degrees)
                    neighbor1 = grad_mag[(y - 1) * width + (x - 1)];
                    neighbor2 = grad_mag[(y + 1) * width + (x + 1)];
                    break;
            }
            
            // Suppress if not local maximum
            if (magnitude >= neighbor1 && magnitude >= neighbor2) {
                output[idx] = magnitude;
            } else {
                output[idx] = 0.0;
            }
        }
    }
}

/**
 * @brief Hysteresis thresholding for edge detection
 */
static void hysteresis_thresholding(
    double *edges,
    int width,
    int height,
    double low_threshold,
    double high_threshold)
{
    // Mark strong edges
    bool *strong = PSF_CALLOC(bool, width * height);
    bool *weak = PSF_CALLOC(bool, width * height);
    
    for (int i = 0; i < width * height; i++) {
        if (edges[i] >= high_threshold) {
            strong[i] = true;
            edges[i] = 1.0;
        } else if (edges[i] >= low_threshold) {
            weak[i] = true;
        } else {
            edges[i] = 0.0;
        }
    }
    
    // Connect weak edges to strong edges
    bool changed = true;
    while (changed) {
        changed = false;
        
        for (int y = 1; y < height - 1; y++) {
            for (int x = 1; x < width - 1; x++) {
                int idx = y * width + x;
                
                if (weak[idx] && !strong[idx]) {
                    // Check 8-connected neighbors
                    bool has_strong_neighbor = false;
                    
                    for (int dy = -1; dy <= 1; dy++) {
                        for (int dx = -1; dx <= 1; dx++) {
                            if (dx == 0 && dy == 0) continue;
                            
                            int neighbor_idx = (y + dy) * width + (x + dx);
                            if (strong[neighbor_idx]) {
                                has_strong_neighbor = true;
                                break;
                            }
                        }
                        if (has_strong_neighbor) break;
                    }
                    
                    if (has_strong_neighbor) {
                        strong[idx] = true;
                        edges[idx] = 1.0;
                        changed = true;
                    }
                }
            }
        }
    }
    
    // Remove weak edges not connected to strong edges
    for (int i = 0; i < width * height; i++) {
        if (weak[i] && !strong[i]) {
            edges[i] = 0.0;
        }
    }
    
    PSF_FREE(strong);
    PSF_FREE(weak);
}

/**
 * @brief Canny edge detection
 */
static PSFError canny_edge_detection(
    const PSFImage *image,
    double low_threshold,
    double high_threshold,
    double sigma,
    double **edge_map)
{
    if (!image || !edge_map) {
        return PSF_ERROR_NULL_POINTER;
    }
    
    int width = image->width;
    int height = image->height;
    int size = width * height;
    
    // Convert to grayscale if needed
    PSFImage *gray = NULL;
    PSFError err = psf_convert_to_grayscale(image, &gray);
    if (err != PSF_SUCCESS) {
        return err;
    }
    
    // Apply Gaussian smoothing
    double *smoothed = PSF_MALLOC(double, size);
    apply_gaussian_filter(gray->data, smoothed, width, height, sigma);
    
    // Compute gradients
    double *grad_x = PSF_CALLOC(double, size);
    double *grad_y = PSF_CALLOC(double, size);
    sobel_filter(smoothed, grad_x, grad_y, width, height);
    
    // Compute gradient magnitude and direction
    double *grad_mag = PSF_MALLOC(double, size);
    double *grad_dir = PSF_MALLOC(double, size);
    
#ifdef _OPENMP
    #pragma omp parallel for
#endif
    for (int i = 0; i < size; i++) {
        grad_mag[i] = sqrt(grad_x[i] * grad_x[i] + grad_y[i] * grad_y[i]);
        grad_dir[i] = atan2(grad_y[i], grad_x[i]);
    }
    
    // Non-maximum suppression
    double *nms = PSF_CALLOC(double, size);
    non_maximum_suppression(grad_mag, grad_dir, nms, width, height);
    
    // Hysteresis thresholding
    hysteresis_thresholding(nms, width, height, low_threshold, high_threshold);
    
    *edge_map = nms;
    
    // Cleanup
    psf_destroy_image(gray);
    PSF_FREE(smoothed);
    PSF_FREE(grad_x);
    PSF_FREE(grad_y);
    PSF_FREE(grad_mag);
    PSF_FREE(grad_dir);
    
    return PSF_SUCCESS;
}

/**
 * @brief Find connected edge segments
 */
static PSFError find_edge_segments(
    const double *edge_map,
    int width,
    int height,
    int min_length,
    PSFEdge ***edges,
    int *num_edges)
{
    if (!edge_map || !edges || !num_edges) {
        return PSF_ERROR_NULL_POINTER;
    }
    
    int size = width * height;
    bool *visited = PSF_CALLOC(bool, size);
    
    // Temporary storage for edge segments
    PSFEdge **temp_edges = PSF_MALLOC(PSFEdge*, 1000);
    int edge_count = 0;
    int max_edges = 1000;
    
    // Find edge segments using connected component analysis
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = y * width + x;
            
            if (edge_map[idx] > 0.5 && !visited[idx]) {
                // Start new edge segment
                PSFPoint2D *points = PSF_MALLOC(PSFPoint2D, size);
                int num_points = 0;
                
                // BFS to find connected edge pixels
                int *queue_x = PSF_MALLOC(int, size);
                int *queue_y = PSF_MALLOC(int, size);
                int queue_start = 0;
                int queue_end = 0;
                
                queue_x[queue_end] = x;
                queue_y[queue_end] = y;
                queue_end++;
                visited[idx] = true;
                
                while (queue_start < queue_end) {
                    int cx = queue_x[queue_start];
                    int cy = queue_y[queue_start];
                    queue_start++;
                    
                    points[num_points].x = cx;
                    points[num_points].y = cy;
                    num_points++;
                    
                    // Check 8-connected neighbors
                    for (int dy = -1; dy <= 1; dy++) {
                        for (int dx = -1; dx <= 1; dx++) {
                            if (dx == 0 && dy == 0) continue;
                            
                            int nx = cx + dx;
                            int ny = cy + dy;
                            
                            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                                int nidx = ny * width + nx;
                                
                                if (edge_map[nidx] > 0.5 && !visited[nidx]) {
                                    queue_x[queue_end] = nx;
                                    queue_y[queue_end] = ny;
                                    queue_end++;
                                    visited[nidx] = true;
                                }
                            }
                        }
                    }
                }
                
                PSF_FREE(queue_x);
                PSF_FREE(queue_y);
                
                // Check if edge segment is long enough
                if (num_points >= min_length) {
                    // Fit line to edge points to determine angle
                    double sum_x = 0.0, sum_y = 0.0;
                    double sum_xx = 0.0, sum_xy = 0.0;
                    
                    for (int i = 0; i < num_points; i++) {
                        sum_x += points[i].x;
                        sum_y += points[i].y;
                        sum_xx += points[i].x * points[i].x;
                        sum_xy += points[i].x * points[i].y;
                    }
                    
                    double mean_x = sum_x / num_points;
                    double mean_y = sum_y / num_points;
                    
                    double slope = (sum_xy - num_points * mean_x * mean_y) /
                                  (sum_xx - num_points * mean_x * mean_x + PSF_EPSILON);
                    double angle = atan(slope);
                    
                    // Compute edge strength (average gradient magnitude)
                    double strength = 0.0;
                    for (int i = 0; i < num_points; i++) {
                        int px = (int)points[i].x;
                        int py = (int)points[i].y;
                        strength += edge_map[py * width + px];
                    }
                    strength /= num_points;
                    
                    // Create edge structure
                    PSFEdge *edge = PSF_MALLOC(PSFEdge, 1);
                    edge->points = PSF_REALLOC(points, PSFPoint2D, num_points);
                    edge->num_points = num_points;
                    edge->angle = angle;
                    edge->strength = strength;
                    edge->sharpness = 0.0; // Will be computed later
                    
                    // Add to edge list
                    if (edge_count >= max_edges) {
                        max_edges *= 2;
                        temp_edges = PSF_REALLOC(temp_edges, PSFEdge*, max_edges);
                    }
                    temp_edges[edge_count++] = edge;
                } else {
                    PSF_FREE(points);
                }
            }
        }
    }
    
    PSF_FREE(visited);
    
    if (edge_count == 0) {
        PSF_FREE(temp_edges);
        return PSF_ERROR_NO_EDGES_FOUND;
    }
    
    // Copy to output
    *edges = PSF_REALLOC(temp_edges, PSFEdge*, edge_count);
    *num_edges = edge_count;
    
    return PSF_SUCCESS;
}

/**
 * @brief Detect edges in image
 */
PSF_API PSFError psf_detect_edges(
    const PSFImage *image,
    double threshold,
    PSFEdge ***edges,
    int *num_edges)
{
    if (!image || !edges || !num_edges) {
        return PSF_ERROR_NULL_POINTER;
    }
    
    // Perform Canny edge detection
    double *edge_map = NULL;
    double low_threshold = threshold * 0.5;
    double high_threshold = threshold;
    double sigma = 1.4;
    
    PSFError err = canny_edge_detection(image, low_threshold, high_threshold, 
                                        sigma, &edge_map);
    if (err != PSF_SUCCESS) {
        return err;
    }
    
    // Find edge segments
    int min_length = 20; // Minimum edge length in pixels
    err = find_edge_segments(edge_map, image->width, image->height,
                            min_length, edges, num_edges);
    
    PSF_FREE(edge_map);
    
    return err;
}

/**
 * @brief Destroy edge structure
 */
PSF_API void psf_destroy_edge(PSFEdge *edge)
{
    if (!edge) return;
    
    PSF_FREE(edge->points);
    PSF_FREE(edge);
}

/**
 * @brief Destroy edge array
 */
PSF_API void psf_destroy_edges(PSFEdge **edges, int num_edges)
{
    if (!edges) return;
    
    for (int i = 0; i < num_edges; i++) {
        psf_destroy_edge(edges[i]);
    }
    
    PSF_FREE(edges);
}

// ============================================================================
// Edge Spread Function (ESF) Extraction
// ============================================================================

/**
 * @brief Extract Edge Spread Function from edge
 */
PSF_API PSFError psf_extract_esf(
    const PSFImage *image,
    const PSFEdge *edge,
    int roi_size,
    int oversample_factor,
    PSFEdgeSpreadFunction **esf)
{
    if (!image || !edge || !esf) {
        return PSF_ERROR_NULL_POINTER;
    }
    
    if (roi_size <= 0 || oversample_factor <= 0) {
        return PSF_ERROR_INVALID_PARAM;
    }
    
    // Convert to grayscale if needed
    PSFImage *gray = NULL;
    PSFError err = psf_convert_to_grayscale(image, &gray);
    if (err != PSF_SUCCESS) {
        return err;
    }
    
    // Compute edge normal direction
    double edge_angle = edge->angle;
    double normal_angle = edge_angle + M_PI / 2.0;
    double nx = cos(normal_angle);
    double ny = sin(normal_angle);
    
    // Allocate ESF structure
    PSFEdgeSpreadFunction *esf_result = PSF_MALLOC(PSFEdgeSpreadFunction, 1);
    if (!esf_result) {
        psf_destroy_image(gray);
        return PSF_ERROR_MEMORY_ALLOCATION;
    }
    
    int esf_length = roi_size * oversample_factor;
    esf_result->data = PSF_CALLOC(double, esf_length);
    esf_result->length = esf_length;
    esf_result->pixel_spacing = 1.0 / oversample_factor;
    
    int *counts = PSF_CALLOC(int, esf_length);
    
    // Sample along edge normal for each edge point
    for (int i = 0; i < edge->num_points; i++) {
        double cx = edge->points[i].x;
        double cy = edge->points[i].y;
        
        // Sample along normal direction
        for (int j = 0; j < esf_length; j++) {
            double offset = (j - esf_length / 2.0) / oversample_factor;
            double sx = cx + offset * nx;
            double sy = cy + offset * ny;
            
            // Bilinear interpolation
            int x0 = (int)floor(sx);
            int y0 = (int)floor(sy);
            
            if (x0 >= 0 && x0 < gray->width - 1 && 
                y0 >= 0 && y0 < gray->height - 1) {
                
                double fx = sx - x0;
                double fy = sy - y0;
                
                double v00 = gray->data[y0 * gray->width + x0];
                double v10 = gray->data[y0 * gray->width + (x0 + 1)];
                double v01 = gray->data[(y0 + 1) * gray->width + x0];
                double v11 = gray->data[(y0 + 1) * gray->width + (x0 + 1)];
                
                double value = v00 * (1 - fx) * (1 - fy) +
                              v10 * fx * (1 - fy) +
                              v01 * (1 - fx) * fy +
                              v11 * fx * fy;
                
                esf_result->data[j] += value;
                counts[j]++;
            }
        }
    }
    
    // Average the samples
    for (int i = 0; i < esf_length; i++) {
        if (counts[i] > 0) {
            esf_result->data[i] /= counts[i];
        }
    }
    
    // Find edge position (maximum gradient)
    double max_grad = 0.0;
    int max_idx = esf_length / 2;
    
    for (int i = 1; i < esf_length - 1; i++) {
        double grad = esf_result->data[i + 1] - esf_result->data[i - 1];
        if (fabs(grad) > max_grad) {
            max_grad = fabs(grad);
            max_idx = i;
        }
    }
    
    esf_result->edge_position = max_idx * esf_result->pixel_spacing;
    
    PSF_FREE(counts);
    psf_destroy_image(gray);
    
    *esf = esf_result;
    
    return PSF_SUCCESS;
}

/**
 * @brief Destroy ESF structure
 */
PSF_API void psf_destroy_esf(PSFEdgeSpreadFunction *esf)
{
    if (!esf) return;
    
    PSF_FREE(esf->data);
    PSF_FREE(esf);
}

// ============================================================================
// Line Spread Function (LSF) Computation
// ============================================================================

/**
 * @brief Compute Line Spread Function from ESF (derivative)
 */
PSF_API PSFError psf_esf_to_lsf(
    const PSFEdgeSpreadFunction *esf,
    PSFLineSpreadFunction **lsf)
{
    if (!esf || !lsf) {
        return PSF_ERROR_NULL_POINTER;
    }
    
    PSFLineSpreadFunction *lsf_result = PSF_MALLOC(PSFLineSpreadFunction, 1);
    if (!lsf_result) {
        return PSF_ERROR_MEMORY_ALLOCATION;
    }
    
    lsf_result->length = esf->length - 2;
    lsf_result->pixel_spacing = esf->pixel_spacing;
    lsf_result->data = PSF_MALLOC(double, lsf_result->length);
    
    // Compute derivative using central differences
    for (int i = 0; i < lsf_result->length; i++) {
        lsf_result->data[i] = (esf->data[i + 2] - esf->data[i]) / 
                             (2.0 * esf->pixel_spacing);
    }
    
    // Find peak position
    double max_value = -DBL_MAX;
    int max_idx = 0;
    
    for (int i = 0; i < lsf_result->length; i++) {
        if (lsf_result->data[i] > max_value) {
            max_value = lsf_result->data[i];
            max_idx = i;
        }
    }
    
    lsf_result->peak_position = max_idx * lsf_result->pixel_spacing;
    
    // Compute FWHM (Full Width at Half Maximum)
    double half_max = max_value / 2.0;
    int left_idx = max_idx;
    int right_idx = max_idx;
    
    // Find left half-maximum point
    while (left_idx > 0 && lsf_result->data[left_idx] > half_max) {
        left_idx--;
    }
    
    // Find right half-maximum point
    while (right_idx < lsf_result->length - 1 && 
           lsf_result->data[right_idx] > half_max) {
        right_idx++;
    }
    
    lsf_result->fwhm = (right_idx - left_idx) * lsf_result->pixel_spacing;
    
    *lsf = lsf_result;
    
    return PSF_SUCCESS;
}

/**
 * @brief Destroy LSF structure
 */
PSF_API void psf_destroy_lsf(PSFLineSpreadFunction *lsf)
{
    if (!lsf) return;
    
    PSF_FREE(lsf->data);
    PSF_FREE(lsf);
}

/**
 * @brief Compute PSF from two orthogonal LSFs
 */
PSF_API PSFError psf_lsf_to_psf(
    const PSFLineSpreadFunction *lsf_x,
    const PSFLineSpreadFunction *lsf_y,
    PSFKernel **psf)
{
    if (!lsf_x || !lsf_y || !psf) {
        return PSF_ERROR_NULL_POINTER;
    }
    
    // Create PSF kernel
    int width = lsf_x->length;
    int height = lsf_y->length;
    
    PSFError err = psf_create_kernel(width, height, psf);
    if (err != PSF_SUCCESS) {
        return err;
    }
    
    // Compute 2D PSF as outer product of LSFs
#ifdef _OPENMP
    #pragma omp parallel for collapse(2)
#endif
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            (*psf)->data[y * width + x] = lsf_x->data[x] * lsf_y->data[y];
        }
    }
    
    // Normalize PSF
    psf_normalize_kernel(*psf);
    psf_center_kernel(*psf);
    
    return PSF_SUCCESS;
}

/**
 * @brief Estimate PSF from edges (complete pipeline)
 */
PSF_API PSFError psf_estimate_from_edges(
    const PSFImage *image,
    const PSFEstimationConfig *config,
    PSFKernel **psf)
{
    if (!image || !psf) {
        return PSF_ERROR_NULL_POINTER;
    }
    
    // Use default config if not provided
    PSFEstimationConfig default_config;
    if (!config) {
        default_config.edge_threshold = 0.1;
        default_config.edge_roi_size = 50;
        default_config.edge_oversample_factor = 4;
        default_config.psf_kernel_size = 15;
        config = &default_config;
    }
    
    // Detect edges
    PSFEdge **edges = NULL;
    int num_edges = 0;
    
    PSFError err = psf_detect_edges(image, config->edge_threshold, 
                                    &edges, &num_edges);
    if (err != PSF_SUCCESS) {
        return err;
    }
    
    if (num_edges == 0) {
        return PSF_ERROR_NO_EDGES_FOUND;
    }
    
    // Find best horizontal and vertical edges
    PSFEdge *best_h_edge = NULL;
    PSFEdge *best_v_edge = NULL;
    double best_h_score = 0.0;
    double best_v_score = 0.0;
    
    for (int i = 0; i < num_edges; i++) {
        double angle = fabs(edges[i]->angle);
        double score = edges[i]->strength * edges[i]->num_points;
        
        // Check if horizontal (angle near 0 or π)
        if (angle < M_PI / 4.0 || angle > 3.0 * M_PI / 4.0) {
            if (score > best_h_score) {
                best_h_score = score;
                best_h_edge = edges[i];
            }
        }
        // Check if vertical (angle near π/2)
        else {
            if (score > best_v_score) {
                best_v_score = score;
                best_v_edge = edges[i];
            }
        }
    }
    
    if (!best_h_edge || !best_v_edge) {
        psf_destroy_edges(edges, num_edges);
        return PSF_ERROR_INSUFFICIENT_DATA;
    }
    
    // Extract ESF from edges
    PSFEdgeSpreadFunction *esf_x = NULL;
    PSFEdgeSpreadFunction *esf_y = NULL;
    
    err = psf_extract_esf(image, best_h_edge, config->edge_roi_size,
                         config->edge_oversample_factor, &esf_x);
    if (err != PSF_SUCCESS) {
        psf_destroy_edges(edges, num_edges);
        return err;
    }
    
    err = psf_extract_esf(image, best_v_edge, config->edge_roi_size,
                         config->edge_oversample_factor, &esf_y);
    if (err != PSF_SUCCESS) {
        psf_destroy_esf(esf_x);
        psf_destroy_edges(edges, num_edges);
        return err;
    }
    
    // Compute LSF from ESF
    PSFLineSpreadFunction *lsf_x = NULL;
    PSFLineSpreadFunction *lsf_y = NULL;
    
    err = psf_esf_to_lsf(esf_x, &lsf_x);
    if (err != PSF_SUCCESS) {
        psf_destroy_esf(esf_x);
        psf_destroy_esf(esf_y);
        psf_destroy_edges(edges, num_edges);
        return err;
    }
    
    err = psf_esf_to_lsf(esf_y, &lsf_y);
    if (err != PSF_SUCCESS) {
        psf_destroy_lsf(lsf_x);
        psf_destroy_esf(esf_x);
        psf_destroy_esf(esf_y);
        psf_destroy_edges(edges, num_edges);
        return err;
    }
    
    // Compute PSF from LSFs
    err = psf_lsf_to_psf(lsf_x, lsf_y, psf);
    
    // Cleanup
    psf_destroy_lsf(lsf_x);
    psf_destroy_lsf(lsf_y);
    psf_destroy_esf(esf_x);
    psf_destroy_esf(esf_y);
    psf_destroy_edges(edges, num_edges);
    
    return err;
}
// ============================================================================
// Star Detection
// ============================================================================

/**
 * @brief Compute local background using median filter
 */
static void compute_local_background(
    const double *image,
    int width,
    int height,
    int window_size,
    double *background)
{
    int half_window = window_size / 2;
    double *window_values = PSF_MALLOC(double, window_size * window_size);
    
#ifdef _OPENMP
    #pragma omp parallel for collapse(2) private(window_values)
#endif
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int count = 0;
            
            // Collect window values
            for (int wy = -half_window; wy <= half_window; wy++) {
                for (int wx = -half_window; wx <= half_window; wx++) {
                    int nx = x + wx;
                    int ny = y + wy;
                    
                    if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                        window_values[count++] = image[ny * width + nx];
                    }
                }
            }
            
            // Compute median
            if (count > 0) {
                // Simple selection sort for median
                for (int i = 0; i < count - 1; i++) {
                    for (int j = i + 1; j < count; j++) {
                        if (window_values[i] > window_values[j]) {
                            double temp = window_values[i];
                            window_values[i] = window_values[j];
                            window_values[j] = temp;
                        }
                    }
                }
                background[y * width + x] = window_values[count / 2];
            } else {
                background[y * width + x] = 0.0;
            }
        }
    }
    
    PSF_FREE(window_values);
}

/**
 * @brief Find local maxima in image
 */
static PSFError find_local_maxima(
    const double *image,
    int width,
    int height,
    double threshold,
    int min_separation,
    PSFPoint2D **maxima,
    int *num_maxima)
{
    if (!image || !maxima || !num_maxima) {
        return PSF_ERROR_NULL_POINTER;
    }
    
    // Temporary storage
    PSFPoint2D *temp_maxima = PSF_MALLOC(PSFPoint2D, width * height);
    int count = 0;
    
    // Find local maxima
    for (int y = min_separation; y < height - min_separation; y++) {
        for (int x = min_separation; x < width - min_separation; x++) {
            double center_value = image[y * width + x];
            
            if (center_value < threshold) {
                continue;
            }
            
            // Check if local maximum
            bool is_maximum = true;
            
            for (int dy = -min_separation; dy <= min_separation && is_maximum; dy++) {
                for (int dx = -min_separation; dx <= min_separation && is_maximum; dx++) {
                    if (dx == 0 && dy == 0) continue;
                    
                    int nx = x + dx;
                    int ny = y + dy;
                    
                    if (image[ny * width + nx] > center_value) {
                        is_maximum = false;
                    }
                }
            }
            
            if (is_maximum) {
                temp_maxima[count].x = x;
                temp_maxima[count].y = y;
                count++;
            }
        }
    }
    
    if (count == 0) {
        PSF_FREE(temp_maxima);
        return PSF_ERROR_NO_STARS_FOUND;
    }
    
    *maxima = PSF_REALLOC(temp_maxima, PSFPoint2D, count);
    *num_maxima = count;
    
    return PSF_SUCCESS;
}

/**
 * @brief Compute centroid of star region
 */
static void compute_centroid(
    const double *image,
    int width,
    int height,
    int cx,
    int cy,
    int box_size,
    double *centroid_x,
    double *centroid_y)
{
    int half_box = box_size / 2;
    double sum = 0.0;
    double sum_x = 0.0;
    double sum_y = 0.0;
    
    for (int dy = -half_box; dy <= half_box; dy++) {
        for (int dx = -half_box; dx <= half_box; dx++) {
            int x = cx + dx;
            int y = cy + dy;
            
            if (x >= 0 && x < width && y >= 0 && y < height) {
                double value = image[y * width + x];
                sum += value;
                sum_x += x * value;
                sum_y += y * value;
            }
        }
    }
    
    if (sum > PSF_EPSILON) {
        *centroid_x = sum_x / sum;
        *centroid_y = sum_y / sum;
    } else {
        *centroid_x = cx;
        *centroid_y = cy;
    }
}

/**
 * @brief Compute star properties (FWHM, ellipticity, etc.)
 */
static void compute_star_properties(
    const double *image,
    int width,
    int height,
    double cx,
    double cy,
    int box_size,
    PSFStar *star)
{
    int half_box = box_size / 2;
    
    // Compute moments
    double m00 = 0.0;  // Total intensity
    double m10 = 0.0, m01 = 0.0;  // First moments
    double m20 = 0.0, m11 = 0.0, m02 = 0.0;  // Second moments
    
    double background = 0.0;
    int bg_count = 0;
    
    // Estimate background from corners
    for (int dy = -half_box; dy <= half_box; dy += half_box * 2) {
        for (int dx = -half_box; dx <= half_box; dx += half_box * 2) {
            int x = (int)(cx + dx);
            int y = (int)(cy + dy);
            
            if (x >= 0 && x < width && y >= 0 && y < height) {
                background += image[y * width + x];
                bg_count++;
            }
        }
    }
    
    if (bg_count > 0) {
        background /= bg_count;
    }
    
    // Compute moments
    for (int dy = -half_box; dy <= half_box; dy++) {
        for (int dx = -half_box; dx <= half_box; dx++) {
            int x = (int)(cx + dx);
            int y = (int)(cy + dy);
            
            if (x >= 0 && x < width && y >= 0 && y < height) {
                double value = image[y * width + x] - background;
                if (value < 0) value = 0;
                
                m00 += value;
                m10 += dx * value;
                m01 += dy * value;
                m20 += dx * dx * value;
                m11 += dx * dy * value;
                m02 += dy * dy * value;
            }
        }
    }
    
    star->brightness = m00;
    
    if (m00 < PSF_EPSILON) {
        star->fwhm = 0.0;
        star->ellipticity = 0.0;
        star->angle = 0.0;
        star->snr = 0.0;
        return;
    }
    
    // Central moments
    double mu20 = m20 / m00 - (m10 / m00) * (m10 / m00);
    double mu11 = m11 / m00 - (m10 / m00) * (m01 / m00);
    double mu02 = m02 / m00 - (m01 / m00) * (m01 / m00);
    
    // Compute eigenvalues of covariance matrix
    double trace = mu20 + mu02;
    double det = mu20 * mu02 - mu11 * mu11;
    
    double lambda1 = 0.5 * (trace + sqrt(trace * trace - 4 * det));
    double lambda2 = 0.5 * (trace - sqrt(trace * trace - 4 * det));
    
    // FWHM (assuming Gaussian)
    double sigma_major = sqrt(lambda1);
    double sigma_minor = sqrt(lambda2);
    star->fwhm = 2.355 * sqrt(sigma_major * sigma_minor);  // Geometric mean
    
    // Ellipticity
    if (lambda1 > PSF_EPSILON) {
        star->ellipticity = 1.0 - lambda2 / lambda1;
    } else {
        star->ellipticity = 0.0;
    }
    
    // Orientation angle
    star->angle = 0.5 * atan2(2.0 * mu11, mu20 - mu02);
    
    // Signal-to-noise ratio
    double noise_variance = 0.0;
    int noise_count = 0;
    
    for (int dy = -half_box; dy <= half_box; dy++) {
        for (int dx = -half_box; dx <= half_box; dx++) {
            int x = (int)(cx + dx);
            int y = (int)(cy + dy);
            
            if (x >= 0 && x < width && y >= 0 && y < height) {
                double value = image[y * width + x] - background;
                if (value < 0) {
                    noise_variance += value * value;
                    noise_count++;
                }
            }
        }
    }
    
    if (noise_count > 0) {
        double noise_std = sqrt(noise_variance / noise_count);
        star->snr = m00 / (noise_std * sqrt(box_size * box_size) + PSF_EPSILON);
    } else {
        star->snr = 0.0;
    }
}

/**
 * @brief Detect stars in image
 */
PSF_API PSFError psf_detect_stars(
    const PSFImage *image,
    const PSFEstimationConfig *config,
    PSFStar **stars,
    int *num_stars)
{
    if (!image || !stars || !num_stars) {
        return PSF_ERROR_NULL_POINTER;
    }
    
    // Convert to grayscale if needed
    PSFImage *gray = NULL;
    PSFError err = psf_convert_to_grayscale(image, &gray);
    if (err != PSF_SUCCESS) {
        return err;
    }
    
    int width = gray->width;
    int height = gray->height;
    
    // Use default config if not provided
    PSFEstimationConfig default_config;
    if (!config) {
        default_config.star_detection_threshold = 0.1;
        default_config.star_min_size = 3;
        default_config.star_max_size = 50;
        default_config.star_roundness_threshold = 0.5;
        config = &default_config;
    }
    
    // Compute local background
    double *background = PSF_MALLOC(double, width * height);
    compute_local_background(gray->data, width, height, 15, background);
    
    // Subtract background
    double *subtracted = PSF_MALLOC(double, width * height);
    
#ifdef _OPENMP
    #pragma omp parallel for
#endif
    for (int i = 0; i < width * height; i++) {
        subtracted[i] = gray->data[i] - background[i];
        if (subtracted[i] < 0) subtracted[i] = 0;
    }
    
    // Find local maxima
    PSFPoint2D *maxima = NULL;
    int num_maxima = 0;
    
    err = find_local_maxima(subtracted, width, height,
                           config->star_detection_threshold,
                           config->star_min_size,
                           &maxima, &num_maxima);
    
    if (err != PSF_SUCCESS) {
        PSF_FREE(background);
        PSF_FREE(subtracted);
        psf_destroy_image(gray);
        return err;
    }
    
    // Analyze each candidate star
    PSFStar *temp_stars = PSF_MALLOC(PSFStar, num_maxima);
    int star_count = 0;
    
    for (int i = 0; i < num_maxima; i++) {
        PSFStar star;
        
        // Refine centroid
        compute_centroid(subtracted, width, height,
                        (int)maxima[i].x, (int)maxima[i].y,
                        config->star_max_size,
                        &star.center.x, &star.center.y);
        
        // Compute star properties
        compute_star_properties(subtracted, width, height,
                               star.center.x, star.center.y,
                               config->star_max_size,
                               &star);
        
        // Filter stars based on criteria
        bool is_valid = true;
        
        // Check FWHM range
        if (star.fwhm < config->star_min_size || 
            star.fwhm > config->star_max_size) {
            is_valid = false;
        }
        
        // Check roundness (1 - ellipticity)
        double roundness = 1.0 - star.ellipticity;
        if (roundness < config->star_roundness_threshold) {
            is_valid = false;
        }
        
        // Check SNR
        if (star.snr < 5.0) {
            is_valid = false;
        }
        
        // Check if too close to edge
        int margin = config->star_max_size;
        if (star.center.x < margin || star.center.x > width - margin ||
            star.center.y < margin || star.center.y > height - margin) {
            is_valid = false;
        }
        
        if (is_valid) {
            temp_stars[star_count++] = star;
        }
    }
    
    PSF_FREE(maxima);
    PSF_FREE(background);
    PSF_FREE(subtracted);
    psf_destroy_image(gray);
    
    if (star_count == 0) {
        PSF_FREE(temp_stars);
        return PSF_ERROR_NO_STARS_FOUND;
    }
    
    // Sort stars by brightness (descending)
    for (int i = 0; i < star_count - 1; i++) {
        for (int j = i + 1; j < star_count; j++) {
            if (temp_stars[j].brightness > temp_stars[i].brightness) {
                PSFStar temp = temp_stars[i];
                temp_stars[i] = temp_stars[j];
                temp_stars[j] = temp;
            }
        }
    }
    
    *stars = PSF_REALLOC(temp_stars, PSFStar, star_count);
    *num_stars = star_count;
    
    return PSF_SUCCESS;
}

/**
 * @brief Destroy star array
 */
PSF_API void psf_destroy_stars(PSFStar *stars)
{
    PSF_FREE(stars);
}

// ============================================================================
// PSF Extraction from Stars
// ============================================================================

/**
 * @brief Extract PSF from single star
 */
PSF_API PSFError psf_extract_from_star(
    const PSFImage *image,
    const PSFStar *star,
    int kernel_size,
    PSFKernel **psf)
{
    if (!image || !star || !psf) {
        return PSF_ERROR_NULL_POINTER;
    }
    
    if (kernel_size <= 0 || kernel_size % 2 == 0) {
        return PSF_ERROR_INVALID_PARAM;
    }
    
    // Convert to grayscale if needed
    PSFImage *gray = NULL;
    PSFError err = psf_convert_to_grayscale(image, &gray);
    if (err != PSF_SUCCESS) {
        return err;
    }
    
    // Create PSF kernel
    err = psf_create_kernel(kernel_size, kernel_size, psf);
    if (err != PSF_SUCCESS) {
        psf_destroy_image(gray);
        return err;
    }
    
    int half_size = kernel_size / 2;
    int cx = (int)(star->center.x + 0.5);
    int cy = (int)(star->center.y + 0.5);
    
    // Extract region around star with sub-pixel interpolation
    for (int y = 0; y < kernel_size; y++) {
        for (int x = 0; x < kernel_size; x++) {
            double sx = star->center.x + (x - half_size);
            double sy = star->center.y + (y - half_size);
            
            // Bilinear interpolation
            int x0 = (int)floor(sx);
            int y0 = (int)floor(sy);
            int x1 = x0 + 1;
            int y1 = y0 + 1;
            
            double fx = sx - x0;
            double fy = sy - y0;
            
            double value = 0.0;
            
            if (x0 >= 0 && x0 < gray->width && y0 >= 0 && y0 < gray->height) {
                value += gray->data[y0 * gray->width + x0] * (1 - fx) * (1 - fy);
            }
            if (x1 >= 0 && x1 < gray->width && y0 >= 0 && y0 < gray->height) {
                value += gray->data[y0 * gray->width + x1] * fx * (1 - fy);
            }
            if (x0 >= 0 && x0 < gray->width && y1 >= 0 && y1 < gray->height) {
                value += gray->data[y1 * gray->width + x0] * (1 - fx) * fy;
            }
            if (x1 >= 0 && x1 < gray->width && y1 >= 0 && y1 < gray->height) {
                value += gray->data[y1 * gray->width + x1] * fx * fy;
            }
            
            (*psf)->data[y * kernel_size + x] = value;
        }
    }
    
    // Estimate and subtract background
    double background = 0.0;
    int bg_count = 0;
    
    // Use edge pixels for background estimation
    for (int i = 0; i < kernel_size; i++) {
        background += (*psf)->data[i];  // Top edge
        background += (*psf)->data[(kernel_size - 1) * kernel_size + i];  // Bottom edge
        background += (*psf)->data[i * kernel_size];  // Left edge
        background += (*psf)->data[i * kernel_size + (kernel_size - 1)];  // Right edge
        bg_count += 4;
    }
    background /= bg_count;
    
    // Subtract background
    for (int i = 0; i < kernel_size * kernel_size; i++) {
        (*psf)->data[i] -= background;
        if ((*psf)->data[i] < 0) (*psf)->data[i] = 0;
    }
    
    // Normalize PSF
    psf_normalize_kernel(*psf);
    psf_center_kernel(*psf);
    
    psf_destroy_image(gray);
    
    return PSF_SUCCESS;
}

/**
 * @brief Estimate PSF from multiple stars (average)
 */
PSF_API PSFError psf_estimate_from_stars(
    const PSFImage *image,
    const PSFEstimationConfig *config,
    PSFKernel **psf)
{
    if (!image || !psf) {
        return PSF_ERROR_NULL_POINTER;
    }
    
    // Use default config if not provided
    PSFEstimationConfig default_config;
    if (!config) {
        default_config.star_detection_threshold = 0.1;
        default_config.star_min_size = 3;
        default_config.star_max_size = 50;
        default_config.star_roundness_threshold = 0.5;
        default_config.psf_kernel_size = 15;
        config = &default_config;
    }
    
    // Detect stars
    PSFStar *stars = NULL;
    int num_stars = 0;
    
    PSFError err = psf_detect_stars(image, config, &stars, &num_stars);
    if (err != PSF_SUCCESS) {
        return err;
    }
    
    // Use top N brightest stars
    int max_stars = PSF_MIN(num_stars, 20);
    
    // Create accumulator for average PSF
    err = psf_create_kernel(config->psf_kernel_size, config->psf_kernel_size, psf);
    if (err != PSF_SUCCESS) {
        psf_destroy_stars(stars);
        return err;
    }
    
    int kernel_size = config->psf_kernel_size;
    int valid_count = 0;
    
    // Extract and accumulate PSF from each star
    for (int i = 0; i < max_stars; i++) {
        PSFKernel *star_psf = NULL;
        
        err = psf_extract_from_star(image, &stars[i], kernel_size, &star_psf);
        
        if (err == PSF_SUCCESS) {
            // Add to accumulator
            for (int j = 0; j < kernel_size * kernel_size; j++) {
                (*psf)->data[j] += star_psf->data[j];
            }
            valid_count++;
            
            psf_destroy_kernel(star_psf);
        }
    }
    
    psf_destroy_stars(stars);
    
    if (valid_count == 0) {
        psf_destroy_kernel(*psf);
        *psf = NULL;
        return PSF_ERROR_INSUFFICIENT_DATA;
    }
    
    // Average the accumulated PSF
    double scale = 1.0 / valid_count;
    for (int i = 0; i < kernel_size * kernel_size; i++) {
        (*psf)->data[i] *= scale;
    }
    
    // Normalize and center
    psf_normalize_kernel(*psf);
    psf_center_kernel(*psf);
    
    return PSF_SUCCESS;
}

/**
 * @brief Estimate spatially varying PSF from stars
 */
PSF_API PSFError psf_estimate_spatially_varying(
    const PSFImage *image,
    const PSFEstimationConfig *config,
    int grid_width,
    int grid_height,
    PSFKernel ***psf_grid)
{
    if (!image || !psf_grid) {
        return PSF_ERROR_NULL_POINTER;
    }
    
    if (grid_width <= 0 || grid_height <= 0) {
        return PSF_ERROR_INVALID_PARAM;
    }
    
    // Use default config if not provided
    PSFEstimationConfig default_config;
    if (!config) {
        default_config.star_detection_threshold = 0.1;
        default_config.star_min_size = 3;
        default_config.star_max_size = 50;
        default_config.star_roundness_threshold = 0.5;
        default_config.psf_kernel_size = 15;
        config = &default_config;
    }
    
    // Detect all stars
    PSFStar *stars = NULL;
    int num_stars = 0;
    
    PSFError err = psf_detect_stars(image, config, &stars, &num_stars);
    if (err != PSF_SUCCESS) {
        return err;
    }
    
    // Allocate PSF grid
    PSFKernel **grid = PSF_MALLOC(PSFKernel*, grid_height);
    for (int i = 0; i < grid_height; i++) {
        grid[i] = PSF_MALLOC(PSFKernel, grid_width);
    }
    
    // Compute cell size
    double cell_width = (double)image->width / grid_width;
    double cell_height = (double)image->height / grid_height;
    
    int kernel_size = config->psf_kernel_size;
    
    // For each grid cell
    for (int gy = 0; gy < grid_height; gy++) {
        for (int gx = 0; gx < grid_width; gx++) {
            // Define cell boundaries
            double x_min = gx * cell_width;
            double x_max = (gx + 1) * cell_width;
            double y_min = gy * cell_height;
            double y_max = (gy + 1) * cell_height;
            
            // Find stars in this cell (with some margin)
            double margin = PSF_MAX(cell_width, cell_height) * 0.5;
            
            PSFKernel *cell_psf = NULL;
            err = psf_create_kernel(kernel_size, kernel_size, &cell_psf);
            
            int star_count = 0;
            
            for (int i = 0; i < num_stars; i++) {
                double sx = stars[i].center.x;
                double sy = stars[i].center.y;
                
                // Check if star is in or near this cell
                if (sx >= x_min - margin && sx <= x_max + margin &&
                    sy >= y_min - margin && sy <= y_max + margin) {
                    
                    PSFKernel *star_psf = NULL;
                    err = psf_extract_from_star(image, &stars[i], 
                                               kernel_size, &star_psf);
                    
                    if (err == PSF_SUCCESS) {
                        // Weight by distance to cell center
                        double cx = (x_min + x_max) / 2.0;
                        double cy = (y_min + y_max) / 2.0;
                        double dist = sqrt((sx - cx) * (sx - cx) + 
                                         (sy - cy) * (sy - cy));
                        double weight = exp(-dist * dist / 
                                          (2.0 * margin * margin));
                        
                        // Add weighted PSF
                        for (int j = 0; j < kernel_size * kernel_size; j++) {
                            cell_psf->data[j] += star_psf->data[j] * weight;
                        }
                        star_count++;
                        
                        psf_destroy_kernel(star_psf);
                    }
                }
            }
            
            if (star_count > 0) {
                psf_normalize_kernel(cell_psf);
                psf_center_kernel(cell_psf);
            } else {
                // No stars in this cell, use default Gaussian PSF
                double sigma = 2.0;
                int half_size = kernel_size / 2;
                
                for (int y = 0; y < kernel_size; y++) {
                    for (int x = 0; x < kernel_size; x++) {
                        double dx = x - half_size;
                        double dy = y - half_size;
                        cell_psf->data[y * kernel_size + x] = 
                            gaussian_2d(dx, dy, sigma, sigma, 0.0);
                    }
                }
                psf_normalize_kernel(cell_psf);
            }
            
            grid[gy][gx] = *cell_psf;
            PSF_FREE(cell_psf);
        }
    }
    
    psf_destroy_stars(stars);
    
    *psf_grid = grid;
    
    return PSF_SUCCESS;
}
// ============================================================================
// Blind Deconvolution
// ============================================================================

/**
 * @brief Richardson-Lucy deconvolution iteration
 */
static void richardson_lucy_iteration(
    const double *blurred,
    const double *psf,
    double *estimate,
    double *temp,
    int width,
    int height,
    int psf_width,
    int psf_height)
{
    int size = width * height;
    int psf_half_w = psf_width / 2;
    int psf_half_h = psf_height / 2;
    
    // Convolve estimate with PSF
    memset(temp, 0, size * sizeof(double));
    
#ifdef _OPENMP
    #pragma omp parallel for collapse(2)
#endif
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            double sum = 0.0;
            
            for (int py = 0; py < psf_height; py++) {
                for (int px = 0; px < psf_width; px++) {
                    int sx = x + px - psf_half_w;
                    int sy = y + py - psf_half_h;
                    
                    if (sx >= 0 && sx < width && sy >= 0 && sy < height) {
                        sum += estimate[sy * width + sx] * 
                               psf[py * psf_width + px];
                    }
                }
            }
            
            temp[y * width + x] = sum;
        }
    }
    
    // Compute ratio image
    for (int i = 0; i < size; i++) {
        if (temp[i] > PSF_EPSILON) {
            temp[i] = blurred[i] / temp[i];
        } else {
            temp[i] = 0.0;
        }
    }
    
    // Correlate ratio with flipped PSF
    double *correction = PSF_MALLOC(double, size);
    memset(correction, 0, size * sizeof(double));
    
#ifdef _OPENMP
    #pragma omp parallel for collapse(2)
#endif
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            double sum = 0.0;
            
            for (int py = 0; py < psf_height; py++) {
                for (int px = 0; px < psf_width; px++) {
                    int sx = x - px + psf_half_w;
                    int sy = y - py + psf_half_h;
                    
                    if (sx >= 0 && sx < width && sy >= 0 && sy < height) {
                        // Flip PSF indices
                        int psf_idx = (psf_height - 1 - py) * psf_width + 
                                     (psf_width - 1 - px);
                        sum += temp[sy * width + sx] * psf[psf_idx];
                    }
                }
            }
            
            correction[y * width + x] = sum;
        }
    }
    
    // Update estimate
    for (int i = 0; i < size; i++) {
        estimate[i] *= correction[i];
        // Enforce non-negativity
        if (estimate[i] < 0) estimate[i] = 0;
    }
    
    PSF_FREE(correction);
}

/**
 * @brief Blind Richardson-Lucy deconvolution
 */
PSF_API PSFError psf_blind_deconvolution(
    const PSFImage *blurred,
    const PSFEstimationConfig *config,
    PSFImage **restored,
    PSFKernel **psf)
{
    if (!blurred || !restored || !psf) {
        return PSF_ERROR_NULL_POINTER;
    }
    
    // Use default config if not provided
    PSFEstimationConfig default_config;
    if (!config) {
        default_config.blind_deconv_iterations = 50;
        default_config.blind_deconv_psf_size = 15;
        default_config.blind_deconv_regularization = 0.001;
        config = &default_config;
    }
    
    // Convert to grayscale if needed
    PSFImage *gray = NULL;
    PSFError err = psf_convert_to_grayscale(blurred, &gray);
    if (err != PSF_SUCCESS) {
        return err;
    }
    
    int width = gray->width;
    int height = gray->height;
    int size = width * height;
    
    int psf_size = config->blind_deconv_psf_size;
    if (psf_size % 2 == 0) psf_size++;
    
    // Initialize PSF with Gaussian
    err = psf_create_kernel(psf_size, psf_size, psf);
    if (err != PSF_SUCCESS) {
        psf_destroy_image(gray);
        return err;
    }
    
    double sigma = psf_size / 6.0;
    int half_size = psf_size / 2;
    
    for (int y = 0; y < psf_size; y++) {
        for (int x = 0; x < psf_size; x++) {
            double dx = x - half_size;
            double dy = y - half_size;
            (*psf)->data[y * psf_size + x] = gaussian_2d(dx, dy, sigma, sigma, 0.0);
        }
    }
    psf_normalize_kernel(*psf);
    
    // Initialize restored image with blurred image
    err = psf_create_image(width, height, 1, restored);
    if (err != PSF_SUCCESS) {
        psf_destroy_kernel(*psf);
        psf_destroy_image(gray);
        return err;
    }
    
    memcpy((*restored)->data, gray->data, size * sizeof(double));
    
    // Temporary buffers
    double *temp = PSF_MALLOC(double, size);
    double *psf_temp = PSF_MALLOC(double, psf_size * psf_size);
    
    // Alternating optimization
    for (int iter = 0; iter < config->blind_deconv_iterations; iter++) {
        // Update image estimate (fix PSF)
        for (int i = 0; i < 5; i++) {
            richardson_lucy_iteration(gray->data, (*psf)->data,
                                     (*restored)->data, temp,
                                     width, height, psf_size, psf_size);
        }
        
        // Update PSF estimate (fix image)
        // Treat restored image as "PSF" and blurred as "image"
        for (int i = 0; i < 2; i++) {
            richardson_lucy_iteration(gray->data, (*restored)->data,
                                     (*psf)->data, psf_temp,
                                     psf_size, psf_size, width, height);
        }
        
        // Regularize PSF (enforce smoothness and non-negativity)
        double lambda = config->blind_deconv_regularization;
        
        for (int y = 1; y < psf_size - 1; y++) {
            for (int x = 1; x < psf_size - 1; x++) {
                int idx = y * psf_size + x;
                
                // Laplacian regularization
                double laplacian = 
                    (*psf)->data[(y-1) * psf_size + x] +
                    (*psf)->data[(y+1) * psf_size + x] +
                    (*psf)->data[y * psf_size + (x-1)] +
                    (*psf)->data[y * psf_size + (x+1)] -
                    4.0 * (*psf)->data[idx];
                
                (*psf)->data[idx] += lambda * laplacian;
                
                // Enforce non-negativity
                if ((*psf)->data[idx] < 0) {
                    (*psf)->data[idx] = 0;
                }
            }
        }
        
        // Normalize PSF
        psf_normalize_kernel(*psf);
        
        // Check convergence (optional)
        if (iter % 10 == 0) {
            // Could compute residual here
        }
    }
    
    PSF_FREE(temp);
    PSF_FREE(psf_temp);
    psf_destroy_image(gray);
    
    return PSF_SUCCESS;
}

/**
 * @brief Wiener deconvolution
 */
PSF_API PSFError psf_wiener_deconvolution(
    const PSFImage *blurred,
    const PSFKernel *psf,
    double noise_variance,
    PSFImage **restored)
{
    if (!blurred || !psf || !restored) {
        return PSF_ERROR_NULL_POINTER;
    }
    
    // Convert to grayscale if needed
    PSFImage *gray = NULL;
    PSFError err = psf_convert_to_grayscale(blurred, &gray);
    if (err != PSF_SUCCESS) {
        return err;
    }
    
    int width = gray->width;
    int height = gray->height;
    int size = width * height;
    
    // Compute FFT of blurred image
    double *blurred_real = PSF_MALLOC(double, size);
    double *blurred_imag = PSF_MALLOC(double, size);
    
    fft_2d(gray->data, blurred_real, blurred_imag, width, height);
    
    // Compute FFT of PSF (zero-padded to image size)
    double *psf_padded = PSF_CALLOC(double, size);
    
    int psf_offset_x = (width - psf->width) / 2;
    int psf_offset_y = (height - psf->height) / 2;
    
    for (int y = 0; y < psf->height; y++) {
        for (int x = 0; x < psf->width; x++) {
            int dst_x = (x + psf_offset_x) % width;
            int dst_y = (y + psf_offset_y) % height;
            psf_padded[dst_y * width + dst_x] = psf->data[y * psf->width + x];
        }
    }
    
    double *psf_real = PSF_MALLOC(double, size);
    double *psf_imag = PSF_MALLOC(double, size);
    
    fft_2d(psf_padded, psf_real, psf_imag, width, height);
    
    // Wiener filter in frequency domain
    double *restored_real = PSF_MALLOC(double, size);
    double *restored_imag = PSF_MALLOC(double, size);
    
#ifdef _OPENMP
    #pragma omp parallel for
#endif
    for (int i = 0; i < size; i++) {
        double psf_mag_sq = psf_real[i] * psf_real[i] + 
                           psf_imag[i] * psf_imag[i];
        
        // Wiener filter: H* / (|H|^2 + K)
        double K = noise_variance;
        double denom = psf_mag_sq + K;
        
        if (denom > PSF_EPSILON) {
            // Complex conjugate of PSF
            double psf_conj_real = psf_real[i];
            double psf_conj_imag = -psf_imag[i];
            
            // Multiply: (PSF* / denom) * Blurred
            double filter_real = psf_conj_real / denom;
            double filter_imag = psf_conj_imag / denom;
            
            restored_real[i] = filter_real * blurred_real[i] - 
                              filter_imag * blurred_imag[i];
            restored_imag[i] = filter_real * blurred_imag[i] + 
                              filter_imag * blurred_real[i];
        } else {
            restored_real[i] = 0.0;
            restored_imag[i] = 0.0;
        }
    }
    
    // Inverse FFT
    err = psf_create_image(width, height, 1, restored);
    if (err != PSF_SUCCESS) {
        PSF_FREE(blurred_real);
        PSF_FREE(blurred_imag);
        PSF_FREE(psf_padded);
        PSF_FREE(psf_real);
        PSF_FREE(psf_imag);
        PSF_FREE(restored_real);
        PSF_FREE(restored_imag);
        psf_destroy_image(gray);
        return err;
    }
    
    ifft_2d(restored_real, restored_imag, (*restored)->data, width, height);
    
    // Cleanup
    PSF_FREE(blurred_real);
    PSF_FREE(blurred_imag);
    PSF_FREE(psf_padded);
    PSF_FREE(psf_real);
    PSF_FREE(psf_imag);
    PSF_FREE(restored_real);
    PSF_FREE(restored_imag);
    psf_destroy_image(gray);
    
    return PSF_SUCCESS;
}

// ============================================================================
// PSF Modeling
// ============================================================================

/**
 * @brief Fit Gaussian PSF model to data
 */
PSF_API PSFError psf_fit_gaussian_model(
    const PSFKernel *measured_psf,
    PSFGaussianModel *model)
{
    if (!measured_psf || !model) {
        return PSF_ERROR_NULL_POINTER;
    }
    
    int width = measured_psf->width;
    int height = measured_psf->height;
    
    // Compute moments to estimate initial parameters
    double m00 = 0.0;  // Total intensity
    double m10 = 0.0, m01 = 0.0;  // First moments
    double m20 = 0.0, m11 = 0.0, m02 = 0.0;  // Second moments
    
    double cx = width / 2.0;
    double cy = height / 2.0;
    
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            double value = measured_psf->data[y * width + x];
            double dx = x - cx;
            double dy = y - cy;
            
            m00 += value;
            m10 += dx * value;
            m01 += dy * value;
            m20 += dx * dx * value;
            m11 += dx * dy * value;
            m02 += dy * dy * value;
        }
    }
    
    if (m00 < PSF_EPSILON) {
        return PSF_ERROR_INVALID_PSF;
    }
    
    // Center
    model->center_x = cx + m10 / m00;
    model->center_y = cy + m01 / m00;
    
    // Central moments
    double mu20 = m20 / m00 - (m10 / m00) * (m10 / m00);
    double mu11 = m11 / m00 - (m10 / m00) * (m01 / m00);
    double mu02 = m02 / m00 - (m01 / m00) * (m01 / m00);
    
    // Compute eigenvalues for sigma_x and sigma_y
    double trace = mu20 + mu02;
    double det = mu20 * mu02 - mu11 * mu11;
    
    double lambda1 = 0.5 * (trace + sqrt(trace * trace - 4 * det));
    double lambda2 = 0.5 * (trace - sqrt(trace * trace - 4 * det));
    
    model->sigma_x = sqrt(PSF_MAX(lambda1, PSF_EPSILON));
    model->sigma_y = sqrt(PSF_MAX(lambda2, PSF_EPSILON));
    
    // Rotation angle
    if (fabs(mu20 - mu02) > PSF_EPSILON) {
        model->rotation = 0.5 * atan2(2.0 * mu11, mu20 - mu02);
    } else {
        model->rotation = 0.0;
    }
    
    model->amplitude = m00;
    
    // Compute fit quality (R-squared)
    double ss_tot = 0.0;
    double ss_res = 0.0;
    double mean = m00 / (width * height);
    
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            double dx = x - model->center_x;
            double dy = y - model->center_y;
            
            double measured = measured_psf->data[y * width + x];
            double fitted = model->amplitude * 
                          gaussian_2d(dx, dy, model->sigma_x, 
                                    model->sigma_y, model->rotation);
            
            ss_tot += (measured - mean) * (measured - mean);
            ss_res += (measured - fitted) * (measured - fitted);
        }
    }
    
    if (ss_tot > PSF_EPSILON) {
        model->fit_quality = 1.0 - ss_res / ss_tot;
    } else {
        model->fit_quality = 0.0;
    }
    
    return PSF_SUCCESS;
}

/**
 * @brief Fit Moffat PSF model to data
 */
PSF_API PSFError psf_fit_moffat_model(
    const PSFKernel *measured_psf,
    PSFMoffatModel *model)
{
    if (!measured_psf || !model) {
        return PSF_ERROR_NULL_POINTER;
    }
    
    int width = measured_psf->width;
    int height = measured_psf->height;
    
    // Find center (maximum value)
    double max_value = -DBL_MAX;
    int max_x = width / 2;
    int max_y = height / 2;
    
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            double value = measured_psf->data[y * width + x];
            if (value > max_value) {
                max_value = value;
                max_x = x;
                max_y = y;
            }
        }
    }
    
    model->center_x = max_x;
    model->center_y = max_y;
    model->amplitude = max_value;
    
    // Estimate alpha from FWHM
    // Find half-maximum radius
    double half_max = max_value / 2.0;
    double fwhm = 0.0;
    int count = 0;
    
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            double dx = x - model->center_x;
            double dy = y - model->center_y;
            double r = sqrt(dx * dx + dy * dy);
            double value = measured_psf->data[y * width + x];
            
            if (fabs(value - half_max) < half_max * 0.1) {
                fwhm += r;
                count++;
            }
        }
    }
    
    if (count > 0) {
        fwhm = 2.0 * fwhm / count;
    } else {
        fwhm = width / 4.0;
    }
    
    // Initial guess for beta
    model->beta = 4.0;
    
    // For Moffat: FWHM = 2 * alpha * sqrt(2^(1/beta) - 1)
    model->alpha = fwhm / (2.0 * sqrt(pow(2.0, 1.0 / model->beta) - 1.0));
    
    // Simple non-linear least squares refinement
    double learning_rate = 0.01;
    int max_iterations = 100;
    
    for (int iter = 0; iter < max_iterations; iter++) {
        double grad_alpha = 0.0;
        double grad_beta = 0.0;
        double error = 0.0;
        
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                double dx = x - model->center_x;
                double dy = y - model->center_y;
                double r = sqrt(dx * dx + dy * dy);
                
                double measured = measured_psf->data[y * width + x];
                double fitted = model->amplitude * 
                              moffat_2d(dx, dy, model->alpha, model->beta);
                
                double residual = fitted - measured;
                error += residual * residual;
                
                // Compute gradients (numerical approximation)
                double h = 0.001;
                
                double fitted_alpha_plus = model->amplitude * 
                    moffat_2d(dx, dy, model->alpha + h, model->beta);
                grad_alpha += residual * (fitted_alpha_plus - fitted) / h;
                
                double fitted_beta_plus = model->amplitude * 
                    moffat_2d(dx, dy, model->alpha, model->beta + h);
                grad_beta += residual * (fitted_beta_plus - fitted) / h;
            }
        }
        
        // Update parameters
        model->alpha -= learning_rate * grad_alpha;
        model->beta -= learning_rate * grad_beta;
        
        // Constrain parameters
        if (model->alpha < 0.1) model->alpha = 0.1;
        if (model->beta < 1.0) model->beta = 1.0;
        if (model->beta > 10.0) model->beta = 10.0;
        
        // Check convergence
        if (fabs(grad_alpha) < 1e-6 && fabs(grad_beta) < 1e-6) {
            break;
        }
    }
    
    // Compute fit quality
    double ss_tot = 0.0;
    double ss_res = 0.0;
    double mean = 0.0;
    
    for (int i = 0; i < width * height; i++) {
        mean += measured_psf->data[i];
    }
    mean /= (width * height);
    
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            double dx = x - model->center_x;
            double dy = y - model->center_y;
            
            double measured = measured_psf->data[y * width + x];
            double fitted = model->amplitude * 
                          moffat_2d(dx, dy, model->alpha, model->beta);
            
            ss_tot += (measured - mean) * (measured - mean);
            ss_res += (measured - fitted) * (measured - fitted);
        }
    }
    
    if (ss_tot > PSF_EPSILON) {
        model->fit_quality = 1.0 - ss_res / ss_tot;
    } else {
        model->fit_quality = 0.0;
    }
    
    return PSF_SUCCESS;
}

/**
 * @brief Generate synthetic PSF from Gaussian model
 */
PSF_API PSFError psf_generate_gaussian(
    const PSFGaussianModel *model,
    int kernel_size,
    PSFKernel **psf)
{
    if (!model || !psf) {
        return PSF_ERROR_NULL_POINTER;
    }
    
    if (kernel_size <= 0 || kernel_size % 2 == 0) {
        return PSF_ERROR_INVALID_PARAM;
    }
    
    PSFError err = psf_create_kernel(kernel_size, kernel_size, psf);
    if (err != PSF_SUCCESS) {
        return err;
    }
    
    int half_size = kernel_size / 2;
    
#ifdef _OPENMP
    #pragma omp parallel for collapse(2)
#endif
    for (int y = 0; y < kernel_size; y++) {
        for (int x = 0; x < kernel_size; x++) {
            double dx = x - half_size;
            double dy = y - half_size;
            
            (*psf)->data[y * kernel_size + x] = 
                model->amplitude * gaussian_2d(dx, dy, model->sigma_x,
                                              model->sigma_y, model->rotation);
        }
    }
    
    psf_normalize_kernel(*psf);
    
    (*psf)->center_x = model->center_x;
    (*psf)->center_y = model->center_y;
    
    return PSF_SUCCESS;
}

/**
 * @brief Generate synthetic PSF from Moffat model
 */
PSF_API PSFError psf_generate_moffat(
    const PSFMoffatModel *model,
    int kernel_size,
    PSFKernel **psf)
{
    if (!model || !psf) {
        return PSF_ERROR_NULL_POINTER;
    }
    
    if (kernel_size <= 0 || kernel_size % 2 == 0) {
        return PSF_ERROR_INVALID_PARAM;
    }
    
    PSFError err = psf_create_kernel(kernel_size, kernel_size, psf);
    if (err != PSF_SUCCESS) {
        return err;
    }
    
    int half_size = kernel_size / 2;
    
#ifdef _OPENMP
    #pragma omp parallel for collapse(2)
#endif
    for (int y = 0; y < kernel_size; y++) {
        for (int x = 0; x < kernel_size; x++) {
            double dx = x - half_size;
            double dy = y - half_size;
            
            (*psf)->data[y * kernel_size + x] = 
                model->amplitude * moffat_2d(dx, dy, model->alpha, model->beta);
        }
    }
    
    psf_normalize_kernel(*psf);
    
    (*psf)->center_x = model->center_x;
    (*psf)->center_y = model->center_y;
    
    return PSF_SUCCESS;
}

/**
 * @brief Generate synthetic Airy disk PSF
 */
PSF_API PSFError psf_generate_airy(
    double radius,
    int kernel_size,
    PSFKernel **psf)
{
    if (!psf) {
        return PSF_ERROR_NULL_POINTER;
    }
    
    if (radius <= 0 || kernel_size <= 0 || kernel_size % 2 == 0) {
        return PSF_ERROR_INVALID_PARAM;
    }
    
    PSFError err = psf_create_kernel(kernel_size, kernel_size, psf);
    if (err != PSF_SUCCESS) {
        return err;
    }
    
    int half_size = kernel_size / 2;
    
#ifdef _OPENMP
    #pragma omp parallel for collapse(2)
#endif
    for (int y = 0; y < kernel_size; y++) {
        for (int x = 0; x < kernel_size; x++) {
            double dx = x - half_size;
            double dy = y - half_size;
            double r = sqrt(dx * dx + dy * dy);
            
            (*psf)->data[y * kernel_size + x] = airy_disk(r, radius);
        }
    }
    
    psf_normalize_kernel(*psf);
    psf_center_kernel(*psf);
    
    return PSF_SUCCESS;
}
// ============================================================================
// Modulation Transfer Function (MTF) Analysis
// ============================================================================

/**
 * @brief Compute 1D MTF from LSF
 */
PSF_API PSFError psf_compute_mtf_from_lsf(
    const PSFLineSpreadFunction *lsf,
    PSFMTF **mtf)
{
    if (!lsf || !mtf) {
        return PSF_ERROR_NULL_POINTER;
    }
    
    PSFMTF *mtf_result = PSF_MALLOC(PSFMTF, 1);
    if (!mtf_result) {
        return PSF_ERROR_MEMORY_ALLOCATION;
    }
    
    int length = lsf->length;
    
    // Allocate MTF arrays
    mtf_result->length = length / 2;  // Nyquist limit
    mtf_result->frequencies = PSF_MALLOC(double, mtf_result->length);
    mtf_result->values = PSF_MALLOC(double, mtf_result->length);
    
    // Compute FFT of LSF
    double *lsf_real = PSF_MALLOC(double, length);
    double *lsf_imag = PSF_CALLOC(double, length);
    
    memcpy(lsf_real, lsf->data, length * sizeof(double));
    
    fft_1d(lsf_real, lsf_imag, length);
    
    // Compute magnitude (MTF)
    double nyquist_freq = 0.5 / lsf->pixel_spacing;
    
    for (int i = 0; i < mtf_result->length; i++) {
        mtf_result->frequencies[i] = (double)i / length * nyquist_freq * 2.0;
        
        double real = lsf_real[i];
        double imag = lsf_imag[i];
        mtf_result->values[i] = sqrt(real * real + imag * imag);
    }
    
    // Normalize MTF (DC component = 1)
    if (mtf_result->values[0] > PSF_EPSILON) {
        double scale = 1.0 / mtf_result->values[0];
        for (int i = 0; i < mtf_result->length; i++) {
            mtf_result->values[i] *= scale;
        }
    }
    
    PSF_FREE(lsf_real);
    PSF_FREE(lsf_imag);
    
    *mtf = mtf_result;
    
    return PSF_SUCCESS;
}

/**
 * @brief Compute 2D MTF from PSF
 */
PSF_API PSFError psf_compute_mtf_from_psf(
    const PSFKernel *psf,
    PSFMTF2D **mtf)
{
    if (!psf || !mtf) {
        return PSF_ERROR_NULL_POINTER;
    }
    
    int width = psf->width;
    int height = psf->height;
    int size = width * height;
    
    // Allocate MTF structure
    PSFMTF2D *mtf_result = PSF_MALLOC(PSFMTF2D, 1);
    if (!mtf_result) {
        return PSF_ERROR_MEMORY_ALLOCATION;
    }
    
    mtf_result->width = width;
    mtf_result->height = height;
    mtf_result->data = PSF_MALLOC(double, size);
    
    // Compute 2D FFT of PSF
    double *psf_real = PSF_MALLOC(double, size);
    double *psf_imag = PSF_CALLOC(double, size);
    
    memcpy(psf_real, psf->data, size * sizeof(double));
    
    fft_2d(psf_real, psf_imag, width, height);
    
    // Compute magnitude (MTF)
#ifdef _OPENMP
    #pragma omp parallel for
#endif
    for (int i = 0; i < size; i++) {
        double real = psf_real[i];
        double imag = psf_imag[i];
        mtf_result->data[i] = sqrt(real * real + imag * imag);
    }
    
    // Normalize MTF
    if (mtf_result->data[0] > PSF_EPSILON) {
        double scale = 1.0 / mtf_result->data[0];
        for (int i = 0; i < size; i++) {
            mtf_result->data[i] *= scale;
        }
    }
    
    // Shift zero frequency to center
    fftshift_2d(mtf_result->data, width, height);
    
    PSF_FREE(psf_real);
    PSF_FREE(psf_imag);
    
    *mtf = mtf_result;
    
    return PSF_SUCCESS;
}

/**
 * @brief Compute radial MTF profile
 */
PSF_API PSFError psf_compute_radial_mtf(
    const PSFMTF2D *mtf,
    int num_bins,
    double **frequencies,
    double **values,
    int *length)
{
    if (!mtf || !frequencies || !values || !length) {
        return PSF_ERROR_NULL_POINTER;
    }
    
    int width = mtf->width;
    int height = mtf->height;
    int cx = width / 2;
    int cy = height / 2;
    
    // Determine maximum radius
    double max_radius = sqrt((double)(cx * cx + cy * cy));
    
    // Allocate bins
    double *bin_sums = PSF_CALLOC(double, num_bins);
    int *bin_counts = PSF_CALLOC(int, num_bins);
    
    // Accumulate values in radial bins
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            double dx = x - cx;
            double dy = y - cy;
            double r = sqrt(dx * dx + dy * dy);
            
            int bin = (int)(r / max_radius * num_bins);
            if (bin >= 0 && bin < num_bins) {
                bin_sums[bin] += mtf->data[y * width + x];
                bin_counts[bin]++;
            }
        }
    }
    
    // Compute average for each bin
    *frequencies = PSF_MALLOC(double, num_bins);
    *values = PSF_MALLOC(double, num_bins);
    
    for (int i = 0; i < num_bins; i++) {
        (*frequencies)[i] = (i + 0.5) / num_bins * max_radius;
        
        if (bin_counts[i] > 0) {
            (*values)[i] = bin_sums[i] / bin_counts[i];
        } else {
            (*values)[i] = 0.0;
        }
    }
    
    *length = num_bins;
    
    PSF_FREE(bin_sums);
    PSF_FREE(bin_counts);
    
    return PSF_SUCCESS;
}

/**
 * @brief Find MTF50 (frequency at 50% contrast)
 */
PSF_API PSFError psf_compute_mtf50(
    const PSFMTF *mtf,
    double *mtf50)
{
    if (!mtf || !mtf50) {
        return PSF_ERROR_NULL_POINTER;
    }
    
    // Find frequency where MTF drops to 0.5
    for (int i = 0; i < mtf->length - 1; i++) {
        if (mtf->values[i] >= 0.5 && mtf->values[i + 1] < 0.5) {
            // Linear interpolation
            double f1 = mtf->frequencies[i];
            double f2 = mtf->frequencies[i + 1];
            double v1 = mtf->values[i];
            double v2 = mtf->values[i + 1];
            
            *mtf50 = f1 + (0.5 - v1) * (f2 - f1) / (v2 - v1);
            return PSF_SUCCESS;
        }
    }
    
    // MTF never drops to 0.5
    *mtf50 = mtf->frequencies[mtf->length - 1];
    
    return PSF_SUCCESS;
}

/**
 * @brief Destroy MTF structure
 */
PSF_API void psf_destroy_mtf(PSFMTF *mtf)
{
    if (!mtf) return;
    
    PSF_FREE(mtf->frequencies);
    PSF_FREE(mtf->values);
    PSF_FREE(mtf);
}

/**
 * @brief Destroy 2D MTF structure
 */
PSF_API void psf_destroy_mtf2d(PSFMTF2D *mtf)
{
    if (!mtf) return;
    
    PSF_FREE(mtf->data);
    PSF_FREE(mtf);
}

// ============================================================================
// Image Quality Metrics
// ============================================================================

/**
 * @brief Compute image sharpness using variance of Laplacian
 */
PSF_API PSFError psf_compute_sharpness(
    const PSFImage *image,
    double *sharpness)
{
    if (!image || !sharpness) {
        return PSF_ERROR_NULL_POINTER;
    }
    
    // Convert to grayscale if needed
    PSFImage *gray = NULL;
    PSFError err = psf_convert_to_grayscale(image, &gray);
    if (err != PSF_SUCCESS) {
        return err;
    }
    
    int width = gray->width;
    int height = gray->height;
    
    // Compute Laplacian
    double *laplacian = PSF_MALLOC(double, width * height);
    
    const int laplacian_kernel[3][3] = {
        {0,  1, 0},
        {1, -4, 1},
        {0,  1, 0}
    };
    
#ifdef _OPENMP
    #pragma omp parallel for collapse(2)
#endif
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            double sum = 0.0;
            
            for (int ky = -1; ky <= 1; ky++) {
                for (int kx = -1; kx <= 1; kx++) {
                    sum += gray->data[(y + ky) * width + (x + kx)] *
                           laplacian_kernel[ky + 1][kx + 1];
                }
            }
            
            laplacian[y * width + x] = sum;
        }
    }
    
    // Compute variance of Laplacian
    double mean = 0.0;
    int count = (width - 2) * (height - 2);
    
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            mean += laplacian[y * width + x];
        }
    }
    mean /= count;
    
    double variance = 0.0;
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            double diff = laplacian[y * width + x] - mean;
            variance += diff * diff;
        }
    }
    variance /= count;
    
    *sharpness = variance;
    
    PSF_FREE(laplacian);
    psf_destroy_image(gray);
    
    return PSF_SUCCESS;
}

/**
 * @brief Compute PSNR (Peak Signal-to-Noise Ratio)
 */
PSF_API PSFError psf_compute_psnr(
    const PSFImage *reference,
    const PSFImage *distorted,
    double *psnr)
{
    if (!reference || !distorted || !psnr) {
        return PSF_ERROR_NULL_POINTER;
    }
    
    if (reference->width != distorted->width ||
        reference->height != distorted->height ||
        reference->channels != distorted->channels) {
        return PSF_ERROR_DIMENSION_MISMATCH;
    }
    
    int size = reference->width * reference->height * reference->channels;
    
    // Compute MSE
    double mse = 0.0;
    
#ifdef _OPENMP
    #pragma omp parallel for reduction(+:mse)
#endif
    for (int i = 0; i < size; i++) {
        double diff = reference->data[i] - distorted->data[i];
        mse += diff * diff;
    }
    mse /= size;
    
    if (mse < PSF_EPSILON) {
        *psnr = INFINITY;
    } else {
        // Assume pixel values in [0, 1]
        *psnr = 10.0 * log10(1.0 / mse);
    }
    
    return PSF_SUCCESS;
}

/**
 * @brief Compute SSIM (Structural Similarity Index)
 */
PSF_API PSFError psf_compute_ssim(
    const PSFImage *reference,
    const PSFImage *distorted,
    double *ssim)
{
    if (!reference || !distorted || !ssim) {
        return PSF_ERROR_NULL_POINTER;
    }
    
    if (reference->width != distorted->width ||
        reference->height != distorted->height) {
        return PSF_ERROR_DIMENSION_MISMATCH;
    }
    
    // Convert to grayscale if needed
    PSFImage *ref_gray = NULL;
    PSFImage *dist_gray = NULL;
    
    PSFError err = psf_convert_to_grayscale(reference, &ref_gray);
    if (err != PSF_SUCCESS) {
        return err;
    }
    
    err = psf_convert_to_grayscale(distorted, &dist_gray);
    if (err != PSF_SUCCESS) {
        psf_destroy_image(ref_gray);
        return err;
    }
    
    int width = ref_gray->width;
    int height = ref_gray->height;
    
    // SSIM parameters
    const double C1 = 0.01 * 0.01;  // (K1 * L)^2
    const double C2 = 0.03 * 0.03;  // (K2 * L)^2
    const int window_size = 11;
    const int half_window = window_size / 2;
    
    // Gaussian window
    double *gaussian_window = PSF_MALLOC(double, window_size * window_size);
    double sigma = 1.5;
    double sum_weights = 0.0;
    
    for (int y = 0; y < window_size; y++) {
        for (int x = 0; x < window_size; x++) {
            double dx = x - half_window;
            double dy = y - half_window;
            double weight = exp(-(dx * dx + dy * dy) / (2.0 * sigma * sigma));
            gaussian_window[y * window_size + x] = weight;
            sum_weights += weight;
        }
    }
    
    // Normalize weights
    for (int i = 0; i < window_size * window_size; i++) {
        gaussian_window[i] /= sum_weights;
    }
    
    // Compute SSIM for each pixel
    double ssim_sum = 0.0;
    int count = 0;
    
    for (int y = half_window; y < height - half_window; y++) {
        for (int x = half_window; x < width - half_window; x++) {
            // Compute local statistics
            double mu_x = 0.0, mu_y = 0.0;
            double sigma_x = 0.0, sigma_y = 0.0, sigma_xy = 0.0;
            
            for (int wy = 0; wy < window_size; wy++) {
                for (int wx = 0; wx < window_size; wx++) {
                    double weight = gaussian_window[wy * window_size + wx];
                    
                    int px = x + wx - half_window;
                    int py = y + wy - half_window;
                    
                    double val_x = ref_gray->data[py * width + px];
                    double val_y = dist_gray->data[py * width + px];
                    
                    mu_x += weight * val_x;
                    mu_y += weight * val_y;
                }
            }
            
            for (int wy = 0; wy < window_size; wy++) {
                for (int wx = 0; wx < window_size; wx++) {
                    double weight = gaussian_window[wy * window_size + wx];
                    
                    int px = x + wx - half_window;
                    int py = y + wy - half_window;
                    
                    double val_x = ref_gray->data[py * width + px];
                    double val_y = dist_gray->data[py * width + px];
                    
                    double diff_x = val_x - mu_x;
                    double diff_y = val_y - mu_y;
                    
                    sigma_x += weight * diff_x * diff_x;
                    sigma_y += weight * diff_y * diff_y;
                    sigma_xy += weight * diff_x * diff_y;
                }
            }
            
            // Compute SSIM for this window
            double numerator = (2.0 * mu_x * mu_y + C1) * 
                             (2.0 * sigma_xy + C2);
            double denominator = (mu_x * mu_x + mu_y * mu_y + C1) *
                               (sigma_x + sigma_y + C2);
            
            double local_ssim = numerator / (denominator + PSF_EPSILON);
            
            ssim_sum += local_ssim;
            count++;
        }
    }
    
    *ssim = ssim_sum / count;
    
    PSF_FREE(gaussian_window);
    psf_destroy_image(ref_gray);
    psf_destroy_image(dist_gray);
    
    return PSF_SUCCESS;
}

/**
 * @brief Compute comprehensive image quality metrics
 */
PSF_API PSFError psf_compute_quality_metrics(
    const PSFImage *reference,
    const PSFImage *distorted,
    PSFQualityMetrics *metrics)
{
    if (!reference || !distorted || !metrics) {
        return PSF_ERROR_NULL_POINTER;
    }
    
    PSFError err;
    
    // Compute PSNR
    err = psf_compute_psnr(reference, distorted, &metrics->psnr);
    if (err != PSF_SUCCESS) {
        return err;
    }
    
    // Compute SSIM
    err = psf_compute_ssim(reference, distorted, &metrics->ssim);
    if (err != PSF_SUCCESS) {
        return err;
    }
    
    // Compute MSE
    int size = reference->width * reference->height * reference->channels;
    double mse = 0.0;
    
#ifdef _OPENMP
    #pragma omp parallel for reduction(+:mse)
#endif
    for (int i = 0; i < size; i++) {
        double diff = reference->data[i] - distorted->data[i];
        mse += diff * diff;
    }
    metrics->mse = mse / size;
    
    // Compute MAE
    double mae = 0.0;
    
#ifdef _OPENMP
    #pragma omp parallel for reduction(+:mae)
#endif
    for (int i = 0; i < size; i++) {
        mae += fabs(reference->data[i] - distorted->data[i]);
    }
    metrics->mae = mae / size;
    
    // Compute sharpness of distorted image
    err = psf_compute_sharpness(distorted, &metrics->sharpness);
    if (err != PSF_SUCCESS) {
        return err;
    }
    
    return PSF_SUCCESS;
}

/**
 * @brief Estimate blur kernel size from image
 */
PSF_API PSFError psf_estimate_blur_amount(
    const PSFImage *image,
    double *blur_amount)
{
    if (!image || !blur_amount) {
        return PSF_ERROR_NULL_POINTER;
    }
    
    // Convert to grayscale if needed
    PSFImage *gray = NULL;
    PSFError err = psf_convert_to_grayscale(image, &gray);
    if (err != PSF_SUCCESS) {
        return err;
    }
    
    int width = gray->width;
    int height = gray->height;
    
    // Compute gradient magnitude
    double *grad_mag = PSF_MALLOC(double, width * height);
    
#ifdef _OPENMP
    #pragma omp parallel for collapse(2)
#endif
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            double gx = gray->data[y * width + (x + 1)] - 
                       gray->data[y * width + (x - 1)];
            double gy = gray->data[(y + 1) * width + x] - 
                       gray->data[(y - 1) * width + x];
            
            grad_mag[y * width + x] = sqrt(gx * gx + gy * gy);
        }
    }
    
    // Find edges (high gradient regions)
    double threshold = 0.1;
    int edge_count = 0;
    double edge_width_sum = 0.0;
    
    for (int y = 2; y < height - 2; y++) {
        for (int x = 2; x < width - 2; x++) {
            if (grad_mag[y * width + x] > threshold) {
                // Measure edge width (distance between gradient peaks)
                double max_grad = grad_mag[y * width + x];
                int width_count = 1;
                
                // Check horizontal direction
                for (int dx = 1; dx < 10; dx++) {
                    if (x + dx < width - 1 && 
                        grad_mag[y * width + (x + dx)] > threshold * 0.5) {
                        width_count++;
                    } else {
                        break;
                    }
                }
                
                edge_width_sum += width_count;
                edge_count++;
            }
        }
    }
    
    if (edge_count > 0) {
        *blur_amount = edge_width_sum / edge_count;
    } else {
        *blur_amount = 1.0;
    }
    
    PSF_FREE(grad_mag);
    psf_destroy_image(gray);
    
    return PSF_SUCCESS;
}

/**
 * @brief Compute frequency domain energy concentration
 */
PSF_API PSFError psf_compute_frequency_concentration(
    const PSFImage *image,
    double *concentration)
{
    if (!image || !concentration) {
        return PSF_ERROR_NULL_POINTER;
    }
    
    // Convert to grayscale if needed
    PSFImage *gray = NULL;
    PSFError err = psf_convert_to_grayscale(image, &gray);
    if (err != PSF_SUCCESS) {
        return err;
    }
    
    int width = gray->width;
    int height = gray->height;
    int size = width * height;
    
    // Compute 2D FFT
    double *fft_real = PSF_MALLOC(double, size);
    double *fft_imag = PSF_CALLOC(double, size);
    
    memcpy(fft_real, gray->data, size * sizeof(double));
    
    fft_2d(fft_real, fft_imag, width, height);
    
    // Compute power spectrum
    double *power = PSF_MALLOC(double, size);
    double total_power = 0.0;
    
    for (int i = 0; i < size; i++) {
        power[i] = fft_real[i] * fft_real[i] + fft_imag[i] * fft_imag[i];
        total_power += power[i];
    }
    
    // Compute energy in low frequency region (center 25%)
    int cx = width / 2;
    int cy = height / 2;
    int radius = PSF_MIN(width, height) / 4;
    
    double low_freq_power = 0.0;
    
    for (int y = cy - radius; y < cy + radius; y++) {
        for (int x = cx - radius; x < cx + radius; x++) {
            if (x >= 0 && x < width && y >= 0 && y < height) {
                low_freq_power += power[y * width + x];
            }
        }
    }
    
    *concentration = low_freq_power / (total_power + PSF_EPSILON);
    
    PSF_FREE(fft_real);
    PSF_FREE(fft_imag);
    PSF_FREE(power);
    psf_destroy_image(gray);
    
    return PSF_SUCCESS;
}
// ============================================================================
// Utility Functions
// ============================================================================

/**
 * @brief Save PSF kernel to file
 */
PSF_API PSFError psf_save_kernel(
    const PSFKernel *kernel,
    const char *filename)
{
    if (!kernel || !filename) {
        return PSF_ERROR_NULL_POINTER;
    }
    
    FILE *fp = fopen(filename, "w");
    if (!fp) {
        return PSF_ERROR_FILE_IO;
    }
    
    // Write header
    fprintf(fp, "# PSF Kernel\n");
    fprintf(fp, "# Width: %d\n", kernel->width);
    fprintf(fp, "# Height: %d\n", kernel->height);
    fprintf(fp, "# Center: (%.6f, %.6f)\n", kernel->center_x, kernel->center_y);
    fprintf(fp, "\n");
    
    // Write data
    for (int y = 0; y < kernel->height; y++) {
        for (int x = 0; x < kernel->width; x++) {
            fprintf(fp, "%.10e", kernel->data[y * kernel->width + x]);
            if (x < kernel->width - 1) {
                fprintf(fp, " ");
            }
        }
        fprintf(fp, "\n");
    }
    
    fclose(fp);
    
    return PSF_SUCCESS;
}

/**
 * @brief Load PSF kernel from file
 */
PSF_API PSFError psf_load_kernel(
    const char *filename,
    PSFKernel **kernel)
{
    if (!filename || !kernel) {
        return PSF_ERROR_NULL_POINTER;
    }
    
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        return PSF_ERROR_FILE_IO;
    }
    
    int width = 0, height = 0;
    double center_x = 0.0, center_y = 0.0;
    
    // Read header
    char line[256];
    while (fgets(line, sizeof(line), fp)) {
        if (line[0] == '#') {
            if (sscanf(line, "# Width: %d", &width) == 1) continue;
            if (sscanf(line, "# Height: %d", &height) == 1) continue;
            if (sscanf(line, "# Center: (%lf, %lf)", &center_x, &center_y) == 2) continue;
        } else if (line[0] != '\n') {
            break;  // End of header
        }
    }
    
    if (width <= 0 || height <= 0) {
        fclose(fp);
        return PSF_ERROR_INVALID_FORMAT;
    }
    
    // Create kernel
    PSFError err = psf_create_kernel(width, height, kernel);
    if (err != PSF_SUCCESS) {
        fclose(fp);
        return err;
    }
    
    (*kernel)->center_x = center_x;
    (*kernel)->center_y = center_y;
    
    // Read data
    rewind(fp);
    
    // Skip header
    while (fgets(line, sizeof(line), fp)) {
        if (line[0] != '#' && line[0] != '\n') {
            break;
        }
    }
    
    // Read first line (already in buffer)
    int idx = 0;
    char *token = strtok(line, " \t\n");
    while (token && idx < width * height) {
        (*kernel)->data[idx++] = atof(token);
        token = strtok(NULL, " \t\n");
    }
    
    // Read remaining lines
    while (fgets(line, sizeof(line), fp) && idx < width * height) {
        token = strtok(line, " \t\n");
        while (token && idx < width * height) {
            (*kernel)->data[idx++] = atof(token);
            token = strtok(NULL, " \t\n");
        }
    }
    
    fclose(fp);
    
    if (idx != width * height) {
        psf_destroy_kernel(*kernel);
        *kernel = NULL;
        return PSF_ERROR_INVALID_FORMAT;
    }
    
    return PSF_SUCCESS;
}

/**
 * @brief Visualize PSF as ASCII art
 */
PSF_API PSFError psf_print_kernel(
    const PSFKernel *kernel,
    int precision)
{
    if (!kernel) {
        return PSF_ERROR_NULL_POINTER;
    }
    
    printf("PSF Kernel (%dx%d), Center: (%.2f, %.2f)\n",
           kernel->width, kernel->height,
           kernel->center_x, kernel->center_y);
    printf("----------------------------------------\n");
    
    // Find max value for normalization
    double max_val = 0.0;
    for (int i = 0; i < kernel->width * kernel->height; i++) {
        if (kernel->data[i] > max_val) {
            max_val = kernel->data[i];
        }
    }
    
    if (max_val < PSF_EPSILON) {
        printf("(Empty kernel)\n");
        return PSF_SUCCESS;
    }
    
    // Print values
    for (int y = 0; y < kernel->height; y++) {
        for (int x = 0; x < kernel->width; x++) {
            double val = kernel->data[y * kernel->width + x];
            printf("%*.*f ", precision + 3, precision, val);
        }
        printf("\n");
    }
    
    // Print ASCII visualization
    printf("\nVisualization:\n");
    const char *chars = " .:-=+*#%@";
    int num_chars = strlen(chars);
    
    for (int y = 0; y < kernel->height; y++) {
        for (int x = 0; x < kernel->width; x++) {
            double val = kernel->data[y * kernel->width + x];
            int char_idx = (int)(val / max_val * (num_chars - 1));
            printf("%c", chars[char_idx]);
        }
        printf("\n");
    }
    
    return PSF_SUCCESS;
}

/**
 * @brief Compare two PSF kernels
 */
PSF_API PSFError psf_compare_kernels(
    const PSFKernel *kernel1,
    const PSFKernel *kernel2,
    double *mse,
    double *correlation)
{
    if (!kernel1 || !kernel2 || !mse || !correlation) {
        return PSF_ERROR_NULL_POINTER;
    }
    
    if (kernel1->width != kernel2->width || 
        kernel1->height != kernel2->height) {
        return PSF_ERROR_DIMENSION_MISMATCH;
    }
    
    int size = kernel1->width * kernel1->height;
    
    // Compute MSE
    double sum_sq_diff = 0.0;
    
    for (int i = 0; i < size; i++) {
        double diff = kernel1->data[i] - kernel2->data[i];
        sum_sq_diff += diff * diff;
    }
    
    *mse = sum_sq_diff / size;
    
    // Compute correlation coefficient
    double mean1 = 0.0, mean2 = 0.0;
    
    for (int i = 0; i < size; i++) {
        mean1 += kernel1->data[i];
        mean2 += kernel2->data[i];
    }
    mean1 /= size;
    mean2 /= size;
    
    double cov = 0.0;
    double var1 = 0.0, var2 = 0.0;
    
    for (int i = 0; i < size; i++) {
        double diff1 = kernel1->data[i] - mean1;
        double diff2 = kernel2->data[i] - mean2;
        
        cov += diff1 * diff2;
        var1 += diff1 * diff1;
        var2 += diff2 * diff2;
    }
    
    double denom = sqrt(var1 * var2);
    if (denom > PSF_EPSILON) {
        *correlation = cov / denom;
    } else {
        *correlation = 0.0;
    }
    
    return PSF_SUCCESS;
}

/**
 * @brief Interpolate PSF at sub-pixel location
 */
PSF_API PSFError psf_interpolate_kernel(
    const PSFKernel *kernel,
    double x,
    double y,
    double *value)
{
    if (!kernel || !value) {
        return PSF_ERROR_NULL_POINTER;
    }
    
    // Bilinear interpolation
    int x0 = (int)floor(x);
    int y0 = (int)floor(y);
    int x1 = x0 + 1;
    int y1 = y0 + 1;
    
    if (x0 < 0 || x1 >= kernel->width || 
        y0 < 0 || y1 >= kernel->height) {
        *value = 0.0;
        return PSF_SUCCESS;
    }
    
    double fx = x - x0;
    double fy = y - y0;
    
    double v00 = kernel->data[y0 * kernel->width + x0];
    double v10 = kernel->data[y0 * kernel->width + x1];
    double v01 = kernel->data[y1 * kernel->width + x0];
    double v11 = kernel->data[y1 * kernel->width + x1];
    
    *value = v00 * (1 - fx) * (1 - fy) +
             v10 * fx * (1 - fy) +
             v01 * (1 - fx) * fy +
             v11 * fx * fy;
    
    return PSF_SUCCESS;
}

/**
 * @brief Compute PSF statistics
 */
PSF_API PSFError psf_compute_statistics(
    const PSFKernel *kernel,
    PSFStatistics *stats)
{
    if (!kernel || !stats) {
        return PSF_ERROR_NULL_POINTER;
    }
    
    int size = kernel->width * kernel->height;
    
    // Find min and max
    stats->min = DBL_MAX;
    stats->max = -DBL_MAX;
    
    for (int i = 0; i < size; i++) {
        if (kernel->data[i] < stats->min) {
            stats->min = kernel->data[i];
        }
        if (kernel->data[i] > stats->max) {
            stats->max = kernel->data[i];
        }
    }
    
    // Compute mean
    stats->mean = 0.0;
    for (int i = 0; i < size; i++) {
        stats->mean += kernel->data[i];
    }
    stats->mean /= size;
    
    // Compute standard deviation
    stats->std_dev = 0.0;
    for (int i = 0; i < size; i++) {
        double diff = kernel->data[i] - stats->mean;
        stats->std_dev += diff * diff;
    }
    stats->std_dev = sqrt(stats->std_dev / size);
    
    // Compute sum
    stats->sum = stats->mean * size;
    
    // Compute energy
    stats->energy = 0.0;
    for (int i = 0; i < size; i++) {
        stats->energy += kernel->data[i] * kernel->data[i];
    }
    
    // Compute entropy
    stats->entropy = 0.0;
    for (int i = 0; i < size; i++) {
        double p = kernel->data[i] / (stats->sum + PSF_EPSILON);
        if (p > PSF_EPSILON) {
            stats->entropy -= p * log2(p);
        }
    }
    
    return PSF_SUCCESS;
}

/**
 * @brief Get library version
 */
PSF_API const char* psf_get_version(void)
{
    return PSF_VERSION_STRING;
}

/**
 * @brief Get error string
 */
PSF_API const char* psf_get_error_string(PSFError error)
{
    switch (error) {
        case PSF_SUCCESS:
            return "Success";
        case PSF_ERROR_NULL_POINTER:
            return "Null pointer error";
        case PSF_ERROR_INVALID_PARAM:
            return "Invalid parameter";
        case PSF_ERROR_MEMORY_ALLOCATION:
            return "Memory allocation failed";
        case PSF_ERROR_DIMENSION_MISMATCH:
            return "Dimension mismatch";
        case PSF_ERROR_FILE_IO:
            return "File I/O error";
        case PSF_ERROR_INVALID_FORMAT:
            return "Invalid format";
        case PSF_ERROR_NOT_IMPLEMENTED:
            return "Not implemented";
        case PSF_ERROR_CONVERGENCE:
            return "Convergence failed";
        case PSF_ERROR_INVALID_PSF:
            return "Invalid PSF";
        case PSF_ERROR_NO_EDGES_FOUND:
            return "No edges found";
        case PSF_ERROR_NO_STARS_FOUND:
            return "No stars found";
        case PSF_ERROR_INSUFFICIENT_DATA:
            return "Insufficient data";
        default:
            return "Unknown error";
    }
}

// ============================================================================
// Complete Usage Examples
// ============================================================================

/**
 * @brief Example 1: Estimate PSF from edge
 */
void example_edge_based_psf(void)
{
    printf("=== Example 1: Edge-based PSF Estimation ===\n\n");
    
    // Load image
    PSFImage *image = NULL;
    PSFError err = psf_load_image("test_image.raw", 640, 480, 1, &image);
    if (err != PSF_SUCCESS) {
        printf("Error loading image: %s\n", psf_get_error_string(err));
        return;
    }
    
    // Configure estimation
    PSFEstimationConfig config = {
        .edge_detection_method = PSF_EDGE_CANNY,
        .edge_threshold_low = 0.05,
        .edge_threshold_high = 0.15,
        .psf_kernel_size = 15
    };
    
    // Estimate PSF
    PSFKernel *psf = NULL;
    err = psf_estimate_from_edge(image, &config, &psf);
    
    if (err == PSF_SUCCESS) {
        printf("PSF estimation successful!\n");
        
        // Print PSF
        psf_print_kernel(psf, 4);
        
        // Compute statistics
        PSFStatistics stats;
        psf_compute_statistics(psf, &stats);
        
        printf("\nPSF Statistics:\n");
        printf("  Mean: %.6f\n", stats.mean);
        printf("  Std Dev: %.6f\n", stats.std_dev);
        printf("  Min: %.6f\n", stats.min);
        printf("  Max: %.6f\n", stats.max);
        printf("  Energy: %.6f\n", stats.energy);
        printf("  Entropy: %.6f\n", stats.entropy);
        
        // Save PSF
        psf_save_kernel(psf, "estimated_psf.txt");
        
        psf_destroy_kernel(psf);
    } else {
        printf("Error estimating PSF: %s\n", psf_get_error_string(err));
    }
    
    psf_destroy_image(image);
}

/**
 * @brief Example 2: Estimate PSF from stars
 */
void example_star_based_psf(void)
{
    printf("\n=== Example 2: Star-based PSF Estimation ===\n\n");
    
    // Load astronomical image
    PSFImage *image = NULL;
    PSFError err = psf_load_image("star_field.raw", 1024, 1024, 1, &image);
    if (err != PSF_SUCCESS) {
        printf("Error loading image: %s\n", psf_get_error_string(err));
        return;
    }
    
    // Configure star detection
    PSFEstimationConfig config = {
        .star_detection_threshold = 0.1,
        .star_min_size = 3,
        .star_max_size = 20,
        .star_roundness_threshold = 0.5,
        .psf_kernel_size = 21
    };
    
    // Detect stars
    PSFStar *stars = NULL;
    int num_stars = 0;
    
    err = psf_detect_stars(image, &config, &stars, &num_stars);
    
    if (err == PSF_SUCCESS) {
        printf("Detected %d stars\n\n", num_stars);
        
        // Print top 5 stars
        printf("Top 5 brightest stars:\n");
        for (int i = 0; i < PSF_MIN(5, num_stars); i++) {
            printf("  Star %d: pos=(%.1f, %.1f), FWHM=%.2f, SNR=%.1f, ellip=%.3f\n",
                   i + 1,
                   stars[i].center.x, stars[i].center.y,
                   stars[i].fwhm,
                   stars[i].snr,
                   stars[i].ellipticity);
        }
        
        // Estimate PSF from stars
        PSFKernel *psf = NULL;
        err = psf_estimate_from_stars(image, &config, &psf);
        
        if (err == PSF_SUCCESS) {
            printf("\nPSF estimation from stars successful!\n");
            
            // Fit Gaussian model
            PSFGaussianModel gaussian_model;
            psf_fit_gaussian_model(psf, &gaussian_model);
            
            printf("\nGaussian Model:\n");
            printf("  Center: (%.2f, %.2f)\n", 
                   gaussian_model.center_x, gaussian_model.center_y);
            printf("  Sigma X: %.3f\n", gaussian_model.sigma_x);
            printf("  Sigma Y: %.3f\n", gaussian_model.sigma_y);
            printf("  Rotation: %.3f rad\n", gaussian_model.rotation);
            printf("  FWHM: %.3f pixels\n", 
                   2.355 * sqrt(gaussian_model.sigma_x * gaussian_model.sigma_y));
            printf("  Fit Quality: %.4f\n", gaussian_model.fit_quality);
            
            // Fit Moffat model
            PSFMoffatModel moffat_model;
            psf_fit_moffat_model(psf, &moffat_model);
            
            printf("\nMoffat Model:\n");
            printf("  Center: (%.2f, %.2f)\n", 
                   moffat_model.center_x, moffat_model.center_y);
            printf("  Alpha: %.3f\n", moffat_model.alpha);
            printf("  Beta: %.3f\n", moffat_model.beta);
            printf("  Fit Quality: %.4f\n", moffat_model.fit_quality);
            
            psf_destroy_kernel(psf);
        }
        
        psf_destroy_stars(stars);
    } else {
        printf("Error detecting stars: %s\n", psf_get_error_string(err));
    }
    
    psf_destroy_image(image);
}

/**
 * @brief Example 3: Blind deconvolution
 */
void example_blind_deconvolution(void)
{
    printf("\n=== Example 3: Blind Deconvolution ===\n\n");
    
    // Load blurred image
    PSFImage *blurred = NULL;
    PSFError err = psf_load_image("blurred.raw", 512, 512, 1, &blurred);
    if (err != PSF_SUCCESS) {
        printf("Error loading image: %s\n", psf_get_error_string(err));
        return;
    }
    
    // Configure blind deconvolution
    PSFEstimationConfig config = {
        .blind_deconv_iterations = 50,
        .blind_deconv_psf_size = 15,
        .blind_deconv_regularization = 0.001
    };
    
    // Perform blind deconvolution
    PSFImage *restored = NULL;
    PSFKernel *psf = NULL;
    
    printf("Running blind deconvolution (50 iterations)...\n");
    
    err = psf_blind_deconvolution(blurred, &config, &restored, &psf);
    
    if (err == PSF_SUCCESS) {
        printf("Blind deconvolution successful!\n\n");
        
        // Print estimated PSF
        printf("Estimated PSF:\n");
        psf_print_kernel(psf, 4);
        
        // Compute quality metrics
        PSFQualityMetrics metrics;
        psf_compute_quality_metrics(blurred, restored, &metrics);
        
        printf("\nQuality Metrics:\n");
        printf("  PSNR: %.2f dB\n", metrics.psnr);
        printf("  SSIM: %.4f\n", metrics.ssim);
        printf("  MSE: %.6f\n", metrics.mse);
        printf("  Sharpness: %.2f\n", metrics.sharpness);
        
        // Save results
        psf_save_image(restored, "restored.raw");
        psf_save_kernel(psf, "estimated_blur_kernel.txt");
        
        psf_destroy_image(restored);
        psf_destroy_kernel(psf);
    } else {
        printf("Error in blind deconvolution: %s\n", psf_get_error_string(err));
    }
    
    psf_destroy_image(blurred);
}

/**
 * @brief Example 4: MTF analysis
 */
void example_mtf_analysis(void)
{
    printf("\n=== Example 4: MTF Analysis ===\n\n");
    
    // Create a test PSF (Gaussian)
    PSFGaussianModel model = {
        .center_x = 7.0,
        .center_y = 7.0,
        .sigma_x = 2.0,
        .sigma_y = 2.0,
        .rotation = 0.0,
        .amplitude = 1.0
    };
    
    PSFKernel *psf = NULL;
    psf_generate_gaussian(&model, 15, &psf);
    
    printf("Generated Gaussian PSF (sigma=2.0)\n\n");
    
    // Compute 2D MTF
    PSFMTF2D *mtf2d = NULL;
    PSFError err = psf_compute_mtf_from_psf(psf, &mtf2d);
    
    if (err == PSF_SUCCESS) {
        // Compute radial MTF profile
        double *frequencies = NULL;
        double *values = NULL;
        int length = 0;
        
        psf_compute_radial_mtf(mtf2d, 50, &frequencies, &values, &length);
        
        printf("Radial MTF Profile:\n");
        printf("Frequency (cycles/pixel)  MTF\n");
        printf("------------------------  -----\n");
        
        for (int i = 0; i < length; i += 5) {
            printf("  %6.3f                  %.4f\n", 
                   frequencies[i], values[i]);
        }
        
        // Compute MTF50
        PSFMTF mtf1d = {
            .length = length,
            .frequencies = frequencies,
            .values = values
        };
        
        double mtf50;
        psf_compute_mtf50(&mtf1d, &mtf50);
        
        printf("\nMTF50: %.3f cycles/pixel\n", mtf50);
        
        PSF_FREE(frequencies);
        PSF_FREE(values);
        psf_destroy_mtf2d(mtf2d);
    }
    
    psf_destroy_kernel(psf);
}

/**
 * @brief Example 5: Image quality assessment
 */
void example_quality_assessment(void)
{
    printf("\n=== Example 5: Image Quality Assessment ===\n\n");
    
    // Load reference and test images
    PSFImage *reference = NULL;
    PSFImage *test = NULL;
    
    PSFError err = psf_load_image("reference.raw", 512, 512, 1, &reference);
    if (err != PSF_SUCCESS) {
        printf("Error loading reference: %s\n", psf_get_error_string(err));
        return;
    }
    
    err = psf_load_image("test.raw", 512, 512, 1, &test);
    if (err != PSF_SUCCESS) {
        printf("Error loading test: %s\n", psf_get_error_string(err));
        psf_destroy_image(reference);
        return;
    }
    
    // Compute quality metrics
    PSFQualityMetrics metrics;
    err = psf_compute_quality_metrics(reference, test, &metrics);
    
    if (err == PSF_SUCCESS) {
        printf("Image Quality Metrics:\n");
        printf("  PSNR: %.2f dB\n", metrics.psnr);
        printf("  SSIM: %.4f\n", metrics.ssim);
        printf("  MSE: %.6f\n", metrics.mse);
        printf("  MAE: %.6f\n", metrics.mae);
        printf("  Sharpness: %.2f\n", metrics.sharpness);
        
        // Estimate blur amount
        double blur_amount;
        psf_estimate_blur_amount(test, &blur_amount);
        printf("  Estimated blur: %.2f pixels\n", blur_amount);
        
        // Compute frequency concentration
        double concentration;
        psf_compute_frequency_concentration(test, &concentration);
        printf("  Frequency concentration: %.4f\n", concentration);
        
        // Interpretation
        printf("\nInterpretation:\n");
        if (metrics.ssim > 0.95) {
            printf("  - Excellent quality (SSIM > 0.95)\n");
        } else if (metrics.ssim > 0.85) {
            printf("  - Good quality (SSIM > 0.85)\n");
        } else if (metrics.ssim > 0.70) {
            printf("  - Fair quality (SSIM > 0.70)\n");
        } else {
            printf("  - Poor quality (SSIM < 0.70)\n");
        }
        
        if (blur_amount > 3.0) {
            printf("  - Significant blur detected\n");
        } else if (blur_amount > 1.5) {
            printf("  - Moderate blur detected\n");
        } else {
            printf("  - Minimal blur\n");
        }
    }
    
    psf_destroy_image(reference);
    psf_destroy_image(test);
}

/**
 * @brief Main function with all examples
 */
int main(int argc, char *argv[])
{
    printf("PSF Estimation Library v%s\n", psf_get_version());
    printf("========================================\n\n");
    
    // Run examples
    example_edge_based_psf();
    example_star_based_psf();
    example_blind_deconvolution();
    example_mtf_analysis();
    example_quality_assessment();
    
    printf("\n========================================\n");
    printf("All examples completed!\n");
    
    return 0;
}

