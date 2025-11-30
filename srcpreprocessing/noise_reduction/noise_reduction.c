// ============================================================================
// noise_reduction.c - Implementation File Part 1/6
// Basic Functions and Image Management
// ============================================================================

#include "noise_reduction.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <time.h>
#include <float.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ============================================================================
// Global Variables
// ============================================================================

static struct {
    bool initialized;
    bool multithreading_enabled;
    int num_threads;
    bool gpu_enabled;
    bool simd_enabled;
    int verbosity;
    bool logging_enabled;
    FILE *log_file;
    size_t memory_limit;
    size_t memory_used;
    char temp_directory[256];
    clock_t last_operation_start;
    double last_operation_time;
} g_config = {
    .initialized = false,
    .multithreading_enabled = true,
    .num_threads = 4,
    .gpu_enabled = false,
    .simd_enabled = true,
    .verbosity = 1,
    .logging_enabled = false,
    .log_file = NULL,
    .memory_limit = 0,
    .memory_used = 0,
    .temp_directory = "/tmp",
    .last_operation_start = 0,
    .last_operation_time = 0.0
};

// ============================================================================
// Internal Helper Functions
// ============================================================================

static void log_message(const char *format, ...) {
    if (!g_config.logging_enabled || !g_config.log_file) return;
    
    va_list args;
    va_start(args, format);
    vfprintf(g_config.log_file, format, args);
    va_end(args);
    fflush(g_config.log_file);
}

static void start_timer(void) {
    g_config.last_operation_start = clock();
}

static void stop_timer(void) {
    clock_t end = clock();
    g_config.last_operation_time = 
        (double)(end - g_config.last_operation_start) / CLOCKS_PER_SEC;
}

static inline int min_int(int a, int b) {
    return a < b ? a : b;
}

static inline int max_int(int a, int b) {
    return a > b ? a : b;
}

static inline double min_double(double a, double b) {
    return a < b ? a : b;
}

static inline double max_double(double a, double b) {
    return a > b ? a : b;
}

static inline int clamp_int(int value, int min_val, int max_val) {
    return max_int(min_val, min_int(max_val, value));
}

static inline double clamp_double(double value, double min_val, double max_val) {
    return max_double(min_val, min_double(max_val, value));
}

// ============================================================================
// Error Handling
// ============================================================================

const char* noise_error_string(NoiseError error) {
    switch (error) {
        case NOISE_SUCCESS:
            return "Success";
        case NOISE_ERROR_INVALID_PARAM:
            return "Invalid parameter";
        case NOISE_ERROR_OUT_OF_MEMORY:
            return "Out of memory";
        case NOISE_ERROR_INVALID_IMAGE:
            return "Invalid image";
        case NOISE_ERROR_UNSUPPORTED_FORMAT:
            return "Unsupported format";
        case NOISE_ERROR_PROCESSING_FAILED:
            return "Processing failed";
        case NOISE_ERROR_FILE_IO:
            return "File I/O error";
        case NOISE_ERROR_DIMENSION_MISMATCH:
            return "Dimension mismatch";
        case NOISE_ERROR_NOT_IMPLEMENTED:
            return "Not implemented";
        default:
            return "Unknown error";
    }
}

const char* image_format_name(ImageFormat format) {
    switch (format) {
        case IMAGE_FORMAT_GRAYSCALE: return "Grayscale";
        case IMAGE_FORMAT_RGB: return "RGB";
        case IMAGE_FORMAT_RGBA: return "RGBA";
        case IMAGE_FORMAT_YUV: return "YUV";
        case IMAGE_FORMAT_HSV: return "HSV";
        case IMAGE_FORMAT_LAB: return "LAB";
        default: return "Unknown";
    }
}

const char* noise_type_name(NoiseType type) {
    switch (type) {
        case NOISE_TYPE_GAUSSIAN: return "Gaussian";
        case NOISE_TYPE_SALT_PEPPER: return "Salt & Pepper";
        case NOISE_TYPE_POISSON: return "Poisson";
        case NOISE_TYPE_SPECKLE: return "Speckle";
        case NOISE_TYPE_UNIFORM: return "Uniform";
        case NOISE_TYPE_IMPULSE: return "Impulse";
        case NOISE_TYPE_PERIODIC: return "Periodic";
        case NOISE_TYPE_UNKNOWN: return "Unknown";
        default: return "Invalid";
    }
}

// ============================================================================
// Image Management Functions
// ============================================================================

size_t get_data_type_size(DataType type) {
    switch (type) {
        case DATA_TYPE_UINT8: return sizeof(uint8_t);
        case DATA_TYPE_UINT16: return sizeof(uint16_t);
        case DATA_TYPE_FLOAT32: return sizeof(float);
        case DATA_TYPE_FLOAT64: return sizeof(double);
        default: return 0;
    }
}

int get_bytes_per_pixel(const Image *image) {
    if (!image) return 0;
    return image->channels * get_data_type_size(image->data_type);
}

bool image_is_valid(const Image *image) {
    if (!image) return false;
    if (!image->data) return false;
    if (image->width <= 0 || image->height <= 0) return false;
    if (image->channels <= 0 || image->channels > 4) return false;
    if (image->stride < image->width * get_bytes_per_pixel(image)) return false;
    return true;
}

bool images_are_compatible(const Image *image1, const Image *image2) {
    if (!image_is_valid(image1) || !image_is_valid(image2)) return false;
    if (image1->width != image2->width) return false;
    if (image1->height != image2->height) return false;
    if (image1->channels != image2->channels) return false;
    if (image1->format != image2->format) return false;
    return true;
}

NoiseError image_create_typed(
    Image **image,
    int width,
    int height,
    int channels,
    ImageFormat format,
    DataType data_type)
{
    if (!image) return NOISE_ERROR_INVALID_PARAM;
    if (width <= 0 || height <= 0) return NOISE_ERROR_INVALID_PARAM;
    if (channels <= 0 || channels > 4) return NOISE_ERROR_INVALID_PARAM;
    
    *image = (Image*)malloc(sizeof(Image));
    if (!*image) return NOISE_ERROR_OUT_OF_MEMORY;
    
    (*image)->width = width;
    (*image)->height = height;
    (*image)->channels = channels;
    (*image)->format = format;
    (*image)->data_type = data_type;
    
    size_t pixel_size = channels * get_data_type_size(data_type);
    (*image)->stride = width * pixel_size;
    (*image)->data_size = (*image)->stride * height;
    
    // Check memory limit
    if (g_config.memory_limit > 0 && 
        g_config.memory_used + (*image)->data_size > g_config.memory_limit) {
        free(*image);
        *image = NULL;
        return NOISE_ERROR_OUT_OF_MEMORY;
    }
    
    (*image)->data = malloc((*image)->data_size);
    if (!(*image)->data) {
        free(*image);
        *image = NULL;
        return NOISE_ERROR_OUT_OF_MEMORY;
    }
    
    memset((*image)->data, 0, (*image)->data_size);
    g_config.memory_used += (*image)->data_size;
    
    log_message("Created image: %dx%d, %d channels, format=%s, type=%d\n",
                width, height, channels, image_format_name(format), data_type);
    
    return NOISE_SUCCESS;
}

NoiseError image_create(
    Image **image,
    int width,
    int height,
    int channels,
    ImageFormat format)
{
    return image_create_typed(image, width, height, channels, 
                             format, DATA_TYPE_UINT8);
}

void image_destroy(Image *image) {
    if (!image) return;
    
    if (image->data) {
        g_config.memory_used -= image->data_size;
        free(image->data);
        image->data = NULL;
    }
    
    free(image);
    
    log_message("Destroyed image\n");
}

NoiseError image_clone(Image **dest, const Image *src) {
    if (!dest || !src) return NOISE_ERROR_INVALID_PARAM;
    if (!image_is_valid(src)) return NOISE_ERROR_INVALID_IMAGE;
    
    NoiseError err = image_create_typed(dest, src->width, src->height,
                                       src->channels, src->format, 
                                       src->data_type);
    if (err != NOISE_SUCCESS) return err;
    
    memcpy((*dest)->data, src->data, src->data_size);
    
    return NOISE_SUCCESS;
}

NoiseError image_copy(Image *dest, const Image *src) {
    if (!dest || !src) return NOISE_ERROR_INVALID_PARAM;
    if (!images_are_compatible(dest, src)) return NOISE_ERROR_DIMENSION_MISMATCH;
    
    memcpy(dest->data, src->data, src->data_size);
    
    return NOISE_SUCCESS;
}

// ============================================================================
// Pixel Access Functions
// ============================================================================

NoiseError image_get_pixel(
    const Image *image,
    int x,
    int y,
    void *pixel)
{
    if (!image || !pixel) return NOISE_ERROR_INVALID_PARAM;
    if (!image_is_valid(image)) return NOISE_ERROR_INVALID_IMAGE;
    if (x < 0 || x >= image->width || y < 0 || y >= image->height) {
        return NOISE_ERROR_INVALID_PARAM;
    }
    
    size_t pixel_size = get_bytes_per_pixel(image);
    uint8_t *src = (uint8_t*)image->data + y * image->stride + x * pixel_size;
    memcpy(pixel, src, pixel_size);
    
    return NOISE_SUCCESS;
}

NoiseError image_set_pixel(
    Image *image,
    int x,
    int y,
    const void *pixel)
{
    if (!image || !pixel) return NOISE_ERROR_INVALID_PARAM;
    if (!image_is_valid(image)) return NOISE_ERROR_INVALID_IMAGE;
    if (x < 0 || x >= image->width || y < 0 || y >= image->height) {
        return NOISE_ERROR_INVALID_PARAM;
    }
    
    size_t pixel_size = get_bytes_per_pixel(image);
    uint8_t *dst = (uint8_t*)image->data + y * image->stride + x * pixel_size;
    memcpy(dst, pixel, pixel_size);
    
    return NOISE_SUCCESS;
}

NoiseError image_get_pixel_safe(
    const Image *image,
    int x,
    int y,
    void *pixel,
    int boundary_mode)
{
    if (!image || !pixel) return NOISE_ERROR_INVALID_PARAM;
    if (!image_is_valid(image)) return NOISE_ERROR_INVALID_IMAGE;
    
    // Handle boundary conditions
    int safe_x = x, safe_y = y;
    
    switch (boundary_mode) {
        case 0: // Clamp
            safe_x = clamp_int(x, 0, image->width - 1);
            safe_y = clamp_int(y, 0, image->height - 1);
            break;
            
        case 1: // Mirror
            if (x < 0) safe_x = -x;
            else if (x >= image->width) safe_x = 2 * image->width - x - 2;
            else safe_x = x;
            
            if (y < 0) safe_y = -y;
            else if (y >= image->height) safe_y = 2 * image->height - y - 2;
            else safe_y = y;
            break;
            
        case 2: // Wrap
            safe_x = (x + image->width) % image->width;
            safe_y = (y + image->height) % image->height;
            break;
            
        default:
            return NOISE_ERROR_INVALID_PARAM;
    }
    
    return image_get_pixel(image, safe_x, safe_y, pixel);
}

// ============================================================================
// Image Statistics Functions
// ============================================================================

NoiseError image_mean(
    double *mean,
    const Image *image,
    int channel)
{
    if (!mean || !image) return NOISE_ERROR_INVALID_PARAM;
    if (!image_is_valid(image)) return NOISE_ERROR_INVALID_IMAGE;
    if (channel < -1 || channel >= image->channels) return NOISE_ERROR_INVALID_PARAM;
    
    double sum = 0.0;
    int count = 0;
    
    if (image->data_type == DATA_TYPE_UINT8) {
        uint8_t *data = (uint8_t*)image->data;
        
        for (int y = 0; y < image->height; y++) {
            for (int x = 0; x < image->width; x++) {
                int idx = y * image->stride + x * image->channels;
                
                if (channel == -1) {
                    // All channels
                    for (int c = 0; c < image->channels; c++) {
                        sum += data[idx + c];
                        count++;
                    }
                } else {
                    sum += data[idx + channel];
                    count++;
                }
            }
        }
    } else if (image->data_type == DATA_TYPE_FLOAT32) {
        float *data = (float*)image->data;
        
        for (int y = 0; y < image->height; y++) {
            for (int x = 0; x < image->width; x++) {
                int idx = (y * image->stride / sizeof(float)) + x * image->channels;
                
                if (channel == -1) {
                    for (int c = 0; c < image->channels; c++) {
                        sum += data[idx + c];
                        count++;
                    }
                } else {
                    sum += data[idx + channel];
                    count++;
                }
            }
        }
    }
    
    *mean = count > 0 ? sum / count : 0.0;
    
    return NOISE_SUCCESS;
}

NoiseError image_variance(
    double *variance,
    const Image *image,
    int channel)
{
    if (!variance || !image) return NOISE_ERROR_INVALID_PARAM;
    
    double mean_val;
    NoiseError err = image_mean(&mean_val, image, channel);
    if (err != NOISE_SUCCESS) return err;
    
    double sum_sq = 0.0;
    int count = 0;
    
    if (image->data_type == DATA_TYPE_UINT8) {
        uint8_t *data = (uint8_t*)image->data;
        
        for (int y = 0; y < image->height; y++) {
            for (int x = 0; x < image->width; x++) {
                int idx = y * image->stride + x * image->channels;
                
                if (channel == -1) {
                    for (int c = 0; c < image->channels; c++) {
                        double diff = data[idx + c] - mean_val;
                        sum_sq += diff * diff;
                        count++;
                    }
                } else {
                    double diff = data[idx + channel] - mean_val;
                    sum_sq += diff * diff;
                    count++;
                }
            }
        }
    } else if (image->data_type == DATA_TYPE_FLOAT32) {
        float *data = (float*)image->data;
        
        for (int y = 0; y < image->height; y++) {
            for (int x = 0; x < image->width; x++) {
                int idx = (y * image->stride / sizeof(float)) + x * image->channels;
                
                if (channel == -1) {
                    for (int c = 0; c < image->channels; c++) {
                        double diff = data[idx + c] - mean_val;
                        sum_sq += diff * diff;
                        count++;
                    }
                } else {
                    double diff = data[idx + channel] - mean_val;
                    sum_sq += diff * diff;
                    count++;
                }
            }
        }
    }
    
    *variance = count > 0 ? sum_sq / count : 0.0;
    
    return NOISE_SUCCESS;
}

NoiseError image_std_dev(
    double *std_dev,
    const Image *image,
    int channel)
{
    if (!std_dev) return NOISE_ERROR_INVALID_PARAM;
    
    double variance;
    NoiseError err = image_variance(&variance, image, channel);
    if (err != NOISE_SUCCESS) return err;
    
    *std_dev = sqrt(variance);
    
    return NOISE_SUCCESS;
}

NoiseError image_min_max(
    double *min_val,
    double *max_val,
    const Image *image,
    int channel)
{
    if (!min_val || !max_val || !image) return NOISE_ERROR_INVALID_PARAM;
    if (!image_is_valid(image)) return NOISE_ERROR_INVALID_IMAGE;
    if (channel < -1 || channel >= image->channels) return NOISE_ERROR_INVALID_PARAM;
    
    *min_val = DBL_MAX;
    *max_val = -DBL_MAX;
    
    if (image->data_type == DATA_TYPE_UINT8) {
        uint8_t *data = (uint8_t*)image->data;
        
        for (int y = 0; y < image->height; y++) {
            for (int x = 0; x < image->width; x++) {
                int idx = y * image->stride + x * image->channels;
                
                if (channel == -1) {
                    for (int c = 0; c < image->channels; c++) {
                        double val = data[idx + c];
                        if (val < *min_val) *min_val = val;
                        if (val > *max_val) *max_val = val;
                    }
                } else {
                    double val = data[idx + channel];
                    if (val < *min_val) *min_val = val;
                    if (val > *max_val) *max_val = val;
                }
            }
        }
    } else if (image->data_type == DATA_TYPE_FLOAT32) {
        float *data = (float*)image->data;
        
        for (int y = 0; y < image->height; y++) {
            for (int x = 0; x < image->width; x++) {
                int idx = (y * image->stride / sizeof(float)) + x * image->channels;
                
                if (channel == -1) {
                    for (int c = 0; c < image->channels; c++) {
                        double val = data[idx + c];
                        if (val < *min_val) *min_val = val;
                        if (val > *max_val) *max_val = val;
                    }
                } else {
                    double val = data[idx + channel];
                    if (val < *min_val) *min_val = val;
                    if (val > *max_val) *max_val = val;
                }
            }
        }
    }
    
    return NOISE_SUCCESS;
}

// End of Part 1/6
// ============================================================================
// noise_reduction.c - Implementation File Part 2/6
// Image Statistics and Noise Detection
// ============================================================================

// ============================================================================
// Histogram Functions
// ============================================================================

NoiseError image_histogram(
    int *histogram,
    int bins,
    const Image *image,
    int channel)
{
    if (!histogram || !image) return NOISE_ERROR_INVALID_PARAM;
    if (!image_is_valid(image)) return NOISE_ERROR_INVALID_IMAGE;
    if (bins <= 0) return NOISE_ERROR_INVALID_PARAM;
    if (channel < 0 || channel >= image->channels) return NOISE_ERROR_INVALID_PARAM;
    
    // Initialize histogram
    memset(histogram, 0, bins * sizeof(int));
    
    double min_val, max_val;
    NoiseError err = image_min_max(&min_val, &max_val, image, channel);
    if (err != NOISE_SUCCESS) return err;
    
    double range = max_val - min_val;
    if (range < 1e-10) range = 1.0;
    
    if (image->data_type == DATA_TYPE_UINT8) {
        uint8_t *data = (uint8_t*)image->data;
        
        for (int y = 0; y < image->height; y++) {
            for (int x = 0; x < image->width; x++) {
                int idx = y * image->stride + x * image->channels + channel;
                double val = data[idx];
                int bin = (int)((val - min_val) / range * (bins - 1));
                bin = clamp_int(bin, 0, bins - 1);
                histogram[bin]++;
            }
        }
    } else if (image->data_type == DATA_TYPE_FLOAT32) {
        float *data = (float*)image->data;
        
        for (int y = 0; y < image->height; y++) {
            for (int x = 0; x < image->width; x++) {
                int idx = (y * image->stride / sizeof(float)) + 
                         x * image->channels + channel;
                double val = data[idx];
                int bin = (int)((val - min_val) / range * (bins - 1));
                bin = clamp_int(bin, 0, bins - 1);
                histogram[bin]++;
            }
        }
    }
    
    return NOISE_SUCCESS;
}

NoiseError image_entropy(
    double *entropy,
    const Image *image,
    int channel)
{
    if (!entropy || !image) return NOISE_ERROR_INVALID_PARAM;
    
    const int bins = 256;
    int histogram[256];
    
    NoiseError err = image_histogram(histogram, bins, image, channel);
    if (err != NOISE_SUCCESS) return err;
    
    int total_pixels = image->width * image->height;
    *entropy = 0.0;
    
    for (int i = 0; i < bins; i++) {
        if (histogram[i] > 0) {
            double p = (double)histogram[i] / total_pixels;
            *entropy -= p * log2(p);
        }
    }
    
    return NOISE_SUCCESS;
}

// ============================================================================
// Quality Metrics
// ============================================================================

NoiseError calculate_psnr(
    double *psnr,
    const Image *original,
    const Image *processed)
{
    if (!psnr || !original || !processed) return NOISE_ERROR_INVALID_PARAM;
    if (!images_are_compatible(original, processed)) {
        return NOISE_ERROR_DIMENSION_MISMATCH;
    }
    
    double mse = 0.0;
    int count = 0;
    
    if (original->data_type == DATA_TYPE_UINT8) {
        uint8_t *data1 = (uint8_t*)original->data;
        uint8_t *data2 = (uint8_t*)processed->data;
        
        for (int y = 0; y < original->height; y++) {
            for (int x = 0; x < original->width; x++) {
                for (int c = 0; c < original->channels; c++) {
                    int idx = y * original->stride + x * original->channels + c;
                    double diff = (double)data1[idx] - (double)data2[idx];
                    mse += diff * diff;
                    count++;
                }
            }
        }
        
        mse /= count;
        *psnr = (mse < 1e-10) ? 100.0 : 10.0 * log10(255.0 * 255.0 / mse);
        
    } else if (original->data_type == DATA_TYPE_FLOAT32) {
        float *data1 = (float*)original->data;
        float *data2 = (float*)processed->data;
        
        for (int y = 0; y < original->height; y++) {
            for (int x = 0; x < original->width; x++) {
                for (int c = 0; c < original->channels; c++) {
                    int idx = (y * original->stride / sizeof(float)) + 
                             x * original->channels + c;
                    double diff = data1[idx] - data2[idx];
                    mse += diff * diff;
                    count++;
                }
            }
        }
        
        mse /= count;
        *psnr = (mse < 1e-10) ? 100.0 : 10.0 * log10(1.0 / mse);
    }
    
    return NOISE_SUCCESS;
}

NoiseError calculate_ssim(
    double *ssim,
    const Image *original,
    const Image *processed)
{
    if (!ssim || !original || !processed) return NOISE_ERROR_INVALID_PARAM;
    if (!images_are_compatible(original, processed)) {
        return NOISE_ERROR_DIMENSION_MISMATCH;
    }
    
    const double C1 = 6.5025;  // (0.01 * 255)^2
    const double C2 = 58.5225; // (0.03 * 255)^2
    const int window_size = 11;
    const int half_window = window_size / 2;
    
    double ssim_sum = 0.0;
    int count = 0;
    
    if (original->data_type == DATA_TYPE_UINT8) {
        uint8_t *data1 = (uint8_t*)original->data;
        uint8_t *data2 = (uint8_t*)processed->data;
        
        for (int y = half_window; y < original->height - half_window; y++) {
            for (int x = half_window; x < original->width - half_window; x++) {
                // Calculate local statistics
                double mean1 = 0.0, mean2 = 0.0;
                double var1 = 0.0, var2 = 0.0, covar = 0.0;
                int window_count = 0;
                
                for (int wy = -half_window; wy <= half_window; wy++) {
                    for (int wx = -half_window; wx <= half_window; wx++) {
                        int idx = (y + wy) * original->stride + 
                                 (x + wx) * original->channels;
                        
                        double val1 = data1[idx];
                        double val2 = data2[idx];
                        
                        mean1 += val1;
                        mean2 += val2;
                        window_count++;
                    }
                }
                
                mean1 /= window_count;
                mean2 /= window_count;
                
                for (int wy = -half_window; wy <= half_window; wy++) {
                    for (int wx = -half_window; wx <= half_window; wx++) {
                        int idx = (y + wy) * original->stride + 
                                 (x + wx) * original->channels;
                        
                        double val1 = data1[idx] - mean1;
                        double val2 = data2[idx] - mean2;
                        
                        var1 += val1 * val1;
                        var2 += val2 * val2;
                        covar += val1 * val2;
                    }
                }
                
                var1 /= window_count;
                var2 /= window_count;
                covar /= window_count;
                
                // Calculate SSIM for this window
                double numerator = (2.0 * mean1 * mean2 + C1) * 
                                  (2.0 * covar + C2);
                double denominator = (mean1 * mean1 + mean2 * mean2 + C1) * 
                                    (var1 + var2 + C2);
                
                ssim_sum += numerator / denominator;
                count++;
            }
        }
    }
    
    *ssim = count > 0 ? ssim_sum / count : 0.0;
    
    return NOISE_SUCCESS;
}

NoiseError calculate_mse(
    double *mse,
    const Image *original,
    const Image *processed)
{
    if (!mse || !original || !processed) return NOISE_ERROR_INVALID_PARAM;
    if (!images_are_compatible(original, processed)) {
        return NOISE_ERROR_DIMENSION_MISMATCH;
    }
    
    double sum_sq_error = 0.0;
    int count = 0;
    
    if (original->data_type == DATA_TYPE_UINT8) {
        uint8_t *data1 = (uint8_t*)original->data;
        uint8_t *data2 = (uint8_t*)processed->data;
        
        for (int y = 0; y < original->height; y++) {
            for (int x = 0; x < original->width; x++) {
                for (int c = 0; c < original->channels; c++) {
                    int idx = y * original->stride + x * original->channels + c;
                    double diff = (double)data1[idx] - (double)data2[idx];
                    sum_sq_error += diff * diff;
                    count++;
                }
            }
        }
    } else if (original->data_type == DATA_TYPE_FLOAT32) {
        float *data1 = (float*)original->data;
        float *data2 = (float*)processed->data;
        
        for (int y = 0; y < original->height; y++) {
            for (int x = 0; x < original->width; x++) {
                for (int c = 0; c < original->channels; c++) {
                    int idx = (y * original->stride / sizeof(float)) + 
                             x * original->channels + c;
                    double diff = data1[idx] - data2[idx];
                    sum_sq_error += diff * diff;
                    count++;
                }
            }
        }
    }
    
    *mse = count > 0 ? sum_sq_error / count : 0.0;
    
    return NOISE_SUCCESS;
}

NoiseError calculate_mae(
    double *mae,
    const Image *original,
    const Image *processed)
{
    if (!mae || !original || !processed) return NOISE_ERROR_INVALID_PARAM;
    if (!images_are_compatible(original, processed)) {
        return NOISE_ERROR_DIMENSION_MISMATCH;
    }
    
    double sum_abs_error = 0.0;
    int count = 0;
    
    if (original->data_type == DATA_TYPE_UINT8) {
        uint8_t *data1 = (uint8_t*)original->data;
        uint8_t *data2 = (uint8_t*)processed->data;
        
        for (int y = 0; y < original->height; y++) {
            for (int x = 0; x < original->width; x++) {
                for (int c = 0; c < original->channels; c++) {
                    int idx = y * original->stride + x * original->channels + c;
                    double diff = fabs((double)data1[idx] - (double)data2[idx]);
                    sum_abs_error += diff;
                    count++;
                }
            }
        }
    }
    
    *mae = count > 0 ? sum_abs_error / count : 0.0;
    
    return NOISE_SUCCESS;
}

// ============================================================================
// Noise Detection Functions
// ============================================================================

NoiseError detect_noise_type(
    NoiseType *type,
    const Image *image)
{
    if (!type || !image) return NOISE_ERROR_INVALID_PARAM;
    if (!image_is_valid(image)) return NOISE_ERROR_INVALID_IMAGE;
    
    // Calculate various statistics
    double mean, variance, entropy;
    image_mean(&mean, image, -1);
    image_variance(&variance, image, -1);
    image_entropy(&entropy, image, 0);
    
    // Count extreme values (for salt & pepper detection)
    int extreme_count = 0;
    int total_pixels = image->width * image->height * image->channels;
    
    if (image->data_type == DATA_TYPE_UINT8) {
        uint8_t *data = (uint8_t*)image->data;
        
        for (int i = 0; i < total_pixels; i++) {
            if (data[i] == 0 || data[i] == 255) {
                extreme_count++;
            }
        }
    }
    
    double extreme_ratio = (double)extreme_count / total_pixels;
    
    // Decision logic
    if (extreme_ratio > 0.05) {
        *type = NOISE_TYPE_SALT_PEPPER;
    } else if (variance > 100.0 && entropy > 7.0) {
        *type = NOISE_TYPE_GAUSSIAN;
    } else if (variance < 50.0) {
        *type = NOISE_TYPE_POISSON;
    } else {
        *type = NOISE_TYPE_UNKNOWN;
    }
    
    log_message("Detected noise type: %s (mean=%.2f, var=%.2f, entropy=%.2f)\n",
                noise_type_name(*type), mean, variance, entropy);
    
    return NOISE_SUCCESS;
}

NoiseError estimate_noise_level(
    double *noise_level,
    const Image *image)
{
    if (!noise_level || !image) return NOISE_ERROR_INVALID_PARAM;
    if (!image_is_valid(image)) return NOISE_ERROR_INVALID_IMAGE;
    
    // Use median absolute deviation (MAD) method
    // Estimate noise from high-frequency components
    
    int width = image->width;
    int height = image->height;
    
    // Allocate array for differences
    int diff_count = (width - 1) * (height - 1);
    double *diffs = (double*)malloc(diff_count * sizeof(double));
    if (!diffs) return NOISE_ERROR_OUT_OF_MEMORY;
    
    int idx = 0;
    
    if (image->data_type == DATA_TYPE_UINT8) {
        uint8_t *data = (uint8_t*)image->data;
        
        for (int y = 0; y < height - 1; y++) {
            for (int x = 0; x < width - 1; x++) {
                int pos = y * image->stride + x * image->channels;
                double val = data[pos];
                double val_right = data[pos + image->channels];
                double val_down = data[pos + image->stride];
                
                // Laplacian-like operator
                double diff = fabs(2.0 * val - val_right - val_down);
                diffs[idx++] = diff;
            }
        }
    }
    
    // Sort differences
    for (int i = 0; i < diff_count - 1; i++) {
        for (int j = i + 1; j < diff_count; j++) {
            if (diffs[i] > diffs[j]) {
                double temp = diffs[i];
                diffs[i] = diffs[j];
                diffs[j] = temp;
            }
        }
    }
    
    // Calculate MAD
    double median = diffs[diff_count / 2];
    *noise_level = median / 0.6745; // Convert MAD to standard deviation
    
    free(diffs);
    
    log_message("Estimated noise level: %.2f\n", *noise_level);
    
    return NOISE_SUCCESS;
}

NoiseError estimate_noise_parameters(
    NoiseParams *params,
    const Image *image)
{
    if (!params || !image) return NOISE_ERROR_INVALID_PARAM;
    if (!image_is_valid(image)) return NOISE_ERROR_INVALID_IMAGE;
    
    // Detect noise type
    NoiseError err = detect_noise_type(&params->type, image);
    if (err != NOISE_SUCCESS) return err;
    
    // Estimate noise level
    err = estimate_noise_level(&params->level, image);
    if (err != NOISE_SUCCESS) return err;
    
    // Calculate additional statistics
    image_mean(&params->mean, image, -1);
    image_variance(&params->variance, image, -1);
    params->std_dev = sqrt(params->variance);
    
    // Estimate SNR
    double signal_power = params->mean * params->mean;
    double noise_power = params->variance;
    params->snr = (noise_power > 0) ? 
                  10.0 * log10(signal_power / noise_power) : 100.0;
    
    log_message("Noise parameters: type=%s, level=%.2f, SNR=%.2f dB\n",
                noise_type_name(params->type), params->level, params->snr);
    
    return NOISE_SUCCESS;
}

NoiseError analyze_noise_distribution(
    NoiseDistribution *dist,
    const Image *image)
{
    if (!dist || !image) return NOISE_ERROR_INVALID_PARAM;
    if (!image_is_valid(image)) return NOISE_ERROR_INVALID_IMAGE;
    
    // Calculate histogram
    const int bins = 256;
    int histogram[256];
    NoiseError err = image_histogram(histogram, bins, image, 0);
    if (err != NOISE_SUCCESS) return err;
    
    // Allocate distribution arrays
    dist->num_bins = bins;
    dist->bin_values = (double*)malloc(bins * sizeof(double));
    dist->frequencies = (double*)malloc(bins * sizeof(double));
    
    if (!dist->bin_values || !dist->frequencies) {
        free(dist->bin_values);
        free(dist->frequencies);
        return NOISE_ERROR_OUT_OF_MEMORY;
    }
    
    // Normalize histogram
    int total = image->width * image->height;
    for (int i = 0; i < bins; i++) {
        dist->bin_values[i] = i;
        dist->frequencies[i] = (double)histogram[i] / total;
    }
    
    // Calculate moments
    dist->mean = 0.0;
    dist->variance = 0.0;
    dist->skewness = 0.0;
    dist->kurtosis = 0.0;
    
    for (int i = 0; i < bins; i++) {
        dist->mean += dist->bin_values[i] * dist->frequencies[i];
    }
    
    for (int i = 0; i < bins; i++) {
        double diff = dist->bin_values[i] - dist->mean;
        dist->variance += diff * diff * dist->frequencies[i];
    }
    
    double std_dev = sqrt(dist->variance);
    
    if (std_dev > 1e-10) {
        for (int i = 0; i < bins; i++) {
            double z = (dist->bin_values[i] - dist->mean) / std_dev;
            dist->skewness += z * z * z * dist->frequencies[i];
            dist->kurtosis += z * z * z * z * dist->frequencies[i];
        }
    }
    
    dist->kurtosis -= 3.0; // Excess kurtosis
    
    return NOISE_SUCCESS;
}

// End of Part 2/6
// ============================================================================
// noise_reduction.c - Implementation File Part 3/6
// Basic Filtering Algorithms
// ============================================================================

// ============================================================================
// Gaussian Filter
// ============================================================================

static double* create_gaussian_kernel(int size, double sigma) {
    double *kernel = (double*)malloc(size * size * sizeof(double));
    if (!kernel) return NULL;
    
    int half = size / 2;
    double sum = 0.0;
    double sigma_sq = 2.0 * sigma * sigma;
    
    for (int y = -half; y <= half; y++) {
        for (int x = -half; x <= half; x++) {
            int idx = (y + half) * size + (x + half);
            double r_sq = x * x + y * y;
            kernel[idx] = exp(-r_sq / sigma_sq);
            sum += kernel[idx];
        }
    }
    
    // Normalize
    for (int i = 0; i < size * size; i++) {
        kernel[i] /= sum;
    }
    
    return kernel;
}

NoiseError gaussian_filter(
    Image *output,
    const Image *input,
    int kernel_size,
    double sigma)
{
    if (!output || !input) return NOISE_ERROR_INVALID_PARAM;
    if (!images_are_compatible(output, input)) return NOISE_ERROR_DIMENSION_MISMATCH;
    if (kernel_size <= 0 || kernel_size % 2 == 0) return NOISE_ERROR_INVALID_PARAM;
    if (sigma <= 0.0) return NOISE_ERROR_INVALID_PARAM;
    
    start_timer();
    
    double *kernel = create_gaussian_kernel(kernel_size, sigma);
    if (!kernel) return NOISE_ERROR_OUT_OF_MEMORY;
    
    int half = kernel_size / 2;
    
    if (input->data_type == DATA_TYPE_UINT8) {
        uint8_t *in_data = (uint8_t*)input->data;
        uint8_t *out_data = (uint8_t*)output->data;
        
        for (int y = 0; y < input->height; y++) {
            for (int x = 0; x < input->width; x++) {
                for (int c = 0; c < input->channels; c++) {
                    double sum = 0.0;
                    
                    for (int ky = -half; ky <= half; ky++) {
                        for (int kx = -half; kx <= half; kx++) {
                            int px = clamp_int(x + kx, 0, input->width - 1);
                            int py = clamp_int(y + ky, 0, input->height - 1);
                            
                            int in_idx = py * input->stride + px * input->channels + c;
                            int k_idx = (ky + half) * kernel_size + (kx + half);
                            
                            sum += in_data[in_idx] * kernel[k_idx];
                        }
                    }
                    
                    int out_idx = y * output->stride + x * output->channels + c;
                    out_data[out_idx] = (uint8_t)clamp_double(sum, 0.0, 255.0);
                }
            }
        }
    } else if (input->data_type == DATA_TYPE_FLOAT32) {
        float *in_data = (float*)input->data;
        float *out_data = (float*)output->data;
        
        for (int y = 0; y < input->height; y++) {
            for (int x = 0; x < input->width; x++) {
                for (int c = 0; c < input->channels; c++) {
                    double sum = 0.0;
                    
                    for (int ky = -half; ky <= half; ky++) {
                        for (int kx = -half; kx <= half; kx++) {
                            int px = clamp_int(x + kx, 0, input->width - 1);
                            int py = clamp_int(y + ky, 0, input->height - 1);
                            
                            int in_idx = (py * input->stride / sizeof(float)) + 
                                        px * input->channels + c;
                            int k_idx = (ky + half) * kernel_size + (kx + half);
                            
                            sum += in_data[in_idx] * kernel[k_idx];
                        }
                    }
                    
                    int out_idx = (y * output->stride / sizeof(float)) + 
                                 x * output->channels + c;
                    out_data[out_idx] = (float)sum;
                }
            }
        }
    }
    
    free(kernel);
    stop_timer();
    
    log_message("Gaussian filter applied: kernel_size=%d, sigma=%.2f, time=%.3fs\n",
                kernel_size, sigma, g_config.last_operation_time);
    
    return NOISE_SUCCESS;
}

// ============================================================================
// Median Filter
// ============================================================================

static int compare_uint8(const void *a, const void *b) {
    return (*(uint8_t*)a - *(uint8_t*)b);
}

static int compare_float(const void *a, const void *b) {
    float diff = *(float*)a - *(float*)b;
    return (diff > 0) ? 1 : (diff < 0) ? -1 : 0;
}

NoiseError median_filter(
    Image *output,
    const Image *input,
    int kernel_size)
{
    if (!output || !input) return NOISE_ERROR_INVALID_PARAM;
    if (!images_are_compatible(output, input)) return NOISE_ERROR_DIMENSION_MISMATCH;
    if (kernel_size <= 0 || kernel_size % 2 == 0) return NOISE_ERROR_INVALID_PARAM;
    
    start_timer();
    
    int half = kernel_size / 2;
    int window_size = kernel_size * kernel_size;
    
    if (input->data_type == DATA_TYPE_UINT8) {
        uint8_t *in_data = (uint8_t*)input->data;
        uint8_t *out_data = (uint8_t*)output->data;
        uint8_t *window = (uint8_t*)malloc(window_size * sizeof(uint8_t));
        
        if (!window) return NOISE_ERROR_OUT_OF_MEMORY;
        
        for (int y = 0; y < input->height; y++) {
            for (int x = 0; x < input->width; x++) {
                for (int c = 0; c < input->channels; c++) {
                    int w_idx = 0;
                    
                    // Collect window values
                    for (int ky = -half; ky <= half; ky++) {
                        for (int kx = -half; kx <= half; kx++) {
                            int px = clamp_int(x + kx, 0, input->width - 1);
                            int py = clamp_int(y + ky, 0, input->height - 1);
                            
                            int in_idx = py * input->stride + px * input->channels + c;
                            window[w_idx++] = in_data[in_idx];
                        }
                    }
                    
                    // Sort and get median
                    qsort(window, window_size, sizeof(uint8_t), compare_uint8);
                    
                    int out_idx = y * output->stride + x * output->channels + c;
                    out_data[out_idx] = window[window_size / 2];
                }
            }
        }
        
        free(window);
        
    } else if (input->data_type == DATA_TYPE_FLOAT32) {
        float *in_data = (float*)input->data;
        float *out_data = (float*)output->data;
        float *window = (float*)malloc(window_size * sizeof(float));
        
        if (!window) return NOISE_ERROR_OUT_OF_MEMORY;
        
        for (int y = 0; y < input->height; y++) {
            for (int x = 0; x < input->width; x++) {
                for (int c = 0; c < input->channels; c++) {
                    int w_idx = 0;
                    
                    for (int ky = -half; ky <= half; ky++) {
                        for (int kx = -half; kx <= half; kx++) {
                            int px = clamp_int(x + kx, 0, input->width - 1);
                            int py = clamp_int(y + ky, 0, input->height - 1);
                            
                            int in_idx = (py * input->stride / sizeof(float)) + 
                                        px * input->channels + c;
                            window[w_idx++] = in_data[in_idx];
                        }
                    }
                    
                    qsort(window, window_size, sizeof(float), compare_float);
                    
                    int out_idx = (y * output->stride / sizeof(float)) + 
                                 x * output->channels + c;
                    out_data[out_idx] = window[window_size / 2];
                }
            }
        }
        
        free(window);
    }
    
    stop_timer();
    
    log_message("Median filter applied: kernel_size=%d, time=%.3fs\n",
                kernel_size, g_config.last_operation_time);
    
    return NOISE_SUCCESS;
}

// ============================================================================
// Bilateral Filter
// ============================================================================

NoiseError bilateral_filter(
    Image *output,
    const Image *input,
    int kernel_size,
    double sigma_color,
    double sigma_space)
{
    if (!output || !input) return NOISE_ERROR_INVALID_PARAM;
    if (!images_are_compatible(output, input)) return NOISE_ERROR_DIMENSION_MISMATCH;
    if (kernel_size <= 0 || kernel_size % 2 == 0) return NOISE_ERROR_INVALID_PARAM;
    if (sigma_color <= 0.0 || sigma_space <= 0.0) return NOISE_ERROR_INVALID_PARAM;
    
    start_timer();
    
    int half = kernel_size / 2;
    double color_coeff = -0.5 / (sigma_color * sigma_color);
    double space_coeff = -0.5 / (sigma_space * sigma_space);
    
    if (input->data_type == DATA_TYPE_UINT8) {
        uint8_t *in_data = (uint8_t*)input->data;
        uint8_t *out_data = (uint8_t*)output->data;
        
        for (int y = 0; y < input->height; y++) {
            for (int x = 0; x < input->width; x++) {
                for (int c = 0; c < input->channels; c++) {
                    int center_idx = y * input->stride + x * input->channels + c;
                    double center_val = in_data[center_idx];
                    
                    double sum = 0.0;
                    double weight_sum = 0.0;
                    
                    for (int ky = -half; ky <= half; ky++) {
                        for (int kx = -half; kx <= half; kx++) {
                            int px = clamp_int(x + kx, 0, input->width - 1);
                            int py = clamp_int(y + ky, 0, input->height - 1);
                            
                            int in_idx = py * input->stride + px * input->channels + c;
                            double pixel_val = in_data[in_idx];
                            
                            // Spatial weight
                            double space_dist = kx * kx + ky * ky;
                            double space_weight = exp(space_dist * space_coeff);
                            
                            // Color weight
                            double color_dist = (pixel_val - center_val) * 
                                              (pixel_val - center_val);
                            double color_weight = exp(color_dist * color_coeff);
                            
                            double weight = space_weight * color_weight;
                            sum += pixel_val * weight;
                            weight_sum += weight;
                        }
                    }
                    
                    int out_idx = y * output->stride + x * output->channels + c;
                    out_data[out_idx] = (uint8_t)clamp_double(sum / weight_sum, 
                                                              0.0, 255.0);
                }
            }
        }
    } else if (input->data_type == DATA_TYPE_FLOAT32) {
        float *in_data = (float*)input->data;
        float *out_data = (float*)output->data;
        
        for (int y = 0; y < input->height; y++) {
            for (int x = 0; x < input->width; x++) {
                for (int c = 0; c < input->channels; c++) {
                    int center_idx = (y * input->stride / sizeof(float)) + 
                                    x * input->channels + c;
                    double center_val = in_data[center_idx];
                    
                    double sum = 0.0;
                    double weight_sum = 0.0;
                    
                    for (int ky = -half; ky <= half; ky++) {
                        for (int kx = -half; kx <= half; kx++) {
                            int px = clamp_int(x + kx, 0, input->width - 1);
                            int py = clamp_int(y + ky, 0, input->height - 1);
                            
                            int in_idx = (py * input->stride / sizeof(float)) + 
                                        px * input->channels + c;
                            double pixel_val = in_data[in_idx];
                            
                            double space_dist = kx * kx + ky * ky;
                            double space_weight = exp(space_dist * space_coeff);
                            
                            double color_dist = (pixel_val - center_val) * 
                                              (pixel_val - center_val);
                            double color_weight = exp(color_dist * color_coeff);
                            
                            double weight = space_weight * color_weight;
                            sum += pixel_val * weight;
                            weight_sum += weight;
                        }
                    }
                    
                    int out_idx = (y * output->stride / sizeof(float)) + 
                                 x * output->channels + c;
                    out_data[out_idx] = (float)(sum / weight_sum);
                }
            }
        }
    }
    
    stop_timer();
    
    log_message("Bilateral filter applied: kernel=%d, sigma_c=%.2f, sigma_s=%.2f, time=%.3fs\n",
                kernel_size, sigma_color, sigma_space, g_config.last_operation_time);
    
    return NOISE_SUCCESS;
}

// ============================================================================
// Mean Filter
// ============================================================================

NoiseError mean_filter(
    Image *output,
    const Image *input,
    int kernel_size)
{
    if (!output || !input) return NOISE_ERROR_INVALID_PARAM;
    if (!images_are_compatible(output, input)) return NOISE_ERROR_DIMENSION_MISMATCH;
    if (kernel_size <= 0 || kernel_size % 2 == 0) return NOISE_ERROR_INVALID_PARAM;
    
    start_timer();
    
    int half = kernel_size / 2;
    int window_size = kernel_size * kernel_size;
    
    if (input->data_type == DATA_TYPE_UINT8) {
        uint8_t *in_data = (uint8_t*)input->data;
        uint8_t *out_data = (uint8_t*)output->data;
        
        for (int y = 0; y < input->height; y++) {
            for (int x = 0; x < input->width; x++) {
                for (int c = 0; c < input->channels; c++) {
                    double sum = 0.0;
                    
                    for (int ky = -half; ky <= half; ky++) {
                        for (int kx = -half; kx <= half; kx++) {
                            int px = clamp_int(x + kx, 0, input->width - 1);
                            int py = clamp_int(y + ky, 0, input->height - 1);
                            
                            int in_idx = py * input->stride + px * input->channels + c;
                            sum += in_data[in_idx];
                        }
                    }
                    
                    int out_idx = y * output->stride + x * output->channels + c;
                    out_data[out_idx] = (uint8_t)(sum / window_size);
                }
            }
        }
    } else if (input->data_type == DATA_TYPE_FLOAT32) {
        float *in_data = (float*)input->data;
        float *out_data = (float*)output->data;
        
        for (int y = 0; y < input->height; y++) {
            for (int x = 0; x < input->width; x++) {
                for (int c = 0; c < input->channels; c++) {
                    double sum = 0.0;
                    
                    for (int ky = -half; ky <= half; ky++) {
                        for (int kx = -half; kx <= half; kx++) {
                            int px = clamp_int(x + kx, 0, input->width - 1);
                            int py = clamp_int(y + ky, 0, input->height - 1);
                            
                            int in_idx = (py * input->stride / sizeof(float)) + 
                                        px * input->channels + c;
                            sum += in_data[in_idx];
                        }
                    }
                    
                    int out_idx = (y * output->stride / sizeof(float)) + 
                                 x * output->channels + c;
                    out_data[out_idx] = (float)(sum / window_size);
                }
            }
        }
    }
    
    stop_timer();
    
    log_message("Mean filter applied: kernel_size=%d, time=%.3fs\n",
                kernel_size, g_config.last_operation_time);
    
    return NOISE_SUCCESS;
}

// ============================================================================
// Wiener Filter
// ============================================================================

NoiseError wiener_filter(
    Image *output,
    const Image *input,
    int kernel_size,
    double noise_variance)
{
    if (!output || !input) return NOISE_ERROR_INVALID_PARAM;
    if (!images_are_compatible(output, input)) return NOISE_ERROR_DIMENSION_MISMATCH;
    if (kernel_size <= 0 || kernel_size % 2 == 0) return NOISE_ERROR_INVALID_PARAM;
    
    start_timer();
    
    int half = kernel_size / 2;
    int window_size = kernel_size * kernel_size;
    
    if (input->data_type == DATA_TYPE_UINT8) {
        uint8_t *in_data = (uint8_t*)input->data;
        uint8_t *out_data = (uint8_t*)output->data;
        
        for (int y = 0; y < input->height; y++) {
            for (int x = 0; x < input->width; x++) {
                for (int c = 0; c < input->channels; c++) {
                    // Calculate local mean
                    double local_mean = 0.0;
                    for (int ky = -half; ky <= half; ky++) {
                        for (int kx = -half; kx <= half; kx++) {
                            int px = clamp_int(x + kx, 0, input->width - 1);
                            int py = clamp_int(y + ky, 0, input->height - 1);
                            int in_idx = py * input->stride + px * input->channels + c;
                            local_mean += in_data[in_idx];
                        }
                    }
                    local_mean /= window_size;
                    
                    // Calculate local variance
                    double local_var = 0.0;
                    for (int ky = -half; ky <= half; ky++) {
                        for (int kx = -half; kx <= half; kx++) {
                            int px = clamp_int(x + kx, 0, input->width - 1);
                            int py = clamp_int(y + ky, 0, input->height - 1);
                            int in_idx = py * input->stride + px * input->channels + c;
                            double diff = in_data[in_idx] - local_mean;
                            local_var += diff * diff;
                        }
                    }
                    local_var /= window_size;
                    
                    // Wiener filter formula
                    int center_idx = y * input->stride + x * input->channels + c;
                    double center_val = in_data[center_idx];
                    
                    double wiener_coeff = max_double(0.0, 
                        (local_var - noise_variance) / (local_var + 1e-10));
                    
                    double result = local_mean + wiener_coeff * (center_val - local_mean);
                    
                    int out_idx = y * output->stride + x * output->channels + c;
                    out_data[out_idx] = (uint8_t)clamp_double(result, 0.0, 255.0);
                }
            }
        }
    }
    
    stop_timer();
    
    log_message("Wiener filter applied: kernel=%d, noise_var=%.2f, time=%.3fs\n",
                kernel_size, noise_variance, g_config.last_operation_time);
    
    return NOISE_SUCCESS;
}

// End of Part 3/6
// ============================================================================
// noise_reduction.c - Implementation File Part 4/6
// Advanced Filtering Algorithms
// ============================================================================

// ============================================================================
// Non-Local Means Filter
// ============================================================================

static double compute_patch_distance(
    const uint8_t *data1,
    const uint8_t *data2,
    int stride,
    int channels,
    int patch_size)
{
    double dist = 0.0;
    int half = patch_size / 2;
    
    for (int py = -half; py <= half; py++) {
        for (int px = -half; px <= half; px++) {
            for (int c = 0; c < channels; c++) {
                int idx1 = py * stride + px * channels + c;
                int idx2 = py * stride + px * channels + c;
                double diff = (double)data1[idx1] - (double)data2[idx2];
                dist += diff * diff;
            }
        }
    }
    
    return dist;
}

NoiseError non_local_means_filter(
    Image *output,
    const Image *input,
    int search_window,
    int patch_size,
    double h)
{
    if (!output || !input) return NOISE_ERROR_INVALID_PARAM;
    if (!images_are_compatible(output, input)) return NOISE_ERROR_DIMENSION_MISMATCH;
    if (search_window <= 0 || search_window % 2 == 0) return NOISE_ERROR_INVALID_PARAM;
    if (patch_size <= 0 || patch_size % 2 == 0) return NOISE_ERROR_INVALID_PARAM;
    if (h <= 0.0) return NOISE_ERROR_INVALID_PARAM;
    
    start_timer();
    
    int search_half = search_window / 2;
    int patch_half = patch_size / 2;
    double h_sq = h * h;
    
    if (input->data_type == DATA_TYPE_UINT8) {
        uint8_t *in_data = (uint8_t*)input->data;
        uint8_t *out_data = (uint8_t*)output->data;
        
        for (int y = patch_half; y < input->height - patch_half; y++) {
            for (int x = patch_half; x < input->width - patch_half; x++) {
                for (int c = 0; c < input->channels; c++) {
                    double sum = 0.0;
                    double weight_sum = 0.0;
                    
                    int center_idx = y * input->stride + x * input->channels;
                    
                    // Search in neighborhood
                    for (int sy = -search_half; sy <= search_half; sy++) {
                        for (int sx = -search_half; sx <= search_half; sx++) {
                            int ny = clamp_int(y + sy, patch_half, 
                                             input->height - patch_half - 1);
                            int nx = clamp_int(x + sx, patch_half, 
                                             input->width - patch_half - 1);
                            
                            int neighbor_idx = ny * input->stride + 
                                             nx * input->channels;
                            
                            // Compute patch distance
                            double dist = compute_patch_distance(
                                in_data + center_idx,
                                in_data + neighbor_idx,
                                input->stride,
                                input->channels,
                                patch_size
                            );
                            
                            // Compute weight
                            double weight = exp(-dist / h_sq);
                            
                            sum += in_data[neighbor_idx + c] * weight;
                            weight_sum += weight;
                        }
                    }
                    
                    int out_idx = y * output->stride + x * output->channels + c;
                    out_data[out_idx] = (uint8_t)clamp_double(sum / weight_sum, 
                                                              0.0, 255.0);
                }
            }
        }
        
        // Handle borders by copying
        for (int y = 0; y < patch_half; y++) {
            memcpy(out_data + y * output->stride,
                   in_data + y * input->stride,
                   input->stride);
        }
        for (int y = input->height - patch_half; y < input->height; y++) {
            memcpy(out_data + y * output->stride,
                   in_data + y * input->stride,
                   input->stride);
        }
    }
    
    stop_timer();
    
    log_message("NLM filter applied: search=%d, patch=%d, h=%.2f, time=%.3fs\n",
                search_window, patch_size, h, g_config.last_operation_time);
    
    return NOISE_SUCCESS;
}

// ============================================================================
// Anisotropic Diffusion (Perona-Malik)
// ============================================================================

static double diffusion_coefficient(double gradient, double kappa, int type) {
    double g_sq = gradient * gradient;
    double k_sq = kappa * kappa;
    
    switch (type) {
        case 1: // Exponential
            return exp(-g_sq / k_sq);
        case 2: // Rational
            return 1.0 / (1.0 + g_sq / k_sq);
        default:
            return 1.0;
    }
}

NoiseError anisotropic_diffusion(
    Image *output,
    const Image *input,
    int iterations,
    double kappa,
    double lambda,
    int option)
{
    if (!output || !input) return NOISE_ERROR_INVALID_PARAM;
    if (!images_are_compatible(output, input)) return NOISE_ERROR_DIMENSION_MISMATCH;
    if (iterations <= 0) return NOISE_ERROR_INVALID_PARAM;
    if (kappa <= 0.0 || lambda <= 0.0 || lambda > 0.25) {
        return NOISE_ERROR_INVALID_PARAM;
    }
    
    start_timer();
    
    // Create temporary image for iterations
    Image *temp = NULL;
    NoiseError err = image_clone(&temp, input);
    if (err != NOISE_SUCCESS) return err;
    
    if (input->data_type == DATA_TYPE_UINT8) {
        uint8_t *data = (uint8_t*)temp->data;
        
        for (int iter = 0; iter < iterations; iter++) {
            for (int y = 1; y < input->height - 1; y++) {
                for (int x = 1; x < input->width - 1; x++) {
                    for (int c = 0; c < input->channels; c++) {
                        int idx = y * temp->stride + x * temp->channels + c;
                        double center = data[idx];
                        
                        // Compute gradients in 4 directions
                        double north = data[idx - temp->stride] - center;
                        double south = data[idx + temp->stride] - center;
                        double east = data[idx + temp->channels] - center;
                        double west = data[idx - temp->channels] - center;
                        
                        // Compute diffusion coefficients
                        double cN = diffusion_coefficient(fabs(north), kappa, option);
                        double cS = diffusion_coefficient(fabs(south), kappa, option);
                        double cE = diffusion_coefficient(fabs(east), kappa, option);
                        double cW = diffusion_coefficient(fabs(west), kappa, option);
                        
                        // Update pixel value
                        double update = lambda * (cN * north + cS * south + 
                                                 cE * east + cW * west);
                        
                        data[idx] = (uint8_t)clamp_double(center + update, 0.0, 255.0);
                    }
                }
            }
        }
    } else if (input->data_type == DATA_TYPE_FLOAT32) {
        float *data = (float*)temp->data;
        int stride_float = temp->stride / sizeof(float);
        
        for (int iter = 0; iter < iterations; iter++) {
            for (int y = 1; y < input->height - 1; y++) {
                for (int x = 1; x < input->width - 1; x++) {
                    for (int c = 0; c < input->channels; c++) {
                        int idx = y * stride_float + x * temp->channels + c;
                        double center = data[idx];
                        
                        double north = data[idx - stride_float] - center;
                        double south = data[idx + stride_float] - center;
                        double east = data[idx + temp->channels] - center;
                        double west = data[idx - temp->channels] - center;
                        
                        double cN = diffusion_coefficient(fabs(north), kappa, option);
                        double cS = diffusion_coefficient(fabs(south), kappa, option);
                        double cE = diffusion_coefficient(fabs(east), kappa, option);
                        double cW = diffusion_coefficient(fabs(west), kappa, option);
                        
                        double update = lambda * (cN * north + cS * south + 
                                                 cE * east + cW * west);
                        
                        data[idx] = (float)(center + update);
                    }
                }
            }
        }
    }
    
    // Copy result to output
    image_copy(output, temp);
    image_destroy(temp);
    
    stop_timer();
    
    log_message("Anisotropic diffusion: iter=%d, kappa=%.2f, lambda=%.3f, time=%.3fs\n",
                iterations, kappa, lambda, g_config.last_operation_time);
    
    return NOISE_SUCCESS;
}

// ============================================================================
// Total Variation Denoising
// ============================================================================

NoiseError total_variation_denoise(
    Image *output,
    const Image *input,
    double lambda,
    int iterations)
{
    if (!output || !input) return NOISE_ERROR_INVALID_PARAM;
    if (!images_are_compatible(output, input)) return NOISE_ERROR_DIMENSION_MISMATCH;
    if (lambda <= 0.0 || iterations <= 0) return NOISE_ERROR_INVALID_PARAM;
    
    start_timer();
    
    Image *temp = NULL;
    NoiseError err = image_clone(&temp, input);
    if (err != NOISE_SUCCESS) return err;
    
    const double dt = 0.25; // Time step
    const double epsilon = 1e-8; // Small constant for numerical stability
    
    if (input->data_type == DATA_TYPE_UINT8) {
        uint8_t *orig_data = (uint8_t*)input->data;
        uint8_t *data = (uint8_t*)temp->data;
        
        for (int iter = 0; iter < iterations; iter++) {
            for (int y = 1; y < input->height - 1; y++) {
                for (int x = 1; x < input->width - 1; x++) {
                    for (int c = 0; c < input->channels; c++) {
                        int idx = y * temp->stride + x * temp->channels + c;
                        double u = data[idx];
                        double f = orig_data[idx]; // Original noisy image
                        
                        // Compute gradients
                        double ux = (data[idx + temp->channels] - 
                                   data[idx - temp->channels]) / 2.0;
                        double uy = (data[idx + temp->stride] - 
                                   data[idx - temp->stride]) / 2.0;
                        
                        // Compute second derivatives
                        double uxx = data[idx + temp->channels] - 2.0 * u + 
                                    data[idx - temp->channels];
                        double uyy = data[idx + temp->stride] - 2.0 * u + 
                                    data[idx - temp->stride];
                        
                        double uxy = (data[idx + temp->stride + temp->channels] -
                                     data[idx + temp->stride - temp->channels] -
                                     data[idx - temp->stride + temp->channels] +
                                     data[idx - temp->stride - temp->channels]) / 4.0;
                        
                        // TV diffusion term
                        double grad_mag = sqrt(ux * ux + uy * uy + epsilon);
                        double tv_term = (uxx * (uy * uy + epsilon) - 
                                        2.0 * ux * uy * uxy + 
                                        uyy * (ux * ux + epsilon)) / 
                                       (grad_mag * grad_mag * grad_mag);
                        
                        // Data fidelity term
                        double fidelity = lambda * (f - u);
                        
                        // Update
                        double update = dt * (tv_term + fidelity);
                        data[idx] = (uint8_t)clamp_double(u + update, 0.0, 255.0);
                    }
                }
            }
        }
    }
    
    image_copy(output, temp);
    image_destroy(temp);
    
    stop_timer();
    
    log_message("TV denoising: lambda=%.3f, iter=%d, time=%.3fs\n",
                lambda, iterations, g_config.last_operation_time);
    
    return NOISE_SUCCESS;
}

// ============================================================================
// Wavelet Denoising
// ============================================================================

static void wavelet_transform_1d(double *data, int n, int forward) {
    if (n < 2) return;
    
    double *temp = (double*)malloc(n * sizeof(double));
    if (!temp) return;
    
    int half = n / 2;
    
    if (forward) {
        // Haar wavelet decomposition
        for (int i = 0; i < half; i++) {
            temp[i] = (data[2*i] + data[2*i+1]) / sqrt(2.0);
            temp[half + i] = (data[2*i] - data[2*i+1]) / sqrt(2.0);
        }
    } else {
        // Haar wavelet reconstruction
        for (int i = 0; i < half; i++) {
            temp[2*i] = (data[i] + data[half + i]) / sqrt(2.0);
            temp[2*i+1] = (data[i] - data[half + i]) / sqrt(2.0);
        }
    }
    
    memcpy(data, temp, n * sizeof(double));
    free(temp);
}

static void soft_threshold(double *data, int n, double threshold) {
    for (int i = 0; i < n; i++) {
        if (data[i] > threshold) {
            data[i] -= threshold;
        } else if (data[i] < -threshold) {
            data[i] += threshold;
        } else {
            data[i] = 0.0;
        }
    }
}

NoiseError wavelet_denoise(
    Image *output,
    const Image *input,
    double threshold,
    int levels)
{
    if (!output || !input) return NOISE_ERROR_INVALID_PARAM;
    if (!images_are_compatible(output, input)) return NOISE_ERROR_DIMENSION_MISMATCH;
    if (threshold < 0.0 || levels <= 0) return NOISE_ERROR_INVALID_PARAM;
    
    start_timer();
    
    // Simplified wavelet denoising using row-column decomposition
    Image *temp = NULL;
    NoiseError err = image_clone(&temp, input);
    if (err != NOISE_SUCCESS) return err;
    
    if (input->data_type == DATA_TYPE_UINT8) {
        uint8_t *in_data = (uint8_t*)input->data;
        
        // Convert to double for processing
        int size = input->width * input->height * input->channels;
        double *work = (double*)malloc(size * sizeof(double));
        if (!work) {
            image_destroy(temp);
            return NOISE_ERROR_OUT_OF_MEMORY;
        }
        
        // Copy and normalize to [0, 1]
        for (int i = 0; i < size; i++) {
            work[i] = in_data[i] / 255.0;
        }
        
        // Process each channel separately
        for (int c = 0; c < input->channels; c++) {
            // Row-wise transform
            for (int y = 0; y < input->height; y++) {
                double *row = work + y * input->width * input->channels + c;
                for (int level = 0; level < levels; level++) {
                    int n = input->width >> level;
                    if (n < 2) break;
                    wavelet_transform_1d(row, n, 1);
                }
            }
            
            // Column-wise transform
            double *col = (double*)malloc(input->height * sizeof(double));
            if (col) {
                for (int x = 0; x < input->width; x++) {
                    // Extract column
                    for (int y = 0; y < input->height; y++) {
                        col[y] = work[(y * input->width + x) * input->channels + c];
                    }
                    
                    // Transform
                    for (int level = 0; level < levels; level++) {
                        int n = input->height >> level;
                        if (n < 2) break;
                        wavelet_transform_1d(col, n, 1);
                    }
                    
                    // Apply threshold
                    soft_threshold(col, input->height, threshold);
                    
                    // Inverse transform
                    for (int level = levels - 1; level >= 0; level--) {
                        int n = input->height >> level;
                        if (n < 2) continue;
                        wavelet_transform_1d(col, n, 0);
                    }
                    
                    // Put back
                    for (int y = 0; y < input->height; y++) {
                        work[(y * input->width + x) * input->channels + c] = col[y];
                    }
                }
                free(col);
            }
            
            // Inverse row-wise transform
            for (int y = 0; y < input->height; y++) {
                double *row = work + y * input->width * input->channels + c;
                for (int level = levels - 1; level >= 0; level--) {
                    int n = input->width >> level;
                    if (n < 2) continue;
                    wavelet_transform_1d(row, n, 0);
                }
            }
        }
        
        // Convert back to uint8
        uint8_t *out_data = (uint8_t*)output->data;
        for (int i = 0; i < size; i++) {
            out_data[i] = (uint8_t)clamp_double(work[i] * 255.0, 0.0, 255.0);
        }
        
        free(work);
    }
    
    image_destroy(temp);
    stop_timer();
    
    log_message("Wavelet denoising: threshold=%.3f, levels=%d, time=%.3fs\n",
                threshold, levels, g_config.last_operation_time);
    
    return NOISE_SUCCESS;
}

// ============================================================================
// Adaptive Filter
// ============================================================================

NoiseError adaptive_filter(
    Image *output,
    const Image *input,
    int kernel_size,
    double noise_variance)
{
    if (!output || !input) return NOISE_ERROR_INVALID_PARAM;
    if (!images_are_compatible(output, input)) return NOISE_ERROR_DIMENSION_MISMATCH;
    if (kernel_size <= 0 || kernel_size % 2 == 0) return NOISE_ERROR_INVALID_PARAM;
    
    start_timer();
    
    int half = kernel_size / 2;
    int window_size = kernel_size * kernel_size;
    
    if (input->data_type == DATA_TYPE_UINT8) {
        uint8_t *in_data = (uint8_t*)input->data;
        uint8_t *out_data = (uint8_t*)output->data;
        
        for (int y = 0; y < input->height; y++) {
            for (int x = 0; x < input->width; x++) {
                for (int c = 0; c < input->channels; c++) {
                    // Calculate local statistics
                    double local_mean = 0.0;
                    double local_var = 0.0;
                    
                    for (int ky = -half; ky <= half; ky++) {
                        for (int kx = -half; kx <= half; kx++) {
                            int px = clamp_int(x + kx, 0, input->width - 1);
                            int py = clamp_int(y + ky, 0, input->height - 1);
                            int idx = py * input->stride + px * input->channels + c;
                            local_mean += in_data[idx];
                        }
                    }
                    local_mean /= window_size;
                    
                    for (int ky = -half; ky <= half; ky++) {
                        for (int kx = -half; kx <= half; kx++) {
                            int px = clamp_int(x + kx, 0, input->width - 1);
                            int py = clamp_int(y + ky, 0, input->height - 1);
                            int idx = py * input->stride + px * input->channels + c;
                            double diff = in_data[idx] - local_mean;
                            local_var += diff * diff;
                        }
                    }
                    local_var /= window_size;
                    
                    // Adaptive filtering
                    int center_idx = y * input->stride + x * input->channels + c;
                    double center_val = in_data[center_idx];
                    
                    double k = max_double(0.0, 
                        (local_var - noise_variance) / (local_var + 1e-10));
                    
                    double result = local_mean + k * (center_val - local_mean);
                    
                    int out_idx = y * output->stride + x * output->channels + c;
                    out_data[out_idx] = (uint8_t)clamp_double(result, 0.0, 255.0);
                }
            }
        }
    }
    
    stop_timer();
    
    log_message("Adaptive filter: kernel=%d, noise_var=%.2f, time=%.3fs\n",
                kernel_size, noise_variance, g_config.last_operation_time);
    
    return NOISE_SUCCESS;
}

// End of Part 4/6
// ============================================================================
// noise_reduction.c - Implementation File Part 5/6
// Automatic Denoising and Batch Processing
// ============================================================================

// ============================================================================
// Automatic Denoising
// ============================================================================

NoiseError auto_denoise(
    Image *output,
    const Image *input,
    DenoiseQuality quality)
{
    if (!output || !input) return NOISE_ERROR_INVALID_PARAM;
    if (!images_are_compatible(output, input)) return NOISE_ERROR_DIMENSION_MISMATCH;
    
    start_timer();
    
    // Estimate noise parameters
    NoiseParams params;
    NoiseError err = estimate_noise_parameters(&params, input);
    if (err != NOISE_SUCCESS) return err;
    
    log_message("Auto-denoise: detected %s noise, level=%.2f, SNR=%.2f dB\n",
                noise_type_name(params.type), params.level, params.snr);
    
    // Select appropriate filter based on noise type and quality
    switch (params.type) {
        case NOISE_TYPE_GAUSSIAN:
            if (quality == DENOISE_QUALITY_FAST) {
                err = gaussian_filter(output, input, 5, params.level / 10.0);
            } else if (quality == DENOISE_QUALITY_BALANCED) {
                err = bilateral_filter(output, input, 7, 
                                      params.level * 2.0, params.level / 5.0);
            } else { // DENOISE_QUALITY_BEST
                err = non_local_means_filter(output, input, 21, 7, 
                                            params.level * 1.5);
            }
            break;
            
        case NOISE_TYPE_SALT_PEPPER:
            if (quality == DENOISE_QUALITY_FAST) {
                err = median_filter(output, input, 3);
            } else if (quality == DENOISE_QUALITY_BALANCED) {
                err = median_filter(output, input, 5);
            } else {
                // Apply median filter followed by bilateral
                Image *temp = NULL;
                err = image_create(&temp, input->width, input->height,
                                  input->channels, input->data_type);
                if (err == NOISE_SUCCESS) {
                    err = median_filter(temp, input, 5);
                    if (err == NOISE_SUCCESS) {
                        err = bilateral_filter(output, temp, 7, 25.0, 5.0);
                    }
                    image_destroy(temp);
                }
            }
            break;
            
        case NOISE_TYPE_POISSON:
            if (quality == DENOISE_QUALITY_FAST) {
                err = anisotropic_diffusion(output, input, 5, 15.0, 0.2, 1);
            } else if (quality == DENOISE_QUALITY_BALANCED) {
                err = anisotropic_diffusion(output, input, 10, 15.0, 0.2, 1);
            } else {
                err = total_variation_denoise(output, input, 0.1, 20);
            }
            break;
            
        case NOISE_TYPE_SPECKLE:
            if (quality == DENOISE_QUALITY_FAST) {
                err = wiener_filter(output, input, 5, params.variance);
            } else if (quality == DENOISE_QUALITY_BALANCED) {
                err = adaptive_filter(output, input, 7, params.variance);
            } else {
                err = wavelet_denoise(output, input, params.level / 100.0, 3);
            }
            break;
            
        default:
            // Unknown noise type - use conservative approach
            if (quality == DENOISE_QUALITY_FAST) {
                err = gaussian_filter(output, input, 3, 1.0);
            } else {
                err = bilateral_filter(output, input, 7, 25.0, 5.0);
            }
            break;
    }
    
    stop_timer();
    
    if (err == NOISE_SUCCESS) {
        log_message("Auto-denoise completed in %.3fs\n", 
                   g_config.last_operation_time);
    }
    
    return err;
}

// ============================================================================
// Batch Processing
// ============================================================================

NoiseError batch_denoise(
    Image **outputs,
    const Image **inputs,
    int count,
    const DenoiseParams *params)
{
    if (!outputs || !inputs || count <= 0 || !params) {
        return NOISE_ERROR_INVALID_PARAM;
    }
    
    start_timer();
    
    int success_count = 0;
    int fail_count = 0;
    
    log_message("Starting batch processing of %d images...\n", count);
    
    for (int i = 0; i < count; i++) {
        if (!inputs[i]) {
            log_message("Image %d: NULL input, skipping\n", i);
            fail_count++;
            continue;
        }
        
        // Create output image
        NoiseError err = image_create(&outputs[i], 
                                     inputs[i]->width,
                                     inputs[i]->height,
                                     inputs[i]->channels,
                                     inputs[i]->data_type);
        
        if (err != NOISE_SUCCESS) {
            log_message("Image %d: Failed to create output (error %d)\n", i, err);
            fail_count++;
            continue;
        }
        
        // Apply denoising based on method
        switch (params->method) {
            case DENOISE_METHOD_GAUSSIAN:
                err = gaussian_filter(outputs[i], inputs[i],
                                     params->kernel_size, params->sigma);
                break;
                
            case DENOISE_METHOD_MEDIAN:
                err = median_filter(outputs[i], inputs[i], params->kernel_size);
                break;
                
            case DENOISE_METHOD_BILATERAL:
                err = bilateral_filter(outputs[i], inputs[i],
                                      params->kernel_size,
                                      params->sigma_color,
                                      params->sigma_space);
                break;
                
            case DENOISE_METHOD_NLM:
                err = non_local_means_filter(outputs[i], inputs[i],
                                            params->search_window,
                                            params->patch_size,
                                            params->h);
                break;
                
            case DENOISE_METHOD_ANISOTROPIC:
                err = anisotropic_diffusion(outputs[i], inputs[i],
                                           params->iterations,
                                           params->kappa,
                                           params->lambda,
                                           params->option);
                break;
                
            case DENOISE_METHOD_TV:
                err = total_variation_denoise(outputs[i], inputs[i],
                                             params->lambda,
                                             params->iterations);
                break;
                
            case DENOISE_METHOD_WAVELET:
                err = wavelet_denoise(outputs[i], inputs[i],
                                     params->threshold,
                                     params->levels);
                break;
                
            case DENOISE_METHOD_WIENER:
                err = wiener_filter(outputs[i], inputs[i],
                                   params->kernel_size,
                                   params->noise_variance);
                break;
                
            case DENOISE_METHOD_ADAPTIVE:
                err = adaptive_filter(outputs[i], inputs[i],
                                     params->kernel_size,
                                     params->noise_variance);
                break;
                
            case DENOISE_METHOD_AUTO:
                err = auto_denoise(outputs[i], inputs[i], params->quality);
                break;
                
            default:
                err = NOISE_ERROR_INVALID_PARAM;
                break;
        }
        
        if (err == NOISE_SUCCESS) {
            success_count++;
            log_message("Image %d: Successfully processed\n", i);
        } else {
            fail_count++;
            log_message("Image %d: Processing failed (error %d)\n", i, err);
            image_destroy(outputs[i]);
            outputs[i] = NULL;
        }
    }
    
    stop_timer();
    
    log_message("Batch processing completed: %d succeeded, %d failed, time=%.3fs\n",
                success_count, fail_count, g_config.last_operation_time);
    
    return (fail_count == 0) ? NOISE_SUCCESS : NOISE_ERROR_PROCESSING_FAILED;
}

// ============================================================================
// Progressive Denoising
// ============================================================================

NoiseError progressive_denoise(
    Image *output,
    const Image *input,
    int stages,
    ProgressCallback callback,
    void *user_data)
{
    if (!output || !input || stages <= 0) return NOISE_ERROR_INVALID_PARAM;
    if (!images_are_compatible(output, input)) return NOISE_ERROR_DIMENSION_MISMATCH;
    
    start_timer();
    
    Image *temp1 = NULL;
    Image *temp2 = NULL;
    NoiseError err;
    
    // Create temporary images
    err = image_create(&temp1, input->width, input->height,
                      input->channels, input->data_type);
    if (err != NOISE_SUCCESS) return err;
    
    err = image_create(&temp2, input->width, input->height,
                      input->channels, input->data_type);
    if (err != NOISE_SUCCESS) {
        image_destroy(temp1);
        return err;
    }
    
    // Copy input to temp1
    image_copy(temp1, input);
    
    log_message("Starting progressive denoising with %d stages\n", stages);
    
    for (int stage = 0; stage < stages; stage++) {
        // Report progress
        if (callback) {
            float progress = (float)(stage + 1) / stages;
            if (!callback(progress, user_data)) {
                log_message("Progressive denoising cancelled at stage %d\n", stage);
                err = NOISE_ERROR_CANCELLED;
                break;
            }
        }
        
        // Estimate current noise level
        double noise_level;
        estimate_noise_level(&noise_level, temp1);
        
        log_message("Stage %d/%d: noise_level=%.2f\n", 
                   stage + 1, stages, noise_level);
        
        // Apply appropriate filter based on remaining noise
        if (noise_level > 20.0) {
            // High noise - use strong filter
            err = bilateral_filter(temp2, temp1, 7, 50.0, 10.0);
        } else if (noise_level > 10.0) {
            // Medium noise - use moderate filter
            err = bilateral_filter(temp2, temp1, 5, 25.0, 5.0);
        } else {
            // Low noise - use gentle filter
            err = gaussian_filter(temp2, temp1, 3, 1.0);
        }
        
        if (err != NOISE_SUCCESS) break;
        
        // Swap buffers
        Image *swap = temp1;
        temp1 = temp2;
        temp2 = swap;
    }
    
    // Copy final result to output
    if (err == NOISE_SUCCESS) {
        image_copy(output, temp1);
    }
    
    image_destroy(temp1);
    image_destroy(temp2);
    
    stop_timer();
    
    if (err == NOISE_SUCCESS) {
        log_message("Progressive denoising completed in %.3fs\n",
                   g_config.last_operation_time);
    }
    
    return err;
}

// ============================================================================
// Multi-scale Denoising
// ============================================================================

NoiseError multiscale_denoise(
    Image *output,
    const Image *input,
    int scales,
    const DenoiseParams *params)
{
    if (!output || !input || scales <= 0 || !params) {
        return NOISE_ERROR_INVALID_PARAM;
    }
    if (!images_are_compatible(output, input)) {
        return NOISE_ERROR_DIMENSION_MISMATCH;
    }
    
    start_timer();
    
    log_message("Starting multi-scale denoising with %d scales\n", scales);
    
    // Create pyramid of images
    Image **pyramid = (Image**)malloc(scales * sizeof(Image*));
    if (!pyramid) return NOISE_ERROR_OUT_OF_MEMORY;
    
    NoiseError err = NOISE_SUCCESS;
    
    // Build pyramid (downsampling)
    pyramid[0] = NULL;
    err = image_clone(&pyramid[0], input);
    if (err != NOISE_SUCCESS) {
        free(pyramid);
        return err;
    }
    
    for (int i = 1; i < scales; i++) {
        int new_width = pyramid[i-1]->width / 2;
        int new_height = pyramid[i-1]->height / 2;
        
        if (new_width < 4 || new_height < 4) {
            scales = i; // Adjust scales if image becomes too small
            break;
        }
        
        err = image_create(&pyramid[i], new_width, new_height,
                          input->channels, input->data_type);
        if (err != NOISE_SUCCESS) break;
        
        // Downsample with Gaussian blur
        Image *blurred = NULL;
        err = image_create(&blurred, pyramid[i-1]->width, pyramid[i-1]->height,
                          input->channels, input->data_type);
        if (err != NOISE_SUCCESS) {
            image_destroy(pyramid[i]);
            break;
        }
        
        gaussian_filter(blurred, pyramid[i-1], 5, 1.0);
        
        // Downsample by 2
        if (input->data_type == DATA_TYPE_UINT8) {
            uint8_t *src = (uint8_t*)blurred->data;
            uint8_t *dst = (uint8_t*)pyramid[i]->data;
            
            for (int y = 0; y < new_height; y++) {
                for (int x = 0; x < new_width; x++) {
                    for (int c = 0; c < input->channels; c++) {
                        int src_idx = (y * 2) * blurred->stride + 
                                     (x * 2) * input->channels + c;
                        int dst_idx = y * pyramid[i]->stride + 
                                     x * input->channels + c;
                        dst[dst_idx] = src[src_idx];
                    }
                }
            }
        }
        
        image_destroy(blurred);
    }
    
    if (err != NOISE_SUCCESS) {
        for (int i = 0; i < scales; i++) {
            image_destroy(pyramid[i]);
        }
        free(pyramid);
        return err;
    }
    
    // Denoise each scale
    for (int i = 0; i < scales; i++) {
        log_message("Processing scale %d/%d (%dx%d)\n", 
                   i + 1, scales, pyramid[i]->width, pyramid[i]->height);
        
        Image *denoised = NULL;
        err = image_create(&denoised, pyramid[i]->width, pyramid[i]->height,
                          input->channels, input->data_type);
        if (err != NOISE_SUCCESS) break;
        
        // Apply denoising with scale-adjusted parameters
        DenoiseParams scale_params = *params;
        scale_params.kernel_size = max_int(3, params->kernel_size / (1 << i));
        
        switch (params->method) {
            case DENOISE_METHOD_BILATERAL:
                err = bilateral_filter(denoised, pyramid[i],
                                      scale_params.kernel_size,
                                      params->sigma_color,
                                      params->sigma_space);
                break;
            case DENOISE_METHOD_NLM:
                err = non_local_means_filter(denoised, pyramid[i],
                                            scale_params.search_window,
                                            scale_params.patch_size,
                                            params->h);
                break;
            default:
                err = gaussian_filter(denoised, pyramid[i],
                                     scale_params.kernel_size,
                                     params->sigma);
                break;
        }
        
        if (err == NOISE_SUCCESS) {
            image_copy(pyramid[i], denoised);
        }
        
        image_destroy(denoised);
        if (err != NOISE_SUCCESS) break;
    }
    
    // Reconstruct from pyramid (upsampling and combining)
    if (err == NOISE_SUCCESS) {
        for (int i = scales - 2; i >= 0; i--) {
            Image *upsampled = NULL;
            err = image_create(&upsampled, pyramid[i]->width, pyramid[i]->height,
                              input->channels, input->data_type);
            if (err != NOISE_SUCCESS) break;
            
            // Upsample from pyramid[i+1] to upsampled
            if (input->data_type == DATA_TYPE_UINT8) {
                uint8_t *src = (uint8_t*)pyramid[i+1]->data;
                uint8_t *dst = (uint8_t*)upsampled->data;
                
                for (int y = 0; y < upsampled->height; y++) {
                    for (int x = 0; x < upsampled->width; x++) {
                        for (int c = 0; c < input->channels; c++) {
                            int src_idx = (y / 2) * pyramid[i+1]->stride + 
                                         (x / 2) * input->channels + c;
                            int dst_idx = y * upsampled->stride + 
                                         x * input->channels + c;
                            dst[dst_idx] = src[src_idx];
                        }
                    }
                }
            }
            
            // Blend with current scale
            if (input->data_type == DATA_TYPE_UINT8) {
                uint8_t *curr = (uint8_t*)pyramid[i]->data;
                uint8_t *up = (uint8_t*)upsampled->data;
                
                for (int y = 0; y < pyramid[i]->height; y++) {
                    for (int x = 0; x < pyramid[i]->width; x++) {
                        for (int c = 0; c < input->channels; c++) {
                            int idx = y * pyramid[i]->stride + 
                                     x * input->channels + c;
                            // Weighted average
                            curr[idx] = (uint8_t)((curr[idx] + up[idx]) / 2);
                        }
                    }
                }
            }
            
            image_destroy(upsampled);
        }
        
        // Copy final result
        image_copy(output, pyramid[0]);
    }
    
    // Cleanup
    for (int i = 0; i < scales; i++) {
        image_destroy(pyramid[i]);
    }
    free(pyramid);
    
    stop_timer();
    
    if (err == NOISE_SUCCESS) {
        log_message("Multi-scale denoising completed in %.3fs\n",
                   g_config.last_operation_time);
    }
    
    return err;
}

// ============================================================================
// Adaptive Multi-method Denoising
// ============================================================================

NoiseError adaptive_multmethod_denoise(
    Image *output,
    const Image *input,
    const Image *reference)
{
    if (!output || !input) return NOISE_ERROR_INVALID_PARAM;
    if (!images_are_compatible(output, input)) {
        return NOISE_ERROR_DIMENSION_MISMATCH;
    }
    
    start_timer();
    
    log_message("Starting adaptive multi-method denoising\n");
    
    // Try multiple methods and select best result
    const int num_methods = 3;
    Image *results[3] = {NULL, NULL, NULL};
    double scores[3] = {0.0, 0.0, 0.0};
    
    NoiseError err;
    
    // Method 1: Bilateral filter
    err = image_create(&results[0], input->width, input->height,
                      input->channels, input->data_type);
    if (err == NOISE_SUCCESS) {
        bilateral_filter(results[0], input, 7, 25.0, 5.0);
        if (reference) {
            calculate_psnr(&scores[0], reference, results[0]);
        } else {
            // Estimate quality without reference
            double noise_level;
            estimate_noise_level(&noise_level, results[0]);
            scores[0] = 100.0 - noise_level;
        }
        log_message("Bilateral filter score: %.2f\n", scores[0]);
    }
    
    // Method 2: Non-local means
    err = image_create(&results[1], input->width, input->height,
                      input->channels, input->data_type);
    if (err == NOISE_SUCCESS) {
        non_local_means_filter(results[1], input, 21, 7, 10.0);
        if (reference) {
            calculate_psnr(&scores[1], reference, results[1]);
        } else {
            double noise_level;
            estimate_noise_level(&noise_level, results[1]);
            scores[1] = 100.0 - noise_level;
        }
        log_message("NLM filter score: %.2f\n", scores[1]);
    }
    
    // Method 3: Wavelet denoising
    err = image_create(&results[2], input->width, input->height,
                      input->channels, input->data_type);
    if (err == NOISE_SUCCESS) {
        wavelet_denoise(results[2], input, 0.1, 3);
        if (reference) {
            calculate_psnr(&scores[2], reference, results[2]);
        } else {
            double noise_level;
            estimate_noise_level(&noise_level, results[2]);
            scores[2] = 100.0 - noise_level;
        }
        log_message("Wavelet filter score: %.2f\n", scores[2]);
    }
    
    // Select best result
    int best_idx = 0;
    for (int i = 1; i < num_methods; i++) {
        if (scores[i] > scores[best_idx]) {
            best_idx = i;
        }
    }
    
    log_message("Selected method %d with score %.2f\n", best_idx + 1, scores[best_idx]);
    
    // Copy best result to output
    if (results[best_idx]) {
        image_copy(output, results[best_idx]);
    } else {
        err = NOISE_ERROR_PROCESSING_FAILED;
    }
    
    // Cleanup
    for (int i = 0; i < num_methods; i++) {
        image_destroy(results[i]);
    }
    
    stop_timer();
    
    if (err == NOISE_SUCCESS) {
        log_message("Adaptive multi-method denoising completed in %.3fs\n",
                   g_config.last_operation_time);
    }
    
    return err;
}

// End of Part 5/6
// ============================================================================
// noise_reduction.c - Implementation File Part 6/6
// Quality Assessment and Utility Functions
// ============================================================================

// ============================================================================
// Quality Metrics
// ============================================================================

NoiseError calculate_psnr(
    double *psnr,
    const Image *reference,
    const Image *test)
{
    if (!psnr || !reference || !test) return NOISE_ERROR_INVALID_PARAM;
    if (!images_are_compatible(reference, test)) {
        return NOISE_ERROR_DIMENSION_MISMATCH;
    }
    
    double mse = 0.0;
    size_t pixel_count = reference->width * reference->height * reference->channels;
    
    if (reference->data_type == DATA_TYPE_UINT8) {
        uint8_t *ref_data = (uint8_t*)reference->data;
        uint8_t *test_data = (uint8_t*)test->data;
        
        for (int y = 0; y < reference->height; y++) {
            for (int x = 0; x < reference->width; x++) {
                for (int c = 0; c < reference->channels; c++) {
                    int idx = y * reference->stride + x * reference->channels + c;
                    double diff = (double)ref_data[idx] - (double)test_data[idx];
                    mse += diff * diff;
                }
            }
        }
        
        mse /= pixel_count;
        
        if (mse < 1e-10) {
            *psnr = 100.0; // Perfect match
        } else {
            *psnr = 10.0 * log10(255.0 * 255.0 / mse);
        }
        
    } else if (reference->data_type == DATA_TYPE_FLOAT32) {
        float *ref_data = (float*)reference->data;
        float *test_data = (float*)test->data;
        
        for (int y = 0; y < reference->height; y++) {
            for (int x = 0; x < reference->width; x++) {
                for (int c = 0; c < reference->channels; c++) {
                    int idx = (y * reference->stride / sizeof(float)) + 
                             x * reference->channels + c;
                    double diff = (double)ref_data[idx] - (double)test_data[idx];
                    mse += diff * diff;
                }
            }
        }
        
        mse /= pixel_count;
        
        if (mse < 1e-10) {
            *psnr = 100.0;
        } else {
            *psnr = 10.0 * log10(1.0 / mse);
        }
    }
    
    log_message("PSNR calculated: %.2f dB\n", *psnr);
    
    return NOISE_SUCCESS;
}

NoiseError calculate_ssim(
    double *ssim,
    const Image *reference,
    const Image *test)
{
    if (!ssim || !reference || !test) return NOISE_ERROR_INVALID_PARAM;
    if (!images_are_compatible(reference, test)) {
        return NOISE_ERROR_DIMENSION_MISMATCH;
    }
    
    const double C1 = 6.5025;  // (0.01 * 255)^2
    const double C2 = 58.5225; // (0.03 * 255)^2
    const int window_size = 11;
    const int half = window_size / 2;
    
    double ssim_sum = 0.0;
    int count = 0;
    
    if (reference->data_type == DATA_TYPE_UINT8) {
        uint8_t *ref_data = (uint8_t*)reference->data;
        uint8_t *test_data = (uint8_t*)test->data;
        
        for (int y = half; y < reference->height - half; y++) {
            for (int x = half; x < reference->width - half; x++) {
                for (int c = 0; c < reference->channels; c++) {
                    double mu1 = 0.0, mu2 = 0.0;
                    double sigma1_sq = 0.0, sigma2_sq = 0.0, sigma12 = 0.0;
                    int window_count = 0;
                    
                    // Calculate means
                    for (int wy = -half; wy <= half; wy++) {
                        for (int wx = -half; wx <= half; wx++) {
                            int idx = (y + wy) * reference->stride + 
                                     (x + wx) * reference->channels + c;
                            mu1 += ref_data[idx];
                            mu2 += test_data[idx];
                            window_count++;
                        }
                    }
                    mu1 /= window_count;
                    mu2 /= window_count;
                    
                    // Calculate variances and covariance
                    for (int wy = -half; wy <= half; wy++) {
                        for (int wx = -half; wx <= half; wx++) {
                            int idx = (y + wy) * reference->stride + 
                                     (x + wx) * reference->channels + c;
                            double diff1 = ref_data[idx] - mu1;
                            double diff2 = test_data[idx] - mu2;
                            sigma1_sq += diff1 * diff1;
                            sigma2_sq += diff2 * diff2;
                            sigma12 += diff1 * diff2;
                        }
                    }
                    sigma1_sq /= window_count;
                    sigma2_sq /= window_count;
                    sigma12 /= window_count;
                    
                    // Calculate SSIM for this window
                    double numerator = (2.0 * mu1 * mu2 + C1) * 
                                      (2.0 * sigma12 + C2);
                    double denominator = (mu1 * mu1 + mu2 * mu2 + C1) * 
                                        (sigma1_sq + sigma2_sq + C2);
                    
                    ssim_sum += numerator / denominator;
                    count++;
                }
            }
        }
    }
    
    *ssim = ssim_sum / count;
    
    log_message("SSIM calculated: %.4f\n", *ssim);
    
    return NOISE_SUCCESS;
}

NoiseError calculate_mse(
    double *mse,
    const Image *reference,
    const Image *test)
{
    if (!mse || !reference || !test) return NOISE_ERROR_INVALID_PARAM;
    if (!images_are_compatible(reference, test)) {
        return NOISE_ERROR_DIMENSION_MISMATCH;
    }
    
    double sum = 0.0;
    size_t pixel_count = reference->width * reference->height * reference->channels;
    
    if (reference->data_type == DATA_TYPE_UINT8) {
        uint8_t *ref_data = (uint8_t*)reference->data;
        uint8_t *test_data = (uint8_t*)test->data;
        
        for (int y = 0; y < reference->height; y++) {
            for (int x = 0; x < reference->width; x++) {
                for (int c = 0; c < reference->channels; c++) {
                    int idx = y * reference->stride + x * reference->channels + c;
                    double diff = (double)ref_data[idx] - (double)test_data[idx];
                    sum += diff * diff;
                }
            }
        }
    } else if (reference->data_type == DATA_TYPE_FLOAT32) {
        float *ref_data = (float*)reference->data;
        float *test_data = (float*)test->data;
        
        for (int y = 0; y < reference->height; y++) {
            for (int x = 0; x < reference->width; x++) {
                for (int c = 0; c < reference->channels; c++) {
                    int idx = (y * reference->stride / sizeof(float)) + 
                             x * reference->channels + c;
                    double diff = (double)ref_data[idx] - (double)test_data[idx];
                    sum += diff * diff;
                }
            }
        }
    }
    
    *mse = sum / pixel_count;
    
    log_message("MSE calculated: %.4f\n", *mse);
    
    return NOISE_SUCCESS;
}

NoiseError calculate_snr(
    double *snr,
    const Image *reference,
    const Image *test)
{
    if (!snr || !reference || !test) return NOISE_ERROR_INVALID_PARAM;
    if (!images_are_compatible(reference, test)) {
        return NOISE_ERROR_DIMENSION_MISMATCH;
    }
    
    double signal_power = 0.0;
    double noise_power = 0.0;
    size_t pixel_count = reference->width * reference->height * reference->channels;
    
    if (reference->data_type == DATA_TYPE_UINT8) {
        uint8_t *ref_data = (uint8_t*)reference->data;
        uint8_t *test_data = (uint8_t*)test->data;
        
        for (int y = 0; y < reference->height; y++) {
            for (int x = 0; x < reference->width; x++) {
                for (int c = 0; c < reference->channels; c++) {
                    int idx = y * reference->stride + x * reference->channels + c;
                    double signal = (double)ref_data[idx];
                    double noise = (double)ref_data[idx] - (double)test_data[idx];
                    signal_power += signal * signal;
                    noise_power += noise * noise;
                }
            }
        }
    }
    
    signal_power /= pixel_count;
    noise_power /= pixel_count;
    
    if (noise_power < 1e-10) {
        *snr = 100.0; // Perfect match
    } else {
        *snr = 10.0 * log10(signal_power / noise_power);
    }
    
    log_message("SNR calculated: %.2f dB\n", *snr);
    
    return NOISE_SUCCESS;
}

// ============================================================================
// Image Comparison and Difference
// ============================================================================

NoiseError compute_difference(
    Image *diff,
    const Image *image1,
    const Image *image2)
{
    if (!diff || !image1 || !image2) return NOISE_ERROR_INVALID_PARAM;
    if (!images_are_compatible(image1, image2)) {
        return NOISE_ERROR_DIMENSION_MISMATCH;
    }
    if (!images_are_compatible(diff, image1)) {
        return NOISE_ERROR_DIMENSION_MISMATCH;
    }
    
    if (image1->data_type == DATA_TYPE_UINT8) {
        uint8_t *data1 = (uint8_t*)image1->data;
        uint8_t *data2 = (uint8_t*)image2->data;
        uint8_t *diff_data = (uint8_t*)diff->data;
        
        for (int y = 0; y < image1->height; y++) {
            for (int x = 0; x < image1->width; x++) {
                for (int c = 0; c < image1->channels; c++) {
                    int idx = y * image1->stride + x * image1->channels + c;
                    int difference = abs((int)data1[idx] - (int)data2[idx]);
                    diff_data[idx] = (uint8_t)clamp_int(difference, 0, 255);
                }
            }
        }
    } else if (image1->data_type == DATA_TYPE_FLOAT32) {
        float *data1 = (float*)image1->data;
        float *data2 = (float*)image2->data;
        float *diff_data = (float*)diff->data;
        
        for (int y = 0; y < image1->height; y++) {
            for (int x = 0; x < image1->width; x++) {
                for (int c = 0; c < image1->channels; c++) {
                    int idx = (y * image1->stride / sizeof(float)) + 
                             x * image1->channels + c;
                    diff_data[idx] = fabs(data1[idx] - data2[idx]);
                }
            }
        }
    }
    
    log_message("Difference image computed\n");
    
    return NOISE_SUCCESS;
}

// ============================================================================
// Benchmarking
// ============================================================================

NoiseError benchmark_filters(
    const Image *input,
    BenchmarkResult *results,
    int *result_count)
{
    if (!input || !results || !result_count) return NOISE_ERROR_INVALID_PARAM;
    
    log_message("Starting filter benchmark on %dx%d image\n", 
               input->width, input->height);
    
    Image *output = NULL;
    NoiseError err = image_create(&output, input->width, input->height,
                                 input->channels, input->data_type);
    if (err != NOISE_SUCCESS) return err;
    
    int idx = 0;
    
    // Benchmark Gaussian filter
    start_timer();
    err = gaussian_filter(output, input, 5, 1.0);
    stop_timer();
    if (err == NOISE_SUCCESS) {
        strncpy(results[idx].method_name, "Gaussian", 63);
        results[idx].execution_time = g_config.last_operation_time;
        results[idx].memory_used = output->stride * output->height;
        calculate_psnr(&results[idx].psnr, input, output);
        idx++;
    }
    
    // Benchmark Median filter
    start_timer();
    err = median_filter(output, input, 5);
    stop_timer();
    if (err == NOISE_SUCCESS) {
        strncpy(results[idx].method_name, "Median", 63);
        results[idx].execution_time = g_config.last_operation_time;
        results[idx].memory_used = output->stride * output->height;
        calculate_psnr(&results[idx].psnr, input, output);
        idx++;
    }
    
    // Benchmark Bilateral filter
    start_timer();
    err = bilateral_filter(output, input, 7, 25.0, 5.0);
    stop_timer();
    if (err == NOISE_SUCCESS) {
        strncpy(results[idx].method_name, "Bilateral", 63);
        results[idx].execution_time = g_config.last_operation_time;
        results[idx].memory_used = output->stride * output->height;
        calculate_psnr(&results[idx].psnr, input, output);
        idx++;
    }
    
    // Benchmark Wiener filter
    start_timer();
    err = wiener_filter(output, input, 5, 100.0);
    stop_timer();
    if (err == NOISE_SUCCESS) {
        strncpy(results[idx].method_name, "Wiener", 63);
        results[idx].execution_time = g_config.last_operation_time;
        results[idx].memory_used = output->stride * output->height;
        calculate_psnr(&results[idx].psnr, input, output);
        idx++;
    }
    
    // Benchmark Anisotropic diffusion
    start_timer();
    err = anisotropic_diffusion(output, input, 5, 15.0, 0.2, 1);
    stop_timer();
    if (err == NOISE_SUCCESS) {
        strncpy(results[idx].method_name, "Anisotropic", 63);
        results[idx].execution_time = g_config.last_operation_time;
        results[idx].memory_used = output->stride * output->height;
        calculate_psnr(&results[idx].psnr, input, output);
        idx++;
    }
    
    *result_count = idx;
    
    image_destroy(output);
    
    log_message("Benchmark completed: %d filters tested\n", idx);
    
    return NOISE_SUCCESS;
}

// ============================================================================
// Utility Functions
// ============================================================================

const char* noise_error_string(NoiseError error) {
    switch (error) {
        case NOISE_SUCCESS:
            return "Success";
        case NOISE_ERROR_INVALID_PARAM:
            return "Invalid parameter";
        case NOISE_ERROR_OUT_OF_MEMORY:
            return "Out of memory";
        case NOISE_ERROR_FILE_NOT_FOUND:
            return "File not found";
        case NOISE_ERROR_FILE_READ:
            return "File read error";
        case NOISE_ERROR_FILE_WRITE:
            return "File write error";
        case NOISE_ERROR_UNSUPPORTED_FORMAT:
            return "Unsupported format";
        case NOISE_ERROR_DIMENSION_MISMATCH:
            return "Dimension mismatch";
        case NOISE_ERROR_PROCESSING_FAILED:
            return "Processing failed";
        case NOISE_ERROR_CANCELLED:
            return "Operation cancelled";
        default:
            return "Unknown error";
    }
}

const char* noise_type_name(NoiseType type) {
    switch (type) {
        case NOISE_TYPE_GAUSSIAN:
            return "Gaussian";
        case NOISE_TYPE_SALT_PEPPER:
            return "Salt & Pepper";
        case NOISE_TYPE_POISSON:
            return "Poisson";
        case NOISE_TYPE_SPECKLE:
            return "Speckle";
        case NOISE_TYPE_UNIFORM:
            return "Uniform";
        default:
            return "Unknown";
    }
}

const char* denoise_method_name(DenoiseMethod method) {
    switch (method) {
        case DENOISE_METHOD_GAUSSIAN:
            return "Gaussian";
        case DENOISE_METHOD_MEDIAN:
            return "Median";
        case DENOISE_METHOD_BILATERAL:
            return "Bilateral";
        case DENOISE_METHOD_NLM:
            return "Non-Local Means";
        case DENOISE_METHOD_ANISOTROPIC:
            return "Anisotropic Diffusion";
        case DENOISE_METHOD_TV:
            return "Total Variation";
        case DENOISE_METHOD_WAVELET:
            return "Wavelet";
        case DENOISE_METHOD_WIENER:
            return "Wiener";
        case DENOISE_METHOD_ADAPTIVE:
            return "Adaptive";
        case DENOISE_METHOD_AUTO:
            return "Auto";
        default:
            return "Unknown";
    }
}

NoiseError get_library_version(int *major, int *minor, int *patch) {
    if (!major || !minor || !patch) return NOISE_ERROR_INVALID_PARAM;
    
    *major = NOISE_REDUCTION_VERSION_MAJOR;
    *minor = NOISE_REDUCTION_VERSION_MINOR;
    *patch = NOISE_REDUCTION_VERSION_PATCH;
    
    return NOISE_SUCCESS;
}

NoiseError get_library_info(LibraryInfo *info) {
    if (!info) return NOISE_ERROR_INVALID_PARAM;
    
    info->version_major = NOISE_REDUCTION_VERSION_MAJOR;
    info->version_minor = NOISE_REDUCTION_VERSION_MINOR;
    info->version_patch = NOISE_REDUCTION_VERSION_PATCH;
    
    strncpy(info->version_string, "1.0.0", 31);
    strncpy(info->build_date, __DATE__, 31);
    strncpy(info->build_time, __TIME__, 31);
    
    info->supports_multithreading = 0;
    info->supports_gpu = 0;
    info->supports_simd = 0;
    
    return NOISE_SUCCESS;
}

// ============================================================================
// Statistics and Analysis
// ============================================================================

NoiseError compute_histogram(
    int *histogram,
    int bins,
    const Image *image,
    int channel)
{
    if (!histogram || !image || bins <= 0) return NOISE_ERROR_INVALID_PARAM;
    if (channel < 0 || channel >= image->channels) return NOISE_ERROR_INVALID_PARAM;
    
    // Initialize histogram
    memset(histogram, 0, bins * sizeof(int));
    
    if (image->data_type == DATA_TYPE_UINT8) {
        uint8_t *data = (uint8_t*)image->data;
        
        for (int y = 0; y < image->height; y++) {
            for (int x = 0; x < image->width; x++) {
                int idx = y * image->stride + x * image->channels + channel;
                int bin = (data[idx] * bins) / 256;
                bin = clamp_int(bin, 0, bins - 1);
                histogram[bin]++;
            }
        }
    }
    
    log_message("Histogram computed: %d bins, channel %d\n", bins, channel);
    
    return NOISE_SUCCESS;
}

NoiseError compute_statistics(
    ImageStatistics *stats,
    const Image *image,
    int channel)
{
    if (!stats || !image) return NOISE_ERROR_INVALID_PARAM;
    if (channel < 0 || channel >= image->channels) return NOISE_ERROR_INVALID_PARAM;
    
    double sum = 0.0;
    double sum_sq = 0.0;
    double min_val = 1e10;
    double max_val = -1e10;
    int count = image->width * image->height;
    
    if (image->data_type == DATA_TYPE_UINT8) {
        uint8_t *data = (uint8_t*)image->data;
        
        for (int y = 0; y < image->height; y++) {
            for (int x = 0; x < image->width; x++) {
                int idx = y * image->stride + x * image->channels + channel;
                double val = data[idx];
                sum += val;
                sum_sq += val * val;
                min_val = min_double(min_val, val);
                max_val = max_double(max_val, val);
            }
        }
    }
    
    stats->mean = sum / count;
    stats->variance = (sum_sq / count) - (stats->mean * stats->mean);
    stats->std_dev = sqrt(stats->variance);
    stats->min = min_val;
    stats->max = max_val;
    
    log_message("Statistics: mean=%.2f, std=%.2f, min=%.2f, max=%.2f\n",
                stats->mean, stats->std_dev, stats->min, stats->max);
    
    return NOISE_SUCCESS;
}

// ============================================================================
// Memory and Performance Profiling
// ============================================================================

NoiseError get_memory_usage(size_t *bytes_used) {
    if (!bytes_used) return NOISE_ERROR_INVALID_PARAM;
    
    // Simple implementation - in real scenario, track all allocations
    *bytes_used = 0;
    
    return NOISE_SUCCESS;
}

NoiseError reset_performance_counters(void) {
    g_config.last_operation_time = 0.0;
    return NOISE_SUCCESS;
}

NoiseError get_performance_stats(PerformanceStats *stats) {
    if (!stats) return NOISE_ERROR_INVALID_PARAM;
    
    stats->total_operations = 0;
    stats->total_time = g_config.last_operation_time;
    stats->average_time = g_config.last_operation_time;
    stats->peak_memory = 0;
    
    return NOISE_SUCCESS;
}

// ============================================================================
// Debug and Validation
// ============================================================================

NoiseError validate_image(const Image *image) {
    if (!image) return NOISE_ERROR_INVALID_PARAM;
    if (!image->data) return NOISE_ERROR_INVALID_PARAM;
    if (image->width <= 0 || image->height <= 0) return NOISE_ERROR_INVALID_PARAM;
    if (image->channels <= 0 || image->channels > 4) return NOISE_ERROR_INVALID_PARAM;
    if (image->stride < image->width * image->channels) return NOISE_ERROR_INVALID_PARAM;
    
    return NOISE_SUCCESS;
}

NoiseError print_image_info(const Image *image) {
    if (!image) return NOISE_ERROR_INVALID_PARAM;
    
    log_message("Image Info:\n");
    log_message("  Dimensions: %dx%d\n", image->width, image->height);
    log_message("  Channels: %d\n", image->channels);
    log_message("  Data type: %s\n", 
               image->data_type == DATA_TYPE_UINT8 ? "uint8" : "float32");
    log_message("  Stride: %d bytes\n", image->stride);
    log_message("  Total size: %zu bytes\n", 
               (size_t)image->stride * image->height);
    
    return NOISE_SUCCESS;
}

// ============================================================================
// Cleanup and Finalization
// ============================================================================

NoiseError cleanup_library(void) {
    log_message("Cleaning up noise reduction library\n");
    
    // Close log file if open
    if (g_config.log_file) {
        fclose(g_config.log_file);
        g_config.log_file = NULL;
    }
    
    return NOISE_SUCCESS;
}

// ============================================================================
// End of noise_reduction.c
// ============================================================================

