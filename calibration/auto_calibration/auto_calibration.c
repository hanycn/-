/**
 * @file auto_calibration.c
 * @brief 自动校准系统实现
 * @author hany
 * @date 2025
 */

#include "auto_calibration.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <time.h>

#ifdef _OPENMP
#include <omp.h>
#endif

// ============================================================================
// 内部宏定义
// ============================================================================

#define AUTO_CALIB_CHECK_NULL(ptr) \
    do { \
        if ((ptr) == NULL) { \
            return AUTO_CALIB_ERROR_NULL_POINTER; \
        } \
    } while(0)

#define AUTO_CALIB_MALLOC(type, count) \
    ((type*)malloc((count) * sizeof(type)))

#define AUTO_CALIB_CALLOC(type, count) \
    ((type*)calloc((count), sizeof(type)))

#define AUTO_CALIB_FREE(ptr) \
    do { \
        if ((ptr) != NULL) { \
            free(ptr); \
            (ptr) = NULL; \
        } \
    } while(0)

// ============================================================================
// 内部结构体定义
// ============================================================================

/**
 * @brief 自动校准上下文内部结构
 */
struct AutoCalibrationContext {
    AutoCalibrationConfig config;
    AutoCalibrationStatus status;
    AutoCalibrationProgress progress;
    
    // 回调函数
    AutoCalibProgressCallback progress_callback;
    void *progress_callback_data;
    AutoCalibImageValidator image_validator;
    void *image_validator_data;
    
    // 取消标志
    volatile bool cancel_requested;
    
    // 时间统计
    time_t start_time;
    time_t last_update_time;
    
    // 临时数据
    float *temp_buffer;
    size_t temp_buffer_size;
    
    // 统计信息
    int total_images_processed;
    int images_rejected;
};

/**
 * @brief 图像统计信息
 */
typedef struct {
    float mean;
    float std_dev;
    float min_val;
    float max_val;
    float median;
    float mad;  // Median Absolute Deviation
} ImageStatistics;

/**
 * @brief 优化参数
 */
typedef struct {
    int max_iterations;
    float convergence_threshold;
    float regularization;
    float step_size;
    bool verbose;
} OptimizationParams;

// ============================================================================
// 内部函数声明
// ============================================================================

// 统计函数
static AutoCalibrationError compute_image_stats(
    const float *image,
    int width,
    int height,
    ImageStatistics *stats);

static float compute_median(float *data, int size);
static float compute_mad(const float *data, int size, float median);

// 异常值检测
static bool is_outlier_image(
    const float *image,
    int width,
    int height,
    const ImageStatistics *reference_stats,
    float threshold);

// 图像对齐
static AutoCalibrationError align_image_pair(
    const float *reference,
    const float *target,
    int width,
    int height,
    float *aligned,
    float *dx,
    float *dy);

// 进度更新
static void update_progress(
    AutoCalibrationContext *context,
    const char *operation,
    int current_step,
    int total_steps);

// 内存管理
static AutoCalibrationError ensure_temp_buffer(
    AutoCalibrationContext *context,
    size_t required_size);

// 验证函数
static bool validate_image_data(
    const float *image,
    int width,
    int height);

// ============================================================================
// 配置管理实现
// ============================================================================

AutoCalibrationConfig auto_calib_create_default_config(void) {
    AutoCalibrationConfig config;
    memset(&config, 0, sizeof(AutoCalibrationConfig));
    
    // 通用设置
    config.priority = AUTO_CALIB_PRIORITY_BALANCED;
    config.max_iterations = 50;
    config.convergence_threshold = 1e-4f;
    config.enable_validation = true;
    config.enable_progress_callback = true;
    
#ifdef _OPENMP
    config.num_threads = omp_get_max_threads();
#else
    config.num_threads = 1;
#endif
    
    // 平场校准设置
    config.flat_method = AUTO_CALIB_FLAT_METHOD_ROBUST_MEAN;
    config.min_flat_images = 5;
    config.flat_outlier_threshold = 3.0f;
    config.flat_enable_smoothing = true;
    config.flat_smoothing_sigma = 1.0f;
    
    // 暗场校准设置
    config.dark_method = AUTO_CALIB_DARK_METHOD_ROBUST_MEAN;
    config.min_dark_images = 5;
    config.dark_outlier_threshold = 3.0f;
    config.dark_enable_temporal_filter = false;
    
    // PSF估计设置
    config.psf_method = AUTO_CALIB_PSF_METHOD_EDGE_BASED;
    config.psf_size = 15;
    config.psf_num_iterations = 20;
    config.psf_regularization = 0.01f;
    config.psf_enforce_symmetry = true;
    config.psf_enforce_positivity = true;
    
    // 颜色矩阵设置
    config.color_method = AUTO_CALIB_COLOR_METHOD_LEAST_SQUARES;
    config.num_color_patches = 24;
    config.color_regularization = 0.001f;
    config.color_preserve_luminance = true;
    
    // 像差估计设置
    config.aberration_method = AUTO_CALIB_ABERR_METHOD_ZERNIKE;
    config.num_zernike_terms = 15;
    config.aberration_regularization = 0.01f;
    
    // 质量控制
    config.min_quality_score = 70.0f;
    config.auto_reject_bad_images = true;
    config.bad_image_threshold = 2.5f;
    
    return config;
}

AutoCalibrationConfig auto_calib_create_fast_config(void) {
    AutoCalibrationConfig config = auto_calib_create_default_config();
    
    config.priority = AUTO_CALIB_PRIORITY_SPEED;
    config.max_iterations = 20;
    config.convergence_threshold = 1e-3f;
    
    config.flat_method = AUTO_CALIB_FLAT_METHOD_MEAN;
    config.min_flat_images = 3;
    config.flat_enable_smoothing = false;
    
    config.dark_method = AUTO_CALIB_DARK_METHOD_MEAN;
    config.min_dark_images = 3;
    
    config.psf_method = AUTO_CALIB_PSF_METHOD_PARAMETRIC;
    config.psf_size = 11;
    config.psf_num_iterations = 10;
    config.psf_enforce_symmetry = true;
    
    config.color_method = AUTO_CALIB_COLOR_METHOD_LEAST_SQUARES;
    config.aberration_method = AUTO_CALIB_ABERR_METHOD_GRID_BASED;
    config.num_zernike_terms = 10;
    
    config.min_quality_score = 60.0f;
    
    return config;
}

AutoCalibrationConfig auto_calib_create_quality_config(void) {
    AutoCalibrationConfig config = auto_calib_create_default_config();
    
    config.priority = AUTO_CALIB_PRIORITY_QUALITY;
    config.max_iterations = 100;
    config.convergence_threshold = 1e-5f;
    
    config.flat_method = AUTO_CALIB_FLAT_METHOD_ADAPTIVE;
    config.min_flat_images = 10;
    config.flat_enable_smoothing = true;
    config.flat_smoothing_sigma = 1.5f;
    
    config.dark_method = AUTO_CALIB_DARK_METHOD_TEMPORAL;
    config.min_dark_images = 10;
    config.dark_enable_temporal_filter = true;
    
    config.psf_method = AUTO_CALIB_PSF_METHOD_BLIND_DECONV;
    config.psf_size = 21;
    config.psf_num_iterations = 50;
    config.psf_regularization = 0.005f;
    config.psf_enforce_symmetry = true;
    config.psf_enforce_positivity = true;
    
    config.color_method = AUTO_CALIB_COLOR_METHOD_CONSTRAINED;
    config.num_color_patches = 24;
    config.color_regularization = 0.0005f;
    
    config.aberration_method = AUTO_CALIB_ABERR_METHOD_PHASE_RETRIEVAL;
    config.num_zernike_terms = 21;
    config.aberration_regularization = 0.005f;
    
    config.min_quality_score = 85.0f;
    config.auto_reject_bad_images = true;
    config.bad_image_threshold = 2.0f;
    
    return config;
}

AutoCalibrationError auto_calib_validate_config(
    const AutoCalibrationConfig *config)
{
    AUTO_CALIB_CHECK_NULL(config);
    
    // 验证迭代次数
    if (config->max_iterations <= 0 || 
        config->max_iterations > AUTO_CALIB_MAX_ITERATIONS) {
        return AUTO_CALIB_ERROR_INVALID_PARAM;
    }
    
    // 验证收敛阈值
    if (config->convergence_threshold <= 0.0f || 
        config->convergence_threshold >= 1.0f) {
        return AUTO_CALIB_ERROR_INVALID_PARAM;
    }
    
    // 验证线程数
    if (config->num_threads <= 0) {
        return AUTO_CALIB_ERROR_INVALID_PARAM;
    }
    
    // 验证平场设置
    if (config->min_flat_images < AUTO_CALIB_MIN_IMAGES) {
        return AUTO_CALIB_ERROR_INVALID_PARAM;
    }
    
    if (config->flat_outlier_threshold <= 0.0f) {
        return AUTO_CALIB_ERROR_INVALID_PARAM;
    }
    
    // 验证暗场设置
    if (config->min_dark_images < AUTO_CALIB_MIN_IMAGES) {
        return AUTO_CALIB_ERROR_INVALID_PARAM;
    }
    
    if (config->dark_outlier_threshold <= 0.0f) {
        return AUTO_CALIB_ERROR_INVALID_PARAM;
    }
    
    // 验证PSF设置
    if (config->psf_size < AUTO_CALIB_MIN_PSF_SIZE || 
        config->psf_size > AUTO_CALIB_MAX_PSF_SIZE) {
        return AUTO_CALIB_ERROR_INVALID_PARAM;
    }
    
    if (config->psf_size % 2 == 0) {
        // PSF大小应该是奇数
        return AUTO_CALIB_ERROR_INVALID_PARAM;
    }
    
    if (config->psf_num_iterations <= 0) {
        return AUTO_CALIB_ERROR_INVALID_PARAM;
    }
    
    if (config->psf_regularization < 0.0f) {
        return AUTO_CALIB_ERROR_INVALID_PARAM;
    }
    
    // 验证颜色矩阵设置
    if (config->num_color_patches <= 0) {
        return AUTO_CALIB_ERROR_INVALID_PARAM;
    }
    
    if (config->color_regularization < 0.0f) {
        return AUTO_CALIB_ERROR_INVALID_PARAM;
    }
    
    // 验证像差设置
    if (config->num_zernike_terms <= 0 || 
        config->num_zernike_terms > CALIB_MAX_ABERRATION_TERMS) {
        return AUTO_CALIB_ERROR_INVALID_PARAM;
    }
    
    if (config->aberration_regularization < 0.0f) {
        return AUTO_CALIB_ERROR_INVALID_PARAM;
    }
    
    // 验证质量控制
    if (config->min_quality_score < 0.0f || 
        config->min_quality_score > 100.0f) {
        return AUTO_CALIB_ERROR_INVALID_PARAM;
    }
    
    if (config->bad_image_threshold <= 0.0f) {
        return AUTO_CALIB_ERROR_INVALID_PARAM;
    }
    
    return AUTO_CALIB_SUCCESS;
}

void auto_calib_print_config(const AutoCalibrationConfig *config) {
    if (!config) {
        printf("No configuration available\n");
        return;
    }
    
    printf("=== Auto Calibration Configuration ===\n");
    
    printf("\nGeneral Settings:\n");
    printf("  Priority: ");
    switch (config->priority) {
        case AUTO_CALIB_PRIORITY_SPEED:
            printf("Speed\n");
            break;
        case AUTO_CALIB_PRIORITY_QUALITY:
            printf("Quality\n");
            break;
        case AUTO_CALIB_PRIORITY_BALANCED:
            printf("Balanced\n");
            break;
    }
    printf("  Max Iterations: %d\n", config->max_iterations);
    printf("  Convergence Threshold: %.6f\n", config->convergence_threshold);
    printf("  Enable Validation: %s\n", config->enable_validation ? "Yes" : "No");
    printf("  Enable Progress Callback: %s\n", 
           config->enable_progress_callback ? "Yes" : "No");
    printf("  Number of Threads: %d\n", config->num_threads);
    
    printf("\nFlat Field Settings:\n");
    printf("  Method: ");
    switch (config->flat_method) {
        case AUTO_CALIB_FLAT_METHOD_MEAN:
            printf("Mean\n");
            break;
        case AUTO_CALIB_FLAT_METHOD_MEDIAN:
            printf("Median\n");
            break;
        case AUTO_CALIB_FLAT_METHOD_ROBUST_MEAN:
            printf("Robust Mean\n");
            break;
        case AUTO_CALIB_FLAT_METHOD_WEIGHTED_MEAN:
            printf("Weighted Mean\n");
            break;
        case AUTO_CALIB_FLAT_METHOD_ADAPTIVE:
            printf("Adaptive\n");
            break;
    }
    printf("  Min Images: %d\n", config->min_flat_images);
    printf("  Outlier Threshold: %.2f\n", config->flat_outlier_threshold);
    printf("  Enable Smoothing: %s\n", 
           config->flat_enable_smoothing ? "Yes" : "No");
    printf("  Smoothing Sigma: %.2f\n", config->flat_smoothing_sigma);
    
    printf("\nDark Field Settings:\n");
    printf("  Method: ");
    switch (config->dark_method) {
        case AUTO_CALIB_DARK_METHOD_MEAN:
            printf("Mean\n");
            break;
        case AUTO_CALIB_DARK_METHOD_MEDIAN:
            printf("Median\n");
            break;
        case AUTO_CALIB_DARK_METHOD_ROBUST_MEAN:
            printf("Robust Mean\n");
            break;
        case AUTO_CALIB_DARK_METHOD_TEMPORAL:
            printf("Temporal\n");
            break;
    }
    printf("  Min Images: %d\n", config->min_dark_images);
    printf("  Outlier Threshold: %.2f\n", config->dark_outlier_threshold);
    printf("  Enable Temporal Filter: %s\n", 
           config->dark_enable_temporal_filter ? "Yes" : "No");
    
    printf("\nPSF Estimation Settings:\n");
    printf("  Method: ");
    switch (config->psf_method) {
        case AUTO_CALIB_PSF_METHOD_BLIND_DECONV:
            printf("Blind Deconvolution\n");
            break;
        case AUTO_CALIB_PSF_METHOD_EDGE_BASED:
            printf("Edge Based\n");
            break;
        case AUTO_CALIB_PSF_METHOD_POINT_SOURCE:
            printf("Point Source\n");
            break;
        case AUTO_CALIB_PSF_METHOD_PARAMETRIC:
            printf("Parametric\n");
            break;
        case AUTO_CALIB_PSF_METHOD_LEARNING_BASED:
            printf("Learning Based\n");
            break;
    }
    printf("  PSF Size: %d\n", config->psf_size);
    printf("  Num Iterations: %d\n", config->psf_num_iterations);
    printf("  Regularization: %.6f\n", config->psf_regularization);
    printf("  Enforce Symmetry: %s\n", 
           config->psf_enforce_symmetry ? "Yes" : "No");
    printf("  Enforce Positivity: %s\n", 
           config->psf_enforce_positivity ? "Yes" : "No");
    
    printf("\nColor Matrix Settings:\n");
    printf("  Method: ");
    switch (config->color_method) {
        case AUTO_CALIB_COLOR_METHOD_LEAST_SQUARES:
            printf("Least Squares\n");
            break;
        case AUTO_CALIB_COLOR_METHOD_ROBUST:
            printf("Robust\n");
            break;
        case AUTO_CALIB_COLOR_METHOD_CONSTRAINED:
            printf("Constrained\n");
            break;
        case AUTO_CALIB_COLOR_METHOD_ITERATIVE:
            printf("Iterative\n");
            break;
    }
    printf("  Num Color Patches: %d\n", config->num_color_patches);
    printf("  Regularization: %.6f\n", config->color_regularization);
    printf("  Preserve Luminance: %s\n", 
           config->color_preserve_luminance ? "Yes" : "No");
    
    printf("\nAberration Settings:\n");
    printf("  Method: ");
    switch (config->aberration_method) {
        case AUTO_CALIB_ABERR_METHOD_ZERNIKE:
            printf("Zernike\n");
            break;
        case AUTO_CALIB_ABERR_METHOD_GRID_BASED:
            printf("Grid Based\n");
            break;
        case AUTO_CALIB_ABERR_METHOD_FEATURE_BASED:
            printf("Feature Based\n");
            break;
        case AUTO_CALIB_ABERR_METHOD_PHASE_RETRIEVAL:
            printf("Phase Retrieval\n");
            break;
    }
    printf("  Num Zernike Terms: %d\n", config->num_zernike_terms);
    printf("  Regularization: %.6f\n", config->aberration_regularization);
    
    printf("\nQuality Control:\n");
    printf("  Min Quality Score: %.2f\n", config->min_quality_score);
    printf("  Auto Reject Bad Images: %s\n", 
           config->auto_reject_bad_images ? "Yes" : "No");
    printf("  Bad Image Threshold: %.2f\n", config->bad_image_threshold);
    
    printf("=======================================\n");
}

// ============================================================================
// 上下文管理实现
// ============================================================================

AutoCalibrationContext* auto_calib_create_context(
    const AutoCalibrationConfig *config)
{
    if (!config) {
        return NULL;
    }
    
    // 验证配置
    if (auto_calib_validate_config(config) != AUTO_CALIB_SUCCESS) {
        return NULL;
    }
    
    // 分配上下文
    AutoCalibrationContext *context = AUTO_CALIB_CALLOC(
        AutoCalibrationContext, 1);
    if (!context) {
        return NULL;
    }
    
    // 复制配置
    memcpy(&context->config, config, sizeof(AutoCalibrationConfig));
    
    // 初始化状态
    context->status = AUTO_CALIB_STATUS_NOT_STARTED;
    context->cancel_requested = false;
    
    // 初始化进度
    memset(&context->progress, 0, sizeof(AutoCalibrationProgress));
    context->progress.status = AUTO_CALIB_STATUS_NOT_STARTED;
    
    // 初始化回调
    context->progress_callback = NULL;
    context->progress_callback_data = NULL;
    context->image_validator = NULL;
    context->image_validator_data = NULL;
    
    // 初始化临时缓冲区
    context->temp_buffer = NULL;
    context->temp_buffer_size = 0;
    
    // 初始化统计
    context->total_images_processed = 0;
    context->images_rejected = 0;
    
    return context;
}

void auto_calib_destroy_context(AutoCalibrationContext *context) {
    if (!context) {
        return;
    }
    
    // 释放临时缓冲区
    AUTO_CALIB_FREE(context->temp_buffer);
    
    // 释放上下文
    free(context);
}

AutoCalibrationError auto_calib_reset_context(
    AutoCalibrationContext *context)
{
    AUTO_CALIB_CHECK_NULL(context);
    
    // 重置状态
    context->status = AUTO_CALIB_STATUS_NOT_STARTED;
    context->cancel_requested = false;
    
    // 重置进度
    memset(&context->progress, 0, sizeof(AutoCalibrationProgress));
    context->progress.status = AUTO_CALIB_STATUS_NOT_STARTED;
    
    // 重置统计
    context->total_images_processed = 0;
    context->images_rejected = 0;
    
    return AUTO_CALIB_SUCCESS;
}

AutoCalibrationError auto_calib_set_progress_callback(
    AutoCalibrationContext *context,
    AutoCalibProgressCallback callback,
    void *user_data)
{
    AUTO_CALIB_CHECK_NULL(context);
    
    context->progress_callback = callback;
    context->progress_callback_data = user_data;
    
    return AUTO_CALIB_SUCCESS;
}

AutoCalibrationError auto_calib_set_image_validator(
    AutoCalibrationContext *context,
    AutoCalibImageValidator validator,
    void *user_data)
{
    AUTO_CALIB_CHECK_NULL(context);
    
    context->image_validator = validator;
    context->image_validator_data = user_data;
    
    return AUTO_CALIB_SUCCESS;
}

AutoCalibrationError auto_calib_cancel(AutoCalibrationContext *context) {
    AUTO_CALIB_CHECK_NULL(context);
    
    context->cancel_requested = true;
    context->status = AUTO_CALIB_STATUS_CANCELLED;
    context->progress.status = AUTO_CALIB_STATUS_CANCELLED;
    
    return AUTO_CALIB_SUCCESS;
}

AutoCalibrationStatus auto_calib_get_status(
    const AutoCalibrationContext *context)
{
    if (!context) {
        return AUTO_CALIB_STATUS_NOT_STARTED;
    }
    
    return context->status;
}

AutoCalibrationError auto_calib_get_progress(
    const AutoCalibrationContext *context,
    AutoCalibrationProgress *progress)
{
    AUTO_CALIB_CHECK_NULL(context);
    AUTO_CALIB_CHECK_NULL(progress);
    
    memcpy(progress, &context->progress, sizeof(AutoCalibrationProgress));
    
    return AUTO_CALIB_SUCCESS;
}

// ============================================================================
// 内部辅助函数实现
// ============================================================================

/**
 * @brief 更新进度信息
 */
static void update_progress(
    AutoCalibrationContext *context,
    const char *operation,
    int current_step,
    int total_steps)
{
    if (!context) {
        return;
    }
    
    // 更新进度信息
    context->progress.current_step = current_step;
    context->progress.total_steps = total_steps;
    context->progress.progress_percentage = 
        (total_steps > 0) ? (100.0f * current_step / total_steps) : 0.0f;
    
    if (operation) {
        strncpy(context->progress.current_operation, operation, 
                sizeof(context->progress.current_operation) - 1);
        context->progress.current_operation[
            sizeof(context->progress.current_operation) - 1] = '\0';
    }
    
    // 计算时间
    time_t current_time = time(NULL);
    context->progress.elapsed_time = 
        difftime(current_time, context->start_time);
    
    if (current_step > 0 && total_steps > 0) {
        double time_per_step = context->progress.elapsed_time / current_step;
        int remaining_steps = total_steps - current_step;
        context->progress.estimated_remaining_time = 
            time_per_step * remaining_steps;
    }
    
    // 调用回调
    if (context->config.enable_progress_callback && 
        context->progress_callback) {
        context->progress_callback(&context->progress, 
                                  context->progress_callback_data);
    }
    
    context->last_update_time = current_time;
}

/**
 * @brief 确保临时缓冲区足够大
 */
static AutoCalibrationError ensure_temp_buffer(
    AutoCalibrationContext *context,
    size_t required_size)
{
    AUTO_CALIB_CHECK_NULL(context);
    
    if (context->temp_buffer_size < required_size) {
        float *new_buffer = AUTO_CALIB_MALLOC(float, required_size);
        if (!new_buffer) {
            return AUTO_CALIB_ERROR_MEMORY_ALLOCATION;
        }
        
        AUTO_CALIB_FREE(context->temp_buffer);
        context->temp_buffer = new_buffer;
        context->temp_buffer_size = required_size;
    }
    
    return AUTO_CALIB_SUCCESS;
}

/**
 * @brief 验证图像数据
 */
static bool validate_image_data(
    const float *image,
    int width,
    int height)
{
    if (!image || width <= 0 || height <= 0) {
        return false;
    }
    
    int num_pixels = width * height;
    
    // 检查是否有NaN或Inf
    for (int i = 0; i < num_pixels; i++) {
        if (isnan(image[i]) || isinf(image[i])) {
            return false;
        }
    }
    
    // 检查是否全为零
    bool all_zero = true;
    for (int i = 0; i < num_pixels; i++) {
        if (image[i] != 0.0f) {
            all_zero = false;
            break;
        }
    }
    
    if (all_zero) {
        return false;
    }
    
    return true;
}
// ============================================================================
// 图像集合管理实现
// ============================================================================

AutoCalibImageSet* auto_calib_create_image_set(
    int num_images,
    int width,
    int height)
{
    if (num_images <= 0 || width <= 0 || height <= 0) {
        return NULL;
    }
    
    AutoCalibImageSet *image_set = AUTO_CALIB_CALLOC(AutoCalibImageSet, 1);
    if (!image_set) {
        return NULL;
    }
    
    // 分配图像指针数组
    image_set->images = AUTO_CALIB_CALLOC(float*, num_images);
    if (!image_set->images) {
        free(image_set);
        return NULL;
    }
    
    // 分配波长数组
    image_set->wavelengths = AUTO_CALIB_CALLOC(float, num_images);
    if (!image_set->wavelengths) {
        free(image_set->images);
        free(image_set);
        return NULL;
    }
    
    // 分配曝光时间数组
    image_set->exposure_times = AUTO_CALIB_CALLOC(float, num_images);
    if (!image_set->exposure_times) {
        free(image_set->wavelengths);
        free(image_set->images);
        free(image_set);
        return NULL;
    }
    
    // 分配温度数组
    image_set->temperatures = AUTO_CALIB_CALLOC(float, num_images);
    if (!image_set->temperatures) {
        free(image_set->exposure_times);
        free(image_set->wavelengths);
        free(image_set->images);
        free(image_set);
        return NULL;
    }
    
    // 分配元数据数组
    image_set->metadata = AUTO_CALIB_CALLOC(char*, num_images);
    if (!image_set->metadata) {
        free(image_set->temperatures);
        free(image_set->exposure_times);
        free(image_set->wavelengths);
        free(image_set->images);
        free(image_set);
        return NULL;
    }
    
    image_set->num_images = 0;  // 实际添加的图像数量
    image_set->width = width;
    image_set->height = height;
    
    return image_set;
}

void auto_calib_destroy_image_set(AutoCalibImageSet *image_set) {
    if (!image_set) {
        return;
    }
    
    // 释放图像数据
    if (image_set->images) {
        for (int i = 0; i < image_set->num_images; i++) {
            AUTO_CALIB_FREE(image_set->images[i]);
        }
        free(image_set->images);
    }
    
    // 释放元数据
    if (image_set->metadata) {
        for (int i = 0; i < image_set->num_images; i++) {
            AUTO_CALIB_FREE(image_set->metadata[i]);
        }
        free(image_set->metadata);
    }
    
    // 释放其他数组
    AUTO_CALIB_FREE(image_set->wavelengths);
    AUTO_CALIB_FREE(image_set->exposure_times);
    AUTO_CALIB_FREE(image_set->temperatures);
    
    free(image_set);
}

AutoCalibrationError auto_calib_add_image(
    AutoCalibImageSet *image_set,
    const float *image,
    float wavelength,
    float exposure_time,
    float temperature,
    const char *metadata)
{
    AUTO_CALIB_CHECK_NULL(image_set);
    AUTO_CALIB_CHECK_NULL(image);
    
    if (image_set->num_images >= AUTO_CALIB_MAX_IMAGES) {
        return AUTO_CALIB_ERROR_INVALID_PARAM;
    }
    
    // 验证图像数据
    if (!validate_image_data(image, image_set->width, image_set->height)) {
        return AUTO_CALIB_ERROR_INVALID_IMAGE;
    }
    
    int idx = image_set->num_images;
    int num_pixels = image_set->width * image_set->height;
    
    // 复制图像数据
    image_set->images[idx] = AUTO_CALIB_MALLOC(float, num_pixels);
    if (!image_set->images[idx]) {
        return AUTO_CALIB_ERROR_MEMORY_ALLOCATION;
    }
    memcpy(image_set->images[idx], image, num_pixels * sizeof(float));
    
    // 保存参数
    image_set->wavelengths[idx] = wavelength;
    image_set->exposure_times[idx] = exposure_time;
    image_set->temperatures[idx] = temperature;
    
    // 复制元数据
    if (metadata) {
        size_t len = strlen(metadata);
        image_set->metadata[idx] = AUTO_CALIB_MALLOC(char, len + 1);
        if (image_set->metadata[idx]) {
            strcpy(image_set->metadata[idx], metadata);
        }
    } else {
        image_set->metadata[idx] = NULL;
    }
    
    image_set->num_images++;
    
    return AUTO_CALIB_SUCCESS;
}

AutoCalibrationError auto_calib_validate_image_set(
    const AutoCalibImageSet *image_set,
    bool *is_valid,
    char *error_message,
    int error_message_size)
{
    AUTO_CALIB_CHECK_NULL(image_set);
    AUTO_CALIB_CHECK_NULL(is_valid);
    
    *is_valid = true;
    
    // 检查图像数量
    if (image_set->num_images < AUTO_CALIB_MIN_IMAGES) {
        *is_valid = false;
        if (error_message) {
            snprintf(error_message, error_message_size,
                    "Insufficient images: %d (minimum %d required)",
                    image_set->num_images, AUTO_CALIB_MIN_IMAGES);
        }
        return AUTO_CALIB_SUCCESS;
    }
    
    // 检查尺寸
    if (image_set->width <= 0 || image_set->height <= 0) {
        *is_valid = false;
        if (error_message) {
            snprintf(error_message, error_message_size,
                    "Invalid image dimensions: %dx%d",
                    image_set->width, image_set->height);
        }
        return AUTO_CALIB_SUCCESS;
    }
    
    // 检查每个图像
    for (int i = 0; i < image_set->num_images; i++) {
        if (!image_set->images[i]) {
            *is_valid = false;
            if (error_message) {
                snprintf(error_message, error_message_size,
                        "Image %d is NULL", i);
            }
            return AUTO_CALIB_SUCCESS;
        }
        
        if (!validate_image_data(image_set->images[i], 
                                image_set->width, 
                                image_set->height)) {
            *is_valid = false;
            if (error_message) {
                snprintf(error_message, error_message_size,
                        "Image %d contains invalid data", i);
            }
            return AUTO_CALIB_SUCCESS;
        }
    }
    
    return AUTO_CALIB_SUCCESS;
}

AutoCalibrationError auto_calib_preprocess_image_set(
    AutoCalibImageSet *image_set,
    bool remove_outliers,
    bool normalize,
    bool align)
{
    AUTO_CALIB_CHECK_NULL(image_set);
    
    if (image_set->num_images == 0) {
        return AUTO_CALIB_ERROR_INSUFFICIENT_DATA;
    }
    
    // 移除异常值
    if (remove_outliers) {
        bool *outlier_mask = AUTO_CALIB_CALLOC(bool, image_set->num_images);
        if (!outlier_mask) {
            return AUTO_CALIB_ERROR_MEMORY_ALLOCATION;
        }
        
        AutoCalibrationError err = auto_calib_detect_outlier_images(
            image_set, 3.0f, outlier_mask);
        
        if (err == AUTO_CALIB_SUCCESS) {
            // 移除标记为异常的图像
            int write_idx = 0;
            for (int read_idx = 0; read_idx < image_set->num_images; read_idx++) {
                if (!outlier_mask[read_idx]) {
                    if (write_idx != read_idx) {
                        // 移动数据
                        image_set->images[write_idx] = image_set->images[read_idx];
                        image_set->wavelengths[write_idx] = image_set->wavelengths[read_idx];
                        image_set->exposure_times[write_idx] = image_set->exposure_times[read_idx];
                        image_set->temperatures[write_idx] = image_set->temperatures[read_idx];
                        image_set->metadata[write_idx] = image_set->metadata[read_idx];
                    }
                    write_idx++;
                } else {
                    // 释放异常图像
                    AUTO_CALIB_FREE(image_set->images[read_idx]);
                    AUTO_CALIB_FREE(image_set->metadata[read_idx]);
                }
            }
            image_set->num_images = write_idx;
        }
        
        free(outlier_mask);
    }
    
    // 归一化
    if (normalize) {
        AutoCalibrationError err = auto_calib_normalize_image_set(image_set);
        if (err != AUTO_CALIB_SUCCESS) {
            return err;
        }
    }
    
    // 对齐
    if (align) {
        AutoCalibrationError err = auto_calib_align_images(image_set, 0);
        if (err != AUTO_CALIB_SUCCESS) {
            return err;
        }
    }
    
    return AUTO_CALIB_SUCCESS;
}

// ============================================================================
// 统计函数实现
// ============================================================================

static AutoCalibrationError compute_image_stats(
    const float *image,
    int width,
    int height,
    ImageStatistics *stats)
{
    if (!image || !stats || width <= 0 || height <= 0) {
        return AUTO_CALIB_ERROR_INVALID_PARAM;
    }
    
    int num_pixels = width * height;
    
    // 计算均值和范围
    double sum = 0.0;
    float min_val = FLT_MAX;
    float max_val = -FLT_MAX;
    
    for (int i = 0; i < num_pixels; i++) {
        float val = image[i];
        sum += val;
        if (val < min_val) min_val = val;
        if (val > max_val) max_val = val;
    }
    
    stats->mean = sum / num_pixels;
    stats->min_val = min_val;
    stats->max_val = max_val;
    
    // 计算标准差
    double sum_sq_diff = 0.0;
    for (int i = 0; i < num_pixels; i++) {
        double diff = image[i] - stats->mean;
        sum_sq_diff += diff * diff;
    }
    stats->std_dev = sqrt(sum_sq_diff / num_pixels);
    
    // 计算中值
    float *sorted = AUTO_CALIB_MALLOC(float, num_pixels);
    if (!sorted) {
        return AUTO_CALIB_ERROR_MEMORY_ALLOCATION;
    }
    memcpy(sorted, image, num_pixels * sizeof(float));
    stats->median = compute_median(sorted, num_pixels);
    
    // 计算MAD
    stats->mad = compute_mad(image, num_pixels, stats->median);
    
    free(sorted);
    
    return AUTO_CALIB_SUCCESS;
}

static float compute_median(float *data, int size) {
    if (!data || size <= 0) {
        return 0.0f;
    }
    
    // 简单的选择排序（对于小数据集）
    // 对于大数据集，应该使用快速选择算法
    for (int i = 0; i < size - 1; i++) {
        for (int j = i + 1; j < size; j++) {
            if (data[j] < data[i]) {
                float temp = data[i];
                data[i] = data[j];
                data[j] = temp;
            }
        }
    }
    
    if (size % 2 == 0) {
        return (data[size/2 - 1] + data[size/2]) / 2.0f;
    } else {
        return data[size/2];
    }
}

static float compute_mad(const float *data, int size, float median) {
    if (!data || size <= 0) {
        return 0.0f;
    }
    
    float *abs_deviations = AUTO_CALIB_MALLOC(float, size);
    if (!abs_deviations) {
        return 0.0f;
    }
    
    for (int i = 0; i < size; i++) {
        abs_deviations[i] = fabsf(data[i] - median);
    }
    
    float mad = compute_median(abs_deviations, size);
    free(abs_deviations);
    
    return mad;
}

// ============================================================================
// 异常值检测实现
// ============================================================================

AutoCalibrationError auto_calib_detect_outlier_images(
    const AutoCalibImageSet *image_set,
    float threshold,
    bool *outlier_mask)
{
    AUTO_CALIB_CHECK_NULL(image_set);
    AUTO_CALIB_CHECK_NULL(outlier_mask);
    
    if (image_set->num_images < AUTO_CALIB_MIN_IMAGES) {
        return AUTO_CALIB_ERROR_INSUFFICIENT_DATA;
    }
    
    int num_pixels = image_set->width * image_set->height;
    
    // 计算每个图像的统计信息
    ImageStatistics *stats = AUTO_CALIB_CALLOC(
        ImageStatistics, image_set->num_images);
    if (!stats) {
        return AUTO_CALIB_ERROR_MEMORY_ALLOCATION;
    }
    
    for (int i = 0; i < image_set->num_images; i++) {
        AutoCalibrationError err = compute_image_stats(
            image_set->images[i],
            image_set->width,
            image_set->height,
            &stats[i]);
        
        if (err != AUTO_CALIB_SUCCESS) {
            free(stats);
            return err;
        }
    }
    
    // 计算统计量的中值和MAD
    float *means = AUTO_CALIB_MALLOC(float, image_set->num_images);
    float *std_devs = AUTO_CALIB_MALLOC(float, image_set->num_images);
    
    if (!means || !std_devs) {
        free(stats);
        AUTO_CALIB_FREE(means);
        AUTO_CALIB_FREE(std_devs);
        return AUTO_CALIB_ERROR_MEMORY_ALLOCATION;
    }
    
    for (int i = 0; i < image_set->num_images; i++) {
        means[i] = stats[i].mean;
        std_devs[i] = stats[i].std_dev;
    }
    
    float median_mean = compute_median(means, image_set->num_images);
    float mad_mean = compute_mad(means, image_set->num_images, median_mean);
    
    float median_std = compute_median(std_devs, image_set->num_images);
    float mad_std = compute_mad(std_devs, image_set->num_images, median_std);
    
    // 检测异常值
    for (int i = 0; i < image_set->num_images; i++) {
        float z_mean = fabsf(stats[i].mean - median_mean) / (mad_mean + 1e-10f);
        float z_std = fabsf(stats[i].std_dev - median_std) / (mad_std + 1e-10f);
        
        outlier_mask[i] = (z_mean > threshold) || (z_std > threshold);
    }
    
    free(stats);
    free(means);
    free(std_devs);
    
    return AUTO_CALIB_SUCCESS;
}

static bool is_outlier_image(
    const float *image,
    int width,
    int height,
    const ImageStatistics *reference_stats,
    float threshold)
{
    if (!image || !reference_stats) {
        return true;
    }
    
    ImageStatistics stats;
    if (compute_image_stats(image, width, height, &stats) != AUTO_CALIB_SUCCESS) {
        return true;
    }
    
    // 使用MAD进行鲁棒的异常值检测
    float z_mean = fabsf(stats.mean - reference_stats->median) / 
                   (reference_stats->mad + 1e-10f);
    float z_std = fabsf(stats.std_dev - reference_stats->std_dev) / 
                  (reference_stats->mad + 1e-10f);
    
    return (z_mean > threshold) || (z_std > threshold);
}

// ============================================================================
// 图像对齐实现
// ============================================================================

AutoCalibrationError auto_calib_align_images(
    AutoCalibImageSet *image_set,
    int reference_index)
{
    AUTO_CALIB_CHECK_NULL(image_set);
    
    if (reference_index < 0 || reference_index >= image_set->num_images) {
        return AUTO_CALIB_ERROR_INVALID_PARAM;
    }
    
    if (image_set->num_images < 2) {
        return AUTO_CALIB_SUCCESS;  // 只有一个图像，无需对齐
    }
    
    int num_pixels = image_set->width * image_set->height;
    const float *reference = image_set->images[reference_index];
    
    // 分配对齐后的图像缓冲区
    float *aligned = AUTO_CALIB_MALLOC(float, num_pixels);
    if (!aligned) {
        return AUTO_CALIB_ERROR_MEMORY_ALLOCATION;
    }
    
    // 对齐每个图像
    for (int i = 0; i < image_set->num_images; i++) {
        if (i == reference_index) {
            continue;  // 跳过参考图像
        }
        
        float dx, dy;
        AutoCalibrationError err = align_image_pair(
            reference,
            image_set->images[i],
            image_set->width,
            image_set->height,
            aligned,
            &dx,
            &dy);
        
        if (err == AUTO_CALIB_SUCCESS) {
            // 替换原图像
            memcpy(image_set->images[i], aligned, num_pixels * sizeof(float));
        }
    }
    
    free(aligned);
    
    return AUTO_CALIB_SUCCESS;
}

static AutoCalibrationError align_image_pair(
    const float *reference,
    const float *target,
    int width,
    int height,
    float *aligned,
    float *dx,
    float *dy)
{
    if (!reference || !target || !aligned || !dx || !dy) {
        return AUTO_CALIB_ERROR_NULL_POINTER;
    }
    
    // 使用相位相关法进行亚像素对齐
    // 这里实现简化版本：整像素平移
    
    int max_shift = 10;  // 最大搜索范围
    float best_correlation = -FLT_MAX;
    int best_shift_x = 0;
    int best_shift_y = 0;
    
    // 搜索最佳平移
    for (int shift_y = -max_shift; shift_y <= max_shift; shift_y++) {
        for (int shift_x = -max_shift; shift_x <= max_shift; shift_x++) {
            double correlation = 0.0;
            int count = 0;
            
            // 计算重叠区域的相关性
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    int target_x = x + shift_x;
                    int target_y = y + shift_y;
                    
                    if (target_x >= 0 && target_x < width &&
                        target_y >= 0 && target_y < height) {
                        
                        float ref_val = reference[y * width + x];
                        float tar_val = target[target_y * width + target_x];
                        
                        correlation += ref_val * tar_val;
                        count++;
                    }
                }
            }
            
            if (count > 0) {
                correlation /= count;
                
                if (correlation > best_correlation) {
                    best_correlation = correlation;
                    best_shift_x = shift_x;
                    best_shift_y = shift_y;
                }
            }
        }
    }
    
    // 应用最佳平移
    *dx = best_shift_x;
    *dy = best_shift_y;
    
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int src_x = x + best_shift_x;
            int src_y = y + best_shift_y;
            
            if (src_x >= 0 && src_x < width &&
                src_y >= 0 && src_y < height) {
                aligned[y * width + x] = target[src_y * width + src_x];
            } else {
                aligned[y * width + x] = 0.0f;
            }
        }
    }
    
    return AUTO_CALIB_SUCCESS;
}

// ============================================================================
// 图像归一化实现
// ============================================================================

AutoCalibrationError auto_calib_normalize_image_set(
    AutoCalibImageSet *image_set)
{
    AUTO_CALIB_CHECK_NULL(image_set);
    
    if (image_set->num_images == 0) {
        return AUTO_CALIB_ERROR_INSUFFICIENT_DATA;
    }
    
    int num_pixels = image_set->width * image_set->height;
    
    // 计算所有图像的全局统计
    double global_mean = 0.0;
    double global_std = 0.0;
    
    for (int i = 0; i < image_set->num_images; i++) {
        ImageStatistics stats;
        AutoCalibrationError err = compute_image_stats(
            image_set->images[i],
            image_set->width,
            image_set->height,
            &stats);
        
        if (err != AUTO_CALIB_SUCCESS) {
            return err;
        }
        
        global_mean += stats.mean;
        global_std += stats.std_dev;
    }
    
    global_mean /= image_set->num_images;
    global_std /= image_set->num_images;
    
    if (global_std < 1e-10f) {
        return AUTO_CALIB_ERROR_INVALID_PARAM;
    }
    
    // 归一化每个图像
    for (int i = 0; i < image_set->num_images; i++) {
        ImageStatistics stats;
        compute_image_stats(
            image_set->images[i],
            image_set->width,
            image_set->height,
            &stats);
        
        float scale = global_std / (stats.std_dev + 1e-10f);
        float offset = global_mean - stats.mean * scale;
        
        for (int j = 0; j < num_pixels; j++) {
            image_set->images[i][j] = 
                image_set->images[i][j] * scale + offset;
        }
    }
    
    return AUTO_CALIB_SUCCESS;
}

// ============================================================================
// 图像统计计算实现
// ============================================================================

AutoCalibrationError auto_calib_compute_image_statistics(
    const float *image,
    int width,
    int height,
    float *mean,
    float *std_dev,
    float *min_val,
    float *max_val)
{
    AUTO_CALIB_CHECK_NULL(image);
    
    if (width <= 0 || height <= 0) {
        return AUTO_CALIB_ERROR_INVALID_PARAM;
    }
    
    ImageStatistics stats;
    AutoCalibrationError err = compute_image_stats(
        image, width, height, &stats);
    
    if (err != AUTO_CALIB_SUCCESS) {
        return err;
    }
    
    if (mean) *mean = stats.mean;
    if (std_dev) *std_dev = stats.std_dev;
    if (min_val) *min_val = stats.min_val;
    if (max_val) *max_val = stats.max_val;
    
    return AUTO_CALIB_SUCCESS;
}
// ============================================================================
// 自动平场校准实现
// ============================================================================

AutoCalibrationError auto_calib_flat_field(
    AutoCalibrationContext *context,
    const AutoCalibImageSet *flat_images,
    float **flat_field,
    int *width,
    int *height,
    float *quality_score)
{
    AUTO_CALIB_CHECK_NULL(context);
    AUTO_CALIB_CHECK_NULL(flat_images);
    AUTO_CALIB_CHECK_NULL(flat_field);
    
    // 验证图像集合
    bool is_valid;
    char error_msg[256];
    AutoCalibrationError err = auto_calib_validate_image_set(
        flat_images, &is_valid, error_msg, sizeof(error_msg));
    
    if (err != AUTO_CALIB_SUCCESS || !is_valid) {
        return AUTO_CALIB_ERROR_INVALID_IMAGE;
    }
    
    // 检查最小图像数量
    if (flat_images->num_images < context->config.min_flat_images) {
        return AUTO_CALIB_ERROR_INSUFFICIENT_DATA;
    }
    
    // 更新状态
    context->status = AUTO_CALIB_STATUS_PROCESSING;
    context->start_time = time(NULL);
    update_progress(context, "Computing flat field", 0, 100);
    
    // 根据方法选择处理
    float *result = NULL;
    float quality = 0.0f;
    
    switch (context->config.flat_method) {
        case AUTO_CALIB_FLAT_METHOD_MEAN:
            err = compute_flat_field_mean(
                flat_images, context, &result, &quality);
            break;
            
        case AUTO_CALIB_FLAT_METHOD_MEDIAN:
            err = compute_flat_field_median(
                flat_images, context, &result, &quality);
            break;
            
        case AUTO_CALIB_FLAT_METHOD_ROBUST_MEAN:
            err = compute_flat_field_robust_mean(
                flat_images, context, &result, &quality);
            break;
            
        case AUTO_CALIB_FLAT_METHOD_WEIGHTED_MEAN:
            err = compute_flat_field_weighted_mean(
                flat_images, context, &result, &quality);
            break;
            
        case AUTO_CALIB_FLAT_METHOD_ADAPTIVE:
            err = compute_flat_field_adaptive(
                flat_images, context, &result, &quality);
            break;
            
        default:
            return AUTO_CALIB_ERROR_INVALID_PARAM;
    }
    
    if (err != AUTO_CALIB_SUCCESS) {
        AUTO_CALIB_FREE(result);
        return err;
    }
    
    // 应用平滑（如果启用）
    if (context->config.flat_enable_smoothing) {
        update_progress(context, "Smoothing flat field", 80, 100);
        
        float *smoothed = NULL;
        err = smooth_flat_field(
            result,
            flat_images->width,
            flat_images->height,
            context->config.flat_smoothing_sigma,
            &smoothed);
        
        if (err == AUTO_CALIB_SUCCESS) {
            free(result);
            result = smoothed;
        }
    }
    
    // 归一化平场
    update_progress(context, "Normalizing flat field", 90, 100);
    err = normalize_flat_field(
        result,
        flat_images->width,
        flat_images->height);
    
    if (err != AUTO_CALIB_SUCCESS) {
        AUTO_CALIB_FREE(result);
        return err;
    }
    
    // 评估质量
    quality = auto_calib_evaluate_flat_quality(
        result,
        flat_images->width,
        flat_images->height);
    
    // 返回结果
    *flat_field = result;
    if (width) *width = flat_images->width;
    if (height) *height = flat_images->height;
    if (quality_score) *quality_score = quality;
    
    update_progress(context, "Flat field calibration completed", 100, 100);
    context->status = AUTO_CALIB_STATUS_COMPLETED;
    
    return AUTO_CALIB_SUCCESS;
}

/**
 * @brief 使用简单平均计算平场
 */
static AutoCalibrationError compute_flat_field_mean(
    const AutoCalibImageSet *flat_images,
    AutoCalibrationContext *context,
    float **result,
    float *quality)
{
    int num_pixels = flat_images->width * flat_images->height;
    
    float *flat = AUTO_CALIB_CALLOC(float, num_pixels);
    if (!flat) {
        return AUTO_CALIB_ERROR_MEMORY_ALLOCATION;
    }
    
    // 累加所有图像
    for (int i = 0; i < flat_images->num_images; i++) {
        if (context->cancel_requested) {
            free(flat);
            return AUTO_CALIB_ERROR_OPERATION_CANCELLED;
        }
        
        for (int j = 0; j < num_pixels; j++) {
            flat[j] += flat_images->images[i][j];
        }
        
        int progress = (i + 1) * 70 / flat_images->num_images;
        update_progress(context, "Computing mean", progress, 100);
    }
    
    // 计算平均
    float scale = 1.0f / flat_images->num_images;
    for (int i = 0; i < num_pixels; i++) {
        flat[i] *= scale;
    }
    
    *result = flat;
    *quality = 80.0f;  // 基本质量分数
    
    return AUTO_CALIB_SUCCESS;
}

/**
 * @brief 使用中值计算平场
 */
static AutoCalibrationError compute_flat_field_median(
    const AutoCalibImageSet *flat_images,
    AutoCalibrationContext *context,
    float **result,
    float *quality)
{
    int num_pixels = flat_images->width * flat_images->height;
    
    float *flat = AUTO_CALIB_MALLOC(float, num_pixels);
    if (!flat) {
        return AUTO_CALIB_ERROR_MEMORY_ALLOCATION;
    }
    
    float *pixel_values = AUTO_CALIB_MALLOC(float, flat_images->num_images);
    if (!pixel_values) {
        free(flat);
        return AUTO_CALIB_ERROR_MEMORY_ALLOCATION;
    }
    
    // 对每个像素计算中值
    for (int i = 0; i < num_pixels; i++) {
        if (context->cancel_requested) {
            free(flat);
            free(pixel_values);
            return AUTO_CALIB_ERROR_OPERATION_CANCELLED;
        }
        
        // 收集该像素在所有图像中的值
        for (int j = 0; j < flat_images->num_images; j++) {
            pixel_values[j] = flat_images->images[j][i];
        }
        
        // 计算中值
        flat[i] = compute_median(pixel_values, flat_images->num_images);
        
        if (i % 1000 == 0) {
            int progress = i * 70 / num_pixels;
            update_progress(context, "Computing median", progress, 100);
        }
    }
    
    free(pixel_values);
    
    *result = flat;
    *quality = 85.0f;  // 中值方法更鲁棒
    
    return AUTO_CALIB_SUCCESS;
}

/**
 * @brief 使用鲁棒平均计算平场
 */
static AutoCalibrationError compute_flat_field_robust_mean(
    const AutoCalibImageSet *flat_images,
    AutoCalibrationContext *context,
    float **result,
    float *quality)
{
    int num_pixels = flat_images->width * flat_images->height;
    
    float *flat = AUTO_CALIB_MALLOC(float, num_pixels);
    if (!flat) {
        return AUTO_CALIB_ERROR_MEMORY_ALLOCATION;
    }
    
    float *pixel_values = AUTO_CALIB_MALLOC(float, flat_images->num_images);
    if (!pixel_values) {
        free(flat);
        return AUTO_CALIB_ERROR_MEMORY_ALLOCATION;
    }
    
    float threshold = context->config.flat_outlier_threshold;
    
    // 对每个像素计算鲁棒平均
    for (int i = 0; i < num_pixels; i++) {
        if (context->cancel_requested) {
            free(flat);
            free(pixel_values);
            return AUTO_CALIB_ERROR_OPERATION_CANCELLED;
        }
        
        // 收集该像素在所有图像中的值
        for (int j = 0; j < flat_images->num_images; j++) {
            pixel_values[j] = flat_images->images[j][i];
        }
        
        // 计算中值和MAD
        float median = compute_median(pixel_values, flat_images->num_images);
        float mad = compute_mad(pixel_values, flat_images->num_images, median);
        
        // 计算鲁棒平均（排除异常值）
        double sum = 0.0;
        int count = 0;
        
        for (int j = 0; j < flat_images->num_images; j++) {
            float z_score = fabsf(pixel_values[j] - median) / (mad + 1e-10f);
            if (z_score <= threshold) {
                sum += pixel_values[j];
                count++;
            }
        }
        
        flat[i] = (count > 0) ? (sum / count) : median;
        
        if (i % 1000 == 0) {
            int progress = i * 70 / num_pixels;
            update_progress(context, "Computing robust mean", progress, 100);
        }
    }
    
    free(pixel_values);
    
    *result = flat;
    *quality = 90.0f;  // 鲁棒方法质量更高
    
    return AUTO_CALIB_SUCCESS;
}

/**
 * @brief 使用加权平均计算平场
 */
static AutoCalibrationError compute_flat_field_weighted_mean(
    const AutoCalibImageSet *flat_images,
    AutoCalibrationContext *context,
    float **result,
    float *quality)
{
    int num_pixels = flat_images->width * flat_images->height;
    
    // 计算每个图像的权重（基于质量）
    float *weights = AUTO_CALIB_MALLOC(float, flat_images->num_images);
    if (!weights) {
        return AUTO_CALIB_ERROR_MEMORY_ALLOCATION;
    }
    
    float total_weight = 0.0f;
    for (int i = 0; i < flat_images->num_images; i++) {
        ImageStatistics stats;
        compute_image_stats(
            flat_images->images[i],
            flat_images->width,
            flat_images->height,
            &stats);
        
        // 权重与信噪比成正比
        float snr = stats.mean / (stats.std_dev + 1e-10f);
        weights[i] = snr;
        total_weight += snr;
    }
    
    // 归一化权重
    for (int i = 0; i < flat_images->num_images; i++) {
        weights[i] /= total_weight;
    }
    
    // 计算加权平均
    float *flat = AUTO_CALIB_CALLOC(float, num_pixels);
    if (!flat) {
        free(weights);
        return AUTO_CALIB_ERROR_MEMORY_ALLOCATION;
    }
    
    for (int i = 0; i < flat_images->num_images; i++) {
        if (context->cancel_requested) {
            free(flat);
            free(weights);
            return AUTO_CALIB_ERROR_OPERATION_CANCELLED;
        }
        
        float weight = weights[i];
        for (int j = 0; j < num_pixels; j++) {
            flat[j] += flat_images->images[i][j] * weight;
        }
        
        int progress = (i + 1) * 70 / flat_images->num_images;
        update_progress(context, "Computing weighted mean", progress, 100);
    }
    
    free(weights);
    
    *result = flat;
    *quality = 88.0f;
    
    return AUTO_CALIB_SUCCESS;
}

/**
 * @brief 使用自适应方法计算平场
 */
static AutoCalibrationError compute_flat_field_adaptive(
    const AutoCalibImageSet *flat_images,
    AutoCalibrationContext *context,
    float **result,
    float *quality)
{
    int num_pixels = flat_images->width * flat_images->height;
    
    float *flat = AUTO_CALIB_MALLOC(float, num_pixels);
    if (!flat) {
        return AUTO_CALIB_ERROR_MEMORY_ALLOCATION;
    }
    
    float *pixel_values = AUTO_CALIB_MALLOC(float, flat_images->num_images);
    if (!pixel_values) {
        free(flat);
        return AUTO_CALIB_ERROR_MEMORY_ALLOCATION;
    }
    
    // 对每个像素自适应选择方法
    for (int i = 0; i < num_pixels; i++) {
        if (context->cancel_requested) {
            free(flat);
            free(pixel_values);
            return AUTO_CALIB_ERROR_OPERATION_CANCELLED;
        }
        
        // 收集该像素在所有图像中的值
        for (int j = 0; j < flat_images->num_images; j++) {
            pixel_values[j] = flat_images->images[j][i];
        }
        
        // 计算统计量
        float median = compute_median(pixel_values, flat_images->num_images);
        float mad = compute_mad(pixel_values, flat_images->num_images, median);
        
        // 计算变异系数
        double mean = 0.0;
        for (int j = 0; j < flat_images->num_images; j++) {
            mean += pixel_values[j];
        }
        mean /= flat_images->num_images;
        
        double variance = 0.0;
        for (int j = 0; j < flat_images->num_images; j++) {
            double diff = pixel_values[j] - mean;
            variance += diff * diff;
        }
        float std_dev = sqrt(variance / flat_images->num_images);
        float cv = std_dev / (mean + 1e-10f);
        
        // 根据变异系数选择方法
        if (cv < 0.1f) {
            // 低变异：使用简单平均
            flat[i] = mean;
        } else if (cv < 0.3f) {
            // 中等变异：使用鲁棒平均
            double sum = 0.0;
            int count = 0;
            float threshold = 2.5f;
            
            for (int j = 0; j < flat_images->num_images; j++) {
                float z_score = fabsf(pixel_values[j] - median) / (mad + 1e-10f);
                if (z_score <= threshold) {
                    sum += pixel_values[j];
                    count++;
                }
            }
            flat[i] = (count > 0) ? (sum / count) : median;
        } else {
            // 高变异：使用中值
            flat[i] = median;
        }
        
        if (i % 1000 == 0) {
            int progress = i * 70 / num_pixels;
            update_progress(context, "Computing adaptive flat", progress, 100);
        }
    }
    
    free(pixel_values);
    
    *result = flat;
    *quality = 92.0f;  // 自适应方法质量最高
    
    return AUTO_CALIB_SUCCESS;
}

/**
 * @brief 平滑平场
 */
static AutoCalibrationError smooth_flat_field(
    const float *flat_field,
    int width,
    int height,
    float sigma,
    float **smoothed)
{
    if (!flat_field || !smoothed || width <= 0 || height <= 0 || sigma <= 0.0f) {
        return AUTO_CALIB_ERROR_INVALID_PARAM;
    }
    
    int num_pixels = width * height;
    float *result = AUTO_CALIB_MALLOC(float, num_pixels);
    if (!result) {
        return AUTO_CALIB_ERROR_MEMORY_ALLOCATION;
    }
    
    // 计算高斯核大小
    int kernel_size = (int)(6 * sigma + 1);
    if (kernel_size % 2 == 0) kernel_size++;
    int kernel_radius = kernel_size / 2;
    
    // 创建高斯核
    float *kernel = AUTO_CALIB_MALLOC(float, kernel_size);
    if (!kernel) {
        free(result);
        return AUTO_CALIB_ERROR_MEMORY_ALLOCATION;
    }
    
    float sum = 0.0f;
    for (int i = 0; i < kernel_size; i++) {
        int x = i - kernel_radius;
        kernel[i] = expf(-(x * x) / (2 * sigma * sigma));
        sum += kernel[i];
    }
    
    // 归一化核
    for (int i = 0; i < kernel_size; i++) {
        kernel[i] /= sum;
    }
    
    // 分配临时缓冲区
    float *temp = AUTO_CALIB_MALLOC(float, num_pixels);
    if (!temp) {
        free(result);
        free(kernel);
        return AUTO_CALIB_ERROR_MEMORY_ALLOCATION;
    }
    
    // 水平方向卷积
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float value = 0.0f;
            float weight_sum = 0.0f;
            
            for (int k = 0; k < kernel_size; k++) {
                int src_x = x + k - kernel_radius;
                if (src_x >= 0 && src_x < width) {
                    value += flat_field[y * width + src_x] * kernel[k];
                    weight_sum += kernel[k];
                }
            }
            
            temp[y * width + x] = value / weight_sum;
        }
    }
    
    // 垂直方向卷积
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float value = 0.0f;
            float weight_sum = 0.0f;
            
            for (int k = 0; k < kernel_size; k++) {
                int src_y = y + k - kernel_radius;
                if (src_y >= 0 && src_y < height) {
                    value += temp[src_y * width + x] * kernel[k];
                    weight_sum += kernel[k];
                }
            }
            
            result[y * width + x] = value / weight_sum;
        }
    }
    
    free(temp);
    free(kernel);
    
    *smoothed = result;
    
    return AUTO_CALIB_SUCCESS;
}

/**
 * @brief 归一化平场
 */
static AutoCalibrationError normalize_flat_field(
    float *flat_field,
    int width,
    int height)
{
    if (!flat_field || width <= 0 || height <= 0) {
        return AUTO_CALIB_ERROR_INVALID_PARAM;
    }
    
    int num_pixels = width * height;
    
    // 计算平均值
    double sum = 0.0;
    for (int i = 0; i < num_pixels; i++) {
        sum += flat_field[i];
    }
    float mean = sum / num_pixels;
    
    if (mean < 1e-10f) {
        return AUTO_CALIB_ERROR_INVALID_PARAM;
    }
    
    // 归一化到平均值为1
    for (int i = 0; i < num_pixels; i++) {
        flat_field[i] /= mean;
        
        // 限制范围，避免极端值
        if (flat_field[i] < 0.1f) flat_field[i] = 0.1f;
        if (flat_field[i] > 10.0f) flat_field[i] = 10.0f;
    }
    
    return AUTO_CALIB_SUCCESS;
}

AutoCalibrationError auto_calib_flat_field_simple(
    const float **images,
    int num_images,
    int width,
    int height,
    AutoCalibFlatMethod method,
    float **flat_field,
    float *quality_score)
{
    AUTO_CALIB_CHECK_NULL(images);
    AUTO_CALIB_CHECK_NULL(flat_field);
    
    if (num_images < AUTO_CALIB_MIN_IMAGES || width <= 0 || height <= 0) {
        return AUTO_CALIB_ERROR_INVALID_PARAM;
    }
    
    // 创建图像集合
    AutoCalibImageSet *image_set = auto_calib_create_image_set(
        num_images, width, height);
    if (!image_set) {
        return AUTO_CALIB_ERROR_MEMORY_ALLOCATION;
    }
    
    // 添加图像
    for (int i = 0; i < num_images; i++) {
        AutoCalibrationError err = auto_calib_add_image(
            image_set, images[i], 0.0f, 0.0f, 0.0f, NULL);
        if (err != AUTO_CALIB_SUCCESS) {
            auto_calib_destroy_image_set(image_set);
            return err;
        }
    }
    
    // 创建配置
    AutoCalibrationConfig config = auto_calib_create_default_config();
    config.flat_method = method;
    
    // 创建上下文
    AutoCalibrationContext *context = auto_calib_create_context(&config);
    if (!context) {
        auto_calib_destroy_image_set(image_set);
        return AUTO_CALIB_ERROR_MEMORY_ALLOCATION;
    }
    
    // 执行校准
    int result_width, result_height;
    AutoCalibrationError err = auto_calib_flat_field(
        context,
        image_set,
        flat_field,
        &result_width,
        &result_height,
        quality_score);
    
    // 清理
    auto_calib_destroy_context(context);
    auto_calib_destroy_image_set(image_set);
    
    return err;
}

float auto_calib_evaluate_flat_quality(
    const float *flat_field,
    int width,
    int height)
{
    if (!flat_field || width <= 0 || height <= 0) {
        return 0.0f;
    }
    
    int num_pixels = width * height;
    
    // 计算统计量
    ImageStatistics stats;
    if (compute_image_stats(flat_field, width, height, &stats) != 
        AUTO_CALIB_SUCCESS) {
        return 0.0f;
    }
    
    // 质量评分因素
    float quality = 100.0f;
    
    // 1. 均匀性（标准差越小越好）
    float uniformity = 1.0f / (1.0f + stats.std_dev);
    quality *= uniformity;
    
    // 2. 平滑度（检查高频噪声）
    float smoothness = 1.0f;
    int count = 0;
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            int idx = y * width + x;
            float laplacian = fabsf(
                4 * flat_field[idx] -
                flat_field[idx - 1] -
                flat_field[idx + 1] -
                flat_field[idx - width] -
                flat_field[idx + width]);
            smoothness += laplacian;
            count++;
        }
    }
    smoothness = 1.0f / (1.0f + smoothness / count);
    quality *= smoothness;
    
    // 3. 动态范围（不应该有极端值）
    float range = stats.max_val - stats.min_val;
    float range_score = (range < 2.0f) ? 1.0f : (2.0f / range);
    quality *= range_score;
    
    // 限制在0-100范围内
    if (quality < 0.0f) quality = 0.0f;
    if (quality > 100.0f) quality = 100.0f;
    
    return quality;
}
// ============================================================================
// 自动暗场校准实现
// ============================================================================

AutoCalibrationError auto_calib_dark_field(
    AutoCalibrationContext *context,
    const AutoCalibImageSet *dark_images,
    float **dark_field,
    int *width,
    int *height,
    float *quality_score)
{
    AUTO_CALIB_CHECK_NULL(context);
    AUTO_CALIB_CHECK_NULL(dark_images);
    AUTO_CALIB_CHECK_NULL(dark_field);
    
    // 验证图像集合
    bool is_valid;
    char error_msg[256];
    AutoCalibrationError err = auto_calib_validate_image_set(
        dark_images, &is_valid, error_msg, sizeof(error_msg));
    
    if (err != AUTO_CALIB_SUCCESS || !is_valid) {
        return AUTO_CALIB_ERROR_INVALID_IMAGE;
    }
    
    // 检查最小图像数量
    if (dark_images->num_images < context->config.min_dark_images) {
        return AUTO_CALIB_ERROR_INSUFFICIENT_DATA;
    }
    
    // 更新状态
    context->status = AUTO_CALIB_STATUS_PROCESSING;
    context->start_time = time(NULL);
    update_progress(context, "Computing dark field", 0, 100);
    
    // 根据方法选择处理
    float *result = NULL;
    float quality = 0.0f;
    
    switch (context->config.dark_method) {
        case AUTO_CALIB_DARK_METHOD_MEAN:
            err = compute_dark_field_mean(
                dark_images, context, &result, &quality);
            break;
            
        case AUTO_CALIB_DARK_METHOD_MEDIAN:
            err = compute_dark_field_median(
                dark_images, context, &result, &quality);
            break;
            
        case AUTO_CALIB_DARK_METHOD_ROBUST_MEAN:
            err = compute_dark_field_robust_mean(
                dark_images, context, &result, &quality);
            break;
            
        case AUTO_CALIB_DARK_METHOD_TEMPORAL:
            err = compute_dark_field_temporal(
                dark_images, context, &result, &quality);
            break;
            
        default:
            return AUTO_CALIB_ERROR_INVALID_PARAM;
    }
    
    if (err != AUTO_CALIB_SUCCESS) {
        AUTO_CALIB_FREE(result);
        return err;
    }
    
    // 应用时域滤波（如果启用）
    if (context->config.dark_enable_temporal_filter) {
        update_progress(context, "Applying temporal filter", 80, 100);
        
        float *filtered = NULL;
        err = apply_temporal_filter(
            result,
            dark_images->width,
            dark_images->height,
            &filtered);
        
        if (err == AUTO_CALIB_SUCCESS) {
            free(result);
            result = filtered;
        }
    }
    
    // 评估质量
    quality = auto_calib_evaluate_dark_quality(
        result,
        dark_images->width,
        dark_images->height);
    
    // 返回结果
    *dark_field = result;
    if (width) *width = dark_images->width;
    if (height) *height = dark_images->height;
    if (quality_score) *quality_score = quality;
    
    update_progress(context, "Dark field calibration completed", 100, 100);
    context->status = AUTO_CALIB_STATUS_COMPLETED;
    
    return AUTO_CALIB_SUCCESS;
}

/**
 * @brief 使用简单平均计算暗场
 */
static AutoCalibrationError compute_dark_field_mean(
    const AutoCalibImageSet *dark_images,
    AutoCalibrationContext *context,
    float **result,
    float *quality)
{
    int num_pixels = dark_images->width * dark_images->height;
    
    float *dark = AUTO_CALIB_CALLOC(float, num_pixels);
    if (!dark) {
        return AUTO_CALIB_ERROR_MEMORY_ALLOCATION;
    }
    
    // 累加所有图像
    for (int i = 0; i < dark_images->num_images; i++) {
        if (context->cancel_requested) {
            free(dark);
            return AUTO_CALIB_ERROR_OPERATION_CANCELLED;
        }
        
        for (int j = 0; j < num_pixels; j++) {
            dark[j] += dark_images->images[i][j];
        }
        
        int progress = (i + 1) * 70 / dark_images->num_images;
        update_progress(context, "Computing dark mean", progress, 100);
    }
    
    // 计算平均
    float scale = 1.0f / dark_images->num_images;
    for (int i = 0; i < num_pixels; i++) {
        dark[i] *= scale;
    }
    
    *result = dark;
    *quality = 80.0f;
    
    return AUTO_CALIB_SUCCESS;
}

/**
 * @brief 使用中值计算暗场
 */
static AutoCalibrationError compute_dark_field_median(
    const AutoCalibImageSet *dark_images,
    AutoCalibrationContext *context,
    float **result,
    float *quality)
{
    int num_pixels = dark_images->width * dark_images->height;
    
    float *dark = AUTO_CALIB_MALLOC(float, num_pixels);
    if (!dark) {
        return AUTO_CALIB_ERROR_MEMORY_ALLOCATION;
    }
    
    float *pixel_values = AUTO_CALIB_MALLOC(float, dark_images->num_images);
    if (!pixel_values) {
        free(dark);
        return AUTO_CALIB_ERROR_MEMORY_ALLOCATION;
    }
    
    // 对每个像素计算中值
    for (int i = 0; i < num_pixels; i++) {
        if (context->cancel_requested) {
            free(dark);
            free(pixel_values);
            return AUTO_CALIB_ERROR_OPERATION_CANCELLED;
        }
        
        // 收集该像素在所有图像中的值
        for (int j = 0; j < dark_images->num_images; j++) {
            pixel_values[j] = dark_images->images[j][i];
        }
        
        // 计算中值
        dark[i] = compute_median(pixel_values, dark_images->num_images);
        
        if (i % 1000 == 0) {
            int progress = i * 70 / num_pixels;
            update_progress(context, "Computing dark median", progress, 100);
        }
    }
    
    free(pixel_values);
    
    *result = dark;
    *quality = 85.0f;
    
    return AUTO_CALIB_SUCCESS;
}

/**
 * @brief 使用鲁棒平均计算暗场
 */
static AutoCalibrationError compute_dark_field_robust_mean(
    const AutoCalibImageSet *dark_images,
    AutoCalibrationContext *context,
    float **result,
    float *quality)
{
    int num_pixels = dark_images->width * dark_images->height;
    
    float *dark = AUTO_CALIB_MALLOC(float, num_pixels);
    if (!dark) {
        return AUTO_CALIB_ERROR_MEMORY_ALLOCATION;
    }
    
    float *pixel_values = AUTO_CALIB_MALLOC(float, dark_images->num_images);
    if (!pixel_values) {
        free(dark);
        return AUTO_CALIB_ERROR_MEMORY_ALLOCATION;
    }
    
    float threshold = context->config.dark_outlier_threshold;
    
    // 对每个像素计算鲁棒平均
    for (int i = 0; i < num_pixels; i++) {
        if (context->cancel_requested) {
            free(dark);
            free(pixel_values);
            return AUTO_CALIB_ERROR_OPERATION_CANCELLED;
        }
        
        // 收集该像素在所有图像中的值
        for (int j = 0; j < dark_images->num_images; j++) {
            pixel_values[j] = dark_images->images[j][i];
        }
        
        // 计算中值和MAD
        float median = compute_median(pixel_values, dark_images->num_images);
        float mad = compute_mad(pixel_values, dark_images->num_images, median);
        
        // 计算鲁棒平均（排除异常值）
        double sum = 0.0;
        int count = 0;
        
        for (int j = 0; j < dark_images->num_images; j++) {
            float z_score = fabsf(pixel_values[j] - median) / (mad + 1e-10f);
            if (z_score <= threshold) {
                sum += pixel_values[j];
                count++;
            }
        }
        
        dark[i] = (count > 0) ? (sum / count) : median;
        
        if (i % 1000 == 0) {
            int progress = i * 70 / num_pixels;
            update_progress(context, "Computing robust dark", progress, 100);
        }
    }
    
    free(pixel_values);
    
    *result = dark;
    *quality = 90.0f;
    
    return AUTO_CALIB_SUCCESS;
}

/**
 * @brief 使用时域方法计算暗场（考虑温度和曝光时间）
 */
static AutoCalibrationError compute_dark_field_temporal(
    const AutoCalibImageSet *dark_images,
    AutoCalibrationContext *context,
    float **result,
    float *quality)
{
    int num_pixels = dark_images->width * dark_images->height;
    
    // 首先按温度和曝光时间分组
    // 这里简化处理，假设所有图像参数相似
    
    float *dark = AUTO_CALIB_MALLOC(float, num_pixels);
    if (!dark) {
        return AUTO_CALIB_ERROR_MEMORY_ALLOCATION;
    }
    
    float *pixel_values = AUTO_CALIB_MALLOC(float, dark_images->num_images);
    float *weights = AUTO_CALIB_MALLOC(float, dark_images->num_images);
    
    if (!pixel_values || !weights) {
        AUTO_CALIB_FREE(dark);
        AUTO_CALIB_FREE(pixel_values);
        AUTO_CALIB_FREE(weights);
        return AUTO_CALIB_ERROR_MEMORY_ALLOCATION;
    }
    
    // 计算每个图像的权重（基于温度和曝光时间的一致性）
    float avg_temp = 0.0f;
    float avg_exposure = 0.0f;
    
    for (int i = 0; i < dark_images->num_images; i++) {
        avg_temp += dark_images->temperatures[i];
        avg_exposure += dark_images->exposure_times[i];
    }
    avg_temp /= dark_images->num_images;
    avg_exposure /= dark_images->num_images;
    
    float total_weight = 0.0f;
    for (int i = 0; i < dark_images->num_images; i++) {
        // 权重与参数一致性成反比
        float temp_diff = fabsf(dark_images->temperatures[i] - avg_temp);
        float exp_diff = fabsf(dark_images->exposure_times[i] - avg_exposure);
        
        weights[i] = 1.0f / (1.0f + temp_diff + exp_diff);
        total_weight += weights[i];
    }
    
    // 归一化权重
    for (int i = 0; i < dark_images->num_images; i++) {
        weights[i] /= total_weight;
    }
    
    // 对每个像素计算加权时域平均
    for (int i = 0; i < num_pixels; i++) {
        if (context->cancel_requested) {
            free(dark);
            free(pixel_values);
            free(weights);
            return AUTO_CALIB_ERROR_OPERATION_CANCELLED;
        }
        
        // 收集该像素在所有图像中的值
        for (int j = 0; j < dark_images->num_images; j++) {
            pixel_values[j] = dark_images->images[j][i];
        }
        
        // 计算加权平均
        double weighted_sum = 0.0;
        for (int j = 0; j < dark_images->num_images; j++) {
            weighted_sum += pixel_values[j] * weights[j];
        }
        
        dark[i] = weighted_sum;
        
        if (i % 1000 == 0) {
            int progress = i * 70 / num_pixels;
            update_progress(context, "Computing temporal dark", progress, 100);
        }
    }
    
    free(pixel_values);
    free(weights);
    
    *result = dark;
    *quality = 92.0f;
    
    return AUTO_CALIB_SUCCESS;
}

/**
 * @brief 应用时域滤波
 */
static AutoCalibrationError apply_temporal_filter(
    const float *dark_field,
    int width,
    int height,
    float **filtered)
{
    if (!dark_field || !filtered || width <= 0 || height <= 0) {
        return AUTO_CALIB_ERROR_INVALID_PARAM;
    }
    
    int num_pixels = width * height;
    float *result = AUTO_CALIB_MALLOC(float, num_pixels);
    if (!result) {
        return AUTO_CALIB_ERROR_MEMORY_ALLOCATION;
    }
    
    // 简单的中值滤波（3x3窗口）
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float values[9];
            int count = 0;
            
            // 收集邻域值
            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    int nx = x + dx;
                    int ny = y + dy;
                    
                    if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                        values[count++] = dark_field[ny * width + nx];
                    }
                }
            }
            
            // 计算中值
            result[y * width + x] = compute_median(values, count);
        }
    }
    
    *filtered = result;
    
    return AUTO_CALIB_SUCCESS;
}

AutoCalibrationError auto_calib_dark_field_simple(
    const float **images,
    int num_images,
    int width,
    int height,
    AutoCalibDarkMethod method,
    float **dark_field,
    float *quality_score)
{
    AUTO_CALIB_CHECK_NULL(images);
    AUTO_CALIB_CHECK_NULL(dark_field);
    
    if (num_images < AUTO_CALIB_MIN_IMAGES || width <= 0 || height <= 0) {
        return AUTO_CALIB_ERROR_INVALID_PARAM;
    }
    
    // 创建图像集合
    AutoCalibImageSet *image_set = auto_calib_create_image_set(
        num_images, width, height);
    if (!image_set) {
        return AUTO_CALIB_ERROR_MEMORY_ALLOCATION;
    }
    
    // 添加图像
    for (int i = 0; i < num_images; i++) {
        AutoCalibrationError err = auto_calib_add_image(
            image_set, images[i], 0.0f, 0.0f, 0.0f, NULL);
        if (err != AUTO_CALIB_SUCCESS) {
            auto_calib_destroy_image_set(image_set);
            return err;
        }
    }
    
    // 创建配置
    AutoCalibrationConfig config = auto_calib_create_default_config();
    config.dark_method = method;
    
    // 创建上下文
    AutoCalibrationContext *context = auto_calib_create_context(&config);
    if (!context) {
        auto_calib_destroy_image_set(image_set);
        return AUTO_CALIB_ERROR_MEMORY_ALLOCATION;
    }
    
    // 执行校准
    int result_width, result_height;
    AutoCalibrationError err = auto_calib_dark_field(
        context,
        image_set,
        dark_field,
        &result_width,
        &result_height,
        quality_score);
    
    // 清理
    auto_calib_destroy_context(context);
    auto_calib_destroy_image_set(image_set);
    
    return err;
}

float auto_calib_evaluate_dark_quality(
    const float *dark_field,
    int width,
    int height)
{
    if (!dark_field || width <= 0 || height <= 0) {
        return 0.0f;
    }
    
    int num_pixels = width * height;
    
    // 计算统计量
    ImageStatistics stats;
    if (compute_image_stats(dark_field, width, height, &stats) != 
        AUTO_CALIB_SUCCESS) {
        return 0.0f;
    }
    
    // 质量评分因素
    float quality = 100.0f;
    
    // 1. 低噪声（标准差应该很小）
    float noise_score = 1.0f / (1.0f + stats.std_dev);
    quality *= noise_score;
    
    // 2. 低均值（暗场应该接近零）
    float mean_score = 1.0f / (1.0f + fabsf(stats.mean));
    quality *= mean_score;
    
    // 3. 平滑度
    float smoothness = 1.0f;
    int count = 0;
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            int idx = y * width + x;
            float gradient = fabsf(
                dark_field[idx] - dark_field[idx - 1]) +
                fabsf(dark_field[idx] - dark_field[idx + 1]) +
                fabsf(dark_field[idx] - dark_field[idx - width]) +
                fabsf(dark_field[idx] - dark_field[idx + width]);
            smoothness += gradient;
            count++;
        }
    }
    smoothness = 1.0f / (1.0f + smoothness / count);
    quality *= smoothness;
    
    // 4. 无热像素（检查极端值）
    int hot_pixels = 0;
    float threshold = stats.median + 5.0f * stats.mad;
    for (int i = 0; i < num_pixels; i++) {
        if (dark_field[i] > threshold) {
            hot_pixels++;
        }
    }
    float hot_pixel_ratio = (float)hot_pixels / num_pixels;
    float hot_pixel_score = 1.0f - hot_pixel_ratio;
    quality *= hot_pixel_score;
    
    // 限制在0-100范围内
    if (quality < 0.0f) quality = 0.0f;
    if (quality > 100.0f) quality = 100.0f;
    
    return quality;
}

// ============================================================================
// 应用校准实现
// ============================================================================

AutoCalibrationError auto_calib_apply_calibration(
    const float *raw_image,
    int width,
    int height,
    const float *dark_field,
    const float *flat_field,
    float **calibrated_image)
{
    AUTO_CALIB_CHECK_NULL(raw_image);
    AUTO_CALIB_CHECK_NULL(calibrated_image);
    
    if (width <= 0 || height <= 0) {
        return AUTO_CALIB_ERROR_INVALID_PARAM;
    }
    
    int num_pixels = width * height;
    
    float *result = AUTO_CALIB_MALLOC(float, num_pixels);
    if (!result) {
        return AUTO_CALIB_ERROR_MEMORY_ALLOCATION;
    }
    
    // 应用校准：(raw - dark) / flat
    for (int i = 0; i < num_pixels; i++) {
        float value = raw_image[i];
        
        // 减去暗场
        if (dark_field) {
            value -= dark_field[i];
        }
        
        // 除以平场
        if (flat_field) {
            float flat_value = flat_field[i];
            if (flat_value > 0.01f) {  // 避免除以零
                value /= flat_value;
            }
        }
        
        result[i] = value;
    }
    
    *calibrated_image = result;
    
    return AUTO_CALIB_SUCCESS;
}

AutoCalibrationError auto_calib_apply_calibration_batch(
    const float **raw_images,
    int num_images,
    int width,
    int height,
    const float *dark_field,
    const float *flat_field,
    float **calibrated_images)
{
    AUTO_CALIB_CHECK_NULL(raw_images);
    AUTO_CALIB_CHECK_NULL(calibrated_images);
    
    if (num_images <= 0 || width <= 0 || height <= 0) {
        return AUTO_CALIB_ERROR_INVALID_PARAM;
    }
    
    // 处理每个图像
    #pragma omp parallel for if(num_images > 4)
    for (int i = 0; i < num_images; i++) {
        AutoCalibrationError err = auto_calib_apply_calibration(
            raw_images[i],
            width,
            height,
            dark_field,
            flat_field,
            &calibrated_images[i]);
        
        if (err != AUTO_CALIB_SUCCESS) {
            // 错误处理
            calibrated_images[i] = NULL;
        }
    }
    
    return AUTO_CALIB_SUCCESS;
}

AutoCalibrationError auto_calib_estimate_noise_level(
    const float *image,
    int width,
    int height,
    float *noise_std)
{
    AUTO_CALIB_CHECK_NULL(image);
    AUTO_CALIB_CHECK_NULL(noise_std);
    
    if (width <= 0 || height <= 0) {
        return AUTO_CALIB_ERROR_INVALID_PARAM;
    }
    
    // 使用MAD估计噪声（鲁棒方法）
    int num_pixels = width * height;
    
    // 计算差分图像（估计高频噪声）
    float *diff = AUTO_CALIB_MALLOC(float, num_pixels);
    if (!diff) {
        return AUTO_CALIB_ERROR_MEMORY_ALLOCATION;
    }
    
    int count = 0;
    for (int y = 0; y < height - 1; y++) {
        for (int x = 0; x < width - 1; x++) {
            int idx = y * width + x;
            // 使用对角差分
            diff[count++] = image[idx] - image[idx + width + 1];
        }
    }
    
    // 计算MAD
    float median = compute_median(diff, count);
    float mad = compute_mad(diff, count, median);
    
    // MAD到标准差的转换（假设高斯噪声）
    *noise_std = 1.4826f * mad;
    
    free(diff);
    
    return AUTO_CALIB_SUCCESS;
}

AutoCalibrationError auto_calib_compute_snr(
    const float *image,
    int width,
    int height,
    float *snr)
{
    AUTO_CALIB_CHECK_NULL(image);
    AUTO_CALIB_CHECK_NULL(snr);
    
    if (width <= 0 || height <= 0) {
        return AUTO_CALIB_ERROR_INVALID_PARAM;
    }
    
    // 计算信号和噪声
    ImageStatistics stats;
    AutoCalibrationError err = compute_image_stats(
        image, width, height, &stats);
    
    if (err != AUTO_CALIB_SUCCESS) {
        return err;
    }
    
    float noise_level;
    err = auto_calib_estimate_noise_level(
        image, width, height, &noise_level);
    
    if (err != AUTO_CALIB_SUCCESS) {
        return err;
    }
    
    // SNR = 信号均值 / 噪声标准差
    if (noise_level > 1e-10f) {
        *snr = stats.mean / noise_level;
    } else {
        *snr = FLT_MAX;
    }
    
    return AUTO_CALIB_SUCCESS;
}
// ============================================================================
// PSF估计实现
// ============================================================================

AutoCalibrationError auto_calib_estimate_psf(
    AutoCalibrationContext *context,
    const float *image,
    int width,
    int height,
    int psf_size,
    float **psf,
    float *quality_score)
{
    AUTO_CALIB_CHECK_NULL(context);
    AUTO_CALIB_CHECK_NULL(image);
    AUTO_CALIB_CHECK_NULL(psf);
    
    if (width <= 0 || height <= 0 || psf_size <= 0 || psf_size % 2 == 0) {
        return AUTO_CALIB_ERROR_INVALID_PARAM;
    }
    
    // 更新状态
    context->status = AUTO_CALIB_STATUS_PROCESSING;
    update_progress(context, "Estimating PSF", 0, 100);
    
    // 根据方法选择处理
    float *result = NULL;
    float quality = 0.0f;
    AutoCalibrationError err;
    
    switch (context->config.psf_method) {
        case AUTO_CALIB_PSF_METHOD_BLIND:
            err = estimate_psf_blind(
                image, width, height, psf_size, context, &result, &quality);
            break;
            
        case AUTO_CALIB_PSF_METHOD_STAR:
            err = estimate_psf_from_stars(
                image, width, height, psf_size, context, &result, &quality);
            break;
            
        case AUTO_CALIB_PSF_METHOD_EDGE:
            err = estimate_psf_from_edges(
                image, width, height, psf_size, context, &result, &quality);
            break;
            
        case AUTO_CALIB_PSF_METHOD_PARAMETRIC:
            err = estimate_psf_parametric(
                image, width, height, psf_size, context, &result, &quality);
            break;
            
        default:
            return AUTO_CALIB_ERROR_INVALID_PARAM;
    }
    
    if (err != AUTO_CALIB_SUCCESS) {
        AUTO_CALIB_FREE(result);
        return err;
    }
    
    // 归一化PSF
    update_progress(context, "Normalizing PSF", 90, 100);
    err = normalize_psf(result, psf_size);
    if (err != AUTO_CALIB_SUCCESS) {
        free(result);
        return err;
    }
    
    // 评估质量
    quality = auto_calib_evaluate_psf_quality(result, psf_size);
    
    // 返回结果
    *psf = result;
    if (quality_score) *quality_score = quality;
    
    update_progress(context, "PSF estimation completed", 100, 100);
    context->status = AUTO_CALIB_STATUS_COMPLETED;
    
    return AUTO_CALIB_SUCCESS;
}

/**
 * @brief 盲PSF估计
 */
static AutoCalibrationError estimate_psf_blind(
    const float *image,
    int width,
    int height,
    int psf_size,
    AutoCalibrationContext *context,
    float **psf,
    float *quality)
{
    // 使用Richardson-Lucy盲反卷积
    int num_pixels = width * height;
    int psf_pixels = psf_size * psf_size;
    
    // 初始化PSF为高斯
    float *current_psf = AUTO_CALIB_MALLOC(float, psf_pixels);
    if (!current_psf) {
        return AUTO_CALIB_ERROR_MEMORY_ALLOCATION;
    }
    
    float sigma = psf_size / 6.0f;
    initialize_gaussian_psf(current_psf, psf_size, sigma);
    
    // 分配工作缓冲区
    float *estimated_image = AUTO_CALIB_MALLOC(float, num_pixels);
    float *error_image = AUTO_CALIB_MALLOC(float, num_pixels);
    float *psf_update = AUTO_CALIB_MALLOC(float, psf_pixels);
    
    if (!estimated_image || !error_image || !psf_update) {
        AUTO_CALIB_FREE(current_psf);
        AUTO_CALIB_FREE(estimated_image);
        AUTO_CALIB_FREE(error_image);
        AUTO_CALIB_FREE(psf_update);
        return AUTO_CALIB_ERROR_MEMORY_ALLOCATION;
    }
    
    // 迭代优化
    int max_iterations = context->config.psf_max_iterations;
    float convergence_threshold = 1e-4f;
    
    for (int iter = 0; iter < max_iterations; iter++) {
        if (context->cancel_requested) {
            AUTO_CALIB_FREE(current_psf);
            AUTO_CALIB_FREE(estimated_image);
            AUTO_CALIB_FREE(error_image);
            AUTO_CALIB_FREE(psf_update);
            return AUTO_CALIB_ERROR_OPERATION_CANCELLED;
        }
        
        // 1. 用当前PSF卷积得到估计图像
        AutoCalibrationError err = convolve_2d(
            image, width, height,
            current_psf, psf_size,
            estimated_image);
        
        if (err != AUTO_CALIB_SUCCESS) {
            AUTO_CALIB_FREE(current_psf);
            AUTO_CALIB_FREE(estimated_image);
            AUTO_CALIB_FREE(error_image);
            AUTO_CALIB_FREE(psf_update);
            return err;
        }
        
        // 2. 计算误差
        float total_error = 0.0f;
        for (int i = 0; i < num_pixels; i++) {
            error_image[i] = image[i] / (estimated_image[i] + 1e-10f);
            total_error += fabsf(image[i] - estimated_image[i]);
        }
        total_error /= num_pixels;
        
        // 3. 更新PSF
        err = update_psf_blind(
            error_image, width, height,
            current_psf, psf_size,
            psf_update);
        
        if (err != AUTO_CALIB_SUCCESS) {
            AUTO_CALIB_FREE(current_psf);
            AUTO_CALIB_FREE(estimated_image);
            AUTO_CALIB_FREE(error_image);
            AUTO_CALIB_FREE(psf_update);
            return err;
        }
        
        // 4. 应用更新
        for (int i = 0; i < psf_pixels; i++) {
            current_psf[i] *= psf_update[i];
        }
        
        // 5. 归一化
        normalize_psf(current_psf, psf_size);
        
        // 6. 检查收敛
        if (total_error < convergence_threshold) {
            break;
        }
        
        int progress = 10 + (iter * 80) / max_iterations;
        update_progress(context, "Blind PSF estimation", progress, 100);
    }
    
    AUTO_CALIB_FREE(estimated_image);
    AUTO_CALIB_FREE(error_image);
    AUTO_CALIB_FREE(psf_update);
    
    *psf = current_psf;
    *quality = 75.0f;
    
    return AUTO_CALIB_SUCCESS;
}

/**
 * @brief 从星点估计PSF
 */
static AutoCalibrationError estimate_psf_from_stars(
    const float *image,
    int width,
    int height,
    int psf_size,
    AutoCalibrationContext *context,
    float **psf,
    float *quality)
{
    // 1. 检测星点
    update_progress(context, "Detecting stars", 10, 100);
    
    StarPoint *stars = NULL;
    int num_stars = 0;
    
    AutoCalibrationError err = detect_stars(
        image, width, height,
        context->config.psf_star_threshold,
        &stars, &num_stars);
    
    if (err != AUTO_CALIB_SUCCESS || num_stars < 3) {
        AUTO_CALIB_FREE(stars);
        return AUTO_CALIB_ERROR_INSUFFICIENT_DATA;
    }
    
    update_progress(context, "Extracting PSF from stars", 30, 100);
    
    // 2. 从每个星点提取PSF
    int psf_pixels = psf_size * psf_size;
    float *accumulated_psf = AUTO_CALIB_CALLOC(float, psf_pixels);
    if (!accumulated_psf) {
        free(stars);
        return AUTO_CALIB_ERROR_MEMORY_ALLOCATION;
    }
    
    int valid_stars = 0;
    int psf_radius = psf_size / 2;
    
    for (int i = 0; i < num_stars; i++) {
        if (context->cancel_requested) {
            free(accumulated_psf);
            free(stars);
            return AUTO_CALIB_ERROR_OPERATION_CANCELLED;
        }
        
        int cx = (int)(stars[i].x + 0.5f);
        int cy = (int)(stars[i].y + 0.5f);
        
        // 检查边界
        if (cx < psf_radius || cx >= width - psf_radius ||
            cy < psf_radius || cy >= height - psf_radius) {
            continue;
        }
        
        // 提取PSF窗口
        float *star_psf = AUTO_CALIB_MALLOC(float, psf_pixels);
        if (!star_psf) continue;
        
        for (int py = 0; py < psf_size; py++) {
            for (int px = 0; px < psf_size; px++) {
                int img_x = cx - psf_radius + px;
                int img_y = cy - psf_radius + py;
                star_psf[py * psf_size + px] = 
                    image[img_y * width + img_x];
            }
        }
        
        // 归一化并累加
        normalize_psf(star_psf, psf_size);
        for (int j = 0; j < psf_pixels; j++) {
            accumulated_psf[j] += star_psf[j];
        }
        
        free(star_psf);
        valid_stars++;
        
        int progress = 30 + (i * 60) / num_stars;
        update_progress(context, "Processing stars", progress, 100);
    }
    
    free(stars);
    
    if (valid_stars == 0) {
        free(accumulated_psf);
        return AUTO_CALIB_ERROR_INSUFFICIENT_DATA;
    }
    
    // 3. 平均PSF
    for (int i = 0; i < psf_pixels; i++) {
        accumulated_psf[i] /= valid_stars;
    }
    
    *psf = accumulated_psf;
    *quality = 85.0f;
    
    return AUTO_CALIB_SUCCESS;
}

/**
 * @brief 从边缘估计PSF
 */
static AutoCalibrationError estimate_psf_from_edges(
    const float *image,
    int width,
    int height,
    int psf_size,
    AutoCalibrationContext *context,
    float **psf,
    float *quality)
{
    // 1. 检测边缘
    update_progress(context, "Detecting edges", 10, 100);
    
    float *edge_map = AUTO_CALIB_MALLOC(float, width * height);
    if (!edge_map) {
        return AUTO_CALIB_ERROR_MEMORY_ALLOCATION;
    }
    
    AutoCalibrationError err = detect_edges(
        image, width, height, edge_map);
    
    if (err != AUTO_CALIB_SUCCESS) {
        free(edge_map);
        return err;
    }
    
    update_progress(context, "Analyzing edge profiles", 30, 100);
    
    // 2. 分析边缘轮廓
    int psf_pixels = psf_size * psf_size;
    float *accumulated_psf = AUTO_CALIB_CALLOC(float, psf_pixels);
    if (!accumulated_psf) {
        free(edge_map);
        return AUTO_CALIB_ERROR_MEMORY_ALLOCATION;
    }
    
    int valid_edges = 0;
    int psf_radius = psf_size / 2;
    
    // 扫描图像寻找强边缘
    for (int y = psf_radius; y < height - psf_radius; y++) {
        for (int x = psf_radius; x < width - psf_radius; x++) {
            if (context->cancel_requested) {
                free(accumulated_psf);
                free(edge_map);
                return AUTO_CALIB_ERROR_OPERATION_CANCELLED;
            }
            
            int idx = y * width + x;
            
            // 检查是否为强边缘
            if (edge_map[idx] < context->config.psf_edge_threshold) {
                continue;
            }
            
            // 提取边缘轮廓的导数作为PSF估计
            float *edge_psf = AUTO_CALIB_MALLOC(float, psf_pixels);
            if (!edge_psf) continue;
            
            for (int py = 0; py < psf_size; py++) {
                for (int px = 0; px < psf_size; px++) {
                    int img_x = x - psf_radius + px;
                    int img_y = y - psf_radius + py;
                    
                    // 计算梯度
                    float gx = 0.0f, gy = 0.0f;
                    if (img_x > 0 && img_x < width - 1) {
                        gx = image[img_y * width + img_x + 1] - 
                             image[img_y * width + img_x - 1];
                    }
                    if (img_y > 0 && img_y < height - 1) {
                        gy = image[(img_y + 1) * width + img_x] - 
                             image[(img_y - 1) * width + img_x];
                    }
                    
                    edge_psf[py * psf_size + px] = sqrtf(gx*gx + gy*gy);
                }
            }
            
            // 归一化并累加
            normalize_psf(edge_psf, psf_size);
            for (int j = 0; j < psf_pixels; j++) {
                accumulated_psf[j] += edge_psf[j];
            }
            
            free(edge_psf);
            valid_edges++;
            
            if (valid_edges >= 100) break;  // 限制边缘数量
        }
        if (valid_edges >= 100) break;
        
        int progress = 30 + (y * 60) / height;
        update_progress(context, "Processing edges", progress, 100);
    }
    
    free(edge_map);
    
    if (valid_edges == 0) {
        free(accumulated_psf);
        return AUTO_CALIB_ERROR_INSUFFICIENT_DATA;
    }
    
    // 3. 平均PSF
    for (int i = 0; i < psf_pixels; i++) {
        accumulated_psf[i] /= valid_edges;
    }
    
    *psf = accumulated_psf;
    *quality = 80.0f;
    
    return AUTO_CALIB_SUCCESS;
}

/**
 * @brief 参数化PSF估计
 */
static AutoCalibrationError estimate_psf_parametric(
    const float *image,
    int width,
    int height,
    int psf_size,
    AutoCalibrationContext *context,
    float **psf,
    float *quality)
{
    // 使用高斯模型拟合PSF
    update_progress(context, "Fitting parametric PSF", 10, 100);
    
    // 1. 检测星点或特征点
    StarPoint *stars = NULL;
    int num_stars = 0;
    
    AutoCalibrationError err = detect_stars(
        image, width, height,
        context->config.psf_star_threshold,
        &stars, &num_stars);
    
    if (err != AUTO_CALIB_SUCCESS || num_stars < 3) {
        AUTO_CALIB_FREE(stars);
        return AUTO_CALIB_ERROR_INSUFFICIENT_DATA;
    }
    
    update_progress(context, "Estimating PSF parameters", 30, 100);
    
    // 2. 对每个星点拟合高斯
    double sum_sigma_x = 0.0;
    double sum_sigma_y = 0.0;
    double sum_theta = 0.0;
    int valid_fits = 0;
    
    for (int i = 0; i < num_stars; i++) {
        if (context->cancel_requested) {
            free(stars);
            return AUTO_CALIB_ERROR_OPERATION_CANCELLED;
        }
        
        GaussianParams params;
        err = fit_gaussian_2d(
            image, width, height,
            stars[i].x, stars[i].y,
            psf_size,
            &params);
        
        if (err == AUTO_CALIB_SUCCESS) {
            sum_sigma_x += params.sigma_x;
            sum_sigma_y += params.sigma_y;
            sum_theta += params.theta;
            valid_fits++;
        }
        
        int progress = 30 + (i * 60) / num_stars;
        update_progress(context, "Fitting Gaussians", progress, 100);
    }
    
    free(stars);
    
    if (valid_fits == 0) {
        return AUTO_CALIB_ERROR_INSUFFICIENT_DATA;
    }
    
    // 3. 平均参数
    GaussianParams avg_params;
    avg_params.sigma_x = sum_sigma_x / valid_fits;
    avg_params.sigma_y = sum_sigma_y / valid_fits;
    avg_params.theta = sum_theta / valid_fits;
    avg_params.amplitude = 1.0f;
    avg_params.x0 = psf_size / 2.0f;
    avg_params.y0 = psf_size / 2.0f;
    
    // 4. 生成PSF
    int psf_pixels = psf_size * psf_size;
    float *result = AUTO_CALIB_MALLOC(float, psf_pixels);
    if (!result) {
        return AUTO_CALIB_ERROR_MEMORY_ALLOCATION;
    }
    
    generate_gaussian_psf(result, psf_size, &avg_params);
    normalize_psf(result, psf_size);
    
    *psf = result;
    *quality = 88.0f;
    
    return AUTO_CALIB_SUCCESS;
}

/**
 * @brief 初始化高斯PSF
 */
static void initialize_gaussian_psf(float *psf, int size, float sigma) {
    int center = size / 2;
    float sum = 0.0f;
    
    for (int y = 0; y < size; y++) {
        for (int x = 0; x < size; x++) {
            float dx = x - center;
            float dy = y - center;
            float r2 = dx*dx + dy*dy;
            float value = expf(-r2 / (2.0f * sigma * sigma));
            psf[y * size + x] = value;
            sum += value;
        }
    }
    
    // 归一化
    for (int i = 0; i < size * size; i++) {
        psf[i] /= sum;
    }
}

/**
 * @brief 生成高斯PSF
 */
static void generate_gaussian_psf(
    float *psf,
    int size,
    const GaussianParams *params)
{
    float cos_theta = cosf(params->theta);
    float sin_theta = sinf(params->theta);
    
    for (int y = 0; y < size; y++) {
        for (int x = 0; x < size; x++) {
            float dx = x - params->x0;
            float dy = y - params->y0;
            
            // 旋转坐标
            float xr = dx * cos_theta + dy * sin_theta;
            float yr = -dx * sin_theta + dy * cos_theta;
            
            // 计算高斯值
            float value = params->amplitude * expf(
                -(xr*xr / (2.0f * params->sigma_x * params->sigma_x) +
                  yr*yr / (2.0f * params->sigma_y * params->sigma_y)));
            
            psf[y * size + x] = value;
        }
    }
}

/**
 * @brief 归一化PSF
 */
static AutoCalibrationError normalize_psf(float *psf, int size) {
    if (!psf || size <= 0) {
        return AUTO_CALIB_ERROR_INVALID_PARAM;
    }
    
    int num_pixels = size * size;
    
    // 计算总和
    double sum = 0.0;
    for (int i = 0; i < num_pixels; i++) {
        sum += psf[i];
    }
    
    if (sum < 1e-10) {
        return AUTO_CALIB_ERROR_INVALID_PARAM;
    }
    
    // 归一化
    float scale = 1.0f / sum;
    for (int i = 0; i < num_pixels; i++) {
        psf[i] *= scale;
    }
    
    return AUTO_CALIB_SUCCESS;
}

/**
 * @brief 检测星点
 */
static AutoCalibrationError detect_stars(
    const float *image,
    int width,
    int height,
    float threshold,
    StarPoint **stars,
    int *num_stars)
{
    if (!image || !stars || !num_stars) {
        return AUTO_CALIB_ERROR_NULL_POINTER;
    }
    
    // 分配临时存储
    int max_stars = 1000;
    StarPoint *detected = AUTO_CALIB_MALLOC(StarPoint, max_stars);
    if (!detected) {
        return AUTO_CALIB_ERROR_MEMORY_ALLOCATION;
    }
    
    int count = 0;
    
    // 简单的局部最大值检测
    int window = 5;
    for (int y = window; y < height - window; y++) {
        for (int x = window; x < width - window; x++) {
            float center = image[y * width + x];
            
            if (center < threshold) continue;
            
            // 检查是否为局部最大值
            bool is_maximum = true;
            for (int dy = -window; dy <= window && is_maximum; dy++) {
                for (int dx = -window; dx <= window; dx++) {
                    if (dx == 0 && dy == 0) continue;
                    
                    float neighbor = image[(y + dy) * width + (x + dx)];
                    if (neighbor >= center) {
                        is_maximum = false;
                        break;
                    }
                }
            }
            
            if (is_maximum && count < max_stars) {
                detected[count].x = x;
                detected[count].y = y;
                detected[count].intensity = center;
                count++;
            }
        }
    }
    
    *stars = detected;
    *num_stars = count;
    
    return AUTO_CALIB_SUCCESS;
}

/**
 * @brief 检测边缘
 */
static AutoCalibrationError detect_edges(
    const float *image,
    int width,
    int height,
    float *edge_map)
{
    if (!image || !edge_map) {
        return AUTO_CALIB_ERROR_NULL_POINTER;
    }
    
    // 使用Sobel算子
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            // Sobel X
            float gx = 
                -image[(y-1)*width + (x-1)] + image[(y-1)*width + (x+1)] +
                -2*image[y*width + (x-1)] + 2*image[y*width + (x+1)] +
                -image[(y+1)*width + (x-1)] + image[(y+1)*width + (x+1)];
            
            // Sobel Y
            float gy = 
                -image[(y-1)*width + (x-1)] - 2*image[(y-1)*width + x] - image[(y-1)*width + (x+1)] +
                image[(y+1)*width + (x-1)] + 2*image[(y+1)*width + x] + image[(y+1)*width + (x+1)];
            
            edge_map[y*width + x] = sqrtf(gx*gx + gy*gy);
        }
    }
    
    return AUTO_CALIB_SUCCESS;
}
// ============================================================================
// 2D高斯拟合和卷积操作
// ============================================================================

/**
 * @brief 拟合2D高斯函数
 */
static AutoCalibrationError fit_gaussian_2d(
    const float *image,
    int width,
    int height,
    float center_x,
    float center_y,
    int window_size,
    GaussianParams *params)
{
    if (!image || !params || window_size <= 0) {
        return AUTO_CALIB_ERROR_INVALID_PARAM;
    }
    
    int cx = (int)(center_x + 0.5f);
    int cy = (int)(center_y + 0.5f);
    int radius = window_size / 2;
    
    // 检查边界
    if (cx < radius || cx >= width - radius ||
        cy < radius || cy >= height - radius) {
        return AUTO_CALIB_ERROR_INVALID_PARAM;
    }
    
    // 提取窗口数据
    int num_points = window_size * window_size;
    float *x_data = AUTO_CALIB_MALLOC(float, num_points);
    float *y_data = AUTO_CALIB_MALLOC(float, num_points);
    float *z_data = AUTO_CALIB_MALLOC(float, num_points);
    
    if (!x_data || !y_data || !z_data) {
        AUTO_CALIB_FREE(x_data);
        AUTO_CALIB_FREE(y_data);
        AUTO_CALIB_FREE(z_data);
        return AUTO_CALIB_ERROR_MEMORY_ALLOCATION;
    }
    
    int idx = 0;
    for (int dy = -radius; dy <= radius; dy++) {
        for (int dx = -radius; dx <= radius; dx++) {
            x_data[idx] = dx;
            y_data[idx] = dy;
            z_data[idx] = image[(cy + dy) * width + (cx + dx)];
            idx++;
        }
    }
    
    // 初始参数估计
    params->x0 = 0.0f;
    params->y0 = 0.0f;
    params->amplitude = z_data[num_points / 2];  // 中心值
    params->sigma_x = window_size / 6.0f;
    params->sigma_y = window_size / 6.0f;
    params->theta = 0.0f;
    
    // Levenberg-Marquardt优化
    int max_iterations = 100;
    float lambda = 0.01f;
    float prev_error = FLT_MAX;
    
    for (int iter = 0; iter < max_iterations; iter++) {
        // 计算雅可比矩阵和残差
        float *jacobian = AUTO_CALIB_CALLOC(float, num_points * 6);
        float *residuals = AUTO_CALIB_MALLOC(float, num_points);
        
        if (!jacobian || !residuals) {
            AUTO_CALIB_FREE(x_data);
            AUTO_CALIB_FREE(y_data);
            AUTO_CALIB_FREE(z_data);
            AUTO_CALIB_FREE(jacobian);
            AUTO_CALIB_FREE(residuals);
            return AUTO_CALIB_ERROR_MEMORY_ALLOCATION;
        }
        
        float total_error = 0.0f;
        
        for (int i = 0; i < num_points; i++) {
            float x = x_data[i];
            float y = y_data[i];
            
            // 计算模型值
            float model = evaluate_gaussian_2d(x, y, params);
            residuals[i] = z_data[i] - model;
            total_error += residuals[i] * residuals[i];
            
            // 计算偏导数
            compute_gaussian_derivatives(
                x, y, params,
                &jacobian[i * 6]);
        }
        
        total_error = sqrtf(total_error / num_points);
        
        // 检查收敛
        if (fabsf(prev_error - total_error) < 1e-6f) {
            free(jacobian);
            free(residuals);
            break;
        }
        
        // 求解正规方程 (J^T J + λI) δ = J^T r
        float JtJ[36] = {0};  // 6x6矩阵
        float Jtr[6] = {0};   // 6x1向量
        
        // 计算 J^T J
        for (int i = 0; i < 6; i++) {
            for (int j = 0; j < 6; j++) {
                for (int k = 0; k < num_points; k++) {
                    JtJ[i * 6 + j] += jacobian[k * 6 + i] * jacobian[k * 6 + j];
                }
            }
            // 添加阻尼项
            JtJ[i * 6 + i] += lambda;
        }
        
        // 计算 J^T r
        for (int i = 0; i < 6; i++) {
            for (int k = 0; k < num_points; k++) {
                Jtr[i] += jacobian[k * 6 + i] * residuals[k];
            }
        }
        
        // 求解线性系统
        float delta[6];
        if (solve_linear_system_6x6(JtJ, Jtr, delta) != AUTO_CALIB_SUCCESS) {
            free(jacobian);
            free(residuals);
            break;
        }
        
        // 更新参数
        params->x0 += delta[0];
        params->y0 += delta[1];
        params->amplitude += delta[2];
        params->sigma_x += delta[3];
        params->sigma_y += delta[4];
        params->theta += delta[5];
        
        // 约束参数
        if (params->sigma_x < 0.5f) params->sigma_x = 0.5f;
        if (params->sigma_y < 0.5f) params->sigma_y = 0.5f;
        if (params->amplitude < 0.0f) params->amplitude = 0.0f;
        
        // 调整lambda
        if (total_error < prev_error) {
            lambda *= 0.1f;
            prev_error = total_error;
        } else {
            lambda *= 10.0f;
        }
        
        free(jacobian);
        free(residuals);
    }
    
    free(x_data);
    free(y_data);
    free(z_data);
    
    return AUTO_CALIB_SUCCESS;
}

/**
 * @brief 计算2D高斯函数值
 */
static float evaluate_gaussian_2d(
    float x,
    float y,
    const GaussianParams *params)
{
    float cos_theta = cosf(params->theta);
    float sin_theta = sinf(params->theta);
    
    // 旋转坐标
    float dx = x - params->x0;
    float dy = y - params->y0;
    float xr = dx * cos_theta + dy * sin_theta;
    float yr = -dx * sin_theta + dy * cos_theta;
    
    // 计算高斯值
    float sx2 = params->sigma_x * params->sigma_x;
    float sy2 = params->sigma_y * params->sigma_y;
    
    return params->amplitude * expf(
        -(xr*xr / (2.0f * sx2) + yr*yr / (2.0f * sy2)));
}

/**
 * @brief 计算高斯函数的偏导数
 */
static void compute_gaussian_derivatives(
    float x,
    float y,
    const GaussianParams *params,
    float *derivatives)
{
    float cos_theta = cosf(params->theta);
    float sin_theta = sinf(params->theta);
    
    float dx = x - params->x0;
    float dy = y - params->y0;
    float xr = dx * cos_theta + dy * sin_theta;
    float yr = -dx * sin_theta + dy * cos_theta;
    
    float sx2 = params->sigma_x * params->sigma_x;
    float sy2 = params->sigma_y * params->sigma_y;
    
    float exp_term = expf(-(xr*xr / (2.0f * sx2) + yr*yr / (2.0f * sy2)));
    float g = params->amplitude * exp_term;
    
    // ∂g/∂x0
    derivatives[0] = g * (xr * cos_theta / sx2 + yr * sin_theta / sy2);
    
    // ∂g/∂y0
    derivatives[1] = g * (xr * sin_theta / sx2 - yr * cos_theta / sy2);
    
    // ∂g/∂A
    derivatives[2] = exp_term;
    
    // ∂g/∂σx
    derivatives[3] = g * xr * xr / (sx2 * params->sigma_x);
    
    // ∂g/∂σy
    derivatives[4] = g * yr * yr / (sy2 * params->sigma_y);
    
    // ∂g/∂θ
    float dxr_dtheta = -dx * sin_theta + dy * cos_theta;
    float dyr_dtheta = -dx * cos_theta - dy * sin_theta;
    derivatives[5] = -g * (xr * dxr_dtheta / sx2 + yr * dyr_dtheta / sy2);
}

/**
 * @brief 求解6x6线性系统
 */
static AutoCalibrationError solve_linear_system_6x6(
    const float *A,
    const float *b,
    float *x)
{
    // 使用高斯消元法
    float aug[6][7];  // 增广矩阵
    
    // 构建增广矩阵
    for (int i = 0; i < 6; i++) {
        for (int j = 0; j < 6; j++) {
            aug[i][j] = A[i * 6 + j];
        }
        aug[i][6] = b[i];
    }
    
    // 前向消元
    for (int k = 0; k < 6; k++) {
        // 找主元
        int max_row = k;
        float max_val = fabsf(aug[k][k]);
        
        for (int i = k + 1; i < 6; i++) {
            if (fabsf(aug[i][k]) > max_val) {
                max_val = fabsf(aug[i][k]);
                max_row = i;
            }
        }
        
        if (max_val < 1e-10f) {
            return AUTO_CALIB_ERROR_INVALID_PARAM;
        }
        
        // 交换行
        if (max_row != k) {
            for (int j = 0; j < 7; j++) {
                float temp = aug[k][j];
                aug[k][j] = aug[max_row][j];
                aug[max_row][j] = temp;
            }
        }
        
        // 消元
        for (int i = k + 1; i < 6; i++) {
            float factor = aug[i][k] / aug[k][k];
            for (int j = k; j < 7; j++) {
                aug[i][j] -= factor * aug[k][j];
            }
        }
    }
    
    // 回代
    for (int i = 5; i >= 0; i--) {
        x[i] = aug[i][6];
        for (int j = i + 1; j < 6; j++) {
            x[i] -= aug[i][j] * x[j];
        }
        x[i] /= aug[i][i];
    }
    
    return AUTO_CALIB_SUCCESS;
}

/**
 * @brief 2D卷积
 */
static AutoCalibrationError convolve_2d(
    const float *image,
    int width,
    int height,
    const float *kernel,
    int kernel_size,
    float *output)
{
    if (!image || !kernel || !output) {
        return AUTO_CALIB_ERROR_NULL_POINTER;
    }
    
    if (width <= 0 || height <= 0 || kernel_size <= 0 || kernel_size % 2 == 0) {
        return AUTO_CALIB_ERROR_INVALID_PARAM;
    }
    
    int radius = kernel_size / 2;
    
    // 对每个输出像素
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            double sum = 0.0;
            double weight_sum = 0.0;
            
            // 卷积核窗口
            for (int ky = 0; ky < kernel_size; ky++) {
                for (int kx = 0; kx < kernel_size; kx++) {
                    int img_x = x + kx - radius;
                    int img_y = y + ky - radius;
                    
                    // 边界处理（镜像）
                    if (img_x < 0) img_x = -img_x;
                    if (img_x >= width) img_x = 2 * width - img_x - 2;
                    if (img_y < 0) img_y = -img_y;
                    if (img_y >= height) img_y = 2 * height - img_y - 2;
                    
                    float kernel_val = kernel[ky * kernel_size + kx];
                    sum += image[img_y * width + img_x] * kernel_val;
                    weight_sum += kernel_val;
                }
            }
            
            output[y * width + x] = sum / weight_sum;
        }
    }
    
    return AUTO_CALIB_SUCCESS;
}

/**
 * @brief 更新盲PSF估计
 */
static AutoCalibrationError update_psf_blind(
    const float *error_image,
    int width,
    int height,
    const float *current_psf,
    int psf_size,
    float *psf_update)
{
    if (!error_image || !current_psf || !psf_update) {
        return AUTO_CALIB_ERROR_NULL_POINTER;
    }
    
    int psf_pixels = psf_size * psf_size;
    int psf_radius = psf_size / 2;
    
    // 初始化更新为1
    for (int i = 0; i < psf_pixels; i++) {
        psf_update[i] = 1.0f;
    }
    
    // 对PSF的每个位置累积误差
    for (int py = 0; py < psf_size; py++) {
        for (int px = 0; px < psf_size; px++) {
            double sum = 0.0;
            int count = 0;
            
            // 在图像中采样
            for (int y = psf_radius; y < height - psf_radius; y += 2) {
                for (int x = psf_radius; x < width - psf_radius; x += 2) {
                    int img_x = x + px - psf_radius;
                    int img_y = y + py - psf_radius;
                    
                    if (img_x >= 0 && img_x < width && 
                        img_y >= 0 && img_y < height) {
                        sum += error_image[img_y * width + img_x];
                        count++;
                    }
                }
            }
            
            if (count > 0) {
                psf_update[py * psf_size + px] = sum / count;
            }
        }
    }
    
    return AUTO_CALIB_SUCCESS;
}

float auto_calib_evaluate_psf_quality(const float *psf, int size) {
    if (!psf || size <= 0) {
        return 0.0f;
    }
    
    int num_pixels = size * size;
    int center = size / 2;
    
    // 质量评分因素
    float quality = 100.0f;
    
    // 1. 中心集中度（PSF应该在中心最强）
    float center_value = psf[center * size + center];
    float max_value = 0.0f;
    for (int i = 0; i < num_pixels; i++) {
        if (psf[i] > max_value) max_value = psf[i];
    }
    float centrality = center_value / (max_value + 1e-10f);
    quality *= centrality;
    
    // 2. 对称性
    float symmetry_error = 0.0f;
    int sym_count = 0;
    for (int y = 0; y < size; y++) {
        for (int x = 0; x < size; x++) {
            int sym_x = size - 1 - x;
            int sym_y = size - 1 - y;
            float diff = fabsf(psf[y * size + x] - psf[sym_y * size + sym_x]);
            symmetry_error += diff;
            sym_count++;
        }
    }
    symmetry_error /= sym_count;
    float symmetry = 1.0f / (1.0f + symmetry_error * 10.0f);
    quality *= symmetry;
    
    // 3. 平滑度（PSF应该平滑）
    float smoothness = 0.0f;
    int smooth_count = 0;
    for (int y = 1; y < size - 1; y++) {
        for (int x = 1; x < size - 1; x++) {
            int idx = y * size + x;
            float laplacian = fabsf(
                4 * psf[idx] -
                psf[idx - 1] - psf[idx + 1] -
                psf[idx - size] - psf[idx + size]);
            smoothness += laplacian;
            smooth_count++;
        }
    }
    smoothness = 1.0f / (1.0f + smoothness / smooth_count);
    quality *= smoothness;
    
    // 4. 能量集中度（大部分能量应该在中心区域）
    float center_energy = 0.0f;
    float total_energy = 0.0f;
    int inner_radius = size / 4;
    
    for (int y = 0; y < size; y++) {
        for (int x = 0; x < size; x++) {
            float value = psf[y * size + x];
            total_energy += value;
            
            int dx = x - center;
            int dy = y - center;
            if (dx*dx + dy*dy <= inner_radius*inner_radius) {
                center_energy += value;
            }
        }
    }
    
    float energy_concentration = center_energy / (total_energy + 1e-10f);
    quality *= energy_concentration;
    
    // 限制在0-100范围内
    if (quality < 0.0f) quality = 0.0f;
    if (quality > 100.0f) quality = 100.0f;
    
    return quality;
}

// ============================================================================
// 反卷积实现
// ============================================================================

AutoCalibrationError auto_calib_deconvolve(
    const float *blurred_image,
    int width,
    int height,
    const float *psf,
    int psf_size,
    AutoCalibDeconvMethod method,
    int max_iterations,
    float **deconvolved_image)
{
    AUTO_CALIB_CHECK_NULL(blurred_image);
    AUTO_CALIB_CHECK_NULL(psf);
    AUTO_CALIB_CHECK_NULL(deconvolved_image);
    
    if (width <= 0 || height <= 0 || psf_size <= 0) {
        return AUTO_CALIB_ERROR_INVALID_PARAM;
    }
    
    AutoCalibrationError err;
    
    switch (method) {
        case AUTO_CALIB_DECONV_RICHARDSON_LUCY:
            err = deconvolve_richardson_lucy(
                blurred_image, width, height,
                psf, psf_size,
                max_iterations,
                deconvolved_image);
            break;
            
        case AUTO_CALIB_DECONV_WIENER:
            err = deconvolve_wiener(
                blurred_image, width, height,
                psf, psf_size,
                0.01f,  // noise-to-signal ratio
                deconvolved_image);
            break;
            
        case AUTO_CALIB_DECONV_LUCY_RICHARDSON_TV:
            err = deconvolve_richardson_lucy_tv(
                blurred_image, width, height,
                psf, psf_size,
                max_iterations,
                0.001f,  // TV regularization parameter
                deconvolved_image);
            break;
            
        default:
            return AUTO_CALIB_ERROR_INVALID_PARAM;
    }
    
    return err;
}

/**
 * @brief Richardson-Lucy反卷积
 */
static AutoCalibrationError deconvolve_richardson_lucy(
    const float *blurred,
    int width,
    int height,
    const float *psf,
    int psf_size,
    int max_iterations,
    float **result)
{
    int num_pixels = width * height;
    
    // 初始化估计为模糊图像
    float *estimate = AUTO_CALIB_MALLOC(float, num_pixels);
    if (!estimate) {
        return AUTO_CALIB_ERROR_MEMORY_ALLOCATION;
    }
    memcpy(estimate, blurred, num_pixels * sizeof(float));
    
    // 分配工作缓冲区
    float *convolved = AUTO_CALIB_MALLOC(float, num_pixels);
    float *ratio = AUTO_CALIB_MALLOC(float, num_pixels);
    float *correction = AUTO_CALIB_MALLOC(float, num_pixels);
    
    if (!convolved || !ratio || !correction) {
        AUTO_CALIB_FREE(estimate);
        AUTO_CALIB_FREE(convolved);
        AUTO_CALIB_FREE(ratio);
        AUTO_CALIB_FREE(correction);
        return AUTO_CALIB_ERROR_MEMORY_ALLOCATION;
    }
    
    // 创建翻转的PSF
    float *psf_flipped = AUTO_CALIB_MALLOC(float, psf_size * psf_size);
    if (!psf_flipped) {
        AUTO_CALIB_FREE(estimate);
        AUTO_CALIB_FREE(convolved);
        AUTO_CALIB_FREE(ratio);
        AUTO_CALIB_FREE(correction);
        return AUTO_CALIB_ERROR_MEMORY_ALLOCATION;
    }
    
    for (int y = 0; y < psf_size; y++) {
        for (int x = 0; x < psf_size; x++) {
            psf_flipped[y * psf_size + x] = 
                psf[(psf_size - 1 - y) * psf_size + (psf_size - 1 - x)];
        }
    }
    
    // Richardson-Lucy迭代
    for (int iter = 0; iter < max_iterations; iter++) {
        // 1. 用当前估计卷积PSF
        AutoCalibrationError err = convolve_2d(
            estimate, width, height,
            psf, psf_size,
            convolved);
        
        if (err != AUTO_CALIB_SUCCESS) {
            AUTO_CALIB_FREE(estimate);
            AUTO_CALIB_FREE(convolved);
            AUTO_CALIB_FREE(ratio);
            AUTO_CALIB_FREE(correction);
            AUTO_CALIB_FREE(psf_flipped);
            return err;
        }
        
        // 2. 计算比率
        for (int i = 0; i < num_pixels; i++) {
            ratio[i] = blurred[i] / (convolved[i] + 1e-10f);
        }
        
        // 3. 用翻转的PSF卷积比率
        err = convolve_2d(
            ratio, width, height,
            psf_flipped, psf_size,
            correction);
        
        if (err != AUTO_CALIB_SUCCESS) {
            AUTO_CALIB_FREE(estimate);
            AUTO_CALIB_FREE(convolved);
            AUTO_CALIB_FREE(ratio);
            AUTO_CALIB_FREE(correction);
            AUTO_CALIB_FREE(psf_flipped);
            return err;
        }
        
        // 4. 更新估计
        for (int i = 0; i < num_pixels; i++) {
            estimate[i] *= correction[i];
            // 确保非负
            if (estimate[i] < 0.0f) estimate[i] = 0.0f;
        }
    }
    
    AUTO_CALIB_FREE(convolved);
    AUTO_CALIB_FREE(ratio);
    AUTO_CALIB_FREE(correction);
    AUTO_CALIB_FREE(psf_flipped);
    
    *result = estimate;
    
    return AUTO_CALIB_SUCCESS;
}
/**
 * @brief Wiener反卷积
 */
static AutoCalibrationError deconvolve_wiener(
    const float *blurred,
    int width,
    int height,
    const float *psf,
    int psf_size,
    float noise_ratio,
    float **result)
{
    // Wiener滤波需要在频域进行
    // 这里实现简化版本（空域近似）
    
    int num_pixels = width * height;
    
    float *output = AUTO_CALIB_MALLOC(float, num_pixels);
    if (!output) {
        return AUTO_CALIB_ERROR_MEMORY_ALLOCATION;
    }
    
    // 计算PSF的能量
    float psf_energy = 0.0f;
    for (int i = 0; i < psf_size * psf_size; i++) {
        psf_energy += psf[i] * psf[i];
    }
    
    // 创建Wiener滤波器
    int psf_pixels = psf_size * psf_size;
    float *wiener_filter = AUTO_CALIB_MALLOC(float, psf_pixels);
    if (!wiener_filter) {
        free(output);
        return AUTO_CALIB_ERROR_MEMORY_ALLOCATION;
    }
    
    // H* / (|H|^2 + K)
    float K = noise_ratio * psf_energy;
    for (int i = 0; i < psf_pixels; i++) {
        wiener_filter[i] = psf[i] / (psf[i] * psf[i] + K);
    }
    
    // 应用滤波器
    AutoCalibrationError err = convolve_2d(
        blurred, width, height,
        wiener_filter, psf_size,
        output);
    
    free(wiener_filter);
    
    if (err != AUTO_CALIB_SUCCESS) {
        free(output);
        return err;
    }
    
    *result = output;
    
    return AUTO_CALIB_SUCCESS;
}

/**
 * @brief Richardson-Lucy反卷积 + TV正则化
 */
static AutoCalibrationError deconvolve_richardson_lucy_tv(
    const float *blurred,
    int width,
    int height,
    const float *psf,
    int psf_size,
    int max_iterations,
    float tv_weight,
    float **result)
{
    int num_pixels = width * height;
    
    // 初始化估计
    float *estimate = AUTO_CALIB_MALLOC(float, num_pixels);
    if (!estimate) {
        return AUTO_CALIB_ERROR_MEMORY_ALLOCATION;
    }
    memcpy(estimate, blurred, num_pixels * sizeof(float));
    
    // 分配工作缓冲区
    float *convolved = AUTO_CALIB_MALLOC(float, num_pixels);
    float *ratio = AUTO_CALIB_MALLOC(float, num_pixels);
    float *correction = AUTO_CALIB_MALLOC(float, num_pixels);
    float *tv_term = AUTO_CALIB_MALLOC(float, num_pixels);
    
    if (!convolved || !ratio || !correction || !tv_term) {
        AUTO_CALIB_FREE(estimate);
        AUTO_CALIB_FREE(convolved);
        AUTO_CALIB_FREE(ratio);
        AUTO_CALIB_FREE(correction);
        AUTO_CALIB_FREE(tv_term);
        return AUTO_CALIB_ERROR_MEMORY_ALLOCATION;
    }
    
    // 创建翻转的PSF
    float *psf_flipped = AUTO_CALIB_MALLOC(float, psf_size * psf_size);
    if (!psf_flipped) {
        AUTO_CALIB_FREE(estimate);
        AUTO_CALIB_FREE(convolved);
        AUTO_CALIB_FREE(ratio);
        AUTO_CALIB_FREE(correction);
        AUTO_CALIB_FREE(tv_term);
        return AUTO_CALIB_ERROR_MEMORY_ALLOCATION;
    }
    
    for (int y = 0; y < psf_size; y++) {
        for (int x = 0; x < psf_size; x++) {
            psf_flipped[y * psf_size + x] = 
                psf[(psf_size - 1 - y) * psf_size + (psf_size - 1 - x)];
        }
    }
    
    // Richardson-Lucy + TV迭代
    for (int iter = 0; iter < max_iterations; iter++) {
        // 1. 标准Richardson-Lucy步骤
        AutoCalibrationError err = convolve_2d(
            estimate, width, height,
            psf, psf_size,
            convolved);
        
        if (err != AUTO_CALIB_SUCCESS) {
            AUTO_CALIB_FREE(estimate);
            AUTO_CALIB_FREE(convolved);
            AUTO_CALIB_FREE(ratio);
            AUTO_CALIB_FREE(correction);
            AUTO_CALIB_FREE(tv_term);
            AUTO_CALIB_FREE(psf_flipped);
            return err;
        }
        
        for (int i = 0; i < num_pixels; i++) {
            ratio[i] = blurred[i] / (convolved[i] + 1e-10f);
        }
        
        err = convolve_2d(
            ratio, width, height,
            psf_flipped, psf_size,
            correction);
        
        if (err != AUTO_CALIB_SUCCESS) {
            AUTO_CALIB_FREE(estimate);
            AUTO_CALIB_FREE(convolved);
            AUTO_CALIB_FREE(ratio);
            AUTO_CALIB_FREE(correction);
            AUTO_CALIB_FREE(tv_term);
            AUTO_CALIB_FREE(psf_flipped);
            return err;
        }
        
        // 2. 计算TV正则化项
        compute_tv_regularization(
            estimate, width, height,
            tv_weight,
            tv_term);
        
        // 3. 组合更新
        for (int i = 0; i < num_pixels; i++) {
            estimate[i] *= (correction[i] + tv_term[i]);
            // 确保非负
            if (estimate[i] < 0.0f) estimate[i] = 0.0f;
        }
    }
    
    AUTO_CALIB_FREE(convolved);
    AUTO_CALIB_FREE(ratio);
    AUTO_CALIB_FREE(correction);
    AUTO_CALIB_FREE(tv_term);
    AUTO_CALIB_FREE(psf_flipped);
    
    *result = estimate;
    
    return AUTO_CALIB_SUCCESS;
}

/**
 * @brief 计算TV正则化项
 */
static void compute_tv_regularization(
    const float *image,
    int width,
    int height,
    float weight,
    float *tv_term)
{
    // 计算各向异性TV: div(∇u / |∇u|)
    
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            int idx = y * width + x;
            
            // 计算梯度
            float dx = image[idx + 1] - image[idx - 1];
            float dy = image[idx + width] - image[idx - width];
            float grad_mag = sqrtf(dx*dx + dy*dy) + 1e-10f;
            
            // 归一化梯度
            float nx = dx / grad_mag;
            float ny = dy / grad_mag;
            
            // 计算散度
            float div_x = 0.0f, div_y = 0.0f;
            
            if (x > 0 && x < width - 1) {
                float nx_left = (image[idx] - image[idx - 1]) / 
                    (sqrtf(powf(image[idx] - image[idx - 1], 2) + 
                           powf(image[idx + width] - image[idx - width], 2)) + 1e-10f);
                div_x = nx - nx_left;
            }
            
            if (y > 0 && y < height - 1) {
                float ny_up = (image[idx] - image[idx - width]) / 
                    (sqrtf(powf(image[idx + 1] - image[idx - 1], 2) + 
                           powf(image[idx] - image[idx - width], 2)) + 1e-10f);
                div_y = ny - ny_up;
            }
            
            tv_term[idx] = weight * (div_x + div_y);
        }
    }
    
    // 边界处理
    for (int x = 0; x < width; x++) {
        tv_term[x] = 0.0f;
        tv_term[(height - 1) * width + x] = 0.0f;
    }
    for (int y = 0; y < height; y++) {
        tv_term[y * width] = 0.0f;
        tv_term[y * width + width - 1] = 0.0f;
    }
}

// ============================================================================
// 完整校准流程实现
// ============================================================================

AutoCalibrationError auto_calib_full_calibration(
    AutoCalibrationContext *context,
    const AutoCalibImageSet *dark_images,
    const AutoCalibImageSet *flat_images,
    const float *sample_image,
    AutoCalibFullResult *result)
{
    AUTO_CALIB_CHECK_NULL(context);
    AUTO_CALIB_CHECK_NULL(result);
    
    // 初始化结果
    memset(result, 0, sizeof(AutoCalibFullResult));
    
    context->status = AUTO_CALIB_STATUS_PROCESSING;
    context->start_time = time(NULL);
    
    // 1. 暗场校准
    if (dark_images && dark_images->num_images > 0) {
        update_progress(context, "Computing dark field", 0, 100);
        
        AutoCalibrationError err = auto_calib_dark_field(
            context,
            dark_images,
            &result->dark_field,
            &result->width,
            &result->height,
            &result->dark_quality);
        
        if (err != AUTO_CALIB_SUCCESS) {
            auto_calib_free_full_result(result);
            return err;
        }
        
        result->has_dark = true;
    }
    
    // 2. 平场校准
    if (flat_images && flat_images->num_images > 0) {
        update_progress(context, "Computing flat field", 25, 100);
        
        AutoCalibrationError err = auto_calib_flat_field(
            context,
            flat_images,
            result->dark_field,  // 可以为NULL
            &result->flat_field,
            &result->width,
            &result->height,
            &result->flat_quality);
        
        if (err != AUTO_CALIB_SUCCESS) {
            auto_calib_free_full_result(result);
            return err;
        }
        
        result->has_flat = true;
    }
    
    // 3. PSF估计
    if (sample_image && context->config.enable_psf_estimation) {
        update_progress(context, "Estimating PSF", 50, 100);
        
        // 首先应用暗场和平场校准
        float *calibrated = NULL;
        AutoCalibrationError err = auto_calib_apply_calibration(
            sample_image,
            result->width,
            result->height,
            result->dark_field,
            result->flat_field,
            &calibrated);
        
        if (err != AUTO_CALIB_SUCCESS) {
            auto_calib_free_full_result(result);
            return err;
        }
        
        // 估计PSF
        err = auto_calib_estimate_psf(
            context,
            calibrated,
            result->width,
            result->height,
            context->config.psf_size,
            &result->psf,
            &result->psf_quality);
        
        free(calibrated);
        
        if (err != AUTO_CALIB_SUCCESS) {
            auto_calib_free_full_result(result);
            return err;
        }
        
        result->has_psf = true;
        result->psf_size = context->config.psf_size;
    }
    
    // 4. 计算整体质量指标
    update_progress(context, "Computing quality metrics", 75, 100);
    
    if (sample_image) {
        // 应用完整校准
        float *calibrated = NULL;
        AutoCalibrationError err = auto_calib_apply_calibration(
            sample_image,
            result->width,
            result->height,
            result->dark_field,
            result->flat_field,
            &calibrated);
        
        if (err == AUTO_CALIB_SUCCESS) {
            // 计算SNR
            auto_calib_compute_snr(
                calibrated,
                result->width,
                result->height,
                &result->snr);
            
            // 估计噪声水平
            auto_calib_estimate_noise_level(
                calibrated,
                result->width,
                result->height,
                &result->noise_level);
            
            free(calibrated);
        }
    }
    
    // 5. 计算总体质量分数
    result->overall_quality = compute_overall_quality(result);
    
    update_progress(context, "Calibration completed", 100, 100);
    context->status = AUTO_CALIB_STATUS_COMPLETED;
    
    return AUTO_CALIB_SUCCESS;
}

/**
 * @brief 计算总体质量分数
 */
static float compute_overall_quality(const AutoCalibFullResult *result) {
    if (!result) return 0.0f;
    
    float quality = 0.0f;
    int count = 0;
    
    if (result->has_dark) {
        quality += result->dark_quality;
        count++;
    }
    
    if (result->has_flat) {
        quality += result->flat_quality;
        count++;
    }
    
    if (result->has_psf) {
        quality += result->psf_quality;
        count++;
    }
    
    if (count > 0) {
        quality /= count;
    }
    
    // 考虑SNR
    if (result->snr > 0.0f) {
        float snr_score = fminf(result->snr / 100.0f, 1.0f) * 100.0f;
        quality = (quality + snr_score) / 2.0f;
    }
    
    return quality;
}

void auto_calib_free_full_result(AutoCalibFullResult *result) {
    if (!result) return;
    
    AUTO_CALIB_FREE(result->dark_field);
    AUTO_CALIB_FREE(result->flat_field);
    AUTO_CALIB_FREE(result->psf);
    
    memset(result, 0, sizeof(AutoCalibFullResult));
}

// ============================================================================
// 保存和加载校准数据
// ============================================================================

AutoCalibrationError auto_calib_save_calibration(
    const char *filename,
    const AutoCalibFullResult *result)
{
    AUTO_CALIB_CHECK_NULL(filename);
    AUTO_CALIB_CHECK_NULL(result);
    
    FILE *fp = fopen(filename, "wb");
    if (!fp) {
        return AUTO_CALIB_ERROR_FILE_IO;
    }
    
    // 写入文件头
    const char magic[] = "AUTOCALIB";
    const uint32_t version = 1;
    
    if (fwrite(magic, 1, 9, fp) != 9 ||
        fwrite(&version, sizeof(uint32_t), 1, fp) != 1) {
        fclose(fp);
        return AUTO_CALIB_ERROR_FILE_IO;
    }
    
    // 写入维度
    if (fwrite(&result->width, sizeof(int), 1, fp) != 1 ||
        fwrite(&result->height, sizeof(int), 1, fp) != 1) {
        fclose(fp);
        return AUTO_CALIB_ERROR_FILE_IO;
    }
    
    // 写入标志
    uint8_t flags = 0;
    if (result->has_dark) flags |= 0x01;
    if (result->has_flat) flags |= 0x02;
    if (result->has_psf) flags |= 0x04;
    
    if (fwrite(&flags, sizeof(uint8_t), 1, fp) != 1) {
        fclose(fp);
        return AUTO_CALIB_ERROR_FILE_IO;
    }
    
    int num_pixels = result->width * result->height;
    
    // 写入暗场
    if (result->has_dark) {
        if (fwrite(result->dark_field, sizeof(float), num_pixels, fp) != 
            (size_t)num_pixels) {
            fclose(fp);
            return AUTO_CALIB_ERROR_FILE_IO;
        }
        if (fwrite(&result->dark_quality, sizeof(float), 1, fp) != 1) {
            fclose(fp);
            return AUTO_CALIB_ERROR_FILE_IO;
        }
    }
    
    // 写入平场
    if (result->has_flat) {
        if (fwrite(result->flat_field, sizeof(float), num_pixels, fp) != 
            (size_t)num_pixels) {
            fclose(fp);
            return AUTO_CALIB_ERROR_FILE_IO;
        }
        if (fwrite(&result->flat_quality, sizeof(float), 1, fp) != 1) {
            fclose(fp);
            return AUTO_CALIB_ERROR_FILE_IO;
        }
    }
    
    // 写入PSF
    if (result->has_psf) {
        if (fwrite(&result->psf_size, sizeof(int), 1, fp) != 1) {
            fclose(fp);
            return AUTO_CALIB_ERROR_FILE_IO;
        }
        
        int psf_pixels = result->psf_size * result->psf_size;
        if (fwrite(result->psf, sizeof(float), psf_pixels, fp) != 
            (size_t)psf_pixels) {
            fclose(fp);
            return AUTO_CALIB_ERROR_FILE_IO;
        }
        if (fwrite(&result->psf_quality, sizeof(float), 1, fp) != 1) {
            fclose(fp);
            return AUTO_CALIB_ERROR_FILE_IO;
        }
    }
    
    // 写入质量指标
    if (fwrite(&result->snr, sizeof(float), 1, fp) != 1 ||
        fwrite(&result->noise_level, sizeof(float), 1, fp) != 1 ||
        fwrite(&result->overall_quality, sizeof(float), 1, fp) != 1) {
        fclose(fp);
        return AUTO_CALIB_ERROR_FILE_IO;
    }
    
    fclose(fp);
    
    return AUTO_CALIB_SUCCESS;
}

AutoCalibrationError auto_calib_load_calibration(
    const char *filename,
    AutoCalibFullResult *result)
{
    AUTO_CALIB_CHECK_NULL(filename);
    AUTO_CALIB_CHECK_NULL(result);
    
    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        return AUTO_CALIB_ERROR_FILE_IO;
    }
    
    // 初始化结果
    memset(result, 0, sizeof(AutoCalibFullResult));
    
    // 读取文件头
    char magic[10] = {0};
    uint32_t version;
    
    if (fread(magic, 1, 9, fp) != 9 ||
        strcmp(magic, "AUTOCALIB") != 0 ||
        fread(&version, sizeof(uint32_t), 1, fp) != 1 ||
        version != 1) {
        fclose(fp);
        return AUTO_CALIB_ERROR_FILE_IO;
    }
    
    // 读取维度
    if (fread(&result->width, sizeof(int), 1, fp) != 1 ||
        fread(&result->height, sizeof(int), 1, fp) != 1) {
        fclose(fp);
        return AUTO_CALIB_ERROR_FILE_IO;
    }
    
    // 读取标志
    uint8_t flags;
    if (fread(&flags, sizeof(uint8_t), 1, fp) != 1) {
        fclose(fp);
        return AUTO_CALIB_ERROR_FILE_IO;
    }
    
    result->has_dark = (flags & 0x01) != 0;
    result->has_flat = (flags & 0x02) != 0;
    result->has_psf = (flags & 0x04) != 0;
    
    int num_pixels = result->width * result->height;
    
    // 读取暗场
    if (result->has_dark) {
        result->dark_field = AUTO_CALIB_MALLOC(float, num_pixels);
        if (!result->dark_field) {
            fclose(fp);
            auto_calib_free_full_result(result);
            return AUTO_CALIB_ERROR_MEMORY_ALLOCATION;
        }
        
        if (fread(result->dark_field, sizeof(float), num_pixels, fp) != 
            (size_t)num_pixels ||
            fread(&result->dark_quality, sizeof(float), 1, fp) != 1) {
            fclose(fp);
            auto_calib_free_full_result(result);
            return AUTO_CALIB_ERROR_FILE_IO;
        }
    }
    
    // 读取平场
    if (result->has_flat) {
        result->flat_field = AUTO_CALIB_MALLOC(float, num_pixels);
        if (!result->flat_field) {
            fclose(fp);
            auto_calib_free_full_result(result);
            return AUTO_CALIB_ERROR_MEMORY_ALLOCATION;
        }
        
        if (fread(result->flat_field, sizeof(float), num_pixels, fp) != 
            (size_t)num_pixels ||
            fread(&result->flat_quality, sizeof(float), 1, fp) != 1) {
            fclose(fp);
            auto_calib_free_full_result(result);
            return AUTO_CALIB_ERROR_FILE_IO;
        }
    }
    
    // 读取PSF
    if (result->has_psf) {
        if (fread(&result->psf_size, sizeof(int), 1, fp) != 1) {
            fclose(fp);
            auto_calib_free_full_result(result);
            return AUTO_CALIB_ERROR_FILE_IO;
        }
        
        int psf_pixels = result->psf_size * result->psf_size;
        result->psf = AUTO_CALIB_MALLOC(float, psf_pixels);
        if (!result->psf) {
            fclose(fp);
            auto_calib_free_full_result(result);
            return AUTO_CALIB_ERROR_MEMORY_ALLOCATION;
        }
        
        if (fread(result->psf, sizeof(float), psf_pixels, fp) != 
            (size_t)psf_pixels ||
            fread(&result->psf_quality, sizeof(float), 1, fp) != 1) {
            fclose(fp);
            auto_calib_free_full_result(result);
            return AUTO_CALIB_ERROR_FILE_IO;
        }
    }
    
    // 读取质量指标
    if (fread(&result->snr, sizeof(float), 1, fp) != 1 ||
        fread(&result->noise_level, sizeof(float), 1, fp) != 1 ||
        fread(&result->overall_quality, sizeof(float), 1, fp) != 1) {
        fclose(fp);
        auto_calib_free_full_result(result);
        return AUTO_CALIB_ERROR_FILE_IO;
    }
    
    fclose(fp);
    
    return AUTO_CALIB_SUCCESS;
}

// ============================================================================
// 导出为常见格式
// ============================================================================

AutoCalibrationError auto_calib_export_to_tiff(
    const char *filename,
    const float *data,
    int width,
    int height,
    bool normalize)
{
    AUTO_CALIB_CHECK_NULL(filename);
    AUTO_CALIB_CHECK_NULL(data);
    
    if (width <= 0 || height <= 0) {
        return AUTO_CALIB_ERROR_INVALID_PARAM;
    }
    
    // 找到数据范围
    float min_val = FLT_MAX;
    float max_val = -FLT_MAX;
    
    int num_pixels = width * height;
    for (int i = 0; i < num_pixels; i++) {
        if (data[i] < min_val) min_val = data[i];
        if (data[i] > max_val) max_val = data[i];
    }
    
    // 转换为16位
    uint16_t *image_16bit = AUTO_CALIB_MALLOC(uint16_t, num_pixels);
    if (!image_16bit) {
        return AUTO_CALIB_ERROR_MEMORY_ALLOCATION;
    }
    
    float scale = (max_val > min_val) ? (65535.0f / (max_val - min_val)) : 1.0f;
    
    for (int i = 0; i < num_pixels; i++) {
        float value = data[i];
        if (normalize) {
            value = (value - min_val) * scale;
        }
        
        if (value < 0.0f) value = 0.0f;
        if (value > 65535.0f) value = 65535.0f;
        
        image_16bit[i] = (uint16_t)value;
    }
    
    // 这里应该使用libtiff库写入TIFF文件
    // 为了简化，这里只是示例代码
    FILE *fp = fopen(filename, "wb");
    if (!fp) {
        free(image_16bit);
        return AUTO_CALIB_ERROR_FILE_IO;
    }
    
    // 简化的TIFF头（实际应该使用libtiff）
    // 这里只是占位符
    fwrite(image_16bit, sizeof(uint16_t), num_pixels, fp);
    
    fclose(fp);
    free(image_16bit);
    
    return AUTO_CALIB_SUCCESS;
}

AutoCalibrationError auto_calib_export_to_fits(
    const char *filename,
    const float *data,
    int width,
    int height)
{
    AUTO_CALIB_CHECK_NULL(filename);
    AUTO_CALIB_CHECK_NULL(data);
    
    if (width <= 0 || height <= 0) {
        return AUTO_CALIB_ERROR_INVALID_PARAM;
    }
    
    // 这里应该使用cfitsio库写入FITS文件
    // 为了简化，这里只是示例代码
    
    FILE *fp = fopen(filename, "wb");
    if (!fp) {
        return AUTO_CALIB_ERROR_FILE_IO;
    }
    
    // 简化的FITS头（实际应该使用cfitsio）
    // 这里只是占位符
    fwrite(data, sizeof(float), width * height, fp);
    
    fclose(fp);
    
    return AUTO_CALIB_SUCCESS;
}

// ============================================================================
// 辅助函数实现
// ============================================================================

const char* auto_calib_error_string(AutoCalibrationError error) {
    switch (error) {
        case AUTO_CALIB_SUCCESS:
            return "Success";
        case AUTO_CALIB_ERROR_NULL_POINTER:
            return "Null pointer error";
        case AUTO_CALIB_ERROR_INVALID_PARAM:
            return "Invalid parameter";
        case AUTO_CALIB_ERROR_MEMORY_ALLOCATION:
            return "Memory allocation failed";
        case AUTO_CALIB_ERROR_FILE_IO:
            return "File I/O error";
        case AUTO_CALIB_ERROR_INVALID_IMAGE:
            return "Invalid image";
        case AUTO_CALIB_ERROR_INSUFFICIENT_DATA:
            return "Insufficient data";
        case AUTO_CALIB_ERROR_OPERATION_CANCELLED:
            return "Operation cancelled";
        case AUTO_CALIB_ERROR_NOT_IMPLEMENTED:
            return "Not implemented";
        default:
            return "Unknown error";
    }
}

void auto_calib_print_result_summary(const AutoCalibFullResult *result) {
    if (!result) return;
    
    printf("\n=== Auto Calibration Result Summary ===\n");
    printf("Image dimensions: %d x %d\n", result->width, result->height);
    printf("\n");
    
    if (result->has_dark) {
        printf("Dark Field:\n");
        printf("  Quality: %.2f%%\n", result->dark_quality);
    }
    
    if (result->has_flat) {
        printf("Flat Field:\n");
        printf("  Quality: %.2f%%\n", result->flat_quality);
    }
    
    if (result->has_psf) {
        printf("PSF:\n");
        printf("  Size: %d x %d\n", result->psf_size, result->psf_size);
        printf("  Quality: %.2f%%\n", result->psf_quality);
    }
    
    printf("\nQuality Metrics:\n");
    printf("  SNR: %.2f\n", result->snr);
    printf("  Noise Level: %.4f\n", result->noise_level);
    printf("  Overall Quality: %.2f%%\n", result->overall_quality);
    printf("\n");
}


