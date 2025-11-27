/**
 * @file calibration_manager.c
 * @brief 校准数据管理模块实现 - 第一部分
 * @details 基础功能：初始化、数据结构管理、PSF操作
 */

#include "calibration_manager.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ============================================================================
// 内部数据结构
// ============================================================================

/**
 * @brief 校准管理器内部结构
 */
struct CalibrationManager {
    bool initialized;                   ///< 是否已初始化
    bool verbose;                       ///< 详细输出模式
    CalibrationData *current_data;      ///< 当前加载的校准数据
    char last_error[256];               ///< 最后的错误信息
    
    // 统计信息
    size_t total_loads;                 ///< 总加载次数
    size_t total_saves;                 ///< 总保存次数
    size_t failed_operations;           ///< 失败操作次数
    
    // 缓存
    PSFData *psf_cache[16];            ///< PSF缓存
    int cache_size;                     ///< 缓存大小
};

// ============================================================================
// 静态全局变量
// ============================================================================

static bool g_verbose = false;          ///< 全局详细输出标志

// ============================================================================
// 内部辅助函数声明
// ============================================================================

static void set_error(CalibrationManager *manager, const char *error);
static uint32_t calculate_crc32(const uint8_t *data, size_t length);
static float bessel_j1(float x);
static void psf_cache_add(CalibrationManager *manager, PSFData *psf);
static PSFData* psf_cache_find(CalibrationManager *manager, 
                               float wavelength, 
                               float na);

// ============================================================================
// 错误处理函数
// ============================================================================

/**
 * @brief 设置错误信息
 */
static void set_error(CalibrationManager *manager, const char *error) {
    if (manager && error) {
        strncpy(manager->last_error, error, sizeof(manager->last_error) - 1);
        manager->last_error[sizeof(manager->last_error) - 1] = '\0';
        manager->failed_operations++;
    }
    
    if (g_verbose || (manager && manager->verbose)) {
        fprintf(stderr, "[CalibManager Error] %s\n", error);
    }
}

/**
 * @brief 获取错误描述字符串
 */
const char* calib_error_string(CalibrationError error) {
    switch (error) {
        case CALIB_SUCCESS:
            return "Success";
        case CALIB_ERROR_INVALID_PARAM:
            return "Invalid parameter";
        case CALIB_ERROR_FILE_NOT_FOUND:
            return "File not found";
        case CALIB_ERROR_FILE_READ:
            return "File read error";
        case CALIB_ERROR_FILE_WRITE:
            return "File write error";
        case CALIB_ERROR_INVALID_FORMAT:
            return "Invalid file format";
        case CALIB_ERROR_VERSION_MISMATCH:
            return "Version mismatch";
        case CALIB_ERROR_MEMORY_ALLOCATION:
            return "Memory allocation failed";
        case CALIB_ERROR_DATA_CORRUPTED:
            return "Data corrupted";
        case CALIB_ERROR_NOT_INITIALIZED:
            return "Not initialized";
        case CALIB_ERROR_ALREADY_INITIALIZED:
            return "Already initialized";
        case CALIB_ERROR_WAVELENGTH_NOT_FOUND:
            return "Wavelength not found";
        case CALIB_ERROR_CHANNEL_NOT_FOUND:
            return "Channel not found";
        case CALIB_ERROR_INVALID_PSF:
            return "Invalid PSF data";
        case CALIB_ERROR_DIMENSION_MISMATCH:
            return "Dimension mismatch";
        case CALIB_ERROR_CHECKSUM_FAILED:
            return "Checksum verification failed";
        default:
            return "Unknown error";
    }
}

// ============================================================================
// 初始化和清理函数
// ============================================================================

/**
 * @brief 创建校准管理器
 */
CalibrationManager* calib_manager_create(void) {
    CalibrationManager *manager = (CalibrationManager*)calloc(1, sizeof(CalibrationManager));
    if (!manager) {
        fprintf(stderr, "[CalibManager] Failed to allocate manager\n");
        return NULL;
    }
    
    manager->initialized = false;
    manager->verbose = false;
    manager->current_data = NULL;
    manager->total_loads = 0;
    manager->total_saves = 0;
    manager->failed_operations = 0;
    manager->cache_size = 0;
    memset(manager->psf_cache, 0, sizeof(manager->psf_cache));
    memset(manager->last_error, 0, sizeof(manager->last_error));
    
    if (g_verbose) {
        printf("[CalibManager] Manager created\n");
    }
    
    return manager;
}

/**
 * @brief 初始化校准管理器
 */
CalibrationError calib_manager_init(CalibrationManager *manager) {
    if (!manager) {
        return CALIB_ERROR_INVALID_PARAM;
    }
    
    if (manager->initialized) {
        set_error(manager, "Manager already initialized");
        return CALIB_ERROR_ALREADY_INITIALIZED;
    }
    
    // 初始化缓存
    manager->cache_size = 0;
    memset(manager->psf_cache, 0, sizeof(manager->psf_cache));
    
    manager->initialized = true;
    
    if (manager->verbose) {
        printf("[CalibManager] Manager initialized\n");
    }
    
    return CALIB_SUCCESS;
}

/**
 * @brief 清理校准管理器资源
 */
void calib_manager_cleanup(CalibrationManager *manager) {
    if (!manager) {
        return;
    }
    
    // 清理当前数据
    if (manager->current_data) {
        calib_data_destroy(manager->current_data);
        manager->current_data = NULL;
    }
    
    // 清理PSF缓存
    for (int i = 0; i < manager->cache_size; i++) {
        if (manager->psf_cache[i]) {
            psf_destroy(manager->psf_cache[i]);
            manager->psf_cache[i] = NULL;
        }
    }
    manager->cache_size = 0;
    
    manager->initialized = false;
    
    if (manager->verbose) {
        printf("[CalibManager] Manager cleaned up\n");
        printf("  Total loads: %zu\n", manager->total_loads);
        printf("  Total saves: %zu\n", manager->total_saves);
        printf("  Failed operations: %zu\n", manager->failed_operations);
    }
}

/**
 * @brief 销毁校准管理器
 */
void calib_manager_destroy(CalibrationManager *manager) {
    if (!manager) {
        return;
    }
    
    calib_manager_cleanup(manager);
    free(manager);
    
    if (g_verbose) {
        printf("[CalibManager] Manager destroyed\n");
    }
}

// ============================================================================
// 校准数据创建和销毁
// ============================================================================

/**
 * @brief 创建空的校准数据结构
 */
CalibrationData* calib_data_create(int num_channels) {
    if (num_channels <= 0 || num_channels > CALIB_MAX_CHANNELS) {
        fprintf(stderr, "[CalibManager] Invalid number of channels: %d\n", num_channels);
        return NULL;
    }
    
    CalibrationData *data = (CalibrationData*)calloc(1, sizeof(CalibrationData));
    if (!data) {
        fprintf(stderr, "[CalibManager] Failed to allocate calibration data\n");
        return NULL;
    }
    
    // 初始化元数据
    memset(&data->metadata, 0, sizeof(CalibrationMetadata));
    data->metadata.type = CALIB_TYPE_UNKNOWN;
    data->metadata.version_major = CALIBRATION_MANAGER_VERSION_MAJOR;
    data->metadata.version_minor = CALIBRATION_MANAGER_VERSION_MINOR;
    
    // 获取当前时间
    time_t now = time(NULL);
    struct tm *tm_info = localtime(&now);
    strftime(data->metadata.creation_date, 32, "%Y-%m-%d %H:%M:%S", tm_info);
    
    // 分配通道数组
    data->num_channels = num_channels;
    data->channels = (ChannelCalibration*)calloc(num_channels, sizeof(ChannelCalibration));
    if (!data->channels) {
        fprintf(stderr, "[CalibManager] Failed to allocate channels\n");
        free(data);
        return NULL;
    }
    
    // 初始化每个通道
    for (int i = 0; i < num_channels; i++) {
        data->channels[i].channel_index = i;
        data->channels[i].num_wavelengths = 0;
        data->channels[i].wavelengths = NULL;
        data->channels[i].has_color_matrix = false;
        memset(data->channels[i].spectral_response, 0, sizeof(float) * 256);
        memset(data->channels[i].color_matrix, 0, sizeof(float) * 9);
        snprintf(data->channels[i].name, 64, "Channel_%d", i);
    }
    
    // 初始化全局参数
    data->sensor_width = 0.0f;
    data->sensor_height = 0.0f;
    data->pixel_pitch = 0.0f;
    data->image_width = 0;
    data->image_height = 0;
    data->focal_length = 0.0f;
    data->f_number = 0.0f;
    data->working_distance = 0.0f;
    
    data->is_valid = false;
    data->internal_data = NULL;
    
    if (g_verbose) {
        printf("[CalibManager] Created calibration data with %d channels\n", num_channels);
    }
    
    return data;
}

/**
 * @brief 销毁校准数据
 */
void calib_data_destroy(CalibrationData *data) {
    if (!data) {
        return;
    }
    
    // 销毁所有通道
    if (data->channels) {
        for (int i = 0; i < data->num_channels; i++) {
            ChannelCalibration *channel = &data->channels[i];
            
            // 销毁波长数据
            if (channel->wavelengths) {
                for (int j = 0; j < channel->num_wavelengths; j++) {
                    WavelengthCalibration *wl = &channel->wavelengths[j];
                    
                    // 销毁PSF数据
                    if (wl->psf.data) {
                        free(wl->psf.data);
                        wl->psf.data = NULL;
                    }
                    
                    // 销毁自定义数据
                    if (wl->custom_data) {
                        free(wl->custom_data);
                        wl->custom_data = NULL;
                    }
                }
                free(channel->wavelengths);
                channel->wavelengths = NULL;
            }
        }
        free(data->channels);
        data->channels = NULL;
    }
    
    // 销毁内部数据
    if (data->internal_data) {
        free(data->internal_data);
        data->internal_data = NULL;
    }
    
    free(data);
    
    if (g_verbose) {
        printf("[CalibManager] Calibration data destroyed\n");
    }
}

/**
 * @brief 复制校准数据
 */
CalibrationData* calib_data_clone(const CalibrationData *src) {
    if (!src) {
        return NULL;
    }
    
    CalibrationData *dst = calib_data_create(src->num_channels);
    if (!dst) {
        return NULL;
    }
    
    // 复制元数据
    memcpy(&dst->metadata, &src->metadata, sizeof(CalibrationMetadata));
    
    // 复制全局参数
    dst->sensor_width = src->sensor_width;
    dst->sensor_height = src->sensor_height;
    dst->pixel_pitch = src->pixel_pitch;
    dst->image_width = src->image_width;
    dst->image_height = src->image_height;
    dst->focal_length = src->focal_length;
    dst->f_number = src->f_number;
    dst->working_distance = src->working_distance;
    dst->is_valid = src->is_valid;
    
    // 复制每个通道
    for (int i = 0; i < src->num_channels; i++) {
        const ChannelCalibration *src_channel = &src->channels[i];
        ChannelCalibration *dst_channel = &dst->channels[i];
        
        // 复制通道基本信息
        strncpy(dst_channel->name, src_channel->name, 64);
        dst_channel->channel_index = src_channel->channel_index;
        dst_channel->has_color_matrix = src_channel->has_color_matrix;
        memcpy(dst_channel->spectral_response, src_channel->spectral_response, sizeof(float) * 256);
        memcpy(dst_channel->color_matrix, src_channel->color_matrix, sizeof(float) * 9);
        
        // 复制波长数据
        dst_channel->num_wavelengths = src_channel->num_wavelengths;
        if (src_channel->num_wavelengths > 0) {
            dst_channel->wavelengths = (WavelengthCalibration*)calloc(
                src_channel->num_wavelengths, sizeof(WavelengthCalibration));
            
            if (!dst_channel->wavelengths) {
                calib_data_destroy(dst);
                return NULL;
            }
            
            for (int j = 0; j < src_channel->num_wavelengths; j++) {
                const WavelengthCalibration *src_wl = &src_channel->wavelengths[j];
                WavelengthCalibration *dst_wl = &dst_channel->wavelengths[j];
                
                dst_wl->wavelength = src_wl->wavelength;
                dst_wl->diffraction_limit = src_wl->diffraction_limit;
                memcpy(dst_wl->aberration_coeff, src_wl->aberration_coeff, sizeof(float) * 10);
                
                // 复制PSF
                dst_wl->psf.width = src_wl->psf.width;
                dst_wl->psf.height = src_wl->psf.height;
                dst_wl->psf.center_x = src_wl->psf.center_x;
                dst_wl->psf.center_y = src_wl->psf.center_y;
                dst_wl->psf.wavelength = src_wl->psf.wavelength;
                dst_wl->psf.numerical_aperture = src_wl->psf.numerical_aperture;
                dst_wl->psf.pixel_size = src_wl->psf.pixel_size;
                dst_wl->psf.is_normalized = src_wl->psf.is_normalized;
                
                if (src_wl->psf.data) {
                    size_t psf_size = src_wl->psf.width * src_wl->psf.height;
                    dst_wl->psf.data = (float*)malloc(psf_size * sizeof(float));
                    if (dst_wl->psf.data) {
                        memcpy(dst_wl->psf.data, src_wl->psf.data, psf_size * sizeof(float));
                    }
                }
                
                // 复制自定义数据
                if (src_wl->custom_data && src_wl->custom_data_size > 0) {
                    dst_wl->custom_data = malloc(src_wl->custom_data_size);
                    if (dst_wl->custom_data) {
                        memcpy(dst_wl->custom_data, src_wl->custom_data, src_wl->custom_data_size);
                        dst_wl->custom_data_size = src_wl->custom_data_size;
                    }
                }
            }
        }
    }
    
    if (g_verbose) {
        printf("[CalibManager] Calibration data cloned\n");
    }
    
    return dst;
}

// ============================================================================
// PSF操作函数
// ============================================================================

/**
 * @brief 创建PSF数据
 */
PSFData* psf_create(int width, int height) {
    if (width <= 0 || height <= 0 || width > 4096 || height > 4096) {
        fprintf(stderr, "[CalibManager] Invalid PSF dimensions: %dx%d\n", width, height);
        return NULL;
    }
    
    PSFData *psf = (PSFData*)calloc(1, sizeof(PSFData));
    if (!psf) {
        fprintf(stderr, "[CalibManager] Failed to allocate PSF\n");
        return NULL;
    }
    
    psf->width = width;
    psf->height = height;
    psf->data = (float*)calloc(width * height, sizeof(float));
    
    if (!psf->data) {
        fprintf(stderr, "[CalibManager] Failed to allocate PSF data\n");
        free(psf);
        return NULL;
    }
    
    psf->center_x = width / 2.0f;
    psf->center_y = height / 2.0f;
    psf->wavelength = 0.0f;
    psf->numerical_aperture = 0.0f;
    psf->pixel_size = 0.0f;
    psf->is_normalized = false;
    
    if (g_verbose) {
        printf("[CalibManager] Created PSF: %dx%d\n", width, height);
    }
    
    return psf;
}

/**
 * @brief 销毁PSF数据
 */
void psf_destroy(PSFData *psf) {
    if (!psf) {
        return;
    }
    
    if (psf->data) {
        free(psf->data);
        psf->data = NULL;
    }
    
    free(psf);
}

/**
 * @brief 复制PSF数据
 */
PSFData* psf_clone(const PSFData *src) {
    if (!src) {
        return NULL;
    }
    
    PSFData *dst = psf_create(src->width, src->height);
    if (!dst) {
        return NULL;
    }
    
    dst->center_x = src->center_x;
    dst->center_y = src->center_y;
    dst->wavelength = src->wavelength;
    dst->numerical_aperture = src->numerical_aperture;
    dst->pixel_size = src->pixel_size;
    dst->is_normalized = src->is_normalized;
    
    size_t data_size = src->width * src->height * sizeof(float);
    memcpy(dst->data, src->data, data_size);
    
    return dst;
}

/**
 * @brief 从数组加载PSF数据
 */
CalibrationError psf_load_from_array(PSFData *psf, 
                                     const float *data,
                                     int width, 
                                     int height) {
    if (!psf || !data || width <= 0 || height <= 0) {
        return CALIB_ERROR_INVALID_PARAM;
    }
    
    if (psf->width != width || psf->height != height) {
        return CALIB_ERROR_DIMENSION_MISMATCH;
    }
    /**
 * @file calibration_manager.c
 * @brief 校准数据管理模块实现 - 第二部分
 * @details PSF归一化、生成、验证和数学函数
 */

// ============================================================================
// PSF归一化和验证
// ============================================================================

/**
 * @brief 归一化PSF数据
 */
CalibrationError psf_normalize(PSFData *psf) {
    if (!psf || !psf->data) {
        return CALIB_ERROR_INVALID_PARAM;
    }
    
    size_t total_pixels = psf->width * psf->height;
    
    // 计算总和
    double sum = 0.0;
    for (size_t i = 0; i < total_pixels; i++) {
        sum += psf->data[i];
    }
    
    if (sum <= 0.0) {
        fprintf(stderr, "[CalibManager] PSF sum is zero or negative: %f\n", sum);
        return CALIB_ERROR_INVALID_PSF;
    }
    
    // 归一化
    float scale = 1.0f / (float)sum;
    for (size_t i = 0; i < total_pixels; i++) {
        psf->data[i] *= scale;
    }
    
    psf->is_normalized = true;
    
    if (g_verbose) {
        printf("[CalibManager] PSF normalized (sum was: %f)\n", sum);
    }
    
    return CALIB_SUCCESS;
}

// ============================================================================
// 数学辅助函数
// ============================================================================

/**
 * @brief 计算第一类贝塞尔函数 J1(x)
 * @details 使用级数展开近似计算
 */
static float bessel_j1(float x) {
    if (x == 0.0f) {
        return 0.0f;
    }
    
    float ax = fabsf(x);
    
    if (ax < 8.0f) {
        // 小参数：使用级数展开
        float y = x * x;
        float ans = x * (72362614232.0f + y * (-7895059235.0f + 
                    y * (242396853.1f + y * (-2972611.439f + 
                    y * (15704.48260f + y * (-30.16036606f))))));
        ans = ans / (144725228442.0f + y * (2300535178.0f + 
              y * (18583304.74f + y * (99447.43394f + 
              y * (376.9991397f + y)))));
        return ans;
    } else {
        // 大参数：使用渐近展开
        float z = 8.0f / ax;
        float y = z * z;
        float xx = ax - 2.356194491f;
        
        float ans1 = 1.0f + y * (0.183105e-2f + y * (-0.3516396496e-4f + 
                     y * (0.2457520174e-5f + y * (-0.240337019e-6f))));
        float ans2 = 0.04687499995f + y * (-0.2002690873e-3f + 
                     y * (0.8449199096e-5f + y * (-0.88228987e-6f + 
                     y * 0.105787412e-6f)));
        
        float ans = sqrtf(0.636619772f / ax) * 
                   (cosf(xx) * ans1 - z * sinf(xx) * ans2);
        
        return (x < 0.0f) ? -ans : ans;
    }
}

/**
 * @brief 计算Airy函数值
 * @param r 归一化半径
 * @return Airy函数值
 */
static float airy_function(float r) {
    if (r < 1e-6f) {
        return 1.0f;
    }
    
    float j1 = bessel_j1(r);
    float val = 2.0f * j1 / r;
    return val * val;
}

// ============================================================================
// PSF生成函数
// ============================================================================

/**
 * @brief 生成理论PSF（Airy盘）
 */
PSFData* psf_generate_airy(float wavelength,
                           float numerical_aperture,
                           float pixel_size,
                           int size) {
    if (wavelength <= 0.0f || numerical_aperture <= 0.0f || 
        pixel_size <= 0.0f || size <= 0 || size > 1024) {
        fprintf(stderr, "[CalibManager] Invalid Airy PSF parameters\n");
        return NULL;
    }
    
    PSFData *psf = psf_create(size, size);
    if (!psf) {
        return NULL;
    }
    
    psf->wavelength = wavelength;
    psf->numerical_aperture = numerical_aperture;
    psf->pixel_size = pixel_size;
    
    // 计算Airy盘参数
    // 第一暗环半径 r = 1.22 * lambda / (2 * NA)
    float lambda_um = wavelength / 1000.0f;  // 转换为微米
    float airy_radius = 1.22f * lambda_um / (2.0f * numerical_aperture);
    
    // 归一化因子
    float k = M_PI * numerical_aperture / lambda_um;
    
    int center_x = size / 2;
    int center_y = size / 2;
    
    // 生成Airy盘
    for (int y = 0; y < size; y++) {
        for (int x = 0; x < size; x++) {
            float dx = (x - center_x) * pixel_size;
            float dy = (y - center_y) * pixel_size;
            float r = sqrtf(dx * dx + dy * dy);
            
            // 计算归一化半径
            float rho = k * r;
            
            // 计算Airy函数值
            float intensity = airy_function(rho);
            
            psf->data[y * size + x] = intensity;
        }
    }
    
    // 归一化PSF
    psf_normalize(psf);
    
    if (g_verbose) {
        printf("[CalibManager] Generated Airy PSF:\n");
        printf("  Size: %dx%d\n", size, size);
        printf("  Wavelength: %.1f nm\n", wavelength);
        printf("  NA: %.3f\n", numerical_aperture);
        printf("  Pixel size: %.3f um\n", pixel_size);
        printf("  Airy radius: %.3f um\n", airy_radius);
    }
    
    return psf;
}

// ============================================================================
// 校准数据验证
// ============================================================================

/**
 * @brief 验证PSF数据的有效性
 */
static bool validate_psf(const PSFData *psf) {
    if (!psf || !psf->data) {
        return false;
    }
    
    if (psf->width <= 0 || psf->height <= 0 || 
        psf->width > 4096 || psf->height > 4096) {
        return false;
    }
    
    // 检查是否有非法值
    size_t total_pixels = psf->width * psf->height;
    double sum = 0.0;
    
    for (size_t i = 0; i < total_pixels; i++) {
        float val = psf->data[i];
        
        if (isnan(val) || isinf(val) || val < 0.0f) {
            return false;
        }
        
        sum += val;
    }
    
    // 检查总和是否合理
    if (sum <= 0.0 || sum > 1e6) {
        return false;
    }
    
    return true;
}

/**
 * @brief 验证波长校准数据
 */
static bool validate_wavelength_calibration(const WavelengthCalibration *wl) {
    if (!wl) {
        return false;
    }
    
    // 检查波长范围（可见光+近红外：300-1100 nm）
    if (wl->wavelength < 300.0f || wl->wavelength > 1100.0f) {
        return false;
    }
    
    // 验证PSF
    if (!validate_psf(&wl->psf)) {
        return false;
    }
    
    // 检查衍射极限
    if (wl->diffraction_limit <= 0.0f || wl->diffraction_limit > 10000.0f) {
        return false;
    }
    
    return true;
}

/**
 * @brief 验证通道校准数据
 */
static bool validate_channel_calibration(const ChannelCalibration *channel) {
    if (!channel) {
        return false;
    }
    
    if (channel->num_wavelengths <= 0 || 
        channel->num_wavelengths > CALIB_MAX_WAVELENGTHS) {
        return false;
    }
    
    if (!channel->wavelengths) {
        return false;
    }
    
    // 验证每个波长
    for (int i = 0; i < channel->num_wavelengths; i++) {
        if (!validate_wavelength_calibration(&channel->wavelengths[i])) {
            return false;
        }
    }
    
    // 如果有颜色矩阵，验证其有效性
    if (channel->has_color_matrix) {
        bool all_zero = true;
        for (int i = 0; i < 9; i++) {
            if (isnan(channel->color_matrix[i]) || 
                isinf(channel->color_matrix[i])) {
                return false;
            }
            if (channel->color_matrix[i] != 0.0f) {
                all_zero = false;
            }
        }
        if (all_zero) {
            return false;
        }
    }
    
    return true;
}

/**
 * @brief 验证校准数据的完整性
 */
bool calib_data_validate(const CalibrationData *data) {
    if (!data) {
        return false;
    }
    
    // 检查通道数量
    if (data->num_channels <= 0 || data->num_channels > CALIB_MAX_CHANNELS) {
        if (g_verbose) {
            fprintf(stderr, "[CalibManager] Invalid number of channels: %d\n", 
                    data->num_channels);
        }
        return false;
    }
    
    if (!data->channels) {
        if (g_verbose) {
            fprintf(stderr, "[CalibManager] Channels array is NULL\n");
        }
        return false;
    }
    
    // 验证每个通道
    for (int i = 0; i < data->num_channels; i++) {
        if (!validate_channel_calibration(&data->channels[i])) {
            if (g_verbose) {
                fprintf(stderr, "[CalibManager] Channel %d validation failed\n", i);
            }
            return false;
        }
    }
    
    // 验证全局参数
    if (data->pixel_pitch <= 0.0f || data->pixel_pitch > 100.0f) {
        if (g_verbose) {
            fprintf(stderr, "[CalibManager] Invalid pixel pitch: %f\n", 
                    data->pixel_pitch);
        }
        return false;
    }
    
    if (data->image_width <= 0 || data->image_height <= 0 ||
        data->image_width > 65536 || data->image_height > 65536) {
        if (g_verbose) {
            fprintf(stderr, "[CalibManager] Invalid image dimensions: %dx%d\n",
                    data->image_width, data->image_height);
        }
        return false;
    }
    
    // 验证光学参数
    if (data->focal_length > 0.0f) {
        if (data->focal_length < 1.0f || data->focal_length > 10000.0f) {
            if (g_verbose) {
                fprintf(stderr, "[CalibManager] Invalid focal length: %f\n",
                        data->focal_length);
            }
            return false;
        }
    }
    
    if (data->f_number > 0.0f) {
        if (data->f_number < 0.5f || data->f_number > 64.0f) {
            if (g_verbose) {
                fprintf(stderr, "[CalibManager] Invalid f-number: %f\n",
                        data->f_number);
            }
            return false;
        }
    }
    
    if (g_verbose) {
        printf("[CalibManager] Calibration data validation passed\n");
    }
    
    return true;
}

// ============================================================================
// 通道和波长操作
// ============================================================================

/**
 * @brief 添加通道校准数据
 */
CalibrationError calib_add_channel(CalibrationData *data,
                                   const char *channel_name,
                                   int channel_index) {
    if (!data || !channel_name) {
        return CALIB_ERROR_INVALID_PARAM;
    }
    
    if (channel_index < 0 || channel_index >= data->num_channels) {
        return CALIB_ERROR_INVALID_PARAM;
    }
    
    ChannelCalibration *channel = &data->channels[channel_index];
    
    // 设置通道名称
    strncpy(channel->name, channel_name, 63);
    channel->name[63] = '\0';
    channel->channel_index = channel_index;
    
    if (g_verbose) {
        printf("[CalibManager] Added channel: %s (index %d)\n", 
               channel_name, channel_index);
    }
    
    return CALIB_SUCCESS;
}

/**
 * @brief 获取通道校准数据
 */
ChannelCalibration* calib_get_channel(const CalibrationData *data,
                                      int channel_index) {
    if (!data || !data->channels) {
        return NULL;
    }
    
    if (channel_index < 0 || channel_index >= data->num_channels) {
        return NULL;
    }
    
    return &data->channels[channel_index];
}

/**
 * @brief 添加波长校准数据到通道
 */
CalibrationError calib_add_wavelength(ChannelCalibration *channel,
                                      float wavelength,
                                      const PSFData *psf) {
    if (!channel || !psf) {
        return CALIB_ERROR_INVALID_PARAM;
    }
    
    if (wavelength < 300.0f || wavelength > 1100.0f) {
        return CALIB_ERROR_INVALID_PARAM;
    }
    
    if (!validate_psf(psf)) {
        return CALIB_ERROR_INVALID_PSF;
    }
    
    // 检查是否已存在该波长
    for (int i = 0; i < channel->num_wavelengths; i++) {
        if (fabsf(channel->wavelengths[i].wavelength - wavelength) < 0.1f) {
            if (g_verbose) {
                fprintf(stderr, "[CalibManager] Wavelength %.1f nm already exists\n",
                        wavelength);
            }
            return CALIB_ERROR_INVALID_PARAM;
        }
    }
    
    // 扩展波长数组
    int new_count = channel->num_wavelengths + 1;
    WavelengthCalibration *new_wavelengths = (WavelengthCalibration*)realloc(
        channel->wavelengths,
        new_count * sizeof(WavelengthCalibration));
    
    if (!new_wavelengths) {
        return CALIB_ERROR_MEMORY_ALLOCATION;
    }
    
    channel->wavelengths = new_wavelengths;
    
    // 初始化新的波长数据
    WavelengthCalibration *wl = &channel->wavelengths[channel->num_wavelengths];
    memset(wl, 0, sizeof(WavelengthCalibration));
    
    wl->wavelength = wavelength;
    
    // 复制PSF数据
    wl->psf.width = psf->width;
    wl->psf.height = psf->height;
    wl->psf.center_x = psf->center_x;
    wl->psf.center_y = psf->center_y;
    wl->psf.wavelength = psf->wavelength;
    wl->psf.numerical_aperture = psf->numerical_aperture;
    wl->psf.pixel_size = psf->pixel_size;
    wl->psf.is_normalized = psf->is_normalized;
    
    size_t psf_size = psf->width * psf->height;
    wl->psf.data = (float*)malloc(psf_size * sizeof(float));
    if (!wl->psf.data) {
        return CALIB_ERROR_MEMORY_ALLOCATION;
    }
    memcpy(wl->psf.data, psf->data, psf_size * sizeof(float));
    
    // 计算衍射极限
    if (psf->numerical_aperture > 0.0f) {
        wl->diffraction_limit = 0.61f * wavelength / psf->numerical_aperture;
    } else {
        wl->diffraction_limit = wavelength / 2.0f;
    }
    
    channel->num_wavelengths = new_count;
    
    if (g_verbose) {
        printf("[CalibManager] Added wavelength %.1f nm to channel %s\n",
               wavelength, channel->name);
        printf("  Diffraction limit: %.1f nm\n", wl->diffraction_limit);
    }
    
    return CALIB_SUCCESS;
}

/**
 * @brief 查找最接近的波长索引
 */
static int find_nearest_wavelength_index(const ChannelCalibration *channel,
                                        float wavelength) {
    if (!channel || channel->num_wavelengths == 0) {
        return -1;
    }
    
    int nearest_idx = 0;
    float min_diff = fabsf(channel->wavelengths[0].wavelength - wavelength);
    
    for (int i = 1; i < channel->num_wavelengths; i++) {
        float diff = fabsf(channel->wavelengths[i].wavelength - wavelength);
        if (diff < min_diff) {
            min_diff = diff;
            nearest_idx = i;
        }
    }
    
    return nearest_idx;
}

/**
 * @brief 获取指定波长的校准数据（最近邻）
 */
WavelengthCalibration* calib_get_wavelength(const ChannelCalibration *channel,
                                            float wavelength,
                                            CalibrationInterpMethod method) {
    if (!channel || !channel->wavelengths) {
        return NULL;
    }
    
    if (channel->num_wavelengths == 0) {
        return NULL;
    }
    
    // 目前只实现最近邻方法
    if (method != CALIB_INTERP_NEAREST) {
        if (g_verbose) {
            fprintf(stderr, "[CalibManager] Only nearest neighbor interpolation is currently supported\n");
        }
    }
    
    int idx = find_nearest_wavelength_index(channel, wavelength);
    if (idx < 0) {
        return NULL;
    }
    
    return &channel->wavelengths[idx];
}

    size_t data_size = width * height * sizeof(float);
    memcpy(psf->data, data, data_size);
    
    psf->is_normalized = false;
    
    if (g_verbose) {
        printf("[CalibManager] Loaded PSF from array: %dx%d\n", width, height);
    }
    
    return CALIB_SUCCESS;
}
/**
 * @file calibration_manager.c
 * @brief 校准数据管理模块实现 - 第三部分
 * @details PSF插值、校验和计算、工具函数
 */

// ============================================================================
// PSF插值函数
// ============================================================================

/**
 * @brief 对PSF进行线性插值
 */
PSFData* psf_interpolate(const PSFData *psf1,
                        const PSFData *psf2,
                        float weight) {
    if (!psf1 || !psf2) {
        return NULL;
    }
    
    if (weight < 0.0f || weight > 1.0f) {
        fprintf(stderr, "[CalibManager] Invalid interpolation weight: %f\n", weight);
        return NULL;
    }
    
    // 检查PSF尺寸是否匹配
    if (psf1->width != psf2->width || psf1->height != psf2->height) {
        fprintf(stderr, "[CalibManager] PSF dimensions mismatch: %dx%d vs %dx%d\n",
                psf1->width, psf1->height, psf2->width, psf2->height);
        return CALIB_ERROR_DIMENSION_MISMATCH;
    }
    
    // 创建结果PSF
    PSFData *result = psf_create(psf1->width, psf1->height);
    if (!result) {
        return NULL;
    }
    
    // 插值参数
    result->center_x = psf1->center_x * (1.0f - weight) + psf2->center_x * weight;
    result->center_y = psf1->center_y * (1.0f - weight) + psf2->center_y * weight;
    result->wavelength = psf1->wavelength * (1.0f - weight) + psf2->wavelength * weight;
    result->numerical_aperture = psf1->numerical_aperture * (1.0f - weight) + 
                                 psf2->numerical_aperture * weight;
    result->pixel_size = psf1->pixel_size * (1.0f - weight) + psf2->pixel_size * weight;
    
    // 插值数据
    size_t total_pixels = psf1->width * psf1->height;
    float w1 = 1.0f - weight;
    float w2 = weight;
    
    for (size_t i = 0; i < total_pixels; i++) {
        result->data[i] = psf1->data[i] * w1 + psf2->data[i] * w2;
    }
    
    // 归一化结果
    psf_normalize(result);
    
    if (g_verbose) {
        printf("[CalibManager] Interpolated PSF with weight %.3f\n", weight);
        printf("  Wavelength: %.1f nm -> %.1f nm -> %.1f nm\n",
               psf1->wavelength, result->wavelength, psf2->wavelength);
    }
    
    return result;
}

/**
 * @brief 对波长校准数据进行插值
 */
static WavelengthCalibration* wavelength_interpolate(
    const WavelengthCalibration *wl1,
    const WavelengthCalibration *wl2,
    float weight) {
    
    if (!wl1 || !wl2) {
        return NULL;
    }
    
    WavelengthCalibration *result = (WavelengthCalibration*)calloc(
        1, sizeof(WavelengthCalibration));
    if (!result) {
        return NULL;
    }
    
    // 插值波长
    result->wavelength = wl1->wavelength * (1.0f - weight) + wl2->wavelength * weight;
    result->diffraction_limit = wl1->diffraction_limit * (1.0f - weight) + 
                                wl2->diffraction_limit * weight;
    
    // 插值像差系数
    for (int i = 0; i < 10; i++) {
        result->aberration_coeff[i] = wl1->aberration_coeff[i] * (1.0f - weight) +
                                      wl2->aberration_coeff[i] * weight;
    }
    
    // 插值PSF
    PSFData *interp_psf = psf_interpolate(&wl1->psf, &wl2->psf, weight);
    if (!interp_psf) {
        free(result);
        return NULL;
    }
    
    result->psf = *interp_psf;
    free(interp_psf);  // 只释放结构体，不释放data
    
    result->custom_data = NULL;
    result->custom_data_size = 0;
    
    return result;
}

// ============================================================================
// 校验和计算
// ============================================================================

/**
 * @brief CRC32查找表
 */
static uint32_t crc32_table[256];
static bool crc32_table_initialized = false;

/**
 * @brief 初始化CRC32查找表
 */
static void init_crc32_table(void) {
    if (crc32_table_initialized) {
        return;
    }
    
    for (uint32_t i = 0; i < 256; i++) {
        uint32_t crc = i;
        for (int j = 0; j < 8; j++) {
            if (crc & 1) {
                crc = (crc >> 1) ^ 0xEDB88320;
            } else {
                crc >>= 1;
            }
        }
        crc32_table[i] = crc;
    }
    
    crc32_table_initialized = true;
}

/**
 * @brief 计算CRC32校验和
 */
static uint32_t calculate_crc32(const uint8_t *data, size_t length) {
    init_crc32_table();
    
    uint32_t crc = 0xFFFFFFFF;
    
    for (size_t i = 0; i < length; i++) {
        uint8_t index = (crc ^ data[i]) & 0xFF;
        crc = (crc >> 8) ^ crc32_table[index];
    }
    
    return crc ^ 0xFFFFFFFF;
}

/**
 * @brief 计算PSF数据的校验和
 */
static uint32_t calculate_psf_checksum(const PSFData *psf) {
    if (!psf || !psf->data) {
        return 0;
    }
    
    size_t data_size = psf->width * psf->height * sizeof(float);
    uint32_t crc = calculate_crc32((const uint8_t*)psf->data, data_size);
    
    // 包含其他参数
    crc ^= calculate_crc32((const uint8_t*)&psf->width, sizeof(int));
    crc ^= calculate_crc32((const uint8_t*)&psf->height, sizeof(int));
    crc ^= calculate_crc32((const uint8_t*)&psf->wavelength, sizeof(float));
    
    return crc;
}

/**
 * @brief 计算波长校准数据的校验和
 */
static uint32_t calculate_wavelength_checksum(const WavelengthCalibration *wl) {
    if (!wl) {
        return 0;
    }
    
    uint32_t crc = 0;
    
    // PSF校验和
    crc ^= calculate_psf_checksum(&wl->psf);
    
    // 波长和衍射极限
    crc ^= calculate_crc32((const uint8_t*)&wl->wavelength, sizeof(float));
    crc ^= calculate_crc32((const uint8_t*)&wl->diffraction_limit, sizeof(float));
    
    // 像差系数
    crc ^= calculate_crc32((const uint8_t*)wl->aberration_coeff, sizeof(float) * 10);
    
    // 自定义数据
    if (wl->custom_data && wl->custom_data_size > 0) {
        crc ^= calculate_crc32((const uint8_t*)wl->custom_data, wl->custom_data_size);
    }
    
    return crc;
}

/**
 * @brief 计算通道校准数据的校验和
 */
static uint32_t calculate_channel_checksum(const ChannelCalibration *channel) {
    if (!channel) {
        return 0;
    }
    
    uint32_t crc = 0;
    
    // 通道名称和索引
    crc ^= calculate_crc32((const uint8_t*)channel->name, strlen(channel->name));
    crc ^= calculate_crc32((const uint8_t*)&channel->channel_index, sizeof(int));
    
    // 所有波长数据
    for (int i = 0; i < channel->num_wavelengths; i++) {
        crc ^= calculate_wavelength_checksum(&channel->wavelengths[i]);
    }
    
    // 光谱响应
    crc ^= calculate_crc32((const uint8_t*)channel->spectral_response, 
                          sizeof(float) * 256);
    
    // 颜色矩阵
    if (channel->has_color_matrix) {
        crc ^= calculate_crc32((const uint8_t*)channel->color_matrix, 
                              sizeof(float) * 9);
    }
    
    return crc;
}

/**
 * @brief 计算校准数据的校验和
 */
uint32_t calib_calculate_checksum(const CalibrationData *data) {
    if (!data) {
        return 0;
    }
    
    uint32_t crc = 0;
    
    // 元数据（排除checksum字段本身）
    crc ^= calculate_crc32((const uint8_t*)data->metadata.name, 
                          strlen(data->metadata.name));
    crc ^= calculate_crc32((const uint8_t*)data->metadata.description,
                          strlen(data->metadata.description));
    crc ^= calculate_crc32((const uint8_t*)&data->metadata.type, sizeof(CalibrationType));
    
    // 所有通道数据
    for (int i = 0; i < data->num_channels; i++) {
        crc ^= calculate_channel_checksum(&data->channels[i]);
    }
    
    // 全局参数
    crc ^= calculate_crc32((const uint8_t*)&data->sensor_width, sizeof(float));
    crc ^= calculate_crc32((const uint8_t*)&data->sensor_height, sizeof(float));
    crc ^= calculate_crc32((const uint8_t*)&data->pixel_pitch, sizeof(float));
    crc ^= calculate_crc32((const uint8_t*)&data->image_width, sizeof(int));
    crc ^= calculate_crc32((const uint8_t*)&data->image_height, sizeof(int));
    crc ^= calculate_crc32((const uint8_t*)&data->focal_length, sizeof(float));
    crc ^= calculate_crc32((const uint8_t*)&data->f_number, sizeof(float));
    crc ^= calculate_crc32((const uint8_t*)&data->working_distance, sizeof(float));
    
    if (g_verbose) {
        printf("[CalibManager] Calculated checksum: 0x%08X\n", crc);
    }
    
    return crc;
}

/**
 * @brief 验证校准数据的校验和
 */
bool calib_verify_checksum(const CalibrationData *data) {
    if (!data) {
        return false;
    }
    
    uint32_t calculated = calib_calculate_checksum(data);
    uint32_t stored = data->metadata.checksum;
    
    bool valid = (calculated == stored);
    
    if (g_verbose) {
        printf("[CalibManager] Checksum verification: %s\n", 
               valid ? "PASSED" : "FAILED");
        printf("  Calculated: 0x%08X\n", calculated);
        printf("  Stored:     0x%08X\n", stored);
    }
    
    return valid;
}

// ============================================================================
// PSF缓存管理
// ============================================================================

/**
 * @brief 添加PSF到缓存
 */
static void psf_cache_add(CalibrationManager *manager, PSFData *psf) {
    if (!manager || !psf) {
        return;
    }
    
    // 如果缓存已满，移除最旧的
    if (manager->cache_size >= 16) {
        if (manager->psf_cache[0]) {
            psf_destroy(manager->psf_cache[0]);
        }
        
        // 移动缓存
        for (int i = 0; i < 15; i++) {
            manager->psf_cache[i] = manager->psf_cache[i + 1];
        }
        manager->cache_size = 15;
    }
    
    // 添加新PSF
    manager->psf_cache[manager->cache_size] = psf_clone(psf);
    manager->cache_size++;
    
    if (manager->verbose) {
        printf("[CalibManager] Added PSF to cache (size: %d)\n", manager->cache_size);
    }
}

/**
 * @brief 在缓存中查找PSF
 */
static PSFData* psf_cache_find(CalibrationManager *manager, 
                               float wavelength, 
                               float na) {
    if (!manager) {
        return NULL;
    }
    
    const float wavelength_tolerance = 1.0f;  // 1 nm
    const float na_tolerance = 0.01f;
    
    for (int i = 0; i < manager->cache_size; i++) {
        PSFData *psf = manager->psf_cache[i];
        if (psf) {
            if (fabsf(psf->wavelength - wavelength) < wavelength_tolerance &&
                fabsf(psf->numerical_aperture - na) < na_tolerance) {
                
                if (manager->verbose) {
                    printf("[CalibManager] Found PSF in cache\n");
                }
                return psf;
            }
        }
    }
    
    return NULL;
}

// ============================================================================
// 信息打印函数
// ============================================================================

/**
 * @brief 打印PSF信息
 */
static void print_psf_info(const PSFData *psf, int indent) {
    if (!psf) {
        return;
    }
    
    const char *indent_str = "";
    if (indent == 1) indent_str = "  ";
    else if (indent == 2) indent_str = "    ";
    else if (indent == 3) indent_str = "      ";
    
    printf("%sPSF: %dx%d pixels\n", indent_str, psf->width, psf->height);
    printf("%s  Center: (%.2f, %.2f)\n", indent_str, psf->center_x, psf->center_y);
    printf("%s  Wavelength: %.1f nm\n", indent_str, psf->wavelength);
    printf("%s  NA: %.3f\n", indent_str, psf->numerical_aperture);
    printf("%s  Pixel size: %.3f µm\n", indent_str, psf->pixel_size);
    printf("%s  Normalized: %s\n", indent_str, psf->is_normalized ? "Yes" : "No");
    
    // 计算PSF统计信息
    if (psf->data) {
        size_t total = psf->width * psf->height;
        float sum = 0.0f, max_val = psf->data[0], min_val = psf->data[0];
        
        for (size_t i = 0; i < total; i++) {
            float val = psf->data[i];
            sum += val;
            if (val > max_val) max_val = val;
            if (val < min_val) min_val = val;
        }
        
        printf("%s  Sum: %.6f, Min: %.6f, Max: %.6f\n", 
               indent_str, sum, min_val, max_val);
    }
}

/**
 * @brief 打印波长校准信息
 */
static void print_wavelength_info(const WavelengthCalibration *wl, int indent) {
    if (!wl) {
        return;
    }
    
    const char *indent_str = "";
    if (indent == 1) indent_str = "  ";
    else if (indent == 2) indent_str = "    ";
    
    printf("%sWavelength: %.1f nm\n", indent_str, wl->wavelength);
    printf("%s  Diffraction limit: %.1f nm\n", indent_str, wl->diffraction_limit);
    
    // 打印像差系数（如果非零）
    bool has_aberration = false;
    for (int i = 0; i < 10; i++) {
        if (fabsf(wl->aberration_coeff[i]) > 1e-6f) {
            has_aberration = true;
            break;
        }
    }
    
    if (has_aberration) {
        printf("%s  Aberration coefficients:\n", indent_str);
        for (int i = 0; i < 10; i++) {
            if (fabsf(wl->aberration_coeff[i]) > 1e-6f) {
                printf("%s    Z%d: %.6f\n", indent_str, i, wl->aberration_coeff[i]);
            }
        }
    }
    
    print_psf_info(&wl->psf, indent + 1);
    
    if (wl->custom_data_size > 0) {
        printf("%s  Custom data: %zu bytes\n", indent_str, wl->custom_data_size);
    }
}

/**
 * @brief 打印通道校准信息
 */
static void print_channel_info(const ChannelCalibration *channel) {
    if (!channel) {
        return;
    }
    
    printf("  Channel: %s (index %d)\n", channel->name, channel->channel_index);
    printf("    Wavelengths: %d\n", channel->num_wavelengths);
    
    for (int i = 0; i < channel->num_wavelengths; i++) {
        printf("    [%d] ", i);
        print_wavelength_info(&channel->wavelengths[i], 2);
    }
    
    if (channel->has_color_matrix) {
        printf("    Color matrix:\n");
        for (int i = 0; i < 3; i++) {
            printf("      [%.4f %.4f %.4f]\n",
                   channel->color_matrix[i*3], 
                   channel->color_matrix[i*3+1],
                   channel->color_matrix[i*3+2]);
        }
    }
    
    // 检查光谱响应
    bool has_spectral = false;
    for (int i = 0; i < 256; i++) {
        if (channel->spectral_response[i] > 0.0f) {
            has_spectral = true;
            break;
        }
    }
    if (has_spectral) {
        printf("    Spectral response: defined\n");
    }
}

/**
 * @brief 打印校准数据信息
 */
void calib_print_info(const CalibrationData *data) {
    if (!data) {
        printf("Calibration data is NULL\n");
        return;
    }
    
    printf("=== Calibration Data ===\n");
    printf("Name: %s\n", data->metadata.name);
    printf("Description: %s\n", data->metadata.description);
    printf("Type: ");
    switch (data->metadata.type) {
        case CALIB_TYPE_MEASURED:
            printf("Measured\n");
            break;
        case CALIB_TYPE_DESIGN:
            printf("Design\n");
            break;
        case CALIB_TYPE_HYBRID:
            printf("Hybrid\n");
            break;
        case CALIB_TYPE_AUTO_GENERATED:
            printf("Auto-generated\n");
            break;
        default:
            printf("Unknown\n");
            break;
    }
    
    printf("Device: %s\n", data->metadata.device_model);
    printf("Lens: %s\n", data->metadata.lens_model);
    printf("Created: %s\n", data->metadata.creation_date);
    printf("Author: %s\n", data->metadata.author);
    printf("Version: %d.%d\n", data->metadata.version_major, 
           data->metadata.version_minor);
    printf("Checksum: 0x%08X\n", data->metadata.checksum);
    
    printf("\nSensor parameters:\n");
    printf("  Size: %.2f x %.2f mm\n", data->sensor_width, data->sensor_height);
    printf("  Pixel pitch: %.3f µm\n", data->pixel_pitch);
    printf("  Image size: %d x %d pixels\n", data->image_width, data->image_height);
    
    printf("\nOptical parameters:\n");
    printf("  Focal length: %.2f mm\n", data->focal_length);
    printf("  F-number: f/%.1f\n", data->f_number);
    printf("  Working distance: %.2f mm\n", data->working_distance);
    
    printf("\nChannels: %d\n", data->num_channels);
    for (int i = 0; i < data->num_channels; i++) {
        print_channel_info(&data->channels[i]);
    }
    
    printf("\nValid: %s\n", data->is_valid ? "Yes" : "No");
    printf("========================\n");
}

// ============================================================================
// 工具函数
// ============================================================================

/**
 * @brief 获取校准管理器版本
 */
const char* calib_manager_get_version(void) {
    static char version[32];
    snprintf(version, sizeof(version), "%d.%d.%d",
             CALIBRATION_MANAGER_VERSION_MAJOR,
             CALIBRATION_MANAGER_VERSION_MINOR,
             CALIBRATION_MANAGER_VERSION_PATCH);
    return version;
}

/**
 * @brief 设置详细输出模式
 */
void calib_set_verbose(bool verbose) {
    g_verbose = verbose;
    
    if (verbose) {
        printf("[CalibManager] Verbose mode enabled\n");
        printf("[CalibManager] Version: %s\n", calib_manager_get_version());
    }
}
/**
 * @file calibration_manager.c
 * @brief 校准数据管理模块实现 - 第四部分
 * @details 文件IO操作：保存和加载校准数据
 */

// ============================================================================
// 文件格式定义
// ============================================================================

/**
 * @brief 文件头魔数
 */
#define CALIB_FILE_MAGIC 0x43414C42  // "CALB"

/**
 * @brief 文件格式版本
 */
#define CALIB_FILE_VERSION 1

/**
 * @brief 文件头结构
 */
typedef struct {
    uint32_t magic;              ///< 魔数
    uint32_t version;            ///< 文件格式版本
    uint32_t header_size;        ///< 头部大小
    uint32_t data_size;          ///< 数据大小
    uint32_t checksum;           ///< 整个文件的校验和
    uint32_t flags;              ///< 标志位
    uint8_t reserved[40];        ///< 保留字段
} CalibFileHeader;

// ============================================================================
// 内部IO辅助函数
// ============================================================================

/**
 * @brief 写入字符串到文件
 */
static bool write_string(FILE *fp, const char *str, size_t max_len) {
    size_t len = strlen(str);
    if (len >= max_len) {
        len = max_len - 1;
    }
    
    // 写入长度
    if (fwrite(&len, sizeof(size_t), 1, fp) != 1) {
        return false;
    }
    
    // 写入字符串
    if (len > 0) {
        if (fwrite(str, 1, len, fp) != len) {
            return false;
        }
    }
    
    return true;
}

/**
 * @brief 从文件读取字符串
 */
static bool read_string(FILE *fp, char *str, size_t max_len) {
    size_t len;
    
    // 读取长度
    if (fread(&len, sizeof(size_t), 1, fp) != 1) {
        return false;
    }
    
    if (len >= max_len) {
        fprintf(stderr, "[CalibManager] String too long: %zu\n", len);
        return false;
    }
    
    // 读取字符串
    if (len > 0) {
        if (fread(str, 1, len, fp) != len) {
            return false;
        }
    }
    
    str[len] = '\0';
    return true;
}

/**
 * @brief 写入PSF数据到文件
 */
static bool write_psf(FILE *fp, const PSFData *psf) {
    if (!fp || !psf) {
        return false;
    }
    
    // 写入PSF参数
    if (fwrite(&psf->width, sizeof(int), 1, fp) != 1) return false;
    if (fwrite(&psf->height, sizeof(int), 1, fp) != 1) return false;
    if (fwrite(&psf->center_x, sizeof(float), 1, fp) != 1) return false;
    if (fwrite(&psf->center_y, sizeof(float), 1, fp) != 1) return false;
    if (fwrite(&psf->wavelength, sizeof(float), 1, fp) != 1) return false;
    if (fwrite(&psf->numerical_aperture, sizeof(float), 1, fp) != 1) return false;
    if (fwrite(&psf->pixel_size, sizeof(float), 1, fp) != 1) return false;
    if (fwrite(&psf->is_normalized, sizeof(bool), 1, fp) != 1) return false;
    
    // 写入PSF数据
    size_t data_size = psf->width * psf->height;
    if (psf->data) {
        if (fwrite(psf->data, sizeof(float), data_size, fp) != data_size) {
            return false;
        }
    }
    
    return true;
}

/**
 * @brief 从文件读取PSF数据
 */
static bool read_psf(FILE *fp, PSFData *psf) {
    if (!fp || !psf) {
        return false;
    }
    
    // 读取PSF参数
    if (fread(&psf->width, sizeof(int), 1, fp) != 1) return false;
    if (fread(&psf->height, sizeof(int), 1, fp) != 1) return false;
    if (fread(&psf->center_x, sizeof(float), 1, fp) != 1) return false;
    if (fread(&psf->center_y, sizeof(float), 1, fp) != 1) return false;
    if (fread(&psf->wavelength, sizeof(float), 1, fp) != 1) return false;
    if (fread(&psf->numerical_aperture, sizeof(float), 1, fp) != 1) return false;
    if (fread(&psf->pixel_size, sizeof(float), 1, fp) != 1) return false;
    if (fread(&psf->is_normalized, sizeof(bool), 1, fp) != 1) return false;
    
    // 验证尺寸
    if (psf->width <= 0 || psf->height <= 0 || 
        psf->width > 4096 || psf->height > 4096) {
        fprintf(stderr, "[CalibManager] Invalid PSF dimensions: %dx%d\n",
                psf->width, psf->height);
        return false;
    }
    
    // 分配并读取PSF数据
    size_t data_size = psf->width * psf->height;
    psf->data = (float*)malloc(data_size * sizeof(float));
    if (!psf->data) {
        fprintf(stderr, "[CalibManager] Failed to allocate PSF data\n");
        return false;
    }
    
    if (fread(psf->data, sizeof(float), data_size, fp) != data_size) {
        free(psf->data);
        psf->data = NULL;
        return false;
    }
    
    return true;
}

/**
 * @brief 写入波长校准数据到文件
 */
static bool write_wavelength(FILE *fp, const WavelengthCalibration *wl) {
    if (!fp || !wl) {
        return false;
    }
    
    // 写入基本参数
    if (fwrite(&wl->wavelength, sizeof(float), 1, fp) != 1) return false;
    if (fwrite(&wl->diffraction_limit, sizeof(float), 1, fp) != 1) return false;
    
    // 写入像差系数
    if (fwrite(wl->aberration_coeff, sizeof(float), 10, fp) != 10) return false;
    
    // 写入PSF
    if (!write_psf(fp, &wl->psf)) return false;
    
    // 写入自定义数据
    if (fwrite(&wl->custom_data_size, sizeof(size_t), 1, fp) != 1) return false;
    if (wl->custom_data_size > 0 && wl->custom_data) {
        if (fwrite(wl->custom_data, 1, wl->custom_data_size, fp) != wl->custom_data_size) {
            return false;
        }
    }
    
    return true;
}

/**
 * @brief 从文件读取波长校准数据
 */
static bool read_wavelength(FILE *fp, WavelengthCalibration *wl) {
    if (!fp || !wl) {
        return false;
    }
    
    memset(wl, 0, sizeof(WavelengthCalibration));
    
    // 读取基本参数
    if (fread(&wl->wavelength, sizeof(float), 1, fp) != 1) return false;
    if (fread(&wl->diffraction_limit, sizeof(float), 1, fp) != 1) return false;
    
    // 读取像差系数
    if (fread(wl->aberration_coeff, sizeof(float), 10, fp) != 10) return false;
    
    // 读取PSF
    if (!read_psf(fp, &wl->psf)) return false;
    
    // 读取自定义数据
    if (fread(&wl->custom_data_size, sizeof(size_t), 1, fp) != 1) return false;
    
    if (wl->custom_data_size > 0) {
        if (wl->custom_data_size > 1024 * 1024) {  // 限制1MB
            fprintf(stderr, "[CalibManager] Custom data too large: %zu bytes\n",
                    wl->custom_data_size);
            return false;
        }
        
        wl->custom_data = malloc(wl->custom_data_size);
        if (!wl->custom_data) {
            fprintf(stderr, "[CalibManager] Failed to allocate custom data\n");
            return false;
        }
        
        if (fread(wl->custom_data, 1, wl->custom_data_size, fp) != wl->custom_data_size) {
            free(wl->custom_data);
            wl->custom_data = NULL;
            return false;
        }
    }
    
    return true;
}

/**
 * @brief 写入通道校准数据到文件
 */
static bool write_channel(FILE *fp, const ChannelCalibration *channel) {
    if (!fp || !channel) {
        return false;
    }
    
    // 写入通道名称和索引
    if (!write_string(fp, channel->name, 64)) return false;
    if (fwrite(&channel->channel_index, sizeof(int), 1, fp) != 1) return false;
    
    // 写入波长数量
    if (fwrite(&channel->num_wavelengths, sizeof(int), 1, fp) != 1) return false;
    
    // 写入每个波长数据
    for (int i = 0; i < channel->num_wavelengths; i++) {
        if (!write_wavelength(fp, &channel->wavelengths[i])) {
            return false;
        }
    }
    
    // 写入光谱响应
    if (fwrite(channel->spectral_response, sizeof(float), 256, fp) != 256) {
        return false;
    }
    
    // 写入颜色矩阵
    if (fwrite(&channel->has_color_matrix, sizeof(bool), 1, fp) != 1) return false;
    if (channel->has_color_matrix) {
        if (fwrite(channel->color_matrix, sizeof(float), 9, fp) != 9) {
            return false;
        }
    }
    
    return true;
}

/**
 * @brief 从文件读取通道校准数据
 */
static bool read_channel(FILE *fp, ChannelCalibration *channel) {
    if (!fp || !channel) {
        return false;
    }
    
    memset(channel, 0, sizeof(ChannelCalibration));
    
    // 读取通道名称和索引
    if (!read_string(fp, channel->name, 64)) return false;
    if (fread(&channel->channel_index, sizeof(int), 1, fp) != 1) return false;
    
    // 读取波长数量
    if (fread(&channel->num_wavelengths, sizeof(int), 1, fp) != 1) return false;
    
    if (channel->num_wavelengths < 0 || 
        channel->num_wavelengths > CALIB_MAX_WAVELENGTHS) {
        fprintf(stderr, "[CalibManager] Invalid wavelength count: %d\n",
                channel->num_wavelengths);
        return false;
    }
    
    // 分配并读取波长数据
    if (channel->num_wavelengths > 0) {
        channel->wavelengths = (WavelengthCalibration*)calloc(
            channel->num_wavelengths, sizeof(WavelengthCalibration));
        
        if (!channel->wavelengths) {
            fprintf(stderr, "[CalibManager] Failed to allocate wavelengths\n");
            return false;
        }
        
        for (int i = 0; i < channel->num_wavelengths; i++) {
            if (!read_wavelength(fp, &channel->wavelengths[i])) {
                // 清理已分配的资源
                for (int j = 0; j < i; j++) {
                    if (channel->wavelengths[j].psf.data) {
                        free(channel->wavelengths[j].psf.data);
                    }
                    if (channel->wavelengths[j].custom_data) {
                        free(channel->wavelengths[j].custom_data);
                    }
                }
                free(channel->wavelengths);
                channel->wavelengths = NULL;
                return false;
            }
        }
    }
    
    // 读取光谱响应
    if (fread(channel->spectral_response, sizeof(float), 256, fp) != 256) {
        return false;
    }
    
    // 读取颜色矩阵
    if (fread(&channel->has_color_matrix, sizeof(bool), 1, fp) != 1) return false;
    if (channel->has_color_matrix) {
        if (fread(channel->color_matrix, sizeof(float), 9, fp) != 9) {
            return false;
        }
    }
    
    return true;
}

// ============================================================================
// 主要文件IO函数
// ============================================================================

/**
 * @brief 保存校准数据到文件
 */
CalibrationError calib_save_to_file(CalibrationManager *manager,
                                    const CalibrationData *data,
                                    const char *filename) {
    if (!manager || !data || !filename) {
        return CALIB_ERROR_INVALID_PARAM;
    }
    
    if (!manager->initialized) {
        set_error(manager, "Manager not initialized");
        return CALIB_ERROR_NOT_INITIALIZED;
    }
    
    // 验证数据
    if (!calib_data_validate(data)) {
        set_error(manager, "Invalid calibration data");
        return CALIB_ERROR_INVALID_FORMAT;
    }
    
    FILE *fp = fopen(filename, "wb");
    if (!fp) {
        char error[256];
        snprintf(error, sizeof(error), "Failed to open file for writing: %s", filename);
        set_error(manager, error);
        return CALIB_ERROR_FILE_WRITE;
    }
    
    if (manager->verbose) {
        printf("[CalibManager] Saving calibration data to: %s\n", filename);
    }
    
    // 写入文件头
    CalibFileHeader header;
    memset(&header, 0, sizeof(CalibFileHeader));
    header.magic = CALIB_FILE_MAGIC;
    header.version = CALIB_FILE_VERSION;
    header.header_size = sizeof(CalibFileHeader);
    header.flags = 0;
    
    // 预留头部空间
    if (fwrite(&header, sizeof(CalibFileHeader), 1, fp) != 1) {
        fclose(fp);
        set_error(manager, "Failed to write file header");
        return CALIB_ERROR_FILE_WRITE;
    }
    
    long data_start = ftell(fp);
    
    // 写入元数据
    CalibrationMetadata *meta = (CalibrationMetadata*)&data->metadata;
    
    if (!write_string(fp, meta->name, 128)) goto write_error;
    if (!write_string(fp, meta->description, 512)) goto write_error;
    if (!write_string(fp, meta->device_model, 64)) goto write_error;
    if (!write_string(fp, meta->lens_model, 64)) goto write_error;
    if (!write_string(fp, meta->creation_date, 32)) goto write_error;
    if (!write_string(fp, meta->author, 64)) goto write_error;
    
    if (fwrite(&meta->type, sizeof(CalibrationType), 1, fp) != 1) goto write_error;
    if (fwrite(&meta->version_major, sizeof(int), 1, fp) != 1) goto write_error;
    if (fwrite(&meta->version_minor, sizeof(int), 1, fp) != 1) goto write_error;
    
    // 写入全局参数
    if (fwrite(&data->num_channels, sizeof(int), 1, fp) != 1) goto write_error;
    if (fwrite(&data->sensor_width, sizeof(float), 1, fp) != 1) goto write_error;
    if (fwrite(&data->sensor_height, sizeof(float), 1, fp) != 1) goto write_error;
    if (fwrite(&data->pixel_pitch, sizeof(float), 1, fp) != 1) goto write_error;
    if (fwrite(&data->image_width, sizeof(int), 1, fp) != 1) goto write_error;
    if (fwrite(&data->image_height, sizeof(int), 1, fp) != 1) goto write_error;
    if (fwrite(&data->focal_length, sizeof(float), 1, fp) != 1) goto write_error;
    if (fwrite(&data->f_number, sizeof(float), 1, fp) != 1) goto write_error;
    if (fwrite(&data->working_distance, sizeof(float), 1, fp) != 1) goto write_error;
    
    // 写入所有通道数据
    for (int i = 0; i < data->num_channels; i++) {
        if (!write_channel(fp, &data->channels[i])) {
            goto write_error;
        }
    }
    
    // 计算数据大小
    long data_end = ftell(fp);
    header.data_size = data_end - data_start;
    
    // 计算校验和
    fseek(fp, data_start, SEEK_SET);
    uint8_t *buffer = (uint8_t*)malloc(header.data_size);
    if (buffer) {
        if (fread(buffer, 1, header.data_size, fp) == header.data_size) {
            header.checksum = calculate_crc32(buffer, header.data_size);
        }
        free(buffer);
    }
    
    // 更新文件头
    fseek(fp, 0, SEEK_SET);
    if (fwrite(&header, sizeof(CalibFileHeader), 1, fp) != 1) {
        fclose(fp);
        set_error(manager, "Failed to update file header");
        return CALIB_ERROR_FILE_WRITE;
    }
    
    fclose(fp);
    
    manager->total_saves++;
    
    if (manager->verbose) {
        printf("[CalibManager] Successfully saved calibration data\n");
        printf("  File size: %ld bytes\n", data_end);
        printf("  Checksum: 0x%08X\n", header.checksum);
    }
    
    return CALIB_SUCCESS;

write_error:
    fclose(fp);
    set_error(manager, "Failed to write calibration data");
    return CALIB_ERROR_FILE_WRITE;
}

/**
 * @brief 从文件加载校准数据
 */
CalibrationError calib_load_from_file(CalibrationManager *manager,
                                      const char *filename,
                                      CalibrationData **out_data) {
    if (!manager || !filename || !out_data) {
        return CALIB_ERROR_INVALID_PARAM;
    }
    
    if (!manager->initialized) {
        set_error(manager, "Manager not initialized");
        return CALIB_ERROR_NOT_INITIALIZED;
    }
    
    *out_data = NULL;
    
    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        char error[256];
        snprintf(error, sizeof(error), "Failed to open file: %s", filename);
        set_error(manager, error);
        return CALIB_ERROR_FILE_NOT_FOUND;
    }
    
    if (manager->verbose) {
        printf("[CalibManager] Loading calibration data from: %s\n", filename);
    }
    
    // 读取文件头
    CalibFileHeader header;
    if (fread(&header, sizeof(CalibFileHeader), 1, fp) != 1) {
        fclose(fp);
        set_error(manager, "Failed to read file header");
        return CALIB_ERROR_FILE_READ;
    }
    
    // 验证魔数
    if (header.magic != CALIB_FILE_MAGIC) {
        fclose(fp);
        set_error(manager, "Invalid file format (magic number mismatch)");
        return CALIB_ERROR_INVALID_FORMAT;
    }
    
    // 验证版本
    if (header.version != CALIB_FILE_VERSION) {
        fclose(fp);
        char error[256];
        snprintf(error, sizeof(error), 
                "Version mismatch: file=%u, expected=%u",
                header.version, CALIB_FILE_VERSION);
        set_error(manager, error);
        return CALIB_ERROR_VERSION_MISMATCH;
    }
    
    // 验证校验和
    if (header.data_size > 0) {
        uint8_t *buffer = (uint8_t*)malloc(header.data_size);
        if (buffer) {
            if (fread(buffer, 1, header.data_size, fp) == header.data_size) {
                uint32_t calculated_checksum = calculate_crc32(buffer, header.data_size);
                if (calculated_checksum != header.checksum) {
                    free(buffer);
                    fclose(fp);
                    set_error(manager, "Checksum verification failed");
                    return CALIB_ERROR_CHECKSUM_FAILED;
                }
            }
            free(buffer);
        }
        
        // 回到数据开始位置
        fseek(fp, sizeof(CalibFileHeader), SEEK_SET);
    }
    
    // 创建临时数据结构
    CalibrationData *data = (CalibrationData*)calloc(1, sizeof(CalibrationData));
    if (!data) {
        fclose(fp);
        set_error(manager, "Failed to allocate calibration data");
        return CALIB_ERROR_MEMORY_ALLOCATION;
    }
    
    // 读取元数据
    if (!read_string(fp, data->metadata.name, 128)) goto read_error;
    if (!read_string(fp, data->metadata.description, 512)) goto read_error;
    if (!read_string(fp, data->metadata.device_model, 64)) goto read_error;
    if (!read_string(fp, data->metadata.lens_model, 64)) goto read_error;
    if (!read_string(fp, data->metadata.creation_date, 32)) goto read_error;
    if (!read_string(fp, data->metadata.author, 64)) goto read_error;
    
    if (fread(&data->metadata.type, sizeof(CalibrationType), 1, fp) != 1) goto read_error;
    if (fread(&data->metadata.version_major, sizeof(int), 1, fp) != 1) goto read_error;
    if (fread(&data->metadata.version_minor, sizeof(int), 1, fp) != 1) goto read_error;
    
    data->metadata.checksum = header.checksum;
    
    // 读取全局参数
    if (fread(&data->num_channels, sizeof(int), 1, fp) != 1) goto read_error;
    
    if (data->num_channels <= 0 || data->num_channels > CALIB_MAX_CHANNELS) {
        fclose(fp);
        calib_data_destroy(data);
        set_error(manager, "Invalid number of channels");
        return CALIB_ERROR_INVALID_FORMAT;
    }
    
    if (fread(&data->sensor_width, sizeof(float), 1, fp) != 1) goto read_error;
    if (fread(&data->sensor_height, sizeof(float), 1, fp) != 1) goto read_error;
    if (fread(&data->pixel_pitch, sizeof(float), 1, fp) != 1) goto read_error;
    if (fread(&data->image_width, sizeof(int), 1, fp) != 1) goto read_error;
    if (fread(&data->image_height, sizeof(int), 1, fp) != 1) goto read_error;
    if (fread(&data->focal_length, sizeof(float), 1, fp) != 1) goto read_error;
    if (fread(&data->f_number, sizeof(float), 1, fp) != 1) goto read_error;
    if (fread(&data->working_distance, sizeof(float), 1, fp) != 1) goto read_error;
    
    // 分配通道数组
    data->channels = (ChannelCalibration*)calloc(
        data->num_channels, sizeof(ChannelCalibration));
    if (!data->channels) {
        fclose(fp);
        calib_data_destroy(data);
        set_error(manager, "Failed to allocate channels");
        return CALIB_ERROR_MEMORY_ALLOCATION;
    }
    
    // 读取所有通道数据
    for (int i = 0; i < data->num_channels; i++) {
        if (!read_channel(fp, &data->channels[i])) {
            fclose(fp);
            calib_data_destroy(data);
            set_error(manager, "Failed to read channel data");
            return CALIB_ERROR_FILE_READ;
        }
    }
    
    fclose(fp);
    
    // 验证加载的数据
    data->is_valid = calib_data_validate(data);
    if (!data->is_valid) {
        calib_data_destroy(data);
        set_error(manager, "Loaded data validation failed");
        return CALIB_ERROR_DATA_CORRUPTED;
    }
    
    *out_data = data;
    manager->total_loads++;
    
    if (manager->verbose) {
        printf("[CalibManager] Successfully loaded calibration data\n");
        printf("  Channels: %d\n", data->num_channels);
        printf("  Image size: %dx%d\n", data->image_width, data->image_height);
    }
    
    return CALIB_SUCCESS;

read_error:
    fclose(fp);
    calib_data_destroy(data);
    set_error(manager, "Failed to read calibration data");
    return CALIB_ERROR_FILE_READ;
}
// ============================================================================
// 校准应用 - PSF卷积
// ============================================================================

CalibrationError calib_manager_apply_psf_convolution(
    CalibrationManager *manager,
    const float *input_image,
    int width,
    int height,
    int channel_index,
    float wavelength,
    float *output_image)
{
    CALIB_CHECK_NULL(manager);
    CALIB_CHECK_NULL(input_image);
    CALIB_CHECK_NULL(output_image);
    
    if (width <= 0 || height <= 0) {
        return CALIB_ERROR_INVALID_PARAM;
    }
    
    // 获取通道
    ChannelCalibration *channel = calib_data_get_channel(
        manager->calib_data, channel_index);
    if (!channel) {
        return CALIB_ERROR_CHANNEL_NOT_FOUND;
    }
    
    // 查找波长
    WavelengthCalibration *wl = calib_channel_find_wavelength(channel, wavelength);
    if (!wl) {
        return CALIB_ERROR_WAVELENGTH_NOT_FOUND;
    }
    
    // 检查PSF
    if (!wl->psf.data || wl->psf.width <= 0 || wl->psf.height <= 0) {
        return CALIB_ERROR_PSF_NOT_FOUND;
    }
    
    // 执行卷积
    return perform_psf_convolution(
        input_image, width, height,
        &wl->psf,
        output_image);
}

/**
 * @brief 执行PSF卷积（内部函数）
 */
static CalibrationError perform_psf_convolution(
    const float *input,
    int width,
    int height,
    const PSFData *psf,
    float *output)
{
    int psf_width = psf->width;
    int psf_height = psf->height;
    int psf_cx = (int)psf->center_x;
    int psf_cy = (int)psf->center_y;
    
    // 初始化输出
    memset(output, 0, width * height * sizeof(float));
    
    // 对每个输出像素进行卷积
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float sum = 0.0f;
            float weight_sum = 0.0f;
            
            // 遍历PSF窗口
            for (int py = 0; py < psf_height; py++) {
                for (int px = 0; px < psf_width; px++) {
                    // 计算输入图像坐标
                    int ix = x + px - psf_cx;
                    int iy = y + py - psf_cy;
                    
                    // 边界检查
                    if (ix >= 0 && ix < width && iy >= 0 && iy < height) {
                        float psf_value = psf->data[py * psf_width + px];
                        float input_value = input[iy * width + ix];
                        
                        sum += input_value * psf_value;
                        weight_sum += psf_value;
                    }
                }
            }
            
            // 归一化（如果PSF未归一化）
            if (!psf->is_normalized && weight_sum > 0.0f) {
                sum /= weight_sum;
            }
            
            output[y * width + x] = sum;
        }
    }
    
    return CALIB_SUCCESS;
}

// ============================================================================
// 校准应用 - 颜色校正
// ============================================================================

CalibrationError calib_manager_apply_color_correction(
    CalibrationManager *manager,
    const float *input_rgb,
    int num_pixels,
    int channel_index,
    float *output_rgb)
{
    CALIB_CHECK_NULL(manager);
    CALIB_CHECK_NULL(input_rgb);
    CALIB_CHECK_NULL(output_rgb);
    
    if (num_pixels <= 0) {
        return CALIB_ERROR_INVALID_PARAM;
    }
    
    // 获取通道
    ChannelCalibration *channel = calib_data_get_channel(
        manager->calib_data, channel_index);
    if (!channel) {
        return CALIB_ERROR_CHANNEL_NOT_FOUND;
    }
    
    // 检查颜色矩阵
    if (!channel->has_color_matrix) {
        return CALIB_ERROR_COLOR_MATRIX_NOT_FOUND;
    }
    
    // 应用颜色矩阵到每个像素
    for (int i = 0; i < num_pixels; i++) {
        float r = input_rgb[i * 3 + 0];
        float g = input_rgb[i * 3 + 1];
        float b = input_rgb[i * 3 + 2];
        
        // 矩阵乘法: [R' G' B'] = [R G B] * M
        output_rgb[i * 3 + 0] = 
            r * channel->color_matrix[0] + 
            g * channel->color_matrix[1] + 
            b * channel->color_matrix[2];
            
        output_rgb[i * 3 + 1] = 
            r * channel->color_matrix[3] + 
            g * channel->color_matrix[4] + 
            b * channel->color_matrix[5];
            
        output_rgb[i * 3 + 2] = 
            r * channel->color_matrix[6] + 
            g * channel->color_matrix[7] + 
            b * channel->color_matrix[8];
        
        // 限制范围
        output_rgb[i * 3 + 0] = fmaxf(0.0f, fminf(1.0f, output_rgb[i * 3 + 0]));
        output_rgb[i * 3 + 1] = fmaxf(0.0f, fminf(1.0f, output_rgb[i * 3 + 1]));
        output_rgb[i * 3 + 2] = fmaxf(0.0f, fminf(1.0f, output_rgb[i * 3 + 2]));
    }
    
    return CALIB_SUCCESS;
}

// ============================================================================
// 校准应用 - 像差校正
// ============================================================================

CalibrationError calib_manager_apply_aberration_correction(
    CalibrationManager *manager,
    const float *input_image,
    int width,
    int height,
    int channel_index,
    float wavelength,
    float *output_image)
{
    CALIB_CHECK_NULL(manager);
    CALIB_CHECK_NULL(input_image);
    CALIB_CHECK_NULL(output_image);
    
    if (width <= 0 || height <= 0) {
        return CALIB_ERROR_INVALID_PARAM;
    }
    
    // 获取通道
    ChannelCalibration *channel = calib_data_get_channel(
        manager->calib_data, channel_index);
    if (!channel) {
        return CALIB_ERROR_CHANNEL_NOT_FOUND;
    }
    
    // 查找波长
    WavelengthCalibration *wl = calib_channel_find_wavelength(channel, wavelength);
    if (!wl) {
        return CALIB_ERROR_WAVELENGTH_NOT_FOUND;
    }
    
    // 检查像差系数
    bool has_aberration = false;
    for (int i = 0; i < CALIB_MAX_ABERRATION_TERMS; i++) {
        if (fabs(wl->aberration_coeff[i]) > 1e-9f) {
            has_aberration = true;
            break;
        }
    }
    
    if (!has_aberration) {
        // 没有像差，直接复制
        memcpy(output_image, input_image, width * height * sizeof(float));
        return CALIB_SUCCESS;
    }
    
    // 执行像差校正
    return perform_aberration_correction(
        input_image, width, height,
        wl->aberration_coeff,
        output_image);
}

/**
 * @brief 执行像差校正（内部函数）
 */
static CalibrationError perform_aberration_correction(
    const float *input,
    int width,
    int height,
    const float *aberration_coeff,
    float *output)
{
    float cx = width / 2.0f;
    float cy = height / 2.0f;
    float max_r = sqrtf(cx * cx + cy * cy);
    
    // 对每个输出像素
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            // 归一化坐标
            float nx = (x - cx) / max_r;
            float ny = (y - cy) / max_r;
            float r = sqrtf(nx * nx + ny * ny);
            
            // 计算径向畸变
            float r2 = r * r;
            float r4 = r2 * r2;
            float r6 = r4 * r2;
            
            float radial_distortion = 
                1.0f + 
                aberration_coeff[0] * r2 + 
                aberration_coeff[1] * r4 + 
                aberration_coeff[2] * r6;
            
            // 计算切向畸变
            float tangential_x = 
                2.0f * aberration_coeff[3] * nx * ny + 
                aberration_coeff[4] * (r2 + 2.0f * nx * nx);
                
            float tangential_y = 
                aberration_coeff[3] * (r2 + 2.0f * ny * ny) + 
                2.0f * aberration_coeff[4] * nx * ny;
            
            // 计算校正后的坐标
            float corrected_nx = nx * radial_distortion + tangential_x;
            float corrected_ny = ny * radial_distortion + tangential_y;
            
            float corrected_x = corrected_nx * max_r + cx;
            float corrected_y = corrected_ny * max_r + cy;
            
            // 双线性插值
            output[y * width + x] = bilinear_interpolate(
                input, width, height, corrected_x, corrected_y);
        }
    }
    
    return CALIB_SUCCESS;
}

/**
 * @brief 双线性插值（内部函数）
 */
static float bilinear_interpolate(
    const float *image,
    int width,
    int height,
    float x,
    float y)
{
    // 边界检查
    if (x < 0 || x >= width - 1 || y < 0 || y >= height - 1) {
        return 0.0f;
    }
    
    int x0 = (int)x;
    int y0 = (int)y;
    int x1 = x0 + 1;
    int y1 = y0 + 1;
    
    float fx = x - x0;
    float fy = y - y0;
    
    float v00 = image[y0 * width + x0];
    float v10 = image[y0 * width + x1];
    float v01 = image[y1 * width + x0];
    float v11 = image[y1 * width + x1];
    
    float v0 = v00 * (1.0f - fx) + v10 * fx;
    float v1 = v01 * (1.0f - fx) + v11 * fx;
    
    return v0 * (1.0f - fy) + v1 * fy;
}

// ============================================================================
// 校准应用 - 光谱响应校正
// ============================================================================

CalibrationError calib_manager_apply_spectral_correction(
    CalibrationManager *manager,
    const float *input_spectrum,
    int num_wavelengths,
    int channel_index,
    float *output_spectrum)
{
    CALIB_CHECK_NULL(manager);
    CALIB_CHECK_NULL(input_spectrum);
    CALIB_CHECK_NULL(output_spectrum);
    
    if (num_wavelengths <= 0 || num_wavelengths > CALIB_MAX_SPECTRAL_SAMPLES) {
        return CALIB_ERROR_INVALID_PARAM;
    }
    
    // 获取通道
    ChannelCalibration *channel = calib_data_get_channel(
        manager->calib_data, channel_index);
    if (!channel) {
        return CALIB_ERROR_CHANNEL_NOT_FOUND;
    }
    
    // 应用光谱响应校正
    for (int i = 0; i < num_wavelengths; i++) {
        if (i < CALIB_MAX_SPECTRAL_SAMPLES) {
            float response = channel->spectral_response[i];
            if (response > 1e-6f) {
                output_spectrum[i] = input_spectrum[i] / response;
            } else {
                output_spectrum[i] = 0.0f;
            }
        } else {
            output_spectrum[i] = input_spectrum[i];
        }
    }
    
    return CALIB_SUCCESS;
}

// ============================================================================
// 校准应用 - 完整流水线
// ============================================================================

CalibrationError calib_manager_apply_full_pipeline(
    CalibrationManager *manager,
    const float *input_image,
    int width,
    int height,
    int channel_index,
    float wavelength,
    CalibrationPipelineFlags flags,
    float *output_image)
{
    CALIB_CHECK_NULL(manager);
    CALIB_CHECK_NULL(input_image);
    CALIB_CHECK_NULL(output_image);
    
    if (width <= 0 || height <= 0) {
        return CALIB_ERROR_INVALID_PARAM;
    }
    
    CalibrationError err;
    
    // 分配临时缓冲区
    float *temp_buffer1 = (float*)malloc(width * height * sizeof(float));
    float *temp_buffer2 = (float*)malloc(width * height * sizeof(float));
    
    if (!temp_buffer1 || !temp_buffer2) {
        free(temp_buffer1);
        free(temp_buffer2);
        return CALIB_ERROR_MEMORY_ALLOCATION;
    }
    
    // 复制输入到临时缓冲区
    memcpy(temp_buffer1, input_image, width * height * sizeof(float));
    
    // 1. 像差校正
    if (flags & CALIB_PIPELINE_ABERRATION) {
        err = calib_manager_apply_aberration_correction(
            manager, temp_buffer1, width, height, 
            channel_index, wavelength, temp_buffer2);
        
        if (err != CALIB_SUCCESS) {
            free(temp_buffer1);
            free(temp_buffer2);
            return err;
        }
        
        // 交换缓冲区
        float *temp = temp_buffer1;
        temp_buffer1 = temp_buffer2;
        temp_buffer2 = temp;
    }
    
    // 2. PSF卷积
    if (flags & CALIB_PIPELINE_PSF) {
        err = calib_manager_apply_psf_convolution(
            manager, temp_buffer1, width, height,
            channel_index, wavelength, temp_buffer2);
        
        if (err != CALIB_SUCCESS) {
            free(temp_buffer1);
            free(temp_buffer2);
            return err;
        }
        
        // 交换缓冲区
        float *temp = temp_buffer1;
        temp_buffer1 = temp_buffer2;
        temp_buffer2 = temp;
    }
    
    // 3. 光谱响应校正（如果需要）
    if (flags & CALIB_PIPELINE_SPECTRAL) {
        // 注意：这里假设图像是单波长的
        // 对于多光谱图像，需要不同的处理方式
        err = calib_manager_apply_spectral_correction(
            manager, temp_buffer1, width * height,
            channel_index, temp_buffer2);
        
        if (err != CALIB_SUCCESS) {
            free(temp_buffer1);
            free(temp_buffer2);
            return err;
        }
        
        // 交换缓冲区
        float *temp = temp_buffer1;
        temp_buffer1 = temp_buffer2;
        temp_buffer2 = temp;
    }
    
    // 复制结果到输出
    memcpy(output_image, temp_buffer1, width * height * sizeof(float));
    
    // 释放临时缓冲区
    free(temp_buffer1);
    free(temp_buffer2);
    
    return CALIB_SUCCESS;
}
// ============================================================================
// 校准应用 - 批量处理
// ============================================================================

CalibrationError calib_manager_apply_batch(
    CalibrationManager *manager,
    const float **input_images,
    int num_images,
    int width,
    int height,
    int channel_index,
    const float *wavelengths,
    CalibrationPipelineFlags flags,
    float **output_images,
    CalibrationProgressCallback progress_callback,
    void *user_data)
{
    CALIB_CHECK_NULL(manager);
    CALIB_CHECK_NULL(input_images);
    CALIB_CHECK_NULL(output_images);
    
    if (num_images <= 0 || width <= 0 || height <= 0) {
        return CALIB_ERROR_INVALID_PARAM;
    }
    
    CalibrationError err = CALIB_SUCCESS;
    
    // 处理每张图像
    for (int i = 0; i < num_images; i++) {
        if (!input_images[i] || !output_images[i]) {
            continue;
        }
        
        float wavelength = wavelengths ? wavelengths[i] : 0.0f;
        
        // 应用校准流水线
        err = calib_manager_apply_full_pipeline(
            manager,
            input_images[i],
            width,
            height,
            channel_index,
            wavelength,
            flags,
            output_images[i]);
        
        if (err != CALIB_SUCCESS) {
            break;
        }
        
        // 调用进度回调
        if (progress_callback) {
            float progress = (float)(i + 1) / num_images;
            if (!progress_callback(progress, user_data)) {
                err = CALIB_ERROR_OPERATION_CANCELLED;
                break;
            }
        }
    }
    
    return err;
}

// ============================================================================
// 校准应用 - 多通道处理
// ============================================================================

CalibrationError calib_manager_apply_multichannel(
    CalibrationManager *manager,
    const float **channel_images,
    int num_channels,
    int width,
    int height,
    const int *channel_indices,
    const float *wavelengths,
    CalibrationPipelineFlags flags,
    float **output_images)
{
    CALIB_CHECK_NULL(manager);
    CALIB_CHECK_NULL(channel_images);
    CALIB_CHECK_NULL(output_images);
    CALIB_CHECK_NULL(channel_indices);
    
    if (num_channels <= 0 || width <= 0 || height <= 0) {
        return CALIB_ERROR_INVALID_PARAM;
    }
    
    CalibrationError err = CALIB_SUCCESS;
    
    // 处理每个通道
    for (int i = 0; i < num_channels; i++) {
        if (!channel_images[i] || !output_images[i]) {
            continue;
        }
        
        int channel_index = channel_indices[i];
        float wavelength = wavelengths ? wavelengths[i] : 0.0f;
        
        // 应用校准流水线
        err = calib_manager_apply_full_pipeline(
            manager,
            channel_images[i],
            width,
            height,
            channel_index,
            wavelength,
            flags,
            output_images[i]);
        
        if (err != CALIB_SUCCESS) {
            break;
        }
    }
    
    return err;
}

// ============================================================================
// 校准应用 - RGB图像处理
// ============================================================================

CalibrationError calib_manager_apply_rgb(
    CalibrationManager *manager,
    const float *input_rgb,
    int width,
    int height,
    int channel_index,
    CalibrationPipelineFlags flags,
    float *output_rgb)
{
    CALIB_CHECK_NULL(manager);
    CALIB_CHECK_NULL(input_rgb);
    CALIB_CHECK_NULL(output_rgb);
    
    if (width <= 0 || height <= 0) {
        return CALIB_ERROR_INVALID_PARAM;
    }
    
    CalibrationError err;
    int num_pixels = width * height;
    
    // 分配临时缓冲区用于分离通道
    float *r_channel = (float*)malloc(num_pixels * sizeof(float));
    float *g_channel = (float*)malloc(num_pixels * sizeof(float));
    float *b_channel = (float*)malloc(num_pixels * sizeof(float));
    
    float *r_output = (float*)malloc(num_pixels * sizeof(float));
    float *g_output = (float*)malloc(num_pixels * sizeof(float));
    float *b_output = (float*)malloc(num_pixels * sizeof(float));
    
    if (!r_channel || !g_channel || !b_channel ||
        !r_output || !g_output || !b_output) {
        free(r_channel);
        free(g_channel);
        free(b_channel);
        free(r_output);
        free(g_output);
        free(b_output);
        return CALIB_ERROR_MEMORY_ALLOCATION;
    }
    
    // 分离RGB通道
    for (int i = 0; i < num_pixels; i++) {
        r_channel[i] = input_rgb[i * 3 + 0];
        g_channel[i] = input_rgb[i * 3 + 1];
        b_channel[i] = input_rgb[i * 3 + 2];
    }
    
    // 处理R通道（假设红光波长约为630nm）
    if (flags & (CALIB_PIPELINE_PSF | CALIB_PIPELINE_ABERRATION)) {
        err = calib_manager_apply_full_pipeline(
            manager, r_channel, width, height,
            channel_index, 630.0f, flags, r_output);
        if (err != CALIB_SUCCESS) {
            goto cleanup;
        }
    } else {
        memcpy(r_output, r_channel, num_pixels * sizeof(float));
    }
    
    // 处理G通道（假设绿光波长约为530nm）
    if (flags & (CALIB_PIPELINE_PSF | CALIB_PIPELINE_ABERRATION)) {
        err = calib_manager_apply_full_pipeline(
            manager, g_channel, width, height,
            channel_index, 530.0f, flags, g_output);
        if (err != CALIB_SUCCESS) {
            goto cleanup;
        }
    } else {
        memcpy(g_output, g_channel, num_pixels * sizeof(float));
    }
    
    // 处理B通道（假设蓝光波长约为470nm）
    if (flags & (CALIB_PIPELINE_PSF | CALIB_PIPELINE_ABERRATION)) {
        err = calib_manager_apply_full_pipeline(
            manager, b_channel, width, height,
            channel_index, 470.0f, flags, b_output);
        if (err != CALIB_SUCCESS) {
            goto cleanup;
        }
    } else {
        memcpy(b_output, b_channel, num_pixels * sizeof(float));
    }
    
    // 合并通道
    for (int i = 0; i < num_pixels; i++) {
        output_rgb[i * 3 + 0] = r_output[i];
        output_rgb[i * 3 + 1] = g_output[i];
        output_rgb[i * 3 + 2] = b_output[i];
    }
    
    // 应用颜色校正（如果需要）
    if (flags & CALIB_PIPELINE_COLOR) {
        err = calib_manager_apply_color_correction(
            manager, output_rgb, num_pixels, channel_index, output_rgb);
        if (err != CALIB_SUCCESS) {
            goto cleanup;
        }
    }
    
    err = CALIB_SUCCESS;

cleanup:
    free(r_channel);
    free(g_channel);
    free(b_channel);
    free(r_output);
    free(g_output);
    free(b_output);
    
    return err;
}

// ============================================================================
// 校准应用 - 自适应处理
// ============================================================================

CalibrationError calib_manager_apply_adaptive(
    CalibrationManager *manager,
    const float *input_image,
    int width,
    int height,
    int channel_index,
    float wavelength,
    const CalibrationAdaptiveParams *params,
    float *output_image)
{
    CALIB_CHECK_NULL(manager);
    CALIB_CHECK_NULL(input_image);
    CALIB_CHECK_NULL(output_image);
    CALIB_CHECK_NULL(params);
    
    if (width <= 0 || height <= 0) {
        return CALIB_ERROR_INVALID_PARAM;
    }
    
    // 获取通道
    ChannelCalibration *channel = calib_data_get_channel(
        manager->calib_data, channel_index);
    if (!channel) {
        return CALIB_ERROR_CHANNEL_NOT_FOUND;
    }
    
    // 查找波长
    WavelengthCalibration *wl = calib_channel_find_wavelength(channel, wavelength);
    if (!wl) {
        return CALIB_ERROR_WAVELENGTH_NOT_FOUND;
    }
    
    // 分配临时缓冲区
    float *temp_buffer = (float*)malloc(width * height * sizeof(float));
    if (!temp_buffer) {
        return CALIB_ERROR_MEMORY_ALLOCATION;
    }
    
    // 根据图像内容自适应调整校准参数
    CalibrationError err = CALIB_SUCCESS;
    
    // 1. 分析图像特征
    float mean_intensity = 0.0f;
    float max_intensity = 0.0f;
    for (int i = 0; i < width * height; i++) {
        mean_intensity += input_image[i];
        if (input_image[i] > max_intensity) {
            max_intensity = input_image[i];
        }
    }
    mean_intensity /= (width * height);
    
    // 2. 根据强度调整PSF强度
    float psf_strength = params->base_psf_strength;
    if (mean_intensity < params->low_intensity_threshold) {
        psf_strength *= params->low_intensity_factor;
    } else if (mean_intensity > params->high_intensity_threshold) {
        psf_strength *= params->high_intensity_factor;
    }
    
    // 3. 应用调整后的PSF卷积
    if (wl->psf.data && psf_strength > 0.0f) {
        // 创建调整后的PSF
        PSFData adjusted_psf = wl->psf;
        float *adjusted_psf_data = (float*)malloc(
            adjusted_psf.width * adjusted_psf.height * sizeof(float));
        
        if (adjusted_psf_data) {
            // 调整PSF强度
            for (int i = 0; i < adjusted_psf.width * adjusted_psf.height; i++) {
                adjusted_psf_data[i] = wl->psf.data[i] * psf_strength;
            }
            adjusted_psf.data = adjusted_psf_data;
            
            // 执行卷积
            err = perform_psf_convolution(
                input_image, width, height,
                &adjusted_psf, temp_buffer);
            
            free(adjusted_psf_data);
            
            if (err != CALIB_SUCCESS) {
                free(temp_buffer);
                return err;
            }
        } else {
            free(temp_buffer);
            return CALIB_ERROR_MEMORY_ALLOCATION;
        }
    } else {
        memcpy(temp_buffer, input_image, width * height * sizeof(float));
    }
    
    // 4. 根据局部对比度调整像差校正
    if (params->enable_local_aberration) {
        float aberration_strength = params->base_aberration_strength;
        
        // 计算局部对比度
        float local_contrast = calculate_local_contrast(
            temp_buffer, width, height, params->contrast_window_size);
        
        if (local_contrast < params->low_contrast_threshold) {
            aberration_strength *= params->low_contrast_factor;
        }
        
        // 调整像差系数
        float adjusted_coeff[CALIB_MAX_ABERRATION_TERMS];
        for (int i = 0; i < CALIB_MAX_ABERRATION_TERMS; i++) {
            adjusted_coeff[i] = wl->aberration_coeff[i] * aberration_strength;
        }
        
        // 应用像差校正
        err = perform_aberration_correction(
            temp_buffer, width, height, adjusted_coeff, output_image);
        
        if (err != CALIB_SUCCESS) {
            free(temp_buffer);
            return err;
        }
    } else {
        memcpy(output_image, temp_buffer, width * height * sizeof(float));
    }
    
    free(temp_buffer);
    return CALIB_SUCCESS;
}

/**
 * @brief 计算局部对比度（内部函数）
 */
static float calculate_local_contrast(
    const float *image,
    int width,
    int height,
    int window_size)
{
    if (window_size <= 0 || window_size > width || window_size > height) {
        return 0.0f;
    }
    
    float total_contrast = 0.0f;
    int num_windows = 0;
    
    int half_window = window_size / 2;
    
    // 在图像上滑动窗口
    for (int y = half_window; y < height - half_window; y += window_size) {
        for (int x = half_window; x < width - half_window; x += window_size) {
            float min_val = FLT_MAX;
            float max_val = -FLT_MAX;
            
            // 计算窗口内的最小和最大值
            for (int wy = -half_window; wy <= half_window; wy++) {
                for (int wx = -half_window; wx <= half_window; wx++) {
                    int px = x + wx;
                    int py = y + wy;
                    
                    if (px >= 0 && px < width && py >= 0 && py < height) {
                        float val = image[py * width + px];
                        if (val < min_val) min_val = val;
                        if (val > max_val) max_val = val;
                    }
                }
            }
            
            // 计算局部对比度
            if (max_val + min_val > 0.0f) {
                float contrast = (max_val - min_val) / (max_val + min_val);
                total_contrast += contrast;
                num_windows++;
            }
        }
    }
    
    return num_windows > 0 ? total_contrast / num_windows : 0.0f;
}

// ============================================================================
// 校准应用 - 区域处理
// ============================================================================

CalibrationError calib_manager_apply_region(
    CalibrationManager *manager,
    const float *input_image,
    int width,
    int height,
    int channel_index,
    float wavelength,
    const CalibrationRegion *region,
    CalibrationPipelineFlags flags,
    float *output_image)
{
    CALIB_CHECK_NULL(manager);
    CALIB_CHECK_NULL(input_image);
    CALIB_CHECK_NULL(output_image);
    CALIB_CHECK_NULL(region);
    
    if (width <= 0 || height <= 0) {
        return CALIB_ERROR_INVALID_PARAM;
    }
    
    // 验证区域
    if (region->x < 0 || region->y < 0 ||
        region->x + region->width > width ||
        region->y + region->height > height) {
        return CALIB_ERROR_INVALID_PARAM;
    }
    
    // 先复制整个图像
    memcpy(output_image, input_image, width * height * sizeof(float));
    
    // 提取区域
    float *region_input = (float*)malloc(region->width * region->height * sizeof(float));
    float *region_output = (float*)malloc(region->width * region->height * sizeof(float));
    
    if (!region_input || !region_output) {
        free(region_input);
        free(region_output);
        return CALIB_ERROR_MEMORY_ALLOCATION;
    }
    
    // 复制区域数据
    for (int y = 0; y < region->height; y++) {
        for (int x = 0; x < region->width; x++) {
            int src_idx = (region->y + y) * width + (region->x + x);
            int dst_idx = y * region->width + x;
            region_input[dst_idx] = input_image[src_idx];
        }
    }
    
    // 处理区域
    CalibrationError err = calib_manager_apply_full_pipeline(
        manager,
        region_input,
        region->width,
        region->height,
        channel_index,
        wavelength,
        flags,
        region_output);
    
    if (err == CALIB_SUCCESS) {
        // 将处理后的区域复制回输出图像
        for (int y = 0; y < region->height; y++) {
            for (int x = 0; x < region->width; x++) {
                int src_idx = y * region->width + x;
                int dst_idx = (region->y + y) * width + (region->x + x);
                output_image[dst_idx] = region_output[src_idx];
            }
        }
    }
    
    free(region_input);
    free(region_output);
    
    return err;
}

// ============================================================================
// 校准应用 - 多区域处理
// ============================================================================

CalibrationError calib_manager_apply_multi_region(
    CalibrationManager *manager,
    const float *input_image,
    int width,
    int height,
    int channel_index,
    float wavelength,
    const CalibrationRegion *regions,
    int num_regions,
    CalibrationPipelineFlags flags,
    float *output_image)
{
    CALIB_CHECK_NULL(manager);
    CALIB_CHECK_NULL(input_image);
    CALIB_CHECK_NULL(output_image);
    CALIB_CHECK_NULL(regions);
    
    if (width <= 0 || height <= 0 || num_regions <= 0) {
        return CALIB_ERROR_INVALID_PARAM;
    }
    
    // 先复制整个图像
    memcpy(output_image, input_image, width * height * sizeof(float));
    
    // 处理每个区域
    for (int i = 0; i < num_regions; i++) {
        CalibrationError err = calib_manager_apply_region(
            manager,
            input_image,
            width,
            height,
            channel_index,
            wavelength,
            &regions[i],
            flags,
            output_image);
        
        if (err != CALIB_SUCCESS) {
            return err;
        }
    }
    
    return CALIB_SUCCESS;
}
// ============================================================================
// 性能优化 - 缓存管理
// ============================================================================

CalibrationError calib_manager_enable_cache(
    CalibrationManager *manager,
    size_t cache_size_mb)
{
    CALIB_CHECK_NULL(manager);
    
    if (cache_size_mb == 0) {
        return CALIB_ERROR_INVALID_PARAM;
    }
    
    // 如果已有缓存，先清理
    if (manager->cache) {
        calib_manager_clear_cache(manager);
        free(manager->cache);
    }
    
    // 分配缓存结构
    manager->cache = (CalibrationCache*)calloc(1, sizeof(CalibrationCache));
    if (!manager->cache) {
        return CALIB_ERROR_MEMORY_ALLOCATION;
    }
    
    manager->cache->max_size = cache_size_mb * 1024 * 1024;
    manager->cache->current_size = 0;
    manager->cache->num_entries = 0;
    manager->cache->entries = NULL;
    
    manager->cache_enabled = true;
    
    return CALIB_SUCCESS;
}

CalibrationError calib_manager_disable_cache(CalibrationManager *manager) {
    CALIB_CHECK_NULL(manager);
    
    if (manager->cache) {
        calib_manager_clear_cache(manager);
        free(manager->cache);
        manager->cache = NULL;
    }
    
    manager->cache_enabled = false;
    
    return CALIB_SUCCESS;
}

CalibrationError calib_manager_clear_cache(CalibrationManager *manager) {
    CALIB_CHECK_NULL(manager);
    
    if (!manager->cache) {
        return CALIB_SUCCESS;
    }
    
    // 释放所有缓存条目
    for (int i = 0; i < manager->cache->num_entries; i++) {
        CalibrationCacheEntry *entry = &manager->cache->entries[i];
        if (entry->data) {
            free(entry->data);
            entry->data = NULL;
        }
    }
    
    if (manager->cache->entries) {
        free(manager->cache->entries);
        manager->cache->entries = NULL;
    }
    
    manager->cache->num_entries = 0;
    manager->cache->current_size = 0;
    
    return CALIB_SUCCESS;
}

/**
 * @brief 查找缓存条目（内部函数）
 */
static CalibrationCacheEntry* find_cache_entry(
    CalibrationCache *cache,
    int channel_index,
    float wavelength,
    CalibrationCacheType type)
{
    if (!cache || !cache->entries) {
        return NULL;
    }
    
    for (int i = 0; i < cache->num_entries; i++) {
        CalibrationCacheEntry *entry = &cache->entries[i];
        
        if (entry->channel_index == channel_index &&
            fabs(entry->wavelength - wavelength) < 0.1f &&
            entry->type == type) {
            
            // 更新访问时间和计数
            entry->last_access_time = time(NULL);
            entry->access_count++;
            
            return entry;
        }
    }
    
    return NULL;
}

/**
 * @brief 添加缓存条目（内部函数）
 */
static CalibrationError add_cache_entry(
    CalibrationCache *cache,
    int channel_index,
    float wavelength,
    CalibrationCacheType type,
    const void *data,
    size_t data_size)
{
    if (!cache || !data || data_size == 0) {
        return CALIB_ERROR_INVALID_PARAM;
    }
    
    // 检查缓存大小限制
    if (cache->current_size + data_size > cache->max_size) {
        // 需要清理一些缓存条目
        evict_cache_entries(cache, data_size);
    }
    
    // 扩展缓存条目数组
    CalibrationCacheEntry *new_entries = (CalibrationCacheEntry*)realloc(
        cache->entries,
        (cache->num_entries + 1) * sizeof(CalibrationCacheEntry));
    
    if (!new_entries) {
        return CALIB_ERROR_MEMORY_ALLOCATION;
    }
    
    cache->entries = new_entries;
    
    // 添加新条目
    CalibrationCacheEntry *entry = &cache->entries[cache->num_entries];
    entry->channel_index = channel_index;
    entry->wavelength = wavelength;
    entry->type = type;
    entry->data_size = data_size;
    entry->creation_time = time(NULL);
    entry->last_access_time = entry->creation_time;
    entry->access_count = 0;
    
    // 复制数据
    entry->data = malloc(data_size);
    if (!entry->data) {
        return CALIB_ERROR_MEMORY_ALLOCATION;
    }
    memcpy(entry->data, data, data_size);
    
    cache->num_entries++;
    cache->current_size += data_size;
    
    return CALIB_SUCCESS;
}

/**
 * @brief 清理缓存条目（内部函数）
 */
static void evict_cache_entries(CalibrationCache *cache, size_t required_size) {
    if (!cache || cache->num_entries == 0) {
        return;
    }
    
    // 使用LRU策略清理缓存
    // 找到最少使用的条目
    while (cache->current_size + required_size > cache->max_size && 
           cache->num_entries > 0) {
        
        int lru_index = 0;
        time_t oldest_time = cache->entries[0].last_access_time;
        
        for (int i = 1; i < cache->num_entries; i++) {
            if (cache->entries[i].last_access_time < oldest_time) {
                oldest_time = cache->entries[i].last_access_time;
                lru_index = i;
            }
        }
        
        // 删除LRU条目
        CalibrationCacheEntry *entry = &cache->entries[lru_index];
        cache->current_size -= entry->data_size;
        
        if (entry->data) {
            free(entry->data);
        }
        
        // 移动后续条目
        for (int i = lru_index; i < cache->num_entries - 1; i++) {
            cache->entries[i] = cache->entries[i + 1];
        }
        
        cache->num_entries--;
    }
}

// ============================================================================
// 性能优化 - 并行处理
// ============================================================================

#ifdef CALIB_ENABLE_OPENMP
#include <omp.h>

CalibrationError calib_manager_set_num_threads(
    CalibrationManager *manager,
    int num_threads)
{
    CALIB_CHECK_NULL(manager);
    
    if (num_threads <= 0) {
        num_threads = omp_get_max_threads();
    }
    
    manager->num_threads = num_threads;
    omp_set_num_threads(num_threads);
    
    return CALIB_SUCCESS;
}

int calib_manager_get_num_threads(const CalibrationManager *manager) {
    if (!manager) {
        return 1;
    }
    return manager->num_threads;
}

/**
 * @brief 并行PSF卷积（内部函数）
 */
static CalibrationError perform_psf_convolution_parallel(
    const float *input,
    int width,
    int height,
    const PSFData *psf,
    float *output,
    int num_threads)
{
    int psf_width = psf->width;
    int psf_height = psf->height;
    int psf_cx = (int)psf->center_x;
    int psf_cy = (int)psf->center_y;
    
    // 初始化输出
    memset(output, 0, width * height * sizeof(float));
    
    // 并行处理每一行
    #pragma omp parallel for num_threads(num_threads) schedule(dynamic)
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float sum = 0.0f;
            float weight_sum = 0.0f;
            
            // 遍历PSF窗口
            for (int py = 0; py < psf_height; py++) {
                for (int px = 0; px < psf_width; px++) {
                    int ix = x + px - psf_cx;
                    int iy = y + py - psf_cy;
                    
                    if (ix >= 0 && ix < width && iy >= 0 && iy < height) {
                        float psf_value = psf->data[py * psf_width + px];
                        float input_value = input[iy * width + ix];
                        
                        sum += input_value * psf_value;
                        weight_sum += psf_value;
                    }
                }
            }
            
            if (!psf->is_normalized && weight_sum > 0.0f) {
                sum /= weight_sum;
            }
            
            output[y * width + x] = sum;
        }
    }
    
    return CALIB_SUCCESS;
}

/**
 * @brief 并行批量处理（内部函数）
 */
static CalibrationError apply_batch_parallel(
    CalibrationManager *manager,
    const float **input_images,
    int num_images,
    int width,
    int height,
    int channel_index,
    const float *wavelengths,
    CalibrationPipelineFlags flags,
    float **output_images,
    int num_threads)
{
    CalibrationError err = CALIB_SUCCESS;
    
    #pragma omp parallel for num_threads(num_threads) schedule(dynamic)
    for (int i = 0; i < num_images; i++) {
        if (!input_images[i] || !output_images[i]) {
            continue;
        }
        
        float wavelength = wavelengths ? wavelengths[i] : 0.0f;
        
        CalibrationError local_err = calib_manager_apply_full_pipeline(
            manager,
            input_images[i],
            width,
            height,
            channel_index,
            wavelength,
            flags,
            output_images[i]);
        
        if (local_err != CALIB_SUCCESS) {
            #pragma omp critical
            {
                err = local_err;
            }
        }
    }
    
    return err;
}

#else // !CALIB_ENABLE_OPENMP

CalibrationError calib_manager_set_num_threads(
    CalibrationManager *manager,
    int num_threads)
{
    CALIB_CHECK_NULL(manager);
    manager->num_threads = 1;
    return CALIB_SUCCESS;
}

int calib_manager_get_num_threads(const CalibrationManager *manager) {
    return 1;
}

#endif // CALIB_ENABLE_OPENMP

// ============================================================================
// 性能优化 - GPU加速
// ============================================================================

#ifdef CALIB_ENABLE_CUDA

CalibrationError calib_manager_enable_gpu(
    CalibrationManager *manager,
    int device_id)
{
    CALIB_CHECK_NULL(manager);
    
    // 初始化CUDA设备
    cudaError_t cuda_err = cudaSetDevice(device_id);
    if (cuda_err != cudaSuccess) {
        return CALIB_ERROR_GPU_INIT_FAILED;
    }
    
    manager->gpu_enabled = true;
    manager->gpu_device_id = device_id;
    
    return CALIB_SUCCESS;
}

CalibrationError calib_manager_disable_gpu(CalibrationManager *manager) {
    CALIB_CHECK_NULL(manager);
    
    if (manager->gpu_enabled) {
        cudaDeviceReset();
        manager->gpu_enabled = false;
        manager->gpu_device_id = -1;
    }
    
    return CALIB_SUCCESS;
}

bool calib_manager_is_gpu_available(void) {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    return (err == cudaSuccess && device_count > 0);
}

/**
 * @brief GPU PSF卷积（内部函数）
 */
static CalibrationError perform_psf_convolution_gpu(
    const float *input,
    int width,
    int height,
    const PSFData *psf,
    float *output)
{
    // 分配GPU内存
    float *d_input = NULL;
    float *d_output = NULL;
    float *d_psf = NULL;
    
    size_t image_size = width * height * sizeof(float);
    size_t psf_size = psf->width * psf->height * sizeof(float);
    
    cudaError_t err;
    
    err = cudaMalloc(&d_input, image_size);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaMalloc(&d_output, image_size);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaMalloc(&d_psf, psf_size);
    if (err != cudaSuccess) goto cleanup;
    
    // 复制数据到GPU
    err = cudaMemcpy(d_input, input, image_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) goto cleanup;
    
    err = cudaMemcpy(d_psf, psf->data, psf_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) goto cleanup;
    
    // 执行GPU卷积
    err = cuda_convolve_2d(
        d_input, width, height,
        d_psf, psf->width, psf->height,
        psf->center_x, psf->center_y,
        psf->is_normalized,
        d_output);
    
    if (err != cudaSuccess) goto cleanup;
    
    // 复制结果回CPU
    err = cudaMemcpy(output, d_output, image_size, cudaMemcpyDeviceToHost);
    
cleanup:
    if (d_input) cudaFree(d_input);
    if (d_output) cudaFree(d_output);
    if (d_psf) cudaFree(d_psf);
    
    return (err == cudaSuccess) ? CALIB_SUCCESS : CALIB_ERROR_GPU_OPERATION_FAILED;
}

#else // !CALIB_ENABLE_CUDA

CalibrationError calib_manager_enable_gpu(
    CalibrationManager *manager,
    int device_id)
{
    return CALIB_ERROR_NOT_SUPPORTED;
}

CalibrationError calib_manager_disable_gpu(CalibrationManager *manager) {
    return CALIB_ERROR_NOT_SUPPORTED;
}

bool calib_manager_is_gpu_available(void) {
    return false;
}

#endif // CALIB_ENABLE_CUDA

// ============================================================================
// 性能分析
// ============================================================================

CalibrationError calib_manager_enable_profiling(
    CalibrationManager *manager,
    bool enable)
{
    CALIB_CHECK_NULL(manager);
    
    manager->profiling_enabled = enable;
    
    if (enable && !manager->profile_data) {
        manager->profile_data = (CalibrationProfileData*)calloc(
            1, sizeof(CalibrationProfileData));
        if (!manager->profile_data) {
            return CALIB_ERROR_MEMORY_ALLOCATION;
        }
    }
    
    return CALIB_SUCCESS;
}

CalibrationError calib_manager_reset_profiling(CalibrationManager *manager) {
    CALIB_CHECK_NULL(manager);
    
    if (manager->profile_data) {
        memset(manager->profile_data, 0, sizeof(CalibrationProfileData));
    }
    
    return CALIB_SUCCESS;
}

CalibrationError calib_manager_get_profile_data(
    const CalibrationManager *manager,
    CalibrationProfileData *profile_data)
{
    CALIB_CHECK_NULL(manager);
    CALIB_CHECK_NULL(profile_data);
    
    if (!manager->profile_data) {
        return CALIB_ERROR_PROFILING_NOT_ENABLED;
    }
    
    memcpy(profile_data, manager->profile_data, sizeof(CalibrationProfileData));
    
    return CALIB_SUCCESS;
}

void calib_manager_print_profile_data(const CalibrationManager *manager) {
    if (!manager || !manager->profile_data) {
        printf("Profiling not enabled\n");
        return;
    }
    
    CalibrationProfileData *data = manager->profile_data;
    
    printf("=== Calibration Performance Profile ===\n");
    printf("PSF Convolution:\n");
    printf("  Count: %d\n", data->psf_convolution_count);
    printf("  Total time: %.3f ms\n", data->psf_convolution_time * 1000.0);
    if (data->psf_convolution_count > 0) {
        printf("  Average time: %.3f ms\n", 
               (data->psf_convolution_time / data->psf_convolution_count) * 1000.0);
    }
    
    printf("\nColor Correction:\n");
    printf("  Count: %d\n", data->color_correction_count);
    printf("  Total time: %.3f ms\n", data->color_correction_time * 1000.0);
    if (data->color_correction_count > 0) {
        printf("  Average time: %.3f ms\n",
               (data->color_correction_time / data->color_correction_count) * 1000.0);
    }
    
    printf("\nAberration Correction:\n");
    printf("  Count: %d\n", data->aberration_correction_count);
    printf("  Total time: %.3f ms\n", data->aberration_correction_time * 1000.0);
    if (data->aberration_correction_count > 0) {
        printf("  Average time: %.3f ms\n",
               (data->aberration_correction_time / data->aberration_correction_count) * 1000.0);
    }
    
    printf("\nSpectral Correction:\n");
    printf("  Count: %d\n", data->spectral_correction_count);
    printf("  Total time: %.3f ms\n", data->spectral_correction_time * 1000.0);
    if (data->spectral_correction_count > 0) {
        printf("  Average time: %.3f ms\n",
               (data->spectral_correction_time / data->spectral_correction_count) * 1000.0);
    }
    
    double total_time = data->psf_convolution_time + 
                       data->color_correction_time +
                       data->aberration_correction_time +
                       data->spectral_correction_time;
    
    printf("\nTotal processing time: %.3f ms\n", total_time * 1000.0);
    
    if (manager->cache) {
        printf("\nCache Statistics:\n");
        printf("  Entries: %d\n", manager->cache->num_entries);
        printf("  Size: %.2f MB / %.2f MB\n",
               manager->cache->current_size / (1024.0 * 1024.0),
               manager->cache->max_size / (1024.0 * 1024.0));
    }
    
    printf("======================================\n");
}

/**
 * @brief 记录性能数据（内部函数）
 */
static void record_profile_time(
    CalibrationProfileData *profile,
    CalibrationOperationType op_type,
    double elapsed_time)
{
    if (!profile) {
        return;
    }
    
    switch (op_type) {
        case CALIB_OP_PSF_CONVOLUTION:
            profile->psf_convolution_count++;
            profile->psf_convolution_time += elapsed_time;
            break;
            
        case CALIB_OP_COLOR_CORRECTION:
            profile->color_correction_count++;
            profile->color_correction_time += elapsed_time;
            break;
            
        case CALIB_OP_ABERRATION_CORRECTION:
            profile->aberration_correction_count++;
            profile->aberration_correction_time += elapsed_time;
            break;
            
        case CALIB_OP_SPECTRAL_CORRECTION:
            profile->spectral_correction_count++;
            profile->spectral_correction_time += elapsed_time;
            break;
            
        default:
            break;
    }
}
// ============================================================================
// 质量评估
// ============================================================================

CalibrationError calib_manager_evaluate_quality(
    CalibrationManager *manager,
    int channel_index,
    CalibrationQualityMetrics *metrics)
{
    CALIB_CHECK_NULL(manager);
    CALIB_CHECK_NULL(metrics);
    
    // 获取通道
    ChannelCalibration *channel = calib_data_get_channel(
        manager->calib_data, channel_index);
    if (!channel) {
        return CALIB_ERROR_CHANNEL_NOT_FOUND;
    }
    
    // 初始化指标
    memset(metrics, 0, sizeof(CalibrationQualityMetrics));
    
    // 评估平场质量
    if (channel->has_flat_field) {
        metrics->flat_field_quality = evaluate_flat_field_quality(
            channel->flat_field, 
            channel->flat_field_width,
            channel->flat_field_height);
    }
    
    // 评估暗场质量
    if (channel->has_dark_field) {
        metrics->dark_field_quality = evaluate_dark_field_quality(
            channel->dark_field,
            channel->dark_field_width,
            channel->dark_field_height);
    }
    
    // 评估PSF质量
    metrics->psf_quality_count = 0;
    for (int i = 0; i < channel->num_wavelengths; i++) {
        WavelengthCalibration *wl = &channel->wavelengths[i];
        if (wl->psf.data) {
            float psf_quality = evaluate_psf_quality(&wl->psf);
            metrics->psf_quality_sum += psf_quality;
            metrics->psf_quality_count++;
            
            if (psf_quality < metrics->min_psf_quality || 
                metrics->psf_quality_count == 1) {
                metrics->min_psf_quality = psf_quality;
            }
            if (psf_quality > metrics->max_psf_quality) {
                metrics->max_psf_quality = psf_quality;
            }
        }
    }
    
    if (metrics->psf_quality_count > 0) {
        metrics->avg_psf_quality = metrics->psf_quality_sum / metrics->psf_quality_count;
    }
    
    // 评估颜色矩阵质量
    if (channel->has_color_matrix) {
        metrics->color_matrix_quality = evaluate_color_matrix_quality(
            channel->color_matrix);
    }
    
    // 评估像差校正质量
    metrics->aberration_quality_count = 0;
    for (int i = 0; i < channel->num_wavelengths; i++) {
        WavelengthCalibration *wl = &channel->wavelengths[i];
        float aberration_quality = evaluate_aberration_quality(
            wl->aberration_coeff);
        
        if (aberration_quality > 0.0f) {
            metrics->aberration_quality_sum += aberration_quality;
            metrics->aberration_quality_count++;
        }
    }
    
    if (metrics->aberration_quality_count > 0) {
        metrics->avg_aberration_quality = 
            metrics->aberration_quality_sum / metrics->aberration_quality_count;
    }
    
    // 计算总体质量分数 (0-100)
    float quality_sum = 0.0f;
    int quality_count = 0;
    
    if (channel->has_flat_field) {
        quality_sum += metrics->flat_field_quality;
        quality_count++;
    }
    if (channel->has_dark_field) {
        quality_sum += metrics->dark_field_quality;
        quality_count++;
    }
    if (metrics->psf_quality_count > 0) {
        quality_sum += metrics->avg_psf_quality;
        quality_count++;
    }
    if (channel->has_color_matrix) {
        quality_sum += metrics->color_matrix_quality;
        quality_count++;
    }
    if (metrics->aberration_quality_count > 0) {
        quality_sum += metrics->avg_aberration_quality;
        quality_count++;
    }
    
    metrics->overall_quality = quality_count > 0 ? 
        quality_sum / quality_count : 0.0f;
    
    return CALIB_SUCCESS;
}

/**
 * @brief 评估平场质量（内部函数）
 */
static float evaluate_flat_field_quality(
    const float *flat_field,
    int width,
    int height)
{
    if (!flat_field || width <= 0 || height <= 0) {
        return 0.0f;
    }
    
    int num_pixels = width * height;
    
    // 计算统计信息
    float mean = 0.0f;
    float min_val = FLT_MAX;
    float max_val = -FLT_MAX;
    
    for (int i = 0; i < num_pixels; i++) {
        float val = flat_field[i];
        mean += val;
        if (val < min_val) min_val = val;
        if (val > max_val) max_val = val;
    }
    mean /= num_pixels;
    
    // 计算标准差
    float variance = 0.0f;
    for (int i = 0; i < num_pixels; i++) {
        float diff = flat_field[i] - mean;
        variance += diff * diff;
    }
    variance /= num_pixels;
    float std_dev = sqrtf(variance);
    
    // 计算变异系数 (CV)
    float cv = (mean > 0.0f) ? (std_dev / mean) : 1.0f;
    
    // 计算均匀性分数 (0-100)
    // CV越小，均匀性越好
    float uniformity_score = fmaxf(0.0f, 100.0f * (1.0f - cv));
    
    // 计算动态范围分数
    float dynamic_range = (max_val > 0.0f) ? (max_val - min_val) / max_val : 0.0f;
    float dynamic_range_score = fminf(100.0f, dynamic_range * 100.0f);
    
    // 综合质量分数
    float quality = (uniformity_score * 0.7f + dynamic_range_score * 0.3f);
    
    return fmaxf(0.0f, fminf(100.0f, quality));
}

/**
 * @brief 评估暗场质量（内部函数）
 */
static float evaluate_dark_field_quality(
    const float *dark_field,
    int width,
    int height)
{
    if (!dark_field || width <= 0 || height <= 0) {
        return 0.0f;
    }
    
    int num_pixels = width * height;
    
    // 计算统计信息
    float mean = 0.0f;
    float max_val = -FLT_MAX;
    
    for (int i = 0; i < num_pixels; i++) {
        float val = dark_field[i];
        mean += val;
        if (val > max_val) max_val = val;
    }
    mean /= num_pixels;
    
    // 计算标准差
    float variance = 0.0f;
    for (int i = 0; i < num_pixels; i++) {
        float diff = dark_field[i] - mean;
        variance += diff * diff;
    }
    variance /= num_pixels;
    float std_dev = sqrtf(variance);
    
    // 暗场应该接近零且噪声小
    // 计算低噪声分数
    float noise_score = fmaxf(0.0f, 100.0f * (1.0f - std_dev / 255.0f));
    
    // 计算低偏置分数
    float bias_score = fmaxf(0.0f, 100.0f * (1.0f - mean / 255.0f));
    
    // 综合质量分数
    float quality = (noise_score * 0.6f + bias_score * 0.4f);
    
    return fmaxf(0.0f, fminf(100.0f, quality));
}

/**
 * @brief 评估PSF质量（内部函数）
 */
static float evaluate_psf_quality(const PSFData *psf) {
    if (!psf || !psf->data || psf->width <= 0 || psf->height <= 0) {
        return 0.0f;
    }
    
    int num_pixels = psf->width * psf->height;
    
    // 计算PSF的总能量
    float total_energy = 0.0f;
    float max_val = -FLT_MAX;
    int max_idx = 0;
    
    for (int i = 0; i < num_pixels; i++) {
        float val = psf->data[i];
        total_energy += val;
        if (val > max_val) {
            max_val = val;
            max_idx = i;
        }
    }
    
    if (total_energy <= 0.0f) {
        return 0.0f;
    }
    
    // 计算峰值位置
    int peak_x = max_idx % psf->width;
    int peak_y = max_idx / psf->width;
    
    // 计算中心偏移
    float center_offset = sqrtf(
        (peak_x - psf->center_x) * (peak_x - psf->center_x) +
        (peak_y - psf->center_y) * (peak_y - psf->center_y));
    
    float max_offset = sqrtf(psf->width * psf->width + psf->height * psf->height) / 2.0f;
    float centering_score = fmaxf(0.0f, 100.0f * (1.0f - center_offset / max_offset));
    
    // 计算能量集中度（中心区域的能量占比）
    int center_radius = fminf(psf->width, psf->height) / 4;
    float center_energy = 0.0f;
    
    for (int y = 0; y < psf->height; y++) {
        for (int x = 0; x < psf->width; x++) {
            float dx = x - psf->center_x;
            float dy = y - psf->center_y;
            float dist = sqrtf(dx * dx + dy * dy);
            
            if (dist <= center_radius) {
                center_energy += psf->data[y * psf->width + x];
            }
        }
    }
    
    float concentration_ratio = center_energy / total_energy;
    float concentration_score = fminf(100.0f, concentration_ratio * 100.0f);
    
    // 计算对称性
    float symmetry_score = evaluate_psf_symmetry(psf);
    
    // 综合质量分数
    float quality = (centering_score * 0.3f + 
                    concentration_score * 0.4f + 
                    symmetry_score * 0.3f);
    
    return fmaxf(0.0f, fminf(100.0f, quality));
}

/**
 * @brief 评估PSF对称性（内部函数）
 */
static float evaluate_psf_symmetry(const PSFData *psf) {
    if (!psf || !psf->data) {
        return 0.0f;
    }
    
    int cx = (int)psf->center_x;
    int cy = (int)psf->center_y;
    
    float symmetry_error = 0.0f;
    int comparison_count = 0;
    
    // 比较关于中心对称的点
    for (int y = 0; y < psf->height; y++) {
        for (int x = 0; x < psf->width; x++) {
            int mirror_x = 2 * cx - x;
            int mirror_y = 2 * cy - y;
            
            if (mirror_x >= 0 && mirror_x < psf->width &&
                mirror_y >= 0 && mirror_y < psf->height) {
                
                float val1 = psf->data[y * psf->width + x];
                float val2 = psf->data[mirror_y * psf->width + mirror_x];
                
                float avg = (val1 + val2) / 2.0f;
                if (avg > 0.0f) {
                    symmetry_error += fabsf(val1 - val2) / avg;
                    comparison_count++;
                }
            }
        }
    }
    
    if (comparison_count == 0) {
        return 0.0f;
    }
    
    float avg_error = symmetry_error / comparison_count;
    float symmetry_score = fmaxf(0.0f, 100.0f * (1.0f - avg_error));
    
    return symmetry_score;
}

/**
 * @brief 评估颜色矩阵质量（内部函数）
 */
static float evaluate_color_matrix_quality(const float *color_matrix) {
    if (!color_matrix) {
        return 0.0f;
    }
    
    // 计算矩阵的行列式
    float det = color_matrix[0] * (color_matrix[4] * color_matrix[8] - 
                                   color_matrix[5] * color_matrix[7]) -
                color_matrix[1] * (color_matrix[3] * color_matrix[8] - 
                                   color_matrix[5] * color_matrix[6]) +
                color_matrix[2] * (color_matrix[3] * color_matrix[7] - 
                                   color_matrix[4] * color_matrix[6]);
    
    // 行列式接近1表示矩阵保持了亮度
    float det_score = fmaxf(0.0f, 100.0f * (1.0f - fabsf(det - 1.0f)));
    
    // 计算对角线元素的平均值（应该接近1）
    float diag_avg = (color_matrix[0] + color_matrix[4] + color_matrix[8]) / 3.0f;
    float diag_score = fmaxf(0.0f, 100.0f * (1.0f - fabsf(diag_avg - 1.0f)));
    
    // 计算非对角线元素的平均值（应该接近0）
    float off_diag_avg = (fabsf(color_matrix[1]) + fabsf(color_matrix[2]) +
                          fabsf(color_matrix[3]) + fabsf(color_matrix[5]) +
                          fabsf(color_matrix[6]) + fabsf(color_matrix[7])) / 6.0f;
    float off_diag_score = fmaxf(0.0f, 100.0f * (1.0f - off_diag_avg));
    
    // 综合质量分数
    float quality = (det_score * 0.4f + diag_score * 0.3f + off_diag_score * 0.3f);
    
    return fmaxf(0.0f, fminf(100.0f, quality));
}

/**
 * @brief 评估像差校正质量（内部函数）
 */
static float evaluate_aberration_quality(const float *aberration_coeff) {
    if (!aberration_coeff) {
        return 0.0f;
    }
    
    // 计算像差系数的总幅度
    float total_aberration = 0.0f;
    for (int i = 0; i < CALIB_MAX_ABERRATION_TERMS; i++) {
        total_aberration += fabsf(aberration_coeff[i]);
    }
    
    // 像差越小，质量越高
    // 假设总像差小于0.1为优秀
    float quality = fmaxf(0.0f, 100.0f * (1.0f - total_aberration / 0.1f));
    
    return fmaxf(0.0f, fminf(100.0f, quality));
}

void calib_manager_print_quality_metrics(const CalibrationQualityMetrics *metrics) {
    if (!metrics) {
        printf("No quality metrics available\n");
        return;
    }
    
    printf("=== Calibration Quality Metrics ===\n");
    printf("Overall Quality: %.2f/100\n", metrics->overall_quality);
    printf("\nFlat Field Quality: %.2f/100\n", metrics->flat_field_quality);
    printf("Dark Field Quality: %.2f/100\n", metrics->dark_field_quality);
    
    if (metrics->psf_quality_count > 0) {
        printf("\nPSF Quality:\n");
        printf("  Average: %.2f/100\n", metrics->avg_psf_quality);
        printf("  Min: %.2f/100\n", metrics->min_psf_quality);
        printf("  Max: %.2f/100\n", metrics->max_psf_quality);
        printf("  Count: %d\n", metrics->psf_quality_count);
    }
    
    printf("\nColor Matrix Quality: %.2f/100\n", metrics->color_matrix_quality);
    
    if (metrics->aberration_quality_count > 0) {
        printf("\nAberration Correction Quality:\n");
        printf("  Average: %.2f/100\n", metrics->avg_aberration_quality);
        printf("  Count: %d\n", metrics->aberration_quality_count);
    }
    
    printf("===================================\n");
}

// ============================================================================
// 诊断和调试
// ============================================================================

CalibrationError calib_manager_diagnose(
    CalibrationManager *manager,
    int channel_index,
    CalibrationDiagnostics *diagnostics)
{
    CALIB_CHECK_NULL(manager);
    CALIB_CHECK_NULL(diagnostics);
    
    // 初始化诊断结果
    memset(diagnostics, 0, sizeof(CalibrationDiagnostics));
    
    // 检查通道
    ChannelCalibration *channel = calib_data_get_channel(
        manager->calib_data, channel_index);
    
    if (!channel) {
        diagnostics->has_errors = true;
        snprintf(diagnostics->error_messages[diagnostics->num_errors++],
                CALIB_MAX_ERROR_MESSAGE_LENGTH,
                "Channel %d not found", channel_index);
        return CALIB_ERROR_CHANNEL_NOT_FOUND;
    }
    
    diagnostics->channel_exists = true;
    
    // 检查平场
    if (channel->has_flat_field) {
        diagnostics->has_flat_field = true;
        
        if (channel->flat_field_width <= 0 || channel->flat_field_height <= 0) {
            diagnostics->has_warnings = true;
            snprintf(diagnostics->warning_messages[diagnostics->num_warnings++],
                    CALIB_MAX_ERROR_MESSAGE_LENGTH,
                    "Invalid flat field dimensions");
        }
        
        // 检查平场数据范围
        float min_val = FLT_MAX;
        float max_val = -FLT_MAX;
        int num_pixels = channel->flat_field_width * channel->flat_field_height;
        
        for (int i = 0; i < num_pixels; i++) {
            float val = channel->flat_field[i];
            if (val < min_val) min_val = val;
            if (val > max_val) max_val = val;
        }
        
        if (min_val <= 0.0f) {
            diagnostics->has_warnings = true;
            snprintf(diagnostics->warning_messages[diagnostics->num_warnings++],
                    CALIB_MAX_ERROR_MESSAGE_LENGTH,
                    "Flat field contains non-positive values");
        }
    } else {
        diagnostics->has_warnings = true;
        snprintf(diagnostics->warning_messages[diagnostics->num_warnings++],
                CALIB_MAX_ERROR_MESSAGE_LENGTH,
                "No flat field calibration");
    }
    
    // 检查暗场
    if (channel->has_dark_field) {
        diagnostics->has_dark_field = true;
        
        if (channel->dark_field_width <= 0 || channel->dark_field_height <= 0) {
            diagnostics->has_warnings = true;
            snprintf(diagnostics->warning_messages[diagnostics->num_warnings++],
                    CALIB_MAX_ERROR_MESSAGE_LENGTH,
                    "Invalid dark field dimensions");
        }
    } else {
        diagnostics->has_warnings = true;
        snprintf(diagnostics->warning_messages[diagnostics->num_warnings++],
                CALIB_MAX_ERROR_MESSAGE_LENGTH,
                "No dark field calibration");
    }
    
    // 检查波长校准
    diagnostics->num_wavelengths = channel->num_wavelengths;
    
    if (channel->num_wavelengths == 0) {
        diagnostics->has_warnings = true;
        snprintf(diagnostics->warning_messages[diagnostics->num_warnings++],
                CALIB_MAX_ERROR_MESSAGE_LENGTH,
                "No wavelength calibrations");
    }
    
    // 检查每个波长的PSF
    diagnostics->num_psf = 0;
    for (int i = 0; i < channel->num_wavelengths; i++) {
        WavelengthCalibration *wl = &channel->wavelengths[i];
        
        if (wl->psf.data) {
            diagnostics->num_psf++;
            diagnostics->has_psf = true;
            
            // 检查PSF有效性
            if (wl->psf.width <= 0 || wl->psf.height <= 0) {
                diagnostics->has_errors = true;
                snprintf(diagnostics->error_messages[diagnostics->num_errors++],
                        CALIB_MAX_ERROR_MESSAGE_LENGTH,
                        "Invalid PSF dimensions at wavelength %.1f nm",
                        wl->wavelength);
            }
        }
    }
    
    // 检查颜色矩阵
    if (channel->has_color_matrix) {
        diagnostics->has_color_matrix = true;
        
        // 检查矩阵的行列式
        float det = channel->color_matrix[0] * 
                   (channel->color_matrix[4] * channel->color_matrix[8] - 
                    channel->color_matrix[5] * channel->color_matrix[7]) -
                   channel->color_matrix[1] * 
                   (channel->color_matrix[3] * channel->color_matrix[8] - 
                    channel->color_matrix[5] * channel->color_matrix[6]) +
                   channel->color_matrix[2] * 
                   (channel->color_matrix[3] * channel->color_matrix[7] - 
                    channel->color_matrix[4] * channel->color_matrix[6]);
        
        if (fabsf(det) < 1e-6f) {
            diagnostics->has_errors = true;
            snprintf(diagnostics->error_messages[diagnostics->num_errors++],
                    CALIB_MAX_ERROR_MESSAGE_LENGTH,
                    "Color matrix is singular (det = %.6f)", det);
        }
    }
    
    // 检查内存使用
    diagnostics->memory_usage = estimate_memory_usage(manager);
    
    // 检查缓存状态
    if (manager->cache) {
        diagnostics->cache_size = manager->cache->current_size;
        diagnostics->cache_entries = manager->cache->num_entries;
    }
    
    return CALIB_SUCCESS;
}

void calib_manager_print_diagnostics(const CalibrationDiagnostics *diagnostics) {
    if (!diagnostics) {
        printf("No diagnostics available\n");
        return;
    }
    
    printf("=== Calibration Diagnostics ===\n");
    
    printf("\nStatus:\n");
    printf("  Channel exists: %s\n", diagnostics->channel_exists ? "Yes" : "No");
    printf("  Has flat field: %s\n", diagnostics->has_flat_field ? "Yes" : "No");
    printf("  Has dark field: %s\n", diagnostics->has_dark_field ? "Yes" : "No");
    printf("  Has PSF: %s\n", diagnostics->has_psf ? "Yes" : "No");
    printf("  Has color matrix: %s\n", diagnostics->has_color_matrix ? "Yes" : "No");
    
    printf("\nStatistics:\n");
    printf("  Number of wavelengths: %d\n", diagnostics->num_wavelengths);
    printf("  Number of PSFs: %d\n", diagnostics->num_psf);
    printf("  Memory usage: %.2f MB\n", diagnostics->memory_usage / (1024.0 * 1024.0));
    
    if (diagnostics->cache_entries > 0) {
        printf("  Cache entries: %d\n", diagnostics->cache_entries);
        printf("  Cache size: %.2f MB\n", diagnostics->cache_size / (1024.0 * 1024.0));
    }
    
    if (diagnostics->has_errors) {
        printf("\nErrors:\n");
        for (int i = 0; i < diagnostics->num_errors; i++) {
            printf("  [ERROR] %s\n", diagnostics->error_messages[i]);
        }
    }
    
    if (diagnostics->has_warnings) {
        printf("\nWarnings:\n");
        for (int i = 0; i < diagnostics->num_warnings; i++) {
            printf("  [WARNING] %s\n", diagnostics->warning_messages[i]);
        }
    }
    
    if (!diagnostics->has_errors && !diagnostics->has_warnings) {
        printf("\nNo issues detected.\n");
    }
    
    printf("================================\n");
}

/**
 * @brief 估算内存使用（内部函数）
 */
static size_t estimate_memory_usage(const CalibrationManager *manager) {
    if (!manager || !manager->calib_data) {
        return 0;
    }
    
    size_t total_size = sizeof(CalibrationManager);
    total_size += sizeof(CalibrationData);
    
    // 计算所有通道的内存使用
    for (int i = 0; i < manager->calib_data->num_channels; i++) {
        ChannelCalibration *channel = &manager->calib_data->channels[i];
        
        total_size += sizeof(ChannelCalibration);
        
        // 平场
        if (channel->has_flat_field) {
            total_size += channel->flat_field_width * 
                         channel->flat_field_height * sizeof(float);
        }
        
        // 暗场
        if (channel->has_dark_field) {
            total_size += channel->dark_field_width * 
                         channel->dark_field_height * sizeof(float);
        }
        
        // 波长校准
        for (int j = 0; j < channel->num_wavelengths; j++) {
            WavelengthCalibration *wl = &channel->wavelengths[j];
            
            total_size += sizeof(WavelengthCalibration);
            
            // PSF
            if (wl->psf.data) {
                total_size += wl->psf.width * wl->psf.height * sizeof(float);
            }
        }
    }
    
    // 缓存
    if (manager->cache) {
        total_size += sizeof(CalibrationCache);
        total_size += manager->cache->current_size;
    }
    
    // 性能分析数据
    if (manager->profile_data) {
        total_size += sizeof(CalibrationProfileData);
    }
    
    return total_size;
}
// ============================================================================
// 实用工具函数
// ============================================================================

CalibrationError calib_manager_compare_images(
    const float *image1,
    const float *image2,
    int width,
    int height,
    CalibrationImageComparisonMetrics *metrics)
{
    CALIB_CHECK_NULL(image1);
    CALIB_CHECK_NULL(image2);
    CALIB_CHECK_NULL(metrics);
    
    if (width <= 0 || height <= 0) {
        return CALIB_ERROR_INVALID_PARAM;
    }
    
    int num_pixels = width * height;
    
    // 初始化指标
    memset(metrics, 0, sizeof(CalibrationImageComparisonMetrics));
    
    // 计算基本统计
    double sum_diff = 0.0;
    double sum_abs_diff = 0.0;
    double sum_squared_diff = 0.0;
    
    float max_diff = -FLT_MAX;
    float min_diff = FLT_MAX;
    
    for (int i = 0; i < num_pixels; i++) {
        float diff = image2[i] - image1[i];
        float abs_diff = fabsf(diff);
        
        sum_diff += diff;
        sum_abs_diff += abs_diff;
        sum_squared_diff += diff * diff;
        
        if (diff > max_diff) max_diff = diff;
        if (diff < min_diff) min_diff = diff;
    }
    
    // 计算均值和标准差
    metrics->mean_difference = sum_diff / num_pixels;
    metrics->mean_absolute_difference = sum_abs_diff / num_pixels;
    metrics->max_difference = max_diff;
    metrics->min_difference = min_diff;
    
    // 计算RMSE
    metrics->rmse = sqrt(sum_squared_diff / num_pixels);
    
    // 计算PSNR
    float max_pixel_value = 0.0f;
    for (int i = 0; i < num_pixels; i++) {
        if (image1[i] > max_pixel_value) max_pixel_value = image1[i];
        if (image2[i] > max_pixel_value) max_pixel_value = image2[i];
    }
    
    if (metrics->rmse > 0.0f && max_pixel_value > 0.0f) {
        metrics->psnr = 20.0f * log10f(max_pixel_value / metrics->rmse);
    } else {
        metrics->psnr = INFINITY;
    }
    
    // 计算SSIM (结构相似性指数)
    metrics->ssim = calculate_ssim(image1, image2, width, height);
    
    // 计算相关系数
    double mean1 = 0.0, mean2 = 0.0;
    for (int i = 0; i < num_pixels; i++) {
        mean1 += image1[i];
        mean2 += image2[i];
    }
    mean1 /= num_pixels;
    mean2 /= num_pixels;
    
    double cov = 0.0, var1 = 0.0, var2 = 0.0;
    for (int i = 0; i < num_pixels; i++) {
        double diff1 = image1[i] - mean1;
        double diff2 = image2[i] - mean2;
        cov += diff1 * diff2;
        var1 += diff1 * diff1;
        var2 += diff2 * diff2;
    }
    
    if (var1 > 0.0 && var2 > 0.0) {
        metrics->correlation = cov / sqrt(var1 * var2);
    } else {
        metrics->correlation = 0.0f;
    }
    
    return CALIB_SUCCESS;
}

/**
 * @brief 计算SSIM（结构相似性指数）
 */
static float calculate_ssim(
    const float *image1,
    const float *image2,
    int width,
    int height)
{
    const float C1 = 6.5025f;  // (0.01 * 255)^2
    const float C2 = 58.5225f; // (0.03 * 255)^2
    
    int num_pixels = width * height;
    
    // 计算均值
    double mean1 = 0.0, mean2 = 0.0;
    for (int i = 0; i < num_pixels; i++) {
        mean1 += image1[i];
        mean2 += image2[i];
    }
    mean1 /= num_pixels;
    mean2 /= num_pixels;
    
    // 计算方差和协方差
    double var1 = 0.0, var2 = 0.0, cov = 0.0;
    for (int i = 0; i < num_pixels; i++) {
        double diff1 = image1[i] - mean1;
        double diff2 = image2[i] - mean2;
        var1 += diff1 * diff1;
        var2 += diff2 * diff2;
        cov += diff1 * diff2;
    }
    var1 /= num_pixels;
    var2 /= num_pixels;
    cov /= num_pixels;
    
    // 计算SSIM
    float numerator = (2.0f * mean1 * mean2 + C1) * (2.0f * cov + C2);
    float denominator = (mean1 * mean1 + mean2 * mean2 + C1) * (var1 + var2 + C2);
    
    if (denominator > 0.0f) {
        return numerator / denominator;
    } else {
        return 1.0f;
    }
}

void calib_manager_print_comparison_metrics(
    const CalibrationImageComparisonMetrics *metrics)
{
    if (!metrics) {
        printf("No comparison metrics available\n");
        return;
    }
    
    printf("=== Image Comparison Metrics ===\n");
    printf("Mean Difference: %.6f\n", metrics->mean_difference);
    printf("Mean Absolute Difference: %.6f\n", metrics->mean_absolute_difference);
    printf("RMSE: %.6f\n", metrics->rmse);
    printf("PSNR: %.2f dB\n", metrics->psnr);
    printf("SSIM: %.6f\n", metrics->ssim);
    printf("Correlation: %.6f\n", metrics->correlation);
    printf("Max Difference: %.6f\n", metrics->max_difference);
    printf("Min Difference: %.6f\n", metrics->min_difference);
    printf("================================\n");
}

// ============================================================================
// 图像转换工具
// ============================================================================

CalibrationError calib_manager_convert_to_uint8(
    const float *input,
    int num_pixels,
    uint8_t *output,
    float min_val,
    float max_val)
{
    CALIB_CHECK_NULL(input);
    CALIB_CHECK_NULL(output);
    
    if (num_pixels <= 0) {
        return CALIB_ERROR_INVALID_PARAM;
    }
    
    // 如果未指定范围，自动计算
    if (min_val >= max_val) {
        min_val = FLT_MAX;
        max_val = -FLT_MAX;
        
        for (int i = 0; i < num_pixels; i++) {
            if (input[i] < min_val) min_val = input[i];
            if (input[i] > max_val) max_val = input[i];
        }
    }
    
    float range = max_val - min_val;
    if (range <= 0.0f) {
        // 所有像素值相同
        memset(output, 128, num_pixels);
        return CALIB_SUCCESS;
    }
    
    // 转换
    for (int i = 0; i < num_pixels; i++) {
        float normalized = (input[i] - min_val) / range;
        normalized = fmaxf(0.0f, fminf(1.0f, normalized));
        output[i] = (uint8_t)(normalized * 255.0f + 0.5f);
    }
    
    return CALIB_SUCCESS;
}

CalibrationError calib_manager_convert_to_uint16(
    const float *input,
    int num_pixels,
    uint16_t *output,
    float min_val,
    float max_val)
{
    CALIB_CHECK_NULL(input);
    CALIB_CHECK_NULL(output);
    
    if (num_pixels <= 0) {
        return CALIB_ERROR_INVALID_PARAM;
    }
    
    // 如果未指定范围，自动计算
    if (min_val >= max_val) {
        min_val = FLT_MAX;
        max_val = -FLT_MAX;
        
        for (int i = 0; i < num_pixels; i++) {
            if (input[i] < min_val) min_val = input[i];
            if (input[i] > max_val) max_val = input[i];
        }
    }
    
    float range = max_val - min_val;
    if (range <= 0.0f) {
        // 所有像素值相同
        for (int i = 0; i < num_pixels; i++) {
            output[i] = 32768;
        }
        return CALIB_SUCCESS;
    }
    
    // 转换
    for (int i = 0; i < num_pixels; i++) {
        float normalized = (input[i] - min_val) / range;
        normalized = fmaxf(0.0f, fminf(1.0f, normalized));
        output[i] = (uint16_t)(normalized * 65535.0f + 0.5f);
    }
    
    return CALIB_SUCCESS;
}

CalibrationError calib_manager_convert_from_uint8(
    const uint8_t *input,
    int num_pixels,
    float *output,
    float min_val,
    float max_val)
{
    CALIB_CHECK_NULL(input);
    CALIB_CHECK_NULL(output);
    
    if (num_pixels <= 0) {
        return CALIB_ERROR_INVALID_PARAM;
    }
    
    float range = max_val - min_val;
    
    for (int i = 0; i < num_pixels; i++) {
        float normalized = input[i] / 255.0f;
        output[i] = min_val + normalized * range;
    }
    
    return CALIB_SUCCESS;
}

CalibrationError calib_manager_convert_from_uint16(
    const uint16_t *input,
    int num_pixels,
    float *output,
    float min_val,
    float max_val)
{
    CALIB_CHECK_NULL(input);
    CALIB_CHECK_NULL(output);
    
    if (num_pixels <= 0) {
        return CALIB_ERROR_INVALID_PARAM;
    }
    
    float range = max_val - min_val;
    
    for (int i = 0; i < num_pixels; i++) {
        float normalized = input[i] / 65535.0f;
        output[i] = min_val + normalized * range;
    }
    
    return CALIB_SUCCESS;
}

// ============================================================================
// 图像处理工具
// ============================================================================

CalibrationError calib_manager_resize_image(
    const float *input,
    int input_width,
    int input_height,
    float *output,
    int output_width,
    int output_height,
    CalibrationInterpolationMethod method)
{
    CALIB_CHECK_NULL(input);
    CALIB_CHECK_NULL(output);
    
    if (input_width <= 0 || input_height <= 0 ||
        output_width <= 0 || output_height <= 0) {
        return CALIB_ERROR_INVALID_PARAM;
    }
    
    float x_ratio = (float)input_width / output_width;
    float y_ratio = (float)input_height / output_height;
    
    switch (method) {
        case CALIB_INTERP_NEAREST:
            for (int y = 0; y < output_height; y++) {
                for (int x = 0; x < output_width; x++) {
                    int src_x = (int)(x * x_ratio);
                    int src_y = (int)(y * y_ratio);
                    
                    src_x = fminf(src_x, input_width - 1);
                    src_y = fminf(src_y, input_height - 1);
                    
                    output[y * output_width + x] = 
                        input[src_y * input_width + src_x];
                }
            }
            break;
            
        case CALIB_INTERP_BILINEAR:
            for (int y = 0; y < output_height; y++) {
                for (int x = 0; x < output_width; x++) {
                    float src_x = x * x_ratio;
                    float src_y = y * y_ratio;
                    
                    int x0 = (int)src_x;
                    int y0 = (int)src_y;
                    int x1 = fminf(x0 + 1, input_width - 1);
                    int y1 = fminf(y0 + 1, input_height - 1);
                    
                    float fx = src_x - x0;
                    float fy = src_y - y0;
                    
                    float v00 = input[y0 * input_width + x0];
                    float v10 = input[y0 * input_width + x1];
                    float v01 = input[y1 * input_width + x0];
                    float v11 = input[y1 * input_width + x1];
                    
                    float v0 = v00 * (1.0f - fx) + v10 * fx;
                    float v1 = v01 * (1.0f - fx) + v11 * fx;
                    
                    output[y * output_width + x] = v0 * (1.0f - fy) + v1 * fy;
                }
            }
            break;
            
        case CALIB_INTERP_BICUBIC:
            // 双三次插值实现
            for (int y = 0; y < output_height; y++) {
                for (int x = 0; x < output_width; x++) {
                    float src_x = x * x_ratio;
                    float src_y = y * y_ratio;
                    
                    int x0 = (int)src_x;
                    int y0 = (int)src_y;
                    
                    float fx = src_x - x0;
                    float fy = src_y - y0;
                    
                    float sum = 0.0f;
                    float weight_sum = 0.0f;
                    
                    // 4x4邻域
                    for (int dy = -1; dy <= 2; dy++) {
                        for (int dx = -1; dx <= 2; dx++) {
                            int px = x0 + dx;
                            int py = y0 + dy;
                            
                            if (px >= 0 && px < input_width &&
                                py >= 0 && py < input_height) {
                                
                                float wx = cubic_weight(fx - dx);
                                float wy = cubic_weight(fy - dy);
                                float weight = wx * wy;
                                
                                sum += input[py * input_width + px] * weight;
                                weight_sum += weight;
                            }
                        }
                    }
                    
                    if (weight_sum > 0.0f) {
                        output[y * output_width + x] = sum / weight_sum;
                    } else {
                        output[y * output_width + x] = 0.0f;
                    }
                }
            }
            break;
            
        default:
            return CALIB_ERROR_INVALID_PARAM;
    }
    
    return CALIB_SUCCESS;
}

/**
 * @brief 三次插值权重函数
 */
static float cubic_weight(float x) {
    x = fabsf(x);
    
    if (x <= 1.0f) {
        return 1.0f - 2.0f * x * x + x * x * x;
    } else if (x < 2.0f) {
        return 4.0f - 8.0f * x + 5.0f * x * x - x * x * x;
    } else {
        return 0.0f;
    }
}

CalibrationError calib_manager_crop_image(
    const float *input,
    int input_width,
    int input_height,
    int crop_x,
    int crop_y,
    int crop_width,
    int crop_height,
    float *output)
{
    CALIB_CHECK_NULL(input);
    CALIB_CHECK_NULL(output);
    
    if (input_width <= 0 || input_height <= 0 ||
        crop_width <= 0 || crop_height <= 0) {
        return CALIB_ERROR_INVALID_PARAM;
    }
    
    // 验证裁剪区域
    if (crop_x < 0 || crop_y < 0 ||
        crop_x + crop_width > input_width ||
        crop_y + crop_height > input_height) {
        return CALIB_ERROR_INVALID_PARAM;
    }
    
    // 复制裁剪区域
    for (int y = 0; y < crop_height; y++) {
        for (int x = 0; x < crop_width; x++) {
            int src_idx = (crop_y + y) * input_width + (crop_x + x);
            int dst_idx = y * crop_width + x;
            output[dst_idx] = input[src_idx];
        }
    }
    
    return CALIB_SUCCESS;
}

CalibrationError calib_manager_rotate_image(
    const float *input,
    int width,
    int height,
    float angle_degrees,
    float *output,
    float background_value)
{
    CALIB_CHECK_NULL(input);
    CALIB_CHECK_NULL(output);
    
    if (width <= 0 || height <= 0) {
        return CALIB_ERROR_INVALID_PARAM;
    }
    
    // 转换角度为弧度
    float angle_rad = angle_degrees * M_PI / 180.0f;
    float cos_angle = cosf(angle_rad);
    float sin_angle = sinf(angle_rad);
    
    // 图像中心
    float cx = width / 2.0f;
    float cy = height / 2.0f;
    
    // 初始化输出为背景值
    for (int i = 0; i < width * height; i++) {
        output[i] = background_value;
    }
    
    // 旋转
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            // 相对于中心的坐标
            float dx = x - cx;
            float dy = y - cy;
            
            // 反向旋转找到源坐标
            float src_x = dx * cos_angle + dy * sin_angle + cx;
            float src_y = -dx * sin_angle + dy * cos_angle + cy;
            
            // 双线性插值
            if (src_x >= 0 && src_x < width - 1 &&
                src_y >= 0 && src_y < height - 1) {
                
                int x0 = (int)src_x;
                int y0 = (int)src_y;
                int x1 = x0 + 1;
                int y1 = y0 + 1;
                
                float fx = src_x - x0;
                float fy = src_y - y0;
                
                float v00 = input[y0 * width + x0];
                float v10 = input[y0 * width + x1];
                float v01 = input[y1 * width + x0];
                float v11 = input[y1 * width + x1];
                
                float v0 = v00 * (1.0f - fx) + v10 * fx;
                float v1 = v01 * (1.0f - fx) + v11 * fx;
                
                output[y * width + x] = v0 * (1.0f - fy) + v1 * fy;
            }
        }
    }
    
    return CALIB_SUCCESS;
}

// ============================================================================
// 版本信息
// ============================================================================

const char* calib_manager_get_version(void) {
    return CALIB_VERSION_STRING;
}

void calib_manager_get_version_info(CalibrationVersionInfo *info) {
    if (!info) {
        return;
    }
    
    info->major = CALIB_VERSION_MAJOR;
    info->minor = CALIB_VERSION_MINOR;
    info->patch = CALIB_VERSION_PATCH;
    strncpy(info->version_string, CALIB_VERSION_STRING, 
            sizeof(info->version_string) - 1);
    info->version_string[sizeof(info->version_string) - 1] = '\0';
    
#ifdef CALIB_ENABLE_OPENMP
    info->has_openmp = true;
#else
    info->has_openmp = false;
#endif

#ifdef CALIB_ENABLE_CUDA
    info->has_cuda = true;
#else
    info->has_cuda = false;
#endif

#ifdef CALIB_ENABLE_SIMD
    info->has_simd = true;
#else
    info->has_simd = false;
#endif
}

void calib_manager_print_version_info(void) {
    CalibrationVersionInfo info;
    calib_manager_get_version_info(&info);
    
    printf("=== Calibration Library Version ===\n");
    printf("Version: %s\n", info.version_string);
    printf("Major: %d\n", info.major);
    printf("Minor: %d\n", info.minor);
    printf("Patch: %d\n", info.patch);
    printf("\nFeatures:\n");
    printf("  OpenMP: %s\n", info.has_openmp ? "Enabled" : "Disabled");
    printf("  CUDA: %s\n", info.has_cuda ? "Enabled" : "Disabled");
    printf("  SIMD: %s\n", info.has_simd ? "Enabled" : "Disabled");
    printf("====================================\n");
}

// ============================================================================
// 错误处理
// ============================================================================

const char* calib_error_string(CalibrationError error) {
    switch (error) {
        case CALIB_SUCCESS:
            return "Success";
        case CALIB_ERROR_NULL_POINTER:
            return "Null pointer";
        case CALIB_ERROR_INVALID_PARAM:
            return "Invalid parameter";
        case CALIB_ERROR_MEMORY_ALLOCATION:
            return "Memory allocation failed";
        case CALIB_ERROR_FILE_NOT_FOUND:
            return "File not found";
        case CALIB_ERROR_FILE_READ:
            return "File read error";
        case CALIB_ERROR_FILE_WRITE:
            return "File write error";
        case CALIB_ERROR_INVALID_FORMAT:
            return "Invalid file format";
        case CALIB_ERROR_CHANNEL_NOT_FOUND:
            return "Channel not found";
        case CALIB_ERROR_WAVELENGTH_NOT_FOUND:
            return "Wavelength not found";
        case CALIB_ERROR_DIMENSION_MISMATCH:
            return "Dimension mismatch";
        case CALIB_ERROR_NOT_INITIALIZED:
            return "Not initialized";
        case CALIB_ERROR_ALREADY_INITIALIZED:
            return "Already initialized";
        case CALIB_ERROR_OPERATION_FAILED:
            return "Operation failed";
        case CALIB_ERROR_NOT_SUPPORTED:
            return "Not supported";
        case CALIB_ERROR_GPU_INIT_FAILED:
            return "GPU initialization failed";
        case CALIB_ERROR_GPU_OPERATION_FAILED:
            return "GPU operation failed";
        case CALIB_ERROR_PROFILING_NOT_ENABLED:
            return "Profiling not enabled";
        case CALIB_ERROR_OPERATION_CANCELLED:
            return "Operation cancelled";
        default:
            return "Unknown error";
    }
}

void calib_print_error(CalibrationError error, const char *context) {
    if (error != CALIB_SUCCESS) {
        if (context) {
            fprintf(stderr, "[CALIB ERROR] %s: %s\n", 
                    context, calib_error_string(error));
        } else {
            fprintf(stderr, "[CALIB ERROR] %s\n", 
                    calib_error_string(error));
        }
    }
}

// ============================================================================
// 清理和销毁
// ============================================================================

void calib_manager_destroy(CalibrationManager *manager) {
    if (!manager) {
        return;
    }
    
    // 清理缓存
    if (manager->cache) {
        calib_manager_clear_cache(manager);
        free(manager->cache);
    }
    
    // 清理性能分析数据
    if (manager->profile_data) {
        free(manager->profile_data);
    }
    
    // 清理校准数据
    if (manager->calib_data) {
        calib_data_destroy(manager->calib_data);
    }
    
    // 禁用GPU（如果启用）
#ifdef CALIB_ENABLE_CUDA
    if (manager->gpu_enabled) {
        calib_manager_disable_gpu(manager);
    }
#endif
    
    // 释放管理器本身
    free(manager);
}

// ============================================================================
// 模块初始化和清理
// ============================================================================

static bool g_calib_module_initialized = false;

CalibrationError calib_module_init(void) {
    if (g_calib_module_initialized) {
        return CALIB_ERROR_ALREADY_INITIALIZED;
    }
    
    // 初始化全局资源
    // （如果需要的话）
    
    g_calib_module_initialized = true;
    
    return CALIB_SUCCESS;
}

void calib_module_cleanup(void) {
    if (!g_calib_module_initialized) {
        return;
    }
    
    // 清理全局资源
    // （如果需要的话）
    
    g_calib_module_initialized = false;
}

bool calib_module_is_initialized(void) {
    return g_calib_module_initialized;
}

