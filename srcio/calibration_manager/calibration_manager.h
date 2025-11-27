/**
 * @file calibration_manager.h
 * @brief 校准数据管理模块头文件
 * @details 提供校准数据的加载、保存、管理和应用功能
 * 
 * 功能包括：
 * - 校准文件的读写
 * - 校准数据的验证和管理
 * - PSF（点扩散函数）数据管理
 * - 波长相关的校准数据
 * - 多通道校准支持
 * - 校准数据插值和应用
 * 
 * @author hany
 * @date 2024
 * @version 1.0
 */

#ifndef CALIBRATION_MANAGER_H
#define CALIBRATION_MANAGER_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

// ============================================================================
// 版本信息
// ============================================================================

#define CALIBRATION_MANAGER_VERSION_MAJOR 1
#define CALIBRATION_MANAGER_VERSION_MINOR 0
#define CALIBRATION_MANAGER_VERSION_PATCH 0

// ============================================================================
// 常量定义
// ============================================================================

#define CALIB_MAX_WAVELENGTHS 32        ///< 最大波长数量
#define CALIB_MAX_CHANNELS 4            ///< 最大通道数（RGB + NIR）
#define CALIB_MAX_NAME_LENGTH 256       ///< 最大名称长度
#define CALIB_MAX_DESCRIPTION_LENGTH 1024 ///< 最大描述长度
#define CALIB_FILE_MAGIC 0x43414C42     ///< 文件魔数 "CALB"
#define CALIB_FILE_VERSION 1            ///< 文件格式版本

// ============================================================================
// 错误码定义
// ============================================================================

typedef enum {
    CALIB_SUCCESS = 0,                  ///< 成功
    CALIB_ERROR_INVALID_PARAM,          ///< 无效参数
    CALIB_ERROR_FILE_NOT_FOUND,         ///< 文件未找到
    CALIB_ERROR_FILE_READ,              ///< 文件读取错误
    CALIB_ERROR_FILE_WRITE,             ///< 文件写入错误
    CALIB_ERROR_INVALID_FORMAT,         ///< 无效的文件格式
    CALIB_ERROR_VERSION_MISMATCH,       ///< 版本不匹配
    CALIB_ERROR_MEMORY_ALLOCATION,      ///< 内存分配失败
    CALIB_ERROR_DATA_CORRUPTED,         ///< 数据损坏
    CALIB_ERROR_NOT_INITIALIZED,        ///< 未初始化
    CALIB_ERROR_ALREADY_INITIALIZED,    ///< 已经初始化
    CALIB_ERROR_WAVELENGTH_NOT_FOUND,   ///< 波长未找到
    CALIB_ERROR_CHANNEL_NOT_FOUND,      ///< 通道未找到
    CALIB_ERROR_INVALID_PSF,            ///< 无效的PSF数据
    CALIB_ERROR_DIMENSION_MISMATCH,     ///< 维度不匹配
    CALIB_ERROR_CHECKSUM_FAILED         ///< 校验和失败
} CalibrationError;

// ============================================================================
// 校准类型定义
// ============================================================================

typedef enum {
    CALIB_TYPE_UNKNOWN = 0,             ///< 未知类型
    CALIB_TYPE_MEASURED,                ///< 实测校准数据
    CALIB_TYPE_DESIGN,                  ///< 基于设计数据的校准
    CALIB_TYPE_HYBRID,                  ///< 混合校准（实测+设计）
    CALIB_TYPE_AUTO_GENERATED           ///< 自动生成的校准
} CalibrationType;

// ============================================================================
// 插值方法定义
// ============================================================================

typedef enum {
    CALIB_INTERP_NEAREST = 0,           ///< 最近邻插值
    CALIB_INTERP_LINEAR,                ///< 线性插值
    CALIB_INTERP_CUBIC,                 ///< 三次插值
    CALIB_INTERP_SPLINE                 ///< 样条插值
} CalibrationInterpMethod;

// ============================================================================
// PSF数据结构
// ============================================================================

/**
 * @brief 点扩散函数（PSF）数据结构
 */
typedef struct {
    int width;                          ///< PSF宽度
    int height;                         ///< PSF高度
    float *data;                        ///< PSF数据（归一化）
    float center_x;                     ///< PSF中心X坐标
    float center_y;                     ///< PSF中心Y坐标
    float wavelength;                   ///< 对应波长（纳米）
    float numerical_aperture;           ///< 数值孔径
    float pixel_size;                   ///< 像素尺寸（微米）
    bool is_normalized;                 ///< 是否已归一化
} PSFData;

// ============================================================================
// 波长校准数据
// ============================================================================

/**
 * @brief 单个波长的校准数据
 */
typedef struct {
    float wavelength;                   ///< 波长（纳米）
    PSFData psf;                        ///< 对应的PSF
    float diffraction_limit;            ///< 衍射极限（纳米）
    float aberration_coeff[10];         ///< 像差系数（Zernike多项式）
    void *custom_data;                  ///< 自定义数据指针
    size_t custom_data_size;            ///< 自定义数据大小
} WavelengthCalibration;

// ============================================================================
// 通道校准数据
// ============================================================================

/**
 * @brief 单个通道的校准数据
 */
typedef struct {
    char name[64];                      ///< 通道名称（如"Red", "Green", "Blue"）
    int channel_index;                  ///< 通道索引
    int num_wavelengths;                ///< 波长数量
    WavelengthCalibration *wavelengths; ///< 波长校准数据数组
    float spectral_response[256];       ///< 光谱响应曲线
    float color_matrix[9];              ///< 颜色矩阵（3x3）
    bool has_color_matrix;              ///< 是否有颜色矩阵
} ChannelCalibration;

// ============================================================================
// 校准元数据
// ============================================================================

/**
 * @brief 校准文件元数据
 */
typedef struct {
    char name[CALIB_MAX_NAME_LENGTH];           ///< 校准名称
    char description[CALIB_MAX_DESCRIPTION_LENGTH]; ///< 描述信息
    CalibrationType type;                       ///< 校准类型
    char device_model[128];                     ///< 设备型号
    char lens_model[128];                       ///< 镜头型号
    char creation_date[32];                     ///< 创建日期
    char author[128];                           ///< 作者
    int version_major;                          ///< 主版本号
    int version_minor;                          ///< 次版本号
    uint32_t checksum;                          ///< 校验和
} CalibrationMetadata;

// ============================================================================
// 校准数据主结构
// ============================================================================

/**
 * @brief 完整的校准数据结构
 */
typedef struct {
    CalibrationMetadata metadata;       ///< 元数据
    int num_channels;                   ///< 通道数量
    ChannelCalibration *channels;       ///< 通道校准数据数组
    
    // 全局参数
    float sensor_width;                 ///< 传感器宽度（毫米）
    float sensor_height;                ///< 传感器高度（毫米）
    float pixel_pitch;                  ///< 像素间距（微米）
    int image_width;                    ///< 图像宽度（像素）
    int image_height;                   ///< 图像高度（像素）
    
    // 光学参数
    float focal_length;                 ///< 焦距（毫米）
    float f_number;                     ///< 光圈数
    float working_distance;             ///< 工作距离（毫米）
    
    // 内部状态
    bool is_valid;                      ///< 数据是否有效
    void *internal_data;                ///< 内部数据指针
} CalibrationData;

// ============================================================================
// 校准管理器句柄
// ============================================================================

typedef struct CalibrationManager CalibrationManager;

// ============================================================================
// 初始化和清理函数
// ============================================================================

/**
 * @brief 创建校准管理器
 * @return 校准管理器句柄，失败返回NULL
 */
CalibrationManager* calib_manager_create(void);

/**
 * @brief 销毁校准管理器
 * @param manager 校准管理器句柄
 */
void calib_manager_destroy(CalibrationManager *manager);

/**
 * @brief 初始化校准管理器
 * @param manager 校准管理器句柄
 * @return 错误码
 */
CalibrationError calib_manager_init(CalibrationManager *manager);

/**
 * @brief 清理校准管理器资源
 * @param manager 校准管理器句柄
 */
void calib_manager_cleanup(CalibrationManager *manager);

// ============================================================================
// 校准数据创建和销毁
// ============================================================================

/**
 * @brief 创建空的校准数据结构
 * @param num_channels 通道数量
 * @return 校准数据指针，失败返回NULL
 */
CalibrationData* calib_data_create(int num_channels);

/**
 * @brief 销毁校准数据
 * @param data 校准数据指针
 */
void calib_data_destroy(CalibrationData *data);

/**
 * @brief 复制校准数据
 * @param src 源校准数据
 * @return 新的校准数据副本，失败返回NULL
 */
CalibrationData* calib_data_clone(const CalibrationData *src);

/**
 * @brief 验证校准数据的完整性
 * @param data 校准数据指针
 * @return true表示有效，false表示无效
 */
bool calib_data_validate(const CalibrationData *data);

// ============================================================================
// 文件IO操作
// ============================================================================

/**
 * @brief 从文件加载校准数据
 * @param manager 校准管理器句柄
 * @param filename 文件路径
 * @param data 输出的校准数据指针
 * @return 错误码
 */
CalibrationError calib_manager_load(CalibrationManager *manager,
                                    const char *filename,
                                    CalibrationData **data);

/**
 * @brief 保存校准数据到文件
 * @param manager 校准管理器句柄
 * @param filename 文件路径
 * @param data 校准数据指针
 * @return 错误码
 */
CalibrationError calib_manager_save(CalibrationManager *manager,
                                    const char *filename,
                                    const CalibrationData *data);

/**
 * @brief 从JSON文件加载校准数据
 * @param manager 校准管理器句柄
 * @param filename JSON文件路径
 * @param data 输出的校准数据指针
 * @return 错误码
 */
CalibrationError calib_manager_load_json(CalibrationManager *manager,
                                         const char *filename,
                                         CalibrationData **data);

/**
 * @brief 保存校准数据为JSON格式
 * @param manager 校准管理器句柄
 * @param filename JSON文件路径
 * @param data 校准数据指针
 * @return 错误码
 */
CalibrationError calib_manager_save_json(CalibrationManager *manager,
                                         const char *filename,
                                         const CalibrationData *data);

// ============================================================================
// PSF操作函数
// ============================================================================

/**
 * @brief 创建PSF数据
 * @param width PSF宽度
 * @param height PSF高度
 * @return PSF数据指针，失败返回NULL
 */
PSFData* psf_create(int width, int height);

/**
 * @brief 销毁PSF数据
 * @param psf PSF数据指针
 */
void psf_destroy(PSFData *psf);

/**
 * @brief 归一化PSF数据
 * @param psf PSF数据指针
 * @return 错误码
 */
CalibrationError psf_normalize(PSFData *psf);

/**
 * @brief 复制PSF数据
 * @param src 源PSF数据
 * @return 新的PSF数据副本，失败返回NULL
 */
PSFData* psf_clone(const PSFData *src);

/**
 * @brief 从数组加载PSF数据
 * @param psf PSF数据指针
 * @param data 数据数组
 * @param width 宽度
 * @param height 高度
 * @return 错误码
 */
CalibrationError psf_load_from_array(PSFData *psf, 
                                     const float *data,
                                     int width, 
                                     int height);

/**
 * @brief 生成理论PSF（Airy盘）
 * @param wavelength 波长（纳米）
 * @param numerical_aperture 数值孔径
 * @param pixel_size 像素尺寸（微米）
 * @param size PSF尺寸
 * @return PSF数据指针，失败返回NULL
 */
PSFData* psf_generate_airy(float wavelength,
                           float numerical_aperture,
                           float pixel_size,
                           int size);

// ============================================================================
// 通道和波长操作
// ============================================================================

/**
 * @brief 添加通道校准数据
 * @param data 校准数据指针
 * @param channel_name 通道名称
 * @param channel_index 通道索引
 * @return 错误码
 */
CalibrationError calib_add_channel(CalibrationData *data,
                                   const char *channel_name,
                                   int channel_index);

/**
 * @brief 获取通道校准数据
 * @param data 校准数据指针
 * @param channel_index 通道索引
 * @return 通道校准数据指针，失败返回NULL
 */
ChannelCalibration* calib_get_channel(const CalibrationData *data,
                                      int channel_index);

/**
 * @brief 添加波长校准数据到通道
 * @param channel 通道校准数据指针
 * @param wavelength 波长（纳米）
 * @param psf PSF数据指针
 * @return 错误码
 */
CalibrationError calib_add_wavelength(ChannelCalibration *channel,
                                      float wavelength,
                                      const PSFData *psf);

/**
 * @brief 获取指定波长的校准数据
 * @param channel 通道校准数据指针
 * @param wavelength 波长（纳米）
 * @param method 插值方法
 * @return 波长校准数据指针，失败返回NULL
 */
WavelengthCalibration* calib_get_wavelength(const ChannelCalibration *channel,
                                            float wavelength,
                                            CalibrationInterpMethod method);

// ============================================================================
// 插值和应用函数
// ============================================================================

/**
 * @brief 对PSF进行插值
 * @param psf1 第一个PSF
 * @param psf2 第二个PSF
 * @param weight 插值权重（0-1）
 * @return 插值后的PSF，失败返回NULL
 */
PSFData* psf_interpolate(const PSFData *psf1,
                        const PSFData *psf2,
                        float weight);

/**
 * @brief 应用校准数据到图像
 * @param manager 校准管理器句柄
 * @param data 校准数据指针
 * @param image 输入图像数据
 * @param width 图像宽度
 * @param height 图像高度
 * @param channels 图像通道数
 * @param output 输出图像数据
 * @return 错误码
 */
CalibrationError calib_apply_to_image(CalibrationManager *manager,
                                      const CalibrationData *data,
                                      const float *image,
                                      int width,
                                      int height,
                                      int channels,
                                      float *output);

// ============================================================================
// 工具函数
// ============================================================================

/**
 * @brief 获取错误描述字符串
 * @param error 错误码
 * @return 错误描述字符串
 */
const char* calib_error_string(CalibrationError error);

/**
 * @brief 打印校准数据信息
 * @param data 校准数据指针
 */
void calib_print_info(const CalibrationData *data);

/**
 * @brief 计算校准数据的校验和
 * @param data 校准数据指针
 * @return 校验和值
 */
uint32_t calib_calculate_checksum(const CalibrationData *data);

/**
 * @brief 验证校准数据的校验和
 * @param data 校准数据指针
 * @return true表示校验通过，false表示失败
 */
bool calib_verify_checksum(const CalibrationData *data);

/**
 * @brief 获取校准管理器版本
 * @return 版本字符串
 */
const char* calib_manager_get_version(void);

/**
 * @brief 设置详细输出模式
 * @param verbose true启用详细输出，false禁用
 */
void calib_set_verbose(bool verbose);

// ============================================================================
// 高级功能
// ============================================================================

/**
 * @brief 合并多个校准数据
 * @param manager 校准管理器句柄
 * @param data_array 校准数据数组
 * @param count 数组大小
 * @param output 输出的合并校准数据
 * @return 错误码
 */
CalibrationError calib_merge(CalibrationManager *manager,
                             const CalibrationData **data_array,
                             int count,
                             CalibrationData **output);

/**
 * @brief 从设计数据生成校准
 * @param manager 校准管理器句柄
 * @param design_file 设计数据文件路径
 * @param output 输出的校准数据
 * @return 错误码
 */
CalibrationError calib_from_design_data(CalibrationManager *manager,
                                        const char *design_file,
                                        CalibrationData **output);

/**
 * @brief 优化校准数据
 * @param manager 校准管理器句柄
 * @param data 校准数据指针
 * @param reference_images 参考图像数组
 * @param num_images 图像数量
 * @return 错误码
 */
CalibrationError calib_optimize(CalibrationManager *manager,
                                CalibrationData *data,
                                const float **reference_images,
                                int num_images);

#ifdef __cplusplus
}
#endif

#endif // CALIBRATION_MANAGER_H
