/**
 * @file jpeg_handler.h
 * @brief JPEG图像处理模块头文件
 * @author hany
 * @date 2025
 * 
 * 提供JPEG图像的读取、写入、压缩、解压缩和元数据处理功能
 * 支持标准JPEG、渐进式JPEG、EXIF元数据等
 */

#ifndef JPEG_HANDLER_H
#define JPEG_HANDLER_H

#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include "image.h"

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// 宏定义
// ============================================================================

/** JPEG库版本 */
#define JPEG_HANDLER_VERSION "1.0.0"

/** 默认JPEG质量 */
#define JPEG_DEFAULT_QUALITY 85

/** 最小JPEG质量 */
#define JPEG_MIN_QUALITY 1

/** 最大JPEG质量 */
#define JPEG_MAX_QUALITY 100

/** 最大EXIF数据大小 */
#define JPEG_MAX_EXIF_SIZE (64 * 1024)

/** 最大注释长度 */
#define JPEG_MAX_COMMENT_LENGTH 256

/** JPEG文件魔数 */
#define JPEG_MAGIC_NUMBER 0xFFD8

// ============================================================================
// 枚举类型
// ============================================================================

/**
 * @brief JPEG色彩空间
 */
typedef enum {
    JPEG_COLORSPACE_UNKNOWN = 0,    /**< 未知色彩空间 */
    JPEG_COLORSPACE_GRAYSCALE = 1,  /**< 灰度 */
    JPEG_COLORSPACE_RGB = 2,        /**< RGB */
    JPEG_COLORSPACE_YCbCr = 3,      /**< YCbCr (标准JPEG) */
    JPEG_COLORSPACE_CMYK = 4,       /**< CMYK */
    JPEG_COLORSPACE_YCCK = 5        /**< YCCK */
} JpegColorSpace;

/**
 * @brief JPEG子采样模式
 */
typedef enum {
    JPEG_SUBSAMPLE_444 = 0,  /**< 4:4:4 (无子采样) */
    JPEG_SUBSAMPLE_422 = 1,  /**< 4:2:2 (水平2倍子采样) */
    JPEG_SUBSAMPLE_420 = 2,  /**< 4:2:0 (水平和垂直2倍子采样) */
    JPEG_SUBSAMPLE_411 = 3,  /**< 4:1:1 (水平4倍子采样) */
    JPEG_SUBSAMPLE_440 = 4   /**< 4:4:0 (垂直2倍子采样) */
} JpegSubsampling;

/**
 * @brief JPEG DCT方法
 */
typedef enum {
    JPEG_DCT_ISLOW = 0,      /**< 慢速整数DCT (最精确) */
    JPEG_DCT_IFAST = 1,      /**< 快速整数DCT */
    JPEG_DCT_FLOAT = 2       /**< 浮点DCT (最快但精度较低) */
} JpegDctMethod;

/**
 * @brief JPEG标记类型
 */
typedef enum {
    JPEG_MARKER_SOI = 0xFFD8,   /**< Start of Image */
    JPEG_MARKER_EOI = 0xFFD9,   /**< End of Image */
    JPEG_MARKER_SOS = 0xFFDA,   /**< Start of Scan */
    JPEG_MARKER_DQT = 0xFFDB,   /**< Define Quantization Table */
    JPEG_MARKER_DHT = 0xFFC4,   /**< Define Huffman Table */
    JPEG_MARKER_SOF0 = 0xFFC0,  /**< Start of Frame (Baseline) */
    JPEG_MARKER_SOF2 = 0xFFC2,  /**< Start of Frame (Progressive) */
    JPEG_MARKER_APP0 = 0xFFE0,  /**< JFIF marker */
    JPEG_MARKER_APP1 = 0xFFE1,  /**< EXIF marker */
    JPEG_MARKER_COM = 0xFFFE    /**< Comment */
} JpegMarker;

/**
 * @brief EXIF方向
 */
typedef enum {
    EXIF_ORIENTATION_NORMAL = 1,           /**< 正常 */
    EXIF_ORIENTATION_FLIP_HORIZONTAL = 2,  /**< 水平翻转 */
    EXIF_ORIENTATION_ROTATE_180 = 3,       /**< 旋转180度 */
    EXIF_ORIENTATION_FLIP_VERTICAL = 4,    /**< 垂直翻转 */
    EXIF_ORIENTATION_TRANSPOSE = 5,        /**< 转置 */
    EXIF_ORIENTATION_ROTATE_90 = 6,        /**< 顺时针旋转90度 */
    EXIF_ORIENTATION_TRANSVERSE = 7,       /**< 横向转置 */
    EXIF_ORIENTATION_ROTATE_270 = 8        /**< 顺时针旋转270度 */
} ExifOrientation;

// ============================================================================
// 结构体定义
// ============================================================================

/**
 * @brief JPEG压缩选项
 */
typedef struct {
    int quality;                    /**< 压缩质量 (1-100) */
    bool progressive;               /**< 是否使用渐进式JPEG */
    bool optimize_coding;           /**< 是否优化霍夫曼编码 */
    JpegSubsampling subsampling;    /**< 色度子采样模式 */
    JpegDctMethod dct_method;       /**< DCT变换方法 */
    int smoothing_factor;           /**< 平滑因子 (0-100) */
    bool arithmetic_coding;         /**< 是否使用算术编码 */
    int restart_interval;           /**< 重启间隔 (0=禁用) */
} JpegCompressOptions;

/**
 * @brief JPEG解压缩选项
 */
typedef struct {
    bool do_fancy_upsampling;       /**< 是否使用高质量上采样 */
    bool do_block_smoothing;        /**< 是否进行块平滑 */
    JpegDctMethod dct_method;       /**< DCT变换方法 */
    bool two_pass_quantize;         /**< 是否使用两遍量化 */
    int dither_mode;                /**< 抖动模式 */
    int scale_num;                  /**< 缩放分子 */
    int scale_denom;                /**< 缩放分母 (支持1/1, 1/2, 1/4, 1/8) */
    bool buffered_image;            /**< 是否使用缓冲图像模式 */
} JpegDecompressOptions;

/**
 * @brief EXIF GPS信息
 */
typedef struct {
    bool has_gps;                   /**< 是否包含GPS信息 */
    double latitude;                /**< 纬度 */
    double longitude;               /**< 经度 */
    double altitude;                /**< 海拔 (米) */
    char latitude_ref[2];           /**< 纬度参考 (N/S) */
    char longitude_ref[2];          /**< 经度参考 (E/W) */
    char altitude_ref;              /**< 海拔参考 (0=海平面以上, 1=海平面以下) */
    char timestamp[32];             /**< GPS时间戳 */
    char datestamp[32];             /**< GPS日期戳 */
} ExifGpsInfo;

/**
 * @brief EXIF元数据
 */
typedef struct {
    // 基本信息
    char make[64];                  /**< 制造商 */
    char model[64];                 /**< 型号 */
    char software[64];              /**< 软件 */
    char datetime[32];              /**< 拍摄时间 */
    char datetime_original[32];     /**< 原始时间 */
    char datetime_digitized[32];    /**< 数字化时间 */
    
    // 图像信息
    int width;                      /**< 图像宽度 */
    int height;                     /**< 图像高度 */
    ExifOrientation orientation;    /**< 方向 */
    int x_resolution;               /**< X分辨率 */
    int y_resolution;               /**< Y分辨率 */
    int resolution_unit;            /**< 分辨率单位 (2=英寸, 3=厘米) */
    
    // 相机设置
    float exposure_time;            /**< 曝光时间 (秒) */
    float f_number;                 /**< 光圈值 */
    int iso_speed;                  /**< ISO感光度 */
    float focal_length;             /**< 焦距 (mm) */
    float focal_length_35mm;        /**< 35mm等效焦距 */
    int flash;                      /**< 闪光灯 */
    int white_balance;              /**< 白平衡 */
    int exposure_program;           /**< 曝光程序 */
    int metering_mode;              /**< 测光模式 */
    
    // 镜头信息
    char lens_make[64];             /**< 镜头制造商 */
    char lens_model[64];            /**< 镜头型号 */
    
    // GPS信息
    ExifGpsInfo gps;                /**< GPS信息 */
    
    // 其他
    char description[256];          /**< 图像描述 */
    char copyright[128];            /**< 版权信息 */
    char artist[64];                /**< 作者 */
    char user_comment[256];         /**< 用户注释 */
} ExifMetadata;

/**
 * @brief JPEG图像信息
 */
typedef struct {
    int width;                      /**< 图像宽度 */
    int height;                     /**< 图像高度 */
    int channels;                   /**< 通道数 */
    JpegColorSpace color_space;     /**< 色彩空间 */
    JpegSubsampling subsampling;    /**< 子采样模式 */
    bool progressive;               /**< 是否为渐进式JPEG */
    int quality;                    /**< 估计的质量 (0-100) */
    size_t file_size;               /**< 文件大小 (字节) */
    bool has_exif;                  /**< 是否包含EXIF数据 */
    bool has_thumbnail;             /**< 是否包含缩略图 */
    char comment[JPEG_MAX_COMMENT_LENGTH]; /**< JPEG注释 */
} JpegInfo;

/**
 * @brief JPEG处理器 (内部使用)
 */
typedef struct JpegProcessor JpegProcessor;

// ============================================================================
// 初始化和清理
// ============================================================================

/**
 * @brief 初始化JPEG处理器
 * @return 成功返回true，失败返回false
 */
bool jpeg_init(void);

/**
 * @brief 清理JPEG处理器
 */
void jpeg_cleanup(void);

/**
 * @brief 获取JPEG处理器版本
 * @return 版本字符串
 */
const char* jpeg_get_version(void);

/**
 * @brief 检查是否支持JPEG
 * @return 支持返回true，不支持返回false
 */
bool jpeg_is_supported(void);

// ============================================================================
// 选项创建
// ============================================================================

/**
 * @brief 创建默认压缩选项
 * @return 默认压缩选项
 */
JpegCompressOptions jpeg_create_default_compress_options(void);

/**
 * @brief 创建默认解压缩选项
 * @return 默认解压缩选项
 */
JpegDecompressOptions jpeg_create_default_decompress_options(void);

/**
 * @brief 创建高质量压缩选项
 * @return 高质量压缩选项
 */
JpegCompressOptions jpeg_create_high_quality_options(void);

/**
 * @brief 创建低质量压缩选项 (用于缩略图等)
 * @return 低质量压缩选项
 */
JpegCompressOptions jpeg_create_low_quality_options(void);

/**
 * @brief 创建渐进式JPEG选项
 * @return 渐进式JPEG选项
 */
JpegCompressOptions jpeg_create_progressive_options(void);

// ============================================================================
// JPEG读取
// ============================================================================

/**
 * @brief 从文件读取JPEG图像
 * @param filename 文件名
 * @return 成功返回Image对象，失败返回NULL
 */
Image* jpeg_read(const char *filename);

/**
 * @brief 从文件读取JPEG图像 (带选项)
 * @param filename 文件名
 * @param options 解压缩选项
 * @return 成功返回Image对象，失败返回NULL
 */
Image* jpeg_read_with_options(const char *filename, const JpegDecompressOptions *options);

/**
 * @brief 从内存读取JPEG图像
 * @param data 内存数据
 * @param size 数据大小
 * @return 成功返回Image对象，失败返回NULL
 */
Image* jpeg_read_from_memory(const unsigned char *data, size_t size);

/**
 * @brief 从内存读取JPEG图像 (带选项)
 * @param data 内存数据
 * @param size 数据大小
 * @param options 解压缩选项
 * @return 成功返回Image对象，失败返回NULL
 */
Image* jpeg_read_from_memory_with_options(const unsigned char *data, size_t size,
                                          const JpegDecompressOptions *options);

/**
 * @brief 读取JPEG缩略图 (快速加载缩小版本)
 * @param filename 文件名
 * @param scale 缩放因子 (1, 2, 4, 8)
 * @return 成功返回Image对象，失败返回NULL
 */
Image* jpeg_read_thumbnail(const char *filename, int scale);

// ============================================================================
// JPEG写入
// ============================================================================

/**
 * @brief 写入JPEG图像到文件
 * @param img 图像对象
 * @param filename 文件名
 * @param quality 质量 (1-100)
 * @return 成功返回true，失败返回false
 */
bool jpeg_write(const Image *img, const char *filename, int quality);

/**
 * @brief 写入JPEG图像到文件 (带选项)
 * @param img 图像对象
 * @param filename 文件名
 * @param options 压缩选项
 * @return 成功返回true，失败返回false
 */
bool jpeg_write_with_options(const Image *img, const char *filename,
                             const JpegCompressOptions *options);

/**
 * @brief 写入JPEG图像到内存
 * @param img 图像对象
 * @param data 输出数据指针 (需要调用者释放)
 * @param size 输出数据大小
 * @param quality 质量 (1-100)
 * @return 成功返回true，失败返回false
 */
bool jpeg_write_to_memory(const Image *img, unsigned char **data, size_t *size, int quality);

/**
 * @brief 写入JPEG图像到内存 (带选项)
 * @param img 图像对象
 * @param data 输出数据指针 (需要调用者释放)
 * @param size 输出数据大小
 * @param options 压缩选项
 * @return 成功返回true，失败返回false
 */
bool jpeg_write_to_memory_with_options(const Image *img, unsigned char **data, size_t *size,
                                       const JpegCompressOptions *options);

// ============================================================================
// JPEG信息获取
// ============================================================================

/**
 * @brief 获取JPEG图像信息 (不解码图像数据)
 * @param filename 文件名
 * @param info 输出信息结构
 * @return 成功返回true，失败返回false
 */
bool jpeg_get_info(const char *filename, JpegInfo *info);

/**
 * @brief 从内存获取JPEG图像信息
 * @param data 内存数据
 * @param size 数据大小
 * @param info 输出信息结构
 * @return 成功返回true，失败返回false
 */
bool jpeg_get_info_from_memory(const unsigned char *data, size_t size, JpegInfo *info);

/**
 * @brief 打印JPEG图像信息
 * @param info JPEG信息结构
 */
void jpeg_print_info(const JpegInfo *info);

/**
 * @brief 验证JPEG文件
 * @param filename 文件名
 * @return 有效返回true，无效返回false
 */
bool jpeg_validate(const char *filename);

// ============================================================================
// EXIF元数据处理
// ============================================================================

/**
 * @brief 读取EXIF元数据
 * @param filename 文件名
 * @param exif 输出EXIF结构
 * @return 成功返回true，失败返回false
 */
bool jpeg_read_exif(const char *filename, ExifMetadata *exif);

/**
 * @brief 从内存读取EXIF元数据
 * @param data 内存数据
 * @param size 数据大小
 * @param exif 输出EXIF结构
 * @return 成功返回true，失败返回false
 */
bool jpeg_read_exif_from_memory(const unsigned char *data, size_t size, ExifMetadata *exif);

/**
 * @brief 写入EXIF元数据
 * @param filename 文件名
 * @param exif EXIF结构
 * @return 成功返回true，失败返回false
 */
bool jpeg_write_exif(const char *filename, const ExifMetadata *exif);

/**
 * @brief 删除EXIF元数据
 * @param filename 文件名
 * @return 成功返回true，失败返回false
 */
bool jpeg_remove_exif(const char *filename);

/**
 * @brief 打印EXIF元数据
 * @param exif EXIF结构
 */
void jpeg_print_exif(const ExifMetadata *exif);

/**
 * @brief 根据EXIF方向旋转图像
 * @param img 图像对象
 * @param orientation EXIF方向
 * @return 成功返回true，失败返回false
 */
bool jpeg_apply_exif_orientation(Image *img, ExifOrientation orientation);

// ============================================================================
// JPEG优化
// ============================================================================

/**
 * @brief 优化JPEG文件 (无损优化)
 * @param input_filename 输入文件名
 * @param output_filename 输出文件名
 * @return 成功返回true，失败返回false
 */
bool jpeg_optimize(const char *input_filename, const char *output_filename);

/**
 * @brief 优化JPEG文件 (有损优化)
 * @param input_filename 输入文件名
 * @param output_filename 输出文件名
 * @param quality 目标质量
 * @return 成功返回true，失败返回false
 */
bool jpeg_optimize_lossy(const char *input_filename, const char *output_filename, int quality);

/**
 * @brief 估计JPEG质量
 * @param filename 文件名
 * @return 估计的质量值 (0-100)，失败返回-1
 */
int jpeg_estimate_quality(const char *filename);

// ============================================================================
// JPEG转换
// ============================================================================

/**
 * @brief 转换为渐进式JPEG
 * @param input_filename 输入文件名
 * @param output_filename 输出文件名
 * @return 成功返回true，失败返回false
 */
bool jpeg_convert_to_progressive(const char *input_filename, const char *output_filename);

/**
 * @brief 转换为基线JPEG
 * @param input_filename 输入文件名
 * @param output_filename 输出文件名
 * @return 成功返回true，失败返回false
 */
bool jpeg_convert_to_baseline(const char *input_filename, const char *output_filename);

/**
 * @brief 转换色彩空间
 * @param img 图像对象
 * @param target_space 目标色彩空间
 * @return 成功返回true，失败返回false
 */
bool jpeg_convert_color_space(Image *img, JpegColorSpace target_space);

// ============================================================================
// 批处理
// ============================================================================

/**
 * @brief 批量压缩JPEG
 * @param input_files 输入文件列表
 * @param output_files 输出文件列表
 * @param count 文件数量
 * @param options 压缩选项
 * @return 成功返回true，失败返回false
 */
bool jpeg_batch_compress(const char **input_files, const char **output_files,
                        int count, const JpegCompressOptions *options);

/**
 * @brief 批量优化JPEG
 * @param input_files 输入文件列表
 * @param output_files 输出文件列表
 * @param count 文件数量
 * @return 成功返回true，失败返回false
 */
bool jpeg_batch_optimize(const char **input_files, const char **output_files, int count);

// ============================================================================
// 工具函数
// ============================================================================

/**
 * @brief 获取色彩空间名称
 * @param space 色彩空间
 * @return 色彩空间名称字符串
 */
const char* jpeg_get_color_space_name(JpegColorSpace space);

/**
 * @brief 获取子采样模式名称
 * @param subsampling 子采样模式
 * @return 子采样模式名称字符串
 */
const char* jpeg_get_subsampling_name(JpegSubsampling subsampling);

/**
 * @brief 获取EXIF方向名称
 * @param orientation EXIF方向
 * @return 方向名称字符串
 */
const char* jpeg_get_orientation_name(ExifOrientation orientation);

/**
 * @brief 计算压缩比
 * @param original_size 原始大小
 * @param compressed_size 压缩后大小
 * @return 压缩比
 */
float jpeg_calculate_compression_ratio(size_t original_size, size_t compressed_size);

#ifdef __cplusplus
}
#endif

#endif /* JPEG_HANDLER_H */
