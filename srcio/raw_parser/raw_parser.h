/**
 * @file raw_parser.h
 * @brief RAW图像格式解析器
 * @details 支持多种相机RAW格式的读取和解析
 * 
 * 支持的RAW格式：
 * - Canon: CR2, CR3
 * - Nikon: NEF
 * - Sony: ARW
 * - Fujifilm: RAF
 * - Olympus: ORF
 * - Panasonic: RW2
 * - Pentax: PEF, DNG
 * - Adobe: DNG
 * 
 * @author Your Name
 * @date 2024
 */

#ifndef RAW_PARSER_H
#define RAW_PARSER_H

#include "image.h"
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// RAW格式类型定义
// ============================================================================

/**
 * @brief RAW图像格式枚举
 */
typedef enum {
    RAW_FORMAT_UNKNOWN = 0,
    RAW_FORMAT_DNG,      // Adobe Digital Negative
    RAW_FORMAT_CR2,      // Canon Raw 2
    RAW_FORMAT_CR3,      // Canon Raw 3
    RAW_FORMAT_NEF,      // Nikon Electronic Format
    RAW_FORMAT_ARW,      // Sony Alpha Raw
    RAW_FORMAT_RAF,      // Fujifilm Raw
    RAW_FORMAT_ORF,      // Olympus Raw Format
    RAW_FORMAT_RW2,      // Panasonic Raw 2
    RAW_FORMAT_PEF,      // Pentax Electronic Format
    RAW_FORMAT_SRW,      // Samsung Raw
    RAW_FORMAT_3FR,      // Hasselblad 3F Raw
    RAW_FORMAT_FFF,      // Hasselblad Flexible File Format
    RAW_FORMAT_MEF,      // Mamiya Electronic Format
    RAW_FORMAT_MOS,      // Leaf Raw
    RAW_FORMAT_IIQ,      // Phase One Raw
    RAW_FORMAT_RWL,      // Leica Raw
    RAW_FORMAT_GPR       // GoPro Raw
} RawFormat;

/**
 * @brief RAW图像Bayer模式
 */
typedef enum {
    BAYER_PATTERN_RGGB = 0,  // Red-Green-Green-Blue
    BAYER_PATTERN_BGGR,      // Blue-Green-Green-Red
    BAYER_PATTERN_GRBG,      // Green-Red-Blue-Green
    BAYER_PATTERN_GBRG,      // Green-Blue-Red-Green
    BAYER_PATTERN_NONE       // 非Bayer模式（如X-Trans）
} BayerPattern;

/**
 * @brief 白平衡系数
 */
typedef struct {
    float r_multiplier;      // 红色通道倍增系数
    float g_multiplier;      // 绿色通道倍增系数
    float b_multiplier;      // 蓝色通道倍增系数
    float g2_multiplier;     // 第二个绿色通道倍增系数（Bayer模式）
} WhiteBalance;

/**
 * @brief 色彩矩阵（3x3或3x4）
 */
typedef struct {
    float matrix[3][4];      // 色彩转换矩阵
    int rows;                // 行数（通常为3）
    int cols;                // 列数（3或4）
} ColorMatrix;

/**
 * @brief RAW图像元数据
 */
typedef struct {
    // 基本信息
    RawFormat format;        // RAW格式类型
    char make[64];           // 相机制造商
    char model[64];          // 相机型号
    char software[64];       // 软件版本
    
    // 图像尺寸
    int width;               // 图像宽度
    int height;              // 图像高度
    int raw_width;           // RAW数据宽度（可能包含黑边）
    int raw_height;          // RAW数据高度
    int top_margin;          // 顶部边距
    int left_margin;         // 左侧边距
    
    // 像素信息
    int bits_per_sample;     // 每样本位数
    BayerPattern bayer_pattern; // Bayer模式
    
    // 曝光信息
    float iso_speed;         // ISO感光度
    float shutter_speed;     // 快门速度（秒）
    float aperture;          // 光圈值
    float focal_length;      // 焦距（mm）
    float exposure_bias;     // 曝光补偿
    
    // 白平衡
    WhiteBalance wb_coeffs;  // 白平衡系数
    WhiteBalance wb_as_shot; // 拍摄时白平衡
    WhiteBalance wb_auto;    // 自动白平衡
    WhiteBalance wb_daylight;// 日光白平衡
    
    // 色彩信息
    ColorMatrix color_matrix1; // 色彩矩阵1（标准光源）
    ColorMatrix color_matrix2; // 色彩矩阵2（备用光源）
    float analog_balance[3];   // 模拟增益平衡
    
    // 黑电平和白电平
    uint16_t black_level[4]; // 黑电平（每个Bayer通道）
    uint16_t white_level;    // 白电平
    
    // 时间信息
    char datetime[32];       // 拍摄日期时间
    
    // 缩略图信息
    bool has_thumbnail;      // 是否有缩略图
    int thumb_width;         // 缩略图宽度
    int thumb_height;        // 缩略图高度
    int thumb_offset;        // 缩略图数据偏移
    int thumb_length;        // 缩略图数据长度
    
    // GPS信息
    bool has_gps;            // 是否有GPS信息
    double gps_latitude;     // 纬度
    double gps_longitude;    // 经度
    double gps_altitude;     // 海拔
    
    // 其他
    int orientation;         // 图像方向（EXIF标准）
    char lens_model[64];     // 镜头型号
    float crop_factor;       // 裁剪系数
} RawMetadata;

/**
 * @brief RAW图像数据结构
 */
typedef struct {
    RawMetadata metadata;    // 元数据
    uint16_t *raw_data;      // RAW像素数据（16位）
    size_t data_size;        // 数据大小（字节）
    
    // 缩略图数据
    unsigned char *thumbnail_data; // 缩略图数据
    size_t thumbnail_size;   // 缩略图大小
    
    // 内部使用
    void *internal_data;     // 内部数据指针（LibRaw等）
} RawImage;

// ============================================================================
// RAW解析选项
// ============================================================================

/**
 * @brief RAW解析选项
 */
typedef struct {
    // 输出选项
    bool output_color;       // 输出彩色图像（否则输出灰度）
    bool use_camera_wb;      // 使用相机白平衡
    bool use_auto_wb;        // 使用自动白平衡
    bool no_auto_bright;     // 禁用自动亮度调整
    bool no_auto_scale;      // 禁用自动缩放
    
    // 去马赛克选项
    int demosaic_algorithm;  // 去马赛克算法
                            // 0: 线性插值
                            // 1: VNG (Variable Number of Gradients)
                            // 2: PPG (Patterned Pixel Grouping)
                            // 3: AHD (Adaptive Homogeneity-Directed)
                            // 4: DCB (Demosaicing using Color-Based)
                            // 11: DHT (Demosaicing using Homogeneity-based Threshold)
    
    // 质量选项
    int quality;             // 输出质量（0-4）
                            // 0: 线性16位
                            // 1: VNG
                            // 2: PPG
                            // 3: AHD
                            // 4: 最高质量
    
    // 色彩空间
    int output_color_space;  // 输出色彩空间
                            // 0: raw
                            // 1: sRGB
                            // 2: Adobe RGB
                            // 3: Wide Gamut RGB
                            // 4: ProPhoto RGB
                            // 5: XYZ
    
    // 伽马校正
    float gamma_power;       // 伽马幂次（默认2.222）
    float gamma_slope;       // 伽马斜率（默认4.5）
    
    // 白平衡
    float user_mul[4];       // 用户自定义白平衡倍增系数
    
    // 裁剪
    bool use_fuji_rotate;    // Fuji相机旋转
    
    // 噪点抑制
    float threshold;         // 噪点阈值
    
    // 高光恢复
    int highlight_mode;      // 高光恢复模式
                            // 0: 裁剪
                            // 1: 不裁剪
                            // 2: 混合
                            // 3+: 重建
    
    // 输出位深
    int output_bps;          // 输出位深（8或16）
    
    // 半尺寸模式
    bool half_size;          // 输出半尺寸图像（快速预览）
    
    // 四色RGB
    bool four_color_rgb;     // 使用四色RGB插值
    
    // 中值滤波
    int median_filter_passes; // 中值滤波次数
} RawParseOptions;

// ============================================================================
// 函数声明
// ============================================================================

/**
 * @brief 初始化RAW解析器
 * @return 成功返回true，失败返回false
 */
bool raw_parser_init(void);

/**
 * @brief 清理RAW解析器
 */
void raw_parser_cleanup(void);

/**
 * @brief 检测RAW格式
 * @param filename 文件名
 * @return RAW格式类型
 */
RawFormat raw_detect_format(const char *filename);

/**
 * @brief 根据文件扩展名获取RAW格式
 * @param filename 文件名
 * @return RAW格式类型
 */
RawFormat raw_get_format_from_extension(const char *filename);

/**
 * @brief 获取RAW格式名称
 * @param format RAW格式
 * @return 格式名称字符串
 */
const char* raw_get_format_name(RawFormat format);

/**
 * @brief 创建默认解析选项
 * @return 默认选项结构
 */
RawParseOptions raw_create_default_options(void);

/**
 * @brief 打开RAW文件
 * @param filename 文件名
 * @return RAW图像对象，失败返回NULL
 */
RawImage* raw_open(const char *filename);

/**
 * @brief 打开RAW文件并指定选项
 * @param filename 文件名
 * @param options 解析选项
 * @return RAW图像对象，失败返回NULL
 */
RawImage* raw_open_with_options(const char *filename, const RawParseOptions *options);

/**
 * @brief 解析RAW图像为RGB图像
 * @param raw RAW图像对象
 * @param options 解析选项（NULL使用默认）
 * @return RGB图像对象，失败返回NULL
 */
Image* raw_process(const RawImage *raw, const RawParseOptions *options);

/**
 * @brief 提取RAW缩略图
 * @param raw RAW图像对象
 * @return 缩略图图像对象，失败返回NULL
 */
Image* raw_extract_thumbnail(const RawImage *raw);

/**
 * @brief 获取RAW元数据
 * @param filename 文件名
 * @param metadata 元数据结构指针
 * @return 成功返回true，失败返回false
 */
bool raw_get_metadata(const char *filename, RawMetadata *metadata);

/**
 * @brief 释放RAW图像对象
 * @param raw RAW图像对象
 */
void raw_close(RawImage *raw);

/**
 * @brief 打印RAW元数据
 * @param metadata 元数据结构
 */
void raw_print_metadata(const RawMetadata *metadata);

/**
 * @brief 获取Bayer模式名称
 * @param pattern Bayer模式
 * @return 模式名称字符串
 */
const char* raw_get_bayer_pattern_name(BayerPattern pattern);

/**
 * @brief 计算自动白平衡
 * @param raw RAW图像对象
 * @param wb 输出白平衡系数
 * @return 成功返回true，失败返回false
 */
bool raw_calculate_auto_wb(const RawImage *raw, WhiteBalance *wb);

/**
 * @brief 应用白平衡
 * @param raw RAW图像对象
 * @param wb 白平衡系数
 * @return 成功返回true，失败返回false
 */
bool raw_apply_white_balance(RawImage *raw, const WhiteBalance *wb);

/**
 * @brief 去马赛克（Bayer转RGB）
 * @param raw RAW图像对象
 * @param algorithm 去马赛克算法
 * @return RGB图像对象，失败返回NULL
 */
Image* raw_demosaic(const RawImage *raw, int algorithm);

/**
 * @brief 应用色彩矩阵
 * @param img 图像对象
 * @param matrix 色彩矩阵
 * @return 成功返回true，失败返回false
 */
bool raw_apply_color_matrix(Image *img, const ColorMatrix *matrix);

/**
 * @brief 应用伽马校正
 * @param img 图像对象
 * @param power 伽马幂次
 * @param slope 伽马斜率
 * @return 成功返回true，失败返回false
 */
bool raw_apply_gamma(Image *img, float power, float slope);

/**
 * @brief 转换色彩空间
 * @param img 图像对象
 * @param color_space 目标色彩空间
 * @return 成功返回true，失败返回false
 */
bool raw_convert_color_space(Image *img, int color_space);

/**
 * @brief 高光恢复
 * @param img 图像对象
 * @param mode 恢复模式
 * @return 成功返回true，失败返回false
 */
bool raw_recover_highlights(Image *img, int mode);

/**
 * @brief 快速加载RAW预览图
 * @param filename 文件名
 * @return 预览图像对象，失败返回NULL
 */
Image* raw_load_preview(const char *filename);

/**
 * @brief 一步加载并处理RAW图像
 * @param filename 文件名
 * @return 处理后的RGB图像，失败返回NULL
 */
Image* raw_load_and_process(const char *filename);

#ifdef __cplusplus
}
#endif

#endif // RAW_PARSER_H
