/**
 * @file image_loader.h
 * @brief Image loading and saving utilities for super-resolution
 * @author hany
 * @date 2025
 */

#ifndef IMAGE_LOADER_H
#define IMAGE_LOADER_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// 类型定义
// ============================================================================

/**
 * @brief 图像格式枚举
 */
typedef enum {
    IMAGE_FORMAT_UNKNOWN = 0,
    IMAGE_FORMAT_PNG,
    IMAGE_FORMAT_JPEG,
    IMAGE_FORMAT_BMP,
    IMAGE_FORMAT_TIFF,
    IMAGE_FORMAT_PPM,
    IMAGE_FORMAT_PGM,
    IMAGE_FORMAT_RAW
} ImageFormat;

/**
 * @brief 颜色空间枚举
 */
typedef enum {
    COLOR_SPACE_RGB = 0,
    COLOR_SPACE_RGBA,
    COLOR_SPACE_GRAYSCALE,
    COLOR_SPACE_YUV,
    COLOR_SPACE_HSV,
    COLOR_SPACE_LAB
} ColorSpace;

/**
 * @brief 像素数据类型
 */
typedef enum {
    PIXEL_TYPE_UINT8 = 0,
    PIXEL_TYPE_UINT16,
    PIXEL_TYPE_FLOAT32,
    PIXEL_TYPE_FLOAT64
} PixelType;

/**
 * @brief 图像结构体
 */
typedef struct {
    int width;              // 图像宽度
    int height;             // 图像高度
    int channels;           // 通道数 (1=灰度, 3=RGB, 4=RGBA)
    float *data;            // 图像数据 (归一化到[0,1])
    ColorSpace color_space; // 颜色空间
    PixelType pixel_type;   // 像素类型
    void *metadata;         // 元数据 (可选)
} Image;

/**
 * @brief 图像元数据
 */
typedef struct {
    char *author;
    char *description;
    char *software;
    char *creation_time;
    int dpi_x;
    int dpi_y;
    float gamma;
    bool has_alpha;
    int bit_depth;
} ImageMetadata;

/**
 * @brief 图像加载选项
 */
typedef struct {
    bool normalize;         // 是否归一化到[0,1]
    bool convert_to_rgb;    // 是否转换为RGB
    bool preserve_alpha;    // 是否保留alpha通道
    int target_channels;    // 目标通道数 (0=保持原样)
    float gamma_correction; // Gamma校正值 (0=不校正)
    bool flip_vertical;     // 是否垂直翻转
    bool flip_horizontal;   // 是否水平翻转
} ImageLoadOptions;

/**
 * @brief 图像保存选项
 */
typedef struct {
    int quality;            // JPEG质量 (1-100)
    int compression_level;  // PNG压缩级别 (0-9)
    bool denormalize;       // 是否反归一化
    float gamma_correction; // Gamma校正
    bool save_metadata;     // 是否保存元数据
    int dpi;                // DPI设置
} ImageSaveOptions;

/**
 * @brief 图像统计信息
 */
typedef struct {
    float min_value;
    float max_value;
    float mean_value;
    float std_dev;
    float *histogram;       // 直方图 (256 bins)
    int histogram_size;
} ImageStats;

// ============================================================================
// 错误代码
// ============================================================================

#define IMAGE_SUCCESS                0
#define IMAGE_ERROR_NULL_POINTER    -1
#define IMAGE_ERROR_INVALID_PARAM   -2
#define IMAGE_ERROR_MEMORY_ALLOC    -3
#define IMAGE_ERROR_FILE_NOT_FOUND  -4
#define IMAGE_ERROR_FILE_READ       -5
#define IMAGE_ERROR_FILE_WRITE      -6
#define IMAGE_ERROR_UNSUPPORTED     -7
#define IMAGE_ERROR_INVALID_FORMAT  -8
#define IMAGE_ERROR_CORRUPTED       -9
#define IMAGE_ERROR_DIMENSION       -10

// ============================================================================
// 图像创建和销毁
// ============================================================================

/**
 * @brief 创建新图像
 * @param width 图像宽度
 * @param height 图像高度
 * @param channels 通道数
 * @return 图像指针，失败返回NULL
 */
Image* image_create(int width, int height, int channels);

/**
 * @brief 创建带初始值的图像
 * @param width 图像宽度
 * @param height 图像高度
 * @param channels 通道数
 * @param init_value 初始值
 * @return 图像指针，失败返回NULL
 */
Image* image_create_with_value(int width, int height, int channels, float init_value);

/**
 * @brief 从数据创建图像
 * @param width 图像宽度
 * @param height 图像高度
 * @param channels 通道数
 * @param data 数据指针 (会被复制)
 * @return 图像指针，失败返回NULL
 */
Image* image_create_from_data(int width, int height, int channels, const float *data);

/**
 * @brief 克隆图像
 * @param src 源图像
 * @return 新图像指针，失败返回NULL
 */
Image* image_clone(const Image *src);

/**
 * @brief 销毁图像
 * @param img 图像指针
 */
void image_destroy(Image *img);

// ============================================================================
// 图像加载和保存
// ============================================================================

/**
 * @brief 加载图像文件
 * @param filename 文件名
 * @return 图像指针，失败返回NULL
 */
Image* image_load(const char *filename);

/**
 * @brief 使用选项加载图像
 * @param filename 文件名
 * @param options 加载选项
 * @return 图像指针，失败返回NULL
 */
Image* image_load_with_options(const char *filename, const ImageLoadOptions *options);

/**
 * @brief 保存图像文件
 * @param img 图像指针
 * @param filename 文件名
 * @return 错误代码
 */
int image_save(const Image *img, const char *filename);

/**
 * @brief 使用选项保存图像
 * @param img 图像指针
 * @param filename 文件名
 * @param options 保存选项
 * @return 错误代码
 */
int image_save_with_options(const Image *img, const char *filename, 
                           const ImageSaveOptions *options);

/**
 * @brief 检测图像格式
 * @param filename 文件名
 * @return 图像格式
 */
ImageFormat image_detect_format(const char *filename);

/**
 * @brief 从扩展名获取格式
 * @param filename 文件名
 * @return 图像格式
 */
ImageFormat image_format_from_extension(const char *filename);

// ============================================================================
// PNG 支持
// ============================================================================

/**
 * @brief 加载PNG图像
 * @param filename 文件名
 * @return 图像指针，失败返回NULL
 */
Image* image_load_png(const char *filename);

/**
 * @brief 保存PNG图像
 * @param img 图像指针
 * @param filename 文件名
 * @param compression_level 压缩级别 (0-9)
 * @return 错误代码
 */
int image_save_png(const Image *img, const char *filename, int compression_level);

// ============================================================================
// JPEG 支持
// ============================================================================

/**
 * @brief 加载JPEG图像
 * @param filename 文件名
 * @return 图像指针，失败返回NULL
 */
Image* image_load_jpeg(const char *filename);

/**
 * @brief 保存JPEG图像
 * @param img 图像指针
 * @param filename 文件名
 * @param quality 质量 (1-100)
 * @return 错误代码
 */
int image_save_jpeg(const Image *img, const char *filename, int quality);

// ============================================================================
// BMP 支持
// ============================================================================

/**
 * @brief 加载BMP图像
 * @param filename 文件名
 * @return 图像指针，失败返回NULL
 */
Image* image_load_bmp(const char *filename);

/**
 * @brief 保存BMP图像
 * @param img 图像指针
 * @param filename 文件名
 * @return 错误代码
 */
int image_save_bmp(const Image *img, const char *filename);

// ============================================================================
// PPM/PGM 支持
// ============================================================================

/**
 * @brief 加载PPM/PGM图像
 * @param filename 文件名
 * @return 图像指针，失败返回NULL
 */
Image* image_load_ppm(const char *filename);

/**
 * @brief 保存PPM/PGM图像
 * @param img 图像指针
 * @param filename 文件名
 * @return 错误代码
 */
int image_save_ppm(const Image *img, const char *filename);

// ============================================================================
// RAW 数据支持
// ============================================================================

/**
 * @brief 加载RAW数据
 * @param filename 文件名
 * @param width 图像宽度
 * @param height 图像高度
 * @param channels 通道数
 * @param pixel_type 像素类型
 * @return 图像指针，失败返回NULL
 */
Image* image_load_raw(const char *filename, int width, int height, 
                     int channels, PixelType pixel_type);

/**
 * @brief 保存RAW数据
 * @param img 图像指针
 * @param filename 文件名
 * @param pixel_type 像素类型
 * @return 错误代码
 */
int image_save_raw(const Image *img, const char *filename, PixelType pixel_type);

// ============================================================================
// 颜色空间转换
// ============================================================================

/**
 * @brief RGB转灰度
 * @param img 源图像
 * @return 灰度图像，失败返回NULL
 */
Image* image_to_grayscale(const Image *img);

/**
 * @brief 灰度转RGB
 * @param img 源图像
 * @return RGB图像，失败返回NULL
 */
Image* image_to_rgb(const Image *img);

/**
 * @brief RGB转YUV
 * @param img 源图像
 * @return YUV图像，失败返回NULL
 */
Image* image_rgb_to_yuv(const Image *img);

/**
 * @brief YUV转RGB
 * @param img 源图像
 * @return RGB图像，失败返回NULL
 */
Image* image_yuv_to_rgb(const Image *img);

/**
 * @brief RGB转HSV
 * @param img 源图像
 * @return HSV图像，失败返回NULL
 */
Image* image_rgb_to_hsv(const Image *img);

/**
 * @brief HSV转RGB
 * @param img 源图像
 * @return RGB图像，失败返回NULL
 */
Image* image_hsv_to_rgb(const Image *img);

/**
 * @brief RGB转LAB
 * @param img 源图像
 * @return LAB图像，失败返回NULL
 */
Image* image_rgb_to_lab(const Image *img);

/**
 * @brief LAB转RGB
 * @param img 源图像
 * @return RGB图像，失败返回NULL
 */
Image* image_lab_to_rgb(const Image *img);

/**
 * @brief 通用颜色空间转换
 * @param img 源图像
 * @param target_space 目标颜色空间
 * @return 转换后的图像，失败返回NULL
 */
Image* image_convert_color_space(const Image *img, ColorSpace target_space);

// ============================================================================
// 通道操作
// ============================================================================

/**
 * @brief 分离通道
 * @param img 源图像
 * @param channels 输出通道数组 (需要预分配)
 * @return 错误代码
 */
int image_split_channels(const Image *img, Image **channels);

/**
 * @brief 合并通道
 * @param channels 通道数组
 * @param num_channels 通道数
 * @return 合并后的图像，失败返回NULL
 */
Image* image_merge_channels(Image **channels, int num_channels);

/**
 * @brief 提取单个通道
 * @param img 源图像
 * @param channel_index 通道索引
 * @return 单通道图像，失败返回NULL
 */
Image* image_extract_channel(const Image *img, int channel_index);

/**
 * @brief 添加Alpha通道
 * @param img 源图像
 * @param alpha_value Alpha值 (0-1)
 * @return 带Alpha的图像，失败返回NULL
 */
Image* image_add_alpha(const Image *img, float alpha_value);

/**
 * @brief 移除Alpha通道
 * @param img 源图像
 * @return 不带Alpha的图像，失败返回NULL
 */
Image* image_remove_alpha(const Image *img);

// ============================================================================
// 像素访问
// ============================================================================

/**
 * @brief 获取像素值
 * @param img 图像指针
 * @param x X坐标
 * @param y Y坐标
 * @param channel 通道索引
 * @return 像素值
 */
float image_get_pixel(const Image *img, int x, int y, int channel);

/**
 * @brief 设置像素值
 * @param img 图像指针
 * @param x X坐标
 * @param y Y坐标
 * @param channel 通道索引
 * @param value 像素值
 */
void image_set_pixel(Image *img, int x, int y, int channel, float value);

/**
 * @brief 获取像素值（带边界检查）
 * @param img 图像指针
 * @param x X坐标
 * @param y Y坐标
 * @param channel 通道索引
 * @param default_value 越界时的默认值
 * @return 像素值
 */
float image_get_pixel_safe(const Image *img, int x, int y, int channel, float default_value);

/**
 * @brief 双线性插值获取像素值
 * @param img 图像指针
 * @param x X坐标 (浮点)
 * @param y Y坐标 (浮点)
 * @param channel 通道索引
 * @return 插值后的像素值
 */
float image_get_pixel_bilinear(const Image *img, float x, float y, int channel);

// ============================================================================
// 图像变换
// ============================================================================

/**
 * @brief 裁剪图像
 * @param img 源图像
 * @param x 起始X坐标
 * @param y 起始Y坐标
 * @param width 裁剪宽度
 * @param height 裁剪高度
 * @return 裁剪后的图像，失败返回NULL
 */
Image* image_crop(const Image *img, int x, int y, int width, int height);

/**
 * @brief 填充图像
 * @param img 源图像
 * @param top 上边填充
 * @param bottom 下边填充
 * @param left 左边填充
 * @param right 右边填充
 * @param fill_value 填充值
 * @return 填充后的图像，失败返回NULL
 */
Image* image_pad(const Image *img, int top, int bottom, int left, int right, float fill_value);

/**
 * @brief 水平翻转
 * @param img 源图像
 * @return 翻转后的图像，失败返回NULL
 */
Image* image_flip_horizontal(const Image *img);

/**
 * @brief 垂直翻转
 * @param img 源图像
 * @return 翻转后的图像，失败返回NULL
 */
Image* image_flip_vertical(const Image *img);

/**
 * @brief 旋转图像
 * @param img 源图像
 * @param angle 旋转角度（度）
 * @return 旋转后的图像，失败返回NULL
 */
Image* image_rotate(const Image *img, float angle);

/**
 * @brief 旋转90度
 * @param img 源图像
 * @param times 旋转次数 (1=90°, 2=180°, 3=270°)
 * @return 旋转后的图像，失败返回NULL
 */
Image* image_rotate_90(const Image *img, int times);

/**
 * @brief 转置图像
 * @param img 源图像
 * @return 转置后的图像，失败返回NULL
 */
Image* image_transpose(const Image *img);

// ============================================================================
// 图像统计
// ============================================================================

/**
 * @brief 计算图像统计信息
 * @param img 图像指针
 * @param stats 统计信息输出
 * @return 错误代码
 */
int image_compute_stats(const Image *img, ImageStats *stats);

/**
 * @brief 计算直方图
 * @param img 图像指针
 * @param channel 通道索引 (-1表示所有通道)
 * @param num_bins 直方图bins数量
 * @return 直方图数组，需要调用者释放
 */
float* image_compute_histogram(const Image *img, int channel, int num_bins);

/**
 * @brief 归一化图像
 * @param img 源图像
 * @param min_val 最小值
 * @param max_val 最大值
 * @return 归一化后的图像，失败返回NULL
 */
Image* image_normalize(const Image *img, float min_val, float max_val);

/**
 * @brief 标准化图像（零均值，单位方差）
 * @param img 源图像
 * @return 标准化后的图像，失败返回NULL
 */
Image* image_standardize(const Image *img);

/**
 * @brief 直方图均衡化
 * @param img 源图像
 * @return 均衡化后的图像，失败返回NULL
 */
Image* image_histogram_equalization(const Image *img);

// ============================================================================
// 元数据操作
// ============================================================================

/**
 * @brief 创建元数据
 * @return 元数据指针，失败返回NULL
 */
ImageMetadata* image_metadata_create(void);

/**
 * @brief 销毁元数据
 * @param metadata 元数据指针
 */
void image_metadata_destroy(ImageMetadata *metadata);

/**
 * @brief 复制元数据
 * @param src 源元数据
 * @return 新元数据指针，失败返回NULL
 */
ImageMetadata* image_metadata_clone(const ImageMetadata *src);

/**
 * @brief 从图像获取元数据
 * @param img 图像指针
 * @return 元数据指针，失败返回NULL
 */
ImageMetadata* image_get_metadata(const Image *img);

/**
 * @brief 设置图像元数据
 * @param img 图像指针
 * @param metadata 元数据指针
 * @return 错误代码
 */
int image_set_metadata(Image *img, const ImageMetadata *metadata);

// ============================================================================
// 工具函数
// ============================================================================

/**
 * @brief 检查图像是否有效
 * @param img 图像指针
 * @return true表示有效
 */
bool image_is_valid(const Image *img);

/**
 * @brief 检查两个图像尺寸是否相同
 * @param img1 图像1
 * @param img2 图像2
 * @return true表示相同
 */
bool image_same_size(const Image *img1, const Image *img2);

/**
 * @brief 复制图像数据
 * @param dst 目标图像
 * @param src 源图像
 * @return 错误代码
 */
int image_copy_data(Image *dst, const Image *src);

/**
 * @brief 填充图像
 * @param img 图像指针
 * @param value 填充值
 */
void image_fill(Image *img, float value);

/**
 * @brief 清空图像（填充0）
 * @param img 图像指针
 */
void image_clear(Image *img);

/**
 * @brief 获取错误信息
 * @param error_code 错误代码
 * @return 错误信息字符串
 */
const char* image_error_string(int error_code);

/**
 * @brief 打印图像信息
 * @param img 图像指针
 */
void image_print_info(const Image *img);

/**
 * @brief 获取默认加载选项
 * @return 默认加载选项
 */
ImageLoadOptions image_get_default_load_options(void);

/**
 * @brief 获取默认保存选项
 * @return 默认保存选项
 */
ImageSaveOptions image_get_default_save_options(void);

#ifdef __cplusplus
}
#endif

#endif // IMAGE_LOADER_H
