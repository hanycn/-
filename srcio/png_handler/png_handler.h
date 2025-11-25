/**
 * @file png_handler.h
 * @brief PNG图像处理库头文件
 * @author hany
 * @date 2025
 * @version 1.0
 * 
 * 提供完整的PNG图像读写、处理和优化功能
 * 依赖: libpng, zlib
 */

#ifndef PNG_HANDLER_H
#define PNG_HANDLER_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>
#include <png.h>
#include <zlib.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// 版本信息
// ============================================================================

#define PNG_HANDLER_VERSION_MAJOR 1
#define PNG_HANDLER_VERSION_MINOR 0
#define PNG_HANDLER_VERSION_PATCH 0
#define PNG_HANDLER_VERSION "1.0.0"

// ============================================================================
// 常量定义
// ============================================================================

// PNG颜色类型
#define PNG_COLOR_TYPE_GRAY_ALPHA   (PNG_COLOR_TYPE_GRAY | PNG_COLOR_MASK_ALPHA)
#define PNG_COLOR_TYPE_RGB_ALPHA    (PNG_COLOR_TYPE_RGB | PNG_COLOR_MASK_ALPHA)

// 压缩级别
#define PNG_COMPRESSION_NONE        0
#define PNG_COMPRESSION_FAST        1
#define PNG_COMPRESSION_DEFAULT     6
#define PNG_COMPRESSION_BEST        9

// 滤波器类型
#define PNG_FILTER_TYPE_NONE        0
#define PNG_FILTER_TYPE_SUB         1
#define PNG_FILTER_TYPE_UP          2
#define PNG_FILTER_TYPE_AVERAGE     3
#define PNG_FILTER_TYPE_PAETH       4
#define PNG_FILTER_TYPE_ALL         5

// 交错类型
#define PNG_INTERLACE_NONE          0
#define PNG_INTERLACE_ADAM7         1

// 最大尺寸限制
#define PNG_MAX_WIDTH               65535
#define PNG_MAX_HEIGHT              65535
#define PNG_MAX_FILE_SIZE           (1024 * 1024 * 100)  // 100MB

// ============================================================================
// 数据结构定义
// ============================================================================

/**
 * @brief 图像结构
 */
typedef struct {
    int width;              // 图像宽度
    int height;             // 图像高度
    int channels;           // 通道数 (1=灰度, 2=灰度+Alpha, 3=RGB, 4=RGBA)
    int bit_depth;          // 位深度 (8 or 16)
    unsigned char *data;    // 图像数据
    size_t data_size;       // 数据大小（字节）
} Image;

/**
 * @brief PNG信息结构
 */
typedef struct {
    int width;              // 图像宽度
    int height;             // 图像高度
    int bit_depth;          // 位深度
    int color_type;         // 颜色类型
    int channels;           // 通道数
    int interlace_type;     // 交错类型
    int compression_type;   // 压缩类型
    int filter_method;      // 滤波方法
    long file_size;         // 文件大小
    bool has_transparency;  // 是否有透明度
    bool has_gamma;         // 是否有gamma信息
    double gamma;           // gamma值
    char color_type_name[32];  // 颜色类型名称
} PNGInfo;

/**
 * @brief PNG文本块信息
 */
typedef struct {
    char key[80];           // 关键字
    char text[2048];        // 文本内容
    int compression;        // 压缩类型
} PNGTextChunk;

/**
 * @brief PNG元数据
 */
typedef struct {
    char software[256];     // 软件信息
    char author[256];       // 作者
    char description[1024]; // 描述
    char copyright[256];    // 版权
    char creation_time[32]; // 创建时间
    int num_text_chunks;    // 文本块数量
    PNGTextChunk *text_chunks;  // 文本块数组
} PNGMetadata;

/**
 * @brief PNG写入选项
 */
typedef struct {
    int compression_level;  // 压缩级别 (0-9)
    int filter_type;        // 滤波器类型
    int interlace_type;     // 交错类型
    bool optimize;          // 是否优化
    bool strip_metadata;    // 是否移除元数据
    double gamma;           // gamma值 (0表示不设置)
} PNGWriteOptions;

/**
 * @brief PNG优化选项
 */
typedef struct {
    bool reduce_colors;     // 是否减少颜色数
    bool reduce_bit_depth;  // 是否减少位深度
    bool remove_alpha;      // 是否移除alpha通道（如果完全不透明）
    bool strip_metadata;    // 是否移除元数据
    int compression_level;  // 压缩级别
    int filter_type;        // 滤波器类型
} PNGOptimizeOptions;

/**
 * @brief 调色板结构
 */
typedef struct {
    int num_colors;         // 颜色数量
    png_color *colors;      // 颜色数组
    png_byte *trans;        // 透明度数组
    int num_trans;          // 透明度数量
} PNGPalette;

/**
 * @brief PNG动画帧（APNG）
 */
typedef struct {
    Image *image;           // 帧图像
    int x_offset;           // X偏移
    int y_offset;           // Y偏移
    int delay_num;          // 延迟分子
    int delay_den;          // 延迟分母
    int dispose_op;         // 处置操作
    int blend_op;           // 混合操作
} APNGFrame;

/**
 * @brief APNG动画信息
 */
typedef struct {
    int num_frames;         // 帧数
    int num_plays;          // 播放次数 (0=无限循环)
    APNGFrame *frames;      // 帧数组
    int width;              // 画布宽度
    int height;             // 画布高度
} APNGAnimation;

// ============================================================================
// 基本图像操作
// ============================================================================

/**
 * @brief 创建图像
 * @param width 宽度
 * @param height 高度
 * @param channels 通道数
 * @param bit_depth 位深度
 * @return 图像指针，失败返回NULL
 */
Image* image_create(int width, int height, int channels, int bit_depth);

/**
 * @brief 销毁图像
 * @param img 图像指针
 */
void image_destroy(Image *img);

/**
 * @brief 克隆图像
 * @param img 源图像
 * @return 新图像指针，失败返回NULL
 */
Image* image_clone(const Image *img);

/**
 * @brief 调整图像大小
 * @param img 源图像
 * @param new_width 新宽度
 * @param new_height 新高度
 * @return 调整后的图像，失败返回NULL
 */
Image* image_resize(const Image *img, int new_width, int new_height);

/**
 * @brief 获取像素值
 * @param img 图像
 * @param x X坐标
 * @param y Y坐标
 * @param channel 通道索引
 * @return 像素值
 */
int image_get_pixel(const Image *img, int x, int y, int channel);

/**
 * @brief 设置像素值
 * @param img 图像
 * @param x X坐标
 * @param y Y坐标
 * @param channel 通道索引
 * @param value 像素值
 */
void image_set_pixel(Image *img, int x, int y, int channel, int value);

// ============================================================================
// PNG基本读写操作
// ============================================================================

/**
 * @brief 读取PNG文件
 * @param filename 文件名
 * @return 图像指针，失败返回NULL
 */
Image* png_read(const char *filename);

/**
 * @brief 写入PNG文件
 * @param img 图像
 * @param filename 文件名
 * @param compression_level 压缩级别 (0-9)
 * @return 成功返回true，失败返回false
 */
bool png_write(const Image *img, const char *filename, int compression_level);

/**
 * @brief 使用选项写入PNG文件
 * @param img 图像
 * @param filename 文件名
 * @param options 写入选项
 * @return 成功返回true，失败返回false
 */
bool png_write_with_options(const Image *img, const char *filename, 
                            const PNGWriteOptions *options);

/**
 * @brief 从内存读取PNG
 * @param buffer 内存缓冲区
 * @param size 缓冲区大小
 * @return 图像指针，失败返回NULL
 */
Image* png_read_from_memory(const unsigned char *buffer, size_t size);

/**
 * @brief 写入PNG到内存
 * @param img 图像
 * @param buffer 输出缓冲区指针
 * @param size 输出大小指针
 * @param compression_level 压缩级别
 * @return 成功返回true，失败返回false
 */
bool png_write_to_memory(const Image *img, unsigned char **buffer, 
                        size_t *size, int compression_level);

// ============================================================================
// PNG信息获取
// ============================================================================

/**
 * @brief 获取PNG文件信息
 * @param filename 文件名
 * @param info 信息结构指针
 * @return 成功返回true，失败返回false
 */
bool png_get_info(const char *filename, PNGInfo *info);

/**
 * @brief 打印PNG信息
 * @param info PNG信息
 */
void png_print_info(const PNGInfo *info);

/**
 * @brief 验证PNG文件
 * @param filename 文件名
 * @param error_msg 错误消息缓冲区
 * @param error_msg_size 错误消息缓冲区大小
 * @return 有效返回true，无效返回false
 */
bool png_validate(const char *filename, char *error_msg, int error_msg_size);

/**
 * @brief 检查是否为PNG文件
 * @param filename 文件名
 * @return 是PNG返回true，否则返回false
 */
bool png_is_png_file(const char *filename);

// ============================================================================
// PNG元数据操作
// ============================================================================

/**
 * @brief 读取PNG元数据
 * @param filename 文件名
 * @param metadata 元数据结构指针
 * @return 成功返回true，失败返回false
 */
bool png_read_metadata(const char *filename, PNGMetadata *metadata);

/**
 * @brief 写入PNG元数据
 * @param filename 文件名
 * @param metadata 元数据
 * @return 成功返回true，失败返回false
 */
bool png_write_metadata(const char *filename, const PNGMetadata *metadata);

/**
 * @brief 打印PNG元数据
 * @param metadata 元数据
 */
void png_print_metadata(const PNGMetadata *metadata);

/**
 * @brief 释放元数据
 * @param metadata 元数据指针
 */
void png_free_metadata(PNGMetadata *metadata);

/**
 * @brief 复制元数据
 * @param src_filename 源文件
 * @param dst_filename 目标文件
 * @return 成功返回true，失败返回false
 */
bool png_copy_metadata(const char *src_filename, const char *dst_filename);

/**
 * @brief 移除元数据
 * @param input_filename 输入文件
 * @param output_filename 输出文件
 * @return 成功返回true，失败返回false
 */
bool png_remove_metadata(const char *input_filename, const char *output_filename);

// ============================================================================
// PNG优化
// ============================================================================

/**
 * @brief 优化PNG文件
 * @param input_filename 输入文件
 * @param output_filename 输出文件
 * @param options 优化选项
 * @return 成功返回true，失败返回false
 */
bool png_optimize(const char *input_filename, const char *output_filename,
                 const PNGOptimizeOptions *options);

/**
 * @brief 使用默认选项优化PNG
 * @param input_filename 输入文件
 * @param output_filename 输出文件
 * @return 成功返回true，失败返回false
 */
bool png_optimize_default(const char *input_filename, const char *output_filename);

/**
 * @brief 减少PNG颜色数
 * @param img 图像
 * @param max_colors 最大颜色数
 * @return 优化后的图像，失败返回NULL
 */
Image* png_reduce_colors(const Image *img, int max_colors);

/**
 * @brief 减少位深度
 * @param img 图像
 * @return 优化后的图像，失败返回NULL
 */
Image* png_reduce_bit_depth(const Image *img);

/**
 * @brief 移除不必要的alpha通道
 * @param img 图像
 * @return 优化后的图像，失败返回NULL
 */
Image* png_remove_alpha_if_opaque(const Image *img);

/**
 * @brief 检测图像是否完全不透明
 * @param img 图像
 * @return 完全不透明返回true，否则返回false
 */
bool png_is_fully_opaque(const Image *img);

// ============================================================================
// PNG转换
// ============================================================================

/**
 * @brief 转换为灰度图像
 * @param img 源图像
 * @return 灰度图像，失败返回NULL
 */
Image* png_convert_to_grayscale(const Image *img);

/**
 * @brief 转换为RGB
 * @param img 源图像
 * @return RGB图像，失败返回NULL
 */
Image* png_convert_to_rgb(const Image *img);

/**
 * @brief 转换为RGBA
 * @param img 源图像
 * @return RGBA图像，失败返回NULL
 */
Image* png_convert_to_rgba(const Image *img);

/**
 * @brief 转换为调色板模式
 * @param img 源图像
 * @param max_colors 最大颜色数
 * @return 调色板图像，失败返回NULL
 */
Image* png_convert_to_palette(const Image *img, int max_colors);

/**
 * @brief 转换位深度
 * @param img 源图像
 * @param new_bit_depth 新位深度
 * @return 转换后的图像，失败返回NULL
 */
Image* png_convert_bit_depth(const Image *img, int new_bit_depth);

// ============================================================================
// PNG图像变换
// ============================================================================

/**
 * @brief 旋转图像
 * @param img 源图像
 * @param angle 角度 (90, 180, 270)
 * @return 旋转后的图像，失败返回NULL
 */
Image* png_rotate(const Image *img, int angle);

/**
 * @brief 水平翻转
 * @param img 源图像
 * @return 翻转后的图像，失败返回NULL
 */
Image* png_flip_horizontal(const Image *img);

/**
 * @brief 垂直翻转
 * @param img 源图像
 * @return 翻转后的图像，失败返回NULL
 */
Image* png_flip_vertical(const Image *img);

/**
 * @brief 裁剪图像
 * @param img 源图像
 * @param x 起始X坐标
 * @param y 起始Y坐标
 * @param width 宽度
 * @param height 高度
 * @return 裁剪后的图像，失败返回NULL
 */
Image* png_crop(const Image *img, int x, int y, int width, int height);

/**
 * @brief 缩放图像（保持宽高比）
 * @param img 源图像
 * @param max_width 最大宽度
 * @param max_height 最大高度
 * @return 缩放后的图像，失败返回NULL
 */
Image* png_scale_to_fit(const Image *img, int max_width, int max_height);

// ============================================================================
// PNG透明度处理
// ============================================================================

/**
 * @brief 添加alpha通道
 * @param img 源图像
 * @param alpha_value 默认alpha值 (0-255)
 * @return 带alpha通道的图像，失败返回NULL
 */
Image* png_add_alpha_channel(const Image *img, int alpha_value);

/**
 * @brief 移除alpha通道
 * @param img 源图像
 * @param bg_color 背景颜色 (RGB)
 * @return 移除alpha后的图像，失败返回NULL
 */
Image* png_remove_alpha_channel(const Image *img, const unsigned char *bg_color);

/**
 * @brief 设置透明色
 * @param img 源图像
 * @param color 透明色 (RGB)
 * @return 处理后的图像，失败返回NULL
 */
Image* png_set_transparent_color(const Image *img, const unsigned char *color);

/**
 * @brief 调整透明度
 * @param img 源图像
 * @param alpha_factor 透明度因子 (0.0-1.0)
 * @return 处理后的图像，失败返回NULL
 */
Image* png_adjust_alpha(const Image *img, double alpha_factor);

/**
 * @brief 预乘alpha
 * @param img 源图像
 * @return 预乘后的图像，失败返回NULL
 */
Image* png_premultiply_alpha(const Image *img);

/**
 * @brief 取消预乘alpha
 * @param img 源图像
 * @return 处理后的图像，失败返回NULL
 */
Image* png_unpremultiply_alpha(const Image *img);

// ============================================================================
// PNG滤波器
// ============================================================================

/**
 * @brief 应用高斯模糊
 * @param img 源图像
 * @param radius 模糊半径
 * @return 模糊后的图像，失败返回NULL
 */
Image* png_gaussian_blur(const Image *img, double radius);

/**
 * @brief 锐化图像
 * @param img 源图像
 * @param amount 锐化程度
 * @return 锐化后的图像，失败返回NULL
 */
Image* png_sharpen(const Image *img, double amount);

/**
 * @brief 调整亮度
 * @param img 源图像
 * @param factor 亮度因子 (-1.0 到 1.0)
 * @return 调整后的图像，失败返回NULL
 */
Image* png_adjust_brightness(const Image *img, double factor);

/**
 * @brief 调整对比度
 * @param img 源图像
 * @param factor 对比度因子 (-1.0 到 1.0)
 * @return 调整后的图像，失败返回NULL
 */
Image* png_adjust_contrast(const Image *img, double factor);

/**
 * @brief 调整饱和度
 * @param img 源图像
 * @param factor 饱和度因子 (-1.0 到 1.0)
 * @return 调整后的图像，失败返回NULL
 */
Image* png_adjust_saturation(const Image *img, double factor);

/**
 * @brief 调整色调
 * @param img 源图像
 * @param hue_shift 色调偏移 (-180 到 180)
 * @return 调整后的图像，失败返回NULL
 */
Image* png_adjust_hue(const Image *img, double hue_shift);

// ============================================================================
// PNG合成操作
// ============================================================================

/**
 * @brief 合成两个图像
 * @param base 基础图像
 * @param overlay 覆盖图像
 * @param x X偏移
 * @param y Y偏移
 * @param opacity 不透明度 (0.0-1.0)
 * @return 合成后的图像，失败返回NULL
 */
Image* png_composite(const Image *base, const Image *overlay, 
                    int x, int y, double opacity);

/**
 * @brief 水平拼接图像
 * @param images 图像数组
 * @param num_images 图像数量
 * @param spacing 间距
 * @return 拼接后的图像，失败返回NULL
 */
Image* png_concat_horizontal(const Image **images, int num_images, int spacing);

/**
 * @brief 垂直拼接图像
 * @param images 图像数组
 * @param num_images 图像数量
 * @param spacing 间距
 * @return 拼接后的图像，失败返回NULL
 */
Image* png_concat_vertical(const Image **images, int num_images, int spacing);

/**
 * @brief 创建图像网格
 * @param images 图像数组
 * @param num_images 图像数量
 * @param cols 列数
 * @param spacing 间距
 * @return 网格图像，失败返回NULL
 */
Image* png_create_grid(const Image **images, int num_images, 
                      int cols, int spacing);

// ============================================================================
// APNG (动画PNG) 支持
// ============================================================================

/**
 * @brief 读取APNG动画
 * @param filename 文件名
 * @return 动画结构指针，失败返回NULL
 */
APNGAnimation* apng_read(const char *filename);

/**
 * @brief 写入APNG动画
 * @param animation 动画结构
 * @param filename 文件名
 * @return 成功返回true，失败返回false
 */
bool apng_write(const APNGAnimation *animation, const char *filename);

/**
 * @brief 创建APNG动画
 * @param width 画布宽度
 * @param height 画布高度
 * @param num_frames 帧数
 * @return 动画结构指针，失败返回NULL
 */
APNGAnimation* apng_create(int width, int height, int num_frames);

/**
 * @brief 销毁APNG动画
 * @param animation 动画指针
 */
void apng_destroy(APNGAnimation *animation);

/**
 * @brief 添加帧到APNG
 * @param animation 动画
 * @param frame_index 帧索引
 * @param image 帧图像
 * @param delay_num 延迟分子
 * @param delay_den 延迟分母
 * @return 成功返回true，失败返回false
 */
bool apng_add_frame(APNGAnimation *animation, int frame_index, 
                   const Image *image, int delay_num, int delay_den);

/**
 * @brief 提取APNG帧
 * @param animation 动画
 * @param frame_index 帧索引
 * @return 帧图像，失败返回NULL
 */
Image* apng_extract_frame(const APNGAnimation *animation, int frame_index);

/**
 * @brief 检查是否为APNG
 * @param filename 文件名
 * @return 是APNG返回true，否则返回false
 */
bool apng_is_animated(const char *filename);

// ============================================================================
// PNG批处理
// ============================================================================

/**
 * @brief 批量优化PNG文件
 * @param input_files 输入文件数组
 * @param num_files 文件数量
 * @param output_dir 输出目录
 * @param options 优化选项
 * @return 成功返回true，失败返回false
 */
bool png_batch_optimize(const char **input_files, int num_files,
                       const char *output_dir, 
                       const PNGOptimizeOptions *options);

/**
 * @brief 批量调整大小
 * @param input_files 输入文件数组
 * @param num_files 文件数量
 * @param output_dir 输出目录
 * @param width 目标宽度
 * @param height 目标高度
 * @return 成功返回true，失败返回false
 */
bool png_batch_resize(const char **input_files, int num_files,
                     const char *output_dir, int width, int height);

/**
 * @brief 批量转换格式
 * @param input_files 输入文件数组
 * @param num_files 文件数量
 * @param output_dir 输出目录
 * @param color_type 目标颜色类型
 * @param bit_depth 目标位深度
 * @return 成功返回true，失败返回false
 */
bool png_batch_convert(const char **input_files, int num_files,
                      const char *output_dir, int color_type, int bit_depth);

// ============================================================================
// 工具函数
// ============================================================================

/**
 * @brief 列出目录中的PNG文件
 * @param directory 目录路径
 * @param num_files 输出文件数量
 * @return 文件名数组，失败返回NULL
 */
char** png_list_files(const char *directory, int *num_files);

/**
 * @brief 释放文件列表
 * @param files 文件名数组
 * @param num_files 文件数量
 */
void png_free_file_list(char **files, int num_files);

/**
 * @brief 格式化文件大小
 * @param size 字节数
 * @return 格式化的字符串
 */
const char* png_format_file_size(long size);

/**
 * @brief 生成输出文件名
 * @param input_filename 输入文件名
 * @param suffix 后缀
 * @param output_dir 输出目录
 * @return 输出文件名，需要调用者释放
 */
char* png_generate_output_filename(const char *input_filename,
                                   const char *suffix,
                                   const char *output_dir);

/**
 * @brief 比较两个PNG文件
 * @param file1 文件1
 * @param file2 文件2
 * @param mse 输出MSE值
 * @param psnr 输出PSNR值
 * @return 成功返回true，失败返回false
 */
bool png_compare_files(const char *file1, const char *file2,
                      double *mse, double *psnr);

/**
 * @brief 打印版本信息
 */
void png_print_version(void);

/**
 * @brief 打印使用帮助
 * @param program_name 程序名
 */
void png_print_usage(const char *program_name);

/**
 * @brief 打印统计信息
 * @param operation 操作名称
 * @param total_files 总文件数
 * @param success_count 成功数
 * @param fail_count 失败数
 * @param size_before 处理前大小
 * @param size_after 处理后大小
 * @param elapsed_time 耗时
 */
void png_print_statistics(const char *operation,
                         int total_files,
                         int success_count,
                         int fail_count,
                         long size_before,
                         long size_after,
                         double elapsed_time);

// ============================================================================
// 错误处理
// ============================================================================

/**
 * @brief 获取最后的错误消息
 * @return 错误消息字符串
 */
const char* png_get_last_error(void);

/**
 * @brief 设置错误消息
 * @param format 格式字符串
 * @param ... 可变参数
 */
void png_set_error(const char *format, ...);

/**
 * @brief 清除错误消息
 */
void png_clear_error(void);

#ifdef __cplusplus
}
#endif

#endif // PNG_HANDLER_H
