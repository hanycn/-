/**
 * @file png_handler.c
 * @brief PNG图像处理库实现
 * @author Your Name
 * @date 2024
 * @version 
 */

#include "png_handler.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <math.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <dirent.h>
#include <errno.h>
#include <ctype.h>

// ============================================================================
// 内部宏定义
// ============================================================================

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define CLAMP(x, min, max) (MIN(MAX(x, min), max))
#define SWAP(a, b, type) do { type temp = a; a = b; b = temp; } while(0)

// 字节序转换
#define PNG_SWAP_BYTES_16(x) ((((x) & 0xFF) << 8) | (((x) >> 8) & 0xFF))

// 错误消息缓冲区
static char g_error_message[512] = {0};

// ============================================================================
// 错误处理
// ============================================================================

/**
 * @brief 设置错误消息
 */
void png_set_error(const char *format, ...)
{
    va_list args;
    va_start(args, format);
    vsnprintf(g_error_message, sizeof(g_error_message), format, args);
    va_end(args);
}

/**
 * @brief 获取最后的错误消息
 */
const char* png_get_last_error(void)
{
    return g_error_message;
}

/**
 * @brief 清除错误消息
 */
void png_clear_error(void)
{
    g_error_message[0] = '\0';
}

// ============================================================================
// 内部辅助函数
// ============================================================================

/**
 * @brief PNG错误回调函数
 */
static void png_error_callback(png_structp png_ptr, png_const_charp error_msg)
{
    png_set_error("PNG错误: %s", error_msg);
    longjmp(png_jmpbuf(png_ptr), 1);
}

/**
 * @brief PNG警告回调函数
 */
static void png_warning_callback(png_structp png_ptr, png_const_charp warning_msg)
{
    fprintf(stderr, "PNG警告: %s\n", warning_msg);
}

/**
 * @brief 计算字节每像素
 */
static inline int bytes_per_pixel(int channels, int bit_depth)
{
    return (channels * bit_depth + 7) / 8;
}

/**
 * @brief 计算行字节数
 */
static inline size_t row_bytes(int width, int channels, int bit_depth)
{
    return (size_t)width * bytes_per_pixel(channels, bit_depth);
}

/**
 * @brief 验证图像参数
 */
static bool validate_image_params(int width, int height, int channels, int bit_depth)
{
    if (width <= 0 || width > PNG_MAX_WIDTH) {
        png_set_error("无效的宽度: %d", width);
        return false;
    }
    
    if (height <= 0 || height > PNG_MAX_HEIGHT) {
        png_set_error("无效的高度: %d", height);
        return false;
    }
    
    if (channels < 1 || channels > 4) {
        png_set_error("无效的通道数: %d", channels);
        return false;
    }
    
    if (bit_depth != 8 && bit_depth != 16) {
        png_set_error("不支持的位深度: %d", bit_depth);
        return false;
    }
    
    return true;
}

/**
 * @brief 获取PNG颜色类型
 */
static int get_png_color_type(int channels)
{
    switch (channels) {
        case 1: return PNG_COLOR_TYPE_GRAY;
        case 2: return PNG_COLOR_TYPE_GRAY_ALPHA;
        case 3: return PNG_COLOR_TYPE_RGB;
        case 4: return PNG_COLOR_TYPE_RGB_ALPHA;
        default: return -1;
    }
}

/**
 * @brief 从PNG颜色类型获取通道数
 */
static int get_channels_from_color_type(int color_type)
{
    switch (color_type) {
        case PNG_COLOR_TYPE_GRAY: return 1;
        case PNG_COLOR_TYPE_GRAY_ALPHA: return 2;
        case PNG_COLOR_TYPE_RGB: return 3;
        case PNG_COLOR_TYPE_RGB_ALPHA: return 4;
        case PNG_COLOR_TYPE_PALETTE: return 3; // 转换为RGB
        default: return 0;
    }
}

/**
 * @brief 获取颜色类型名称
 */
static const char* get_color_type_name(int color_type)
{
    switch (color_type) {
        case PNG_COLOR_TYPE_GRAY: return "Grayscale";
        case PNG_COLOR_TYPE_GRAY_ALPHA: return "Grayscale + Alpha";
        case PNG_COLOR_TYPE_RGB: return "RGB";
        case PNG_COLOR_TYPE_RGB_ALPHA: return "RGBA";
        case PNG_COLOR_TYPE_PALETTE: return "Palette";
        default: return "Unknown";
    }
}

// ============================================================================
// 基本图像操作
// ============================================================================

/**
 * @brief 创建图像
 */
Image* image_create(int width, int height, int channels, int bit_depth)
{
    png_clear_error();
    
    if (!validate_image_params(width, height, channels, bit_depth)) {
        return NULL;
    }
    
    Image *img = (Image*)calloc(1, sizeof(Image));
    if (!img) {
        png_set_error("内存分配失败");
        return NULL;
    }
    
    img->width = width;
    img->height = height;
    img->channels = channels;
    img->bit_depth = bit_depth;
    
    // 计算数据大小
    img->data_size = (size_t)height * row_bytes(width, channels, bit_depth);
    
    // 分配数据内存
    img->data = (unsigned char*)calloc(img->data_size, 1);
    if (!img->data) {
        png_set_error("图像数据内存分配失败");
        free(img);
        return NULL;
    }
    
    return img;
}

/**
 * @brief 销毁图像
 */
void image_destroy(Image *img)
{
    if (!img) {
        return;
    }
    
    if (img->data) {
        free(img->data);
        img->data = NULL;
    }
    
    free(img);
}

/**
 * @brief 克隆图像
 */
Image* image_clone(const Image *img)
{
    png_clear_error();
    
    if (!img) {
        png_set_error("源图像为空");
        return NULL;
    }
    
    Image *clone = image_create(img->width, img->height, 
                               img->channels, img->bit_depth);
    if (!clone) {
        return NULL;
    }
    
    // 复制数据
    memcpy(clone->data, img->data, img->data_size);
    
    return clone;
}

/**
 * @brief 获取像素值（8位）
 */
static inline unsigned char get_pixel_8bit(const Image *img, int x, int y, int channel)
{
    if (x < 0 || x >= img->width || y < 0 || y >= img->height || 
        channel < 0 || channel >= img->channels) {
        return 0;
    }
    
    size_t index = (size_t)y * img->width * img->channels + 
                   (size_t)x * img->channels + channel;
    return img->data[index];
}

/**
 * @brief 设置像素值（8位）
 */
static inline void set_pixel_8bit(Image *img, int x, int y, int channel, 
                                 unsigned char value)
{
    if (x < 0 || x >= img->width || y < 0 || y >= img->height || 
        channel < 0 || channel >= img->channels) {
        return;
    }
    
    size_t index = (size_t)y * img->width * img->channels + 
                   (size_t)x * img->channels + channel;
    img->data[index] = value;
}

/**
 * @brief 获取像素值（16位）
 */
static inline unsigned short get_pixel_16bit(const Image *img, int x, int y, int channel)
{
    if (x < 0 || x >= img->width || y < 0 || y >= img->height || 
        channel < 0 || channel >= img->channels) {
        return 0;
    }
    
    size_t index = ((size_t)y * img->width * img->channels + 
                    (size_t)x * img->channels + channel) * 2;
    
    // 大端序
    return ((unsigned short)img->data[index] << 8) | img->data[index + 1];
}

/**
 * @brief 设置像素值（16位）
 */
static inline void set_pixel_16bit(Image *img, int x, int y, int channel, 
                                  unsigned short value)
{
    if (x < 0 || x >= img->width || y < 0 || y >= img->height || 
        channel < 0 || channel >= img->channels) {
        return;
    }
    
    size_t index = ((size_t)y * img->width * img->channels + 
                    (size_t)x * img->channels + channel) * 2;
    
    // 大端序
    img->data[index] = (value >> 8) & 0xFF;
    img->data[index + 1] = value & 0xFF;
}

/**
 * @brief 获取像素值（通用接口）
 */
int image_get_pixel(const Image *img, int x, int y, int channel)
{
    if (!img) {
        return 0;
    }
    
    if (img->bit_depth == 8) {
        return get_pixel_8bit(img, x, y, channel);
    } else if (img->bit_depth == 16) {
        return get_pixel_16bit(img, x, y, channel);
    }
    
    return 0;
}

/**
 * @brief 设置像素值（通用接口）
 */
void image_set_pixel(Image *img, int x, int y, int channel, int value)
{
    if (!img) {
        return;
    }
    
    if (img->bit_depth == 8) {
        set_pixel_8bit(img, x, y, channel, (unsigned char)value);
    } else if (img->bit_depth == 16) {
        set_pixel_16bit(img, x, y, channel, (unsigned short)value);
    }
}

/**
 * @brief 双线性插值
 */
static double bilinear_interpolate(const Image *img, double x, double y, int channel)
{
    int x0 = (int)floor(x);
    int y0 = (int)floor(y);
    int x1 = x0 + 1;
    int y1 = y0 + 1;
    
    double dx = x - x0;
    double dy = y - y0;
    
    // 边界处理
    x0 = CLAMP(x0, 0, img->width - 1);
    x1 = CLAMP(x1, 0, img->width - 1);
    y0 = CLAMP(y0, 0, img->height - 1);
    y1 = CLAMP(y1, 0, img->height - 1);
    
    // 获取四个角的像素值
    double p00 = image_get_pixel(img, x0, y0, channel);
    double p10 = image_get_pixel(img, x1, y0, channel);
    double p01 = image_get_pixel(img, x0, y1, channel);
    double p11 = image_get_pixel(img, x1, y1, channel);
    
    // 双线性插值
    double p0 = p00 * (1 - dx) + p10 * dx;
    double p1 = p01 * (1 - dx) + p11 * dx;
    double p = p0 * (1 - dy) + p1 * dy;
    
    return p;
}

/**
 * @brief 调整图像大小
 */
Image* image_resize(const Image *img, int new_width, int new_height)
{
    png_clear_error();
    
    if (!img) {
        png_set_error("源图像为空");
        return NULL;
    }
    
    if (new_width <= 0 || new_height <= 0) {
        png_set_error("无效的目标尺寸");
        return NULL;
    }
    
    // 创建新图像
    Image *resized = image_create(new_width, new_height, 
                                 img->channels, img->bit_depth);
    if (!resized) {
        return NULL;
    }
    
    // 计算缩放比例
    double scale_x = (double)img->width / new_width;
    double scale_y = (double)img->height / new_height;
    
    // 最大值（用于归一化）
    int max_value = (img->bit_depth == 8) ? 255 : 65535;
    
    // 对每个像素进行插值
    for (int y = 0; y < new_height; y++) {
        for (int x = 0; x < new_width; x++) {
            // 计算源图像中的对应位置
            double src_x = (x + 0.5) * scale_x - 0.5;
            double src_y = (y + 0.5) * scale_y - 0.5;
            
            // 对每个通道进行插值
            for (int c = 0; c < img->channels; c++) {
                double value = bilinear_interpolate(img, src_x, src_y, c);
                value = CLAMP(value, 0, max_value);
                image_set_pixel(resized, x, y, c, (int)round(value));
            }
        }
    }
    
    return resized;
}

/**
 * @brief 填充图像
 */
static void image_fill(Image *img, const unsigned char *color)
{
    if (!img || !color) {
        return;
    }
    
    if (img->bit_depth == 8) {
        for (int y = 0; y < img->height; y++) {
            for (int x = 0; x < img->width; x++) {
                for (int c = 0; c < img->channels; c++) {
                    set_pixel_8bit(img, x, y, c, color[c]);
                }
            }
        }
    } else if (img->bit_depth == 16) {
        unsigned short color16[4];
        for (int c = 0; c < img->channels; c++) {
            color16[c] = ((unsigned short)color[c] << 8) | color[c];
        }
        
        for (int y = 0; y < img->height; y++) {
            for (int x = 0; x < img->width; x++) {
                for (int c = 0; c < img->channels; c++) {
                    set_pixel_16bit(img, x, y, c, color16[c]);
                }
            }
        }
    }
}

/**
 * @brief 复制图像区域
 */
static void image_copy_region(Image *dst, int dst_x, int dst_y,
                             const Image *src, int src_x, int src_y,
                             int width, int height)
{
    if (!dst || !src) {
        return;
    }
    
    // 确保通道数和位深度相同
    if (dst->channels != src->channels || dst->bit_depth != src->bit_depth) {
        return;
    }
    
    // 裁剪到有效范围
    if (src_x < 0) {
        width += src_x;
        dst_x -= src_x;
        src_x = 0;
    }
    if (src_y < 0) {
        height += src_y;
        dst_y -= src_y;
        src_y = 0;
    }
    if (dst_x < 0) {
        width += dst_x;
        src_x -= dst_x;
        dst_x = 0;
    }
    if (dst_y < 0) {
        height += dst_y;
        src_y -= dst_y;
        dst_y = 0;
    }
    
    width = MIN(width, MIN(src->width - src_x, dst->width - dst_x));
    height = MIN(height, MIN(src->height - src_y, dst->height - dst_y));
    
    if (width <= 0 || height <= 0) {
        return;
    }
    
    // 复制数据
    size_t row_size = row_bytes(width, src->channels, src->bit_depth);
    size_t src_row_stride = row_bytes(src->width, src->channels, src->bit_depth);
    size_t dst_row_stride = row_bytes(dst->width, dst->channels, dst->bit_depth);
    
    for (int y = 0; y < height; y++) {
        unsigned char *src_ptr = src->data + 
            (src_y + y) * src_row_stride + 
            src_x * bytes_per_pixel(src->channels, src->bit_depth);
        
        unsigned char *dst_ptr = dst->data + 
            (dst_y + y) * dst_row_stride + 
            dst_x * bytes_per_pixel(dst->channels, dst->bit_depth);
        
        memcpy(dst_ptr, src_ptr, row_size);
    }
}

/**
 * @brief 创建子图像（视图，不复制数据）
 */
typedef struct {
    const Image *parent;
    int x_offset;
    int y_offset;
    int width;
    int height;
} ImageView;

/**
 * @brief 从视图获取像素
 */
static int image_view_get_pixel(const ImageView *view, int x, int y, int channel)
{
    if (!view || !view->parent) {
        return 0;
    }
    
    if (x < 0 || x >= view->width || y < 0 || y >= view->height) {
        return 0;
    }
    
    return image_get_pixel(view->parent, 
                          view->x_offset + x, 
                          view->y_offset + y, 
                          channel);
}

/**
 * @brief 计算图像直方图
 */
typedef struct {
    int histogram[256];
    int total_pixels;
} Histogram;

/**
 * @brief 计算单通道直方图
 */
static void compute_histogram(const Image *img, int channel, Histogram *hist)
{
    if (!img || !hist || channel < 0 || channel >= img->channels) {
        return;
    }
    
    memset(hist, 0, sizeof(Histogram));
    
    if (img->bit_depth == 8) {
        for (int y = 0; y < img->height; y++) {
            for (int x = 0; x < img->width; x++) {
                int value = get_pixel_8bit(img, x, y, channel);
                hist->histogram[value]++;
                hist->total_pixels++;
            }
        }
    } else if (img->bit_depth == 16) {
        // 16位转8位进行统计
        for (int y = 0; y < img->height; y++) {
            for (int x = 0; x < img->width; x++) {
                int value = get_pixel_16bit(img, x, y, channel) >> 8;
                hist->histogram[value]++;
                hist->total_pixels++;
            }
        }
    }
}

/**
 * @brief 直方图均衡化
 */
static void histogram_equalize_channel(Image *img, int channel)
{
    if (!img || channel < 0 || channel >= img->channels) {
        return;
    }
    
    // 计算直方图
    Histogram hist;
    compute_histogram(img, channel, &hist);
    
    // 计算累积分布函数
    int cdf[256] = {0};
    cdf[0] = hist.histogram[0];
    for (int i = 1; i < 256; i++) {
        cdf[i] = cdf[i-1] + hist.histogram[i];
    }
    
    // 找到第一个非零值
    int cdf_min = 0;
    for (int i = 0; i < 256; i++) {
        if (cdf[i] > 0) {
            cdf_min = cdf[i];
            break;
        }
    }
    
    // 计算映射表
    unsigned char map[256];
    for (int i = 0; i < 256; i++) {
        if (cdf[i] == 0) {
            map[i] = 0;
        } else {
            map[i] = (unsigned char)round(
                255.0 * (cdf[i] - cdf_min) / (hist.total_pixels - cdf_min)
            );
        }
    }
    
    // 应用映射
    if (img->bit_depth == 8) {
        for (int y = 0; y < img->height; y++) {
            for (int x = 0; x < img->width; x++) {
                unsigned char value = get_pixel_8bit(img, x, y, channel);
                set_pixel_8bit(img, x, y, channel, map[value]);
            }
        }
    } else if (img->bit_depth == 16) {
        for (int y = 0; y < img->height; y++) {
            for (int x = 0; x < img->width; x++) {
                unsigned short value = get_pixel_16bit(img, x, y, channel);
                unsigned char value8 = value >> 8;
                unsigned short new_value = ((unsigned short)map[value8] << 8) | 
                                          map[value8];
                set_pixel_16bit(img, x, y, channel, new_value);
            }
        }
    }
}

/**
 * @brief 图像直方图均衡化
 */
Image* image_histogram_equalize(const Image *img)
{
    png_clear_error();
    
    if (!img) {
        png_set_error("源图像为空");
        return NULL;
    }
    
    Image *result = image_clone(img);
    if (!result) {
        return NULL;
    }
    
    // 对每个通道进行均衡化（不包括alpha通道）
    int num_channels = (img->channels == 2 || img->channels == 4) ? 
                       img->channels - 1 : img->channels;
    
    for (int c = 0; c < num_channels; c++) {
        histogram_equalize_channel(result, c);
    }
    
    return result;
}

// 第一部分结束
// ============================================================================
// PNG基本读写操作
// ============================================================================

/**
 * @brief 检查PNG文件签名
 */
bool png_is_png_file(const char *filename)
{
    png_clear_error();
    
    if (!filename) {
        png_set_error("文件名为空");
        return false;
    }
    
    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        png_set_error("无法打开文件: %s", filename);
        return false;
    }
    
    unsigned char header[8];
    size_t read_size = fread(header, 1, 8, fp);
    fclose(fp);
    
    if (read_size != 8) {
        return false;
    }
    
    return (png_sig_cmp(header, 0, 8) == 0);
}

/**
 * @brief 读取PNG文件
 */
Image* png_read(const char *filename)
{
    png_clear_error();
    
    if (!filename) {
        png_set_error("文件名为空");
        return NULL;
    }
    
    // 打开文件
    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        png_set_error("无法打开文件: %s", filename);
        return NULL;
    }
    
    // 检查PNG签名
    unsigned char header[8];
    if (fread(header, 1, 8, fp) != 8) {
        png_set_error("读取文件头失败");
        fclose(fp);
        return NULL;
    }
    
    if (png_sig_cmp(header, 0, 8) != 0) {
        png_set_error("不是有效的PNG文件");
        fclose(fp);
        return NULL;
    }
    
    // 创建PNG读取结构
    png_structp png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING,
                                                 NULL,
                                                 png_error_callback,
                                                 png_warning_callback);
    if (!png_ptr) {
        png_set_error("创建PNG读取结构失败");
        fclose(fp);
        return NULL;
    }
    
    // 创建PNG信息结构
    png_infop info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr) {
        png_set_error("创建PNG信息结构失败");
        png_destroy_read_struct(&png_ptr, NULL, NULL);
        fclose(fp);
        return NULL;
    }
    
    // 设置错误处理
    if (setjmp(png_jmpbuf(png_ptr))) {
        png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
        fclose(fp);
        return NULL;
    }
    
    // 初始化IO
    png_init_io(png_ptr, fp);
    png_set_sig_bytes(png_ptr, 8);
    
    // 读取PNG信息
    png_read_info(png_ptr, info_ptr);
    
    // 获取图像信息
    int width = png_get_image_width(png_ptr, info_ptr);
    int height = png_get_image_height(png_ptr, info_ptr);
    int color_type = png_get_color_type(png_ptr, info_ptr);
    int bit_depth = png_get_bit_depth(png_ptr, info_ptr);
    
    // 转换处理
    // 调色板转RGB
    if (color_type == PNG_COLOR_TYPE_PALETTE) {
        png_set_palette_to_rgb(png_ptr);
        color_type = PNG_COLOR_TYPE_RGB;
    }
    
    // 灰度<8位扩展到8位
    if (color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8) {
        png_set_expand_gray_1_2_4_to_8(png_ptr);
        bit_depth = 8;
    }
    
    // tRNS块转alpha通道
    if (png_get_valid(png_ptr, info_ptr, PNG_INFO_tRNS)) {
        png_set_tRNS_to_alpha(png_ptr);
        if (color_type == PNG_COLOR_TYPE_GRAY) {
            color_type = PNG_COLOR_TYPE_GRAY_ALPHA;
        } else if (color_type == PNG_COLOR_TYPE_RGB) {
            color_type = PNG_COLOR_TYPE_RGB_ALPHA;
        }
    }
    
    // 16位大端序转换
    if (bit_depth == 16) {
        png_set_swap(png_ptr);
    }
    
    // 更新信息
    png_read_update_info(png_ptr, info_ptr);
    
    // 获取通道数
    int channels = get_channels_from_color_type(color_type);
    if (channels == 0) {
        png_set_error("不支持的颜色类型");
        png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
        fclose(fp);
        return NULL;
    }
    
    // 创建图像
    Image *img = image_create(width, height, channels, bit_depth);
    if (!img) {
        png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
        fclose(fp);
        return NULL;
    }
    
    // 分配行指针
    png_bytep *row_pointers = (png_bytep*)malloc(sizeof(png_bytep) * height);
    if (!row_pointers) {
        png_set_error("分配行指针内存失败");
        image_destroy(img);
        png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
        fclose(fp);
        return NULL;
    }
    
    // 设置行指针
    size_t row_stride = row_bytes(width, channels, bit_depth);
    for (int y = 0; y < height; y++) {
        row_pointers[y] = img->data + y * row_stride;
    }
    
    // 读取图像数据
    png_read_image(png_ptr, row_pointers);
    
    // 读取结束
    png_read_end(png_ptr, NULL);
    
    // 清理
    free(row_pointers);
    png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
    fclose(fp);
    
    return img;
}

/**
 * @brief 初始化默认写入选项
 */
static void init_default_write_options(PNGWriteOptions *options)
{
    options->compression_level = PNG_COMPRESSION_DEFAULT;
    options->filter_type = PNG_FILTER_TYPE_ALL;
    options->interlace_type = PNG_INTERLACE_NONE;
    options->optimize = false;
    options->strip_metadata = false;
    options->gamma = 0.0;
}

/**
 * @brief 写入PNG文件（带选项）
 */
bool png_write_with_options(const Image *img, const char *filename,
                            const PNGWriteOptions *options)
{
    png_clear_error();
    
    if (!img || !filename) {
        png_set_error("参数为空");
        return false;
    }
    
    if (!img->data) {
        png_set_error("图像数据为空");
        return false;
    }
    
    // 打开文件
    FILE *fp = fopen(filename, "wb");
    if (!fp) {
        png_set_error("无法创建文件: %s", filename);
        return false;
    }
    
    // 创建PNG写入结构
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING,
                                                  NULL,
                                                  png_error_callback,
                                                  png_warning_callback);
    if (!png_ptr) {
        png_set_error("创建PNG写入结构失败");
        fclose(fp);
        return false;
    }
    
    // 创建PNG信息结构
    png_infop info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr) {
        png_set_error("创建PNG信息结构失败");
        png_destroy_write_struct(&png_ptr, NULL);
        fclose(fp);
        return false;
    }
    
    // 设置错误处理
    if (setjmp(png_jmpbuf(png_ptr))) {
        png_destroy_write_struct(&png_ptr, &info_ptr);
        fclose(fp);
        return false;
    }
    
    // 初始化IO
    png_init_io(png_ptr, fp);
    
    // 设置压缩级别
    int compression = options ? options->compression_level : PNG_COMPRESSION_DEFAULT;
    compression = CLAMP(compression, 0, 9);
    png_set_compression_level(png_ptr, compression);
    
    // 设置滤波器
    int filter = options ? options->filter_type : PNG_FILTER_TYPE_ALL;
    if (filter >= 0 && filter <= PNG_FILTER_TYPE_ALL) {
        if (filter == PNG_FILTER_TYPE_ALL) {
            png_set_filter(png_ptr, 0, PNG_ALL_FILTERS);
        } else {
            png_set_filter(png_ptr, 0, 1 << filter);
        }
    }
    
    // 获取颜色类型
    int color_type = get_png_color_type(img->channels);
    if (color_type < 0) {
        png_set_error("不支持的通道数: %d", img->channels);
        png_destroy_write_struct(&png_ptr, &info_ptr);
        fclose(fp);
        return false;
    }
    
    // 设置交错类型
    int interlace = options ? options->interlace_type : PNG_INTERLACE_NONE;
    
    // 设置IHDR
    png_set_IHDR(png_ptr, info_ptr,
                img->width, img->height,
                img->bit_depth,
                color_type,
                interlace,
                PNG_COMPRESSION_TYPE_DEFAULT,
                PNG_FILTER_TYPE_DEFAULT);
    
    // 设置gamma
    if (options && options->gamma > 0.0) {
        png_set_gAMA(png_ptr, info_ptr, options->gamma);
    }
    
    // 16位字节序转换
    if (img->bit_depth == 16) {
        png_set_swap(png_ptr);
    }
    
    // 写入信息
    png_write_info(png_ptr, info_ptr);
    
    // 分配行指针
    png_bytep *row_pointers = (png_bytep*)malloc(sizeof(png_bytep) * img->height);
    if (!row_pointers) {
        png_set_error("分配行指针内存失败");
        png_destroy_write_struct(&png_ptr, &info_ptr);
        fclose(fp);
        return false;
    }
    
    // 设置行指针
    size_t row_stride = row_bytes(img->width, img->channels, img->bit_depth);
    for (int y = 0; y < img->height; y++) {
        row_pointers[y] = img->data + y * row_stride;
    }
    
    // 写入图像数据
    png_write_image(png_ptr, row_pointers);
    
    // 写入结束
    png_write_end(png_ptr, NULL);
    
    // 清理
    free(row_pointers);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
    
    return true;
}

/**
 * @brief 写入PNG文件（简化版）
 */
bool png_write(const Image *img, const char *filename, int compression_level)
{
    PNGWriteOptions options;
    init_default_write_options(&options);
    options.compression_level = compression_level;
    
    return png_write_with_options(img, filename, &options);
}
// ============================================================================
// PNG内存读写
// ============================================================================

/**
 * @brief 内存读取回调结构
 */
typedef struct {
    const unsigned char *buffer;
    size_t size;
    size_t offset;
} MemoryReadState;

/**
 * @brief 内存读取回调函数
 */
static void png_read_from_memory_callback(png_structp png_ptr,
                                         png_bytep data,
                                         png_size_t length)
{
    MemoryReadState *state = (MemoryReadState*)png_get_io_ptr(png_ptr);
    
    if (state->offset + length > state->size) {
        png_error(png_ptr, "读取超出缓冲区范围");
        return;
    }
    
    memcpy(data, state->buffer + state->offset, length);
    state->offset += length;
}

/**
 * @brief 从内存读取PNG
 */
Image* png_read_from_memory(const unsigned char *buffer, size_t size)
{
    png_clear_error();
    
    if (!buffer || size < 8) {
        png_set_error("无效的缓冲区");
        return NULL;
    }
    
    // 检查PNG签名
    if (png_sig_cmp(buffer, 0, 8) != 0) {
        png_set_error("不是有效的PNG数据");
        return NULL;
    }
    
    // 创建PNG读取结构
    png_structp png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING,
                                                 NULL,
                                                 png_error_callback,
                                                 png_warning_callback);
    if (!png_ptr) {
        png_set_error("创建PNG读取结构失败");
        return NULL;
    }
    
    // 创建PNG信息结构
    png_infop info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr) {
        png_set_error("创建PNG信息结构失败");
        png_destroy_read_struct(&png_ptr, NULL, NULL);
        return NULL;
    }
    
    // 设置错误处理
    if (setjmp(png_jmpbuf(png_ptr))) {
        png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
        return NULL;
    }
    
    // 设置内存读取
    MemoryReadState state;
    state.buffer = buffer;
    state.size = size;
    state.offset = 8; // 跳过签名
    
    png_set_read_fn(png_ptr, &state, png_read_from_memory_callback);
    png_set_sig_bytes(png_ptr, 8);
    
    // 读取PNG信息
    png_read_info(png_ptr, info_ptr);
    
    // 获取图像信息
    int width = png_get_image_width(png_ptr, info_ptr);
    int height = png_get_image_height(png_ptr, info_ptr);
    int color_type = png_get_color_type(png_ptr, info_ptr);
    int bit_depth = png_get_bit_depth(png_ptr, info_ptr);
    
    // 转换处理（与文件读取相同）
    if (color_type == PNG_COLOR_TYPE_PALETTE) {
        png_set_palette_to_rgb(png_ptr);
        color_type = PNG_COLOR_TYPE_RGB;
    }
    
    if (color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8) {
        png_set_expand_gray_1_2_4_to_8(png_ptr);
        bit_depth = 8;
    }
    
    if (png_get_valid(png_ptr, info_ptr, PNG_INFO_tRNS)) {
        png_set_tRNS_to_alpha(png_ptr);
        if (color_type == PNG_COLOR_TYPE_GRAY) {
            color_type = PNG_COLOR_TYPE_GRAY_ALPHA;
        } else if (color_type == PNG_COLOR_TYPE_RGB) {
            color_type = PNG_COLOR_TYPE_RGB_ALPHA;
        }
    }
    
    if (bit_depth == 16) {
        png_set_swap(png_ptr);
    }
    
    png_read_update_info(png_ptr, info_ptr);
    
    // 获取通道数
    int channels = get_channels_from_color_type(color_type);
    if (channels == 0) {
        png_set_error("不支持的颜色类型");
        png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
        return NULL;
    }
    
    // 创建图像
    Image *img = image_create(width, height, channels, bit_depth);
    if (!img) {
        png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
        return NULL;
    }
    
    // 分配行指针
    png_bytep *row_pointers = (png_bytep*)malloc(sizeof(png_bytep) * height);
    if (!row_pointers) {
        png_set_error("分配行指针内存失败");
        image_destroy(img);
        png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
        return NULL;
    }
    
    // 设置行指针
    size_t row_stride = row_bytes(width, channels, bit_depth);
    for (int y = 0; y < height; y++) {
        row_pointers[y] = img->data + y * row_stride;
    }
    
    // 读取图像数据
    png_read_image(png_ptr, row_pointers);
    png_read_end(png_ptr, NULL);
    
    // 清理
    free(row_pointers);
    png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
    
    return img;
}

/**
 * @brief 内存写入回调结构
 */
typedef struct {
    unsigned char *buffer;
    size_t size;
    size_t capacity;
} MemoryWriteState;

/**
 * @brief 内存写入回调函数
 */
static void png_write_to_memory_callback(png_structp png_ptr,
                                        png_bytep data,
                                        png_size_t length)
{
    MemoryWriteState *state = (MemoryWriteState*)png_get_io_ptr(png_ptr);
    
    // 扩展缓冲区
    if (state->size + length > state->capacity) {
        size_t new_capacity = state->capacity * 2;
        if (new_capacity < state->size + length) {
            new_capacity = state->size + length;
        }
        
        unsigned char *new_buffer = (unsigned char*)realloc(state->buffer, 
                                                            new_capacity);
        if (!new_buffer) {
            png_error(png_ptr, "内存分配失败");
            return;
        }
        
        state->buffer = new_buffer;
        state->capacity = new_capacity;
    }
    
    memcpy(state->buffer + state->size, data, length);
    state->size += length;
}

/**
 * @brief 内存写入刷新回调（空操作）
 */
static void png_write_to_memory_flush(png_structp png_ptr)
{
    // 不需要刷新
    (void)png_ptr;
}

/**
 * @brief 写入PNG到内存
 */
bool png_write_to_memory(const Image *img, unsigned char **buffer,
                        size_t *size, int compression_level)
{
    png_clear_error();
    
    if (!img || !buffer || !size) {
        png_set_error("参数为空");
        return false;
    }
    
    if (!img->data) {
        png_set_error("图像数据为空");
        return false;
    }
    
    // 创建PNG写入结构
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING,
                                                  NULL,
                                                  png_error_callback,
                                                  png_warning_callback);
    if (!png_ptr) {
        png_set_error("创建PNG写入结构失败");
        return false;
    }
    
    // 创建PNG信息结构
    png_infop info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr) {
        png_set_error("创建PNG信息结构失败");
        png_destroy_write_struct(&png_ptr, NULL);
        return false;
    }
    
    // 设置错误处理
    if (setjmp(png_jmpbuf(png_ptr))) {
        png_destroy_write_struct(&png_ptr, &info_ptr);
        return false;
    }
    
    // 初始化内存写入状态
    MemoryWriteState state;
    state.capacity = 65536; // 初始64KB
    state.size = 0;
    state.buffer = (unsigned char*)malloc(state.capacity);
    if (!state.buffer) {
        png_set_error("分配内存缓冲区失败");
        png_destroy_write_struct(&png_ptr, &info_ptr);
        return false;
    }
    
    // 设置内存写入
    png_set_write_fn(png_ptr, &state,
                    png_write_to_memory_callback,
                    png_write_to_memory_flush);
    
    // 设置压缩级别
    compression_level = CLAMP(compression_level, 0, 9);
    png_set_compression_level(png_ptr, compression_level);
    
    // 获取颜色类型
    int color_type = get_png_color_type(img->channels);
    if (color_type < 0) {
        png_set_error("不支持的通道数: %d", img->channels);
        free(state.buffer);
        png_destroy_write_struct(&png_ptr, &info_ptr);
        return false;
    }
    
    // 设置IHDR
    png_set_IHDR(png_ptr, info_ptr,
                img->width, img->height,
                img->bit_depth,
                color_type,
                PNG_INTERLACE_NONE,
                PNG_COMPRESSION_TYPE_DEFAULT,
                PNG_FILTER_TYPE_DEFAULT);
    
    // 16位字节序转换
    if (img->bit_depth == 16) {
        png_set_swap(png_ptr);
    }
    
    // 写入信息
    png_write_info(png_ptr, info_ptr);
    
    // 分配行指针
    png_bytep *row_pointers = (png_bytep*)malloc(sizeof(png_bytep) * img->height);
    if (!row_pointers) {
        png_set_error("分配行指针内存失败");
        free(state.buffer);
        png_destroy_write_struct(&png_ptr, &info_ptr);
        return false;
    }
    
    // 设置行指针
    size_t row_stride = row_bytes(img->width, img->channels, img->bit_depth);
    for (int y = 0; y < img->height; y++) {
        row_pointers[y] = img->data + y * row_stride;
    }
    
    // 写入图像数据
    png_write_image(png_ptr, row_pointers);
    png_write_end(png_ptr, NULL);
    
    // 清理
    free(row_pointers);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    
    // 返回结果
    *buffer = state.buffer;
    *size = state.size;
    
    return true;
}

/**
 * @brief 释放内存缓冲区
 */
void png_free_memory(unsigned char *buffer)
{
    if (buffer) {
        free(buffer);
    }
}
// ============================================================================
// PNG信息获取
// ============================================================================

/**
 * @brief 获取PNG文件信息
 */
bool png_get_info(const char *filename, PNGInfo *info)
{
    png_clear_error();
    
    if (!filename || !info) {
        png_set_error("参数为空");
        return false;
    }
    
    memset(info, 0, sizeof(PNGInfo));
    
    // 打开文件
    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        png_set_error("无法打开文件: %s", filename);
        return false;
    }
    
    // 获取文件大小
    struct stat st;
    if (fstat(fileno(fp), &st) == 0) {
        info->file_size = st.st_size;
    }
    
    // 检查PNG签名
    unsigned char header[8];
    if (fread(header, 1, 8, fp) != 8) {
        png_set_error("读取文件头失败");
        fclose(fp);
        return false;
    }
    
    if (png_sig_cmp(header, 0, 8) != 0) {
        png_set_error("不是有效的PNG文件");
        fclose(fp);
        return false;
    }
    
    // 创建PNG读取结构
    png_structp png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING,
                                                 NULL, NULL, NULL);
    if (!png_ptr) {
        png_set_error("创建PNG读取结构失败");
        fclose(fp);
        return false;
    }
    
    // 创建PNG信息结构
    png_infop info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr) {
        png_set_error("创建PNG信息结构失败");
        png_destroy_read_struct(&png_ptr, NULL, NULL);
        fclose(fp);
        return false;
    }
    
    // 设置错误处理
    if (setjmp(png_jmpbuf(png_ptr))) {
        png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
        fclose(fp);
        return false;
    }
    
    // 初始化IO
    png_init_io(png_ptr, fp);
    png_set_sig_bytes(png_ptr, 8);
    
    // 读取PNG信息
    png_read_info(png_ptr, info_ptr);
    
    // 获取基本信息
    info->width = png_get_image_width(png_ptr, info_ptr);
    info->height = png_get_image_height(png_ptr, info_ptr);
    info->bit_depth = png_get_bit_depth(png_ptr, info_ptr);
    info->color_type = png_get_color_type(png_ptr, info_ptr);
    info->interlace_type = png_get_interlace_type(png_ptr, info_ptr);
    info->compression_type = png_get_compression_type(png_ptr, info_ptr);
    info->filter_method = png_get_filter_type(png_ptr, info_ptr);
    
    // 获取通道数
    info->channels = get_channels_from_color_type(info->color_type);
    
    // 检查透明度
    info->has_transparency = (png_get_valid(png_ptr, info_ptr, PNG_INFO_tRNS) != 0) ||
                            (info->color_type == PNG_COLOR_TYPE_GRAY_ALPHA) ||
                            (info->color_type == PNG_COLOR_TYPE_RGB_ALPHA);
    
    // 检查gamma
    info->has_gamma = (png_get_valid(png_ptr, info_ptr, PNG_INFO_gAMA) != 0);
    if (info->has_gamma) {
        png_get_gAMA(png_ptr, info_ptr, &info->gamma);
    }
    
    // 获取颜色类型名称
    strncpy(info->color_type_name, 
            get_color_type_name(info->color_type),
            sizeof(info->color_type_name) - 1);
    
    // 清理
    png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
    fclose(fp);
    
    return true;
}

/**
 * @brief 打印PNG信息
 */
void png_print_info(const PNGInfo *info)
{
    if (!info) {
        return;
    }
    
    printf("PNG图像信息:\n");
    printf("  尺寸: %dx%d\n", info->width, info->height);
    printf("  位深度: %d\n", info->bit_depth);
    printf("  颜色类型: %s (%d)\n", info->color_type_name, info->color_type);
    printf("  通道数: %d\n", info->channels);
    printf("  交错类型: %s\n", 
           info->interlace_type == PNG_INTERLACE_NONE ? "无" : "Adam7");
    printf("  压缩类型: %d\n", info->compression_type);
    printf("  滤波方法: %d\n", info->filter_method);
    printf("  透明度: %s\n", info->has_transparency ? "是" : "否");
    
    if (info->has_gamma) {
        printf("  Gamma: %.4f\n", info->gamma);
    }
    
    if (info->file_size > 0) {
        printf("  文件大小: %s\n", png_format_file_size(info->file_size));
    }
}

/**
 * @brief 验证PNG文件
 */
bool png_validate(const char *filename, char *error_msg, int error_msg_size)
{
    png_clear_error();
    
    if (!filename) {
        if (error_msg) {
            snprintf(error_msg, error_msg_size, "文件名为空");
        }
        return false;
    }
    
    // 检查文件是否存在
    struct stat st;
    if (stat(filename, &st) != 0) {
        if (error_msg) {
            snprintf(error_msg, error_msg_size, "文件不存在");
        }
        return false;
    }
    
    // 检查文件大小
    if (st.st_size < 8) {
        if (error_msg) {
            snprintf(error_msg, error_msg_size, "文件太小，不是有效的PNG");
        }
        return false;
    }
    
    // 打开文件
    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        if (error_msg) {
            snprintf(error_msg, error_msg_size, "无法打开文件");
        }
        return false;
    }
    
    // 检查PNG签名
    unsigned char header[8];
    if (fread(header, 1, 8, fp) != 8) {
        if (error_msg) {
            snprintf(error_msg, error_msg_size, "读取文件头失败");
        }
        fclose(fp);
        return false;
    }
    
    if (png_sig_cmp(header, 0, 8) != 0) {
        if (error_msg) {
            snprintf(error_msg, error_msg_size, "不是有效的PNG文件签名");
        }
        fclose(fp);
        return false;
    }
    
    // 创建PNG读取结构
    png_structp png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING,
                                                 NULL, NULL, NULL);
    if (!png_ptr) {
        if (error_msg) {
            snprintf(error_msg, error_msg_size, "创建PNG读取结构失败");
        }
        fclose(fp);
        return false;
    }
    
    // 创建PNG信息结构
    png_infop info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr) {
        if (error_msg) {
            snprintf(error_msg, error_msg_size, "创建PNG信息结构失败");
        }
        png_destroy_read_struct(&png_ptr, NULL, NULL);
        fclose(fp);
        return false;
    }
    
    // 设置错误处理
    if (setjmp(png_jmpbuf(png_ptr))) {
        if (error_msg) {
            const char *err = png_get_last_error();
            snprintf(error_msg, error_msg_size, "PNG验证失败: %s", 
                    err ? err : "未知错误");
        }
        png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
        fclose(fp);
        return false;
    }
    
    // 初始化IO
    png_init_io(png_ptr, fp);
    png_set_sig_bytes(png_ptr, 8);
    
    // 读取并验证PNG信息
    png_read_info(png_ptr, info_ptr);
    
    // 获取图像信息进行基本验证
    int width = png_get_image_width(png_ptr, info_ptr);
    int height = png_get_image_height(png_ptr, info_ptr);
    int bit_depth = png_get_bit_depth(png_ptr, info_ptr);
    int color_type = png_get_color_type(png_ptr, info_ptr);
    
    // 验证尺寸
    if (width <= 0 || height <= 0) {
        if (error_msg) {
            snprintf(error_msg, error_msg_size, "无效的图像尺寸: %dx%d", 
                    width, height);
        }
        png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
        fclose(fp);
        return false;
    }
    
    // 验证位深度
    if (bit_depth != 1 && bit_depth != 2 && bit_depth != 4 && 
        bit_depth != 8 && bit_depth != 16) {
        if (error_msg) {
            snprintf(error_msg, error_msg_size, "无效的位深度: %d", bit_depth);
        }
        png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
        fclose(fp);
        return false;
    }
    
    // 验证颜色类型
    if (color_type != PNG_COLOR_TYPE_GRAY &&
        color_type != PNG_COLOR_TYPE_GRAY_ALPHA &&
        color_type != PNG_COLOR_TYPE_PALETTE &&
        color_type != PNG_COLOR_TYPE_RGB &&
        color_type != PNG_COLOR_TYPE_RGB_ALPHA) {
        if (error_msg) {
            snprintf(error_msg, error_msg_size, "无效的颜色类型: %d", color_type);
        }
        png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
        fclose(fp);
        return false;
    }
    
    // 尝试读取整个图像以验证数据完整性
    png_bytep *row_pointers = (png_bytep*)malloc(sizeof(png_bytep) * height);
    if (!row_pointers) {
        if (error_msg) {
            snprintf(error_msg, error_msg_size, "内存分配失败");
        }
        png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
        fclose(fp);
        return false;
    }
    
    // 分配临时行缓冲区
    size_t rowbytes = png_get_rowbytes(png_ptr, info_ptr);
    for (int y = 0; y < height; y++) {
        row_pointers[y] = (png_bytep)malloc(rowbytes);
        if (!row_pointers[y]) {
            // 清理已分配的行
            for (int i = 0; i < y; i++) {
                free(row_pointers[i]);
            }
            free(row_pointers);
            if (error_msg) {
                snprintf(error_msg, error_msg_size, "内存分配失败");
            }
            png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
            fclose(fp);
            return false;
        }
    }
    
    // 读取图像数据
    png_read_image(png_ptr, row_pointers);
    png_read_end(png_ptr, NULL);
    
    // 清理
    for (int y = 0; y < height; y++) {
        free(row_pointers[y]);
    }
    free(row_pointers);
    png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
    fclose(fp);
    
    if (error_msg) {
        snprintf(error_msg, error_msg_size, "验证成功");
    }
    
    return true;
}

/**
 * @brief 比较两个PNG文件
 */
bool png_compare(const char *filename1, const char *filename2, 
                PNGCompareResult *result)
{
    png_clear_error();
    
    if (!filename1 || !filename2 || !result) {
        png_set_error("参数为空");
        return false;
    }
    
    memset(result, 0, sizeof(PNGCompareResult));
    
    // 读取两个图像
    Image *img1 = png_read(filename1);
    if (!img1) {
        return false;
    }
    
    Image *img2 = png_read(filename2);
    if (!img2) {
        image_destroy(img1);
        return false;
    }
    
    // 比较尺寸
    result->same_dimensions = (img1->width == img2->width && 
                              img1->height == img2->height);
    
    // 比较通道数
    result->same_channels = (img1->channels == img2->channels);
    
    // 比较位深度
    result->same_bit_depth = (img1->bit_depth == img2->bit_depth);
    
    // 如果格式不同，无法比较像素
    if (!result->same_dimensions || !result->same_channels || 
        !result->same_bit_depth) {
        result->identical = false;
        result->difference_percentage = 100.0;
        image_destroy(img1);
        image_destroy(img2);
        return true;
    }
    
    // 比较像素数据
    size_t total_bytes = img1->width * img1->height * img1->channels * 
                        (img1->bit_depth / 8);
    size_t diff_bytes = 0;
    
    for (size_t i = 0; i < total_bytes; i++) {
        if (img1->data[i] != img2->data[i]) {
            diff_bytes++;
        }
    }
    
    result->identical = (diff_bytes == 0);
    result->difference_percentage = (double)diff_bytes / total_bytes * 100.0;
    
    image_destroy(img1);
    image_destroy(img2);
    
    return true;
}

/**
 * @brief 打印比较结果
 */
void png_print_compare_result(const PNGCompareResult *result)
{
    if (!result) {
        return;
    }
    
    printf("PNG比较结果:\n");
    printf("  尺寸相同: %s\n", result->same_dimensions ? "是" : "否");
    printf("  通道相同: %s\n", result->same_channels ? "是" : "否");
    printf("  位深度相同: %s\n", result->same_bit_depth ? "是" : "否");
    printf("  完全相同: %s\n", result->identical ? "是" : "否");
    
    if (!result->identical) {
        printf("  差异百分比: %.2f%%\n", result->difference_percentage);
    }
}
// ============================================================================
// 图像转换
// ============================================================================

/**
 * @brief 转换为灰度图像
 */
Image* png_convert_to_grayscale(const Image *img)
{
    png_clear_error();
    
    if (!img || !img->data) {
        png_set_error("图像为空");
        return NULL;
    }
    
    // 如果已经是灰度图，直接复制
    if (img->channels == 1) {
        return image_clone(img);
    }
    
    // 创建灰度图像
    Image *gray = image_create(img->width, img->height, 1, img->bit_depth);
    if (!gray) {
        return NULL;
    }
    
    if (img->bit_depth == 8) {
        // 8位处理
        for (int y = 0; y < img->height; y++) {
            for (int x = 0; x < img->width; x++) {
                size_t src_idx = (y * img->width + x) * img->channels;
                size_t dst_idx = y * img->width + x;
                
                if (img->channels >= 3) {
                    // RGB或RGBA转灰度：0.299*R + 0.587*G + 0.114*B
                    uint8_t r = img->data[src_idx];
                    uint8_t g = img->data[src_idx + 1];
                    uint8_t b = img->data[src_idx + 2];
                    gray->data[dst_idx] = (uint8_t)(0.299 * r + 0.587 * g + 0.114 * b);
                } else if (img->channels == 2) {
                    // 灰度+Alpha，取灰度值
                    gray->data[dst_idx] = img->data[src_idx];
                }
            }
        }
    } else if (img->bit_depth == 16) {
        // 16位处理
        uint16_t *src_data = (uint16_t*)img->data;
        uint16_t *dst_data = (uint16_t*)gray->data;
        
        for (int y = 0; y < img->height; y++) {
            for (int x = 0; x < img->width; x++) {
                size_t src_idx = (y * img->width + x) * img->channels;
                size_t dst_idx = y * img->width + x;
                
                if (img->channels >= 3) {
                    uint16_t r = src_data[src_idx];
                    uint16_t g = src_data[src_idx + 1];
                    uint16_t b = src_data[src_idx + 2];
                    dst_data[dst_idx] = (uint16_t)(0.299 * r + 0.587 * g + 0.114 * b);
                } else if (img->channels == 2) {
                    dst_data[dst_idx] = src_data[src_idx];
                }
            }
        }
    }
    
    return gray;
}

/**
 * @brief 转换为RGB图像
 */
Image* png_convert_to_rgb(const Image *img)
{
    png_clear_error();
    
    if (!img || !img->data) {
        png_set_error("图像为空");
        return NULL;
    }
    
    // 如果已经是RGB，直接复制
    if (img->channels == 3) {
        return image_clone(img);
    }
    
    // 创建RGB图像
    Image *rgb = image_create(img->width, img->height, 3, img->bit_depth);
    if (!rgb) {
        return NULL;
    }
    
    if (img->bit_depth == 8) {
        // 8位处理
        for (int y = 0; y < img->height; y++) {
            for (int x = 0; x < img->width; x++) {
                size_t src_idx = (y * img->width + x) * img->channels;
                size_t dst_idx = (y * img->width + x) * 3;
                
                if (img->channels == 1) {
                    // 灰度转RGB
                    uint8_t gray = img->data[src_idx];
                    rgb->data[dst_idx] = gray;
                    rgb->data[dst_idx + 1] = gray;
                    rgb->data[dst_idx + 2] = gray;
                } else if (img->channels == 2) {
                    // 灰度+Alpha转RGB（忽略Alpha）
                    uint8_t gray = img->data[src_idx];
                    rgb->data[dst_idx] = gray;
                    rgb->data[dst_idx + 1] = gray;
                    rgb->data[dst_idx + 2] = gray;
                } else if (img->channels == 4) {
                    // RGBA转RGB（忽略Alpha）
                    rgb->data[dst_idx] = img->data[src_idx];
                    rgb->data[dst_idx + 1] = img->data[src_idx + 1];
                    rgb->data[dst_idx + 2] = img->data[src_idx + 2];
                }
            }
        }
    } else if (img->bit_depth == 16) {
        // 16位处理
        uint16_t *src_data = (uint16_t*)img->data;
        uint16_t *dst_data = (uint16_t*)rgb->data;
        
        for (int y = 0; y < img->height; y++) {
            for (int x = 0; x < img->width; x++) {
                size_t src_idx = (y * img->width + x) * img->channels;
                size_t dst_idx = (y * img->width + x) * 3;
                
                if (img->channels == 1) {
                    uint16_t gray = src_data[src_idx];
                    dst_data[dst_idx] = gray;
                    dst_data[dst_idx + 1] = gray;
                    dst_data[dst_idx + 2] = gray;
                } else if (img->channels == 2) {
                    uint16_t gray = src_data[src_idx];
                    dst_data[dst_idx] = gray;
                    dst_data[dst_idx + 1] = gray;
                    dst_data[dst_idx + 2] = gray;
                } else if (img->channels == 4) {
                    dst_data[dst_idx] = src_data[src_idx];
                    dst_data[dst_idx + 1] = src_data[src_idx + 1];
                    dst_data[dst_idx + 2] = src_data[src_idx + 2];
                }
            }
        }
    }
    
    return rgb;
}

/**
 * @brief 转换为RGBA图像
 */
Image* png_convert_to_rgba(const Image *img)
{
    png_clear_error();
    
    if (!img || !img->data) {
        png_set_error("图像为空");
        return NULL;
    }
    
    // 如果已经是RGBA，直接复制
    if (img->channels == 4) {
        return image_clone(img);
    }
    
    // 创建RGBA图像
    Image *rgba = image_create(img->width, img->height, 4, img->bit_depth);
    if (!rgba) {
        return NULL;
    }
    
    if (img->bit_depth == 8) {
        // 8位处理
        uint8_t max_alpha = 255;
        
        for (int y = 0; y < img->height; y++) {
            for (int x = 0; x < img->width; x++) {
                size_t src_idx = (y * img->width + x) * img->channels;
                size_t dst_idx = (y * img->width + x) * 4;
                
                if (img->channels == 1) {
                    // 灰度转RGBA
                    uint8_t gray = img->data[src_idx];
                    rgba->data[dst_idx] = gray;
                    rgba->data[dst_idx + 1] = gray;
                    rgba->data[dst_idx + 2] = gray;
                    rgba->data[dst_idx + 3] = max_alpha;
                } else if (img->channels == 2) {
                    // 灰度+Alpha转RGBA
                    uint8_t gray = img->data[src_idx];
                    uint8_t alpha = img->data[src_idx + 1];
                    rgba->data[dst_idx] = gray;
                    rgba->data[dst_idx + 1] = gray;
                    rgba->data[dst_idx + 2] = gray;
                    rgba->data[dst_idx + 3] = alpha;
                } else if (img->channels == 3) {
                    // RGB转RGBA
                    rgba->data[dst_idx] = img->data[src_idx];
                    rgba->data[dst_idx + 1] = img->data[src_idx + 1];
                    rgba->data[dst_idx + 2] = img->data[src_idx + 2];
                    rgba->data[dst_idx + 3] = max_alpha;
                }
            }
        }
    } else if (img->bit_depth == 16) {
        // 16位处理
        uint16_t *src_data = (uint16_t*)img->data;
        uint16_t *dst_data = (uint16_t*)rgba->data;
        uint16_t max_alpha = 65535;
        
        for (int y = 0; y < img->height; y++) {
            for (int x = 0; x < img->width; x++) {
                size_t src_idx = (y * img->width + x) * img->channels;
                size_t dst_idx = (y * img->width + x) * 4;
                
                if (img->channels == 1) {
                    uint16_t gray = src_data[src_idx];
                    dst_data[dst_idx] = gray;
                    dst_data[dst_idx + 1] = gray;
                    dst_data[dst_idx + 2] = gray;
                    dst_data[dst_idx + 3] = max_alpha;
                } else if (img->channels == 2) {
                    uint16_t gray = src_data[src_idx];
                    uint16_t alpha = src_data[src_idx + 1];
                    dst_data[dst_idx] = gray;
                    dst_data[dst_idx + 1] = gray;
                    dst_data[dst_idx + 2] = gray;
                    dst_data[dst_idx + 3] = alpha;
                } else if (img->channels == 3) {
                    dst_data[dst_idx] = src_data[src_idx];
                    dst_data[dst_idx + 1] = src_data[src_idx + 1];
                    dst_data[dst_idx + 2] = src_data[src_idx + 2];
                    dst_data[dst_idx + 3] = max_alpha;
                }
            }
        }
    }
    
    return rgba;
}

/**
 * @brief 转换位深度
 */
Image* png_convert_bit_depth(const Image *img, int target_bit_depth)
{
    png_clear_error();
    
    if (!img || !img->data) {
        png_set_error("图像为空");
        return NULL;
    }
    
    if (target_bit_depth != 8 && target_bit_depth != 16) {
        png_set_error("不支持的目标位深度: %d", target_bit_depth);
        return NULL;
    }
    
    // 如果位深度相同，直接复制
    if (img->bit_depth == target_bit_depth) {
        return image_clone(img);
    }
    
    // 创建目标图像
    Image *result = image_create(img->width, img->height, 
                                img->channels, target_bit_depth);
    if (!result) {
        return NULL;
    }
    
    size_t pixel_count = img->width * img->height * img->channels;
    
    if (img->bit_depth == 8 && target_bit_depth == 16) {
        // 8位转16位
        uint16_t *dst_data = (uint16_t*)result->data;
        for (size_t i = 0; i < pixel_count; i++) {
            // 扩展到16位：value * 257 (等同于 value * 65535 / 255)
            dst_data[i] = img->data[i] * 257;
        }
    } else if (img->bit_depth == 16 && target_bit_depth == 8) {
        // 16位转8位
        uint16_t *src_data = (uint16_t*)img->data;
        for (size_t i = 0; i < pixel_count; i++) {
            // 缩减到8位：value / 257
            result->data[i] = src_data[i] / 257;
        }
    }
    
    return result;
}

/**
 * @brief 移除Alpha通道
 */
Image* png_remove_alpha(const Image *img, uint8_t bg_r, uint8_t bg_g, uint8_t bg_b)
{
    png_clear_error();
    
    if (!img || !img->data) {
        png_set_error("图像为空");
        return NULL;
    }
    
    // 如果没有Alpha通道，直接复制
    if (img->channels != 2 && img->channels != 4) {
        return image_clone(img);
    }
    
    // 确定目标通道数
    int target_channels = (img->channels == 4) ? 3 : 1;
    
    // 创建目标图像
    Image *result = image_create(img->width, img->height, 
                                target_channels, img->bit_depth);
    if (!result) {
        return NULL;
    }
    
    if (img->bit_depth == 8) {
        // 8位处理
        for (int y = 0; y < img->height; y++) {
            for (int x = 0; x < img->width; x++) {
                size_t src_idx = (y * img->width + x) * img->channels;
                size_t dst_idx = (y * img->width + x) * target_channels;
                
                if (img->channels == 2) {
                    // 灰度+Alpha
                    uint8_t gray = img->data[src_idx];
                    uint8_t alpha = img->data[src_idx + 1];
                    float a = alpha / 255.0f;
                    result->data[dst_idx] = (uint8_t)(gray * a + bg_r * (1 - a));
                } else if (img->channels == 4) {
                    // RGBA
                    uint8_t r = img->data[src_idx];
                    uint8_t g = img->data[src_idx + 1];
                    uint8_t b = img->data[src_idx + 2];
                    uint8_t alpha = img->data[src_idx + 3];
                    float a = alpha / 255.0f;
                    result->data[dst_idx] = (uint8_t)(r * a + bg_r * (1 - a));
                    result->data[dst_idx + 1] = (uint8_t)(g * a + bg_g * (1 - a));
                    result->data[dst_idx + 2] = (uint8_t)(b * a + bg_b * (1 - a));
                }
            }
        }
    } else if (img->bit_depth == 16) {
        // 16位处理
        uint16_t *src_data = (uint16_t*)img->data;
        uint16_t *dst_data = (uint16_t*)result->data;
        uint16_t bg_r16 = bg_r * 257;
        uint16_t bg_g16 = bg_g * 257;
        uint16_t bg_b16 = bg_b * 257;
        
        for (int y = 0; y < img->height; y++) {
            for (int x = 0; x < img->width; x++) {
                size_t src_idx = (y * img->width + x) * img->channels;
                size_t dst_idx = (y * img->width + x) * target_channels;
                
                if (img->channels == 2) {
                    uint16_t gray = src_data[src_idx];
                    uint16_t alpha = src_data[src_idx + 1];
                    float a = alpha / 65535.0f;
                    dst_data[dst_idx] = (uint16_t)(gray * a + bg_r16 * (1 - a));
                } else if (img->channels == 4) {
                    uint16_t r = src_data[src_idx];
                    uint16_t g = src_data[src_idx + 1];
                    uint16_t b = src_data[src_idx + 2];
                    uint16_t alpha = src_data[src_idx + 3];
                    float a = alpha / 65535.0f;
                    dst_data[dst_idx] = (uint16_t)(r * a + bg_r16 * (1 - a));
                    dst_data[dst_idx + 1] = (uint16_t)(g * a + bg_g16 * (1 - a));
                    dst_data[dst_idx + 2] = (uint16_t)(b * a + bg_b16 * (1 - a));
                }
            }
        }
    }
    
    return result;
}
// ============================================================================
// 图像处理操作
// ============================================================================

/**
 * @brief 调整图像大小（最近邻插值）
 */
Image* png_resize_nearest(const Image *img, int new_width, int new_height)
{
    png_clear_error();
    
    if (!img || !img->data) {
        png_set_error("图像为空");
        return NULL;
    }
    
    if (new_width <= 0 || new_height <= 0) {
        png_set_error("无效的目标尺寸: %dx%d", new_width, new_height);
        return NULL;
    }
    
    // 创建目标图像
    Image *result = image_create(new_width, new_height, 
                                img->channels, img->bit_depth);
    if (!result) {
        return NULL;
    }
    
    // 计算缩放比例
    float x_ratio = (float)img->width / new_width;
    float y_ratio = (float)img->height / new_height;
    
    if (img->bit_depth == 8) {
        // 8位处理
        for (int y = 0; y < new_height; y++) {
            for (int x = 0; x < new_width; x++) {
                int src_x = (int)(x * x_ratio);
                int src_y = (int)(y * y_ratio);
                
                // 确保不越界
                src_x = CLAMP(src_x, 0, img->width - 1);
                src_y = CLAMP(src_y, 0, img->height - 1);
                
                size_t src_idx = (src_y * img->width + src_x) * img->channels;
                size_t dst_idx = (y * new_width + x) * img->channels;
                
                for (int c = 0; c < img->channels; c++) {
                    result->data[dst_idx + c] = img->data[src_idx + c];
                }
            }
        }
    } else if (img->bit_depth == 16) {
        // 16位处理
        uint16_t *src_data = (uint16_t*)img->data;
        uint16_t *dst_data = (uint16_t*)result->data;
        
        for (int y = 0; y < new_height; y++) {
            for (int x = 0; x < new_width; x++) {
                int src_x = (int)(x * x_ratio);
                int src_y = (int)(y * y_ratio);
                
                src_x = CLAMP(src_x, 0, img->width - 1);
                src_y = CLAMP(src_y, 0, img->height - 1);
                
                size_t src_idx = (src_y * img->width + src_x) * img->channels;
                size_t dst_idx = (y * new_width + x) * img->channels;
                
                for (int c = 0; c < img->channels; c++) {
                    dst_data[dst_idx + c] = src_data[src_idx + c];
                }
            }
        }
    }
    
    return result;
}

/**
 * @brief 调整图像大小（双线性插值）
 */
Image* png_resize_bilinear(const Image *img, int new_width, int new_height)
{
    png_clear_error();
    
    if (!img || !img->data) {
        png_set_error("图像为空");
        return NULL;
    }
    
    if (new_width <= 0 || new_height <= 0) {
        png_set_error("无效的目标尺寸: %dx%d", new_width, new_height);
        return NULL;
    }
    
    // 创建目标图像
    Image *result = image_create(new_width, new_height, 
                                img->channels, img->bit_depth);
    if (!result) {
        return NULL;
    }
    
    // 计算缩放比例
    float x_ratio = (float)(img->width - 1) / new_width;
    float y_ratio = (float)(img->height - 1) / new_height;
    
    if (img->bit_depth == 8) {
        // 8位处理
        for (int y = 0; y < new_height; y++) {
            for (int x = 0; x < new_width; x++) {
                float src_x = x * x_ratio;
                float src_y = y * y_ratio;
                
                int x1 = (int)src_x;
                int y1 = (int)src_y;
                int x2 = MIN(x1 + 1, img->width - 1);
                int y2 = MIN(y1 + 1, img->height - 1);
                
                float dx = src_x - x1;
                float dy = src_y - y1;
                
                size_t dst_idx = (y * new_width + x) * img->channels;
                
                for (int c = 0; c < img->channels; c++) {
                    // 获取四个角的像素值
                    uint8_t p11 = img->data[(y1 * img->width + x1) * img->channels + c];
                    uint8_t p12 = img->data[(y1 * img->width + x2) * img->channels + c];
                    uint8_t p21 = img->data[(y2 * img->width + x1) * img->channels + c];
                    uint8_t p22 = img->data[(y2 * img->width + x2) * img->channels + c];
                    
                    // 双线性插值
                    float val = p11 * (1 - dx) * (1 - dy) +
                               p12 * dx * (1 - dy) +
                               p21 * (1 - dx) * dy +
                               p22 * dx * dy;
                    
                    result->data[dst_idx + c] = (uint8_t)CLAMP(val, 0, 255);
                }
            }
        }
    } else if (img->bit_depth == 16) {
        // 16位处理
        uint16_t *src_data = (uint16_t*)img->data;
        uint16_t *dst_data = (uint16_t*)result->data;
        
        for (int y = 0; y < new_height; y++) {
            for (int x = 0; x < new_width; x++) {
                float src_x = x * x_ratio;
                float src_y = y * y_ratio;
                
                int x1 = (int)src_x;
                int y1 = (int)src_y;
                int x2 = MIN(x1 + 1, img->width - 1);
                int y2 = MIN(y1 + 1, img->height - 1);
                
                float dx = src_x - x1;
                float dy = src_y - y1;
                
                size_t dst_idx = (y * new_width + x) * img->channels;
                
                for (int c = 0; c < img->channels; c++) {
                    uint16_t p11 = src_data[(y1 * img->width + x1) * img->channels + c];
                    uint16_t p12 = src_data[(y1 * img->width + x2) * img->channels + c];
                    uint16_t p21 = src_data[(y2 * img->width + x1) * img->channels + c];
                    uint16_t p22 = src_data[(y2 * img->width + x2) * img->channels + c];
                    
                    float val = p11 * (1 - dx) * (1 - dy) +
                               p12 * dx * (1 - dy) +
                               p21 * (1 - dx) * dy +
                               p22 * dx * dy;
                    
                    dst_data[dst_idx + c] = (uint16_t)CLAMP(val, 0, 65535);
                }
            }
        }
    }
    
    return result;
}

/**
 * @brief 裁剪图像
 */
Image* png_crop(const Image *img, int x, int y, int width, int height)
{
    png_clear_error();
    
    if (!img || !img->data) {
        png_set_error("图像为空");
        return NULL;
    }
    
    // 验证裁剪区域
    if (x < 0 || y < 0 || width <= 0 || height <= 0) {
        png_set_error("无效的裁剪区域");
        return NULL;
    }
    
    if (x + width > img->width || y + height > img->height) {
        png_set_error("裁剪区域超出图像边界");
        return NULL;
    }
    
    // 创建裁剪后的图像
    Image *result = image_create(width, height, img->channels, img->bit_depth);
    if (!result) {
        return NULL;
    }
    
    size_t bytes_per_pixel = img->channels * (img->bit_depth / 8);
    size_t src_row_stride = img->width * bytes_per_pixel;
    size_t dst_row_stride = width * bytes_per_pixel;
    
    // 复制裁剪区域
    for (int row = 0; row < height; row++) {
        size_t src_offset = ((y + row) * img->width + x) * bytes_per_pixel;
        size_t dst_offset = row * dst_row_stride;
        memcpy(result->data + dst_offset, 
               img->data + src_offset, 
               dst_row_stride);
    }
    
    return result;
}

/**
 * @brief 旋转图像90度
 */
Image* png_rotate_90(const Image *img, bool clockwise)
{
    png_clear_error();
    
    if (!img || !img->data) {
        png_set_error("图像为空");
        return NULL;
    }
    
    // 创建旋转后的图像（宽高互换）
    Image *result = image_create(img->height, img->width, 
                                img->channels, img->bit_depth);
    if (!result) {
        return NULL;
    }
    
    if (img->bit_depth == 8) {
        // 8位处理
        for (int y = 0; y < img->height; y++) {
            for (int x = 0; x < img->width; x++) {
                size_t src_idx = (y * img->width + x) * img->channels;
                size_t dst_idx;
                
                if (clockwise) {
                    // 顺时针90度：(x, y) -> (height-1-y, x)
                    dst_idx = ((img->height - 1 - y) + x * result->width) * img->channels;
                } else {
                    // 逆时针90度：(x, y) -> (y, width-1-x)
                    dst_idx = (y + (img->width - 1 - x) * result->width) * img->channels;
                }
                
                for (int c = 0; c < img->channels; c++) {
                    result->data[dst_idx + c] = img->data[src_idx + c];
                }
            }
        }
    } else if (img->bit_depth == 16) {
        // 16位处理
        uint16_t *src_data = (uint16_t*)img->data;
        uint16_t *dst_data = (uint16_t*)result->data;
        
        for (int y = 0; y < img->height; y++) {
            for (int x = 0; x < img->width; x++) {
                size_t src_idx = (y * img->width + x) * img->channels;
                size_t dst_idx;
                
                if (clockwise) {
                    dst_idx = ((img->height - 1 - y) + x * result->width) * img->channels;
                } else {
                    dst_idx = (y + (img->width - 1 - x) * result->width) * img->channels;
                }
                
                for (int c = 0; c < img->channels; c++) {
                    dst_data[dst_idx + c] = src_data[src_idx + c];
                }
            }
        }
    }
    
    return result;
}

/**
 * @brief 旋转图像180度
 */
Image* png_rotate_180(const Image *img)
{
    png_clear_error();
    
    if (!img || !img->data) {
        png_set_error("图像为空");
        return NULL;
    }
    
    // 创建旋转后的图像
    Image *result = image_create(img->width, img->height, 
                                img->channels, img->bit_depth);
    if (!result) {
        return NULL;
    }
    
    if (img->bit_depth == 8) {
        // 8位处理
        for (int y = 0; y < img->height; y++) {
            for (int x = 0; x < img->width; x++) {
                size_t src_idx = (y * img->width + x) * img->channels;
                // 180度：(x, y) -> (width-1-x, height-1-y)
                size_t dst_idx = ((img->height - 1 - y) * img->width + 
                                 (img->width - 1 - x)) * img->channels;
                
                for (int c = 0; c < img->channels; c++) {
                    result->data[dst_idx + c] = img->data[src_idx + c];
                }
            }
        }
    } else if (img->bit_depth == 16) {
        // 16位处理
        uint16_t *src_data = (uint16_t*)img->data;
        uint16_t *dst_data = (uint16_t*)result->data;
        
        for (int y = 0; y < img->height; y++) {
            for (int x = 0; x < img->width; x++) {
                size_t src_idx = (y * img->width + x) * img->channels;
                size_t dst_idx = ((img->height - 1 - y) * img->width + 
                                 (img->width - 1 - x)) * img->channels;
                
                for (int c = 0; c < img->channels; c++) {
                    dst_data[dst_idx + c] = src_data[src_idx + c];
                }
            }
        }
    }
    
    return result;
}

/**
 * @brief 水平翻转图像
 */
Image* png_flip_horizontal(const Image *img)
{
    png_clear_error();
    
    if (!img || !img->data) {
        png_set_error("图像为空");
        return NULL;
    }
    
    // 创建翻转后的图像
    Image *result = image_create(img->width, img->height, 
                                img->channels, img->bit_depth);
    if (!result) {
        return NULL;
    }
    
    if (img->bit_depth == 8) {
        // 8位处理
        for (int y = 0; y < img->height; y++) {
            for (int x = 0; x < img->width; x++) {
                size_t src_idx = (y * img->width + x) * img->channels;
                size_t dst_idx = (y * img->width + (img->width - 1 - x)) * img->channels;
                
                for (int c = 0; c < img->channels; c++) {
                    result->data[dst_idx + c] = img->data[src_idx + c];
                }
            }
        }
    } else if (img->bit_depth == 16) {
        // 16位处理
        uint16_t *src_data = (uint16_t*)img->data;
        uint16_t *dst_data = (uint16_t*)result->data;
        
        for (int y = 0; y < img->height; y++) {
            for (int x = 0; x < img->width; x++) {
                size_t src_idx = (y * img->width + x) * img->channels;
                size_t dst_idx = (y * img->width + (img->width - 1 - x)) * img->channels;
                
                for (int c = 0; c < img->channels; c++) {
                    dst_data[dst_idx + c] = src_data[src_idx + c];
                }
            }
        }
    }
    
    return result;
}

/**
 * @brief 垂直翻转图像
 */
Image* png_flip_vertical(const Image *img)
{
    png_clear_error();
    
    if (!img || !img->data) {
        png_set_error("图像为空");
        return NULL;
    }
    
    // 创建翻转后的图像
    Image *result = image_create(img->width, img->height, 
                                img->channels, img->bit_depth);
    if (!result) {
        return NULL;
    }
    
    size_t row_stride = row_bytes(img->width, img->channels, img->bit_depth);
    
    // 逐行复制（倒序）
    for (int y = 0; y < img->height; y++) {
        size_t src_offset = y * row_stride;
        size_t dst_offset = (img->height - 1 - y) * row_stride;
        memcpy(result->data + dst_offset, 
               img->data + src_offset, 
               row_stride);
    }
    
    return result;
}
// ============================================================================
// 图像滤镜和效果
// ============================================================================

/**
 * @brief 调整亮度
 */
Image* png_adjust_brightness(const Image *img, float factor)
{
    png_clear_error();
    
    if (!img || !img->data) {
        png_set_error("图像为空");
        return NULL;
    }
    
    if (factor < -1.0f || factor > 1.0f) {
        png_set_error("亮度因子必须在-1.0到1.0之间");
        return NULL;
    }
    
    // 创建结果图像
    Image *result = image_clone(img);
    if (!result) {
        return NULL;
    }
    
    if (img->bit_depth == 8) {
        // 8位处理
        int adjustment = (int)(factor * 255);
        
        for (int y = 0; y < img->height; y++) {
            for (int x = 0; x < img->width; x++) {
                size_t idx = (y * img->width + x) * img->channels;
                
                // 调整每个颜色通道（跳过Alpha通道）
                int channels_to_adjust = (img->channels == 4 || img->channels == 2) ? 
                                        img->channels - 1 : img->channels;
                
                for (int c = 0; c < channels_to_adjust; c++) {
                    int val = img->data[idx + c] + adjustment;
                    result->data[idx + c] = (uint8_t)CLAMP(val, 0, 255);
                }
            }
        }
    } else if (img->bit_depth == 16) {
        // 16位处理
        uint16_t *src_data = (uint16_t*)img->data;
        uint16_t *dst_data = (uint16_t*)result->data;
        int adjustment = (int)(factor * 65535);
        
        for (int y = 0; y < img->height; y++) {
            for (int x = 0; x < img->width; x++) {
                size_t idx = (y * img->width + x) * img->channels;
                
                int channels_to_adjust = (img->channels == 4 || img->channels == 2) ? 
                                        img->channels - 1 : img->channels;
                
                for (int c = 0; c < channels_to_adjust; c++) {
                    int val = src_data[idx + c] + adjustment;
                    dst_data[idx + c] = (uint16_t)CLAMP(val, 0, 65535);
                }
            }
        }
    }
    
    return result;
}

/**
 * @brief 调整对比度
 */
Image* png_adjust_contrast(const Image *img, float factor)
{
    png_clear_error();
    
    if (!img || !img->data) {
        png_set_error("图像为空");
        return NULL;
    }
    
    if (factor < 0.0f) {
        png_set_error("对比度因子必须大于等于0");
        return NULL;
    }
    
    // 创建结果图像
    Image *result = image_clone(img);
    if (!result) {
        return NULL;
    }
    
    if (img->bit_depth == 8) {
        // 8位处理
        float contrast = (259.0f * (factor * 255.0f + 255.0f)) / 
                        (255.0f * (259.0f - factor * 255.0f));
        
        for (int y = 0; y < img->height; y++) {
            for (int x = 0; x < img->width; x++) {
                size_t idx = (y * img->width + x) * img->channels;
                
                int channels_to_adjust = (img->channels == 4 || img->channels == 2) ? 
                                        img->channels - 1 : img->channels;
                
                for (int c = 0; c < channels_to_adjust; c++) {
                    float val = contrast * (img->data[idx + c] - 128.0f) + 128.0f;
                    result->data[idx + c] = (uint8_t)CLAMP(val, 0, 255);
                }
            }
        }
    } else if (img->bit_depth == 16) {
        // 16位处理
        uint16_t *src_data = (uint16_t*)img->data;
        uint16_t *dst_data = (uint16_t*)result->data;
        float contrast = (259.0f * (factor * 65535.0f + 65535.0f)) / 
                        (65535.0f * (259.0f - factor * 65535.0f));
        
        for (int y = 0; y < img->height; y++) {
            for (int x = 0; x < img->width; x++) {
                size_t idx = (y * img->width + x) * img->channels;
                
                int channels_to_adjust = (img->channels == 4 || img->channels == 2) ? 
                                        img->channels - 1 : img->channels;
                
                for (int c = 0; c < channels_to_adjust; c++) {
                    float val = contrast * (src_data[idx + c] - 32768.0f) + 32768.0f;
                    dst_data[idx + c] = (uint16_t)CLAMP(val, 0, 65535);
                }
            }
        }
    }
    
    return result;
}

/**
 * @brief 反转颜色
 */
Image* png_invert(const Image *img)
{
    png_clear_error();
    
    if (!img || !img->data) {
        png_set_error("图像为空");
        return NULL;
    }
    
    // 创建结果图像
    Image *result = image_clone(img);
    if (!result) {
        return NULL;
    }
    
    if (img->bit_depth == 8) {
        // 8位处理
        for (int y = 0; y < img->height; y++) {
            for (int x = 0; x < img->width; x++) {
                size_t idx = (y * img->width + x) * img->channels;
                
                // 反转每个颜色通道（跳过Alpha通道）
                int channels_to_invert = (img->channels == 4 || img->channels == 2) ? 
                                        img->channels - 1 : img->channels;
                
                for (int c = 0; c < channels_to_invert; c++) {
                    result->data[idx + c] = 255 - img->data[idx + c];
                }
            }
        }
    } else if (img->bit_depth == 16) {
        // 16位处理
        uint16_t *src_data = (uint16_t*)img->data;
        uint16_t *dst_data = (uint16_t*)result->data;
        
        for (int y = 0; y < img->height; y++) {
            for (int x = 0; x < img->width; x++) {
                size_t idx = (y * img->width + x) * img->channels;
                
                int channels_to_invert = (img->channels == 4 || img->channels == 2) ? 
                                        img->channels - 1 : img->channels;
                
                for (int c = 0; c < channels_to_invert; c++) {
                    dst_data[idx + c] = 65535 - src_data[idx + c];
                }
            }
        }
    }
    
    return result;
}

/**
 * @brief 应用阈值（二值化）
 */
Image* png_threshold(const Image *img, int threshold)
{
    png_clear_error();
    
    if (!img || !img->data) {
        png_set_error("图像为空");
        return NULL;
    }
    
    if (threshold < 0 || threshold > 255) {
        png_set_error("阈值必须在0-255之间");
        return NULL;
    }
    
    // 先转换为灰度图
    Image *gray = png_convert_to_grayscale(img);
    if (!gray) {
        return NULL;
    }
    
    if (gray->bit_depth == 8) {
        // 8位处理
        for (size_t i = 0; i < gray->width * gray->height; i++) {
            gray->data[i] = (gray->data[i] >= threshold) ? 255 : 0;
        }
    } else if (gray->bit_depth == 16) {
        // 16位处理
        uint16_t *data = (uint16_t*)gray->data;
        uint16_t threshold16 = threshold * 257;
        
        for (size_t i = 0; i < gray->width * gray->height; i++) {
            data[i] = (data[i] >= threshold16) ? 65535 : 0;
        }
    }
    
    return gray;
}

/**
 * @brief 应用模糊滤镜（简单平均）
 */
Image* png_blur(const Image *img, int radius)
{
    png_clear_error();
    
    if (!img || !img->data) {
        png_set_error("图像为空");
        return NULL;
    }
    
    if (radius < 1) {
        png_set_error("模糊半径必须大于0");
        return NULL;
    }
    
    // 创建结果图像
    Image *result = image_create(img->width, img->height, 
                                img->channels, img->bit_depth);
    if (!result) {
        return NULL;
    }
    
    int kernel_size = radius * 2 + 1;
    float weight = 1.0f / (kernel_size * kernel_size);
    
    if (img->bit_depth == 8) {
        // 8位处理
        for (int y = 0; y < img->height; y++) {
            for (int x = 0; x < img->width; x++) {
                size_t dst_idx = (y * img->width + x) * img->channels;
                
                // 对每个通道应用模糊
                for (int c = 0; c < img->channels; c++) {
                    float sum = 0.0f;
                    
                    // 遍历卷积核
                    for (int ky = -radius; ky <= radius; ky++) {
                        for (int kx = -radius; kx <= radius; kx++) {
                            int px = CLAMP(x + kx, 0, img->width - 1);
                            int py = CLAMP(y + ky, 0, img->height - 1);
                            size_t src_idx = (py * img->width + px) * img->channels + c;
                            sum += img->data[src_idx];
                        }
                    }
                    
                    result->data[dst_idx + c] = (uint8_t)(sum * weight);
                }
            }
        }
    } else if (img->bit_depth == 16) {
        // 16位处理
        uint16_t *src_data = (uint16_t*)img->data;
        uint16_t *dst_data = (uint16_t*)result->data;
        
        for (int y = 0; y < img->height; y++) {
            for (int x = 0; x < img->width; x++) {
                size_t dst_idx = (y * img->width + x) * img->channels;
                
                for (int c = 0; c < img->channels; c++) {
                    float sum = 0.0f;
                    
                    for (int ky = -radius; ky <= radius; ky++) {
                        for (int kx = -radius; kx <= radius; kx++) {
                            int px = CLAMP(x + kx, 0, img->width - 1);
                            int py = CLAMP(y + ky, 0, img->height - 1);
                            size_t src_idx = (py * img->width + px) * img->channels + c;
                            sum += src_data[src_idx];
                        }
                    }
                    
                    dst_data[dst_idx + c] = (uint16_t)(sum * weight);
                }
            }
        }
    }
    
    return result;
}

/**
 * @brief 应用锐化滤镜
 */
Image* png_sharpen(const Image *img)
{
    png_clear_error();
    
    if (!img || !img->data) {
        png_set_error("图像为空");
        return NULL;
    }
    
    // 创建结果图像
    Image *result = image_create(img->width, img->height, 
                                img->channels, img->bit_depth);
    if (!result) {
        return NULL;
    }
    
    // 锐化卷积核
    // [ 0 -1  0]
    // [-1  5 -1]
    // [ 0 -1  0]
    int kernel[3][3] = {
        { 0, -1,  0},
        {-1,  5, -1},
        { 0, -1,  0}
    };
    
    if (img->bit_depth == 8) {
        // 8位处理
        for (int y = 0; y < img->height; y++) {
            for (int x = 0; x < img->width; x++) {
                size_t dst_idx = (y * img->width + x) * img->channels;
                
                // 对每个通道应用锐化
                for (int c = 0; c < img->channels; c++) {
                    int sum = 0;
                    
                    // 应用卷积核
                    for (int ky = -1; ky <= 1; ky++) {
                        for (int kx = -1; kx <= 1; kx++) {
                            int px = CLAMP(x + kx, 0, img->width - 1);
                            int py = CLAMP(y + ky, 0, img->height - 1);
                            size_t src_idx = (py * img->width + px) * img->channels + c;
                            sum += img->data[src_idx] * kernel[ky + 1][kx + 1];
                        }
                    }
                    
                    result->data[dst_idx + c] = (uint8_t)CLAMP(sum, 0, 255);
                }
            }
        }
    } else if (img->bit_depth == 16) {
        // 16位处理
        uint16_t *src_data = (uint16_t*)img->data;
        uint16_t *dst_data = (uint16_t*)result->data;
        
        for (int y = 0; y < img->height; y++) {
            for (int x = 0; x < img->width; x++) {
                size_t dst_idx = (y * img->width + x) * img->channels;
                
                for (int c = 0; c < img->channels; c++) {
                    int sum = 0;
                    
                    for (int ky = -1; ky <= 1; ky++) {
                        for (int kx = -1; kx <= 1; kx++) {
                            int px = CLAMP(x + kx, 0, img->width - 1);
                            int py = CLAMP(y + ky, 0, img->height - 1);
                            size_t src_idx = (py * img->width + px) * img->channels + c;
                            sum += src_data[src_idx] * kernel[ky + 1][kx + 1];
                        }
                    }
                    
                    dst_data[dst_idx + c] = (uint16_t)CLAMP(sum, 0, 65535);
                }
            }
        }
    }
    
    return result;
}

/**
 * @brief 应用边缘检测（Sobel算子）
 */
Image* png_edge_detect(const Image *img)
{
    png_clear_error();
    
    if (!img || !img->data) {
        png_set_error("图像为空");
        return NULL;
    }
    
    // 先转换为灰度图
    Image *gray = png_convert_to_grayscale(img);
    if (!gray) {
        return NULL;
    }
    
    // 创建结果图像
    Image *result = image_create(gray->width, gray->height, 1, gray->bit_depth);
    if (!result) {
        image_destroy(gray);
        return NULL;
    }
    
    // Sobel算子
    int sobel_x[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };
    
    int sobel_y[3][3] = {
        {-1, -2, -1},
        { 0,  0,  0},
        { 1,  2,  1}
    };
    
    if (gray->bit_depth == 8) {
        // 8位处理
        for (int y = 0; y < gray->height; y++) {
            for (int x = 0; x < gray->width; x++) {
                int gx = 0, gy = 0;
                
                // 应用Sobel算子
                for (int ky = -1; ky <= 1; ky++) {
                    for (int kx = -1; kx <= 1; kx++) {
                        int px = CLAMP(x + kx, 0, gray->width - 1);
                        int py = CLAMP(y + ky, 0, gray->height - 1);
                        size_t idx = py * gray->width + px;
                        
                        gx += gray->data[idx] * sobel_x[ky + 1][kx + 1];
                        gy += gray->data[idx] * sobel_y[ky + 1][kx + 1];
                    }
                }
                
                // 计算梯度幅值
                int magnitude = (int)sqrt(gx * gx + gy * gy);
                size_t dst_idx = y * gray->width + x;
                result->data[dst_idx] = (uint8_t)CLAMP(magnitude, 0, 255);
            }
        }
    } else if (gray->bit_depth == 16) {
        // 16位处理
        uint16_t *src_data = (uint16_t*)gray->data;
        uint16_t *dst_data = (uint16_t*)result->data;
        
        for (int y = 0; y < gray->height; y++) {
            for (int x = 0; x < gray->width; x++) {
                int gx = 0, gy = 0;
                
                for (int ky = -1; ky <= 1; ky++) {
                    for (int kx = -1; kx <= 1; kx++) {
                        int px = CLAMP(x + kx, 0, gray->width - 1);
                        int py = CLAMP(y + ky, 0, gray->height - 1);
                        size_t idx = py * gray->width + px;
                        
                        gx += src_data[idx] * sobel_x[ky + 1][kx + 1];
                        gy += src_data[idx] * sobel_y[ky + 1][kx + 1];
                    }
                }
                
                int magnitude = (int)sqrt(gx * gx + gy * gy);
                size_t dst_idx = y * gray->width + x;
                dst_data[dst_idx] = (uint16_t)CLAMP(magnitude, 0, 65535);
            }
        }
    }
    
    image_destroy(gray);
    return result;
}
// ============================================================================
// 图像合成
// ============================================================================

/**
 * @brief Alpha混合两个图像
 */
Image* png_alpha_blend(const Image *bottom, const Image *top, 
                       int offset_x, int offset_y)
{
    png_clear_error();
    
    if (!bottom || !bottom->data || !top || !top->data) {
        png_set_error("图像为空");
        return NULL;
    }
    
    if (bottom->bit_depth != top->bit_depth) {
        png_set_error("两个图像的位深度必须相同");
        return NULL;
    }
    
    // 创建结果图像（复制底层图像）
    Image *result = image_clone(bottom);
    if (!result) {
        return NULL;
    }
    
    // 确保顶层图像有Alpha通道
    Image *top_rgba = NULL;
    if (top->channels != 4 && top->channels != 2) {
        top_rgba = png_convert_to_rgba(top);
        if (!top_rgba) {
            image_destroy(result);
            return NULL;
        }
    } else {
        top_rgba = (Image*)top;
    }
    
    // 确保底层图像有Alpha通道
    Image *bottom_rgba = NULL;
    if (bottom->channels != 4 && bottom->channels != 2) {
        bottom_rgba = png_convert_to_rgba(bottom);
        if (!bottom_rgba) {
            if (top_rgba != top) image_destroy(top_rgba);
            image_destroy(result);
            return NULL;
        }
        image_destroy(result);
        result = bottom_rgba;
    }
    
    if (result->bit_depth == 8) {
        // 8位处理
        for (int y = 0; y < top_rgba->height; y++) {
            int dst_y = y + offset_y;
            if (dst_y < 0 || dst_y >= result->height) continue;
            
            for (int x = 0; x < top_rgba->width; x++) {
                int dst_x = x + offset_x;
                if (dst_x < 0 || dst_x >= result->width) continue;
                
                size_t top_idx = (y * top_rgba->width + x) * top_rgba->channels;
                size_t dst_idx = (dst_y * result->width + dst_x) * result->channels;
                
                if (top_rgba->channels == 4) {
                    // RGBA混合
                    float alpha_top = top_rgba->data[top_idx + 3] / 255.0f;
                    float alpha_bottom = result->data[dst_idx + 3] / 255.0f;
                    float alpha_out = alpha_top + alpha_bottom * (1.0f - alpha_top);
                    
                    if (alpha_out > 0) {
                        for (int c = 0; c < 3; c++) {
                            float color_top = top_rgba->data[top_idx + c] * alpha_top;
                            float color_bottom = result->data[dst_idx + c] * 
                                               alpha_bottom * (1.0f - alpha_top);
                            result->data[dst_idx + c] = 
                                (uint8_t)((color_top + color_bottom) / alpha_out);
                        }
                        result->data[dst_idx + 3] = (uint8_t)(alpha_out * 255);
                    }
                } else if (top_rgba->channels == 2) {
                    // 灰度+Alpha混合
                    float alpha_top = top_rgba->data[top_idx + 1] / 255.0f;
                    float alpha_bottom = result->data[dst_idx + 1] / 255.0f;
                    float alpha_out = alpha_top + alpha_bottom * (1.0f - alpha_top);
                    
                    if (alpha_out > 0) {
                        float color_top = top_rgba->data[top_idx] * alpha_top;
                        float color_bottom = result->data[dst_idx] * 
                                           alpha_bottom * (1.0f - alpha_top);
                        result->data[dst_idx] = 
                            (uint8_t)((color_top + color_bottom) / alpha_out);
                        result->data[dst_idx + 1] = (uint8_t)(alpha_out * 255);
                    }
                }
            }
        }
    } else if (result->bit_depth == 16) {
        // 16位处理
        uint16_t *top_data = (uint16_t*)top_rgba->data;
        uint16_t *dst_data = (uint16_t*)result->data;
        
        for (int y = 0; y < top_rgba->height; y++) {
            int dst_y = y + offset_y;
            if (dst_y < 0 || dst_y >= result->height) continue;
            
            for (int x = 0; x < top_rgba->width; x++) {
                int dst_x = x + offset_x;
                if (dst_x < 0 || dst_x >= result->width) continue;
                
                size_t top_idx = (y * top_rgba->width + x) * top_rgba->channels;
                size_t dst_idx = (dst_y * result->width + dst_x) * result->channels;
                
                if (top_rgba->channels == 4) {
                    float alpha_top = top_data[top_idx + 3] / 65535.0f;
                    float alpha_bottom = dst_data[dst_idx + 3] / 65535.0f;
                    float alpha_out = alpha_top + alpha_bottom * (1.0f - alpha_top);
                    
                    if (alpha_out > 0) {
                        for (int c = 0; c < 3; c++) {
                            float color_top = top_data[top_idx + c] * alpha_top;
                            float color_bottom = dst_data[dst_idx + c] * 
                                               alpha_bottom * (1.0f - alpha_top);
                            dst_data[dst_idx + c] = 
                                (uint16_t)((color_top + color_bottom) / alpha_out);
                        }
                        dst_data[dst_idx + 3] = (uint16_t)(alpha_out * 65535);
                    }
                } else if (top_rgba->channels == 2) {
                    float alpha_top = top_data[top_idx + 1] / 65535.0f;
                    float alpha_bottom = dst_data[dst_idx + 1] / 65535.0f;
                    float alpha_out = alpha_top + alpha_bottom * (1.0f - alpha_top);
                    
                    if (alpha_out > 0) {
                        float color_top = top_data[top_idx] * alpha_top;
                        float color_bottom = dst_data[dst_idx] * 
                                           alpha_bottom * (1.0f - alpha_top);
                        dst_data[dst_idx] = 
                            (uint16_t)((color_top + color_bottom) / alpha_out);
                        dst_data[dst_idx + 1] = (uint16_t)(alpha_out * 65535);
                    }
                }
            }
        }
    }
    
    // 清理临时图像
    if (top_rgba != top) image_destroy(top_rgba);
    
    return result;
}

/**
 * @brief 水平拼接图像
 */
Image* png_concat_horizontal(const Image *left, const Image *right)
{
    png_clear_error();
    
    if (!left || !left->data || !right || !right->data) {
        png_set_error("图像为空");
        return NULL;
    }
    
    if (left->height != right->height) {
        png_set_error("两个图像的高度必须相同");
        return NULL;
    }
    
    if (left->channels != right->channels || left->bit_depth != right->bit_depth) {
        png_set_error("两个图像的通道数和位深度必须相同");
        return NULL;
    }
    
    // 创建拼接后的图像
    int new_width = left->width + right->width;
    Image *result = image_create(new_width, left->height, 
                                left->channels, left->bit_depth);
    if (!result) {
        return NULL;
    }
    
    size_t bytes_per_pixel = left->channels * (left->bit_depth / 8);
    size_t left_row_bytes = left->width * bytes_per_pixel;
    size_t right_row_bytes = right->width * bytes_per_pixel;
    
    // 逐行复制
    for (int y = 0; y < left->height; y++) {
        size_t dst_offset = y * new_width * bytes_per_pixel;
        size_t left_offset = y * left->width * bytes_per_pixel;
        size_t right_offset = y * right->width * bytes_per_pixel;
        
        // 复制左图
        memcpy(result->data + dst_offset, 
               left->data + left_offset, 
               left_row_bytes);
        
        // 复制右图
        memcpy(result->data + dst_offset + left_row_bytes, 
               right->data + right_offset, 
               right_row_bytes);
    }
    
    return result;
}

/**
 * @brief 垂直拼接图像
 */
Image* png_concat_vertical(const Image *top, const Image *bottom)
{
    png_clear_error();
    
    if (!top || !top->data || !bottom || !bottom->data) {
        png_set_error("图像为空");
        return NULL;
    }
    
    if (top->width != bottom->width) {
        png_set_error("两个图像的宽度必须相同");
        return NULL;
    }
    
    if (top->channels != bottom->channels || top->bit_depth != bottom->bit_depth) {
        png_set_error("两个图像的通道数和位深度必须相同");
        return NULL;
    }
    
    // 创建拼接后的图像
    int new_height = top->height + bottom->height;
    Image *result = image_create(top->width, new_height, 
                                top->channels, top->bit_depth);
    if (!result) {
        return NULL;
    }
    
    size_t top_size = image_data_size(top);
    size_t bottom_size = image_data_size(bottom);
    
    // 复制顶部图像
    memcpy(result->data, top->data, top_size);
    
    // 复制底部图像
    memcpy(result->data + top_size, bottom->data, bottom_size);
    
    return result;
}

/**
 * @brief 创建图像网格
 */
Image* png_create_grid(Image **images, int rows, int cols)
{
    png_clear_error();
    
    if (!images || rows <= 0 || cols <= 0) {
        png_set_error("无效的参数");
        return NULL;
    }
    
    // 检查所有图像
    int width = 0, height = 0;
    int channels = 0, bit_depth = 0;
    
    for (int i = 0; i < rows * cols; i++) {
        if (!images[i] || !images[i]->data) {
            png_set_error("图像数组中包含空图像");
            return NULL;
        }
        
        if (i == 0) {
            width = images[i]->width;
            height = images[i]->height;
            channels = images[i]->channels;
            bit_depth = images[i]->bit_depth;
        } else {
            if (images[i]->width != width || images[i]->height != height ||
                images[i]->channels != channels || images[i]->bit_depth != bit_depth) {
                png_set_error("所有图像必须具有相同的尺寸、通道数和位深度");
                return NULL;
            }
        }
    }
    
    // 创建网格图像
    int grid_width = width * cols;
    int grid_height = height * rows;
    Image *result = image_create(grid_width, grid_height, channels, bit_depth);
    if (!result) {
        return NULL;
    }
    
    size_t bytes_per_pixel = channels * (bit_depth / 8);
    size_t row_bytes = width * bytes_per_pixel;
    
    // 填充网格
    for (int row = 0; row < rows; row++) {
        for (int col = 0; col < cols; col++) {
            Image *img = images[row * cols + col];
            
            // 复制每一行
            for (int y = 0; y < height; y++) {
                size_t src_offset = y * width * bytes_per_pixel;
                size_t dst_offset = ((row * height + y) * grid_width + 
                                    col * width) * bytes_per_pixel;
                
                memcpy(result->data + dst_offset, 
                       img->data + src_offset, 
                       row_bytes);
            }
        }
    }
    
    return result;
}

// ============================================================================
// 批处理操作
// ============================================================================

/**
 * @brief 批处理回调函数类型
 */
typedef Image* (*BatchProcessFunc)(const Image *img, void *user_data);

/**
 * @brief 批处理图像
 */
bool png_batch_process(const char **input_files, int count,
                      const char *output_dir,
                      BatchProcessFunc process_func,
                      void *user_data)
{
    png_clear_error();
    
    if (!input_files || count <= 0 || !output_dir || !process_func) {
        png_set_error("无效的参数");
        return false;
    }
    
    bool all_success = true;
    
    for (int i = 0; i < count; i++) {
        // 读取图像
        Image *img = png_read(input_files[i]);
        if (!img) {
            fprintf(stderr, "无法读取图像: %s\n", input_files[i]);
            all_success = false;
            continue;
        }
        
        // 处理图像
        Image *processed = process_func(img, user_data);
        image_destroy(img);
        
        if (!processed) {
            fprintf(stderr, "处理图像失败: %s\n", input_files[i]);
            all_success = false;
            continue;
        }
        
        // 构建输出文件名
        const char *filename = strrchr(input_files[i], '/');
        if (!filename) filename = strrchr(input_files[i], '\\');
        if (!filename) filename = input_files[i];
        else filename++;
        
        char output_path[512];
        snprintf(output_path, sizeof(output_path), "%s/%s", output_dir, filename);
        
        // 写入图像
        if (!png_write(output_path, processed, 6)) {
            fprintf(stderr, "无法写入图像: %s\n", output_path);
            all_success = false;
        }
        
        image_destroy(processed);
    }
    
    return all_success;
}

/**
 * @brief 批量调整大小
 */
bool png_batch_resize(const char **input_files, int count,
                     const char *output_dir,
                     int new_width, int new_height,
                     bool use_bilinear)
{
    typedef struct {
        int width;
        int height;
        bool bilinear;
    } ResizeData;
    
    ResizeData data = {new_width, new_height, use_bilinear};
    
    BatchProcessFunc resize_func = NULL;
    if (use_bilinear) {
        resize_func = (BatchProcessFunc)png_resize_bilinear;
    } else {
        resize_func = (BatchProcessFunc)png_resize_nearest;
    }
    
    // 包装函数以传递参数
    Image* resize_wrapper(const Image *img, void *user_data) {
        ResizeData *rd = (ResizeData*)user_data;
        if (rd->bilinear) {
            return png_resize_bilinear(img, rd->width, rd->height);
        } else {
            return png_resize_nearest(img, rd->width, rd->height);
        }
    }
    
    return png_batch_process(input_files, count, output_dir, 
                           resize_wrapper, &data);
}

/**
 * @brief 批量转换格式
 */
bool png_batch_convert(const char **input_files, int count,
                      const char *output_dir,
                      int target_channels)
{
    typedef struct {
        int channels;
    } ConvertData;
    
    ConvertData data = {target_channels};
    
    Image* convert_wrapper(const Image *img, void *user_data) {
        ConvertData *cd = (ConvertData*)user_data;
        
        switch (cd->channels) {
            case 1:
                return png_convert_to_grayscale(img);
            case 3:
                return png_convert_to_rgb(img);
            case 4:
                return png_convert_to_rgba(img);
            default:
                png_set_error("不支持的目标通道数: %d", cd->channels);
                return NULL;
        }
    }
    
    return png_batch_process(input_files, count, output_dir, 
                           convert_wrapper, &data);
}

/**
 * @brief 批量应用滤镜
 */
bool png_batch_filter(const char **input_files, int count,
                     const char *output_dir,
                     const char *filter_name)
{
    Image* filter_wrapper(const Image *img, void *user_data) {
        const char *filter = (const char*)user_data;
        
        if (strcmp(filter, "grayscale") == 0) {
            return png_convert_to_grayscale(img);
        } else if (strcmp(filter, "invert") == 0) {
            return png_invert(img);
        } else if (strcmp(filter, "blur") == 0) {
            return png_blur(img, 3);
        } else if (strcmp(filter, "sharpen") == 0) {
            return png_sharpen(img);
        } else if (strcmp(filter, "edge") == 0) {
            return png_edge_detect(img);
        } else {
            png_set_error("未知的滤镜: %s", filter);
            return NULL;
        }
    }
    
    return png_batch_process(input_files, count, output_dir, 
                           filter_wrapper, (void*)filter_name);
}
// ============================================================================
// 图像信息和统计
// ============================================================================

/**
 * @brief 获取图像信息
 */
void png_get_info(const Image *img, ImageInfo *info)
{
    if (!img || !info) {
        return;
    }
    
    info->width = img->width;
    info->height = img->height;
    info->channels = img->channels;
    info->bit_depth = img->bit_depth;
    info->data_size = image_data_size(img);
    
    // 确定颜色类型
    if (img->channels == 1) {
        info->color_type = PNG_COLOR_TYPE_GRAY;
    } else if (img->channels == 2) {
        info->color_type = PNG_COLOR_TYPE_GRAY_ALPHA;
    } else if (img->channels == 3) {
        info->color_type = PNG_COLOR_TYPE_RGB;
    } else if (img->channels == 4) {
        info->color_type = PNG_COLOR_TYPE_RGB_ALPHA;
    }
}

/**
 * @brief 打印图像信息
 */
void png_print_info(const Image *img)
{
    if (!img) {
        printf("图像为空\n");
        return;
    }
    
    printf("图像信息:\n");
    printf("  尺寸: %d x %d\n", img->width, img->height);
    printf("  通道数: %d\n", img->channels);
    printf("  位深度: %d\n", img->bit_depth);
    
    const char *color_type_str = "未知";
    if (img->channels == 1) {
        color_type_str = "灰度";
    } else if (img->channels == 2) {
        color_type_str = "灰度+Alpha";
    } else if (img->channels == 3) {
        color_type_str = "RGB";
    } else if (img->channels == 4) {
        color_type_str = "RGBA";
    }
    printf("  颜色类型: %s\n", color_type_str);
    
    size_t data_size = image_data_size(img);
    printf("  数据大小: %zu 字节 (%.2f MB)\n", 
           data_size, data_size / (1024.0 * 1024.0));
}

/**
 * @brief 计算图像直方图
 */
bool png_calculate_histogram(const Image *img, int channel, 
                             unsigned int *histogram, int bins)
{
    png_clear_error();
    
    if (!img || !img->data || !histogram) {
        png_set_error("无效的参数");
        return false;
    }
    
    if (channel < 0 || channel >= img->channels) {
        png_set_error("无效的通道索引: %d", channel);
        return false;
    }
    
    if (bins <= 0 || bins > 65536) {
        png_set_error("无效的bins数量: %d", bins);
        return false;
    }
    
    // 初始化直方图
    memset(histogram, 0, bins * sizeof(unsigned int));
    
    if (img->bit_depth == 8) {
        // 8位处理
        int scale = 256 / bins;
        
        for (int y = 0; y < img->height; y++) {
            for (int x = 0; x < img->width; x++) {
                size_t idx = (y * img->width + x) * img->channels + channel;
                int bin = img->data[idx] / scale;
                if (bin >= bins) bin = bins - 1;
                histogram[bin]++;
            }
        }
    } else if (img->bit_depth == 16) {
        // 16位处理
        uint16_t *data = (uint16_t*)img->data;
        int scale = 65536 / bins;
        
        for (int y = 0; y < img->height; y++) {
            for (int x = 0; x < img->width; x++) {
                size_t idx = (y * img->width + x) * img->channels + channel;
                int bin = data[idx] / scale;
                if (bin >= bins) bin = bins - 1;
                histogram[bin]++;
            }
        }
    }
    
    return true;
}

/**
 * @brief 计算图像统计信息
 */
bool png_calculate_statistics(const Image *img, int channel, 
                              ImageStatistics *stats)
{
    png_clear_error();
    
    if (!img || !img->data || !stats) {
        png_set_error("无效的参数");
        return false;
    }
    
    if (channel < 0 || channel >= img->channels) {
        png_set_error("无效的通道索引: %d", channel);
        return false;
    }
    
    double sum = 0.0;
    double sum_sq = 0.0;
    int min_val = INT_MAX;
    int max_val = INT_MIN;
    size_t pixel_count = img->width * img->height;
    
    if (img->bit_depth == 8) {
        // 8位处理
        for (int y = 0; y < img->height; y++) {
            for (int x = 0; x < img->width; x++) {
                size_t idx = (y * img->width + x) * img->channels + channel;
                int val = img->data[idx];
                
                sum += val;
                sum_sq += val * val;
                if (val < min_val) min_val = val;
                if (val > max_val) max_val = val;
            }
        }
    } else if (img->bit_depth == 16) {
        // 16位处理
        uint16_t *data = (uint16_t*)img->data;
        
        for (int y = 0; y < img->height; y++) {
            for (int x = 0; x < img->width; x++) {
                size_t idx = (y * img->width + x) * img->channels + channel;
                int val = data[idx];
                
                sum += val;
                sum_sq += val * val;
                if (val < min_val) min_val = val;
                if (val > max_val) max_val = val;
            }
        }
    }
    
    stats->min = min_val;
    stats->max = max_val;
    stats->mean = sum / pixel_count;
    stats->variance = (sum_sq / pixel_count) - (stats->mean * stats->mean);
    stats->std_dev = sqrt(stats->variance);
    
    return true;
}

/**
 * @brief 比较两个图像是否相同
 */
bool png_compare(const Image *img1, const Image *img2)
{
    png_clear_error();
    
    if (!img1 || !img1->data || !img2 || !img2->data) {
        png_set_error("图像为空");
        return false;
    }
    
    // 检查基本属性
    if (img1->width != img2->width || 
        img1->height != img2->height ||
        img1->channels != img2->channels ||
        img1->bit_depth != img2->bit_depth) {
        return false;
    }
    
    // 比较数据
    size_t data_size = image_data_size(img1);
    return memcmp(img1->data, img2->data, data_size) == 0;
}

/**
 * @brief 计算两个图像的差异（MSE）
 */
double png_calculate_mse(const Image *img1, const Image *img2)
{
    png_clear_error();
    
    if (!img1 || !img1->data || !img2 || !img2->data) {
        png_set_error("图像为空");
        return -1.0;
    }
    
    if (img1->width != img2->width || 
        img1->height != img2->height ||
        img1->channels != img2->channels ||
        img1->bit_depth != img2->bit_depth) {
        png_set_error("图像尺寸或格式不匹配");
        return -1.0;
    }
    
    double mse = 0.0;
    size_t pixel_count = img1->width * img1->height * img1->channels;
    
    if (img1->bit_depth == 8) {
        // 8位处理
        for (size_t i = 0; i < pixel_count; i++) {
            double diff = img1->data[i] - img2->data[i];
            mse += diff * diff;
        }
    } else if (img1->bit_depth == 16) {
        // 16位处理
        uint16_t *data1 = (uint16_t*)img1->data;
        uint16_t *data2 = (uint16_t*)img2->data;
        
        for (size_t i = 0; i < pixel_count; i++) {
            double diff = data1[i] - data2[i];
            mse += diff * diff;
        }
    }
    
    return mse / pixel_count;
}

/**
 * @brief 计算PSNR（峰值信噪比）
 */
double png_calculate_psnr(const Image *img1, const Image *img2)
{
    double mse = png_calculate_mse(img1, img2);
    if (mse < 0.0 || mse == 0.0) {
        return -1.0;
    }
    
    double max_val = (img1->bit_depth == 8) ? 255.0 : 65535.0;
    return 10.0 * log10((max_val * max_val) / mse);
}

// ============================================================================
// 颜色空间转换
// ============================================================================

/**
 * @brief RGB转HSV
 */
static void rgb_to_hsv(uint8_t r, uint8_t g, uint8_t b,
                      float *h, float *s, float *v)
{
    float rf = r / 255.0f;
    float gf = g / 255.0f;
    float bf = b / 255.0f;
    
    float max = fmaxf(rf, fmaxf(gf, bf));
    float min = fminf(rf, fminf(gf, bf));
    float delta = max - min;
    
    // Value
    *v = max;
    
    // Saturation
    if (max > 0.0f) {
        *s = delta / max;
    } else {
        *s = 0.0f;
        *h = 0.0f;
        return;
    }
    
    // Hue
    if (delta == 0.0f) {
        *h = 0.0f;
    } else if (max == rf) {
        *h = 60.0f * fmodf((gf - bf) / delta, 6.0f);
    } else if (max == gf) {
        *h = 60.0f * ((bf - rf) / delta + 2.0f);
    } else {
        *h = 60.0f * ((rf - gf) / delta + 4.0f);
    }
    
    if (*h < 0.0f) {
        *h += 360.0f;
    }
}

/**
 * @brief HSV转RGB
 */
static void hsv_to_rgb(float h, float s, float v,
                      uint8_t *r, uint8_t *g, uint8_t *b)
{
    if (s == 0.0f) {
        // 灰色
        *r = *g = *b = (uint8_t)(v * 255);
        return;
    }
    
    float hh = h / 60.0f;
    int i = (int)hh;
    float f = hh - i;
    
    float p = v * (1.0f - s);
    float q = v * (1.0f - s * f);
    float t = v * (1.0f - s * (1.0f - f));
    
    float rf, gf, bf;
    
    switch (i) {
        case 0:
            rf = v; gf = t; bf = p;
            break;
        case 1:
            rf = q; gf = v; bf = p;
            break;
        case 2:
            rf = p; gf = v; bf = t;
            break;
        case 3:
            rf = p; gf = q; bf = v;
            break;
        case 4:
            rf = t; gf = p; bf = v;
            break;
        default:
            rf = v; gf = p; bf = q;
            break;
    }
    
    *r = (uint8_t)(rf * 255);
    *g = (uint8_t)(gf * 255);
    *b = (uint8_t)(bf * 255);
}

/**
 * @brief 调整色相
 */
Image* png_adjust_hue(const Image *img, float hue_shift)
{
    png_clear_error();
    
    if (!img || !img->data) {
        png_set_error("图像为空");
        return NULL;
    }
    
    if (img->channels < 3) {
        png_set_error("图像必须是RGB或RGBA格式");
        return NULL;
    }
    
    if (img->bit_depth != 8) {
        png_set_error("目前仅支持8位图像");
        return NULL;
    }
    
    // 创建结果图像
    Image *result = image_clone(img);
    if (!result) {
        return NULL;
    }
    
    // 处理每个像素
    for (int y = 0; y < img->height; y++) {
        for (int x = 0; x < img->width; x++) {
            size_t idx = (y * img->width + x) * img->channels;
            
            uint8_t r = img->data[idx];
            uint8_t g = img->data[idx + 1];
            uint8_t b = img->data[idx + 2];
            
            // RGB转HSV
            float h, s, v;
            rgb_to_hsv(r, g, b, &h, &s, &v);
            
            // 调整色相
            h += hue_shift;
            while (h < 0.0f) h += 360.0f;
            while (h >= 360.0f) h -= 360.0f;
            
            // HSV转RGB
            hsv_to_rgb(h, s, v, &r, &g, &b);
            
            result->data[idx] = r;
            result->data[idx + 1] = g;
            result->data[idx + 2] = b;
        }
    }
    
    return result;
}

/**
 * @brief 调整饱和度
 */
Image* png_adjust_saturation(const Image *img, float saturation_factor)
{
    png_clear_error();
    
    if (!img || !img->data) {
        png_set_error("图像为空");
        return NULL;
    }
    
    if (img->channels < 3) {
        png_set_error("图像必须是RGB或RGBA格式");
        return NULL;
    }
    
    if (img->bit_depth != 8) {
        png_set_error("目前仅支持8位图像");
        return NULL;
    }
    
    if (saturation_factor < 0.0f) {
        png_set_error("饱和度因子必须大于等于0");
        return NULL;
    }
    
    // 创建结果图像
    Image *result = image_clone(img);
    if (!result) {
        return NULL;
    }
    
    // 处理每个像素
    for (int y = 0; y < img->height; y++) {
        for (int x = 0; x < img->width; x++) {
            size_t idx = (y * img->width + x) * img->channels;
            
            uint8_t r = img->data[idx];
            uint8_t g = img->data[idx + 1];
            uint8_t b = img->data[idx + 2];
            
            // RGB转HSV
            float h, s, v;
            rgb_to_hsv(r, g, b, &h, &s, &v);
            
            // 调整饱和度
            s *= saturation_factor;
            s = CLAMP(s, 0.0f, 1.0f);
            
            // HSV转RGB
            hsv_to_rgb(h, s, v, &r, &g, &b);
            
            result->data[idx] = r;
            result->data[idx + 1] = g;
            result->data[idx + 2] = b;
        }
    }
    
    return result;
}

// ============================================================================
// 绘图功能
// ============================================================================

/**
 * @brief 在图像上绘制矩形
 */
bool png_draw_rectangle(Image *img, int x, int y, int width, int height,
                       uint8_t r, uint8_t g, uint8_t b, uint8_t a,
                       int thickness)
{
    png_clear_error();
    
    if (!img || !img->data) {
        png_set_error("图像为空");
        return false;
    }
    
    if (img->bit_depth != 8) {
        png_set_error("目前仅支持8位图像");
        return false;
    }
    
    if (thickness < 1) {
        png_set_error("线条粗细必须大于0");
        return false;
    }
    
    // 绘制四条边
    for (int t = 0; t < thickness; t++) {
        // 上边
        for (int i = x; i < x + width; i++) {
            int py = y + t;
            if (i >= 0 && i < img->width && py >= 0 && py < img->height) {
                size_t idx = (py * img->width + i) * img->channels;
                
                if (img->channels >= 3) {
                    img->data[idx] = r;
                    img->data[idx + 1] = g;
                    img->data[idx + 2] = b;
                    if (img->channels == 4) {
                        img->data[idx + 3] = a;
                    }
                } else {
                    // 灰度图：使用亮度值
                    img->data[idx] = (uint8_t)(0.299 * r + 0.587 * g + 0.114 * b);
                    if (img->channels == 2) {
                        img->data[idx + 1] = a;
                    }
                }
            }
        }
        
        // 下边
        for (int i = x; i < x + width; i++) {
            int py = y + height - 1 - t;
            if (i >= 0 && i < img->width && py >= 0 && py < img->height) {
                size_t idx = (py * img->width + i) * img->channels;
                
                if (img->channels >= 3) {
                    img->data[idx] = r;
                    img->data[idx + 1] = g;
                    img->data[idx + 2] = b;
                    if (img->channels == 4) {
                        img->data[idx + 3] = a;
                    }
                } else {
                    img->data[idx] = (uint8_t)(0.299 * r + 0.587 * g + 0.114 * b);
                    if (img->channels == 2) {
                        img->data[idx + 1] = a;
                    }
                }
            }
        }
        
        // 左边
        for (int i = y; i < y + height; i++) {
            int px = x + t;
            if (px >= 0 && px < img->width && i >= 0 && i < img->height) {
                size_t idx = (i * img->width + px) * img->channels;
                
                if (img->channels >= 3) {
                    img->data[idx] = r;
                    img->data[idx + 1] = g;
                    img->data[idx + 2] = b;
                    if (img->channels == 4) {
                        img->data[idx + 3] = a;
                    }
                } else {
                    img->data[idx] = (uint8_t)(0.299 * r + 0.587 * g + 0.114 * b);
                    if (img->channels == 2) {
                        img->data[idx + 1] = a;
                    }
                }
            }
        }
        
        // 右边
        for (int i = y; i < y + height; i++) {
            int px = x + width - 1 - t;
            if (px >= 0 && px < img->width && i >= 0 && i < img->height) {
                size_t idx = (i * img->width + px) * img->channels;
                
                if (img->channels >= 3) {
                    img->data[idx] = r;
                    img->data[idx + 1] = g;
                    img->data[idx + 2] = b;
                    if (img->channels == 4) {
                        img->data[idx + 3] = a;
                    }
                } else {
                    img->data[idx] = (uint8_t)(0.299 * r + 0.587 * g + 0.114 * b);
                    if (img->channels == 2) {
                        img->data[idx + 1] = a;
                    }
                }
            }
        }
    }
    
    return true;
}

/**
 * @brief 填充矩形区域
 */
bool png_fill_rectangle(Image *img, int x, int y, int width, int height,
                       uint8_t r, uint8_t g, uint8_t b, uint8_t a)
{
    png_clear_error();
    
    if (!img || !img->data) {
        png_set_error("图像为空");
        return false;
    }
    
    if (img->bit_depth != 8) {
        png_set_error("目前仅支持8位图像");
        return false;
    }
    
    // 裁剪到图像边界
    int x1 = MAX(x, 0);
    int y1 = MAX(y, 0);
    int x2 = MIN(x + width, img->width);
    int y2 = MIN(y + height, img->height);
    
    for (int py = y1; py < y2; py++) {
        for (int px = x1; px < x2; px++) {
            size_t idx = (py * img->width + px) * img->channels;
            
            if (img->channels >= 3) {
                img->data[idx] = r;
                img->data[idx + 1] = g;
                img->data[idx + 2] = b;
                if (img->channels == 4) {
                    img->data[idx + 3] = a;
                }
            } else {
                img->data[idx] = (uint8_t)(0.299 * r + 0.587 * g + 0.114 * b);
                if (img->channels == 2) {
                    img->data[idx + 1] = a;
                }
            }
        }
    }
    
    return true;
}

