/**
 * @file image_loader.c
 * @brief Image loading and saving implementation - Part 1
 * @author hany
 * @date 2025
 */

#include "image_loader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <ctype.h>

// ============================================================================
// 内部辅助宏定义
// ============================================================================

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define CLAMP(x, min, max) (MIN(MAX((x), (min)), (max)))

// ============================================================================
// 内部辅助函数声明
// ============================================================================

static inline float clamp_float(float value, float min, float max);
static inline int clamp_int(int value, int min, int max);
static inline uint8_t float_to_uint8(float value);
static inline float uint8_to_float(uint8_t value);
static inline uint16_t float_to_uint16(float value);
static inline float uint16_to_float(uint16_t value);
static bool is_valid_image_params(int width, int height, int channels);
static size_t get_image_data_size(int width, int height, int channels);

// ============================================================================
// 内部辅助函数实现
// ============================================================================

/**
 * @brief 限制浮点数范围
 */
static inline float clamp_float(float value, float min, float max)
{
    if (value < min) return min;
    if (value > max) return max;
    return value;
}

/**
 * @brief 限制整数范围
 */
static inline int clamp_int(int value, int min, int max)
{
    if (value < min) return min;
    if (value > max) return max;
    return value;
}

/**
 * @brief 浮点数转uint8（0-1映射到0-255）
 */
static inline uint8_t float_to_uint8(float value)
{
    return (uint8_t)(clamp_float(value, 0.0f, 1.0f) * 255.0f + 0.5f);
}

/**
 * @brief uint8转浮点数（0-255映射到0-1）
 */
static inline float uint8_to_float(uint8_t value)
{
    return (float)value / 255.0f;
}

/**
 * @brief 浮点数转uint16（0-1映射到0-65535）
 */
static inline uint16_t float_to_uint16(float value)
{
    return (uint16_t)(clamp_float(value, 0.0f, 1.0f) * 65535.0f + 0.5f);
}

/**
 * @brief uint16转浮点数（0-65535映射到0-1）
 */
static inline float uint16_to_float(uint16_t value)
{
    return (float)value / 65535.0f;
}

/**
 * @brief 验证图像参数是否有效
 */
static bool is_valid_image_params(int width, int height, int channels)
{
    if (width <= 0 || height <= 0) {
        return false;
    }
    if (channels <= 0 || channels > 4) {
        return false;
    }
    // 检查是否会溢出
    if (width > INT_MAX / height || 
        (size_t)width * height > SIZE_MAX / channels / sizeof(float)) {
        return false;
    }
    return true;
}

/**
 * @brief 计算图像数据大小
 */
static size_t get_image_data_size(int width, int height, int channels)
{
    return (size_t)width * height * channels;
}

// ============================================================================
// 图像创建和销毁函数
// ============================================================================

/**
 * @brief 创建新图像
 */
Image* image_create(int width, int height, int channels)
{
    // 参数验证
    if (!is_valid_image_params(width, height, channels)) {
        fprintf(stderr, "Error: Invalid image parameters (w=%d, h=%d, c=%d)\n", 
                width, height, channels);
        return NULL;
    }

    // 分配图像结构
    Image *img = (Image*)malloc(sizeof(Image));
    if (!img) {
        fprintf(stderr, "Error: Failed to allocate memory for Image structure\n");
        return NULL;
    }

    // 初始化基本属性
    img->width = width;
    img->height = height;
    img->channels = channels;
    img->pixel_type = PIXEL_TYPE_FLOAT32;
    img->metadata = NULL;

    // 设置颜色空间
    if (channels == 1) {
        img->color_space = COLOR_SPACE_GRAYSCALE;
    } else if (channels == 3) {
        img->color_space = COLOR_SPACE_RGB;
    } else if (channels == 4) {
        img->color_space = COLOR_SPACE_RGBA;
    } else {
        img->color_space = COLOR_SPACE_RGB;
    }

    // 分配数据内存
    size_t data_size = get_image_data_size(width, height, channels);
    img->data = (float*)calloc(data_size, sizeof(float));
    
    if (!img->data) {
        fprintf(stderr, "Error: Failed to allocate memory for image data (%zu bytes)\n", 
                data_size * sizeof(float));
        free(img);
        return NULL;
    }

    return img;
}

/**
 * @brief 创建带初始值的图像
 */
Image* image_create_with_value(int width, int height, int channels, float init_value)
{
    Image *img = image_create(width, height, channels);
    if (!img) {
        return NULL;
    }

    // 填充初始值
    size_t size = get_image_data_size(width, height, channels);
    for (size_t i = 0; i < size; i++) {
        img->data[i] = init_value;
    }

    return img;
}

/**
 * @brief 从数据创建图像
 */
Image* image_create_from_data(int width, int height, int channels, const float *data)
{
    if (!data) {
        fprintf(stderr, "Error: NULL data pointer\n");
        return NULL;
    }

    Image *img = image_create(width, height, channels);
    if (!img) {
        return NULL;
    }

    // 复制数据
    size_t size = get_image_data_size(width, height, channels);
    memcpy(img->data, data, size * sizeof(float));

    return img;
}

/**
 * @brief 克隆图像
 */
Image* image_clone(const Image *src)
{
    if (!src) {
        fprintf(stderr, "Error: NULL source image\n");
        return NULL;
    }

    if (!src->data) {
        fprintf(stderr, "Error: Source image has NULL data\n");
        return NULL;
    }

    // 创建新图像
    Image *dst = image_create(src->width, src->height, src->channels);
    if (!dst) {
        return NULL;
    }

    // 复制数据
    size_t size = get_image_data_size(src->width, src->height, src->channels);
    memcpy(dst->data, src->data, size * sizeof(float));
    
    // 复制属性
    dst->color_space = src->color_space;
    dst->pixel_type = src->pixel_type;

    // 复制元数据（如果存在）
    if (src->metadata) {
        dst->metadata = image_metadata_clone((ImageMetadata*)src->metadata);
    }

    return dst;
}

/**
 * @brief 销毁图像
 */
void image_destroy(Image *img)
{
    if (img) {
        // 释放数据
        if (img->data) {
            free(img->data);
            img->data = NULL;
        }
        
        // 释放元数据
        if (img->metadata) {
            image_metadata_destroy((ImageMetadata*)img->metadata);
            img->metadata = NULL;
        }
        
        // 释放结构本身
        free(img);
    }
}

// ============================================================================
// 像素访问函数
// ============================================================================

/**
 * @brief 获取像素值
 */
float image_get_pixel(const Image *img, int x, int y, int channel)
{
    if (!img || !img->data) {
        return 0.0f;
    }

    // 边界检查
    if (x < 0 || x >= img->width || y < 0 || y >= img->height || 
        channel < 0 || channel >= img->channels) {
        return 0.0f;
    }

    int idx = (y * img->width + x) * img->channels + channel;
    return img->data[idx];
}

/**
 * @brief 设置像素值
 */
void image_set_pixel(Image *img, int x, int y, int channel, float value)
{
    if (!img || !img->data) {
        return;
    }

    // 边界检查
    if (x < 0 || x >= img->width || y < 0 || y >= img->height || 
        channel < 0 || channel >= img->channels) {
        return;
    }

    int idx = (y * img->width + x) * img->channels + channel;
    img->data[idx] = value;
}

/**
 * @brief 获取像素值（带边界检查和默认值）
 */
float image_get_pixel_safe(const Image *img, int x, int y, int channel, float default_value)
{
    if (!img || !img->data) {
        return default_value;
    }

    // 边界检查
    if (x < 0 || x >= img->width || y < 0 || y >= img->height || 
        channel < 0 || channel >= img->channels) {
        return default_value;
    }

    int idx = (y * img->width + x) * img->channels + channel;
    return img->data[idx];
}

/**
 * @brief 双线性插值获取像素值
 */
float image_get_pixel_bilinear(const Image *img, float x, float y, int channel)
{
    if (!img || !img->data) {
        return 0.0f;
    }

    if (channel < 0 || channel >= img->channels) {
        return 0.0f;
    }

    // 获取四个最近的整数坐标
    int x0 = (int)floor(x);
    int y0 = (int)floor(y);
    int x1 = x0 + 1;
    int y1 = y0 + 1;

    // 计算插值权重
    float wx = x - x0;
    float wy = y - y0;

    // 获取四个角的像素值
    float p00 = image_get_pixel_safe(img, x0, y0, channel, 0.0f);
    float p10 = image_get_pixel_safe(img, x1, y0, channel, 0.0f);
    float p01 = image_get_pixel_safe(img, x0, y1, channel, 0.0f);
    float p11 = image_get_pixel_safe(img, x1, y1, channel, 0.0f);

    // 双线性插值
    float p0 = p00 * (1.0f - wx) + p10 * wx;
    float p1 = p01 * (1.0f - wx) + p11 * wx;
    float result = p0 * (1.0f - wy) + p1 * wy;

    return result;
}

// ============================================================================
// 工具函数
// ============================================================================

/**
 * @brief 检查图像是否有效
 */
bool image_is_valid(const Image *img)
{
    if (!img) {
        return false;
    }
    if (!img->data) {
        return false;
    }
    if (!is_valid_image_params(img->width, img->height, img->channels)) {
        return false;
    }
    return true;
}

/**
 * @brief 检查两个图像尺寸是否相同
 */
bool image_same_size(const Image *img1, const Image *img2)
{
    if (!img1 || !img2) {
        return false;
    }
    return (img1->width == img2->width && 
            img1->height == img2->height && 
            img1->channels == img2->channels);
}

/**
 * @brief 复制图像数据
 */
int image_copy_data(Image *dst, const Image *src)
{
    if (!dst || !src) {
        return IMAGE_ERROR_NULL_POINTER;
    }

    if (!dst->data || !src->data) {
        return IMAGE_ERROR_NULL_POINTER;
    }

    if (!image_same_size(dst, src)) {
        return IMAGE_ERROR_DIMENSION;
    }

    size_t size = get_image_data_size(src->width, src->height, src->channels);
    memcpy(dst->data, src->data, size * sizeof(float));

    return IMAGE_SUCCESS;
}

/**
 * @brief 填充图像
 */
void image_fill(Image *img, float value)
{
    if (!img || !img->data) {
        return;
    }

    size_t size = get_image_data_size(img->width, img->height, img->channels);
    for (size_t i = 0; i < size; i++) {
        img->data[i] = value;
    }
}

/**
 * @brief 清空图像（填充0）
 */
void image_clear(Image *img)
{
    if (!img || !img->data) {
        return;
    }

    size_t size = get_image_data_size(img->width, img->height, img->channels);
    memset(img->data, 0, size * sizeof(float));
}

/**
 * @brief 获取错误信息
 */
const char* image_error_string(int error_code)
{
    switch (error_code) {
        case IMAGE_SUCCESS:
            return "Success";
        case IMAGE_ERROR_NULL_POINTER:
            return "NULL pointer error";
        case IMAGE_ERROR_INVALID_PARAM:
            return "Invalid parameter";
        case IMAGE_ERROR_MEMORY_ALLOC:
            return "Memory allocation failed";
        case IMAGE_ERROR_FILE_NOT_FOUND:
            return "File not found";
        case IMAGE_ERROR_FILE_READ:
            return "File read error";
        case IMAGE_ERROR_FILE_WRITE:
            return "File write error";
        case IMAGE_ERROR_UNSUPPORTED:
            return "Unsupported operation";
        case IMAGE_ERROR_INVALID_FORMAT:
            return "Invalid image format";
        case IMAGE_ERROR_CORRUPTED:
            return "Corrupted image data";
        case IMAGE_ERROR_DIMENSION:
            return "Dimension mismatch";
        default:
            return "Unknown error";
    }
}

/**
 * @brief 打印图像信息
 */
void image_print_info(const Image *img)
{
    if (!img) {
        printf("Image: NULL\n");
        return;
    }

    printf("Image Information:\n");
    printf("  Size: %d x %d\n", img->width, img->height);
    printf("  Channels: %d\n", img->channels);
    
    const char *color_space_str = "Unknown";
    switch (img->color_space) {
        case COLOR_SPACE_RGB: color_space_str = "RGB"; break;
        case COLOR_SPACE_RGBA: color_space_str = "RGBA"; break;
        case COLOR_SPACE_GRAYSCALE: color_space_str = "Grayscale"; break;
        case COLOR_SPACE_YUV: color_space_str = "YUV"; break;
        case COLOR_SPACE_HSV: color_space_str = "HSV"; break;
        case COLOR_SPACE_LAB: color_space_str = "LAB"; break;
    }
    printf("  Color Space: %s\n", color_space_str);
    
    const char *pixel_type_str = "Unknown";
    switch (img->pixel_type) {
        case PIXEL_TYPE_UINT8: pixel_type_str = "UINT8"; break;
        case PIXEL_TYPE_UINT16: pixel_type_str = "UINT16"; break;
        case PIXEL_TYPE_FLOAT32: pixel_type_str = "FLOAT32"; break;
        case PIXEL_TYPE_FLOAT64: pixel_type_str = "FLOAT64"; break;
    }
    printf("  Pixel Type: %s\n", pixel_type_str);
    
    printf("  Data: %s\n", img->data ? "Valid" : "NULL");
    printf("  Metadata: %s\n", img->metadata ? "Present" : "None");
}

/**
 * @brief 获取默认加载选项
 */
ImageLoadOptions image_get_default_load_options(void)
{
    ImageLoadOptions options;
    options.normalize = true;
    options.convert_to_rgb = false;
    options.preserve_alpha = true;
    options.target_channels = 0;  // 保持原样
    options.gamma_correction = 0.0f;  // 不校正
    options.flip_vertical = false;
    options.flip_horizontal = false;
    return options;
}

/**
 * @brief 获取默认保存选项
 */
ImageSaveOptions image_get_default_save_options(void)
{
    ImageSaveOptions options;
    options.quality = 95;  // JPEG质量
    options.compression_level = 6;  // PNG压缩级别
    options.denormalize = true;
    options.gamma_correction = 0.0f;  // 不校正
    options.save_metadata = true;
    options.dpi = 96;
    return options;
}
// ============================================================================
// 格式检测函数
// ============================================================================

/**
 * @brief 通过文件头检测图像格式
 */
ImageFormat image_detect_format(const char *filename)
{
    if (!filename) {
        return IMAGE_FORMAT_UNKNOWN;
    }

    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        return IMAGE_FORMAT_UNKNOWN;
    }

    unsigned char header[16];
    size_t bytes_read = fread(header, 1, 16, fp);
    fclose(fp);

    if (bytes_read < 8) {
        return IMAGE_FORMAT_UNKNOWN;
    }

    // PNG: 89 50 4E 47 0D 0A 1A 0A
    if (header[0] == 0x89 && header[1] == 0x50 && 
        header[2] == 0x4E && header[3] == 0x47 &&
        header[4] == 0x0D && header[5] == 0x0A &&
        header[6] == 0x1A && header[7] == 0x0A) {
        return IMAGE_FORMAT_PNG;
    }

    // JPEG: FF D8 FF
    if (header[0] == 0xFF && header[1] == 0xD8 && header[2] == 0xFF) {
        return IMAGE_FORMAT_JPEG;
    }

    // BMP: 42 4D (BM)
    if (header[0] == 0x42 && header[1] == 0x4D) {
        return IMAGE_FORMAT_BMP;
    }

    // TIFF: 49 49 2A 00 (little-endian) or 4D 4D 00 2A (big-endian)
    if ((header[0] == 0x49 && header[1] == 0x49 && 
         header[2] == 0x2A && header[3] == 0x00) ||
        (header[0] == 0x4D && header[1] == 0x4D && 
         header[2] == 0x00 && header[3] == 0x2A)) {
        return IMAGE_FORMAT_TIFF;
    }

    // PPM/PGM: P5 or P6
    if (header[0] == 'P') {
        if (header[1] == '5') return IMAGE_FORMAT_PGM;
        if (header[1] == '6') return IMAGE_FORMAT_PPM;
        if (header[1] == '2') return IMAGE_FORMAT_PGM;  // ASCII PGM
        if (header[1] == '3') return IMAGE_FORMAT_PPM;  // ASCII PPM
    }

    // WebP: RIFF....WEBP
    if (bytes_read >= 12 &&
        header[0] == 'R' && header[1] == 'I' && 
        header[2] == 'F' && header[3] == 'F' &&
        header[8] == 'W' && header[9] == 'E' && 
        header[10] == 'B' && header[11] == 'P') {
        return
        return IMAGE_FORMAT_WEBP;
    }

    return IMAGE_FORMAT_UNKNOWN;
}

/**
 * @brief 根据文件扩展名获取格式
 */
ImageFormat image_get_format_from_extension(const char *filename)
{
    if (!filename) {
        return IMAGE_FORMAT_UNKNOWN;
    }

    const char *ext = strrchr(filename, '.');
    if (!ext) {
        return IMAGE_FORMAT_UNKNOWN;
    }
    ext++; // 跳过点号

    if (strcasecmp(ext, "png") == 0) return IMAGE_FORMAT_PNG;
    if (strcasecmp(ext, "jpg") == 0 || strcasecmp(ext, "jpeg") == 0) return IMAGE_FORMAT_JPEG;
    if (strcasecmp(ext, "bmp") == 0) return IMAGE_FORMAT_BMP;
    if (strcasecmp(ext, "tif") == 0 || strcasecmp(ext, "tiff") == 0) return IMAGE_FORMAT_TIFF;
    if (strcasecmp(ext, "ppm") == 0) return IMAGE_FORMAT_PPM;
    if (strcasecmp(ext, "pgm") == 0) return IMAGE_FORMAT_PGM;
    if (strcasecmp(ext, "webp") == 0) return IMAGE_FORMAT_WEBP;

    return IMAGE_FORMAT_UNKNOWN;
}

// ============================================================================
// PNG 加载函数
// ============================================================================

/**
 * @brief 加载 PNG 图像
 */
static Image* load_png(const char *filename)
{
    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        fprintf(stderr, "无法打开文件: %s\n", filename);
        return NULL;
    }

    // 读取文件头验证
    unsigned char header[8];
    if (fread(header, 1, 8, fp) != 8) {
        fprintf(stderr, "读取PNG文件头失败\n");
        fclose(fp);
        return NULL;
    }

    if (png_sig_cmp(header, 0, 8)) {
        fprintf(stderr, "不是有效的PNG文件\n");
        fclose(fp);
        return NULL;
    }

    // 创建PNG读取结构
    png_structp png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png_ptr) {
        fprintf(stderr, "创建PNG读取结构失败\n");
        fclose(fp);
        return NULL;
    }

    png_infop info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr) {
        fprintf(stderr, "创建PNG信息结构失败\n");
        png_destroy_read_struct(&png_ptr, NULL, NULL);
        fclose(fp);
        return NULL;
    }

    // 设置错误处理
    if (setjmp(png_jmpbuf(png_ptr))) {
        fprintf(stderr, "PNG读取过程中发生错误\n");
        png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
        fclose(fp);
        return NULL;
    }

    // 初始化IO
    png_init_io(png_ptr, fp);
    png_set_sig_bytes(png_ptr, 8);

    // 读取PNG信息
    png_read_info(png_ptr, info_ptr);

    int width = png_get_image_width(png_ptr, info_ptr);
    int height = png_get_image_height(png_ptr, info_ptr);
    png_byte color_type = png_get_color_type(png_ptr, info_ptr);
    png_byte bit_depth = png_get_bit_depth(png_ptr, info_ptr);

    // 转换为标准格式
    if (bit_depth == 16) {
        png_set_strip_16(png_ptr);
    }

    if (color_type == PNG_COLOR_TYPE_PALETTE) {
        png_set_palette_to_rgb(png_ptr);
    }

    if (color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8) {
        png_set_expand_gray_1_2_4_to_8(png_ptr);
    }

    if (png_get_valid(png_ptr, info_ptr, PNG_INFO_tRNS)) {
        png_set_tRNS_to_alpha(png_ptr);
    }

    if (color_type == PNG_COLOR_TYPE_RGB ||
        color_type == PNG_COLOR_TYPE_GRAY ||
        color_type == PNG_COLOR_TYPE_PALETTE) {
        png_set_filler(png_ptr, 0xFF, PNG_FILLER_AFTER);
    }

    if (color_type == PNG_COLOR_TYPE_GRAY ||
        color_type == PNG_COLOR_TYPE_GRAY_ALPHA) {
        png_set_gray_to_rgb(png_ptr);
    }

    png_read_update_info(png_ptr, info_ptr);

    // 创建图像对象
    Image *img = image_create(width, height, 4);
    if (!img) {
        fprintf(stderr, "创建图像对象失败\n");
        png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
        fclose(fp);
        return NULL;
    }

    // 分配行指针
    png_bytep *row_pointers = (png_bytep*)malloc(sizeof(png_bytep) * height);
    if (!row_pointers) {
        fprintf(stderr, "分配行指针内存失败\n");
        image_destroy(img);
        png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
        fclose(fp);
        return NULL;
    }

    for (int y = 0; y < height; y++) {
        row_pointers[y] = img->data + y * img->stride;
    }

    // 读取图像数据
    png_read_image(png_ptr, row_pointers);

    // 清理
    free(row_pointers);
    png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
    fclose(fp);

    img->format = IMAGE_FORMAT_PNG;
    return img;
}

// ============================================================================
// JPEG 加载函数
// ============================================================================

/**
 * @brief 加载 JPEG 图像
 */
static Image* load_jpeg(const char *filename)
{
    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        fprintf(stderr, "无法打开文件: %s\n", filename);
        return NULL;
    }

    struct jpeg_decompress_struct cinfo;
    struct jpeg_error_mgr jerr;

    // 设置错误处理
    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_decompress(&cinfo);

    // 指定数据源
    jpeg_stdio_src(&cinfo, fp);

    // 读取文件头
    jpeg_read_header(&cinfo, TRUE);

    // 开始解压
    jpeg_start_decompress(&cinfo);

    int width = cinfo.output_width;
    int height = cinfo.output_height;
    int channels = cinfo.output_components;

    // 创建图像对象
    Image *img = image_create(width, height, channels);
    if (!img) {
        fprintf(stderr, "创建图像对象失败\n");
        jpeg_destroy_decompress(&cinfo);
        fclose(fp);
        return NULL;
    }

    // 逐行读取
    JSAMPARRAY buffer = (*cinfo.mem->alloc_sarray)
        ((j_common_ptr)&cinfo, JPOOL_IMAGE, width * channels, 1);

    while (cinfo.output_scanline < cinfo.output_height) {
        int y = cinfo.output_scanline;
        jpeg_read_scanlines(&cinfo, buffer, 1);
        memcpy(img->data + y * img->stride, buffer[0], width * channels);
    }

    // 完成解压
    jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);
    fclose(fp);

    img->format = IMAGE_FORMAT_JPEG;
    return img;
}

// ============================================================================
// BMP 加载函数
// ============================================================================

#pragma pack(push, 1)
typedef struct {
    uint16_t type;
    uint32_t size;
    uint16_t reserved1;
    uint16_t reserved2;
    uint32_t offset;
} BMPFileHeader;

typedef struct {
    uint32_t size;
    int32_t width;
    int32_t height;
    uint16_t planes;
    uint16_t bit_count;
    uint32_t compression;
    uint32_t size_image;
    int32_t x_pels_per_meter;
    int32_t y_pels_per_meter;
    uint32_t clr_used;
    uint32_t clr_important;
} BMPInfoHeader;
#pragma pack(pop)

/**
 * @brief 加载 BMP 图像
 */
static Image* load_bmp(const char *filename)
{
    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        fprintf(stderr, "无法打开文件: %s\n", filename);
        return NULL;
    }

    BMPFileHeader file_header;
    BMPInfoHeader info_header;

    // 读取文件头
    if (fread(&file_header, sizeof(BMPFileHeader), 1, fp) != 1) {
        fprintf(stderr, "读取BMP文件头失败\n");
        fclose(fp);
        return NULL;
    }

    // 验证BMP标识
    if (file_header.type != 0x4D42) { // 'BM'
        fprintf(stderr, "不是有效的BMP文件\n");
        fclose(fp);
        return NULL;
    }

    // 读取信息头
    if (fread(&info_header, sizeof(BMPInfoHeader), 1, fp) != 1) {
        fprintf(stderr, "读取BMP信息头失败\n");
        fclose(fp);
        return NULL;
    }

    int width = info_header.width;
    int height = abs(info_header.height);
    int bit_count = info_header.bit_count;
    bool top_down = (info_header.height < 0);

    // 只支持24位和32位BMP
    if (bit_count != 24 && bit_count != 32) {
        fprintf(stderr, "不支持的BMP位深度: %d\n", bit_count);
        fclose(fp);
        return NULL;
    }

    int channels = (bit_count == 24) ? 3 : 4;
    
    // 创建图像对象
    Image *img = image_create(width, height, channels);
    if (!img) {
        fprintf(stderr, "创建图像对象失败\n");
        fclose(fp);
        return NULL;
    }

    // 定位到像素数据
    fseek(fp, file_header.offset, SEEK_SET);

    // BMP行对齐到4字节
    int row_size = ((width * bit_count + 31) / 32) * 4;
    unsigned char *row_buffer = (unsigned char*)malloc(row_size);
    if (!row_buffer) {
        fprintf(stderr, "分配行缓冲区失败\n");
        image_destroy(img);
        fclose(fp);
        return NULL;
    }

    // 读取像素数据（BMP是从下到上存储的）
    for (int y = 0; y < height; y++) {
        int target_y = top_down ? y : (height - 1 - y);
        
        if (fread(row_buffer, 1, row_size, fp) != row_size) {
            fprintf(stderr, "读取像素数据失败\n");
            free(row_buffer);
            image_destroy(img);
            fclose(fp);
            return NULL;
        }

        unsigned char *dst = img->data + target_y * img->stride;
        
        // BMP是BGR格式，需要转换为RGB
        for (int x = 0; x < width; x++) {
            if (channels == 3) {
                dst[x * 3 + 0] = row_buffer[x * 3 + 2]; // R
                dst[x * 3 + 1] = row_buffer[x * 3 + 1]; // G
                dst[x * 3 + 2] = row_buffer[x * 3 + 0]; // B
            } else {
                dst[x * 4 + 0] = row_buffer[x * 4 + 2]; // R
                dst[x * 4 + 1] = row_buffer[x * 4 + 1]; // G
                dst[x * 4 + 2] = row_buffer[x * 4 + 0]; // B
                dst[x * 4 + 3] = row_buffer[x * 4 + 3]; // A
            }
        }
    }

    free(row_buffer);
    fclose(fp);

    img->format = IMAGE_FORMAT_BMP;
    return img;
}

// 第二部分结束
// ============================================================================
// PPM/PGM 加载函数
// ============================================================================

/**
 * @brief 跳过PPM/PGM文件中的注释和空白
 */
static void skip_ppm_comments(FILE *fp)
{
    int c;
    while ((c = fgetc(fp)) != EOF) {
        if (c == '#') {
            // 跳过注释行
            while ((c = fgetc(fp)) != EOF && c != '\n');
        } else if (!isspace(c)) {
            ungetc(c, fp);
            break;
        }
    }
}

/**
 * @brief 加载 PPM 图像 (P6格式)
 */
static Image* load_ppm(const char *filename)
{
    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        fprintf(stderr, "无法打开文件: %s\n", filename);
        return NULL;
    }

    char magic[3];
    if (fread(magic, 1, 2, fp) != 2) {
        fprintf(stderr, "读取PPM魔数失败\n");
        fclose(fp);
        return NULL;
    }
    magic[2] = '\0';

    bool is_ascii = false;
    if (strcmp(magic, "P3") == 0) {
        is_ascii = true;
    } else if (strcmp(magic, "P6") != 0) {
        fprintf(stderr, "不支持的PPM格式: %s\n", magic);
        fclose(fp);
        return NULL;
    }

    skip_ppm_comments(fp);

    int width, height, maxval;
    if (fscanf(fp, "%d", &width) != 1) {
        fprintf(stderr, "读取PPM宽度失败\n");
        fclose(fp);
        return NULL;
    }

    skip_ppm_comments(fp);

    if (fscanf(fp, "%d", &height) != 1) {
        fprintf(stderr, "读取PPM高度失败\n");
        fclose(fp);
        return NULL;
    }

    skip_ppm_comments(fp);

    if (fscanf(fp, "%d", &maxval) != 1) {
        fprintf(stderr, "读取PPM最大值失败\n");
        fclose(fp);
        return NULL;
    }

    // 跳过最后一个空白字符
    fgetc(fp);

    if (maxval != 255) {
        fprintf(stderr, "只支持8位PPM图像\n");
        fclose(fp);
        return NULL;
    }

    // 创建图像对象
    Image *img = image_create(width, height, 3);
    if (!img) {
        fprintf(stderr, "创建图像对象失败\n");
        fclose(fp);
        return NULL;
    }

    if (is_ascii) {
        // ASCII格式
        for (int y = 0; y < height; y++) {
            unsigned char *row = img->data + y * img->stride;
            for (int x = 0; x < width; x++) {
                int r, g, b;
                if (fscanf(fp, "%d %d %d", &r, &g, &b) != 3) {
                    fprintf(stderr, "读取PPM像素数据失败\n");
                    image_destroy(img);
                    fclose(fp);
                    return NULL;
                }
                row[x * 3 + 0] = (unsigned char)r;
                row[x * 3 + 1] = (unsigned char)g;
                row[x * 3 + 2] = (unsigned char)b;
            }
        }
    } else {
        // 二进制格式
        size_t data_size = width * height * 3;
        if (fread(img->data, 1, data_size, fp) != data_size) {
            fprintf(stderr, "读取PPM像素数据失败\n");
            image_destroy(img);
            fclose(fp);
            return NULL;
        }
    }

    fclose(fp);
    img->format = IMAGE_FORMAT_PPM;
    return img;
}

/**
 * @brief 加载 PGM 图像 (P5格式)
 */
static Image* load_pgm(const char *filename)
{
    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        fprintf(stderr, "无法打开文件: %s\n", filename);
        return NULL;
    }

    char magic[3];
    if (fread(magic, 1, 2, fp) != 2) {
        fprintf(stderr, "读取PGM魔数失败\n");
        fclose(fp);
        return NULL;
    }
    magic[2] = '\0';

    bool is_ascii = false;
    if (strcmp(magic, "P2") == 0) {
        is_ascii = true;
    } else if (strcmp(magic, "P5") != 0) {
        fprintf(stderr, "不支持的PGM格式: %s\n", magic);
        fclose(fp);
        return NULL;
    }

    skip_ppm_comments(fp);

    int width, height, maxval;
    if (fscanf(fp, "%d", &width) != 1) {
        fprintf(stderr, "读取PGM宽度失败\n");
        fclose(fp);
        return NULL;
    }

    skip_ppm_comments(fp);

    if (fscanf(fp, "%d", &height) != 1) {
        fprintf(stderr, "读取PGM高度失败\n");
        fclose(fp);
        return NULL;
    }

    skip_ppm_comments(fp);

    if (fscanf(fp, "%d", &maxval) != 1) {
        fprintf(stderr, "读取PGM最大值失败\n");
        fclose(fp);
        return NULL;
    }

    fgetc(fp); // 跳过空白

    if (maxval != 255) {
        fprintf(stderr, "只支持8位PGM图像\n");
        fclose(fp);
        return NULL;
    }

    // 创建图像对象
    Image *img = image_create(width, height, 1);
    if (!img) {
        fprintf(stderr, "创建图像对象失败\n");
        fclose(fp);
        return NULL;
    }

    if (is_ascii) {
        // ASCII格式
        for (int y = 0; y < height; y++) {
            unsigned char *row = img->data + y * img->stride;
            for (int x = 0; x < width; x++) {
                int gray;
                if (fscanf(fp, "%d", &gray) != 1) {
                    fprintf(stderr, "读取PGM像素数据失败\n");
                    image_destroy(img);
                    fclose(fp);
                    return NULL;
                }
                row[x] = (unsigned char)gray;
            }
        }
    } else {
        // 二进制格式
        size_t data_size = width * height;
        if (fread(img->data, 1, data_size, fp) != data_size) {
            fprintf(stderr, "读取PGM像素数据失败\n");
            image_destroy(img);
            fclose(fp);
            return NULL;
        }
    }

    fclose(fp);
    img->format = IMAGE_FORMAT_PGM;
    return img;
}

// ============================================================================
// TIFF 加载函数
// ============================================================================

/**
 * @brief 加载 TIFF 图像
 */
static Image* load_tiff(const char *filename)
{
    TIFF *tif = TIFFOpen(filename, "r");
    if (!tif) {
        fprintf(stderr, "无法打开TIFF文件: %s\n", filename);
        return NULL;
    }

    uint32_t width, height;
    uint16_t samples_per_pixel, bits_per_sample;
    uint16_t photometric;

    // 读取TIFF标签
    TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &width);
    TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &height);
    TIFFGetField(tif, TIFFTAG_SAMPLESPERPIXEL, &samples_per_pixel);
    TIFFGetField(tif, TIFFTAG_BITSPERSAMPLE, &bits_per_sample);
    TIFFGetField(tif, TIFFTAG_PHOTOMETRIC, &photometric);

    if (bits_per_sample != 8 && bits_per_sample != 16) {
        fprintf(stderr, "不支持的TIFF位深度: %d\n", bits_per_sample);
        TIFFClose(tif);
        return NULL;
    }

    // 创建图像对象
    Image *img = image_create(width, height, samples_per_pixel);
    if (!img) {
        fprintf(stderr, "创建图像对象失败\n");
        TIFFClose(tif);
        return NULL;
    }

    // 使用RGBA接口读取（最简单的方法）
    uint32_t *raster = (uint32_t*)_TIFFmalloc(width * height * sizeof(uint32_t));
    if (!raster) {
        fprintf(stderr, "分配TIFF光栅缓冲区失败\n");
        image_destroy(img);
        TIFFClose(tif);
        return NULL;
    }

    if (!TIFFReadRGBAImageOriented(tif, width, height, raster, ORIENTATION_TOPLEFT, 0)) {
        fprintf(stderr, "读取TIFF图像数据失败\n");
        _TIFFfree(raster);
        image_destroy(img);
        TIFFClose(tif);
        return NULL;
    }

    // 转换RGBA数据到图像格式
    for (uint32_t y = 0; y < height; y++) {
        unsigned char *row = img->data + y * img->stride;
        for (uint32_t x = 0; x < width; x++) {
            uint32_t pixel = raster[y * width + x];
            
            if (samples_per_pixel >= 3) {
                row[x * samples_per_pixel + 0] = TIFFGetR(pixel);
                row[x * samples_per_pixel + 1] = TIFFGetG(pixel);
                row[x * samples_per_pixel + 2] = TIFFGetB(pixel);
                if (samples_per_pixel == 4) {
                    row[x * samples_per_pixel + 3] = TIFFGetA(pixel);
                }
            } else {
                // 灰度图像
                row[x] = TIFFGetR(pixel);
            }
        }
    }

    _TIFFfree(raster);
    TIFFClose(tif);

    img->format = IMAGE_FORMAT_TIFF;
    return img;
}

// ============================================================================
// WebP 加载函数
// ============================================================================

/**
 * @brief 加载 WebP 图像
 */
static Image* load_webp(const char *filename)
{
    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        fprintf(stderr, "无法打开文件: %s\n", filename);
        return NULL;
    }

    // 获取文件大小
    fseek(fp, 0, SEEK_END);
    size_t file_size = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    // 读取整个文件
    uint8_t *file_data = (uint8_t*)malloc(file_size);
    if (!file_data) {
        fprintf(stderr, "分配文件缓冲区失败\n");
        fclose(fp);
        return NULL;
    }

    if (fread(file_data, 1, file_size, fp) != file_size) {
        fprintf(stderr, "读取WebP文件失败\n");
        free(file_data);
        fclose(fp);
        return NULL;
    }
    fclose(fp);

    // 获取图像信息
    int width, height;
    if (!WebPGetInfo(file_data, file_size, &width, &height)) {
        fprintf(stderr, "无效的WebP文件\n");
        free(file_data);
        return NULL;
    }

    // 创建图像对象
    Image *img = image_create(width, height, 4); // WebP解码为RGBA
    if (!img) {
        fprintf(stderr, "创建图像对象失败\n");
        free(file_data);
        return NULL;
    }

    // 解码WebP
    uint8_t *decoded = WebPDecodeRGBAInto(
        file_data, file_size,
        img->data, img->stride * height,
        img->stride
    );

    if (!decoded) {
        fprintf(stderr, "WebP解码失败\n");
        free(file_data);
        image_destroy(img);
        return NULL;
    }

    free(file_data);
    img->format = IMAGE_FORMAT_WEBP;
    return img;
}

// ============================================================================
// 统一加载接口
// ============================================================================

/**
 * @brief 加载图像（自动检测格式）
 */
Image* image_load(const char *filename)
{
    if (!filename) {
        fprintf(stderr, "文件名为空\n");
        return NULL;
    }

    // 首先尝试通过文件头检测格式
    ImageFormat format = image_detect_format(filename);
    
    // 如果检测失败，尝试通过扩展名
    if (format == IMAGE_FORMAT_UNKNOWN) {
        format = image_get_format_from_extension(filename);
    }

    if (format == IMAGE_FORMAT_UNKNOWN) {
        fprintf(stderr, "无法识别图像格式: %s\n", filename);
        return NULL;
    }

    // 根据格式调用相应的加载函数
    Image *img = NULL;
    switch (format) {
        case IMAGE_FORMAT_PNG:
            img = load_png(filename);
            break;
        case IMAGE_FORMAT_JPEG:
            img = load_jpeg(filename);
            break;
        case IMAGE_FORMAT_BMP:
            img = load_bmp(filename);
            break;
        case IMAGE_FORMAT_TIFF:
            img = load_tiff(filename);
            break;
        case IMAGE_FORMAT_PPM:
            img = load_ppm(filename);
            break;
        case IMAGE_FORMAT_PGM:
            img = load_pgm(filename);
            break;
        case IMAGE_FORMAT_WEBP:
            img = load_webp(filename);
            break;
        default:
            fprintf(stderr, "不支持的图像格式\n");
            return NULL;
    }

    if (img) {
        printf("成功加载图像: %s (%dx%d, %d通道)\n", 
               filename, img->width, img->height, img->channels);
    }

    return img;
}

/**
 * @brief 加载指定格式的图像
 */
Image* image_load_format(const char *filename, ImageFormat format)
{
    if (!filename) {
        fprintf(stderr, "文件名为空\n");
        return NULL;
    }

    Image *img = NULL;
    switch (format) {
        case IMAGE_FORMAT_PNG:
            img = load_png(filename);
            break;
        case IMAGE_FORMAT_JPEG:
            img = load_jpeg(filename);
            break;
        case IMAGE_FORMAT_BMP:
            img = load_bmp(filename);
            break;
        case IMAGE_FORMAT_TIFF:
            img = load_tiff(filename);
            break;
        case IMAGE_FORMAT_PPM:
            img = load_ppm(filename);
            break;
        case IMAGE_FORMAT_PGM:
            img = load_pgm(filename);
            break;
        case IMAGE_FORMAT_WEBP:
            img = load_webp(filename);
            break;
        default:
            fprintf(stderr, "不支持的图像格式\n");
            return NULL;
    }

    return img;
}

// 第三部分结束
// ============================================================================
// PNG 保存函数
// ============================================================================

/**
 * @brief 保存为 PNG 格式
 */
static bool save_png(const Image *img, const char *filename)
{
    if (!img || !filename) {
        return false;
    }

    FILE *fp = fopen(filename, "wb");
    if (!fp) {
        fprintf(stderr, "无法创建文件: %s\n", filename);
        return false;
    }

    // 创建PNG写入结构
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png_ptr) {
        fprintf(stderr, "创建PNG写入结构失败\n");
        fclose(fp);
        return false;
    }

    png_infop info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr) {
        fprintf(stderr, "创建PNG信息结构失败\n");
        png_destroy_write_struct(&png_ptr, NULL);
        fclose(fp);
        return false;
    }

    // 设置错误处理
    if (setjmp(png_jmpbuf(png_ptr))) {
        fprintf(stderr, "PNG写入过程中发生错误\n");
        png_destroy_write_struct(&png_ptr, &info_ptr);
        fclose(fp);
        return false;
    }

    // 初始化IO
    png_init_io(png_ptr, fp);

    // 确定颜色类型
    int color_type;
    switch (img->channels) {
        case 1:
            color_type = PNG_COLOR_TYPE_GRAY;
            break;
        case 2:
            color_type = PNG_COLOR_TYPE_GRAY_ALPHA;
            break;
        case 3:
            color_type = PNG_COLOR_TYPE_RGB;
            break;
        case 4:
            color_type = PNG_COLOR_TYPE_RGBA;
            break;
        default:
            fprintf(stderr, "不支持的通道数: %d\n", img->channels);
            png_destroy_write_struct(&png_ptr, &info_ptr);
            fclose(fp);
            return false;
    }

    // 设置图像信息
    png_set_IHDR(png_ptr, info_ptr, img->width, img->height,
                 8, color_type, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);

    // 写入文件头
    png_write_info(png_ptr, info_ptr);

    // 分配行指针
    png_bytep *row_pointers = (png_bytep*)malloc(sizeof(png_bytep) * img->height);
    if (!row_pointers) {
        fprintf(stderr, "分配行指针内存失败\n");
        png_destroy_write_struct(&png_ptr, &info_ptr);
        fclose(fp);
        return false;
    }

    for (int y = 0; y < img->height; y++) {
        row_pointers[y] = img->data + y * img->stride;
    }

    // 写入图像数据
    png_write_image(png_ptr, row_pointers);

    // 完成写入
    png_write_end(png_ptr, NULL);

    // 清理
    free(row_pointers);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);

    return true;
}

// ============================================================================
// JPEG 保存函数
// ============================================================================

/**
 * @brief 保存为 JPEG 格式
 */
static bool save_jpeg(const Image *img, const char *filename, int quality)
{
    if (!img || !filename) {
        return false;
    }

    // JPEG只支持1或3通道
    if (img->channels != 1 && img->channels != 3) {
        fprintf(stderr, "JPEG只支持灰度或RGB图像\n");
        return false;
    }

    FILE *fp = fopen(filename, "wb");
    if (!fp) {
        fprintf(stderr, "无法创建文件: %s\n", filename);
        return false;
    }

    struct jpeg_compress_struct cinfo;
    struct jpeg_error_mgr jerr;

    // 设置错误处理
    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_compress(&cinfo);

    // 指定输出文件
    jpeg_stdio_dest(&cinfo, fp);

    // 设置图像参数
    cinfo.image_width = img->width;
    cinfo.image_height = img->height;
    cinfo.input_components = img->channels;
    cinfo.in_color_space = (img->channels == 1) ? JCS_GRAYSCALE : JCS_RGB;

    // 设置默认压缩参数
    jpeg_set_defaults(&cinfo);
    
    // 设置质量
    jpeg_set_quality(&cinfo, quality, TRUE);

    // 开始压缩
    jpeg_start_compress(&cinfo, TRUE);

    // 逐行写入
    JSAMPROW row_pointer[1];
    while (cinfo.next_scanline < cinfo.image_height) {
        row_pointer[0] = img->data + cinfo.next_scanline * img->stride;
        jpeg_write_scanlines(&cinfo, row_pointer, 1);
    }

    // 完成压缩
    jpeg_finish_compress(&cinfo);
    jpeg_destroy_compress(&cinfo);
    fclose(fp);

    return true;
}

// ============================================================================
// BMP 保存函数
// ============================================================================

/**
 * @brief 保存为 BMP 格式
 */
static bool save_bmp(const Image *img, const char *filename)
{
    if (!img || !filename) {
        return false;
    }

    // BMP只支持3或4通道
    if (img->channels != 3 && img->channels != 4) {
        fprintf(stderr, "BMP只支持RGB或RGBA图像\n");
        return false;
    }

    FILE *fp = fopen(filename, "wb");
    if (!fp) {
        fprintf(stderr, "无法创建文件: %s\n", filename);
        return false;
    }

    int channels = img->channels;
    int row_size = ((img->width * channels * 8 + 31) / 32) * 4;
    int pixel_data_size = row_size * img->height;

    // 填充文件头
    BMPFileHeader file_header;
    file_header.type = 0x4D42; // 'BM'
    file_header.size = sizeof(BMPFileHeader) + sizeof(BMPInfoHeader) + pixel_data_size;
    file_header.reserved1 = 0;
    file_header.reserved2 = 0;
    file_header.offset = sizeof(BMPFileHeader) + sizeof(BMPInfoHeader);

    // 填充信息头
    BMPInfoHeader info_header;
    info_header.size = sizeof(BMPInfoHeader);
    info_header.width = img->width;
    info_header.height = img->height; // 正值表示从下到上
    info_header.planes = 1;
    info_header.bit_count = channels * 8;
    info_header.compression = 0; // BI_RGB
    info_header.size_image = pixel_data_size;
    info_header.x_pels_per_meter = 0;
    info_header.y_pels_per_meter = 0;
    info_header.clr_used = 0;
    info_header.clr_important = 0;

    // 写入文件头
    if (fwrite(&file_header, sizeof(BMPFileHeader), 1, fp) != 1) {
        fprintf(stderr, "写入BMP文件头失败\n");
        fclose(fp);
        return false;
    }

    // 写入信息头
    if (fwrite(&info_header, sizeof(BMPInfoHeader), 1, fp) != 1) {
        fprintf(stderr, "写入BMP信息头失败\n");
        fclose(fp);
        return false;
    }

    // 分配行缓冲区
    unsigned char *row_buffer = (unsigned char*)calloc(1, row_size);
    if (!row_buffer) {
        fprintf(stderr, "分配行缓冲区失败\n");
        fclose(fp);
        return false;
    }

    // 写入像素数据（从下到上，RGB转BGR）
    for (int y = img->height - 1; y >= 0; y--) {
        unsigned char *src = img->data + y * img->stride;
        
        for (int x = 0; x < img->width; x++) {
            if (channels == 3) {
                row_buffer[x * 3 + 0] = src[x * 3 + 2]; // B
                row_buffer[x * 3 + 1] = src[x * 3 + 1]; // G
                row_buffer[x * 3 + 2] = src[x * 3 + 0]; // R
            } else {
                row_buffer[x * 4 + 0] = src[x * 4 + 2]; // B
                row_buffer[x * 4 + 1] = src[x * 4 + 1]; // G
                row_buffer[x * 4 + 2] = src[x * 4 + 0]; // R
                row_buffer[x * 4 + 3] = src[x * 4 + 3]; // A
            }
        }

        if (fwrite(row_buffer, 1, row_size, fp) != row_size) {
            fprintf(stderr, "写入BMP像素数据失败\n");
            free(row_buffer);
            fclose(fp);
            return false;
        }
    }

    free(row_buffer);
    fclose(fp);
    return true;
}

// ============================================================================
// PPM/PGM 保存函数
// ============================================================================

/**
 * @brief 保存为 PPM 格式
 */
static bool save_ppm(const Image *img, const char *filename)
{
    if (!img || !filename) {
        return false;
    }

    if (img->channels != 3) {
        fprintf(stderr, "PPM只支持RGB图像\n");
        return false;
    }

    FILE *fp = fopen(filename, "wb");
    if (!fp) {
        fprintf(stderr, "无法创建文件: %s\n", filename);
        return false;
    }

    // 写入PPM头
    fprintf(fp, "P6\n%d %d\n255\n", img->width, img->height);

    // 写入像素数据
    size_t data_size = img->width * img->height * 3;
    if (fwrite(img->data, 1, data_size, fp) != data_size) {
        fprintf(stderr, "写入PPM像素数据失败\n");
        fclose(fp);
        return false;
    }

    fclose(fp);
    return true;
}

/**
 * @brief 保存为 PGM 格式
 */
static bool save_pgm(const Image *img, const char *filename)
{
    if (!img || !filename) {
        return false;
    }

    if (img->channels != 1) {
        fprintf(stderr, "PGM只支持灰度图像\n");
        return false;
    }

    FILE *fp = fopen(filename, "wb");
    if (!fp) {
        fprintf(stderr, "无法创建文件: %s\n", filename);
        return false;
    }

    // 写入PGM头
    fprintf(fp, "P5\n%d %d\n255\n", img->width, img->height);

    // 写入像素数据
    size_t data_size = img->width * img->height;
    if (fwrite(img->data, 1, data_size, fp) != data_size) {
        fprintf(stderr, "写入PGM像素数据失败\n");
        fclose(fp);
        return false;
    }

    fclose(fp);
    return true;
}

// ============================================================================
// TIFF 保存函数
// ============================================================================

/**
 * @brief 保存为 TIFF 格式
 */
static bool save_tiff(const Image *img, const char *filename)
{
    if (!img || !filename) {
        return false;
    }

    TIFF *tif = TIFFOpen(filename, "w");
    if (!tif) {
        fprintf(stderr, "无法创建TIFF文件: %s\n", filename);
        return false;
    }

    // 设置TIFF标签
    TIFFSetField(tif, TIFFTAG_IMAGEWIDTH, img->width);
    TIFFSetField(tif, TIFFTAG_IMAGELENGTH, img->height);
    TIFFSetField(tif, TIFFTAG_SAMPLESPERPIXEL, img->channels);
    TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE, 8);
    TIFFSetField(tif, TIFFTAG_ORIENTATION, ORIENTATION_TOPLEFT);
    TIFFSetField(tif, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);

    // 设置光度解释
    uint16_t photometric;
    if (img->channels == 1) {
        photometric = PHOTOMETRIC_MINISBLACK;
    } else if (img->channels == 3 || img->channels == 4) {
        photometric = PHOTOMETRIC_RGB;
    } else {
        fprintf(stderr, "不支持的通道数: %d\n", img->channels);
        TIFFClose(tif);
        return false;
    }
    TIFFSetField(tif, TIFFTAG_PHOTOMETRIC, photometric);

    // 如果有alpha通道
    if (img->channels == 2 || img->channels == 4) {
        uint16_t extra_samples = EXTRASAMPLE_ASSOCALPHA;
        TIFFSetField(tif, TIFFTAG_EXTRASAMPLES, 1, &extra_samples);
    }

    // 设置压缩
    TIFFSetField(tif, TIFFTAG_COMPRESSION, COMPRESSION_LZW);

    // 逐行写入
    for (int y = 0; y < img->height; y++) {
        if (TIFFWriteScanline(tif, img->data + y * img->stride, y, 0) < 0) {
            fprintf(stderr, "写入TIFF扫描线失败\n");
            TIFFClose(tif);
            return false;
        }
    }

    TIFFClose(tif);
    return true;
}

// ============================================================================
// WebP 保存函数
// ============================================================================

/**
 * @brief 保存为 WebP 格式
 */
static bool save_webp(const Image *img, const char *filename, float quality)
{
    if (!img || !filename) {
        return false;
    }

    // WebP支持RGB和RGBA
    if (img->channels != 3 && img->channels != 4) {
        fprintf(stderr, "WebP只支持RGB或RGBA图像\n");
        return false;
    }

    uint8_t *output = NULL;
    size_t output_size;

    // 编码WebP
    if (img->channels == 3) {
        output_size = WebPEncodeRGB(img->data, img->width, img->height, 
                                    img->stride, quality, &output);
    } else {
        output_size = WebPEncodeRGBA(img->data, img->width, img->height, 
                                     img->stride, quality, &output);
    }

    if (output_size == 0) {
        fprintf(stderr, "WebP编码失败\n");
        return false;
    }

    // 写入文件
    FILE *fp = fopen(filename, "wb");
    if (!fp) {
        fprintf(stderr, "无法创建文件: %s\n", filename);
        WebPFree(output);
        return false;
    }

    if (fwrite(output, 1, output_size, fp) != output_size) {
        fprintf(stderr, "写入WebP文件失败\n");
        WebPFree(output);
        fclose(fp);
        return false;
    }

    WebPFree(output);
    fclose(fp);
    return true;
}

// ============================================================================
// 统一保存接口
// ============================================================================

/**
 * @brief 保存图像（根据文件扩展名自动选择格式）
 */
bool image_save(const Image *img, const char *filename)
{
    if (!img || !filename) {
        return false;
    }

    ImageFormat format = image_get_format_from_extension(filename);
    if (format == IMAGE_FORMAT_UNKNOWN) {
        fprintf(stderr, "无法从文件名确定保存格式: %s\n", filename);
        return false;
    }

    return image_save_format(img, filename, format);
}

/**
 * @brief 保存为指定格式
 */
bool image_save_format(const Image *img, const char *filename, ImageFormat format)
{
    if (!img || !filename) {
        return false;
    }

    bool success = false;
    switch (format) {
        case IMAGE_FORMAT_PNG:
            success = save_png(img, filename);
            break;
        case IMAGE_FORMAT_JPEG:
            success = save_jpeg(img, filename, 95); // 默认质量95
            break;
        case IMAGE_FORMAT_BMP:
            success = save_bmp(img, filename);
            break;
        case IMAGE_FORMAT_TIFF:
            success = save_tiff(img, filename);
            break;
        case IMAGE_FORMAT_PPM:
            success = save_ppm(img, filename);
            break;
        case IMAGE_FORMAT_PGM:
            success = save_pgm(img, filename);
            break;
        case IMAGE_FORMAT_WEBP:
            success = save_webp(img, filename, 90.0f); // 默认质量90
            break;
        default:
            fprintf(stderr, "不支持的保存格式\n");
            return false;
    }

    if (success) {
        printf("成功保存图像: %s\n", filename);
    }

    return success;
}

/**
 * @brief 保存JPEG（指定质量）
 */
bool image_save_jpeg_quality(const Image *img, const char *filename, int quality)
{
    if (!img || !filename) {
        return false;
    }

    if (quality < 0 || quality > 100) {
        fprintf(stderr, "JPEG质量必须在0-100之间\n");
        return false;
    }

    return save_jpeg(img, filename, quality);
}

/**
 * @brief 保存WebP（指定质量）
 */
bool image_save_webp_quality(const Image *img, const char *filename, float quality)
{
    if (!img || !filename) {
        return false;
    }

    if (quality < 0.0f || quality > 100.0f) {
        fprintf(stderr, "WebP质量必须在0-100之间\n");
        return false;
    }

    return save_webp(img, filename, quality);
}

// image_loader.c 完成

