/**
 * @file jpeg_handler.c
 * @brief JPEG图像处理模块实现
 * @author Your Name
 * @date 2024
 */

#include "jpeg_handler.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <jpeglib.h>
#include <jerror.h>
#include <setjmp.h>

// ============================================================================
// 内部结构体和全局变量
// ============================================================================

/**
 * @brief 错误处理结构
 */
typedef struct {
    struct jpeg_error_mgr pub;  /**< 公共错误管理器 */
    jmp_buf setjmp_buffer;      /**< 错误跳转缓冲区 */
    char message[JMSG_LENGTH_MAX]; /**< 错误消息 */
} JpegErrorManager;

/**
 * @brief JPEG处理器内部结构
 */
struct JpegProcessor {
    bool initialized;           /**< 是否已初始化 */
    int error_count;           /**< 错误计数 */
    char last_error[256];      /**< 最后的错误消息 */
};

/** 全局JPEG处理器实例 */
static JpegProcessor g_jpeg_processor = {
    .initialized = false,
    .error_count = 0,
    .last_error = {0}
};

// ============================================================================
// 错误处理
// ============================================================================

/**
 * @brief JPEG错误退出处理
 */
static void jpeg_error_exit(j_common_ptr cinfo)
{
    JpegErrorManager *err = (JpegErrorManager*)cinfo->err;
    
    // 格式化错误消息
    (*cinfo->err->format_message)(cinfo, err->message);
    
    // 记录错误
    fprintf(stderr, "JPEG错误: %s\n", err->message);
    snprintf(g_jpeg_processor.last_error, sizeof(g_jpeg_processor.last_error),
             "%s", err->message);
    g_jpeg_processor.error_count++;
    
    // 跳转回安全点
    longjmp(err->setjmp_buffer, 1);
}

/**
 * @brief JPEG输出消息处理
 */
static void jpeg_output_message(j_common_ptr cinfo)
{
    char buffer[JMSG_LENGTH_MAX];
    
    // 格式化消息
    (*cinfo->err->format_message)(cinfo, buffer);
    
    // 输出警告消息
    fprintf(stderr, "JPEG警告: %s\n", buffer);
}

/**
 * @brief 设置JPEG错误管理器
 */
static void jpeg_setup_error_manager(JpegErrorManager *err)
{
    // 设置标准错误处理
    jpeg_std_error(&err->pub);
    
    // 覆盖错误处理函数
    err->pub.error_exit = jpeg_error_exit;
    err->pub.output_message = jpeg_output_message;
    
    // 清空错误消息
    memset(err->message, 0, sizeof(err->message));
}

// ============================================================================
// 初始化和清理
// ============================================================================

/**
 * @brief 初始化JPEG处理器
 */
bool jpeg_init(void)
{
    if (g_jpeg_processor.initialized) {
        return true;
    }
    
    printf("初始化JPEG处理器...\n");
    
    // 重置错误计数
    g_jpeg_processor.error_count = 0;
    memset(g_jpeg_processor.last_error, 0, sizeof(g_jpeg_processor.last_error));
    
    g_jpeg_processor.initialized = true;
    
    printf("JPEG处理器初始化完成\n");
    printf("libjpeg版本: %d\n", JPEG_LIB_VERSION);
    
    return true;
}

/**
 * @brief 清理JPEG处理器
 */
void jpeg_cleanup(void)
{
    if (!g_jpeg_processor.initialized) {
        return;
    }
    
    printf("清理JPEG处理器...\n");
    
    if (g_jpeg_processor.error_count > 0) {
        printf("处理过程中发生了 %d 个错误\n", g_jpeg_processor.error_count);
    }
    
    g_jpeg_processor.initialized = false;
    
    printf("JPEG处理器清理完成\n");
}

/**
 * @brief 获取JPEG处理器版本
 */
const char* jpeg_get_version(void)
{
    return JPEG_HANDLER_VERSION;
}

/**
 * @brief 检查是否支持JPEG
 */
bool jpeg_is_supported(void)
{
    return true; // libjpeg总是可用的
}

// ============================================================================
// 选项创建
// ============================================================================

/**
 * @brief 创建默认压缩选项
 */
JpegCompressOptions jpeg_create_default_compress_options(void)
{
    JpegCompressOptions options;
    
    options.quality = JPEG_DEFAULT_QUALITY;
    options.progressive = false;
    options.optimize_coding = true;
    options.subsampling = JPEG_SUBSAMPLE_420;
    options.dct_method = JPEG_DCT_ISLOW;
    options.smoothing_factor = 0;
    options.arithmetic_coding = false;
    options.restart_interval = 0;
    
    return options;
}

/**
 * @brief 创建默认解压缩选项
 */
JpegDecompressOptions jpeg_create_default_decompress_options(void)
{
    JpegDecompressOptions options;
    
    options.do_fancy_upsampling = true;
    options.do_block_smoothing = true;
    options.dct_method = JPEG_DCT_ISLOW;
    options.two_pass_quantize = true;
    options.dither_mode = JDITHER_FS;
    options.scale_num = 1;
    options.scale_denom = 1;
    options.buffered_image = false;
    
    return options;
}

/**
 * @brief 创建高质量压缩选项
 */
JpegCompressOptions jpeg_create_high_quality_options(void)
{
    JpegCompressOptions options = jpeg_create_default_compress_options();
    
    options.quality = 95;
    options.progressive = true;
    options.optimize_coding = true;
    options.subsampling = JPEG_SUBSAMPLE_444; // 无子采样
    options.dct_method = JPEG_DCT_ISLOW;
    options.smoothing_factor = 0;
    
    return options;
}

/**
 * @brief 创建低质量压缩选项
 */
JpegCompressOptions jpeg_create_low_quality_options(void)
{
    JpegCompressOptions options = jpeg_create_default_compress_options();
    
    options.quality = 60;
    options.progressive = false;
    options.optimize_coding = false;
    options.subsampling = JPEG_SUBSAMPLE_420;
    options.dct_method = JPEG_DCT_IFAST;
    options.smoothing_factor = 10;
    
    return options;
}

/**
 * @brief 创建渐进式JPEG选项
 */
JpegCompressOptions jpeg_create_progressive_options(void)
{
    JpegCompressOptions options = jpeg_create_default_compress_options();
    
    options.progressive = true;
    options.optimize_coding = true;
    
    return options;
}

// ============================================================================
// 内部辅助函数
// ============================================================================

/**
 * @brief 转换libjpeg色彩空间到内部枚举
 */
static JpegColorSpace convert_jpeg_colorspace(J_COLOR_SPACE space)
{
    switch (space) {
        case JCS_GRAYSCALE:
            return JPEG_COLORSPACE_GRAYSCALE;
        case JCS_RGB:
            return JPEG_COLORSPACE_RGB;
        case JCS_YCbCr:
            return JPEG_COLORSPACE_YCbCr;
        case JCS_CMYK:
            return JPEG_COLORSPACE_CMYK;
        case JCS_YCCK:
            return JPEG_COLORSPACE_YCCK;
        default:
            return JPEG_COLORSPACE_UNKNOWN;
    }
}

/**
 * @brief 转换内部色彩空间枚举到libjpeg
 */
static J_COLOR_SPACE convert_to_jpeg_colorspace(JpegColorSpace space)
{
    switch (space) {
        case JPEG_COLORSPACE_GRAYSCALE:
            return JCS_GRAYSCALE;
        case JPEG_COLORSPACE_RGB:
            return JCS_RGB;
        case JPEG_COLORSPACE_YCbCr:
            return JCS_YCbCr;
        case JPEG_COLORSPACE_CMYK:
            return JCS_CMYK;
        case JPEG_COLORSPACE_YCCK:
            return JCS_YCCK;
        default:
            return JCS_UNKNOWN;
    }
}

/**
 * @brief 检测子采样模式
 */
static JpegSubsampling detect_subsampling(struct jpeg_decompress_struct *cinfo)
{
    if (cinfo->num_components < 3) {
        return JPEG_SUBSAMPLE_444; // 灰度图像
    }
    
    int h_samp_0 = cinfo->comp_info[0].h_samp_factor;
    int v_samp_0 = cinfo->comp_info[0].v_samp_factor;
    int h_samp_1 = cinfo->comp_info[1].h_samp_factor;
    int v_samp_1 = cinfo->comp_info[1].v_samp_factor;
    
    if (h_samp_0 == 2 && v_samp_0 == 2 && h_samp_1 == 1 && v_samp_1 == 1) {
        return JPEG_SUBSAMPLE_420; // 4:2:0
    } else if (h_samp_0 == 2 && v_samp_0 == 1 && h_samp_1 == 1 && v_samp_1 == 1) {
        return JPEG_SUBSAMPLE_422; // 4:2:2
    } else if (h_samp_0 == 1 && v_samp_0 == 1 && h_samp_1 == 1 && v_samp_1 == 1) {
        return JPEG_SUBSAMPLE_444; // 4:4:4
    } else if (h_samp_0 == 4 && v_samp_0 == 1 && h_samp_1 == 1 && v_samp_1 == 1) {
        return JPEG_SUBSAMPLE_411; // 4:1:1
    } else if (h_samp_0 == 1 && v_samp_0 == 2 && h_samp_1 == 1 && v_samp_1 == 1) {
        return JPEG_SUBSAMPLE_440; // 4:4:0
    }
    
    return JPEG_SUBSAMPLE_444; // 默认
}

/**
 * @brief 设置子采样模式
 */
static void set_subsampling(struct jpeg_compress_struct *cinfo, JpegSubsampling subsampling)
{
    if (cinfo->num_components < 3) {
        return; // 灰度图像不需要子采样
    }
    
    switch (subsampling) {
        case JPEG_SUBSAMPLE_444:
            // 4:4:4 - 无子采样
            cinfo->comp_info[0].h_samp_factor = 1;
            cinfo->comp_info[0].v_samp_factor = 1;
            cinfo->comp_info[1].h_samp_factor = 1;
            cinfo->comp_info[1].v_samp_factor = 1;
            cinfo->comp_info[2].h_samp_factor = 1;
            cinfo->comp_info[2].v_samp_factor = 1;
            break;
            
        case JPEG_SUBSAMPLE_422:
            // 4:2:2 - 水平2倍子采样
            cinfo->comp_info[0].h_samp_factor = 2;
            cinfo->comp_info[0].v_samp_factor = 1;
            cinfo->comp_info[1].h_samp_factor = 1;
            cinfo->comp_info[1].v_samp_factor = 1;
            cinfo->comp_info[2].h_samp_factor = 1;
            cinfo->comp_info[2].v_samp_factor = 1;
            break;
            
        case JPEG_SUBSAMPLE_420:
            // 4:2:0 - 水平和垂直2倍子采样
            cinfo->comp_info[0].h_samp_factor = 2;
            cinfo->comp_info[0].v_samp_factor = 2;
            cinfo->comp_info[1].h_samp_factor = 1;
            cinfo->comp_info[1].v_samp_factor = 1;
            cinfo->comp_info[2].h_samp_factor = 1;
            cinfo->comp_info[2].v_samp_factor = 1;
            break;
            
        case JPEG_SUBSAMPLE_411:
            // 4:1:1 - 水平4倍子采样
            cinfo->comp_info[0].h_samp_factor = 4;
            cinfo->comp_info[0].v_samp_factor = 1;
            cinfo->comp_info[1].h_samp_factor = 1;
            cinfo->comp_info[1].v_samp_factor = 1;
            cinfo->comp_info[2].h_samp_factor = 1;
            cinfo->comp_info[2].v_samp_factor = 1;
            break;
            
        case JPEG_SUBSAMPLE_440:
            // 4:4:0 - 垂直2倍子采样
            cinfo->comp_info[0].h_samp_factor = 1;
            cinfo->comp_info[0].v_samp_factor = 2;
            cinfo->comp_info[1].h_samp_factor = 1;
            cinfo->comp_info[1].v_samp_factor = 1;
            cinfo->comp_info[2].h_samp_factor = 1;
            cinfo->comp_info[2].v_samp_factor = 1;
            break;
    }
}

/**
 * @brief 转换DCT方法
 */
static J_DCT_METHOD convert_dct_method(JpegDctMethod method)
{
    switch (method) {
        case JPEG_DCT_ISLOW:
            return JDCT_ISLOW;
        case JPEG_DCT_IFAST:
            return JDCT_IFAST;
        case JPEG_DCT_FLOAT:
            return JDCT_FLOAT;
        default:
            return JDCT_ISLOW;
    }
}

/**
 * @brief 验证质量参数
 */
static int validate_quality(int quality)
{
    if (quality < JPEG_MIN_QUALITY) {
        fprintf(stderr, "警告: 质量值 %d 太低，使用最小值 %d\n", 
                quality, JPEG_MIN_QUALITY);
        return JPEG_MIN_QUALITY;
    }
    
    if (quality > JPEG_MAX_QUALITY) {
        fprintf(stderr, "警告: 质量值 %d 太高，使用最大值 %d\n", 
                quality, JPEG_MAX_QUALITY);
        return JPEG_MAX_QUALITY;
    }
    
    return quality;
}

/**
 * @brief 应用压缩选项
 */
static void apply_compress_options(struct jpeg_compress_struct *cinfo,
                                   const JpegCompressOptions *options)
{
    if (!options) {
        return;
    }
    
    // 设置质量
    jpeg_set_quality(cinfo, validate_quality(options->quality), TRUE);
    
    // 设置渐进式模式
    if (options->progressive) {
        jpeg_simple_progression(cinfo);
    }
    
    // 设置优化编码
    cinfo->optimize_coding = options->optimize_coding ? TRUE : FALSE;
    
    // 设置DCT方法
    cinfo->dct_method = convert_dct_method(options->dct_method);
    
    // 设置平滑因子
    if (options->smoothing_factor > 0 && options->smoothing_factor <= 100) {
        cinfo->smoothing_factor = options->smoothing_factor;
    }
    
    // 设置算术编码 (如果支持)
#ifdef C_ARITH_CODING_SUPPORTED
    if (options->arithmetic_coding) {
        cinfo->arith_code = TRUE;
    }
#endif
    
    // 设置重启间隔
    if (options->restart_interval > 0) {
        cinfo->restart_interval = options->restart_interval;
    }
    
    // 设置子采样模式
    set_subsampling(cinfo, options->subsampling);
}

/**
 * @brief 应用解压缩选项
 */
static void apply_decompress_options(struct jpeg_decompress_struct *cinfo,
                                     const JpegDecompressOptions *options)
{
    if (!options) {
        return;
    }
    
    // 设置上采样
    cinfo->do_fancy_upsampling = options->do_fancy_upsampling ? TRUE : FALSE;
    
    // 设置块平滑
    cinfo->do_block_smoothing = options->do_block_smoothing ? TRUE : FALSE;
    
    // 设置DCT方法
    cinfo->dct_method = convert_dct_method(options->dct_method);
    
    // 设置量化
    cinfo->two_pass_quantize = options->two_pass_quantize ? TRUE : FALSE;
    cinfo->dither_mode = (J_DITHER_MODE)options->dither_mode;
    
    // 设置缩放
    if (options->scale_denom > 0) {
        cinfo->scale_num = options->scale_num;
        cinfo->scale_denom = options->scale_denom;
    }
    
    // 设置缓冲图像模式
    cinfo->buffered_image = options->buffered_image ? TRUE : FALSE;
}

/**
 * @brief 检查文件是否为JPEG格式
 */
static bool is_jpeg_file(const char *filename)
{
    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        return false;
    }
    
    unsigned char magic[2];
    size_t read = fread(magic, 1, 2, fp);
    fclose(fp);
    
    if (read != 2) {
        return false;
    }
    
    // 检查JPEG魔数 (0xFF 0xD8)
    return (magic[0] == 0xFF && magic[1] == 0xD8);
}

/**
 * @brief 检查内存数据是否为JPEG格式
 */
static bool is_jpeg_memory(const unsigned char *data, size_t size)
{
    if (!data || size < 2) {
        return false;
    }
    
    // 检查JPEG魔数
    return (data[0] == 0xFF && data[1] == 0xD8);
}

/**
 * @brief 获取文件大小
 */
static size_t get_file_size(const char *filename)
{
    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        return 0;
    }
    
    fseek(fp, 0, SEEK_END);
    size_t size = ftell(fp);
    fclose(fp);
    
    return size;
}

// ============================================================================
// JPEG读取实现
// ============================================================================

/**
 * @brief 从文件读取JPEG图像 (内部实现)
 */
static Image* jpeg_read_internal(FILE *fp, const JpegDecompressOptions *options)
{
    if (!fp) {
        return NULL;
    }
    
    struct jpeg_decompress_struct cinfo;
    JpegErrorManager jerr;
    Image *img = NULL;
    JSAMPARRAY buffer = NULL;
    
    // 设置错误处理
    jpeg_setup_error_manager(&jerr);
    cinfo.err = (struct jpeg_error_mgr*)&jerr;
    
    // 设置错误跳转点
    if (setjmp(jerr.setjmp_buffer)) {
        // 发生错误，清理并返回
        jpeg_destroy_decompress(&cinfo);
        if (img) {
            image_destroy(img);
        }
        return NULL;
    }
    
    // 创建解压缩对象
    jpeg_create_decompress(&cinfo);
    
    // 指定数据源
    jpeg_stdio_src(&cinfo, fp);
    
    // 读取JPEG头
    jpeg_read_header(&cinfo, TRUE);
    
    // 应用解压缩选项
    apply_decompress_options(&cinfo, options);
    
    // 开始解压缩
    jpeg_start_decompress(&cinfo);
    
    // 创建图像对象
    int width = cinfo.output_width;
    int height = cinfo.output_height;
    int channels = cinfo.output_components;
    
    img = image_create(width, height, channels);
    if (!img) {
        fprintf(stderr, "创建图像对象失败\n");
        jpeg_destroy_decompress(&cinfo);
        return NULL;
    }
    
    // 分配行缓冲区
    int row_stride = width * channels;
    buffer = (*cinfo.mem->alloc_sarray)
        ((j_common_ptr)&cinfo, JPOOL_IMAGE, row_stride, 1);
    
    // 逐行读取图像数据
    int row = 0;
    while (cinfo.output_scanline < cinfo.output_height) {
        jpeg_read_scanlines(&cinfo, buffer, 1);
        
        // 复制数据到图像对象
        memcpy(img->data + row * row_stride, buffer[0], row_stride);
        row++;
    }
    
    // 完成解压缩
    jpeg_finish_decompress(&cinfo);
    
    // 清理
    jpeg_destroy_decompress(&cinfo);
    
    return img;
}

/**
 * @brief 从文件读取JPEG图像
 */
Image* jpeg_read(const char *filename)
{
    return jpeg_read_with_options(filename, NULL);
}

/**
 * @brief 从文件读取JPEG图像 (带选项)
 */
Image* jpeg_read_with_options(const char *filename, const JpegDecompressOptions *options)
{
    if (!filename) {
        fprintf(stderr, "文件名为空\n");
        return NULL;
    }
    
    // 检查文件格式
    if (!is_jpeg_file(filename)) {
        fprintf(stderr, "不是有效的JPEG文件: %s\n", filename);
        return NULL;
    }
    
    printf("读取JPEG图像: %s\n", filename);
    
    // 打开文件
    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        fprintf(stderr, "无法打开文件: %s\n", filename);
        return NULL;
    }
    
    // 读取图像
    Image *img = jpeg_read_internal(fp, options);
    
    // 关闭文件
    fclose(fp);
    
    if (img) {
        printf("成功读取JPEG图像: %dx%d, %d通道\n", 
               img->width, img->height, img->channels);
    }
    
    return img;
}

// 第一部分结束
// ============================================================================
// JPEG读取实现（续）
// ============================================================================

/**
 * @brief 从内存读取JPEG图像
 */
Image* jpeg_read_from_memory(const unsigned char *data, size_t size)
{
    return jpeg_read_from_memory_with_options(data, size, NULL);
}

/**
 * @brief 从内存读取JPEG图像 (带选项)
 */
Image* jpeg_read_from_memory_with_options(const unsigned char *data, size_t size,
                                          const JpegDecompressOptions *options)
{
    if (!data || size == 0) {
        fprintf(stderr, "内存数据为空\n");
        return NULL;
    }
    
    // 检查JPEG格式
    if (!is_jpeg_memory(data, size)) {
        fprintf(stderr, "不是有效的JPEG数据\n");
        return NULL;
    }
    
    printf("从内存读取JPEG图像 (%zu 字节)\n", size);
    
    struct jpeg_decompress_struct cinfo;
    JpegErrorManager jerr;
    Image *img = NULL;
    JSAMPARRAY buffer = NULL;
    
    // 设置错误处理
    jpeg_setup_error_manager(&jerr);
    cinfo.err = (struct jpeg_error_mgr*)&jerr;
    
    // 设置错误跳转点
    if (setjmp(jerr.setjmp_buffer)) {
        jpeg_destroy_decompress(&cinfo);
        if (img) {
            image_destroy(img);
        }
        return NULL;
    }
    
    // 创建解压缩对象
    jpeg_create_decompress(&cinfo);
    
    // 指定内存数据源
    jpeg_mem_src(&cinfo, data, size);
    
    // 读取JPEG头
    jpeg_read_header(&cinfo, TRUE);
    
    // 应用解压缩选项
    apply_decompress_options(&cinfo, options);
    
    // 开始解压缩
    jpeg_start_decompress(&cinfo);
    
    // 创建图像对象
    int width = cinfo.output_width;
    int height = cinfo.output_height;
    int channels = cinfo.output_components;
    
    img = image_create(width, height, channels);
    if (!img) {
        fprintf(stderr, "创建图像对象失败\n");
        jpeg_destroy_decompress(&cinfo);
        return NULL;
    }
    
    // 分配行缓冲区
    int row_stride = width * channels;
    buffer = (*cinfo.mem->alloc_sarray)
        ((j_common_ptr)&cinfo, JPOOL_IMAGE, row_stride, 1);
    
    // 逐行读取图像数据
    int row = 0;
    while (cinfo.output_scanline < cinfo.output_height) {
        jpeg_read_scanlines(&cinfo, buffer, 1);
        memcpy(img->data + row * row_stride, buffer[0], row_stride);
        row++;
    }
    
    // 完成解压缩
    jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);
    
    printf("成功从内存读取JPEG图像: %dx%d, %d通道\n", 
           img->width, img->height, img->channels);
    
    return img;
}

/**
 * @brief 读取JPEG缩略图
 */
Image* jpeg_read_thumbnail(const char *filename, int scale)
{
    if (!filename) {
        fprintf(stderr, "文件名为空\n");
        return NULL;
    }
    
    // 验证缩放因子
    if (scale != 1 && scale != 2 && scale != 4 && scale != 8) {
        fprintf(stderr, "无效的缩放因子: %d (必须是1, 2, 4或8)\n", scale);
        return NULL;
    }
    
    printf("读取JPEG缩略图: %s (缩放: 1/%d)\n", filename, scale);
    
    // 创建解压缩选项
    JpegDecompressOptions options = jpeg_create_default_decompress_options();
    options.scale_num = 1;
    options.scale_denom = scale;
    options.do_fancy_upsampling = false; // 快速模式
    options.do_block_smoothing = false;
    options.dct_method = JPEG_DCT_IFAST;
    
    return jpeg_read_with_options(filename, &options);
}

// ============================================================================
// JPEG写入实现
// ============================================================================

/**
 * @brief 写入JPEG图像到文件 (内部实现)
 */
static bool jpeg_write_internal(const Image *img, FILE *fp, 
                                const JpegCompressOptions *options)
{
    if (!img || !fp) {
        return false;
    }
    
    struct jpeg_compress_struct cinfo;
    JpegErrorManager jerr;
    JSAMPROW row_pointer[1];
    
    // 设置错误处理
    jpeg_setup_error_manager(&jerr);
    cinfo.err = (struct jpeg_error_mgr*)&jerr;
    
    // 设置错误跳转点
    if (setjmp(jerr.setjmp_buffer)) {
        jpeg_destroy_compress(&cinfo);
        return false;
    }
    
    // 创建压缩对象
    jpeg_create_compress(&cinfo);
    
    // 指定输出目标
    jpeg_stdio_dest(&cinfo, fp);
    
    // 设置图像参数
    cinfo.image_width = img->width;
    cinfo.image_height = img->height;
    cinfo.input_components = img->channels;
    
    // 设置色彩空间
    if (img->channels == 1) {
        cinfo.in_color_space = JCS_GRAYSCALE;
    } else if (img->channels == 3) {
        cinfo.in_color_space = JCS_RGB;
    } else if (img->channels == 4) {
        // RGBA需要转换为RGB
        fprintf(stderr, "警告: RGBA图像将转换为RGB\n");
        cinfo.in_color_space = JCS_RGB;
        cinfo.input_components = 3;
    } else {
        fprintf(stderr, "不支持的通道数: %d\n", img->channels);
        jpeg_destroy_compress(&cinfo);
        return false;
    }
    
    // 设置默认压缩参数
    jpeg_set_defaults(&cinfo);
    
    // 应用压缩选项
    apply_compress_options(&cinfo, options);
    
    // 开始压缩
    jpeg_start_compress(&cinfo, TRUE);
    
    // 逐行写入图像数据
    int row_stride = img->width * img->channels;
    
    if (img->channels == 4) {
        // RGBA转RGB
        unsigned char *rgb_row = (unsigned char*)malloc(img->width * 3);
        if (!rgb_row) {
            fprintf(stderr, "内存分配失败\n");
            jpeg_destroy_compress(&cinfo);
            return false;
        }
        
        while (cinfo.next_scanline < cinfo.image_height) {
            // 转换RGBA到RGB
            for (int x = 0; x < img->width; x++) {
                int src_idx = cinfo.next_scanline * row_stride + x * 4;
                int dst_idx = x * 3;
                rgb_row[dst_idx + 0] = img->data[src_idx + 0];
                rgb_row[dst_idx + 1] = img->data[src_idx + 1];
                rgb_row[dst_idx + 2] = img->data[src_idx + 2];
            }
            
            row_pointer[0] = rgb_row;
            jpeg_write_scanlines(&cinfo, row_pointer, 1);
        }
        
        free(rgb_row);
    } else {
        // 直接写入
        while (cinfo.next_scanline < cinfo.image_height) {
            row_pointer[0] = &img->data[cinfo.next_scanline * row_stride];
            jpeg_write_scanlines(&cinfo, row_pointer, 1);
        }
    }
    
    // 完成压缩
    jpeg_finish_compress(&cinfo);
    jpeg_destroy_compress(&cinfo);
    
    return true;
}

/**
 * @brief 写入JPEG图像到文件
 */
bool jpeg_write(const Image *img, const char *filename, int quality)
{
    if (!img || !filename) {
        fprintf(stderr, "参数无效\n");
        return false;
    }
    
    // 创建压缩选项
    JpegCompressOptions options = jpeg_create_default_compress_options();
    options.quality = validate_quality(quality);
    
    return jpeg_write_with_options(img, filename, &options);
}

/**
 * @brief 写入JPEG图像到文件 (带选项)
 */
bool jpeg_write_with_options(const Image *img, const char *filename,
                             const JpegCompressOptions *options)
{
    if (!img || !filename) {
        fprintf(stderr, "参数无效\n");
        return false;
    }
    
    printf("写入JPEG图像: %s\n", filename);
    if (options) {
        printf("  质量: %d\n", options->quality);
        printf("  渐进式: %s\n", options->progressive ? "是" : "否");
        printf("  优化编码: %s\n", options->optimize_coding ? "是" : "否");
    }
    
    // 打开文件
    FILE *fp = fopen(filename, "wb");
    if (!fp) {
        fprintf(stderr, "无法创建文件: %s\n", filename);
        return false;
    }
    
    // 写入图像
    bool result = jpeg_write_internal(img, fp, options);
    
    // 关闭文件
    fclose(fp);
    
    if (result) {
        size_t file_size = get_file_size(filename);
        printf("成功写入JPEG图像: %zu 字节\n", file_size);
    }
    
    return result;
}

/**
 * @brief 写入JPEG图像到内存
 */
bool jpeg_write_to_memory(const Image *img, unsigned char **data, size_t *size, int quality)
{
    if (!img || !data || !size) {
        fprintf(stderr, "参数无效\n");
        return false;
    }
    
    // 创建压缩选项
    JpegCompressOptions options = jpeg_create_default_compress_options();
    options.quality = validate_quality(quality);
    
    return jpeg_write_to_memory_with_options(img, data, size, &options);
}

/**
 * @brief 写入JPEG图像到内存 (带选项)
 */
bool jpeg_write_to_memory_with_options(const Image *img, unsigned char **data, size_t *size,
                                       const JpegCompressOptions *options)
{
    if (!img || !data || !size) {
        fprintf(stderr, "参数无效\n");
        return false;
    }
    
    printf("写入JPEG图像到内存\n");
    
    struct jpeg_compress_struct cinfo;
    JpegErrorManager jerr;
    JSAMPROW row_pointer[1];
    unsigned char *mem_buffer = NULL;
    unsigned long mem_size = 0;
    
    // 设置错误处理
    jpeg_setup_error_manager(&jerr);
    cinfo.err = (struct jpeg_error_mgr*)&jerr;
    
    // 设置错误跳转点
    if (setjmp(jerr.setjmp_buffer)) {
        jpeg_destroy_compress(&cinfo);
        if (mem_buffer) {
            free(mem_buffer);
        }
        return false;
    }
    
    // 创建压缩对象
    jpeg_create_compress(&cinfo);
    
    // 指定内存输出目标
    jpeg_mem_dest(&cinfo, &mem_buffer, &mem_size);
    
    // 设置图像参数
    cinfo.image_width = img->width;
    cinfo.image_height = img->height;
    cinfo.input_components = img->channels;
    
    if (img->channels == 1) {
        cinfo.in_color_space = JCS_GRAYSCALE;
    } else if (img->channels == 3) {
        cinfo.in_color_space = JCS_RGB;
    } else if (img->channels == 4) {
        cinfo.in_color_space = JCS_RGB;
        cinfo.input_components = 3;
    } else {
        fprintf(stderr, "不支持的通道数: %d\n", img->channels);
        jpeg_destroy_compress(&cinfo);
        return false;
    }
    
    // 设置默认压缩参数
    jpeg_set_defaults(&cinfo);
    
    // 应用压缩选项
    apply_compress_options(&cinfo, options);
    
    // 开始压缩
    jpeg_start_compress(&cinfo, TRUE);
    
    // 逐行写入图像数据
    int row_stride = img->width * img->channels;
    
    if (img->channels == 4) {
        // RGBA转RGB
        unsigned char *rgb_row = (unsigned char*)malloc(img->width * 3);
        if (!rgb_row) {
            fprintf(stderr, "内存分配失败\n");
            jpeg_destroy_compress(&cinfo);
            if (mem_buffer) free(mem_buffer);
            return false;
        }
        
        while (cinfo.next_scanline < cinfo.image_height) {
            for (int x = 0; x < img->width; x++) {
                int src_idx = cinfo.next_scanline * row_stride + x * 4;
                int dst_idx = x * 3;
                rgb_row[dst_idx + 0] = img->data[src_idx + 0];
                rgb_row[dst_idx + 1] = img->data[src_idx + 1];
                rgb_row[dst_idx + 2] = img->data[src_idx + 2];
            }
            
            row_pointer[0] = rgb_row;
            jpeg_write_scanlines(&cinfo, row_pointer, 1);
        }
        
        free(rgb_row);
    } else {
        while (cinfo.next_scanline < cinfo.image_height) {
            row_pointer[0] = &img->data[cinfo.next_scanline * row_stride];
            jpeg_write_scanlines(&cinfo, row_pointer, 1);
        }
    }
    
    // 完成压缩
    jpeg_finish_compress(&cinfo);
    jpeg_destroy_compress(&cinfo);
    
    // 返回结果
    *data = mem_buffer;
    *size = mem_size;
    
    printf("成功写入JPEG到内存: %zu 字节\n", *size);
    
    return true;
}

// ============================================================================
// JPEG信息获取
// ============================================================================

/**
 * @brief 获取JPEG图像信息 (不解码图像数据)
 */
bool jpeg_get_info(const char *filename, JpegInfo *info)
{
    if (!filename || !info) {
        fprintf(stderr, "参数无效\n");
        return false;
    }
    
    // 检查文件格式
    if (!is_jpeg_file(filename)) {
        fprintf(stderr, "不是有效的JPEG文件: %s\n", filename);
        return false;
    }
    
    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        fprintf(stderr, "无法打开文件: %s\n", filename);
        return false;
    }
    
    struct jpeg_decompress_struct cinfo;
    JpegErrorManager jerr;
    
    // 设置错误处理
    jpeg_setup_error_manager(&jerr);
    cinfo.err = (struct jpeg_error_mgr*)&jerr;
    
    // 设置错误跳转点
    if (setjmp(jerr.setjmp_buffer)) {
        jpeg_destroy_decompress(&cinfo);
        fclose(fp);
        return false;
    }
    
    // 创建解压缩对象
    jpeg_create_decompress(&cinfo);
    
    // 指定数据源
    jpeg_stdio_src(&cinfo, fp);
    
    // 保存标记数据
    jpeg_save_markers(&cinfo, JPEG_COM, 0xFFFF);
    jpeg_save_markers(&cinfo, JPEG_APP0, 0xFFFF);
    jpeg_save_markers(&cinfo, JPEG_APP0 + 1, 0xFFFF);
    
    // 读取JPEG头
    jpeg_read_header(&cinfo, TRUE);
    
    // 填充信息结构
    memset(info, 0, sizeof(JpegInfo));
    
    info->width = cinfo.image_width;
    info->height = cinfo.image_height;
    info->channels = cinfo.num_components;
    info->color_space = convert_jpeg_colorspace(cinfo.jpeg_color_space);
    info->subsampling = detect_subsampling(&cinfo);
    info->progressive = (cinfo.progressive_mode == TRUE);
    info->file_size = get_file_size(filename);
    
    // 检查EXIF
    jpeg_saved_marker_ptr marker = cinfo.marker_list;
    while (marker) {
        if (marker->marker == JPEG_APP0 + 1) {
            // EXIF标记
            if (marker->data_length > 6 &&
                memcmp(marker->data, "Exif\0\0", 6) == 0) {
                info->has_exif = true;
            }
        } else if (marker->marker == JPEG_COM) {
            // 注释
            size_t len = marker->data_length;
            if (len > JPEG_MAX_COMMENT_LENGTH - 1) {
                len = JPEG_MAX_COMMENT_LENGTH - 1;
            }
            memcpy(info->comment, marker->data, len);
            info->comment[len] = '\0';
        }
        marker = marker->next;
    }
    
    // 估计质量 (基于量化表)
    info->quality = jpeg_estimate_quality(filename);
    
    // 清理
    jpeg_destroy_decompress(&cinfo);
    fclose(fp);
    
    return true;
}

/**
 * @brief 从内存获取JPEG图像信息
 */
bool jpeg_get_info_from_memory(const unsigned char *data, size_t size, JpegInfo *info)
{
    if (!data || size == 0 || !info) {
        fprintf(stderr, "参数无效\n");
        return false;
    }
    
    // 检查JPEG格式
    if (!is_jpeg_memory(data, size)) {
        fprintf(stderr, "不是有效的JPEG数据\n");
        return false;
    }
    
    struct jpeg_decompress_struct cinfo;
    JpegErrorManager jerr;
    
    // 设置错误处理
    jpeg_setup_error_manager(&jerr);
    cinfo.err = (struct jpeg_error_mgr*)&jerr;
    
    // 设置错误跳转点
    if (setjmp(jerr.setjmp_buffer)) {
        jpeg_destroy_decompress(&cinfo);
        return false;
    }
    
    // 创建解压缩对象
    jpeg_create_decompress(&cinfo);
    
    // 指定内存数据源
    jpeg_mem_src(&cinfo, data, size);
    
    // 保存标记数据
    jpeg_save_markers(&cinfo, JPEG_COM, 0xFFFF);
    jpeg_save_markers(&cinfo, JPEG_APP0, 0xFFFF);
    jpeg_save_markers(&cinfo, JPEG_APP0 + 1, 0xFFFF);
    
    // 读取JPEG头
    jpeg_read_header(&cinfo, TRUE);
    
    // 填充信息结构
    memset(info, 0, sizeof(JpegInfo));
    
    info->width = cinfo.image_width;
    info->height = cinfo.image_height;
    info->channels = cinfo.num_components;
    info->color_space = convert_jpeg_colorspace(cinfo.jpeg_color_space);
    info->subsampling = detect_subsampling(&cinfo);
    info->progressive = (cinfo.progressive_mode == TRUE);
    info->file_size = size;
    
    // 检查EXIF
    jpeg_saved_marker_ptr marker = cinfo.marker_list;
    while (marker) {
        if (marker->marker == JPEG_APP0 + 1) {
            if (marker->data_length > 6 &&
                memcmp(marker->data, "Exif\0\0", 6) == 0) {
                info->has_exif = true;
            }
        } else if (marker->marker == JPEG_COM) {
            size_t len = marker->data_length;
            if (len > JPEG_MAX_COMMENT_LENGTH - 1) {
                len = JPEG_MAX_COMMENT_LENGTH - 1;
            }
            memcpy(info->comment, marker->data, len);
            info->comment[len] = '\0';
        }
        marker = marker->next;
    }
    
    // 清理
    jpeg_destroy_decompress(&cinfo);
    
    return true;
}

/**
 * @brief 打印JPEG图像信息
 */
void jpeg_print_info(const JpegInfo *info)
{
    if (!info) {
        return;
    }
    
    printf("\n========== JPEG图像信息 ==========\n");
    printf("尺寸: %dx%d\n", info->width, info->height);
    printf("通道数: %d\n", info->channels);
    printf("色彩空间: %s\n", jpeg_get_color_space_name(info->color_space));
    printf("子采样: %s\n", jpeg_get_subsampling_name(info->subsampling));
    printf("渐进式: %s\n", info->progressive ? "是" : "否");
    printf("估计质量: %d\n", info->quality);
    printf("文件大小: %zu 字节 (%.2f KB)\n", 
           info->file_size, info->file_size / 1024.0);
    printf("包含EXIF: %s\n", info->has_exif ? "是" : "否");
    printf("包含缩略图: %s\n", info->has_thumbnail ? "是" : "否");
    
    if (info->comment[0] != '\0') {
        printf("注释: %s\n", info->comment);
    }
    
    printf("====================================\n\n");
}

/**
 * @brief 验证JPEG文件
 */
bool jpeg_validate(const char *filename)
{
    if (!filename) {
        return false;
    }
    
    printf("验证JPEG文件: %s\n", filename);
    
    // 检查文件格式
    if (!is_jpeg_file(filename)) {
        fprintf(stderr, "不是有效的JPEG文件\n");
        return false;
    }
    
    // 尝试获取信息
    JpegInfo info;
    if (!jpeg_get_info(filename, &info)) {
        fprintf(stderr, "无法读取JPEG信息\n");
        return false;
    }
    
    // 验证基本参数
    if (info.width <= 0 || info.height <= 0) {
        fprintf(stderr, "无效的图像尺寸\n");
        return false;
    }
    
    if (info.channels <= 0 || info.channels > 4) {
        fprintf(stderr, "无效的通道数\n");
        return false;
    }
    
    printf("JPEG文件验证通过\n");
    return true;
}

// 第二部分结束
// ============================================================================
// EXIF元数据处理
// ============================================================================

#include <libexif/exif-data.h>
#include <libexif/exif-loader.h>

/**
 * @brief 读取EXIF有理数值
 */
static double exif_get_rational_value(ExifEntry *entry, int index)
{
    if (!entry || entry->format != EXIF_FORMAT_RATIONAL) {
        return 0.0;
    }
    
    if (index >= (int)entry->components) {
        return 0.0;
    }
    
    ExifByteOrder order = exif_data_get_byte_order(entry->parent->parent);
    ExifRational rational = exif_get_rational(entry->data + index * 8, order);
    
    if (rational.denominator == 0) {
        return 0.0;
    }
    
    return (double)rational.numerator / (double)rational.denominator;
}

/**
 * @brief 读取EXIF短整型值
 */
static int exif_get_short_value(ExifEntry *entry)
{
    if (!entry || entry->size < 2) {
        return 0;
    }
    
    ExifByteOrder order = exif_data_get_byte_order(entry->parent->parent);
    return exif_get_short(entry->data, order);
}

/**
 * @brief 读取EXIF长整型值
 */
static int exif_get_long_value(ExifEntry *entry)
{
    if (!entry || entry->size < 4) {
        return 0;
    }
    
    ExifByteOrder order = exif_data_get_byte_order(entry->parent->parent);
    return exif_get_long(entry->data, order);
}

/**
 * @brief 读取EXIF字符串值
 */
static void exif_get_string_value(ExifEntry *entry, char *buffer, size_t size)
{
    if (!entry || !buffer || size == 0) {
        return;
    }
    
    char value[1024];
    exif_entry_get_value(entry, value, sizeof(value));
    
    // 复制并确保以null结尾
    strncpy(buffer, value, size - 1);
    buffer[size - 1] = '\0';
}

/**
 * @brief 解析GPS坐标
 */
static double parse_gps_coordinate(ExifEntry *entry)
{
    if (!entry || entry->format != EXIF_FORMAT_RATIONAL || entry->components < 3) {
        return 0.0;
    }
    
    // 读取度、分、秒
    double degrees = exif_get_rational_value(entry, 0);
    double minutes = exif_get_rational_value(entry, 1);
    double seconds = exif_get_rational_value(entry, 2);
    
    return degrees + minutes / 60.0 + seconds / 3600.0;
}

/**
 * @brief 解析GPS信息
 */
static void parse_gps_info(ExifData *exif, ExifGpsInfo *gps)
{
    if (!exif || !gps) {
        return;
    }
    
    memset(gps, 0, sizeof(ExifGpsInfo));
    
    ExifContent *content = exif->ifd[EXIF_IFD_GPS];
    if (!content) {
        return;
    }
    
    ExifEntry *entry;
    
    // GPS纬度
    entry = exif_content_get_entry(content, EXIF_TAG_GPS_LATITUDE);
    if (entry) {
        gps->latitude = parse_gps_coordinate(entry);
        gps->has_gps = true;
    }
    
    // GPS纬度参考 (N/S)
    entry = exif_content_get_entry(content, EXIF_TAG_GPS_LATITUDE_REF);
    if (entry && entry->size >= 1) {
        gps->latitude_ref[0] = entry->data[0];
        gps->latitude_ref[1] = '\0';
        
        // 南纬为负
        if (gps->latitude_ref[0] == 'S') {
            gps->latitude = -gps->latitude;
        }
    }
    
    // GPS经度
    entry = exif_content_get_entry(content, EXIF_TAG_GPS_LONGITUDE);
    if (entry) {
        gps->longitude = parse_gps_coordinate(entry);
        gps->has_gps = true;
    }
    
    // GPS经度参考 (E/W)
    entry = exif_content_get_entry(content, EXIF_TAG_GPS_LONGITUDE_REF);
    if (entry && entry->size >= 1) {
        gps->longitude_ref[0] = entry->data[0];
        gps->longitude_ref[1] = '\0';
        
        // 西经为负
        if (gps->longitude_ref[0] == 'W') {
            gps->longitude = -gps->longitude;
        }
    }
    
    // GPS海拔
    entry = exif_content_get_entry(content, EXIF_TAG_GPS_ALTITUDE);
    if (entry) {
        gps->altitude = exif_get_rational_value(entry, 0);
    }
    
    // GPS海拔参考
    entry = exif_content_get_entry(content, EXIF_TAG_GPS_ALTITUDE_REF);
    if (entry && entry->size >= 1) {
        gps->altitude_ref = entry->data[0];
        
        // 海平面以下为负
        if (gps->altitude_ref == 1) {
            gps->altitude = -gps->altitude;
        }
    }
    
    // GPS时间戳
    entry = exif_content_get_entry(content, EXIF_TAG_GPS_TIME_STAMP);
    if (entry) {
        exif_get_string_value(entry, gps->timestamp, sizeof(gps->timestamp));
    }
    
    // GPS日期戳
    entry = exif_content_get_entry(content, EXIF_TAG_GPS_DATE_STAMP);
    if (entry) {
        exif_get_string_value(entry, gps->datestamp, sizeof(gps->datestamp));
    }
}

/**
 * @brief 解析基本EXIF信息
 */
static void parse_basic_exif(ExifData *ed, ExifMetadata *exif)
{
    ExifEntry *entry;
    ExifContent *content;
    
    // IFD0 - 基本信息
    content = ed->ifd[EXIF_IFD_0];
    if (content) {
        // 制造商
        entry = exif_content_get_entry(content, EXIF_TAG_MAKE);
        if (entry) {
            exif_get_string_value(entry, exif->make, sizeof(exif->make));
        }
        
        // 型号
        entry = exif_content_get_entry(content, EXIF_TAG_MODEL);
        if (entry) {
            exif_get_string_value(entry, exif->model, sizeof(exif->model));
        }
        
        // 软件
        entry = exif_content_get_entry(content, EXIF_TAG_SOFTWARE);
        if (entry) {
            exif_get_string_value(entry, exif->software, sizeof(exif->software));
        }
        
        // 方向
        entry = exif_content_get_entry(content, EXIF_TAG_ORIENTATION);
        if (entry) {
            exif->orientation = (ExifOrientation)exif_get_short_value(entry);
        } else {
            exif->orientation = EXIF_ORIENTATION_NORMAL;
        }
        
        // X分辨率
        entry = exif_content_get_entry(content, EXIF_TAG_X_RESOLUTION);
        if (entry) {
            exif->x_resolution = (int)exif_get_rational_value(entry, 0);
        }
        
        // Y分辨率
        entry = exif_content_get_entry(content, EXIF_TAG_Y_RESOLUTION);
        if (entry) {
            exif->y_resolution = (int)exif_get_rational_value(entry, 0);
        }
        
        // 分辨率单位
        entry = exif_content_get_entry(content, EXIF_TAG_RESOLUTION_UNIT);
        if (entry) {
            exif->resolution_unit = exif_get_short_value(entry);
        }
        
        // 日期时间
        entry = exif_content_get_entry(content, EXIF_TAG_DATE_TIME);
        if (entry) {
            exif_get_string_value(entry, exif->datetime, sizeof(exif->datetime));
        }
        
        // 图像描述
        entry = exif_content_get_entry(content, EXIF_TAG_IMAGE_DESCRIPTION);
        if (entry) {
            exif_get_string_value(entry, exif->description, sizeof(exif->description));
        }
        
        // 版权
        entry = exif_content_get_entry(content, EXIF_TAG_COPYRIGHT);
        if (entry) {
            exif_get_string_value(entry, exif->copyright, sizeof(exif->copyright));
        }
        
        // 作者
        entry = exif_content_get_entry(content, EXIF_TAG_ARTIST);
        if (entry) {
            exif_get_string_value(entry, exif->artist, sizeof(exif->artist));
        }
    }
}

/**
 * @brief 解析相机设置EXIF信息
 */
static void parse_camera_exif(ExifData *ed, ExifMetadata *exif)
{
    ExifEntry *entry;
    ExifContent *content;
    
    // EXIF IFD - 相机设置
    content = ed->ifd[EXIF_IFD_EXIF];
    if (!content) {
        return;
    }
    
    // 图像宽度
    entry = exif_content_get_entry(content, EXIF_TAG_PIXEL_X_DIMENSION);
    if (entry) {
        exif->width = exif_get_long_value(entry);
    }
    
    // 图像高度
    entry = exif_content_get_entry(content, EXIF_TAG_PIXEL_Y_DIMENSION);
    if (entry) {
        exif->height = exif_get_long_value(entry);
    }
    
    // 曝光时间
    entry = exif_content_get_entry(content, EXIF_TAG_EXPOSURE_TIME);
    if (entry) {
        exif->exposure_time = exif_get_rational_value(entry, 0);
    }
    
    // 光圈值
    entry = exif_content_get_entry(content, EXIF_TAG_FNUMBER);
    if (entry) {
        exif->f_number = exif_get_rational_value(entry, 0);
    }
    
    // ISO感光度
    entry = exif_content_get_entry(content, EXIF_TAG_ISO_SPEED_RATINGS);
    if (entry) {
        exif->iso_speed = exif_get_short_value(entry);
    }
    
    // 焦距
    entry = exif_content_get_entry(content, EXIF_TAG_FOCAL_LENGTH);
    if (entry) {
        exif->focal_length = exif_get_rational_value(entry, 0);
    }
    
    // 35mm等效焦距
    entry = exif_content_get_entry(content, EXIF_TAG_FOCAL_LENGTH_IN_35MM_FILM);
    if (entry) {
        exif->focal_length_35mm = exif_get_short_value(entry);
    }
    
    // 闪光灯
    entry = exif_content_get_entry(content, EXIF_TAG_FLASH);
    if (entry) {
        exif->flash = exif_get_short_value(entry);
    }
    
    // 白平衡
    entry = exif_content_get_entry(content, EXIF_TAG_WHITE_BALANCE);
    if (entry) {
        exif->white_balance = exif_get_short_value(entry);
    }
    
    // 曝光程序
    entry = exif_content_get_entry(content, EXIF_TAG_EXPOSURE_PROGRAM);
    if (entry) {
        exif->exposure_program = exif_get_short_value(entry);
    }
    
    // 测光模式
    entry = exif_content_get_entry(content, EXIF_TAG_METERING_MODE);
    if (entry) {
        exif->metering_mode = exif_get_short_value(entry);
    }
    
    // 原始日期时间
    entry = exif_content_get_entry(content, EXIF_TAG_DATE_TIME_ORIGINAL);
    if (entry) {
        exif_get_string_value(entry, exif->datetime_original, 
                             sizeof(exif->datetime_original));
    }
    
    // 数字化日期时间
    entry = exif_content_get_entry(content, EXIF_TAG_DATE_TIME_DIGITIZED);
    if (entry) {
        exif_get_string_value(entry, exif->datetime_digitized, 
                             sizeof(exif->datetime_digitized));
    }
    
    // 用户注释
    entry = exif_content_get_entry(content, EXIF_TAG_USER_COMMENT);
    if (entry) {
        exif_get_string_value(entry, exif->user_comment, sizeof(exif->user_comment));
    }
    
    // 镜头制造商
    entry = exif_content_get_entry(content, EXIF_TAG_LENS_MAKE);
    if (entry) {
        exif_get_string_value(entry, exif->lens_make, sizeof(exif->lens_make));
    }
    
    // 镜头型号
    entry = exif_content_get_entry(content, EXIF_TAG_LENS_MODEL);
    if (entry) {
        exif_get_string_value(entry, exif->lens_model, sizeof(exif->lens_model));
    }
}

/**
 * @brief 读取EXIF元数据
 */
bool jpeg_read_exif(const char *filename, ExifMetadata *exif)
{
    if (!filename || !exif) {
        fprintf(stderr, "参数无效\n");
        return false;
    }
    
    printf("读取EXIF元数据: %s\n", filename);
    
    // 初始化EXIF结构
    memset(exif, 0, sizeof(ExifMetadata));
    
    // 加载EXIF数据
    ExifData *ed = exif_data_new_from_file(filename);
    if (!ed) {
        fprintf(stderr, "无法读取EXIF数据\n");
        return false;
    }
    
    // 解析基本信息
    parse_basic_exif(ed, exif);
    
    // 解析相机设置
    parse_camera_exif(ed, exif);
    
    // 解析GPS信息
    parse_gps_info(ed, &exif->gps);
    
    // 释放EXIF数据
    exif_data_unref(ed);
    
    printf("成功读取EXIF元数据\n");
    return true;
}

/**
 * @brief 从内存读取EXIF元数据
 */
bool jpeg_read_exif_from_memory(const unsigned char *data, size_t size, 
                                ExifMetadata *exif)
{
    if (!data || size == 0 || !exif) {
        fprintf(stderr, "参数无效\n");
        return false;
    }
    
    printf("从内存读取EXIF元数据 (%zu 字节)\n", size);
    
    // 初始化EXIF结构
    memset(exif, 0, sizeof(ExifMetadata));
    
    // 创建EXIF加载器
    ExifLoader *loader = exif_loader_new();
    if (!loader) {
        fprintf(stderr, "创建EXIF加载器失败\n");
        return false;
    }
    
    // 写入数据
    exif_loader_write(loader, (unsigned char*)data, size);
    
    // 获取EXIF数据
    ExifData *ed = exif_loader_get_data(loader);
    exif_loader_unref(loader);
    
    if (!ed) {
        fprintf(stderr, "无法读取EXIF数据\n");
        return false;
    }
    
    // 解析基本信息
    parse_basic_exif(ed, exif);
    
    // 解析相机设置
    parse_camera_exif(ed, exif);
    
    // 解析GPS信息
    parse_gps_info(ed, &exif->gps);
    
    // 释放EXIF数据
    exif_data_unref(ed);
    
    printf("成功从内存读取EXIF元数据\n");
    return true;
}

// 第三部分A结束
// ============================================================================
// EXIF写入和操作
// ============================================================================

/**
 * @brief 设置EXIF字符串条目
 */
static void exif_set_string_entry(ExifContent *content, ExifTag tag, const char *value)
{
    if (!content || !value || value[0] == '\0') {
        return;
    }
    
    ExifEntry *entry = exif_content_get_entry(content, tag);
    if (!entry) {
        entry = exif_entry_new();
        exif_content_add_entry(content, entry);
        exif_entry_initialize(entry, tag);
        exif_entry_unref(entry);
    }
    
    // 设置字符串值
    if (entry->data) {
        free(entry->data);
    }
    
    size_t len = strlen(value) + 1;
    entry->data = (unsigned char*)malloc(len);
    entry->size = len;
    memcpy(entry->data, value, len);
    entry->components = len;
    entry->format = EXIF_FORMAT_ASCII;
}

/**
 * @brief 设置EXIF短整型条目
 */
static void exif_set_short_entry(ExifContent *content, ExifTag tag, int value)
{
    if (!content) {
        return;
    }
    
    ExifEntry *entry = exif_content_get_entry(content, tag);
    if (!entry) {
        entry = exif_entry_new();
        exif_content_add_entry(content, entry);
        exif_entry_initialize(entry, tag);
        exif_entry_unref(entry);
    }
    
    // 设置短整型值
    if (entry->data) {
        free(entry->data);
    }
    
    entry->data = (unsigned char*)malloc(2);
    entry->size = 2;
    entry->components = 1;
    entry->format = EXIF_FORMAT_SHORT;
    
    ExifByteOrder order = exif_data_get_byte_order(entry->parent->parent);
    exif_set_short(entry->data, order, (ExifShort)value);
}

/**
 * @brief 设置EXIF有理数条目
 */
static void exif_set_rational_entry(ExifContent *content, ExifTag tag, double value)
{
    if (!content) {
        return;
    }
    
    ExifEntry *entry = exif_content_get_entry(content, tag);
    if (!entry) {
        entry = exif_entry_new();
        exif_content_add_entry(content, entry);
        exif_entry_initialize(entry, tag);
        exif_entry_unref(entry);
    }
    
    // 设置有理数值
    if (entry->data) {
        free(entry->data);
    }
    
    entry->data = (unsigned char*)malloc(8);
    entry->size = 8;
    entry->components = 1;
    entry->format = EXIF_FORMAT_RATIONAL;
    
    // 转换为有理数
    ExifRational rational;
    if (value < 1.0) {
        rational.numerator = 1;
        rational.denominator = (ExifLong)(1.0 / value + 0.5);
    } else {
        rational.numerator = (ExifLong)(value * 1000.0 + 0.5);
        rational.denominator = 1000;
    }
    
    ExifByteOrder order = exif_data_get_byte_order(entry->parent->parent);
    exif_set_rational(entry->data, order, rational);
}

/**
 * @brief 写入基本EXIF信息
 */
static void write_basic_exif(ExifData *ed, const ExifMetadata *exif)
{
    ExifContent *content = ed->ifd[EXIF_IFD_0];
    if (!content) {
        return;
    }
    
    // 制造商
    if (exif->make[0] != '\0') {
        exif_set_string_entry(content, EXIF_TAG_MAKE, exif->make);
    }
    
    // 型号
    if (exif->model[0] != '\0') {
        exif_set_string_entry(content, EXIF_TAG_MODEL, exif->model);
    }
    
    // 软件
    if (exif->software[0] != '\0') {
        exif_set_string_entry(content, EXIF_TAG_SOFTWARE, exif->software);
    }
    
    // 方向
    if (exif->orientation != EXIF_ORIENTATION_NORMAL) {
        exif_set_short_entry(content, EXIF_TAG_ORIENTATION, exif->orientation);
    }
    
    // X分辨率
    if (exif->x_resolution > 0) {
        exif_set_rational_entry(content, EXIF_TAG_X_RESOLUTION, exif->x_resolution);
    }
    
    // Y分辨率
    if (exif->y_resolution > 0) {
        exif_set_rational_entry(content, EXIF_TAG_Y_RESOLUTION, exif->y_resolution);
    }
    
    // 分辨率单位
    if (exif->resolution_unit > 0) {
        exif_set_short_entry(content, EXIF_TAG_RESOLUTION_UNIT, exif->resolution_unit);
    }
    
    // 日期时间
    if (exif->datetime[0] != '\0') {
        exif_set_string_entry(content, EXIF_TAG_DATE_TIME, exif->datetime);
    }
    
    // 图像描述
    if (exif->description[0] != '\0') {
        exif_set_string_entry(content, EXIF_TAG_IMAGE_DESCRIPTION, exif->description);
    }
    
    // 版权
    if (exif->copyright[0] != '\0') {
        exif_set_string_entry(content, EXIF_TAG_COPYRIGHT, exif->copyright);
    }
    
    // 作者
    if (exif->artist[0] != '\0') {
        exif_set_string_entry(content, EXIF_TAG_ARTIST, exif->artist);
    }
}

/**
 * @brief 写入相机设置EXIF信息
 */
static void write_camera_exif(ExifData *ed, const ExifMetadata *exif)
{
    ExifContent *content = ed->ifd[EXIF_IFD_EXIF];
    if (!content) {
        return;
    }
    
    // 曝光时间
    if (exif->exposure_time > 0) {
        exif_set_rational_entry(content, EXIF_TAG_EXPOSURE_TIME, exif->exposure_time);
    }
    
    // 光圈值
    if (exif->f_number > 0) {
        exif_set_rational_entry(content, EXIF_TAG_FNUMBER, exif->f_number);
    }
    
    // ISO感光度
    if (exif->iso_speed > 0) {
        exif_set_short_entry(content, EXIF_TAG_ISO_SPEED_RATINGS, exif->iso_speed);
    }
    
    // 焦距
    if (exif->focal_length > 0) {
        exif_set_rational_entry(content, EXIF_TAG_FOCAL_LENGTH, exif->focal_length);
    }
    
    // 35mm等效焦距
    if (exif->focal_length_35mm > 0) {
        exif_set_short_entry(content, EXIF_TAG_FOCAL_LENGTH_IN_35MM_FILM, 
                            exif->focal_length_35mm);
    }
    
    // 闪光灯
    if (exif->flash >= 0) {
        exif_set_short_entry(content, EXIF_TAG_FLASH, exif->flash);
    }
    
    // 白平衡
    if (exif->white_balance >= 0) {
        exif_set_short_entry(content, EXIF_TAG_WHITE_BALANCE, exif->white_balance);
    }
    
    // 曝光程序
    if (exif->exposure_program >= 0) {
        exif_set_short_entry(content, EXIF_TAG_EXPOSURE_PROGRAM, exif->exposure_program);
    }
    
    // 测光模式
    if (exif->metering_mode >= 0) {
        exif_set_short_entry(content, EXIF_TAG_METERING_MODE, exif->metering_mode);
    }
    
    // 原始日期时间
    if (exif->datetime_original[0] != '\0') {
        exif_set_string_entry(content, EXIF_TAG_DATE_TIME_ORIGINAL, 
                             exif->datetime_original);
    }
    
    // 数字化日期时间
    if (exif->datetime_digitized[0] != '\0') {
        exif_set_string_entry(content, EXIF_TAG_DATE_TIME_DIGITIZED, 
                             exif->datetime_digitized);
    }
    
    // 用户注释
    if (exif->user_comment[0] != '\0') {
        exif_set_string_entry(content, EXIF_TAG_USER_COMMENT, exif->user_comment);
    }
    
    // 镜头制造商
    if (exif->lens_make[0] != '\0') {
        exif_set_string_entry(content, EXIF_TAG_LENS_MAKE, exif->lens_make);
    }
    
    // 镜头型号
    if (exif->lens_model[0] != '\0') {
        exif_set_string_entry(content, EXIF_TAG_LENS_MODEL, exif->lens_model);
    }
}

/**
 * @brief 写入EXIF元数据
 */
bool jpeg_write_exif(const char *filename, const ExifMetadata *exif)
{
    if (!filename || !exif) {
        fprintf(stderr, "参数无效\n");
        return false;
    }
    
    printf("写入EXIF元数据: %s\n", filename);
    
    // 加载现有EXIF数据或创建新的
    ExifData *ed = exif_data_new_from_file(filename);
    if (!ed) {
        ed = exif_data_new();
        if (!ed) {
            fprintf(stderr, "创建EXIF数据失败\n");
            return false;
        }
        
        // 设置字节序
        exif_data_set_byte_order(ed, EXIF_BYTE_ORDER_INTEL);
        
        // 修复EXIF数据
        exif_data_fix(ed);
    }
    
    // 写入基本信息
    write_basic_exif(ed, exif);
    
    // 写入相机设置
    write_camera_exif(ed, exif);
    
    // 保存EXIF数据
    unsigned char *exif_data = NULL;
    unsigned int exif_size = 0;
    exif_data_save_data(ed, &exif_data, &exif_size);
    
    if (!exif_data) {
        fprintf(stderr, "保存EXIF数据失败\n");
        exif_data_unref(ed);
        return false;
    }
    
    // 读取原始JPEG
    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        fprintf(stderr, "无法打开文件: %s\n", filename);
        free(exif_data);
        exif_data_unref(ed);
        return false;
    }
    
    fseek(fp, 0, SEEK_END);
    size_t jpeg_size = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    
    unsigned char *jpeg_data = (unsigned char*)malloc(jpeg_size);
    if (!jpeg_data) {
        fprintf(stderr, "内存分配失败\n");
        fclose(fp);
        free(exif_data);
        exif_data_unref(ed);
        return false;
    }
    
    fread(jpeg_data, 1, jpeg_size, fp);
    fclose(fp);
    
    // 创建临时文件
    char temp_filename[512];
    snprintf(temp_filename, sizeof(temp_filename), "%s.tmp", filename);
    
    fp = fopen(temp_filename, "wb");
    if (!fp) {
        fprintf(stderr, "无法创建临时文件\n");
        free(jpeg_data);
        free(exif_data);
        exif_data_unref(ed);
        return false;
    }
    
    // 写入JPEG SOI标记
    fwrite(jpeg_data, 1, 2, fp);
    
    // 写入APP1标记 (EXIF)
    unsigned char app1_marker[2] = {0xFF, 0xE1};
    fwrite(app1_marker, 1, 2, fp);
    
    // 写入APP1长度
    unsigned short app1_length = exif_size + 2;
    unsigned char length_bytes[2];
    length_bytes[0] = (app1_length >> 8) & 0xFF;
    length_bytes[1] = app1_length & 0xFF;
    fwrite(length_bytes, 1, 2, fp);
    
    // 写入EXIF数据
    fwrite(exif_data, 1, exif_size, fp);
    
    // 跳过原始JPEG中的APP标记，写入剩余数据
    size_t pos = 2;
    while (pos < jpeg_size - 1) {
        if (jpeg_data[pos] == 0xFF && 
            (jpeg_data[pos + 1] >= 0xE0 && jpeg_data[pos + 1] <= 0xEF)) {
            // 跳过APP标记
            unsigned short marker_length = (jpeg_data[pos + 2] << 8) | jpeg_data[pos + 3];
            pos += 2 + marker_length;
        } else {
            break;
        }
    }
    
    // 写入剩余的JPEG数据
    fwrite(jpeg_data + pos, 1, jpeg_size - pos, fp);
    
    fclose(fp);
    free(jpeg_data);
    free(exif_data);
    exif_data_unref(ed);
    
    // 替换原文件
    remove(filename);
    rename(temp_filename, filename);
    
    printf("成功写入EXIF元数据\n");
    return true;
}

/**
 * @brief 删除EXIF元数据
 */
bool jpeg_remove_exif(const char *filename)
{
    if (!filename) {
        fprintf(stderr, "文件名为空\n");
        return false;
    }
    
    printf("删除EXIF元数据: %s\n", filename);
    
    // 读取JPEG图像
    Image *img = jpeg_read(filename);
    if (!img) {
        fprintf(stderr, "读取JPEG失败\n");
        return false;
    }
    
    // 重新写入（不包含EXIF）
    bool result = jpeg_write(img, filename, JPEG_DEFAULT_QUALITY);
    
    image_destroy(img);
    
    if (result) {
        printf("成功删除EXIF元数据\n");
    }
    
    return result;
}

/**
 * @brief 复制EXIF元数据
 */
bool jpeg_copy_exif(const char *src_filename, const char *dst_filename)
{
    if (!src_filename || !dst_filename) {
        fprintf(stderr, "文件名无效\n");
        return false;
    }
    
    printf("复制EXIF元数据: %s -> %s\n", src_filename, dst_filename);
    
    // 读取源文件的EXIF
    ExifMetadata exif;
    if (!jpeg_read_exif(src_filename, &exif)) {
        fprintf(stderr, "读取源文件EXIF失败\n");
        return false;
    }
    
    // 写入到目标文件
    if (!jpeg_write_exif(dst_filename, &exif)) {
        fprintf(stderr, "写入目标文件EXIF失败\n");
        return false;
    }
    
    printf("成功复制EXIF元数据\n");
    return true;
}

/**
 * @brief 更新EXIF方向
 */
bool jpeg_update_exif_orientation(const char *filename, ExifOrientation orientation)
{
    if (!filename) {
        fprintf(stderr, "文件名为空\n");
        return false;
    }
    
    printf("更新EXIF方向: %s -> %s\n", 
           filename, jpeg_get_orientation_name(orientation));
    
    // 读取现有EXIF
    ExifMetadata exif;
    memset(&exif, 0, sizeof(ExifMetadata));
    
    // 尝试读取现有EXIF（如果失败则创建新的）
    jpeg_read_exif(filename, &exif);
    
    // 更新方向
    exif.orientation = orientation;
    
    // 写回EXIF
    return jpeg_write_exif(filename, &exif);
}

/**
 * @brief 根据EXIF方向旋转图像
 */
bool jpeg_apply_exif_orientation(Image *img, ExifOrientation orientation)
{
    if (!img) {
        return false;
    }
    
    printf("应用EXIF方向: %s\n", jpeg_get_orientation_name(orientation));
    
    switch (orientation) {
        case EXIF_ORIENTATION_NORMAL:
            // 无需操作
            return true;
            
        case EXIF_ORIENTATION_FLIP_HORIZONTAL:
            return image_flip_horizontal(img);
            
        case EXIF_ORIENTATION_ROTATE_180:
            return image_rotate_180(img);
            
        case EXIF_ORIENTATION_FLIP_VERTICAL:
            return image_flip_vertical(img);
            
        case EXIF_ORIENTATION_TRANSPOSE:
            // 转置 = 顺时针90度 + 水平翻转
            if (!image_rotate_90_cw(img)) return false;
            return image_flip_horizontal(img);
            
        case EXIF_ORIENTATION_ROTATE_90_CW:
            return image_rotate_90_cw(img);
            
        case EXIF_ORIENTATION_TRANSVERSE:
            // 横向转置 = 顺时针90度 + 垂直翻转
            if (!image_rotate_90_cw(img)) return false;
            return image_flip_vertical(img);
            
        case EXIF_ORIENTATION_ROTATE_90_CCW:
            return image_rotate_90_ccw(img);
            
        default:
            fprintf(stderr, "未知的EXIF方向: %d\n", orientation);
            return false;
    }
}

/**
 * @brief 自动旋转JPEG图像（根据EXIF方向）
 */
Image* jpeg_read_auto_orient(const char *filename)
{
    if (!filename) {
        fprintf(stderr, "文件名为空\n");
        return NULL;
    }
    
    printf("读取并自动旋转JPEG: %s\n", filename);
    
    // 读取EXIF获取方向
    ExifMetadata exif;
    ExifOrientation orientation = EXIF_ORIENTATION_NORMAL;
    
    if (jpeg_read_exif(filename, &exif)) {
        orientation = exif.orientation;
    }
    
    // 读取图像
    Image *img = jpeg_read(filename);
    if (!img) {
        return NULL;
    }
    
    // 应用方向
    if (orientation != EXIF_ORIENTATION_NORMAL) {
        if (!jpeg_apply_exif_orientation(img, orientation)) {
            fprintf(stderr, "应用EXIF方向失败\n");
            image_destroy(img);
            return NULL;
        }
    }
    
    return img;
}

// 第三部分B结束
// ============================================================================
// EXIF打印和辅助函数
// ============================================================================

/**
 * @brief 获取方向名称
 */
const char* jpeg_get_orientation_name(ExifOrientation orientation)
{
    switch (orientation) {
        case EXIF_ORIENTATION_NORMAL:
            return "正常";
        case EXIF_ORIENTATION_FLIP_HORIZONTAL:
            return "水平翻转";
        case EXIF_ORIENTATION_ROTATE_180:
            return "旋转180度";
        case EXIF_ORIENTATION_FLIP_VERTICAL:
            return "垂直翻转";
        case EXIF_ORIENTATION_TRANSPOSE:
            return "转置";
        case EXIF_ORIENTATION_ROTATE_90_CW:
            return "顺时针旋转90度";
        case EXIF_ORIENTATION_TRANSVERSE:
            return "横向转置";
        case EXIF_ORIENTATION_ROTATE_90_CCW:
            return "逆时针旋转90度";
        default:
            return "未知";
    }
}

/**
 * @brief 获取曝光程序名称
 */
static const char* get_exposure_program_name(int program)
{
    switch (program) {
        case 0: return "未定义";
        case 1: return "手动";
        case 2: return "程序自动";
        case 3: return "光圈优先";
        case 4: return "快门优先";
        case 5: return "创意程序";
        case 6: return "动作程序";
        case 7: return "人像模式";
        case 8: return "风景模式";
        default: return "未知";
    }
}

/**
 * @brief 获取测光模式名称
 */
static const char* get_metering_mode_name(int mode)
{
    switch (mode) {
        case 0: return "未知";
        case 1: return "平均测光";
        case 2: return "中央重点测光";
        case 3: return "点测光";
        case 4: return "多点测光";
        case 5: return "评价测光";
        case 6: return "局部测光";
        default: return "其他";
    }
}

/**
 * @brief 格式化曝光时间
 */
static void format_exposure_time(double exposure, char *buffer, size_t size)
{
    if (!buffer || size == 0) {
        return;
    }
    
    if (exposure < 1.0 && exposure > 0) {
        // 显示为分数形式
        int denominator = (int)(1.0 / exposure + 0.5);
        snprintf(buffer, size, "1/%d 秒", denominator);
    } else if (exposure >= 1.0) {
        snprintf(buffer, size, "%.1f 秒", exposure);
    } else {
        snprintf(buffer, size, "未知");
    }
}

/**
 * @brief 打印EXIF元数据
 */
void jpeg_print_exif(const ExifMetadata *exif)
{
    if (!exif) {
        return;
    }
    
    printf("\n");
    printf("╔════════════════════════════════════════════════════════════╗\n");
    printf("║                      EXIF 元数据                           ║\n");
    printf("╠════════════════════════════════════════════════════════════╣\n");
    
    // 基本信息
    bool has_basic = false;
    if (exif->make[0] != '\0' || exif->model[0] != '\0' || exif->software[0] != '\0') {
        printf("║ 基本信息:                                                  ║\n");
        has_basic = true;
    }
    
    if (exif->make[0] != '\0') {
        printf("║   制造商: %-48s ║\n", exif->make);
    }
    if (exif->model[0] != '\0') {
        printf("║   型号: %-50s ║\n", exif->model);
    }
    if (exif->software[0] != '\0') {
        printf("║   软件: %-50s ║\n", exif->software);
    }
    
    if (has_basic) {
        printf("║                                                            ║\n");
    }
    
    // 日期时间
    if (exif->datetime[0] != '\0' || exif->datetime_original[0] != '\0') {
        printf("║ 日期时间:                                                  ║\n");
        
        if (exif->datetime_original[0] != '\0') {
            printf("║   拍摄时间: %-46s ║\n", exif->datetime_original);
        }
        if (exif->datetime[0] != '\0') {
            printf("║   修改时间: %-46s ║\n", exif->datetime);
        }
        if (exif->datetime_digitized[0] != '\0') {
            printf("║   数字化时间: %-44s ║\n", exif->datetime_digitized);
        }
        printf("║                                                            ║\n");
    }
    
    // 图像信息
    if (exif->width > 0 || exif->height > 0 || 
        exif->orientation != EXIF_ORIENTATION_NORMAL) {
        printf("║ 图像信息:                                                  ║\n");
        
        if (exif->width > 0 && exif->height > 0) {
            printf("║   尺寸: %d x %d 像素%*s║\n", 
                   exif->width, exif->height,
                   (int)(38 - snprintf(NULL, 0, "%d x %d 像素", 
                                      exif->width, exif->height)), "");
        }
        
        printf("║   方向: %-50s ║\n", 
               jpeg_get_orientation_name(exif->orientation));
        
        if (exif->x_resolution > 0 && exif->y_resolution > 0) {
            const char *unit = exif->resolution_unit == 2 ? "dpi" : "dpcm";
            printf("║   分辨率: %d x %d %s%*s║\n",
                   exif->x_resolution, exif->y_resolution, unit,
                   (int)(42 - snprintf(NULL, 0, "%d x %d %s",
                                      exif->x_resolution, exif->y_resolution, unit)), "");
        }
        printf("║                                                            ║\n");
    }
    
    // 相机设置
    if (exif->exposure_time > 0 || exif->f_number > 0 || 
        exif->iso_speed > 0 || exif->focal_length > 0) {
        printf("║ 相机设置:                                                  ║\n");
        
        // 曝光时间
        if (exif->exposure_time > 0) {
            char exp_str[64];
            format_exposure_time(exif->exposure_time, exp_str, sizeof(exp_str));
            printf("║   曝光时间: %-46s ║\n", exp_str);
        }
        
        // 光圈
        if (exif->f_number > 0) {
            printf("║   光圈: f/%.1f%*s║\n", 
                   exif->f_number,
                   (int)(48 - snprintf(NULL, 0, "f/%.1f", exif->f_number)), "");
        }
        
        // ISO
        if (exif->iso_speed > 0) {
            printf("║   ISO: %d%*s║\n",
                   exif->iso_speed,
                   (int)(51 - snprintf(NULL, 0, "%d", exif->iso_speed)), "");
        }
        
        // 焦距
        if (exif->focal_length > 0) {
            if (exif->focal_length_35mm > 0) {
                printf("║   焦距: %.1f mm (35mm等效: %d mm)%*s║\n",
                       exif->focal_length, exif->focal_length_35mm,
                       (int)(26 - snprintf(NULL, 0, "%.1f mm (35mm等效: %d mm)",
                                          exif->focal_length, exif->focal_length_35mm)), "");
            } else {
                printf("║   焦距: %.1f mm%*s║\n",
                       exif->focal_length,
                       (int)(46 - snprintf(NULL, 0, "%.1f mm", exif->focal_length)), "");
            }
        }
        
        // 闪光灯
        if (exif->flash >= 0) {
            const char *flash_str = (exif->flash & 0x01) ? "开启" : "关闭";
            printf("║   闪光灯: %-47s ║\n", flash_str);
        }
        
        // 白平衡
        if (exif->white_balance >= 0) {
            const char *wb_str = exif->white_balance == 0 ? "自动" : "手动";
            printf("║   白平衡: %-47s ║\n", wb_str);
        }
        
        // 曝光程序
        if (exif->exposure_program >= 0) {
            printf("║   曝光程序: %-45s ║\n",
                   get_exposure_program_name(exif->exposure_program));
        }
        
        // 测光模式
        if (exif->metering_mode >= 0) {
            printf("║   测光模式: %-45s ║\n",
                   get_metering_mode_name(exif->metering_mode));
        }
        
        printf("║                                                            ║\n");
    }
    
    // 镜头信息
    if (exif->lens_make[0] != '\0' || exif->lens_model[0] != '\0') {
        printf("║ 镜头信息:                                                  ║\n");
        
        if (exif->lens_make[0] != '\0') {
            printf("║   制造商: %-47s ║\n", exif->lens_make);
        }
        if (exif->lens_model[0] != '\0') {
            printf("║   型号: %-49s ║\n", exif->lens_model);
        }
        printf("║                                                            ║\n");
    }
    
    // GPS信息
    if (exif->gps.has_gps) {
        printf("║ GPS 信息:                                                  ║\n");
        
        // 纬度
        char lat_dir = exif->gps.latitude >= 0 ? 'N' : 'S';
        printf("║   纬度: %.6f° %c%*s║\n",
               fabs(exif->gps.latitude), lat_dir,
               (int)(42 - snprintf(NULL, 0, "%.6f° %c",
                                  fabs(exif->gps.latitude), lat_dir)), "");
        
        // 经度
        char lon_dir = exif->gps.longitude >= 0 ? 'E' : 'W';
        printf("║   经度: %.6f° %c%*s║\n",
               fabs(exif->gps.longitude), lon_dir,
               (int)(42 - snprintf(NULL, 0, "%.6f° %c",
                                  fabs(exif->gps.longitude), lon_dir)), "");
        
        // 海拔
        if (exif->gps.altitude != 0) {
            printf("║   海拔: %.1f 米%*s║\n",
                   exif->gps.altitude,
                   (int)(45 - snprintf(NULL, 0, "%.1f 米", exif->gps.altitude)), "");
        }
        
        // GPS时间
        if (exif->gps.timestamp[0] != '\0') {
            if (exif->gps.datestamp[0] != '\0') {
                printf("║   GPS时间: %s %s%*s║\n",
                       exif->gps.datestamp, exif->gps.timestamp,
                       (int)(38 - strlen(exif->gps.datestamp) - 
                            strlen(exif->gps.timestamp)), "");
            } else {
                printf("║   GPS时间: %-44s ║\n", exif->gps.timestamp);
            }
        }
        
        printf("║                                                            ║\n");
    }
    
    // 其他信息
    bool has_other = false;
    if (exif->description[0] != '\0' || exif->copyright[0] != '\0' || 
        exif->artist[0] != '\0' || exif->user_comment[0] != '\0') {
        printf("║ 其他信息:                                                  ║\n");
        has_other = true;
    }
    
    if (exif->description[0] != '\0') {
        printf("║   描述: %-50s ║\n", exif->description);
    }
    if (exif->artist[0] != '\0') {
        printf("║   作者: %-50s ║\n", exif->artist);
    }
    if (exif->copyright[0] != '\0') {
        printf("║   版权: %-50s ║\n", exif->copyright);
    }
    if (exif->user_comment[0] != '\0') {
        // 用户注释可能很长，需要截断
        char comment[49];
        strncpy(comment, exif->user_comment, sizeof(comment) - 1);
        comment[sizeof(comment) - 1] = '\0';
        printf("║   注释: %-50s ║\n", comment);
    }
    
    if (has_other) {
        printf("║                                                            ║\n");
    }
    
    printf("╚════════════════════════════════════════════════════════════╝\n");
    printf("\n");
}

/**
 * @brief 打印简化的EXIF信息
 */
void jpeg_print_exif_brief(const ExifMetadata *exif)
{
    if (!exif) {
        return;
    }
    
    printf("EXIF: ");
    
    // 相机信息
    if (exif->make[0] != '\0' || exif->model[0] != '\0') {
        if (exif->make[0] != '\0') {
            printf("%s ", exif->make);
        }
        if (exif->model[0] != '\0') {
            printf("%s", exif->model);
        }
        printf(" | ");
    }
    
    // 图像尺寸
    if (exif->width > 0 && exif->height > 0) {
        printf("%dx%d | ", exif->width, exif->height);
    }
    
    // 曝光参数
    if (exif->exposure_time > 0) {
        if (exif->exposure_time < 1.0) {
            printf("1/%.0f ", 1.0 / exif->exposure_time);
        } else {
            printf("%.1fs ", exif->exposure_time);
        }
    }
    
    if (exif->f_number > 0) {
        printf("f/%.1f ", exif->f_number);
    }
    
    if (exif->iso_speed > 0) {
        printf("ISO%d ", exif->iso_speed);
    }
    
    if (exif->focal_length > 0) {
        printf("%.0fmm", exif->focal_length);
    }
    
    // 日期
    if (exif->datetime_original[0] != '\0') {
        printf(" | %s", exif->datetime_original);
    }
    
    printf("\n");
}

/**
 * @brief 比较两个EXIF元数据
 */
bool jpeg_compare_exif(const ExifMetadata *exif1, const ExifMetadata *exif2)
{
    if (!exif1 || !exif2) {
        return false;
    }
    
    // 比较主要字段
    if (strcmp(exif1->make, exif2->make) != 0) return false;
    if (strcmp(exif1->model, exif2->model) != 0) return false;
    if (strcmp(exif1->datetime, exif2->datetime) != 0) return false;
    if (exif1->orientation != exif2->orientation) return false;
    if (exif1->width != exif2->width) return false;
    if (exif1->height != exif2->height) return false;
    
    // 比较相机设置（允许小误差）
    if (fabs(exif1->exposure_time - exif2->exposure_time) > 0.001) return false;
    if (fabs(exif1->f_number - exif2->f_number) > 0.1) return false;
    if (exif1->iso_speed != exif2->iso_speed) return false;
    if (fabs(exif1->focal_length - exif2->focal_length) > 0.1) return false;
    
    return true;
}

/**
 * @brief 验证EXIF数据完整性
 */
bool jpeg_validate_exif(const ExifMetadata *exif)
{
    if (!exif) {
        return false;
    }
    
    // 检查方向值是否有效
    if (exif->orientation < EXIF_ORIENTATION_NORMAL || 
        exif->orientation > EXIF_ORIENTATION_ROTATE_90_CCW) {
        fprintf(stderr, "无效的EXIF方向值: %d\n", exif->orientation);
        return false;
    }
    
    // 检查图像尺寸
    if (exif->width < 0 || exif->height < 0) {
        fprintf(stderr, "无效的图像尺寸: %dx%d\n", exif->width, exif->height);
        return false;
    }
    
    // 检查曝光时间
    if (exif->exposure_time < 0) {
        fprintf(stderr, "无效的曝光时间: %f\n", exif->exposure_time);
        return false;
    }
    
    // 检查光圈值
    if (exif->f_number < 0) {
        fprintf(stderr, "无效的光圈值: %f\n", exif->f_number);
        return false;
    }
    
    // 检查ISO
    if (exif->iso_speed < 0) {
        fprintf(stderr, "无效的ISO值: %d\n", exif->iso_speed);
        return false;
    }
    
    // 检查焦距
    if (exif->focal_length < 0) {
        fprintf(stderr, "无效的焦距: %f\n", exif->focal_length);
        return false;
    }
    
    // 检查GPS坐标范围
    if (exif->gps.has_gps) {
        if (exif->gps.latitude < -90.0 || exif->gps.latitude > 90.0) {
            fprintf(stderr, "无效的GPS纬度: %f\n", exif->gps.latitude);
            return false;
        }
        if (exif->gps.longitude < -180.0 || exif->gps.longitude > 180.0) {
            fprintf(stderr, "无效的GPS经度: %f\n", exif->gps.longitude);
            return false;
        }
    }
    
    return true;
}

/**
 * @brief 清空EXIF元数据
 */
void jpeg_clear_exif(ExifMetadata *exif)
{
    if (exif) {
        memset(exif, 0, sizeof(ExifMetadata));
        exif->orientation = EXIF_ORIENTATION_NORMAL;
    }
}

/**
 * @brief 克隆EXIF元数据
 */
bool jpeg_clone_exif(const ExifMetadata *src, ExifMetadata *dst)
{
    if (!src || !dst) {
        return false;
    }
    
    memcpy(dst, src, sizeof(ExifMetadata));
    return true;
}

// 第三部分C结束
// ============================================================================
// 渐进式JPEG处理
// ============================================================================

/**
 * @brief 写入渐进式JPEG
 */
bool jpeg_write_progressive(const Image *img, const char *filename, int quality)
{
    if (!img || !filename) {
        fprintf(stderr, "参数无效\n");
        return false;
    }
    
    printf("写入渐进式JPEG: %s (质量: %d)\n", filename, quality);
    
    // 打开输出文件
    FILE *outfile = fopen(filename, "wb");
    if (!outfile) {
        fprintf(stderr, "无法创建文件: %s\n", filename);
        return false;
    }
    
    // 初始化JPEG压缩对象
    struct jpeg_compress_struct cinfo;
    struct jpeg_error_mgr jerr;
    
    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_compress(&cinfo);
    jpeg_stdio_dest(&cinfo, outfile);
    
    // 设置图像参数
    cinfo.image_width = img->width;
    cinfo.image_height = img->height;
    cinfo.input_components = img->channels;
    
    if (img->channels == 1) {
        cinfo.in_color_space = JCS_GRAYSCALE;
    } else if (img->channels == 3) {
        cinfo.in_color_space = JCS_RGB;
    } else {
        fprintf(stderr, "不支持的通道数: %d\n", img->channels);
        jpeg_destroy_compress(&cinfo);
        fclose(outfile);
        return false;
    }
    
    // 设置默认压缩参数
    jpeg_set_defaults(&cinfo);
    jpeg_set_quality(&cinfo, quality, TRUE);
    
    // 启用渐进式模式
    jpeg_simple_progression(&cinfo);
    
    // 开始压缩
    jpeg_start_compress(&cinfo, TRUE);
    
    // 写入扫描线
    JSAMPROW row_pointer[1];
    int row_stride = img->width * img->channels;
    
    while (cinfo.next_scanline < cinfo.image_height) {
        row_pointer[0] = &img->data[cinfo.next_scanline * row_stride];
        jpeg_write_scanlines(&cinfo, row_pointer, 1);
    }
    
    // 完成压缩
    jpeg_finish_compress(&cinfo);
    jpeg_destroy_compress(&cinfo);
    fclose(outfile);
    
    printf("成功写入渐进式JPEG\n");
    return true;
}

/**
 * @brief 检查是否为渐进式JPEG
 */
bool jpeg_is_progressive(const char *filename)
{
    if (!filename) {
        return false;
    }
    
    FILE *infile = fopen(filename, "rb");
    if (!infile) {
        return false;
    }
    
    struct jpeg_decompress_struct cinfo;
    struct jpeg_error_mgr jerr;
    
    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_decompress(&cinfo);
    jpeg_stdio_src(&cinfo, infile);
    
    // 读取JPEG头
    jpeg_read_header(&cinfo, TRUE);
    
    // 检查是否为渐进式
    bool is_progressive = (cinfo.progressive_mode != 0);
    
    jpeg_destroy_decompress(&cinfo);
    fclose(infile);
    
    return is_progressive;
}

/**
 * @brief 转换为渐进式JPEG
 */
bool jpeg_convert_to_progressive(const char *input_filename, 
                                 const char *output_filename,
                                 int quality)
{
    if (!input_filename || !output_filename) {
        fprintf(stderr, "文件名无效\n");
        return false;
    }
    
    printf("转换为渐进式JPEG: %s -> %s\n", input_filename, output_filename);
    
    // 读取图像
    Image *img = jpeg_read(input_filename);
    if (!img) {
        fprintf(stderr, "读取JPEG失败\n");
        return false;
    }
    
    // 写入渐进式JPEG
    bool result = jpeg_write_progressive(img, output_filename, quality);
    
    // 复制EXIF数据
    if (result) {
        jpeg_copy_exif(input_filename, output_filename);
    }
    
    image_destroy(img);
    return result;
}

// ============================================================================
// JPEG优化
// ============================================================================

/**
 * @brief 优化JPEG文件（无损优化）
 */
bool jpeg_optimize(const char *input_filename, const char *output_filename)
{
    if (!input_filename || !output_filename) {
        fprintf(stderr, "文件名无效\n");
        return false;
    }
    
    printf("优化JPEG: %s -> %s\n", input_filename, output_filename);
    
    // 打开输入文件
    FILE *infile = fopen(input_filename, "rb");
    if (!infile) {
        fprintf(stderr, "无法打开输入文件: %s\n", input_filename);
        return false;
    }
    
    // 打开输出文件
    FILE *outfile = fopen(output_filename, "wb");
    if (!outfile) {
        fprintf(stderr, "无法创建输出文件: %s\n", output_filename);
        fclose(infile);
        return false;
    }
    
    // 初始化解压缩对象
    struct jpeg_decompress_struct srcinfo;
    struct jpeg_error_mgr jsrcerr;
    srcinfo.err = jpeg_std_error(&jsrcerr);
    jpeg_create_decompress(&srcinfo);
    jpeg_stdio_src(&srcinfo, infile);
    
    // 读取JPEG头
    jpeg_read_header(&srcinfo, TRUE);
    
    // 初始化压缩对象
    struct jpeg_compress_struct dstinfo;
    struct jpeg_error_mgr jdsterr;
    dstinfo.err = jpeg_std_error(&jdsterr);
    jpeg_create_compress(&dstinfo);
    jpeg_stdio_dest(&dstinfo, outfile);
    
    // 复制压缩参数
    dstinfo.image_width = srcinfo.image_width;
    dstinfo.image_height = srcinfo.image_height;
    dstinfo.input_components = srcinfo.num_components;
    dstinfo.in_color_space = srcinfo.jpeg_color_space;
    
    jpeg_set_defaults(&dstinfo);
    
    // 复制量化表
    jpeg_copy_critical_parameters(&srcinfo, &dstinfo);
    
    // 优化霍夫曼表
    dstinfo.optimize_coding = TRUE;
    
    // 开始压缩
    jpeg_start_compress(&dstinfo, TRUE);
    
    // 开始解压缩
    jpeg_start_decompress(&srcinfo);
    
    // 分配扫描线缓冲区
    JSAMPARRAY buffer = (*srcinfo.mem->alloc_sarray)
        ((j_common_ptr)&srcinfo, JPOOL_IMAGE,
         srcinfo.output_width * srcinfo.output_components, 1);
    
    // 复制图像数据
    while (srcinfo.output_scanline < srcinfo.output_height) {
        jpeg_read_scanlines(&srcinfo, buffer, 1);
        jpeg_write_scanlines(&dstinfo, buffer, 1);
    }
    
    // 完成压缩和解压缩
    jpeg_finish_compress(&dstinfo);
    jpeg_finish_decompress(&srcinfo);
    
    // 清理
    jpeg_destroy_compress(&dstinfo);
    jpeg_destroy_decompress(&srcinfo);
    fclose(outfile);
    fclose(infile);
    
    printf("成功优化JPEG\n");
    return true;
}

/**
 * @brief 优化JPEG文件大小（有损优化）
 */
bool jpeg_optimize_size(const char *input_filename, 
                       const char *output_filename,
                       int target_quality)
{
    if (!input_filename || !output_filename) {
        fprintf(stderr, "文件名无效\n");
        return false;
    }
    
    printf("优化JPEG大小: %s -> %s (目标质量: %d)\n", 
           input_filename, output_filename, target_quality);
    
    // 读取图像
    Image *img = jpeg_read(input_filename);
    if (!img) {
        fprintf(stderr, "读取JPEG失败\n");
        return false;
    }
    
    // 以目标质量写入
    bool result = jpeg_write(img, output_filename, target_quality);
    
    // 复制EXIF数据
    if (result) {
        jpeg_copy_exif(input_filename, output_filename);
    }
    
    image_destroy(img);
    
    if (result) {
        // 显示文件大小变化
        struct stat st_in, st_out;
        if (stat(input_filename, &st_in) == 0 && 
            stat(output_filename, &st_out) == 0) {
            long size_in = st_in.st_size;
            long size_out = st_out.st_size;
            double ratio = 100.0 * (size_in - size_out) / size_in;
            
            printf("文件大小: %ld -> %ld 字节 (减少 %.1f%%)\n",
                   size_in, size_out, ratio);
        }
    }
    
    return result;
}

/**
 * @brief 自动优化JPEG质量
 */
bool jpeg_auto_optimize(const char *input_filename,
                       const char *output_filename,
                       long target_size_kb)
{
    if (!input_filename || !output_filename || target_size_kb <= 0) {
        fprintf(stderr, "参数无效\n");
        return false;
    }
    
    printf("自动优化JPEG: %s -> %s (目标大小: %ld KB)\n",
           input_filename, output_filename, target_size_kb);
    
    // 读取图像
    Image *img = jpeg_read(input_filename);
    if (!img) {
        fprintf(stderr, "读取JPEG失败\n");
        return false;
    }
    
    // 二分查找最佳质量
    int quality_min = 1;
    int quality_max = 100;
    int best_quality = 85;
    long target_size = target_size_kb * 1024;
    
    char temp_filename[512];
    snprintf(temp_filename, sizeof(temp_filename), "%s.tmp", output_filename);
    
    while (quality_min <= quality_max) {
        int quality = (quality_min + quality_max) / 2;
        
        // 尝试当前质量
        if (!jpeg_write(img, temp_filename, quality)) {
            fprintf(stderr, "写入临时文件失败\n");
            image_destroy(img);
            return false;
        }
        
        // 检查文件大小
        struct stat st;
        if (stat(temp_filename, &st) != 0) {
            fprintf(stderr, "获取文件大小失败\n");
            image_destroy(img);
            remove(temp_filename);
            return false;
        }
        
        long file_size = st.st_size;
        
        printf("  质量 %d: %ld 字节\n", quality, file_size);
        
        if (file_size <= target_size) {
            // 文件大小符合要求，尝试更高质量
            best_quality = quality;
            quality_min = quality + 1;
        } else {
            // 文件太大，降低质量
            quality_max = quality - 1;
        }
    }
    
    // 使用最佳质量写入最终文件
    bool result = jpeg_write(img, output_filename, best_quality);
    
    // 复制EXIF数据
    if (result) {
        jpeg_copy_exif(input_filename, output_filename);
        
        struct stat st;
        if (stat(output_filename, &st) == 0) {
            printf("最终质量: %d, 文件大小: %ld 字节 (%.1f KB)\n",
                   best_quality, st.st_size, st.st_size / 1024.0);
        }
    }
    
    // 清理
    image_destroy(img);
    remove(temp_filename);
    
    return result;
}

/**
 * @brief 批量优化JPEG文件
 */
bool jpeg_batch_optimize(const char **input_files, int num_files,
                        const char *output_dir, int quality)
{
    if (!input_files || num_files <= 0 || !output_dir) {
        fprintf(stderr, "参数无效\n");
        return false;
    }
    
    printf("批量优化 %d 个JPEG文件\n", num_files);
    
    // 创建输出目录
    mkdir(output_dir, 0755);
    
    int success_count = 0;
    int fail_count = 0;
    
    for (int i = 0; i < num_files; i++) {
        const char *input_file = input_files[i];
        
        // 提取文件名
        const char *filename = strrchr(input_file, '/');
        if (!filename) {
            filename = strrchr(input_file, '\\');
        }
        if (!filename) {
            filename = input_file;
        } else {
            filename++;
        }
        
        // 构建输出文件路径
        char output_file[512];
        snprintf(output_file, sizeof(output_file), "%s/%s", output_dir, filename);
        
        printf("[%d/%d] 优化: %s\n", i + 1, num_files, filename);
        
        // 优化文件
        if (jpeg_optimize_size(input_file, output_file, quality)) {
            success_count++;
        } else {
            fail_count++;
            fprintf(stderr, "优化失败: %s\n", input_file);
        }
    }
    
    printf("\n批量优化完成: 成功 %d, 失败 %d\n", success_count, fail_count);
    
    return (fail_count == 0);
}

// ============================================================================
// JPEG质量评估
// ============================================================================

/**
 * @brief 估算JPEG质量
 */
int jpeg_estimate_quality(const char *filename)
{
    if (!filename) {
        return -1;
    }
    
    FILE *infile = fopen(filename, "rb");
    if (!infile) {
        return -1;
    }
    
    struct jpeg_decompress_struct cinfo;
    struct jpeg_error_mgr jerr;
    
    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_decompress(&cinfo);
    jpeg_stdio_src(&cinfo, infile);
    
    // 读取JPEG头
    jpeg_read_header(&cinfo, TRUE);
    
    // 获取量化表
    int quality = -1;
    
    if (cinfo.quant_tbl_ptrs[0] != NULL) {
        // 计算量化表的平均值
        int sum = 0;
        for (int i = 0; i < DCTSIZE2; i++) {
            sum += cinfo.quant_tbl_ptrs[0]->quantval[i];
        }
        int avg = sum / DCTSIZE2;
        
        // 根据平均值估算质量
        // 这是一个简化的估算方法
        if (avg <= 2) {
            quality = 100;
        } else if (avg <= 4) {
            quality = 95;
        } else if (avg <= 8) {
            quality = 90;
        } else if (avg <= 16) {
            quality = 85;
        } else if (avg <= 24) {
            quality = 80;
        } else if (avg <= 32) {
            quality = 75;
        } else if (avg <= 48) {
            quality = 70;
        } else if (avg <= 64) {
            quality = 60;
        } else if (avg <= 96) {
            quality = 50;
        } else {
            quality = 40;
        }
    }
    
    jpeg_destroy_decompress(&cinfo);
    fclose(infile);
    
    return quality;
}

/**
 * @brief 计算JPEG压缩比
 */
double jpeg_calculate_compression_ratio(const char *filename)
{
    if (!filename) {
        return 0.0;
    }
    
    // 获取文件大小
    struct stat st;
    if (stat(filename, &st) != 0) {
        return 0.0;
    }
    long compressed_size = st.st_size;
    
    // 读取图像信息
    FILE *infile = fopen(filename, "rb");
    if (!infile) {
        return 0.0;
    }
    
    struct jpeg_decompress_struct cinfo;
    struct jpeg_error_mgr jerr;
    
    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_decompress(&cinfo);
    jpeg_stdio_src(&cinfo, infile);
    jpeg_read_header(&cinfo, TRUE);
    
    // 计算未压缩大小
    long uncompressed_size = (long)cinfo.image_width * 
                            cinfo.image_height * 
                            cinfo.num_components;
    
    jpeg_destroy_decompress(&cinfo);
    fclose(infile);
    
    if (uncompressed_size == 0) {
        return 0.0;
    }
    
    return (double)uncompressed_size / compressed_size;
}

// 第四部分A结束
// ============================================================================
// 色彩空间转换
// ============================================================================

/**
 * @brief RGB转YCbCr
 */
static void rgb_to_ycbcr(unsigned char r, unsigned char g, unsigned char b,
                         unsigned char *y, unsigned char *cb, unsigned char *cr)
{
    *y  = (unsigned char)(0.299 * r + 0.587 * g + 0.114 * b);
    *cb = (unsigned char)(128 - 0.168736 * r - 0.331264 * g + 0.5 * b);
    *cr = (unsigned char)(128 + 0.5 * r - 0.418688 * g - 0.081312 * b);
}

/**
 * @brief YCbCr转RGB
 */
static void ycbcr_to_rgb(unsigned char y, unsigned char cb, unsigned char cr,
                         unsigned char *r, unsigned char *g, unsigned char *b)
{
    int r_tmp = y + 1.402 * (cr - 128);
    int g_tmp = y - 0.344136 * (cb - 128) - 0.714136 * (cr - 128);
    int b_tmp = y + 1.772 * (cb - 128);
    
    *r = (unsigned char)CLAMP(r_tmp, 0, 255);
    *g = (unsigned char)CLAMP(g_tmp, 0, 255);
    *b = (unsigned char)CLAMP(b_tmp, 0, 255);
}

/**
 * @brief 转换图像色彩空间为YCbCr
 */
Image* jpeg_convert_to_ycbcr(const Image *img)
{
    if (!img || img->channels != 3) {
        fprintf(stderr, "只支持RGB图像转换\n");
        return NULL;
    }
    
    printf("转换色彩空间: RGB -> YCbCr\n");
    
    Image *result = image_create(img->width, img->height, 3);
    if (!result) {
        return NULL;
    }
    
    for (int i = 0; i < img->width * img->height; i++) {
        int idx = i * 3;
        unsigned char r = img->data[idx];
        unsigned char g = img->data[idx + 1];
        unsigned char b = img->data[idx + 2];
        
        rgb_to_ycbcr(r, g, b,
                    &result->data[idx],
                    &result->data[idx + 1],
                    &result->data[idx + 2]);
    }
    
    return result;
}

/**
 * @brief 转换图像色彩空间为RGB
 */
Image* jpeg_convert_to_rgb(const Image *img)
{
    if (!img || img->channels != 3) {
        fprintf(stderr, "只支持YCbCr图像转换\n");
        return NULL;
    }
    
    printf("转换色彩空间: YCbCr -> RGB\n");
    
    Image *result = image_create(img->width, img->height, 3);
    if (!result) {
        return NULL;
    }
    
    for (int i = 0; i < img->width * img->height; i++) {
        int idx = i * 3;
        unsigned char y = img->data[idx];
        unsigned char cb = img->data[idx + 1];
        unsigned char cr = img->data[idx + 2];
        
        ycbcr_to_rgb(y, cb, cr,
                    &result->data[idx],
                    &result->data[idx + 1],
                    &result->data[idx + 2]);
    }
    
    return result;
}

/**
 * @brief 转换为灰度图像
 */
Image* jpeg_convert_to_grayscale(const Image *img)
{
    if (!img) {
        return NULL;
    }
    
    printf("转换为灰度图像\n");
    
    if (img->channels == 1) {
        // 已经是灰度图像，直接复制
        return image_clone(img);
    }
    
    Image *result = image_create(img->width, img->height, 1);
    if (!result) {
        return NULL;
    }
    
    for (int i = 0; i < img->width * img->height; i++) {
        if (img->channels == 3) {
            int idx = i * 3;
            unsigned char r = img->data[idx];
            unsigned char g = img->data[idx + 1];
            unsigned char b = img->data[idx + 2];
            
            // 使用标准灰度转换公式
            result->data[i] = (unsigned char)(0.299 * r + 0.587 * g + 0.114 * b);
        } else if (img->channels == 4) {
            int idx = i * 4;
            unsigned char r = img->data[idx];
            unsigned char g = img->data[idx + 1];
            unsigned char b = img->data[idx + 2];
            
            result->data[i] = (unsigned char)(0.299 * r + 0.587 * g + 0.114 * b);
        }
    }
    
    return result;
}

// ============================================================================
// 色度子采样
// ============================================================================

/**
 * @brief 应用4:2:0色度子采样
 */
Image* jpeg_apply_chroma_subsampling_420(const Image *img)
{
    if (!img || img->channels != 3) {
        fprintf(stderr, "只支持RGB图像\n");
        return NULL;
    }
    
    printf("应用4:2:0色度子采样\n");
    
    // 转换为YCbCr
    Image *ycbcr = jpeg_convert_to_ycbcr(img);
    if (!ycbcr) {
        return NULL;
    }
    
    // 对Cb和Cr通道进行2x2下采样
    int width = img->width;
    int height = img->height;
    
    for (int y = 0; y < height; y += 2) {
        for (int x = 0; x < width; x += 2) {
            // 计算2x2块的平均Cb和Cr值
            int sum_cb = 0, sum_cr = 0;
            int count = 0;
            
            for (int dy = 0; dy < 2 && (y + dy) < height; dy++) {
                for (int dx = 0; dx < 2 && (x + dx) < width; dx++) {
                    int idx = ((y + dy) * width + (x + dx)) * 3;
                    sum_cb += ycbcr->data[idx + 1];
                    sum_cr += ycbcr->data[idx + 2];
                    count++;
                }
            }
            
            unsigned char avg_cb = sum_cb / count;
            unsigned char avg_cr = sum_cr / count;
            
            // 将平均值应用到2x2块
            for (int dy = 0; dy < 2 && (y + dy) < height; dy++) {
                for (int dx = 0; dx < 2 && (x + dx) < width; dx++) {
                    int idx = ((y + dy) * width + (x + dx)) * 3;
                    ycbcr->data[idx + 1] = avg_cb;
                    ycbcr->data[idx + 2] = avg_cr;
                }
            }
        }
    }
    
    // 转换回RGB
    Image *result = jpeg_convert_to_rgb(ycbcr);
    image_destroy(ycbcr);
    
    return result;
}

/**
 * @brief 应用4:2:2色度子采样
 */
Image* jpeg_apply_chroma_subsampling_422(const Image *img)
{
    if (!img || img->channels != 3) {
        fprintf(stderr, "只支持RGB图像\n");
        return NULL;
    }
    
    printf("应用4:2:2色度子采样\n");
    
    // 转换为YCbCr
    Image *ycbcr = jpeg_convert_to_ycbcr(img);
    if (!ycbcr) {
        return NULL;
    }
    
    // 对Cb和Cr通道进行水平2:1下采样
    int width = img->width;
    int height = img->height;
    
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x += 2) {
            // 计算相邻两个像素的平均Cb和Cr值
            int idx1 = (y * width + x) * 3;
            int idx2 = (y * width + x + 1) * 3;
            
            unsigned char avg_cb, avg_cr;
            
            if (x + 1 < width) {
                avg_cb = (ycbcr->data[idx1 + 1] + ycbcr->data[idx2 + 1]) / 2;
                avg_cr = (ycbcr->data[idx1 + 2] + ycbcr->data[idx2 + 2]) / 2;
                
                ycbcr->data[idx1 + 1] = avg_cb;
                ycbcr->data[idx1 + 2] = avg_cr;
                ycbcr->data[idx2 + 1] = avg_cb;
                ycbcr->data[idx2 + 2] = avg_cr;
            }
        }
    }
    
    // 转换回RGB
    Image *result = jpeg_convert_to_rgb(ycbcr);
    image_destroy(ycbcr);
    
    return result;
}

// ============================================================================
// JPEG伪影处理
// ============================================================================

/**
 * @brief 检测JPEG块效应
 */
double jpeg_detect_blocking_artifacts(const Image *img)
{
    if (!img || img->channels != 3) {
        fprintf(stderr, "只支持RGB图像\n");
        return 0.0;
    }
    
    printf("检测JPEG块效应\n");
    
    // 转换为灰度图像
    Image *gray = jpeg_convert_to_grayscale(img);
    if (!gray) {
        return 0.0;
    }
    
    int width = gray->width;
    int height = gray->height;
    double total_diff = 0.0;
    int count = 0;
    
    // 检测8x8块边界的不连续性
    for (int y = 8; y < height; y += 8) {
        for (int x = 0; x < width; x++) {
            int idx1 = (y - 1) * width + x;
            int idx2 = y * width + x;
            
            int diff = abs(gray->data[idx1] - gray->data[idx2]);
            total_diff += diff;
            count++;
        }
    }
    
    for (int y = 0; y < height; y++) {
        for (int x = 8; x < width; x += 8) {
            int idx1 = y * width + (x - 1);
            int idx2 = y * width + x;
            
            int diff = abs(gray->data[idx1] - gray->data[idx2]);
            total_diff += diff;
            count++;
        }
    }
    
    image_destroy(gray);
    
    if (count == 0) {
        return 0.0;
    }
    
    double avg_diff = total_diff / count;
    printf("平均块边界差异: %.2f\n", avg_diff);
    
    return avg_diff;
}

/**
 * @brief 减少JPEG块效应（简单平滑）
 */
Image* jpeg_reduce_blocking_artifacts(const Image *img, int strength)
{
    if (!img || img->channels != 3) {
        fprintf(stderr, "只支持RGB图像\n");
        return NULL;
    }
    
    printf("减少JPEG块效应 (强度: %d)\n", strength);
    
    Image *result = image_clone(img);
    if (!result) {
        return NULL;
    }
    
    int width = img->width;
    int height = img->height;
    
    // 对8x8块边界进行平滑
    for (int y = 8; y < height; y += 8) {
        for (int x = 0; x < width; x++) {
            for (int c = 0; c < 3; c++) {
                int idx1 = ((y - 1) * width + x) * 3 + c;
                int idx2 = (y * width + x) * 3 + c;
                
                int val1 = result->data[idx1];
                int val2 = result->data[idx2];
                int avg = (val1 + val2) / 2;
                
                // 根据强度进行混合
                result->data[idx1] = (val1 * (100 - strength) + avg * strength) / 100;
                result->data[idx2] = (val2 * (100 - strength) + avg * strength) / 100;
            }
        }
    }
    
    for (int y = 0; y < height; y++) {
        for (int x = 8; x < width; x += 8) {
            for (int c = 0; c < 3; c++) {
                int idx1 = (y * width + (x - 1)) * 3 + c;
                int idx2 = (y * width + x) * 3 + c;
                
                int val1 = result->data[idx1];
                int val2 = result->data[idx2];
                int avg = (val1 + val2) / 2;
                
                result->data[idx1] = (val1 * (100 - strength) + avg * strength) / 100;
                result->data[idx2] = (val2 * (100 - strength) + avg * strength) / 100;
            }
        }
    }
    
    return result;
}

/**
 * @brief 锐化JPEG图像
 */
Image* jpeg_sharpen(const Image *img, double amount)
{
    if (!img) {
        return NULL;
    }
    
    printf("锐化图像 (强度: %.2f)\n", amount);
    
    Image *result = image_clone(img);
    if (!result) {
        return NULL;
    }
    
    int width = img->width;
    int height = img->height;
    int channels = img->channels;
    
    // 使用拉普拉斯算子进行锐化
    // 锐化核: [-1 -1 -1]
    //         [-1  9 -1]
    //         [-1 -1 -1]
    
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            for (int c = 0; c < channels; c++) {
                int sum = 0;
                
                // 应用锐化核
                for (int dy = -1; dy <= 1; dy++) {
                    for (int dx = -1; dx <= 1; dx++) {
                        int idx = ((y + dy) * width + (x + dx)) * channels + c;
                        int weight = (dx == 0 && dy == 0) ? 9 : -1;
                        sum += img->data[idx] * weight;
                    }
                }
                
                // 混合原始值和锐化值
                int idx = (y * width + x) * channels + c;
                int original = img->data[idx];
                int sharpened = sum;
                int final = original + (sharpened - original) * amount;
                
                result->data[idx] = CLAMP(final, 0, 255);
            }
        }
    }
    
    return result;
}

/**
 * @brief 应用去噪滤波
 */
Image* jpeg_denoise(const Image *img, int radius)
{
    if (!img) {
        return NULL;
    }
    
    printf("去噪处理 (半径: %d)\n", radius);
    
    Image *result = image_create(img->width, img->height, img->channels);
    if (!result) {
        return NULL;
    }
    
    int width = img->width;
    int height = img->height;
    int channels = img->channels;
    
    // 使用中值滤波进行去噪
    int window_size = radius * 2 + 1;
    int *values = (int*)malloc(window_size * window_size * sizeof(int));
    
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            for (int c = 0; c < channels; c++) {
                int count = 0;
                
                // 收集窗口内的值
                for (int dy = -radius; dy <= radius; dy++) {
                    for (int dx = -radius; dx <= radius; dx++) {
                        int ny = y + dy;
                        int nx = x + dx;
                        
                        if (ny >= 0 && ny < height && nx >= 0 && nx < width) {
                            int idx = (ny * width + nx) * channels + c;
                            values[count++] = img->data[idx];
                        }
                    }
                }
                
                // 排序并取中值
                for (int i = 0; i < count - 1; i++) {
                    for (int j = i + 1; j < count; j++) {
                        if (values[i] > values[j]) {
                            int temp = values[i];
                            values[i] = values[j];
                            values[j] = temp;
                        }
                    }
                }
                
                int idx = (y * width + x) * channels + c;
                result->data[idx] = values[count / 2];
            }
        }
    }
    
    free(values);
    return result;
}

// ============================================================================
// JPEG质量增强
// ============================================================================

/**
 * @brief 增强JPEG图像质量
 */
Image* jpeg_enhance_quality(const Image *img)
{
    if (!img) {
        return NULL;
    }
    
    printf("增强JPEG图像质量\n");
    
    // 1. 减少块效应
    Image *temp1 = jpeg_reduce_blocking_artifacts(img, 30);
    if (!temp1) {
        return NULL;
    }
    
    // 2. 去噪
    Image *temp2 = jpeg_denoise(temp1, 1);
    image_destroy(temp1);
    if (!temp2) {
        return NULL;
    }
    
    // 3. 轻微锐化
    Image *result = jpeg_sharpen(temp2, 0.3);
    image_destroy(temp2);
    
    return result;
}

/**
 * @brief 自适应增强JPEG图像
 */
Image* jpeg_adaptive_enhance(const Image *img)
{
    if (!img) {
        return NULL;
    }
    
    printf("自适应增强JPEG图像\n");
    
    // 检测块效应程度
    double blocking = jpeg_detect_blocking_artifacts(img);
    
    // 根据块效应程度选择处理强度
    int denoise_strength = 1;
    int deblock_strength = 30;
    double sharpen_amount = 0.3;
    
    if (blocking > 20.0) {
        // 严重块效应
        denoise_strength = 2;
        deblock_strength = 50;
        sharpen_amount = 0.2;
        printf("检测到严重块效应，使用强力处理\n");
    } else if (blocking > 10.0) {
        // 中等块效应
        denoise_strength = 1;
        deblock_strength = 30;
        sharpen_amount = 0.3;
        printf("检测到中等块效应，使用标准处理\n");
    } else {
        // 轻微块效应
        denoise_strength = 1;
        deblock_strength = 20;
        sharpen_amount = 0.4;
        printf("检测到轻微块效应，使用轻度处理\n");
    }
    
    // 应用处理
    Image *temp1 = jpeg_reduce_blocking_artifacts(img, deblock_strength);
    if (!temp1) {
        return NULL;
    }
    
    Image *temp2 = jpeg_denoise(temp1, denoise_strength);
    image_destroy(temp1);
    if (!temp2) {
        return NULL;
    }
    
    Image *result = jpeg_sharpen(temp2, sharpen_amount);
    image_destroy(temp2);
    
    return result;
}

// 第四部分B结束
// ============================================================================
// 缩略图处理
// ============================================================================

/**
 * @brief 从JPEG提取嵌入的缩略图
 */
Image* jpeg_extract_thumbnail(const char *filename)
{
    if (!filename) {
        fprintf(stderr, "文件名无效\n");
        return NULL;
    }
    
    printf("提取JPEG缩略图: %s\n", filename);
    
    FILE *infile = fopen(filename, "rb");
    if (!infile) {
        fprintf(stderr, "无法打开文件: %s\n", filename);
        return NULL;
    }
    
    struct jpeg_decompress_struct cinfo;
    struct jpeg_error_mgr jerr;
    
    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_decompress(&cinfo);
    jpeg_stdio_src(&cinfo, infile);
    
    // 读取JPEG头，包括EXIF数据
    jpeg_save_markers(&cinfo, JPEG_APP0 + 1, 0xFFFF);  // APP1 (EXIF)
    jpeg_read_header(&cinfo, TRUE);
    
    Image *thumbnail = NULL;
    
    // 查找EXIF标记
    jpeg_saved_marker_ptr marker = cinfo.marker_list;
    while (marker != NULL) {
        if (marker->marker == JPEG_APP0 + 1) {  // APP1
            // 检查是否为EXIF标记
            if (marker->data_length > 6 &&
                memcmp(marker->data, "Exif\0\0", 6) == 0) {
                
                // 解析EXIF数据查找缩略图
                unsigned char *exif_data = marker->data + 6;
                int exif_length = marker->data_length - 6;
                
                // 简化的缩略图提取（实际实现需要完整的EXIF解析）
                // 这里只是示例代码
                printf("找到EXIF数据，长度: %d\n", exif_length);
                
                // TODO: 实现完整的EXIF缩略图提取
                // 这需要解析TIFF格式的EXIF数据
            }
        }
        marker = marker->next;
    }
    
    jpeg_destroy_decompress(&cinfo);
    fclose(infile);
    
    if (!thumbnail) {
        printf("未找到嵌入的缩略图\n");
    }
    
    return thumbnail;
}

/**
 * @brief 生成JPEG缩略图
 */
Image* jpeg_create_thumbnail(const Image *img, int max_width, int max_height)
{
    if (!img || max_width <= 0 || max_height <= 0) {
        fprintf(stderr, "参数无效\n");
        return NULL;
    }
    
    printf("生成缩略图: %dx%d -> 最大 %dx%d\n", 
           img->width, img->height, max_width, max_height);
    
    // 计算缩放比例
    double scale_w = (double)max_width / img->width;
    double scale_h = (double)max_height / img->height;
    double scale = (scale_w < scale_h) ? scale_w : scale_h;
    
    // 如果图像已经小于目标尺寸，不需要缩放
    if (scale >= 1.0) {
        printf("图像已经小于目标尺寸，返回原图\n");
        return image_clone(img);
    }
    
    int new_width = (int)(img->width * scale);
    int new_height = (int)(img->height * scale);
    
    printf("缩略图尺寸: %dx%d (缩放比例: %.2f)\n", 
           new_width, new_height, scale);
    
    return image_resize(img, new_width, new_height);
}

/**
 * @brief 生成并保存JPEG缩略图
 */
bool jpeg_save_thumbnail(const char *input_filename,
                        const char *output_filename,
                        int max_width, int max_height,
                        int quality)
{
    if (!input_filename || !output_filename) {
        fprintf(stderr, "文件名无效\n");
        return false;
    }
    
    printf("生成缩略图: %s -> %s\n", input_filename, output_filename);
    
    // 读取原图
    Image *img = jpeg_read(input_filename);
    if (!img) {
        fprintf(stderr, "读取图像失败\n");
        return false;
    }
    
    // 生成缩略图
    Image *thumbnail = jpeg_create_thumbnail(img, max_width, max_height);
    image_destroy(img);
    
    if (!thumbnail) {
        fprintf(stderr, "生成缩略图失败\n");
        return false;
    }
    
    // 保存缩略图
    bool result = jpeg_write(thumbnail, output_filename, quality);
    image_destroy(thumbnail);
    
    if (result) {
        printf("成功保存缩略图\n");
    }
    
    return result;
}

/**
 * @brief 批量生成缩略图
 */
bool jpeg_batch_create_thumbnails(const char **input_files, int num_files,
                                  const char *output_dir,
                                  int max_width, int max_height,
                                  int quality)
{
    if (!input_files || num_files <= 0 || !output_dir) {
        fprintf(stderr, "参数无效\n");
        return false;
    }
    
    printf("批量生成 %d 个缩略图\n", num_files);
    printf("输出目录: %s\n", output_dir);
    printf("最大尺寸: %dx%d\n", max_width, max_height);
    printf("质量: %d\n", quality);
    
    // 创建输出目录
    mkdir(output_dir, 0755);
    
    int success_count = 0;
    int fail_count = 0;
    
    for (int i = 0; i < num_files; i++) {
        const char *input_file = input_files[i];
        
        // 提取文件名
        const char *filename = strrchr(input_file, '/');
        if (!filename) {
            filename = strrchr(input_file, '\\');
        }
        if (!filename) {
            filename = input_file;
        } else {
            filename++;
        }
        
        // 构建输出文件路径
        char output_file[512];
        snprintf(output_file, sizeof(output_file), "%s/thumb_%s", 
                output_dir, filename);
        
        printf("[%d/%d] 处理: %s\n", i + 1, num_files, filename);
        
        // 生成缩略图
        if (jpeg_save_thumbnail(input_file, output_file, 
                               max_width, max_height, quality)) {
            success_count++;
        } else {
            fail_count++;
            fprintf(stderr, "失败: %s\n", input_file);
        }
    }
    
    printf("\n批量处理完成: 成功 %d, 失败 %d\n", success_count, fail_count);
    
    return (fail_count == 0);
}

// ============================================================================
// 图像拼接和组合
// ============================================================================

/**
 * @brief 水平拼接多个JPEG图像
 */
Image* jpeg_concat_horizontal(const Image **images, int num_images)
{
    if (!images || num_images <= 0) {
        fprintf(stderr, "参数无效\n");
        return NULL;
    }
    
    printf("水平拼接 %d 个图像\n", num_images);
    
    // 计算总宽度和最大高度
    int total_width = 0;
    int max_height = 0;
    int channels = images[0]->channels;
    
    for (int i = 0; i < num_images; i++) {
        if (!images[i]) {
            fprintf(stderr, "图像 %d 无效\n", i);
            return NULL;
        }
        if (images[i]->channels != channels) {
            fprintf(stderr, "图像通道数不一致\n");
            return NULL;
        }
        
        total_width += images[i]->width;
        if (images[i]->height > max_height) {
            max_height = images[i]->height;
        }
    }
    
    printf("输出尺寸: %dx%d\n", total_width, max_height);
    
    // 创建输出图像
    Image *result = image_create(total_width, max_height, channels);
    if (!result) {
        return NULL;
    }
    
    // 填充白色背景
    memset(result->data, 255, total_width * max_height * channels);
    
    // 拼接图像
    int x_offset = 0;
    for (int i = 0; i < num_images; i++) {
        const Image *img = images[i];
        
        // 垂直居中
        int y_offset = (max_height - img->height) / 2;
        
        // 复制图像数据
        for (int y = 0; y < img->height; y++) {
            int src_idx = y * img->width * channels;
            int dst_idx = ((y + y_offset) * total_width + x_offset) * channels;
            memcpy(&result->data[dst_idx], 
                   &img->data[src_idx], 
                   img->width * channels);
        }
        
        x_offset += img->width;
    }
    
    return result;
}

/**
 * @brief 垂直拼接多个JPEG图像
 */
Image* jpeg_concat_vertical(const Image **images, int num_images)
{
    if (!images || num_images <= 0) {
        fprintf(stderr, "参数无效\n");
        return NULL;
    }
    
    printf("垂直拼接 %d 个图像\n", num_images);
    
    // 计算最大宽度和总高度
    int max_width = 0;
    int total_height = 0;
    int channels = images[0]->channels;
    
    for (int i = 0; i < num_images; i++) {
        if (!images[i]) {
            fprintf(stderr, "图像 %d 无效\n", i);
            return NULL;
        }
        if (images[i]->channels != channels) {
            fprintf(stderr, "图像通道数不一致\n");
            return NULL;
        }
        
        if (images[i]->width > max_width) {
            max_width = images[i]->width;
        }
        total_height += images[i]->height;
    }
    
    printf("输出尺寸: %dx%d\n", max_width, total_height);
    
    // 创建输出图像
    Image *result = image_create(max_width, total_height, channels);
    if (!result) {
        return NULL;
    }
    
    // 填充白色背景
    memset(result->data, 255, max_width * total_height * channels);
    
    // 拼接图像
    int y_offset = 0;
    for (int i = 0; i < num_images; i++) {
        const Image *img = images[i];
        
        // 水平居中
        int x_offset = (max_width - img->width) / 2;
        
        // 复制图像数据
        for (int y = 0; y < img->height; y++) {
            int src_idx = y * img->width * channels;
            int dst_idx = ((y + y_offset) * max_width + x_offset) * channels;
            memcpy(&result->data[dst_idx], 
                   &img->data[src_idx], 
                   img->width * channels);
        }
        
        y_offset += img->height;
    }
    
    return result;
}

/**
 * @brief 创建图像网格
 */
Image* jpeg_create_grid(const Image **images, int num_images, 
                       int cols, int spacing)
{
    if (!images || num_images <= 0 || cols <= 0) {
        fprintf(stderr, "参数无效\n");
        return NULL;
    }
    
    int rows = (num_images + cols - 1) / cols;
    printf("创建图像网格: %d行 x %d列 (间距: %d)\n", rows, cols, spacing);
    
    // 找到最大的图像尺寸
    int max_img_width = 0;
    int max_img_height = 0;
    int channels = images[0]->channels;
    
    for (int i = 0; i < num_images; i++) {
        if (!images[i]) continue;
        
        if (images[i]->width > max_img_width) {
            max_img_width = images[i]->width;
        }
        if (images[i]->height > max_img_height) {
            max_img_height = images[i]->height;
        }
    }
    
    // 计算网格总尺寸
    int grid_width = cols * max_img_width + (cols + 1) * spacing;
    int grid_height = rows * max_img_height + (rows + 1) * spacing;
    
    printf("网格尺寸: %dx%d\n", grid_width, grid_height);
    
    // 创建输出图像
    Image *result = image_create(grid_width, grid_height, channels);
    if (!result) {
        return NULL;
    }
    
    // 填充白色背景
    memset(result->data, 255, grid_width * grid_height * channels);
    
    // 放置图像
    for (int i = 0; i < num_images; i++) {
        const Image *img = images[i];
        if (!img) continue;
        
        int row = i / cols;
        int col = i % cols;
        
        // 计算位置（居中）
        int x = spacing + col * (max_img_width + spacing) + 
                (max_img_width - img->width) / 2;
        int y = spacing + row * (max_img_height + spacing) + 
                (max_img_height - img->height) / 2;
        
        // 复制图像数据
        for (int py = 0; py < img->height; py++) {
            int src_idx = py * img->width * channels;
            int dst_idx = ((y + py) * grid_width + x) * channels;
            memcpy(&result->data[dst_idx], 
                   &img->data[src_idx], 
                   img->width * channels);
        }
    }
    
    return result;
}

// ============================================================================
// 批处理工具
// ============================================================================

/**
 * @brief 批量转换JPEG质量
 */
bool jpeg_batch_convert_quality(const char **input_files, int num_files,
                                const char *output_dir, int quality)
{
    if (!input_files || num_files <= 0 || !output_dir) {
        fprintf(stderr, "参数无效\n");
        return false;
    }
    
    printf("批量转换 %d 个JPEG文件质量为 %d\n", num_files, quality);
    
    // 创建输出目录
    mkdir(output_dir, 0755);
    
    int success_count = 0;
    int fail_count = 0;
    long total_size_before = 0;
    long total_size_after = 0;
    
    for (int i = 0; i < num_files; i++) {
        const char *input_file = input_files[i];
        
        // 获取原文件大小
        struct stat st;
        if (stat(input_file, &st) == 0) {
            total_size_before += st.st_size;
        }
        
        // 提取文件名
        const char *filename = strrchr(input_file, '/');
        if (!filename) {
            filename = strrchr(input_file, '\\');
        }
        if (!filename) {
            filename = input_file;
        } else {
            filename++;
        }
        
        // 构建输出文件路径
        char output_file[512];
        snprintf(output_file, sizeof(output_file), "%s/%s", 
                output_dir, filename);
        
        printf("[%d/%d] 转换: %s\n", i + 1, num_files, filename);
        
        // 转换质量
        if (jpeg_optimize_size(input_file, output_file, quality)) {
            success_count++;
            
            // 获取新文件大小
            if (stat(output_file, &st) == 0) {
                total_size_after += st.st_size;
            }
        } else {
            fail_count++;
            fprintf(stderr, "失败: %s\n", input_file);
        }
    }
    
    printf("\n批量转换完成: 成功 %d, 失败 %d\n", success_count, fail_count);
    
    if (total_size_before > 0) {
        double ratio = 100.0 * (total_size_before - total_size_after) / 
                      total_size_before;
        printf("总大小: %ld -> %ld 字节 (减少 %.1f%%)\n",
               total_size_before, total_size_after, ratio);
    }
    
    return (fail_count == 0);
}

/**
 * @brief 批量调整JPEG尺寸
 */
bool jpeg_batch_resize(const char **input_files, int num_files,
                      const char *output_dir,
                      int target_width, int target_height,
                      int quality)
{
    if (!input_files || num_files <= 0 || !output_dir) {
        fprintf(stderr, "参数无效\n");
        return false;
    }
    
    printf("批量调整 %d 个JPEG文件尺寸为 %dx%d\n", 
           num_files, target_width, target_height);
    
    // 创建输出目录
    mkdir(output_dir, 0755);
    
    int success_count = 0;
    int fail_count = 0;
    
    for (int i = 0; i < num_files; i++) {
        const char *input_file = input_files[i];
        
        // 提取文件名
        const char *filename = strrchr(input_file, '/');
        if (!filename) {
            filename = strrchr(input_file, '\\');
        }
        if (!filename) {
            filename = input_file;
        } else {
            filename++;
        }
        
        // 构建输出文件路径
        char output_file[512];
        snprintf(output_file, sizeof(output_file), "%s/%s", 
                output_dir, filename);
        
        printf("[%d/%d] 调整: %s\n", i + 1, num_files, filename);
        
        // 读取图像
        Image *img = jpeg_read(input_file);
        if (!img) {
            fail_count++;
            fprintf(stderr, "读取失败: %s\n", input_file);
            continue;
        }
        
        // 调整尺寸
        Image *resized = image_resize(img, target_width, target_height);
        image_destroy(img);
        
        if (!resized) {
            fail_count++;
            fprintf(stderr, "调整尺寸失败: %s\n", input_file);
            continue;
        }
        
        // 保存
        if (jpeg_write(resized, output_file, quality)) {
            success_count++;
            
            // 复制EXIF数据
            jpeg_copy_exif(input_file, output_file);
        } else {
            fail_count++;
            fprintf(stderr, "保存失败: %s\n", output_file);
        }
        
        image_destroy(resized);
    }
    
    printf("\n批量调整完成: 成功 %d, 失败 %d\n", success_count, fail_count);
    
    return (fail_count == 0);
}

/**
 * @brief 批量转换为渐进式JPEG
 */
bool jpeg_batch_convert_progressive(const char **input_files, int num_files,
                                    const char *output_dir, int quality)
{
    if (!input_files || num_files <= 0 || !output_dir) {
        fprintf(stderr, "参数无效\n");
        return false;
    }
    
    printf("批量转换 %d 个文件为渐进式JPEG\n", num_files);
    
    // 创建输出目录
    mkdir(output_dir, 0755);
    
    int success_count = 0;
    int fail_count = 0;
    
    for (int i = 0; i < num_files; i++) {
        const char *input_file = input_files[i];
        
        // 提取文件名
        const char *filename = strrchr(input_file, '/');
        if (!filename) {
            filename = strrchr(input_file, '\\');
        }
        if (!filename) {
            filename = input_file;
        } else {
            filename++;
        }
        
        // 构建输出文件路径
        char output_file[512];
        snprintf(output_file, sizeof(output_file), "%s/%s", 
                output_dir, filename);
        
        printf("[%d/%d] 转换: %s\n", i + 1, num_files, filename);
        
        // 转换为渐进式
        if (jpeg_convert_to_progressive(input_file, output_file, quality)) {
            success_count++;
        } else {
            fail_count++;
            fprintf(stderr, "失败: %s\n", input_file);
        }
    }
    
    printf("\n批量转换完成: 成功 %d, 失败 %d\n", success_count, fail_count);
    
    return (fail_count == 0);
}

// 第四部分C结束
// ============================================================================
// 错误处理
// ============================================================================

/**
 * @brief 自定义JPEG错误管理器
 */
typedef struct {
    struct jpeg_error_mgr pub;  // 公共字段
    jmp_buf setjmp_buffer;      // 用于错误恢复
    char error_message[JMSG_LENGTH_MAX];
} jpeg_error_handler;

/**
 * @brief JPEG错误退出处理函数
 */
static void jpeg_error_exit(j_common_ptr cinfo)
{
    jpeg_error_handler *err = (jpeg_error_handler*)cinfo->err;
    
    // 格式化错误消息
    (*cinfo->err->format_message)(cinfo, err->error_message);
    
    // 跳转回设置点
    longjmp(err->setjmp_buffer, 1);
}

/**
 * @brief JPEG输出消息处理函数
 */
static void jpeg_output_message(j_common_ptr cinfo)
{
    char buffer[JMSG_LENGTH_MAX];
    
    // 格式化消息
    (*cinfo->err->format_message)(cinfo, buffer);
    
    // 输出到stderr
    fprintf(stderr, "JPEG警告: %s\n", buffer);
}

/**
 * @brief 安全读取JPEG（带错误处理）
 */
Image* jpeg_read_safe(const char *filename, char *error_msg, int error_msg_size)
{
    if (!filename) {
        if (error_msg) {
            snprintf(error_msg, error_msg_size, "文件名为空");
        }
        return NULL;
    }
    
    FILE *infile = fopen(filename, "rb");
    if (!infile) {
        if (error_msg) {
            snprintf(error_msg, error_msg_size, "无法打开文件: %s", filename);
        }
        return NULL;
    }
    
    struct jpeg_decompress_struct cinfo;
    jpeg_error_handler jerr;
    
    // 设置错误处理
    cinfo.err = jpeg_std_error(&jerr.pub);
    jerr.pub.error_exit = jpeg_error_exit;
    jerr.pub.output_message = jpeg_output_message;
    
    // 设置错误恢复点
    if (setjmp(jerr.setjmp_buffer)) {
        // 发生错误，清理并返回
        if (error_msg) {
            snprintf(error_msg, error_msg_size, "JPEG解码错误: %s", 
                    jerr.error_message);
        }
        jpeg_destroy_decompress(&cinfo);
        fclose(infile);
        return NULL;
    }
    
    jpeg_create_decompress(&cinfo);
    jpeg_stdio_src(&cinfo, infile);
    
    // 读取JPEG头
    jpeg_read_header(&cinfo, TRUE);
    
    // 开始解压缩
    jpeg_start_decompress(&cinfo);
    
    // 创建图像
    Image *img = image_create(cinfo.output_width, 
                             cinfo.output_height, 
                             cinfo.output_components);
    if (!img) {
        if (error_msg) {
            snprintf(error_msg, error_msg_size, "内存分配失败");
        }
        jpeg_destroy_decompress(&cinfo);
        fclose(infile);
        return NULL;
    }
    
    // 读取扫描线
    JSAMPROW row_pointer[1];
    int row_stride = cinfo.output_width * cinfo.output_components;
    
    while (cinfo.output_scanline < cinfo.output_height) {
        row_pointer[0] = &img->data[cinfo.output_scanline * row_stride];
        jpeg_read_scanlines(&cinfo, row_pointer, 1);
    }
    
    // 完成解压缩
    jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);
    fclose(infile);
    
    if (error_msg) {
        snprintf(error_msg, error_msg_size, "成功");
    }
    
    return img;
}

/**
 * @brief 安全写入JPEG（带错误处理）
 */
bool jpeg_write_safe(const Image *img, const char *filename, int quality,
                    char *error_msg, int error_msg_size)
{
    if (!img || !filename) {
        if (error_msg) {
            snprintf(error_msg, error_msg_size, "参数无效");
        }
        return false;
    }
    
    FILE *outfile = fopen(filename, "wb");
    if (!outfile) {
        if (error_msg) {
            snprintf(error_msg, error_msg_size, "无法创建文件: %s", filename);
        }
        return false;
    }
    
    struct jpeg_compress_struct cinfo;
    jpeg_error_handler jerr;
    
    // 设置错误处理
    cinfo.err = jpeg_std_error(&jerr.pub);
    jerr.pub.error_exit = jpeg_error_exit;
    jerr.pub.output_message = jpeg_output_message;
    
    // 设置错误恢复点
    if (setjmp(jerr.setjmp_buffer)) {
        // 发生错误，清理并返回
        if (error_msg) {
            snprintf(error_msg, error_msg_size, "JPEG编码错误: %s", 
                    jerr.error_message);
        }
        jpeg_destroy_compress(&cinfo);
        fclose(outfile);
        return false;
    }
    
    jpeg_create_compress(&cinfo);
    jpeg_stdio_dest(&cinfo, outfile);
    
    // 设置图像参数
    cinfo.image_width = img->width;
    cinfo.image_height = img->height;
    cinfo.input_components = img->channels;
    
    if (img->channels == 1) {
        cinfo.in_color_space = JCS_GRAYSCALE;
    } else if (img->channels == 3) {
        cinfo.in_color_space = JCS_RGB;
    } else {
        if (error_msg) {
            snprintf(error_msg, error_msg_size, "不支持的通道数: %d", 
                    img->channels);
        }
        jpeg_destroy_compress(&cinfo);
        fclose(outfile);
        return false;
    }
    
    jpeg_set_defaults(&cinfo);
    jpeg_set_quality(&cinfo, quality, TRUE);
    
    // 开始压缩
    jpeg_start_compress(&cinfo, TRUE);
    
    // 写入扫描线
    JSAMPROW row_pointer[1];
    int row_stride = img->width * img->channels;
    
    while (cinfo.next_scanline < cinfo.image_height) {
        row_pointer[0] = &img->data[cinfo.next_scanline * row_stride];
        jpeg_write_scanlines(&cinfo, row_pointer, 1);
    }
    
    // 完成压缩
    jpeg_finish_compress(&cinfo);
    jpeg_destroy_compress(&cinfo);
    fclose(outfile);
    
    if (error_msg) {
        snprintf(error_msg, error_msg_size, "成功");
    }
    
    return true;
}

/**
 * @brief 验证JPEG文件
 */
bool jpeg_validate(const char *filename, char *error_msg, int error_msg_size)
{
    if (!filename) {
        if (error_msg) {
            snprintf(error_msg, error_msg_size, "文件名为空");
        }
        return false;
    }
    
    FILE *infile = fopen(filename, "rb");
    if (!infile) {
        if (error_msg) {
            snprintf(error_msg, error_msg_size, "无法打开文件");
        }
        return false;
    }
    
    // 检查JPEG魔数
    unsigned char magic[2];
    if (fread(magic, 1, 2, infile) != 2) {
        if (error_msg) {
            snprintf(error_msg, error_msg_size, "文件太小");
        }
        fclose(infile);
        return false;
    }
    
    if (magic[0] != 0xFF || magic[1] != 0xD8) {
        if (error_msg) {
            snprintf(error_msg, error_msg_size, "不是有效的JPEG文件");
        }
        fclose(infile);
        return false;
    }
    
    // 重置文件指针
    fseek(infile, 0, SEEK_SET);
    
    struct jpeg_decompress_struct cinfo;
    jpeg_error_handler jerr;
    
    cinfo.err = jpeg_std_error(&jerr.pub);
    jerr.pub.error_exit = jpeg_error_exit;
    jerr.pub.output_message = jpeg_output_message;
    
    if (setjmp(jerr.setjmp_buffer)) {
        if (error_msg) {
            snprintf(error_msg, error_msg_size, "JPEG验证失败: %s", 
                    jerr.error_message);
        }
        jpeg_destroy_decompress(&cinfo);
        fclose(infile);
        return false;
    }
    
    jpeg_create_decompress(&cinfo);
    jpeg_stdio_src(&cinfo, infile);
    
    // 只读取头部信息
    jpeg_read_header(&cinfo, TRUE);
    
    // 验证基本参数
    if (cinfo.image_width == 0 || cinfo.image_height == 0) {
        if (error_msg) {
            snprintf(error_msg, error_msg_size, "无效的图像尺寸");
        }
        jpeg_destroy_decompress(&cinfo);
        fclose(infile);
        return false;
    }
    
    if (cinfo.num_components != 1 && cinfo.num_components != 3) {
        if (error_msg) {
            snprintf(error_msg, error_msg_size, "不支持的颜色分量数: %d", 
                    cinfo.num_components);
        }
        jpeg_destroy_decompress(&cinfo);
        fclose(infile);
        return false;
    }
    
    jpeg_destroy_decompress(&cinfo);
    fclose(infile);
    
    if (error_msg) {
        snprintf(error_msg, error_msg_size, "JPEG文件有效");
    }
    
    return true;
}

// ============================================================================
// 工具函数
// ============================================================================

/**
 * @brief 获取JPEG文件扩展名
 */
const char* jpeg_get_extension(const char *filename)
{
    if (!filename) {
        return NULL;
    }
    
    const char *dot = strrchr(filename, '.');
    if (!dot || dot == filename) {
        return NULL;
    }
    
    return dot + 1;
}

/**
 * @brief 检查是否为JPEG文件（根据扩展名）
 */
bool jpeg_is_jpeg_file(const char *filename)
{
    const char *ext = jpeg_get_extension(filename);
    if (!ext) {
        return false;
    }
    
    return (strcasecmp(ext, "jpg") == 0 || 
            strcasecmp(ext, "jpeg") == 0 ||
            strcasecmp(ext, "jpe") == 0);
}

/**
 * @brief 生成输出文件名
 */
char* jpeg_generate_output_filename(const char *input_filename, 
                                    const char *suffix,
                                    const char *output_dir)
{
    if (!input_filename) {
        return NULL;
    }
    
    // 提取文件名（不含路径）
    const char *filename = strrchr(input_filename, '/');
    if (!filename) {
        filename = strrchr(input_filename, '\\');
    }
    if (!filename) {
        filename = input_filename;
    } else {
        filename++;
    }
    
    // 提取文件名（不含扩展名）
    char basename[256];
    strncpy(basename, filename, sizeof(basename) - 1);
    basename[sizeof(basename) - 1] = '\0';
    
    char *dot = strrchr(basename, '.');
    if (dot) {
        *dot = '\0';
    }
    
    // 构建输出文件名
    char *output = (char*)malloc(512);
    if (!output) {
        return NULL;
    }
    
    if (output_dir) {
        if (suffix) {
            snprintf(output, 512, "%s/%s_%s.jpg", output_dir, basename, suffix);
        } else {
            snprintf(output, 512, "%s/%s.jpg", output_dir, basename);
        }
    } else {
        if (suffix) {
            snprintf(output, 512, "%s_%s.jpg", basename, suffix);
        } else {
            snprintf(output, 512, "%s.jpg", basename);
        }
    }
    
    return output;
}

/**
 * @brief 格式化文件大小
 */
char* jpeg_format_file_size(long size)
{
    static char buffer[64];
    
    if (size < 1024) {
        snprintf(buffer, sizeof(buffer), "%ld B", size);
    } else if (size < 1024 * 1024) {
        snprintf(buffer, sizeof(buffer), "%.2f KB", size / 1024.0);
    } else if (size < 1024 * 1024 * 1024) {
        snprintf(buffer, sizeof(buffer), "%.2f MB", size / (1024.0 * 1024.0));
    } else {
        snprintf(buffer, sizeof(buffer), "%.2f GB", 
                size / (1024.0 * 1024.0 * 1024.0));
    }
    
    return buffer;
}

/**
 * @brief 计算处理时间
 */
double jpeg_calculate_elapsed_time(struct timeval start, struct timeval end)
{
    return (end.tv_sec - start.tv_sec) + 
           (end.tv_usec - start.tv_usec) / 1000000.0;
}

/**
 * @brief 打印处理统计信息
 */
void jpeg_print_statistics(const char *operation, 
                          int total_files,
                          int success_count,
                          int fail_count,
                          long size_before,
                          long size_after,
                          double elapsed_time)
{
    printf("\n========================================\n");
    printf("操作: %s\n", operation);
    printf("========================================\n");
    printf("总文件数: %d\n", total_files);
    printf("成功: %d\n", success_count);
    printf("失败: %d\n", fail_count);
    
    if (size_before > 0 && size_after > 0) {
        printf("原始大小: %s\n", jpeg_format_file_size(size_before));
        printf("处理后大小: %s\n", jpeg_format_file_size(size_after));
        
        double ratio = 100.0 * (size_before - size_after) / size_before;
        printf("大小变化: %.1f%%\n", ratio);
    }
    
    if (elapsed_time > 0) {
        printf("处理时间: %.2f 秒\n", elapsed_time);
        if (total_files > 0) {
            printf("平均速度: %.2f 文件/秒\n", total_files / elapsed_time);
        }
    }
    
    printf("========================================\n");
}

/**
 * @brief 创建目录（递归）
 */
bool jpeg_create_directory(const char *path)
{
    if (!path) {
        return false;
    }
    
    char tmp[512];
    char *p = NULL;
    size_t len;
    
    snprintf(tmp, sizeof(tmp), "%s", path);
    len = strlen(tmp);
    
    if (tmp[len - 1] == '/') {
        tmp[len - 1] = 0;
    }
    
    for (p = tmp + 1; *p; p++) {
        if (*p == '/') {
            *p = 0;
            mkdir(tmp, 0755);
            *p = '/';
        }
    }
    
    return (mkdir(tmp, 0755) == 0 || errno == EEXIST);
}

/**
 * @brief 检查目录是否存在
 */
bool jpeg_directory_exists(const char *path)
{
    if (!path) {
        return false;
    }
    
    struct stat st;
    if (stat(path, &st) == 0) {
        return S_ISDIR(st.st_mode);
    }
    
    return false;
}

/**
 * @brief 列出目录中的JPEG文件
 */
char** jpeg_list_files(const char *directory, int *num_files)
{
    if (!directory || !num_files) {
        return NULL;
    }
    
    *num_files = 0;
    
    DIR *dir = opendir(directory);
    if (!dir) {
        fprintf(stderr, "无法打开目录: %s\n", directory);
        return NULL;
    }
    
    // 第一遍：计数
    struct dirent *entry;
    while ((entry = readdir(dir)) != NULL) {
        if (entry->d_type == DT_REG) {
            if (jpeg_is_jpeg_file(entry->d_name)) {
                (*num_files)++;
            }
        }
    }
    
    if (*num_files == 0) {
        closedir(dir);
        return NULL;
    }
    
    // 分配内存
    char **files = (char**)malloc(*num_files * sizeof(char*));
    if (!files) {
        closedir(dir);
        *num_files = 0;
        return NULL;
    }
    
    // 第二遍：收集文件名
    rewinddir(dir);
    int index = 0;
    
    while ((entry = readdir(dir)) != NULL && index < *num_files) {
        if (entry->d_type == DT_REG) {
            if (jpeg_is_jpeg_file(entry->d_name)) {
                char filepath[512];
                snprintf(filepath, sizeof(filepath), "%s/%s", 
                        directory, entry->d_name);
                
                files[index] = strdup(filepath);
                if (!files[index]) {
                    // 内存分配失败，清理已分配的内存
                    for (int i = 0; i < index; i++) {
                        free(files[i]);
                    }
                    free(files);
                    closedir(dir);
                    *num_files = 0;
                    return NULL;
                }
                index++;
            }
        }
    }
    
    closedir(dir);
    *num_files = index;
    
    return files;
}

/**
 * @brief 释放文件列表
 */
void jpeg_free_file_list(char **files, int num_files)
{
    if (!files) {
        return;
    }
    
    for (int i = 0; i < num_files; i++) {
        if (files[i]) {
            free(files[i]);
        }
    }
    
    free(files);
}

/**
 * @brief 比较两个JPEG文件
 */
bool jpeg_compare_files(const char *file1, const char *file2, 
                       double *mse, double *psnr)
{
    if (!file1 || !file2) {
        return false;
    }
    
    // 读取两个图像
    Image *img1 = jpeg_read(file1);
    Image *img2 = jpeg_read(file2);
    
    if (!img1 || !img2) {
        if (img1) image_destroy(img1);
        if (img2) image_destroy(img2);
        return false;
    }
    
    // 检查尺寸是否相同
    if (img1->width != img2->width || 
        img1->height != img2->height ||
        img1->channels != img2->channels) {
        fprintf(stderr, "图像尺寸或通道数不匹配\n");
        image_destroy(img1);
        image_destroy(img2);
        return false;
    }
    
    // 计算MSE
    long long sum_squared_diff = 0;
    int num_pixels = img1->width * img1->height * img1->channels;
    
    for (int i = 0; i < num_pixels; i++) {
        int diff = img1->data[i] - img2->data[i];
        sum_squared_diff += diff * diff;
    }
    
    *mse = (double)sum_squared_diff / num_pixels;
    
    // 计算PSNR
    if (*mse > 0) {
        *psnr = 10.0 * log10((255.0 * 255.0) / *mse);
    } else {
        *psnr = INFINITY;
    }
    
    image_destroy(img1);
    image_destroy(img2);
    
    return true;
}

/**
 * @brief 打印JPEG处理器版本信息
 */
void jpeg_print_version(void)
{
    printf("JPEG处理器 v1.0\n");
    printf("libjpeg版本: %d\n", JPEG_LIB_VERSION);
    printf("编译日期: %s %s\n", __DATE__, __TIME__);
    printf("\n支持的功能:\n");
    printf("  - JPEG读写\n");
    printf("  - 渐进式JPEG\n");
    printf("  - EXIF数据处理\n");
    printf("  - 图像变换\n");
    printf("  - 质量优化\n");
    printf("  - 批处理\n");
    printf("  - 缩略图生成\n");
    printf("  - 色彩空间转换\n");
    printf("  - 伪影处理\n");
}

/**
 * @brief 打印使用帮助
 */
void jpeg_print_usage(const char *program_name)
{
    printf("用法: %s [选项] <输入文件> [输出文件]\n\n", program_name);
    printf("选项:\n");
    printf("  -q <质量>        设置JPEG质量 (1-100, 默认85)\n");
    printf("  -p               生成渐进式JPEG\n");
    printf("  -o               优化JPEG文件\n");
    printf("  -r <宽>x<高>     调整图像尺寸\n");
    printf("  -t <宽>x<高>     生成缩略图\n");
    printf("  -g               转换为灰度图像\n");
    printf("  -e               增强图像质量\n");
    printf("  -i               显示图像信息\n");
    printf("  -v               显示版本信息\n");
    printf("  -h               显示此帮助信息\n");
    printf("\n示例:\n");
    printf("  %s -q 90 input.jpg output.jpg\n", program_name);
    printf("  %s -p -q 85 input.jpg output.jpg\n", program_name);
    printf("  %s -r 800x600 input.jpg output.jpg\n", program_name);
    printf("  %s -t 200x200 input.jpg thumb.jpg\n", program_name);
    printf("  %s -i input.jpg\n", program_name);
}

// 第五部分结束


