/**
 * @file raw_parser.c
 * @brief RAW图像格式解析器实现
 * @details 使用LibRaw库实现RAW格式的读取和处理
 */

#include "raw_parser.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <ctype.h>

// LibRaw库
#ifdef USE_LIBRAW
#include <libraw/libraw.h>
#endif

// ============================================================================
// 内部数据结构
// ============================================================================

/**
 * @brief LibRaw处理器包装
 */
typedef struct {
    #ifdef USE_LIBRAW
    libraw_data_t *raw_processor;
    #else
    void *raw_processor;
    #endif
    bool is_opened;
} RawProcessor;

// 全局RAW处理器（可选）
static bool g_raw_initialized = false;

// ============================================================================
// 格式检测和识别
// ============================================================================

/**
 * @brief RAW格式魔数定义
 */
typedef struct {
    const char *magic;       // 魔数
    size_t magic_len;        // 魔数长度
    size_t offset;           // 魔数在文件中的偏移
    RawFormat format;        // 对应的格式
} RawMagic;

static const RawMagic raw_magic_table[] = {
    // TIFF-based formats (CR2, NEF, ARW, etc.)
    {"II\x2a\x00", 4, 0, RAW_FORMAT_DNG},  // Little-endian TIFF
    {"MM\x00\x2a", 4, 0, RAW_FORMAT_DNG},  // Big-endian TIFF
    
    // Canon CR2
    {"II\x2a\x00\x10\x00\x00\x00CR", 12, 0, RAW_FORMAT_CR2},
    
    // Canon CR3 (ISO Base Media File Format)
    {"ftypcrx ", 8, 4, RAW_FORMAT_CR3},
    
    // Nikon NEF
    {"MM\x00\x2a", 4, 0, RAW_FORMAT_NEF},
    
    // Sony ARW
    {"II\x2a\x00", 4, 0, RAW_FORMAT_ARW},
    
    // Fujifilm RAF
    {"FUJIFILMCCD-RAW", 16, 0, RAW_FORMAT_RAF},
    
    // Olympus ORF
    {"IIRO", 4, 0, RAW_FORMAT_ORF},
    {"MMOR", 4, 0, RAW_FORMAT_ORF},
    {"IIRS", 4, 0, RAW_FORMAT_ORF},
    
    // Panasonic RW2
    {"IIU\x00", 4, 0, RAW_FORMAT_RW2},
    
    // Pentax PEF
    {"II\x2a\x00", 4, 0, RAW_FORMAT_PEF},
    
    // Samsung SRW
    {"MM\x00\x2a", 4, 0, RAW_FORMAT_SRW},
    
    // Hasselblad 3FR
    {"II\x2a\x00", 4, 0, RAW_FORMAT_3FR},
    
    // Phase One IIQ
    {"II\x49\x49", 4, 0, RAW_FORMAT_IIQ},
    
    {NULL, 0, 0, RAW_FORMAT_UNKNOWN}
};

/**
 * @brief 文件扩展名到格式的映射
 */
typedef struct {
    const char *extension;
    RawFormat format;
} RawExtensionMap;

static const RawExtensionMap raw_extension_table[] = {
    {"dng", RAW_FORMAT_DNG},
    {"cr2", RAW_FORMAT_CR2},
    {"cr3", RAW_FORMAT_CR3},
    {"nef", RAW_FORMAT_NEF},
    {"nrw", RAW_FORMAT_NEF},
    {"arw", RAW_FORMAT_ARW},
    {"srf", RAW_FORMAT_ARW},
    {"sr2", RAW_FORMAT_ARW},
    {"raf", RAW_FORMAT_RAF},
    {"orf", RAW_FORMAT_ORF},
    {"rw2", RAW_FORMAT_RW2},
    {"pef", RAW_FORMAT_PEF},
    {"ptx", RAW_FORMAT_PEF},
    {"srw", RAW_FORMAT_SRW},
    {"3fr", RAW_FORMAT_3FR},
    {"fff", RAW_FORMAT_FFF},
    {"mef", RAW_FORMAT_MEF},
    {"mos", RAW_FORMAT_MOS},
    {"iiq", RAW_FORMAT_IIQ},
    {"rwl", RAW_FORMAT_RWL},
    {"gpr", RAW_FORMAT_GPR},
    {NULL, RAW_FORMAT_UNKNOWN}
};

/**
 * @brief 格式名称表
 */
static const char* raw_format_names[] = {
    "Unknown",
    "Adobe DNG",
    "Canon CR2",
    "Canon CR3",
    "Nikon NEF",
    "Sony ARW",
    "Fujifilm RAF",
    "Olympus ORF",
    "Panasonic RW2",
    "Pentax PEF",
    "Samsung SRW",
    "Hasselblad 3FR",
    "Hasselblad FFF",
    "Mamiya MEF",
    "Leaf MOS",
    "Phase One IIQ",
    "Leica RWL",
    "GoPro GPR"
};

/**
 * @brief Bayer模式名称表
 */
static const char* bayer_pattern_names[] = {
    "RGGB",
    "BGGR",
    "GRBG",
    "GBRG",
    "None"
};

// ============================================================================
// 初始化和清理
// ============================================================================

/**
 * @brief 初始化RAW解析器
 */
bool raw_parser_init(void)
{
    if (g_raw_initialized) {
        return true;
    }

    #ifdef USE_LIBRAW
    // LibRaw不需要全局初始化
    g_raw_initialized = true;
    printf("RAW解析器初始化成功 (LibRaw)\n");
    return true;
    #else
    fprintf(stderr, "RAW解析器未编译LibRaw支持\n");
    return false;
    #endif
}

/**
 * @brief 清理RAW解析器
 */
void raw_parser_cleanup(void)
{
    if (!g_raw_initialized) {
        return;
    }

    g_raw_initialized = false;
    printf("RAW解析器已清理\n");
}

// ============================================================================
// 格式检测函数
// ============================================================================

/**
 * @brief 从文件扩展名获取RAW格式
 */
RawFormat raw_get_format_from_extension(const char *filename)
{
    if (!filename) {
        return RAW_FORMAT_UNKNOWN;
    }

    // 查找最后一个点
    const char *dot = strrchr(filename, '.');
    if (!dot || dot == filename) {
        return RAW_FORMAT_UNKNOWN;
    }

    // 转换为小写
    char ext[16];
    const char *p = dot + 1;
    int i = 0;
    while (*p && i < 15) {
        ext[i++] = tolower(*p++);
    }
    ext[i] = '\0';

    // 查找扩展名
    for (int j = 0; raw_extension_table[j].extension != NULL; j++) {
        if (strcmp(ext, raw_extension_table[j].extension) == 0) {
            return raw_extension_table[j].format;
        }
    }

    return RAW_FORMAT_UNKNOWN;
}

/**
 * @brief 通过文件头检测RAW格式
 */
RawFormat raw_detect_format(const char *filename)
{
    if (!filename) {
        return RAW_FORMAT_UNKNOWN;
    }

    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        return RAW_FORMAT_UNKNOWN;
    }

    // 读取文件头（最多128字节）
    unsigned char header[128];
    size_t read_size = fread(header, 1, sizeof(header), fp);
    fclose(fp);

    if (read_size < 4) {
        return RAW_FORMAT_UNKNOWN;
    }

    // 检查魔数
    for (int i = 0; raw_magic_table[i].magic != NULL; i++) {
        const RawMagic *magic = &raw_magic_table[i];
        
        if (magic->offset + magic->magic_len > read_size) {
            continue;
        }

        if (memcmp(header + magic->offset, magic->magic, magic->magic_len) == 0) {
            // 对于TIFF-based格式，需要进一步区分
            if (magic->format == RAW_FORMAT_DNG) {
                // 尝试通过扩展名进一步识别
                RawFormat ext_format = raw_get_format_from_extension(filename);
                if (ext_format != RAW_FORMAT_UNKNOWN && ext_format != RAW_FORMAT_DNG) {
                    return ext_format;
                }
            }
            return magic->format;
        }
    }

    // 如果魔数检测失败，尝试扩展名
    return raw_get_format_from_extension(filename);
}

/**
 * @brief 获取RAW格式名称
 */
const char* raw_get_format_name(RawFormat format)
{
    if (format < 0 || format >= sizeof(raw_format_names) / sizeof(raw_format_names[0])) {
        return "Unknown";
    }
    return raw_format_names[format];
}

/**
 * @brief 获取Bayer模式名称
 */
const char* raw_get_bayer_pattern_name(BayerPattern pattern)
{
    if (pattern < 0 || pattern >= sizeof(bayer_pattern_names) / sizeof(bayer_pattern_names[0])) {
        return "Unknown";
    }
    return bayer_pattern_names[pattern];
}

// ============================================================================
// 默认选项创建
// ============================================================================

/**
 * @brief 创建默认解析选项
 */
RawParseOptions raw_create_default_options(void)
{
    RawParseOptions options;
    memset(&options, 0, sizeof(RawParseOptions));

    // 输出选项
    options.output_color = true;
    options.use_camera_wb = true;
    options.use_auto_wb = false;
    options.no_auto_bright = false;
    options.no_auto_scale = false;

    // 去马赛克算法（AHD - 高质量）
    options.demosaic_algorithm = 3;
    options.quality = 3;

    // 色彩空间（sRGB）
    options.output_color_space = 1;

    // 伽马校正（sRGB标准）
    options.gamma_power = 2.222f;
    options.gamma_slope = 4.5f;

    // 白平衡（使用相机设置）
    for (int i = 0; i < 4; i++) {
        options.user_mul[i] = 0.0f;
    }

    // Fuji旋转
    options.use_fuji_rotate = true;

    // 噪点阈值
    options.threshold = 0.0f;

    // 高光恢复（混合模式）
    options.highlight_mode = 2;

    // 输出位深（8位）
    options.output_bps = 8;

    // 其他选项
    options.half_size = false;
    options.four_color_rgb = false;
    options.median_filter_passes = 0;

    return options;
}

// ============================================================================
// 内部辅助函数
// ============================================================================

/**
 * @brief 创建RAW处理器
 */
static RawProcessor* create_raw_processor(void)
{
    RawProcessor *processor = (RawProcessor*)calloc(1, sizeof(RawProcessor));
    if (!processor) {
        fprintf(stderr, "分配RAW处理器内存失败\n");
        return NULL;
    }

    #ifdef USE_LIBRAW
    processor->raw_processor = libraw_init(0);
    if (!processor->raw_processor) {
        fprintf(stderr, "初始化LibRaw处理器失败\n");
        free(processor);
        return NULL;
    }
    #else
    fprintf(stderr, "未编译LibRaw支持\n");
    free(processor);
    return NULL;
    #endif

    processor->is_opened = false;
    return processor;
}

/**
 * @brief 销毁RAW处理器
 */
static void destroy_raw_processor(RawProcessor *processor)
{
    if (!processor) {
        return;
    }

    #ifdef USE_LIBRAW
    if (processor->raw_processor) {
        if (processor->is_opened) {
            libraw_recycle(processor->raw_processor);
        }
        libraw_close(processor->raw_processor);
    }
    #endif

    free(processor);
}

/**
 * @brief 从LibRaw获取Bayer模式
 */
#ifdef USE_LIBRAW
static BayerPattern get_bayer_pattern_from_libraw(libraw_data_t *raw)
{
    if (!raw) {
        return BAYER_PATTERN_NONE;
    }

    // LibRaw的cdesc字段描述了颜色滤镜阵列
    // 例如: "RGBG" 表示 RGGB
    const char *desc = raw->idata.cdesc;
    
    if (desc[0] == 'R' && desc[1] == 'G' && desc[2] == 'B' && desc[3] == 'G') {
        return BAYER_PATTERN_RGGB;
    } else if (desc[0] == 'B' && desc[1] == 'G' && desc[2] == 'R' && desc[3] == 'G') {
        return BAYER_PATTERN_BGGR;
    } else if (desc[0] == 'G' && desc[1] == 'R' && desc[2] == 'B' && desc[3] == 'G') {
        return BAYER_PATTERN_GRBG;
    } else if (desc[0] == 'G' && desc[1] == 'B' && desc[2] == 'R' && desc[3] == 'G') {
        return BAYER_PATTERN_GBRG;
    }

    return BAYER_PATTERN_NONE;
}
#endif

/**
 * @brief 填充元数据
 */
#ifdef USE_LIBRAW
static void fill_metadata(RawMetadata *metadata, libraw_data_t *raw)
{
    if (!metadata || !raw) {
        return;
    }

    memset(metadata, 0, sizeof(RawMetadata));

    // 基本信息
    strncpy(metadata->make, raw->idata.make, sizeof(metadata->make) - 1);
    strncpy(metadata->model, raw->idata.model, sizeof(metadata->model) - 1);
    strncpy(metadata->software, raw->idata.software, sizeof(metadata->software) - 1);

    // 图像尺寸
    metadata->width = raw->sizes.width;
    metadata->height = raw->sizes.height;
    metadata->raw_width = raw->sizes.raw_width;
    metadata->raw_height = raw->sizes.raw_height;
    metadata->top_margin = raw->sizes.top_margin;
    metadata->left_margin = raw->sizes.left_margin;

    // 像素信息
    metadata->bits_per_sample = 16; // LibRaw内部使用16位
    metadata->bayer_pattern = get_bayer_pattern_from_libraw(raw);

    // 曝光信息
    metadata->iso_speed = raw->other.iso_speed;
    metadata->shutter_speed = raw->other.shutter;
    metadata->aperture = raw->other.aperture;
    metadata->focal_length = raw->other.focal_len;

    // 白平衡系数
    metadata->wb_coeffs.r_multiplier = raw->color.cam_mul[0];
    metadata->wb_coeffs.g_multiplier = raw->color.cam_mul[1];
    metadata->wb_coeffs.b_multiplier = raw->color.cam_mul[2];
    metadata->wb_coeffs.g2_multiplier = raw->color.cam_mul[3];

    // 黑电平和白电平
    for (int i = 0; i < 4; i++) {
        metadata->black_level[i] = raw->color.black + raw->color.cblack[i];
    }
    metadata->white_level = raw->color.maximum;

    // 色彩矩阵
    metadata->color_matrix1.rows = 3;
    metadata->color_matrix1.cols = 3;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            metadata->color_matrix1.matrix[i][j] = raw->color.rgb_cam[i][j];
        }
    }

    // 时间信息
    if (raw->other.timestamp > 0) {
        time_t timestamp = raw->other.timestamp;
        struct tm *tm_info = localtime(&timestamp);
        strftime(metadata->datetime, sizeof(metadata->datetime), 
                 "%Y:%m:%d %H:%M:%S", tm_info);
    }

    // 缩略图信息
    metadata->has_thumbnail = (raw->thumbnail.tlength > 0);
    if (metadata->has_thumbnail) {
        metadata->thumb_width = raw->thumbnail.twidth;
        metadata->thumb_height = raw->thumbnail.theight;
    }

    // 方向
    metadata->orientation = raw->sizes.flip;

    // 镜头信息
    if (raw->lens.Lens[0]) {
        strncpy(metadata->lens_model, raw->lens.Lens, sizeof(metadata->lens_model) - 1);
    }
}
#endif

/**
 * @brief 应用解析选项到LibRaw
 */
#ifdef USE_LIBRAW
static void apply_options_to_libraw(libraw_data_t *raw, const RawParseOptions *options)
{
    if (!raw || !options) {
        return;
    }

    libraw_output_params_t *params = &raw->params;

    // 输出选项
    params->output_color = options->output_color ? 1 : 0;
    params->use_camera_wb = options->use_camera_wb ? 1 : 0;
    params->use_auto_wb = options->use_auto_wb ? 1 : 0;
    params->no_auto_bright = options->no_auto_bright ? 1 : 0;
    params->no_auto_scale = options->no_auto_scale ? 1 : 0;

    // 去马赛克算法
    params->user_qual = options->demosaic_algorithm;

    // 色彩空间
    params->output_color = options->output_color_space;

    // 伽马校正
    params->gamm[0] = 1.0 / options->gamma_power;
    params->gamm[1] = options->gamma_slope;

    // 用户白平衡
    if (options->user_mul[0] > 0) {
        for (int i = 0; i < 4; i++) {
            params->user_mul[i] = options->user_mul[i];
        }
    }

    // Fuji旋转
    params->use_fuji_rotate = options->use_fuji_rotate ? 1 : 0;

    // 噪点阈值
    params->threshold = options->threshold;

    // 高光恢复
    params->highlight = options->highlight_mode;

    // 输出位深
    params->output_bps = options->output_bps;

    // 半尺寸
    params->half_size = options->half_size ? 1 : 0;

    // 四色RGB
    params->four_color_rgb = options->four_color_rgb ? 1 : 0;

    // 中值滤波
    params->med_passes = options->median_filter_passes;
}
#endif

// 第一部分结束
// ============================================================================
// RAW文件打开和读取
// ============================================================================

/**
 * @brief 打开RAW文件
 */
RawImage* raw_open(const char *filename)
{
    RawParseOptions options = raw_create_default_options();
    return raw_open_with_options(filename, &options);
}

/**
 * @brief 打开RAW文件并指定选项
 */
RawImage* raw_open_with_options(const char *filename, const RawParseOptions *options)
{
    if (!filename) {
        fprintf(stderr, "文件名为空\n");
        return NULL;
    }

    #ifndef USE_LIBRAW
    fprintf(stderr, "未编译LibRaw支持\n");
    return NULL;
    #else

    // 检测格式
    RawFormat format = raw_detect_format(filename);
    if (format == RAW_FORMAT_UNKNOWN) {
        fprintf(stderr, "无法识别RAW格式: %s\n", filename);
        return NULL;
    }

    printf("检测到RAW格式: %s\n", raw_get_format_name(format));

    // 创建处理器
    RawProcessor *processor = create_raw_processor();
    if (!processor) {
        return NULL;
    }

    // 应用选项
    if (options) {
        apply_options_to_libraw(processor->raw_processor, options);
    }

    // 打开文件
    int ret = libraw_open_file(processor->raw_processor, filename);
    if (ret != LIBRAW_SUCCESS) {
        fprintf(stderr, "打开RAW文件失败: %s (错误码: %d)\n", 
                libraw_strerror(ret), ret);
        destroy_raw_processor(processor);
        return NULL;
    }

    processor->is_opened = true;

    // 解包RAW数据
    ret = libraw_unpack(processor->raw_processor);
    if (ret != LIBRAW_SUCCESS) {
        fprintf(stderr, "解包RAW数据失败: %s\n", libraw_strerror(ret));
        destroy_raw_processor(processor);
        return NULL;
    }

    // 创建RawImage对象
    RawImage *raw_img = (RawImage*)calloc(1, sizeof(RawImage));
    if (!raw_img) {
        fprintf(stderr, "分配RawImage内存失败\n");
        destroy_raw_processor(processor);
        return NULL;
    }

    // 填充元数据
    raw_img->metadata.format = format;
    fill_metadata(&raw_img->metadata, processor->raw_processor);

    // 获取RAW数据
    libraw_data_t *raw = processor->raw_processor;
    int raw_width = raw->sizes.raw_width;
    int raw_height = raw->sizes.raw_height;
    
    raw_img->data_size = raw_width * raw_height * sizeof(uint16_t);
    raw_img->raw_data = (uint16_t*)malloc(raw_img->data_size);
    
    if (!raw_img->raw_data) {
        fprintf(stderr, "分配RAW数据内存失败\n");
        free(raw_img);
        destroy_raw_processor(processor);
        return NULL;
    }

    // 复制RAW数据
    memcpy(raw_img->raw_data, raw->rawdata.raw_image, raw_img->data_size);

    // 提取缩略图（如果有）
    if (raw->thumbnail.tlength > 0) {
        raw_img->thumbnail_size = raw->thumbnail.tlength;
        raw_img->thumbnail_data = (unsigned char*)malloc(raw_img->thumbnail_size);
        
        if (raw_img->thumbnail_data) {
            ret = libraw_unpack_thumb(raw);
            if (ret == LIBRAW_SUCCESS) {
                memcpy(raw_img->thumbnail_data, raw->thumbnail.thumb, 
                       raw_img->thumbnail_size);
            } else {
                free(raw_img->thumbnail_data);
                raw_img->thumbnail_data = NULL;
                raw_img->thumbnail_size = 0;
            }
        }
    }

    // 保存处理器指针
    raw_img->internal_data = processor;

    printf("成功打开RAW文件: %dx%d, %d位, %s\n",
           raw_img->metadata.width, raw_img->metadata.height,
           raw_img->metadata.bits_per_sample,
           raw_get_bayer_pattern_name(raw_img->metadata.bayer_pattern));

    return raw_img;
    #endif
}

/**
 * @brief 释放RAW图像对象
 */
void raw_close(RawImage *raw)
{
    if (!raw) {
        return;
    }

    // 释放RAW数据
    if (raw->raw_data) {
        free(raw->raw_data);
    }

    // 释放缩略图数据
    if (raw->thumbnail_data) {
        free(raw->thumbnail_data);
    }

    // 释放处理器
    if (raw->internal_data) {
        destroy_raw_processor((RawProcessor*)raw->internal_data);
    }

    free(raw);
}

// ============================================================================
// RAW图像处理
// ============================================================================

/**
 * @brief 解析RAW图像为RGB图像
 */
Image* raw_process(const RawImage *raw, const RawParseOptions *options)
{
    if (!raw) {
        fprintf(stderr, "RAW图像对象为空\n");
        return NULL;
    }

    #ifndef USE_LIBRAW
    fprintf(stderr, "未编译LibRaw支持\n");
    return NULL;
    #else

    RawProcessor *processor = (RawProcessor*)raw->internal_data;
    if (!processor || !processor->raw_processor) {
        fprintf(stderr, "无效的RAW处理器\n");
        return NULL;
    }

    libraw_data_t *libraw = processor->raw_processor;

    // 如果提供了新选项，重新应用
    if (options) {
        apply_options_to_libraw(libraw, options);
    }

    // 处理RAW数据
    int ret = libraw_dcraw_process(libraw);
    if (ret != LIBRAW_SUCCESS) {
        fprintf(stderr, "处理RAW数据失败: %s\n", libraw_strerror(ret));
        return NULL;
    }

    // 获取处理后的图像
    libraw_processed_image_t *processed = libraw_dcraw_make_mem_image(libraw, &ret);
    if (!processed) {
        fprintf(stderr, "生成内存图像失败: %s\n", libraw_strerror(ret));
        return NULL;
    }

    // 创建Image对象
    Image *img = image_create(processed->width, processed->height, processed->colors);
    if (!img) {
        fprintf(stderr, "创建图像对象失败\n");
        libraw_dcraw_clear_mem(processed);
        return NULL;
    }

    // 复制图像数据
    size_t data_size = processed->width * processed->height * processed->colors;
    
    if (processed->bits == 8) {
        // 8位数据直接复制
        memcpy(img->data, processed->data, data_size);
    } else if (processed->bits == 16) {
        // 16位数据转换为8位
        uint16_t *src = (uint16_t*)processed->data;
        unsigned char *dst = img->data;
        
        for (size_t i = 0; i < data_size; i++) {
            dst[i] = src[i] >> 8; // 简单右移8位
        }
    }

    // 释放LibRaw处理后的图像
    libraw_dcraw_clear_mem(processed);

    printf("成功处理RAW图像: %dx%d, %d通道\n",
           img->width, img->height, img->channels);

    return img;
    #endif
}

/**
 * @brief 提取RAW缩略图
 */
Image* raw_extract_thumbnail(const RawImage *raw)
{
    if (!raw) {
        fprintf(stderr, "RAW图像对象为空\n");
        return NULL;
    }

    if (!raw->metadata.has_thumbnail) {
        fprintf(stderr, "RAW文件不包含缩略图\n");
        return NULL;
    }

    #ifndef USE_LIBRAW
    fprintf(stderr, "未编译LibRaw支持\n");
    return NULL;
    #else

    RawProcessor *processor = (RawProcessor*)raw->internal_data;
    if (!processor || !processor->raw_processor) {
        fprintf(stderr, "无效的RAW处理器\n");
        return NULL;
    }

    libraw_data_t *libraw = processor->raw_processor;

    // 解包缩略图
    int ret = libraw_unpack_thumb(libraw);
    if (ret != LIBRAW_SUCCESS) {
        fprintf(stderr, "解包缩略图失败: %s\n", libraw_strerror(ret));
        return NULL;
    }

    // 获取缩略图数据
    libraw_processed_image_t *thumb = libraw_dcraw_make_mem_thumb(libraw, &ret);
    if (!thumb) {
        fprintf(stderr, "生成缩略图失败: %s\n", libraw_strerror(ret));
        return NULL;
    }

    // 创建Image对象
    Image *img = NULL;

    if (thumb->type == LIBRAW_IMAGE_JPEG) {
        // JPEG缩略图，需要解码
        // 这里简化处理，实际应该使用JPEG解码器
        fprintf(stderr, "JPEG缩略图需要额外解码\n");
    } else if (thumb->type == LIBRAW_IMAGE_BITMAP) {
        // 位图缩略图
        img = image_create(thumb->width, thumb->height, thumb->colors);
        if (img) {
            size_t data_size = thumb->width * thumb->height * thumb->colors;
            
            if (thumb->bits == 8) {
                memcpy(img->data, thumb->data, data_size);
            } else if (thumb->bits == 16) {
                uint16_t *src = (uint16_t*)thumb->data;
                unsigned char *dst = img->data;
                for (size_t i = 0; i < data_size; i++) {
                    dst[i] = src[i] >> 8;
                }
            }
        }
    }

    libraw_dcraw_clear_mem(thumb);

    if (img) {
        printf("成功提取缩略图: %dx%d\n", img->width, img->height);
    }

    return img;
    #endif
}

/**
 * @brief 获取RAW元数据
 */
bool raw_get_metadata(const char *filename, RawMetadata *metadata)
{
    if (!filename || !metadata) {
        return false;
    }

    #ifndef USE_LIBRAW
    fprintf(stderr, "未编译LibRaw支持\n");
    return false;
    #else

    // 创建临时处理器
    RawProcessor *processor = create_raw_processor();
    if (!processor) {
        return false;
    }

    // 打开文件
    int ret = libraw_open_file(processor->raw_processor, filename);
    if (ret != LIBRAW_SUCCESS) {
        fprintf(stderr, "打开RAW文件失败: %s\n", libraw_strerror(ret));
        destroy_raw_processor(processor);
        return false;
    }

    // 解包（只需要元数据）
    ret = libraw_unpack(processor->raw_processor);
    if (ret != LIBRAW_SUCCESS) {
        fprintf(stderr, "解包RAW数据失败: %s\n", libraw_strerror(ret));
        destroy_raw_processor(processor);
        return false;
    }

    // 填充元数据
    metadata->format = raw_detect_format(filename);
    fill_metadata(metadata, processor->raw_processor);

    // 清理
    destroy_raw_processor(processor);

    return true;
    #endif
}

/**
 * @brief 打印RAW元数据
 */
void raw_print_metadata(const RawMetadata *metadata)
{
    if (!metadata) {
        return;
    }

    printf("\n========== RAW元数据 ==========\n");
    printf("格式: %s\n", raw_get_format_name(metadata->format));
    printf("制造商: %s\n", metadata->make);
    printf("型号: %s\n", metadata->model);
    printf("软件: %s\n", metadata->software);
    printf("\n");

    printf("图像尺寸: %dx%d\n", metadata->width, metadata->height);
    printf("RAW尺寸: %dx%d\n", metadata->raw_width, metadata->raw_height);
    printf("边距: 上=%d, 左=%d\n", metadata->top_margin, metadata->left_margin);
    printf("位深: %d位\n", metadata->bits_per_sample);
    printf("Bayer模式: %s\n", raw_get_bayer_pattern_name(metadata->bayer_pattern));
    printf("\n");

    printf("ISO: %.0f\n", metadata->iso_speed);
    printf("快门速度: 1/%.0f秒\n", 1.0 / metadata->shutter_speed);
    printf("光圈: f/%.1f\n", metadata->aperture);
    printf("焦距: %.1fmm\n", metadata->focal_length);
    printf("\n");

    printf("白平衡系数:\n");
    printf("  R: %.4f\n", metadata->wb_coeffs.r_multiplier);
    printf("  G: %.4f\n", metadata->wb_coeffs.g_multiplier);
    printf("  B: %.4f\n", metadata->wb_coeffs.b_multiplier);
    printf("  G2: %.4f\n", metadata->wb_coeffs.g2_multiplier);
    printf("\n");

    printf("黑电平: [%d, %d, %d, %d]\n",
           metadata->black_level[0], metadata->black_level[1],
           metadata->black_level[2], metadata->black_level[3]);
    printf("白电平: %d\n", metadata->white_level);
    printf("\n");

    if (metadata->datetime[0]) {
        printf("拍摄时间: %s\n", metadata->datetime);
    }

    if (metadata->lens_model[0]) {
        printf("镜头: %s\n", metadata->lens_model);
    }

    if (metadata->has_thumbnail) {
        printf("缩略图: %dx%d\n", metadata->thumb_width, metadata->thumb_height);
    }

    if (metadata->has_gps) {
        printf("GPS: 纬度=%.6f, 经度=%.6f, 海拔=%.1fm\n",
               metadata->gps_latitude, metadata->gps_longitude, metadata->gps_altitude);
    }

    printf("================================\n\n");
}

// ============================================================================
// 白平衡处理
// ============================================================================

/**
 * @brief 计算自动白平衡
 */
bool raw_calculate_auto_wb(const RawImage *raw, WhiteBalance *wb)
{
    if (!raw || !wb) {
        return false;
    }

    #ifndef USE_LIBRAW
    fprintf(stderr, "未编译LibRaw支持\n");
    return false;
    #else

    RawProcessor *processor = (RawProcessor*)raw->internal_data;
    if (!processor || !processor->raw_processor) {
        fprintf(stderr, "无效的RAW处理器\n");
        return false;
    }

    libraw_data_t *libraw = processor->raw_processor;

    // 使用LibRaw的自动白平衡
    // 这里使用预乘系数
    float pre_mul[4];
    
    // 计算灰度世界白平衡
    double sum[4] = {0, 0, 0, 0};
    int count[4] = {0, 0, 0, 0};
    
    int width = libraw->sizes.width;
    int height = libraw->sizes.height;
    uint16_t *data = libraw->rawdata.raw_image;
    
    // 采样中心区域
    int x_start = width / 4;
    int x_end = width * 3 / 4;
    int y_start = height / 4;
    int y_end = height * 3 / 4;
    
    for (int y = y_start; y < y_end; y += 4) {
        for (int x = x_start; x < x_end; x += 4) {
            int idx = y * width + x;
            int color = libraw->COLOR(y, x);
            
            if (color < 4) {
                sum[color] += data[idx];
                count[color]++;
            }
        }
    }
    
    // 计算平均值
    for (int i = 0; i < 4; i++) {
        if (count[i] > 0) {
            pre_mul[i] = sum[i] / count[i];
        } else {
            pre_mul[i] = 1.0f;
        }
    }
    
    // 归一化到绿色通道
    float green_avg = (pre_mul[1] + pre_mul[3]) / 2.0f;
    
    wb->r_multiplier = green_avg / pre_mul[0];
    wb->g_multiplier = 1.0f;
    wb->b_multiplier = green_avg / pre_mul[2];
    wb->g2_multiplier = 1.0f;

    printf("自动白平衡: R=%.4f, G=%.4f, B=%.4f\n",
           wb->r_multiplier, wb->g_multiplier, wb->b_multiplier);

    return true;
    #endif
}

/**
 * @brief 应用白平衡
 */
bool raw_apply_white_balance(RawImage *raw, const WhiteBalance *wb)
{
    if (!raw || !wb) {
        return false;
    }

    #ifndef USE_LIBRAW
    fprintf(stderr, "未编译LibRaw支持\n");
    return false;
    #else

    RawProcessor *processor = (RawProcessor*)raw->internal_data;
    if (!processor || !processor->raw_processor) {
        fprintf(stderr, "无效的RAW处理器\n");
        return false;
    }

    libraw_data_t *libraw = processor->raw_processor;

    // 设置用户白平衡倍增系数
    libraw->params.user_mul[0] = wb->r_multiplier;
    libraw->params.user_mul[1] = wb->g_multiplier;
    libraw->params.user_mul[2] = wb->b_multiplier;
    libraw->params.user_mul[3] = wb->g2_multiplier;

    // 禁用相机白平衡和自动白平衡
    libraw->params.use_camera_wb = 0;
    libraw->params.use_auto_wb = 0;

    printf("应用白平衡: R=%.4f, G=%.4f, B=%.4f\n",
           wb->r_multiplier, wb->g_multiplier, wb->b_multiplier);

    return true;
    #endif
}

// ============================================================================
// 去马赛克（Demosaicing）
// ============================================================================

/**
 * @brief 去马赛克（Bayer转RGB）
 */
Image* raw_demosaic(const RawImage *raw, int algorithm)
{
    if (!raw) {
        fprintf(stderr, "RAW图像对象为空\n");
        return NULL;
    }

    #ifndef USE_LIBRAW
    fprintf(stderr, "未编译LibRaw支持\n");
    return NULL;
    #else

    RawProcessor *processor = (RawProcessor*)raw->internal_data;
    if (!processor || !processor->raw_processor) {
        fprintf(stderr, "无效的RAW处理器\n");
        return NULL;
    }

    libraw_data_t *libraw = processor->raw_processor;

    // 设置去马赛克算法
    libraw->params.user_qual = algorithm;

    // 创建临时选项
    RawParseOptions options = raw_create_default_options();
    options.demosaic_algorithm = algorithm;

    // 处理图像
    return raw_process(raw, &options);
    #endif
}

// 第二部分结束
// ============================================================================
// 色彩矩阵和色彩空间转换
// ============================================================================

/**
 * @brief 应用色彩矩阵
 */
bool raw_apply_color_matrix(Image *img, const ColorMatrix *matrix)
{
    if (!img || !matrix) {
        return false;
    }

    if (img->channels != 3) {
        fprintf(stderr, "色彩矩阵只能应用于RGB图像\n");
        return false;
    }

    if (matrix->rows != 3 || (matrix->cols != 3 && matrix->cols != 4)) {
        fprintf(stderr, "无效的色彩矩阵尺寸: %dx%d\n", matrix->rows, matrix->cols);
        return false;
    }

    printf("应用色彩矩阵 (%dx%d)...\n", matrix->rows, matrix->cols);

    int width = img->width;
    int height = img->height;

    // 遍历每个像素
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = (y * width + x) * 3;
            
            // 获取原始RGB值（归一化到0-1）
            float r = img->data[idx + 0] / 255.0f;
            float g = img->data[idx + 1] / 255.0f;
            float b = img->data[idx + 2] / 255.0f;

            // 应用色彩矩阵
            float new_r, new_g, new_b;

            if (matrix->cols == 3) {
                // 3x3矩阵
                new_r = matrix->matrix[0][0] * r + matrix->matrix[0][1] * g + matrix->matrix[0][2] * b;
                new_g = matrix->matrix[1][0] * r + matrix->matrix[1][1] * g + matrix->matrix[1][2] * b;
                new_b = matrix->matrix[2][0] * r + matrix->matrix[2][1] * g + matrix->matrix[2][2] * b;
            } else {
                // 3x4矩阵（包含偏移）
                new_r = matrix->matrix[0][0] * r + matrix->matrix[0][1] * g + 
                        matrix->matrix[0][2] * b + matrix->matrix[0][3];
                new_g = matrix->matrix[1][0] * r + matrix->matrix[1][1] * g + 
                        matrix->matrix[1][2] * b + matrix->matrix[1][3];
                new_b = matrix->matrix[2][0] * r + matrix->matrix[2][1] * g + 
                        matrix->matrix[2][2] * b + matrix->matrix[2][3];
            }

            // 裁剪到有效范围
            new_r = fmaxf(0.0f, fminf(1.0f, new_r));
            new_g = fmaxf(0.0f, fminf(1.0f, new_g));
            new_b = fmaxf(0.0f, fminf(1.0f, new_b));

            // 转换回8位
            img->data[idx + 0] = (unsigned char)(new_r * 255.0f + 0.5f);
            img->data[idx + 1] = (unsigned char)(new_g * 255.0f + 0.5f);
            img->data[idx + 2] = (unsigned char)(new_b * 255.0f + 0.5f);
        }
    }

    printf("色彩矩阵应用完成\n");
    return true;
}

/**
 * @brief 应用伽马校正
 */
bool raw_apply_gamma(Image *img, float power, float slope)
{
    if (!img) {
        return false;
    }

    if (power <= 0.0f) {
        fprintf(stderr, "无效的伽马幂次: %.3f\n", power);
        return false;
    }

    printf("应用伽马校正: power=%.3f, slope=%.3f\n", power, slope);

    int total_pixels = img->width * img->height * img->channels;

    // 创建查找表以提高性能
    unsigned char lut[256];
    
    for (int i = 0; i < 256; i++) {
        float normalized = i / 255.0f;
        float corrected;

        // sRGB伽马校正
        if (normalized <= 0.0031308f) {
            corrected = normalized * slope;
        } else {
            corrected = 1.055f * powf(normalized, 1.0f / power) - 0.055f;
        }

        // 裁剪并转换回8位
        corrected = fmaxf(0.0f, fminf(1.0f, corrected));
        lut[i] = (unsigned char)(corrected * 255.0f + 0.5f);
    }

    // 应用查找表
    for (int i = 0; i < total_pixels; i++) {
        img->data[i] = lut[img->data[i]];
    }

    printf("伽马校正完成\n");
    return true;
}

/**
 * @brief RGB转XYZ色彩空间
 */
static void rgb_to_xyz(float r, float g, float b, float *x, float *y, float *z)
{
    // sRGB到XYZ的转换矩阵
    *x = 0.4124564f * r + 0.3575761f * g + 0.1804375f * b;
    *y = 0.2126729f * r + 0.7151522f * g + 0.0721750f * b;
    *z = 0.0193339f * r + 0.1191920f * g + 0.9503041f * b;
}

/**
 * @brief XYZ转RGB色彩空间
 */
static void xyz_to_rgb(float x, float y, float z, float *r, float *g, float *b)
{
    // XYZ到sRGB的转换矩阵
    *r =  3.2404542f * x - 1.5371385f * y - 0.4985314f * z;
    *g = -0.9692660f * x + 1.8760108f * y + 0.0415560f * z;
    *b =  0.0556434f * x - 0.2040259f * y + 1.0572252f * z;
}

/**
 * @brief RGB转Adobe RGB
 */
static void rgb_to_adobe_rgb(float r, float g, float b, float *ar, float *ag, float *ab)
{
    // 先转到XYZ
    float x, y, z;
    rgb_to_xyz(r, g, b, &x, &y, &z);

    // XYZ到Adobe RGB的转换矩阵
    *ar =  2.0413690f * x - 0.5649464f * y - 0.3446944f * z;
    *ag = -0.9692660f * x + 1.8760108f * y + 0.0415560f * z;
    *ab =  0.0134474f * x - 0.1183897f * y + 1.0154096f * z;
}

/**
 * @brief RGB转ProPhoto RGB
 */
static void rgb_to_prophoto_rgb(float r, float g, float b, float *pr, float *pg, float *pb)
{
    // 先转到XYZ
    float x, y, z;
    rgb_to_xyz(r, g, b, &x, &y, &z);

    // XYZ到ProPhoto RGB的转换矩阵
    *pr =  1.3459433f * x - 0.2556075f * y - 0.0511118f * z;
    *pg = -0.5445989f * x + 1.5081673f * y + 0.0205351f * z;
    *pb =  0.0000000f * x + 0.0000000f * y + 1.2118128f * z;
}

/**
 * @brief 转换色彩空间
 */
bool raw_convert_color_space(Image *img, int color_space)
{
    if (!img) {
        return false;
    }

    if (img->channels != 3) {
        fprintf(stderr, "色彩空间转换只能应用于RGB图像\n");
        return false;
    }

    const char *space_names[] = {
        "Raw", "sRGB", "Adobe RGB", "Wide Gamut RGB", "ProPhoto RGB", "XYZ"
    };

    if (color_space < 0 || color_space > 5) {
        fprintf(stderr, "无效的色彩空间: %d\n", color_space);
        return false;
    }

    printf("转换色彩空间到: %s\n", space_names[color_space]);

    // 0 = raw (不转换)
    if (color_space == 0) {
        return true;
    }

    // 1 = sRGB (默认，不需要转换)
    if (color_space == 1) {
        return true;
    }

    int width = img->width;
    int height = img->height;

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = (y * width + x) * 3;
            
            // 归一化到0-1
            float r = img->data[idx + 0] / 255.0f;
            float g = img->data[idx + 1] / 255.0f;
            float b = img->data[idx + 2] / 255.0f;

            float new_r, new_g, new_b;

            switch (color_space) {
                case 2: // Adobe RGB
                    rgb_to_adobe_rgb(r, g, b, &new_r, &new_g, &new_b);
                    break;

                case 3: // Wide Gamut RGB
                    // 简化处理，使用Adobe RGB
                    rgb_to_adobe_rgb(r, g, b, &new_r, &new_g, &new_b);
                    break;

                case 4: // ProPhoto RGB
                    rgb_to_prophoto_rgb(r, g, b, &new_r, &new_g, &new_b);
                    break;

                case 5: // XYZ
                    rgb_to_xyz(r, g, b, &new_r, &new_g, &new_b);
                    break;

                default:
                    new_r = r;
                    new_g = g;
                    new_b = b;
                    break;
            }

            // 裁剪到有效范围
            new_r = fmaxf(0.0f, fminf(1.0f, new_r));
            new_g = fmaxf(0.0f, fminf(1.0f, new_g));
            new_b = fmaxf(0.0f, fminf(1.0f, new_b));

            // 转换回8位
            img->data[idx + 0] = (unsigned char)(new_r * 255.0f + 0.5f);
            img->data[idx + 1] = (unsigned char)(new_g * 255.0f + 0.5f);
            img->data[idx + 2] = (unsigned char)(new_b * 255.0f + 0.5f);
        }
    }

    printf("色彩空间转换完成\n");
    return true;
}

// ============================================================================
// 高光恢复
// ============================================================================

/**
 * @brief 高光恢复
 */
bool raw_recover_highlights(Image *img, int mode)
{
    if (!img) {
        return false;
    }

    if (img->channels != 3) {
        fprintf(stderr, "高光恢复只能应用于RGB图像\n");
        return false;
    }

    const char *mode_names[] = {
        "裁剪", "不裁剪", "混合", "重建"
    };

    if (mode < 0 || mode > 3) {
        fprintf(stderr, "无效的高光恢复模式: %d\n", mode);
        return false;
    }

    printf("应用高光恢复: %s模式\n", mode_names[mode]);

    int width = img->width;
    int height = img->height;

    // 模式0: 裁剪（默认行为，不需要处理）
    if (mode == 0) {
        return true;
    }

    // 模式1: 不裁剪（保留过曝值）
    if (mode == 1) {
        return true;
    }

    // 模式2: 混合
    if (mode == 2) {
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int idx = (y * width + x) * 3;
                
                unsigned char r = img->data[idx + 0];
                unsigned char g = img->data[idx + 1];
                unsigned char b = img->data[idx + 2];

                // 检测过曝像素（至少一个通道达到255）
                if (r == 255 || g == 255 || b == 255) {
                    // 使用未过曝通道的信息重建
                    int max_val = (r > g) ? ((r > b) ? r : b) : ((g > b) ? g : b);
                    int min_val = (r < g) ? ((r < b) ? r : b) : ((g < b) ? g : b);

                    if (max_val > 200 && min_val < 200) {
                        // 混合过曝和未过曝通道
                        float blend = (max_val - 200) / 55.0f;
                        blend = fmaxf(0.0f, fminf(1.0f, blend));

                        img->data[idx + 0] = (unsigned char)(r * (1.0f - blend) + min_val * blend);
                        img->data[idx + 1] = (unsigned char)(g * (1.0f - blend) + min_val * blend);
                        img->data[idx + 2] = (unsigned char)(b * (1.0f - blend) + min_val * blend);
                    }
                }
            }
        }
    }

    // 模式3: 重建（更复杂的算法）
    if (mode == 3) {
        for (int y = 1; y < height - 1; y++) {
            for (int x = 1; x < width - 1; x++) {
                int idx = (y * width + x) * 3;
                
                unsigned char r = img->data[idx + 0];
                unsigned char g = img->data[idx + 1];
                unsigned char b = img->data[idx + 2];

                // 检测过曝像素
                if (r >= 250 || g >= 250 || b >= 250) {
                    // 使用周围像素的信息重建
                    int sum_r = 0, sum_g = 0, sum_b = 0;
                    int count = 0;

                    for (int dy = -1; dy <= 1; dy++) {
                        for (int dx = -1; dx <= 1; dx++) {
                            if (dx == 0 && dy == 0) continue;

                            int neighbor_idx = ((y + dy) * width + (x + dx)) * 3;
                            unsigned char nr = img->data[neighbor_idx + 0];
                            unsigned char ng = img->data[neighbor_idx + 1];
                            unsigned char nb = img->data[neighbor_idx + 2];

                            // 只使用未过曝的邻居
                            if (nr < 250 && ng < 250 && nb < 250) {
                                sum_r += nr;
                                sum_g += ng;
                                sum_b += nb;
                                count++;
                            }
                        }
                    }

                    if (count > 0) {
                        // 使用邻居的平均值
                        img->data[idx + 0] = sum_r / count;
                        img->data[idx + 1] = sum_g / count;
                        img->data[idx + 2] = sum_b / count;
                    }
                }
            }
        }
    }

    printf("高光恢复完成\n");
    return true;
}

// ============================================================================
// 快速加载和便捷函数
// ============================================================================

/**
 * @brief 快速加载RAW预览图
 */
Image* raw_load_preview(const char *filename)
{
    if (!filename) {
        return NULL;
    }

    printf("快速加载RAW预览: %s\n", filename);

    // 尝试提取缩略图
    RawImage *raw = raw_open(filename);
    if (!raw) {
        return NULL;
    }

    Image *preview = raw_extract_thumbnail(raw);
    
    // 如果没有缩略图，使用半尺寸模式
    if (!preview) {
        printf("无缩略图，使用半尺寸模式\n");
        
        RawParseOptions options = raw_create_default_options();
        options.half_size = true;
        options.demosaic_algorithm = 0; // 使用最快的算法
        
        preview = raw_process(raw, &options);
    }

    raw_close(raw);
    return preview;
}

/**
 * @brief 一步加载并处理RAW图像
 */
Image* raw_load_and_process(const char *filename)
{
    if (!filename) {
        return NULL;
    }

    printf("加载并处理RAW图像: %s\n", filename);

    // 打开RAW文件
    RawImage *raw = raw_open(filename);
    if (!raw) {
        return NULL;
    }

    // 使用默认选项处理
    Image *img = raw_process(raw, NULL);

    // 关闭RAW文件
    raw_close(raw);

    return img;
}

// ============================================================================
// 高级处理函数
// ============================================================================

/**
 * @brief 噪点抑制（简单中值滤波）
 */
static bool raw_denoise_median(Image *img, int radius)
{
    if (!img || radius < 1) {
        return false;
    }

    printf("应用中值滤波降噪: radius=%d\n", radius);

    int width = img->width;
    int height = img->height;
    int channels = img->channels;

    // 创建临时缓冲区
    unsigned char *temp = (unsigned char*)malloc(width * height * channels);
    if (!temp) {
        fprintf(stderr, "分配临时缓冲区失败\n");
        return false;
    }

    memcpy(temp, img->data, width * height * channels);

    // 窗口大小
    int window_size = (2 * radius + 1) * (2 * radius + 1);
    unsigned char *window = (unsigned char*)malloc(window_size);

    // 对每个通道应用中值滤波
    for (int c = 0; c < channels; c++) {
        for (int y = radius; y < height - radius; y++) {
            for (int x = radius; x < width - radius; x++) {
                int count = 0;

                // 收集窗口内的值
                for (int dy = -radius; dy <= radius; dy++) {
                    for (int dx = -radius; dx <= radius; dx++) {
                        int idx = ((y + dy) * width + (x + dx)) * channels + c;
                        window[count++] = temp[idx];
                    }
                }

                // 排序找中值（简单冒泡排序）
                for (int i = 0; i < count - 1; i++) {
                    for (int j = 0; j < count - i - 1; j++) {
                        if (window[j] > window[j + 1]) {
                            unsigned char t = window[j];
                            window[j] = window[j + 1];
                            window[j + 1] = t;
                        }
                    }
                }

                // 设置中值
                int idx = (y * width + x) * channels + c;
                img->data[idx] = window[count / 2];
            }
        }
    }

    free(window);
    free(temp);

    printf("降噪完成\n");
    return true;
}

/**
 * @brief 锐化处理
 */
static bool raw_sharpen(Image *img, float amount)
{
    if (!img || amount <= 0.0f) {
        return false;
    }

    printf("应用锐化: amount=%.2f\n", amount);

    int width = img->width;
    int height = img->height;
    int channels = img->channels;

    // 拉普拉斯锐化核
    float kernel[3][3] = {
        { 0, -1,  0},
        {-1,  5, -1},
        { 0, -1,  0}
    };

    // 调整核强度
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            if (i == 1 && j == 1) {
                kernel[i][j] = 1.0f + 4.0f * amount;
            } else if (kernel[i][j] != 0) {
                kernel[i][j] = -amount;
            }
        }
    }

    // 创建临时缓冲区
    unsigned char *temp = (unsigned char*)malloc(width * height * channels);
    if (!temp) {
        fprintf(stderr, "分配临时缓冲区失败\n");
        return false;
    }

    memcpy(temp, img->data, width * height * channels);

    // 应用卷积
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            for (int c = 0; c < channels; c++) {
                float sum = 0.0f;

                for (int ky = 0; ky < 3; ky++) {
                    for (int kx = 0; kx < 3; kx++) {
                        int idx = ((y + ky - 1) * width + (x + kx - 1)) * channels + c;
                        sum += temp[idx] * kernel[ky][kx];
                    }
                }

                // 裁剪到有效范围
                sum = fmaxf(0.0f, fminf(255.0f, sum));
                
                int idx = (y * width + x) * channels + c;
                img->data[idx] = (unsigned char)(sum + 0.5f);
            }
        }
    }

    free(temp);

    printf("锐化完成\n");
    return true;
}

// ============================================================================
// 批处理辅助函数
// ============================================================================

/**
 * @brief 批量处理RAW文件
 */
bool raw_batch_process(const char **filenames, int count, 
                       const RawParseOptions *options,
                       bool (*callback)(const char*, Image*, void*),
                       void *user_data)
{
    if (!filenames || count <= 0) {
        return false;
    }

    printf("开始批量处理 %d 个RAW文件\n", count);

    int success_count = 0;
    int fail_count = 0;

    for (int i = 0; i < count; i++) {
        printf("\n[%d/%d] 处理: %s\n", i + 1, count, filenames[i]);

        RawImage *raw = raw_open(filenames[i]);
        if (!raw) {
            fprintf(stderr, "打开失败: %s\n", filenames[i]);
            fail_count++;
            continue;
        }

        Image *img = raw_process(raw, options);
        raw_close(raw);

        if (!img) {
            fprintf(stderr, "处理失败: %s\n", filenames[i]);
            fail_count++;
            continue;
        }

        // 调用回调函数
        bool result = true;
        if (callback) {
            result = callback(filenames[i], img, user_data);
        }

        image_destroy(img);

        if (result) {
            success_count++;
        } else {
            fail_count++;
        }
    }

    printf("\n批量处理完成: 成功=%d, 失败=%d\n", success_count, fail_count);
    return (fail_count == 0);
}

// 第三部分结束
// ============================================================================
// 调试和诊断函数
// ============================================================================

/**
 * @brief 验证RAW数据完整性
 */
bool raw_validate_data(const RawImage *raw)
{
    if (!raw) {
        fprintf(stderr, "RAW图像对象为空\n");
        return false;
    }

    printf("验证RAW数据完整性...\n");

    // 检查基本字段
    if (raw->metadata.width <= 0 || raw->metadata.height <= 0) {
        fprintf(stderr, "无效的图像尺寸: %dx%d\n", 
                raw->metadata.width, raw->metadata.height);
        return false;
    }

    if (raw->metadata.raw_width <= 0 || raw->metadata.raw_height <= 0) {
        fprintf(stderr, "无效的RAW尺寸: %dx%d\n", 
                raw->metadata.raw_width, raw->metadata.raw_height);
        return false;
    }

    if (!raw->raw_data) {
        fprintf(stderr, "RAW数据指针为空\n");
        return false;
    }

    if (raw->data_size == 0) {
        fprintf(stderr, "RAW数据大小为0\n");
        return false;
    }

    // 检查数据范围
    size_t pixel_count = raw->metadata.raw_width * raw->metadata.raw_height;
    uint16_t min_val = 65535;
    uint16_t max_val = 0;
    uint64_t sum = 0;

    for (size_t i = 0; i < pixel_count; i++) {
        uint16_t val = raw->raw_data[i];
        if (val < min_val) min_val = val;
        if (val > max_val) max_val = val;
        sum += val;
    }

    float avg_val = (float)sum / pixel_count;

    printf("  像素统计:\n");
    printf("    最小值: %u\n", min_val);
    printf("    最大值: %u\n", max_val);
    printf("    平均值: %.2f\n", avg_val);
    printf("    动态范围: %u\n", max_val - min_val);

    // 检查是否有异常值
    if (max_val > raw->metadata.white_level) {
        fprintf(stderr, "警告: 检测到超过白电平的像素值\n");
    }

    if (min_val < raw->metadata.black_level[0]) {
        fprintf(stderr, "警告: 检测到低于黑电平的像素值\n");
    }

    // 检查白平衡系数
    if (raw->metadata.wb_coeffs.r_multiplier <= 0 ||
        raw->metadata.wb_coeffs.g_multiplier <= 0 ||
        raw->metadata.wb_coeffs.b_multiplier <= 0) {
        fprintf(stderr, "警告: 无效的白平衡系数\n");
    }

    printf("RAW数据验证通过\n");
    return true;
}

/**
 * @brief 生成RAW数据直方图
 */
bool raw_generate_histogram(const RawImage *raw, int *histogram, int bins)
{
    if (!raw || !histogram || bins <= 0) {
        return false;
    }

    printf("生成RAW数据直方图 (%d bins)...\n", bins);

    // 初始化直方图
    memset(histogram, 0, bins * sizeof(int));

    size_t pixel_count = raw->metadata.raw_width * raw->metadata.raw_height;
    uint16_t max_value = raw->metadata.white_level;

    if (max_value == 0) {
        max_value = 65535;
    }

    // 统计像素分布
    for (size_t i = 0; i < pixel_count; i++) {
        uint16_t val = raw->raw_data[i];
        int bin = (int)((float)val / max_value * (bins - 1));
        
        if (bin >= 0 && bin < bins) {
            histogram[bin]++;
        }
    }

    // 打印简单的ASCII直方图
    printf("\n直方图分布:\n");
    int max_count = 0;
    for (int i = 0; i < bins; i++) {
        if (histogram[i] > max_count) {
            max_count = histogram[i];
        }
    }

    const int bar_width = 50;
    for (int i = 0; i < bins; i += bins / 20) {
        int bar_len = (int)((float)histogram[i] / max_count * bar_width);
        printf("%3d%% |", i * 100 / bins);
        for (int j = 0; j < bar_len; j++) {
            printf("█");
        }
        printf(" %d\n", histogram[i]);
    }

    printf("\n");
    return true;
}

/**
 * @brief 导出RAW数据为CSV（用于分析）
 */
bool raw_export_csv(const RawImage *raw, const char *filename, int sample_rate)
{
    if (!raw || !filename || sample_rate <= 0) {
        return false;
    }

    printf("导出RAW数据到CSV: %s (采样率: 1/%d)\n", filename, sample_rate);

    FILE *fp = fopen(filename, "w");
    if (!fp) {
        fprintf(stderr, "无法创建CSV文件: %s\n", filename);
        return false;
    }

    // 写入头部
    fprintf(fp, "X,Y,Value,Color\n");

    int width = raw->metadata.raw_width;
    int height = raw->metadata.raw_height;

    // 采样并写入数据
    int count = 0;
    for (int y = 0; y < height; y += sample_rate) {
        for (int x = 0; x < width; x += sample_rate) {
            int idx = y * width + x;
            uint16_t val = raw->raw_data[idx];

            // 确定颜色（基于Bayer模式）
            const char *color = "?";
            switch (raw->metadata.bayer_pattern) {
                case BAYER_PATTERN_RGGB:
                    color = ((y % 2 == 0) ? ((x % 2 == 0) ? "R" : "G") : 
                                           ((x % 2 == 0) ? "G" : "B"));
                    break;
                case BAYER_PATTERN_BGGR:
                    color = ((y % 2 == 0) ? ((x % 2 == 0) ? "B" : "G") : 
                                           ((x % 2 == 0) ? "G" : "R"));
                    break;
                case BAYER_PATTERN_GRBG:
                    color = ((y % 2 == 0) ? ((x % 2 == 0) ? "G" : "R") : 
                                           ((x % 2 == 0) ? "B" : "G"));
                    break;
                case BAYER_PATTERN_GBRG:
                    color = ((y % 2 == 0) ? ((x % 2 == 0) ? "G" : "B") : 
                                           ((x % 2 == 0) ? "R" : "G"));
                    break;
                default:
                    color = "?";
                    break;
            }

            fprintf(fp, "%d,%d,%u,%s\n", x, y, val, color);
            count++;
        }
    }

    fclose(fp);
    printf("成功导出 %d 个采样点\n", count);
    return true;
}

/**
 * @brief 比较两个RAW文件的元数据
 */
void raw_compare_metadata(const RawMetadata *meta1, const RawMetadata *meta2)
{
    if (!meta1 || !meta2) {
        return;
    }

    printf("\n========== RAW元数据比较 ==========\n");

    // 格式
    printf("格式: %s vs %s %s\n", 
           raw_get_format_name(meta1->format),
           raw_get_format_name(meta2->format),
           (meta1->format == meta2->format) ? "✓" : "✗");

    // 制造商和型号
    printf("制造商: %s vs %s %s\n", 
           meta1->make, meta2->make,
           (strcmp(meta1->make, meta2->make) == 0) ? "✓" : "✗");
    
    printf("型号: %s vs %s %s\n", 
           meta1->model, meta2->model,
           (strcmp(meta1->model, meta2->model) == 0) ? "✓" : "✗");

    // 尺寸
    printf("图像尺寸: %dx%d vs %dx%d %s\n",
           meta1->width, meta1->height,
           meta2->width, meta2->height,
           (meta1->width == meta2->width && meta1->height == meta2->height) ? "✓" : "✗");

    // Bayer模式
    printf("Bayer模式: %s vs %s %s\n",
           raw_get_bayer_pattern_name(meta1->bayer_pattern),
           raw_get_bayer_pattern_name(meta2->bayer_pattern),
           (meta1->bayer_pattern == meta2->bayer_pattern) ? "✓" : "✗");

    // 曝光参数
    printf("ISO: %.0f vs %.0f %s\n",
           meta1->iso_speed, meta2->iso_speed,
           (fabs(meta1->iso_speed - meta2->iso_speed) < 0.1) ? "✓" : "✗");

    printf("快门速度: 1/%.0f vs 1/%.0f %s\n",
           1.0 / meta1->shutter_speed, 1.0 / meta2->shutter_speed,
           (fabs(meta1->shutter_speed - meta2->shutter_speed) < 0.001) ? "✓" : "✗");

    printf("光圈: f/%.1f vs f/%.1f %s\n",
           meta1->aperture, meta2->aperture,
           (fabs(meta1->aperture - meta2->aperture) < 0.1) ? "✓" : "✗");

    printf("====================================\n\n");
}

// ============================================================================
// 性能测试函数
// ============================================================================

/**
 * @brief 测试RAW解析性能
 */
void raw_benchmark(const char *filename, int iterations)
{
    if (!filename || iterations <= 0) {
        return;
    }

    printf("\n========== RAW解析性能测试 ==========\n");
    printf("文件: %s\n", filename);
    printf("迭代次数: %d\n\n", iterations);

    // 测试1: 打开文件
    clock_t start = clock();
    for (int i = 0; i < iterations; i++) {
        RawImage *raw = raw_open(filename);
        if (raw) {
            raw_close(raw);
        }
    }
    clock_t end = clock();
    double open_time = (double)(end - start) / CLOCKS_PER_SEC / iterations;
    printf("平均打开时间: %.3f 秒\n", open_time);

    // 测试2: 处理图像
    RawImage *raw = raw_open(filename);
    if (raw) {
        start = clock();
        for (int i = 0; i < iterations; i++) {
            Image *img = raw_process(raw, NULL);
            if (img) {
                image_destroy(img);
            }
        }
        end = clock();
        double process_time = (double)(end - start) / CLOCKS_PER_SEC / iterations;
        printf("平均处理时间: %.3f 秒\n", process_time);

        // 测试3: 提取缩略图
        start = clock();
        for (int i = 0; i < iterations; i++) {
            Image *thumb = raw_extract_thumbnail(raw);
            if (thumb) {
                image_destroy(thumb);
            }
        }
        end = clock();
        double thumb_time = (double)(end - start) / CLOCKS_PER_SEC / iterations;
        printf("平均缩略图提取时间: %.3f 秒\n", thumb_time);

        raw_close(raw);
    }

    printf("======================================\n\n");
}

// ============================================================================
// 示例和测试代码
// ============================================================================

/**
 * @brief 示例1: 基本RAW文件读取
 */
void raw_example_basic(const char *filename)
{
    printf("\n========== 示例1: 基本RAW文件读取 ==========\n");

    // 1. 检测格式
    RawFormat format = raw_detect_format(filename);
    printf("检测到格式: %s\n", raw_get_format_name(format));

    // 2. 打开RAW文件
    RawImage *raw = raw_open(filename);
    if (!raw) {
        fprintf(stderr, "打开RAW文件失败\n");
        return;
    }

    // 3. 打印元数据
    raw_print_metadata(&raw->metadata);

    // 4. 验证数据
    raw_validate_data(raw);

    // 5. 处理为RGB图像
    Image *img = raw_process(raw, NULL);
    if (img) {
        printf("成功处理为RGB图像: %dx%d\n", img->width, img->height);
        
        // 保存结果
        char output[256];
        snprintf(output, sizeof(output), "output_basic.png");
        image_save(img, output);
        printf("保存到: %s\n", output);
        
        image_destroy(img);
    }

    // 6. 清理
    raw_close(raw);
    printf("==========================================\n\n");
}

/**
 * @brief 示例2: 自定义处理选项
 */
void raw_example_custom_options(const char *filename)
{
    printf("\n========== 示例2: 自定义处理选项 ==========\n");

    // 1. 创建自定义选项
    RawParseOptions options = raw_create_default_options();
    
    // 使用自动白平衡
    options.use_camera_wb = false;
    options.use_auto_wb = true;
    
    // 使用高质量去马赛克算法 (AHD)
    options.demosaic_algorithm = 3;
    
    // 高光恢复
    options.highlight_mode = 2;
    
    // Adobe RGB色彩空间
    options.output_color_space = 2;

    printf("处理选项:\n");
    printf("  自动白平衡: %s\n", options.use_auto_wb ? "是" : "否");
    printf("  去马赛克算法: %d\n", options.demosaic_algorithm);
    printf("  高光恢复模式: %d\n", options.highlight_mode);
    printf("  色彩空间: %d\n", options.output_color_space);

    // 2. 打开并处理
    RawImage *raw = raw_open_with_options(filename, &options);
    if (!raw) {
        fprintf(stderr, "打开RAW文件失败\n");
        return;
    }

    Image *img = raw_process(raw, &options);
    if (img) {
        // 保存结果
        char output[256];
        snprintf(output, sizeof(output), "output_custom.png");
        image_save(img, output);
        printf("保存到: %s\n", output);
        
        image_destroy(img);
    }

    raw_close(raw);
    printf("==========================================\n\n");
}

/**
 * @brief 示例3: 白平衡调整
 */
void raw_example_white_balance(const char *filename)
{
    printf("\n========== 示例3: 白平衡调整 ==========\n");

    RawImage *raw = raw_open(filename);
    if (!raw) {
        return;
    }

    // 1. 使用相机白平衡
    printf("\n1. 相机白平衡:\n");
    RawParseOptions options1 = raw_create_default_options();
    options1.use_camera_wb = true;
    Image *img1 = raw_process(raw, &options1);
    if (img1) {
        image_save(img1, "output_wb_camera.png");
        image_destroy(img1);
    }

    // 2. 使用自动白平衡
    printf("\n2. 自动白平衡:\n");
    RawParseOptions options2 = raw_create_default_options();
    options2.use_auto_wb = true;
    options2.use_camera_wb = false;
    Image *img2 = raw_process(raw, &options2);
    if (img2) {
        image_save(img2, "output_wb_auto.png");
        image_destroy(img2);
    }

    // 3. 自定义白平衡
    printf("\n3. 自定义白平衡:\n");
    WhiteBalance custom_wb;
    custom_wb.r_multiplier = 2.0f;
    custom_wb.g_multiplier = 1.0f;
    custom_wb.b_multiplier = 1.5f;
    custom_wb.g2_multiplier = 1.0f;
    
    raw_apply_white_balance(raw, &custom_wb);
    
    RawParseOptions options3 = raw_create_default_options();
    options3.use_camera_wb = false;
    options3.use_auto_wb = false;
    Image *img3 = raw_process(raw, &options3);
    if (img3) {
        image_save(img3, "output_wb_custom.png");
        image_destroy(img3);
    }

    raw_close(raw);
    printf("==========================================\n\n");
}

/**
 * @brief 示例4: 去马赛克算法比较
 */
void raw_example_demosaic_comparison(const char *filename)
{
    printf("\n========== 示例4: 去马赛克算法比较 ==========\n");

    RawImage *raw = raw_open(filename);
    if (!raw) {
        return;
    }

    const char *algorithm_names[] = {
        "Linear", "VNG", "PPG", "AHD", "DCB", "Modified AHD", "AFD", "VCD"
    };

    // 测试不同的去马赛克算法
    for (int algo = 0; algo <= 3; algo++) {
        printf("\n测试算法 %d: %s\n", algo, 
               (algo < 8) ? algorithm_names[algo] : "Unknown");

        clock_t start = clock();
        Image *img = raw_demosaic(raw, algo);
        clock_t end = clock();
        
        double time_taken = (double)(end - start) / CLOCKS_PER_SEC;
        
        if (img) {
            char output[256];
            snprintf(output, sizeof(output), "output_demosaic_%d.png", algo);
            image_save(img, output);
            printf("  处理时间: %.3f 秒\n", time_taken);
            printf("  保存到: %s\n", output);
            image_destroy(img);
        }
    }

    raw_close(raw);
    printf("==========================================\n\n");
}

/**
 * @brief 示例5: 批量处理
 */
static bool batch_callback(const char *filename, Image *img, void *user_data)
{
    // 生成输出文件名
    char output[512];
    const char *basename = strrchr(filename, '/');
    if (!basename) {
        basename = strrchr(filename, '\\');
    }
    if (!basename) {
        basename = filename;
    } else {
        basename++;
    }

    // 移除扩展名
    char name_without_ext[256];
    strncpy(name_without_ext, basename, sizeof(name_without_ext) - 1);
    char *dot = strrchr(name_without_ext, '.');
    if (dot) {
        *dot = '\0';
    }

    snprintf(output, sizeof(output), "batch_%s.jpg", name_without_ext);

    // 保存图像
    bool result = image_save(img, output);
    if (result) {
        printf("  保存成功: %s\n", output);
    }

    return result;
}

void raw_example_batch_processing(const char **filenames, int count)
{
    printf("\n========== 示例5: 批量处理 ==========\n");

    RawParseOptions options = raw_create_default_options();
    options.use_auto_wb = true;
    options.demosaic_algorithm = 3;

    raw_batch_process(filenames, count, &options, batch_callback, NULL);

    printf("==========================================\n\n");
}

/**
 * @brief 示例6: 高级后处理
 */
void raw_example_post_processing(const char *filename)
{
    printf("\n========== 示例6: 高级后处理 ==========\n");

    RawImage *raw = raw_open(filename);
    if (!raw) {
        return;
    }

    Image *img = raw_process(raw, NULL);
    if (!img) {
        raw_close(raw);
        return;
    }

    // 1. 高光恢复
    printf("应用高光恢复...\n");
    raw_recover_highlights(img, 2);
    image_save(img, "output_post_1_highlights.png");

    // 2. 色彩空间转换
    printf("转换到Adobe RGB...\n");
    raw_convert_color_space(img, 2);
    image_save(img, "output_post_2_colorspace.png");

    // 3. 伽马校正
    printf("应用伽马校正...\n");
    raw_apply_gamma(img, 2.2f, 4.5f);
    image_save(img, "output_post_3_gamma.png");

    // 4. 降噪
    printf("应用降噪...\n");
    raw_denoise_median(img, 1);
    image_save(img, "output_post_4_denoise.png");

    // 5. 锐化
    printf("应用锐化...\n");
    raw_sharpen(img, 0.5f);
    image_save(img, "output_post_5_sharpen.png");

    image_destroy(img);
    raw_close(raw);
    printf("==========================================\n\n");
}

/**
 * @brief 主测试函数
 */
void raw_run_all_examples(const char *filename)
{
    printf("\n");
    printf("╔════════════════════════════════════════════════╗\n");
    printf("║        RAW图像解析器 - 完整示例测试           ║\n");
    printf("╚════════════════════════════════════════════════╝\n");
    printf("\n");

    // 初始化
    if (!raw_parser_init()) {
        fprintf(stderr, "RAW解析器初始化失败\n");
        return;
    }

    // 运行所有示例
    raw_example_basic(filename);
    raw_example_custom_options(filename);
    raw_example_white_balance(filename);
    raw_example_demosaic_comparison(filename);
    raw_example_post_processing(filename);

    // 性能测试
    raw_benchmark(filename, 3);

    // 清理
    raw_parser_cleanup();

    printf("\n所有示例测试完成！\n\n");
}

// raw_parser.c 

