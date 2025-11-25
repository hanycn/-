/**
 * @file result_exporter.c
 * @brief 结果导出模块实现
 */

#include "result_exporter.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/stat.h>
#include <errno.h>

#ifdef _WIN32
#include <direct.h>
#define mkdir(path, mode) _mkdir(path)
#else
#include <sys/types.h>
#endif

// ============================================================================
// 内部状态和错误处理
// ============================================================================

static char g_error_message[512] = {0};
static bool g_verbose = false;

/**
 * @brief 设置错误信息
 */
static void export_set_error(const char *format, ...)
{
    va_list args;
    va_start(args, format);
    vsnprintf(g_error_message, sizeof(g_error_message), format, args);
    va_end(args);
    
    if (g_verbose) {
        fprintf(stderr, "[Export Error] %s\n", g_error_message);
    }
}

const char* export_get_error(void)
{
    return g_error_message;
}

void export_clear_error(void)
{
    g_error_message[0] = '\0';
}

void export_set_verbose(bool verbose)
{
    g_verbose = verbose;
}

const char* export_get_version(void)
{
    return "1.0.0";
}

// ============================================================================
// 初始化和清理
// ============================================================================

bool exporter_init(void)
{
    export_clear_error();
    return true;
}

void exporter_cleanup(void)
{
    export_clear_error();
}

// ============================================================================
// 默认选项
// ============================================================================

ExportOptions exporter_default_options(void)
{
    ExportOptions options = {
        .include_metadata = true,
        .include_statistics = true,
        .include_confidence = true,
        .include_coordinates = true,
        .include_timestamp = true,
        .pretty_print = true,
        .use_relative_paths = false,
        .precision = 6,
        .encoding = "UTF-8"
    };
    return options;
}

ImageExportOptions exporter_default_image_options(void)
{
    ImageExportOptions options = {
        .draw_boxes = true,
        .draw_labels = true,
        .draw_confidence = true,
        .draw_grid = false,
        .box_thickness = 2,
        .font_size = 12,
        .box_color = {255, 0, 0, 255},      // 红色
        .text_color = {255, 255, 255, 255}, // 白色
        .bg_color = {0, 0, 0, 128},         // 半透明黑色
        .opacity = 0.7f
    };
    return options;
}

ReportConfig exporter_default_report_config(void)
{
    ReportConfig config = {
        .title = "Detection Report",
        .author = "AI Detector",
        .description = "Automated detection results",
        .version = "1.0",
        .include_summary = true,
        .include_details = true,
        .include_images = true,
        .include_charts = false
    };
    return config;
}

// ============================================================================
// 路径和文件操作
// ============================================================================

/**
 * @brief 创建目录（递归）
 */
static bool create_directory_recursive(const char *path)
{
    char tmp[512];
    char *p = NULL;
    size_t len;
    
    snprintf(tmp, sizeof(tmp), "%s", path);
    len = strlen(tmp);
    
    if (tmp[len - 1] == '/' || tmp[len - 1] == '\\') {
        tmp[len - 1] = 0;
    }
    
    for (p = tmp + 1; *p; p++) {
        if (*p == '/' || *p == '\\') {
            *p = 0;
            if (mkdir(tmp, 0755) != 0 && errno != EEXIST) {
                return false;
            }
            *p = '/';
        }
    }
    
    if (mkdir(tmp, 0755) != 0 && errno != EEXIST) {
        return false;
    }
    
    return true;
}

bool export_validate_path(const char *output_path, bool create_dirs)
{
    if (!output_path || strlen(output_path) == 0) {
        export_set_error("输出路径为空");
        return false;
    }
    
    // 提取目录路径
    char dir_path[512];
    const char *last_slash = strrchr(output_path, '/');
    const char *last_backslash = strrchr(output_path, '\\');
    const char *separator = (last_slash > last_backslash) ? last_slash : last_backslash;
    
    if (separator) {
        size_t dir_len = separator - output_path;
        if (dir_len >= sizeof(dir_path)) {
            export_set_error("路径太长");
            return false;
        }
        strncpy(dir_path, output_path, dir_len);
        dir_path[dir_len] = '\0';
        
        // 检查目录是否存在
        struct stat st;
        if (stat(dir_path, &st) != 0) {
            if (create_dirs) {
                if (!create_directory_recursive(dir_path)) {
                    export_set_error("无法创建目录: %s", dir_path);
                    return false;
                }
            } else {
                export_set_error("目录不存在: %s", dir_path);
                return false;
            }
        } else if (!S_ISDIR(st.st_mode)) {
            export_set_error("路径不是目录: %s", dir_path);
            return false;
        }
    }
    
    return true;
}

const char* export_get_extension(ExportFormat format)
{
    switch (format) {
        case EXPORT_FORMAT_JSON:     return "json";
        case EXPORT_FORMAT_XML:      return "xml";
        case EXPORT_FORMAT_CSV:      return "csv";
        case EXPORT_FORMAT_TXT:      return "txt";
        case EXPORT_FORMAT_HTML:     return "html";
        case EXPORT_FORMAT_MARKDOWN: return "md";
        case EXPORT_FORMAT_YAML:     return "yaml";
        case EXPORT_FORMAT_LATEX:    return "tex";
        default:                     return "txt";
    }
}

ExportFormat export_format_from_extension(const char *extension)
{
    if (!extension) return EXPORT_FORMAT_JSON;
    
    if (strcmp(extension, "json") == 0) return EXPORT_FORMAT_JSON;
    if (strcmp(extension, "xml") == 0) return EXPORT_FORMAT_XML;
    if (strcmp(extension, "csv") == 0) return EXPORT_FORMAT_CSV;
    if (strcmp(extension, "txt") == 0) return EXPORT_FORMAT_TXT;
    if (strcmp(extension, "html") == 0 || strcmp(extension, "htm") == 0) 
        return EXPORT_FORMAT_HTML;
    if (strcmp(extension, "md") == 0 || strcmp(extension, "markdown") == 0) 
        return EXPORT_FORMAT_MARKDOWN;
    if (strcmp(extension, "yaml") == 0 || strcmp(extension, "yml") == 0) 
        return EXPORT_FORMAT_YAML;
    if (strcmp(extension, "tex") == 0) return EXPORT_FORMAT_LATEX;
    
    return EXPORT_FORMAT_JSON;
}

// ============================================================================
// JSON导出
// ============================================================================

/**
 * @brief 转义JSON字符串
 */
static void json_escape_string(const char *str, char *output, size_t output_size)
{
    size_t i = 0, j = 0;
    
    while (str[i] && j < output_size - 2) {
        switch (str[i]) {
            case '"':
                if (j < output_size - 3) {
                    output[j++] = '\\';
                    output[j++] = '"';
                }
                break;
            case '\\':
                if (j < output_size - 3) {
                    output[j++] = '\\';
                    output[j++] = '\\';
                }
                break;
            case '\n':
                if (j < output_size - 3) {
                    output[j++] = '\\';
                    output[j++] = 'n';
                }
                break;
            case '\r':
                if (j < output_size - 3) {
                    output[j++] = '\\';
                    output[j++] = 'r';
                }
                break;
            case '\t':
                if (j < output_size - 3) {
                    output[j++] = '\\';
                    output[j++] = 't';
                }
                break;
            default:
                output[j++] = str[i];
                break;
        }
        i++;
    }
    output[j] = '\0';
}

/**
 * @brief 写入缩进
 */
static void write_indent(FILE *fp, int level, bool pretty)
{
    if (pretty) {
        for (int i = 0; i < level; i++) {
            fprintf(fp, "  ");
        }
    }
}

bool export_to_json(const DetectionResult *results, int count,
                   const char *output_file,
                   const ExportOptions *options)
{
    export_clear_error();
    
    if (!results || count <= 0 || !output_file) {
        export_set_error("无效的参数");
        return false;
    }
    
    if (!export_validate_path(output_file, true)) {
        return false;
    }
    
    // 使用默认选项
    ExportOptions default_opts = exporter_default_options();
    if (!options) {
        options = &default_opts;
    }
    
    FILE *fp = fopen(output_file, "w");
    if (!fp) {
        export_set_error("无法打开文件: %s", output_file);
        return false;
    }
    
    bool pretty = options->pretty_print;
    const char *nl = pretty ? "\n" : "";
    const char *sp = pretty ? " " : "";
    
    // 开始JSON对象
    fprintf(fp, "{%s", nl);
    
    // 元数据
    if (options->include_metadata) {
        write_indent(fp, 1, pretty);
        fprintf(fp, "\"metadata\":%s{%s", sp, nl);
        
        write_indent(fp, 2, pretty);
        fprintf(fp, "\"version\":%s\"%s\",%s", sp, export_get_version(), nl);
        
        if (options->include_timestamp) {
            write_indent(fp, 2, pretty);
            time_t now = time(NULL);
            fprintf(fp, "\"timestamp\":%s%ld,%s", sp, (long)now, nl);
            
            char time_str[64];
            strftime(time_str, sizeof(time_str), "%Y-%m-%d %H:%M:%S", 
                    localtime(&now));
            write_indent(fp, 2, pretty);
            fprintf(fp, "\"timestamp_str\":%s\"%s\",%s", sp, time_str, nl);
        }
        
        write_indent(fp, 2, pretty);
        fprintf(fp, "\"total_detections\":%s%d%s", sp, count, nl);
        
        write_indent(fp, 1, pretty);
        fprintf(fp, "},%s", nl);
    }
    
    // 检测结果数组
    write_indent(fp, 1, pretty);
    fprintf(fp, "\"detections\":%s[%s", sp, nl);
    
    for (int i = 0; i < count; i++) {
        const DetectionResult *result = &results[i];
        
        write_indent(fp, 2, pretty);
        fprintf(fp, "{%s", nl);
        
        // ID
        write_indent(fp, 3, pretty);
        fprintf(fp, "\"id\":%s%d,%s", sp, i, nl);
        
        // 类型
        write_indent(fp, 3, pretty);
        const char *type_str = (result->type == DETECTION_AI_GENERATED) ? 
                              "ai_generated" : "real";
        fprintf(fp, "\"type\":%s\"%s\",%s", sp, type_str, nl);
        
        // 置信度
        if (options->include_confidence) {
            write_indent(fp, 3, pretty);
            fprintf(fp, "\"confidence\":%s%.*f,%s", 
                   sp, options->precision, result->confidence, nl);
        }
        
        // 坐标
        if (options->include_coordinates && result->has_region) {
            write_indent(fp, 3, pretty);
            fprintf(fp, "\"region\":%s{%s", sp, nl);
            
            write_indent(fp, 4, pretty);
            fprintf(fp, "\"x\":%s%d,%s", sp, result->region.x, nl);
            
            write_indent(fp, 4, pretty);
            fprintf(fp, "\"y\":%s%d,%s", sp, result->region.y, nl);
            
            write_indent(fp, 4, pretty);
            fprintf(fp, "\"width\":%s%d,%s", sp, result->region.width, nl);
            
            write_indent(fp, 4, pretty);
            fprintf(fp, "\"height\":%s%d%s", sp, result->region.height, nl);
            
            write_indent(fp, 3, pretty);
            fprintf(fp, "},%s", nl);
        }
        
        // 特征
        write_indent(fp, 3, pretty);
        fprintf(fp, "\"features\":%s{%s", sp, nl);
        
        write_indent(fp, 4, pretty);
        fprintf(fp, "\"noise_level\":%s%.*f,%s", 
               sp, options->precision, result->features.noise_level, nl);
        
        write_indent(fp, 4, pretty);
        fprintf(fp, "\"compression_artifacts\":%s%.*f,%s", 
               sp, options->precision, result->features.compression_artifacts, nl);
        
        write_indent(fp, 4, pretty);
        fprintf(fp, "\"edge_consistency\":%s%.*f,%s", 
               sp, options->precision, result->features.edge_consistency, nl);
        
        write_indent(fp, 4, pretty);
        fprintf(fp, "\"color_distribution\":%s%.*f,%s", 
               sp, options->precision, result->features.color_distribution, nl);
        
        write_indent(fp, 4, pretty);
        fprintf(fp, "\"texture_pattern\":%s%.*f%s", 
               sp, options->precision, result->features.texture_pattern, nl);
        
        write_indent(fp, 3, pretty);
        fprintf(fp, "}%s", nl);
        
        write_indent(fp, 2, pretty);
        fprintf(fp, "}");
        
        if (i < count - 1) {
            fprintf(fp, ",");
        }
        fprintf(fp, "%s", nl);
    }
    
    write_indent(fp, 1, pretty);
    fprintf(fp, "]%s", nl);
    
    // 统计信息
    if (options->include_statistics) {
        fprintf(fp, ",%s", nl);
        
        ExportStatistics stats;
        if (export_calculate_statistics(results, count, &stats)) {
            write_indent(fp, 1, pretty);
            fprintf(fp, "\"statistics\":%s{%s", sp, nl);
            
            write_indent(fp, 2, pretty);
            fprintf(fp, "\"total_detections\":%s%d,%s", 
                   sp, stats.total_detections, nl);
            
            write_indent(fp, 2, pretty);
            fprintf(fp, "\"average_confidence\":%s%.*f%s", 
                   sp, options->precision, stats.average_confidence, nl);
            
            write_indent(fp, 1, pretty);
            fprintf(fp, "}%s", nl);
        }
    }
    
    // 结束JSON对象
    fprintf(fp, "}%s", nl);
    
    fclose(fp);
    
    if (g_verbose) {
        printf("成功导出JSON到: %s\n", output_file);
    }
    
    return true;
}

// ============================================================================
// XML导出
// ============================================================================

/**
 * @brief 转义XML字符串
 */
static void xml_escape_string(const char *str, char *output, size_t output_size)
{
    size_t i = 0, j = 0;
    
    while (str[i] && j < output_size - 6) {
        switch (str[i]) {
            case '<':
                strcpy(&output[j], "&lt;");
                j += 4;
                break;
            case '>':
                strcpy(&output[j], "&gt;");
                j += 4;
                break;
            case '&':
                strcpy(&output[j], "&amp;");
                j += 5;
                break;
            case '"':
                strcpy(&output[j], "&quot;");
                j += 6;
                break;
            case '\'':
                strcpy(&output[j], "&apos;");
                j += 6;
                break;
            default:
                output[j++] = str[i];
                break;
        }
        i++;
    }
    output[j] = '\0';
}

bool export_to_xml(const DetectionResult *results, int count,
                  const char *output_file,
                  const ExportOptions *options)
{
    export_clear_error();
    
    if (!results || count <= 0 || !output_file) {
        export_set_error("无效的参数");
        return false;
    }
    
    if (!export_validate_path(output_file, true)) {
        return false;
    }
    
    ExportOptions default_opts = exporter_default_options();
    if (!options) {
        options = &default_opts;
    }
    
    FILE *fp = fopen(output_file, "w");
    if (!fp) {
        export_set_error("无法打开文件: %s", output_file);
        return false;
    }
    
    bool pretty = options->pretty_print;
    const char *nl = pretty ? "\n" : "";
    
    // XML声明
    fprintf(fp, "<?xml version=\"1.0\" encoding=\"%s\"?>%s", 
           options->encoding, nl);
    
    // 根元素
    fprintf(fp, "<detection_results>%s", nl);
    
    // 元数据
    if (options->include_metadata) {
        write_indent(fp, 1, pretty);
        fprintf(fp, "<metadata>%s", nl);
        
        write_indent(fp, 2, pretty);
        fprintf(fp, "<version>%s</version>%s", export_get_version(), nl);
        
        if (options->include_timestamp) {
            time_t now = time(NULL);
            char time_str[64];
            strftime(time_str, sizeof(time_str), "%Y-%m-%d %H:%M:%S", 
                    localtime(&now));
            
            write_indent(fp, 2, pretty);
            fprintf(fp, "<timestamp>%ld</timestamp>%s", (long)now, nl);
            
            write_indent(fp, 2, pretty);
            fprintf(fp, "<timestamp_str>%s</timestamp_str>%s", time_str, nl);
        }
        
        write_indent(fp, 2, pretty);
        fprintf(fp, "<total_detections>%d</total_detections>%s", count, nl);
        
        write_indent(fp, 1, pretty);
        fprintf(fp, "</metadata>%s", nl);
    }
    
    // 检测结果
    write_indent(fp, 1, pretty);
    fprintf(fp, "<detections>%s", nl);
    
    for (int i = 0; i < count; i++) {
        const DetectionResult *result = &results[i];
        
        write_indent(fp, 2, pretty);
        fprintf(fp, "<detection id=\"%d\">%s", i, nl);
        
        // 类型
        write_indent(fp, 3, pretty);
        const char *type_str = (result->type == DETECTION_AI_GENERATED) ? 
                              "ai_generated" : "real";
        fprintf(fp, "<type>%s</type>%s", type_str, nl);
        
        // 置信度
        if (options->include_confidence) {
            write_indent(fp, 3, pretty);
            fprintf(fp, "<confidence>%.*f</confidence>%s", 
                   options->precision, result->confidence, nl);
        }
        
        // 区域
        if (options->include_coordinates && result->has_region) {
            write_indent(fp, 3, pretty);
            fprintf(fp, "<region>%s", nl);
            
            write_indent(fp, 4, pretty);
            fprintf(fp, "<x>%d</x>%s", result->region.x, nl);
            
            write_indent(fp, 4, pretty);
            fprintf(fp, "<y>%d</y>%s", result->region.y, nl);
            
            write_indent(fp, 4, pretty);
            fprintf(fp, "<width>%d</width>%s", result->region.width, nl);
            
            write_indent(fp, 4, pretty);
            fprintf(fp, "<height>%d</height>%s", result->region.height, nl);
            
            write_indent(fp, 3, pretty);
            fprintf(fp, "</region>%s", nl);
        }
        
        // 特征
        write_indent(fp, 3, pretty);
        fprintf(fp, "<features>%s", nl);
        
        write_indent(fp, 4, pretty);
        fprintf(fp, "<noise_level>%.*f</noise_level>%s", 
               options->precision, result->features.noise_level, nl);
        
        write_indent(fp, 4, pretty);
        fprintf(fp, "<compression_artifacts>%.*f</compression_artifacts>%s", 
               options->precision, result->features.compression_artifacts, nl);
        
        write_indent(fp, 4, pretty);
        fprintf(fp, "<edge_consistency>%.*f</edge_consistency>%s", 
               options->precision, result->features.edge_consistency, nl);
        
        write_indent(fp, 4, pretty);
        fprintf(fp, "<color_distribution>%.*f</color_distribution>%s", 
               options->precision, result->features.color_distribution, nl);
        
        write_indent(fp, 4, pretty);
        fprintf(fp, "<texture_pattern>%.*f</texture_pattern>%s", 
               options->precision, result->features.texture_pattern, nl);
        
        write_indent(fp, 3, pretty);
        fprintf(fp, "</features>%s", nl);
        
        write_indent(fp, 2, pretty);
        fprintf(fp, "</detection>%s", nl);
    }
    
    write_indent(fp, 1, pretty);
    fprintf(fp, "</detections>%s", nl);
    
    // 统计信息
    if (options->include_statistics) {
        ExportStatistics stats;
        if (export_calculate_statistics(results, count, &stats)) {
            write_indent(fp, 1, pretty);
            fprintf(fp, "<statistics>%s", nl);
            
            write_indent(fp, 2, pretty);
            fprintf(fp, "<total_detections>%d</total_detections>%s", 
                   stats.total_detections, nl);
            
            write_indent(fp, 2, pretty);
            fprintf(fp, "<average_confidence>%.*f</average_confidence>%s", 
                   options->precision, stats.average_confidence, nl);
            
            write_indent(fp, 1, pretty);
            fprintf(fp, "</statistics>%s", nl);
        }
    }
    
    // 关闭根元素
    fprintf(fp, "</detection_results>%s", nl);
    
    fclose(fp);
    
    if (g_verbose) {
        printf("成功导出XML到: %s\n", output_file);
    }
    
    return true;
}
// ============================================================================
// CSV导出
// ============================================================================

bool export_to_csv(const DetectionResult *results, int count,
                  const char *output_file,
                  const ExportOptions *options)
{
    export_clear_error();
    
    if (!results || count <= 0 || !output_file) {
        export_set_error("无效的参数");
        return false;
    }
    
    if (!export_validate_path(output_file, true)) {
        return false;
    }
    
    ExportOptions default_opts = exporter_default_options();
    if (!options) {
        options = &default_opts;
    }
    
    FILE *fp = fopen(output_file, "w");
    if (!fp) {
        export_set_error("无法打开文件: %s", output_file);
        return false;
    }
    
    // 写入CSV头部
    fprintf(fp, "ID,Type");
    
    if (options->include_confidence) {
        fprintf(fp, ",Confidence");
    }
    
    if (options->include_coordinates) {
        fprintf(fp, ",Region_X,Region_Y,Region_Width,Region_Height");
    }
    
    fprintf(fp, ",Noise_Level,Compression_Artifacts,Edge_Consistency,");
    fprintf(fp, "Color_Distribution,Texture_Pattern");
    
    if (options->include_timestamp) {
        fprintf(fp, ",Timestamp");
    }
    
    fprintf(fp, "\n");
    
    // 写入数据行
    time_t now = time(NULL);
    char time_str[64];
    strftime(time_str, sizeof(time_str), "%Y-%m-%d %H:%M:%S", localtime(&now));
    
    for (int i = 0; i < count; i++) {
        const DetectionResult *result = &results[i];
        
        // ID
        fprintf(fp, "%d,", i);
        
        // 类型
        const char *type_str = (result->type == DETECTION_AI_GENERATED) ? 
                              "AI_Generated" : "Real";
        fprintf(fp, "%s", type_str);
        
        // 置信度
        if (options->include_confidence) {
            fprintf(fp, ",%.*f", options->precision, result->confidence);
        }
        
        // 区域坐标
        if (options->include_coordinates) {
            if (result->has_region) {
                fprintf(fp, ",%d,%d,%d,%d",
                       result->region.x, result->region.y,
                       result->region.width, result->region.height);
            } else {
                fprintf(fp, ",,,");
            }
        }
        
        // 特征
        fprintf(fp, ",%.*f,%.*f,%.*f,%.*f,%.*f",
               options->precision, result->features.noise_level,
               options->precision, result->features.compression_artifacts,
               options->precision, result->features.edge_consistency,
               options->precision, result->features.color_distribution,
               options->precision, result->features.texture_pattern);
        
        // 时间戳
        if (options->include_timestamp) {
            fprintf(fp, ",%s", time_str);
        }
        
        fprintf(fp, "\n");
    }
    
    fclose(fp);
    
    if (g_verbose) {
        printf("成功导出CSV到: %s\n", output_file);
    }
    
    return true;
}

// ============================================================================
// 纯文本导出
// ============================================================================

bool export_to_text(const DetectionResult *results, int count,
                   const char *output_file,
                   const ExportOptions *options)
{
    export_clear_error();
    
    if (!results || count <= 0 || !output_file) {
        export_set_error("无效的参数");
        return false;
    }
    
    if (!export_validate_path(output_file, true)) {
        return false;
    }
    
    ExportOptions default_opts = exporter_default_options();
    if (!options) {
        options = &default_opts;
    }
    
    FILE *fp = fopen(output_file, "w");
    if (!fp) {
        export_set_error("无法打开文件: %s", output_file);
        return false;
    }
    
    // 标题
    fprintf(fp, "========================================\n");
    fprintf(fp, "AI Detection Results Report\n");
    fprintf(fp, "========================================\n\n");
    
    // 元数据
    if (options->include_metadata) {
        fprintf(fp, "Metadata:\n");
        fprintf(fp, "  Version: %s\n", export_get_version());
        
        if (options->include_timestamp) {
            time_t now = time(NULL);
            char time_str[64];
            strftime(time_str, sizeof(time_str), "%Y-%m-%d %H:%M:%S", 
                    localtime(&now));
            fprintf(fp, "  Generated: %s\n", time_str);
        }
        
        fprintf(fp, "  Total Detections: %d\n", count);
        fprintf(fp, "\n");
    }
    
    // 统计信息
    if (options->include_statistics) {
        ExportStatistics stats;
        if (export_calculate_statistics(results, count, &stats)) {
            fprintf(fp, "Statistics:\n");
            fprintf(fp, "  Total Detections: %d\n", stats.total_detections);
            fprintf(fp, "  Average Confidence: %.*f\n", 
                   options->precision, stats.average_confidence);
            fprintf(fp, "\n");
        }
    }
    
    // 检测结果
    fprintf(fp, "Detection Results:\n");
    fprintf(fp, "----------------------------------------\n\n");
    
    for (int i = 0; i < count; i++) {
        const DetectionResult *result = &results[i];
        
        fprintf(fp, "Detection #%d:\n", i + 1);
        
        // 类型
        const char *type_str = (result->type == DETECTION_AI_GENERATED) ? 
                              "AI Generated" : "Real";
        fprintf(fp, "  Type: %s\n", type_str);
        
        // 置信度
        if (options->include_confidence) {
            fprintf(fp, "  Confidence: %.*f%%\n", 
                   options->precision, result->confidence * 100.0);
        }
        
        // 区域
        if (options->include_coordinates && result->has_region) {
            fprintf(fp, "  Region:\n");
            fprintf(fp, "    Position: (%d, %d)\n", 
                   result->region.x, result->region.y);
            fprintf(fp, "    Size: %d x %d\n", 
                   result->region.width, result->region.height);
        }
        
        // 特征
        fprintf(fp, "  Features:\n");
        fprintf(fp, "    Noise Level: %.*f\n", 
               options->precision, result->features.noise_level);
        fprintf(fp, "    Compression Artifacts: %.*f\n", 
               options->precision, result->features.compression_artifacts);
        fprintf(fp, "    Edge Consistency: %.*f\n", 
               options->precision, result->features.edge_consistency);
        fprintf(fp, "    Color Distribution: %.*f\n", 
               options->precision, result->features.color_distribution);
        fprintf(fp, "    Texture Pattern: %.*f\n", 
               options->precision, result->features.texture_pattern);
        
        fprintf(fp, "\n");
    }
    
    fprintf(fp, "========================================\n");
    fprintf(fp, "End of Report\n");
    fprintf(fp, "========================================\n");
    
    fclose(fp);
    
    if (g_verbose) {
        printf("成功导出文本到: %s\n", output_file);
    }
    
    return true;
}

// ============================================================================
// YAML导出
// ============================================================================

bool export_to_yaml(const DetectionResult *results, int count,
                   const char *output_file,
                   const ExportOptions *options)
{
    export_clear_error();
    
    if (!results || count <= 0 || !output_file) {
        export_set_error("无效的参数");
        return false;
    }
    
    if (!export_validate_path(output_file, true)) {
        return false;
    }
    
    ExportOptions default_opts = exporter_default_options();
    if (!options) {
        options = &default_opts;
    }
    
    FILE *fp = fopen(output_file, "w");
    if (!fp) {
        export_set_error("无法打开文件: %s", output_file);
        return false;
    }
    
    // YAML文档开始
    fprintf(fp, "---\n");
    
    // 元数据
    if (options->include_metadata) {
        fprintf(fp, "metadata:\n");
        fprintf(fp, "  version: \"%s\"\n", export_get_version());
        
        if (options->include_timestamp) {
            time_t now = time(NULL);
            char time_str[64];
            strftime(time_str, sizeof(time_str), "%Y-%m-%d %H:%M:%S", 
                    localtime(&now));
            fprintf(fp, "  timestamp: %ld\n", (long)now);
            fprintf(fp, "  timestamp_str: \"%s\"\n", time_str);
        }
        
        fprintf(fp, "  total_detections: %d\n", count);
        fprintf(fp, "\n");
    }
    
    // 检测结果
    fprintf(fp, "detections:\n");
    
    for (int i = 0; i < count; i++) {
        const DetectionResult *result = &results[i];
        
        fprintf(fp, "  - id: %d\n", i);
        
        // 类型
        const char *type_str = (result->type == DETECTION_AI_GENERATED) ? 
                              "ai_generated" : "real";
        fprintf(fp, "    type: %s\n", type_str);
        
        // 置信度
        if (options->include_confidence) {
            fprintf(fp, "    confidence: %.*f\n", 
                   options->precision, result->confidence);
        }
        
        // 区域
        if (options->include_coordinates && result->has_region) {
            fprintf(fp, "    region:\n");
            fprintf(fp, "      x: %d\n", result->region.x);
            fprintf(fp, "      y: %d\n", result->region.y);
            fprintf(fp, "      width: %d\n", result->region.width);
            fprintf(fp, "      height: %d\n", result->region.height);
        }
        
        // 特征
        fprintf(fp, "    features:\n");
        fprintf(fp, "      noise_level: %.*f\n", 
               options->precision, result->features.noise_level);
        fprintf(fp, "      compression_artifacts: %.*f\n", 
               options->precision, result->features.compression_artifacts);
        fprintf(fp, "      edge_consistency: %.*f\n", 
               options->precision, result->features.edge_consistency);
        fprintf(fp, "      color_distribution: %.*f\n", 
               options->precision, result->features.color_distribution);
        fprintf(fp, "      texture_pattern: %.*f\n", 
               options->precision, result->features.texture_pattern);
        
        if (i < count - 1) {
            fprintf(fp, "\n");
        }
    }
    
    // 统计信息
    if (options->include_statistics) {
        fprintf(fp, "\n");
        
        ExportStatistics stats;
        if (export_calculate_statistics(results, count, &stats)) {
            fprintf(fp, "statistics:\n");
            fprintf(fp, "  total_detections: %d\n", stats.total_detections);
            fprintf(fp, "  average_confidence: %.*f\n", 
                   options->precision, stats.average_confidence);
        }
    }
    
    fclose(fp);
    
    if (g_verbose) {
        printf("成功导出YAML到: %s\n", output_file);
    }
    
    return true;
}

// ============================================================================
// HTML报告生成
// ============================================================================

bool export_html_report(const DetectionResult *results, int count,
                       const char *output_file,
                       const ReportConfig *config,
                       const ExportOptions *options)
{
    export_clear_error();
    
    if (!results || count <= 0 || !output_file) {
        export_set_error("无效的参数");
        return false;
    }
    
    if (!export_validate_path(output_file, true)) {
        return false;
    }
    
    ExportOptions default_opts = exporter_default_options();
    if (!options) {
        options = &default_opts;
    }
    
    ReportConfig default_config = exporter_default_report_config();
    if (!config) {
        config = &default_config;
    }
    
    FILE *fp = fopen(output_file, "w");
    if (!fp) {
        export_set_error("无法打开文件: %s", output_file);
        return false;
    }
    
    // HTML头部
    fprintf(fp, "<!DOCTYPE html>\n");
    fprintf(fp, "<html lang=\"en\">\n");
    fprintf(fp, "<head>\n");
    fprintf(fp, "  <meta charset=\"UTF-8\">\n");
    fprintf(fp, "  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n");
    fprintf(fp, "  <title>%s</title>\n", config->title);
    
    // CSS样式
    fprintf(fp, "  <style>\n");
    fprintf(fp, "    body {\n");
    fprintf(fp, "      font-family: Arial, sans-serif;\n");
    fprintf(fp, "      max-width: 1200px;\n");
    fprintf(fp, "      margin: 0 auto;\n");
    fprintf(fp, "      padding: 20px;\n");
    fprintf(fp, "      background-color: #f5f5f5;\n");
    fprintf(fp, "    }\n");
    fprintf(fp, "    .header {\n");
    fprintf(fp, "      background-color: #2c3e50;\n");
    fprintf(fp, "      color: white;\n");
    fprintf(fp, "      padding: 30px;\n");
    fprintf(fp, "      border-radius: 5px;\n");
    fprintf(fp, "      margin-bottom: 20px;\n");
    fprintf(fp, "    }\n");
    fprintf(fp, "    .header h1 {\n");
    fprintf(fp, "      margin: 0 0 10px 0;\n");
    fprintf(fp, "    }\n");
    fprintf(fp, "    .metadata {\n");
    fprintf(fp, "      background-color: white;\n");
    fprintf(fp, "      padding: 20px;\n");
    fprintf(fp, "      border-radius: 5px;\n");
    fprintf(fp, "      margin-bottom: 20px;\n");
    fprintf(fp, "      box-shadow: 0 2px 4px rgba(0,0,0,0.1);\n");
    fprintf(fp, "    }\n");
    fprintf(fp, "    .statistics {\n");
    fprintf(fp, "      background-color: white;\n");
    fprintf(fp, "      padding: 20px;\n");
    fprintf(fp, "      border-radius: 5px;\n");
    fprintf(fp, "      margin-bottom: 20px;\n");
    fprintf(fp, "      box-shadow: 0 2px 4px rgba(0,0,0,0.1);\n");
    fprintf(fp, "    }\n");
    fprintf(fp, "    .detection {\n");
    fprintf(fp, "      background-color: white;\n");
    fprintf(fp, "      padding: 20px;\n");
    fprintf(fp, "      border-radius: 5px;\n");
    fprintf(fp, "      margin-bottom: 15px;\n");
    fprintf(fp, "      box-shadow: 0 2px 4px rgba(0,0,0,0.1);\n");
    fprintf(fp, "    }\n");
    fprintf(fp, "    .detection-header {\n");
    fprintf(fp, "      display: flex;\n");
    fprintf(fp, "      justify-content: space-between;\n");
    fprintf(fp, "      align-items: center;\n");
    fprintf(fp, "      margin-bottom: 15px;\n");
    fprintf(fp, "      padding-bottom: 10px;\n");
    fprintf(fp, "      border-bottom: 2px solid #ecf0f1;\n");
    fprintf(fp, "    }\n");
    fprintf(fp, "    .badge {\n");
    fprintf(fp, "      padding: 5px 15px;\n");
    fprintf(fp, "      border-radius: 20px;\n");
    fprintf(fp, "      font-weight: bold;\n");
    fprintf(fp, "      font-size: 14px;\n");
    fprintf(fp, "    }\n");
    fprintf(fp, "    .badge-ai {\n");
    fprintf(fp, "      background-color: #e74c3c;\n");
    fprintf(fp, "      color: white;\n");
    fprintf(fp, "    }\n");
    fprintf(fp, "    .badge-real {\n");
    fprintf(fp, "      background-color: #27ae60;\n");
    fprintf(fp, "      color: white;\n");
    fprintf(fp, "    }\n");
    fprintf(fp, "    .confidence {\n");
    fprintf(fp, "      font-size: 18px;\n");
    fprintf(fp, "      font-weight: bold;\n");
    fprintf(fp, "      color: #2c3e50;\n");
    fprintf(fp, "    }\n");
    fprintf(fp, "    .features {\n");
    fprintf(fp, "      display: grid;\n");
    fprintf(fp, "      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));\n");
    fprintf(fp, "      gap: 15px;\n");
    fprintf(fp, "    }\n");
    fprintf(fp, "    .feature-item {\n");
    fprintf(fp, "      padding: 10px;\n");
    fprintf(fp, "      background-color: #ecf0f1;\n");
    fprintf(fp, "      border-radius: 5px;\n");
    fprintf(fp, "    }\n");
    fprintf(fp, "    .feature-label {\n");
    fprintf(fp, "      font-size: 12px;\n");
    fprintf(fp, "      color: #7f8c8d;\n");
    fprintf(fp, "      margin-bottom: 5px;\n");
    fprintf(fp, "    }\n");
    fprintf(fp, "    .feature-value {\n");
    fprintf(fp, "      font-size: 16px;\n");
    fprintf(fp, "      font-weight: bold;\n");
    fprintf(fp, "      color: #2c3e50;\n");
    fprintf(fp, "    }\n");
    fprintf(fp, "    .progress-bar {\n");
    fprintf(fp, "      width: 100%%;\n");
    fprintf(fp, "      height: 8px;\n");
    fprintf(fp, "      background-color: #ecf0f1;\n");
    fprintf(fp, "      border-radius: 4px;\n");
    fprintf(fp, "      overflow: hidden;\n");
    fprintf(fp, "      margin-top: 5px;\n");
    fprintf(fp, "    }\n");
    fprintf(fp, "    .progress-fill {\n");
    fprintf(fp, "      height: 100%%;\n");
    fprintf(fp, "      background-color: #3498db;\n");
    fprintf(fp, "      transition: width 0.3s ease;\n");
    fprintf(fp, "    }\n");
    fprintf(fp, "    table {\n");
    fprintf(fp, "      width: 100%%;\n");
    fprintf(fp, "      border-collapse: collapse;\n");
    fprintf(fp, "    }\n");
    fprintf(fp, "    th, td {\n");
    fprintf(fp, "      padding: 12px;\n");
    fprintf(fp, "      text-align: left;\n");
    fprintf(fp, "      border-bottom: 1px solid #ecf0f1;\n");
    fprintf(fp, "    }\n");
    fprintf(fp, "    th {\n");
    fprintf(fp, "      background-color: #34495e;\n");
    fprintf(fp, "      color: white;\n");
    fprintf(fp, "      font-weight: bold;\n");
    fprintf(fp, "    }\n");
    fprintf(fp, "    tr:hover {\n");
    fprintf(fp, "      background-color: #f8f9fa;\n");
    fprintf(fp, "    }\n");
    fprintf(fp, "  </style>\n");
    fprintf(fp, "</head>\n");
    fprintf(fp, "<body>\n");
    
    // 页面头部
    fprintf(fp, "  <div class=\"header\">\n");
    fprintf(fp, "    <h1>%s</h1>\n", config->title);
    if (config->description) {
        fprintf(fp, "    <p>%s</p>\n", config->description);
    }
    if (config->author) {
        fprintf(fp, "    <p><small>Author: %s</small></p>\n", config->author);
    }
    fprintf(fp, "  </div>\n\n");
    
    // 元数据
    if (options->include_metadata) {
        fprintf(fp, "  <div class=\"metadata\">\n");
        fprintf(fp, "    <h2>Metadata</h2>\n");
        fprintf(fp, "    <table>\n");
        fprintf(fp, "      <tr><td><strong>Version:</strong></td><td>%s</td></tr>\n", 
               export_get_version());
        
        if (options->include_timestamp) {
            time_t now = time(NULL);
            char time_str[64];
            strftime(time_str, sizeof(time_str), "%Y-%m-%d %H:%M:%S", 
                    localtime(&now));
            fprintf(fp, "      <tr><td><strong>Generated:</strong></td><td>%s</td></tr>\n", 
                   time_str);
        }
        
        fprintf(fp, "      <tr><td><strong>Total Detections:</strong></td><td>%d</td></tr>\n", 
               count);
        fprintf(fp, "    </table>\n");
        fprintf(fp, "  </div>\n\n");
    }
    
    // 统计信息
    if (config->include_summary && options->include_statistics) {
        ExportStatistics stats;
        if (export_calculate_statistics(results, count, &stats)) {
            fprintf(fp, "  <div class=\"statistics\">\n");
            fprintf(fp, "    <h2>Statistics Summary</h2>\n");
            fprintf(fp, "    <table>\n");
            fprintf(fp, "      <tr><td><strong>Total Detections:</strong></td><td>%d</td></tr>\n", 
                   stats.total_detections);
            fprintf(fp, "      <tr><td><strong>Average Confidence:</strong></td><td>%.2f%%</td></tr>\n", 
                   stats.average_confidence * 100.0);
            fprintf(fp, "    </table>\n");
            fprintf(fp, "  </div>\n\n");
        }
    }
    
    // 检测结果详情
    if (config->include_details) {
        fprintf(fp, "  <h2>Detection Results</h2>\n\n");
        
        for (int i = 0; i < count; i++) {
            const DetectionResult *result = &results[i];
            
            fprintf(fp, "  <div class=\"detection\">\n");
            fprintf(fp, "    <div class=\"detection-header\">\n");
            fprintf(fp, "      <div>\n");
            fprintf(fp, "        <h3>Detection #%d</h3>\n", i + 1);
            
            const char *badge_class = (result->type == DETECTION_AI_GENERATED) ? 
                                     "badge-ai" : "badge-real";
            const char *type_str = (result->type == DETECTION_AI_GENERATED) ? 
                                  "AI Generated" : "Real";
            fprintf(fp, "        <span class=\"badge %s\">%s</span>\n", 
                   badge_class, type_str);
            fprintf(fp, "      </div>\n");
            
            if (options->include_confidence) {
                fprintf(fp, "      <div class=\"confidence\">%.2f%%</div>\n", 
                       result->confidence * 100.0);
            }
            
            fprintf(fp, "    </div>\n");
            
            // 区域信息
            if (options->include_coordinates && result->has_region) {
                fprintf(fp, "    <p><strong>Region:</strong> ");
                fprintf(fp, "Position (%d, %d), Size %d x %d</p>\n",
                       result->region.x, result->region.y,
                       result->region.width, result->region.height);
            }
            
            // 特征
            fprintf(fp, "    <h4>Features</h4>\n");
            fprintf(fp, "    <div class=\"features\">\n");
            
            // 噪声水平
            fprintf(fp, "      <div class=\"feature-item\">\n");
            fprintf(fp, "        <div class=\"feature-label\">Noise Level</div>\n");
            fprintf(fp, "        <div class=\"feature-value\">%.3f</div>\n", 
                   result->features.noise_level);
            fprintf(fp, "        <div class=\"progress-bar\">\n");
            fprintf(fp, "          <div class=\"progress-fill\" style=\"width: %.1f%%\"></div>\n",
                   result->features.noise_level * 100.0);
            fprintf(fp, "        </div>\n");
            fprintf(fp, "      </div>\n");
            
            // 压缩伪影
            fprintf(fp, "      <div class=\"feature-item\">\n");
            fprintf(fp, "        <div class=\"feature-label\">Compression Artifacts</div>\n");
            fprintf(fp, "        <div class=\"feature-value\">%.3f</div>\n", 
                   result->features.compression_artifacts);
            fprintf(fp, "        <div class=\"progress-bar\">\n");
            fprintf(fp, "          <div class=\"progress-fill\" style=\"width: %.1f%%\"></div>\n",
                   result->features.compression_artifacts * 100.0);
            fprintf(fp, "        </div>\n");
            fprintf(fp, "      </div>\n");
            
            // 边缘一致性
            fprintf(fp, "      <div class=\"feature-item\">\n");
            fprintf(fp, "        <div class=\"feature-label\">Edge Consistency</div>\n");
            fprintf(fp, "        <div class=\"feature-value\">%.3f</div>\n", 
                   result->features.edge_consistency);
            fprintf(fp, "        <div class=\"progress-bar\">\n");
            fprintf(fp, "          <div class=\"progress-fill\" style=\"width: %.1f%%\"></div>\n",
                   result->features.edge_consistency * 100.0);
            fprintf(fp, "        </div>\n");
            fprintf(fp, "      </div>\n");
            
            // 颜色分布
            fprintf(fp, "      <div class=\"feature-item\">\n");
            fprintf(fp, "        <div class=\"feature-label\">Color Distribution</div>\n");
            fprintf(fp, "        <div class=\"feature-value\">%.3f</div>\n", 
                   result->features.color_distribution);
            fprintf(fp, "        <div class=\"progress-bar\">\n");
            fprintf(fp, "          <div class=\"progress-fill\" style=\"width: %.1f%%\"></div>\n",
                   result->features.color_distribution * 100.0);
            fprintf(fp, "        </div>\n");
            fprintf(fp, "      </div>\n");
            
            // 纹理模式
            fprintf(fp, "      <div class=\"feature-item\">\n");
            fprintf(fp, "        <div class=\"feature-label\">Texture Pattern</div>\n");
            fprintf(fp, "        <div class=\"feature-value\">%.3f</div>\n", 
                   result->features.texture_pattern);
            fprintf(fp, "        <div class=\"progress-bar\">\n");
            fprintf(fp, "          <div class=\"progress-fill\" style=\"width: %.1f%%\"></div>\n",
                   result->features.texture_pattern * 100.0);
            fprintf(fp, "        </div>\n");
            fprintf(fp, "      </div>\n");
            
            fprintf(fp, "    </div>\n");
            fprintf(fp, "  </div>\n\n");
        }
    }
    
    // HTML尾部
    fprintf(fp, "</body>\n");
    fprintf(fp, "</html>\n");
    
    fclose(fp);
    
    if (g_verbose) {
        printf("成功导出HTML报告到: %s\n", output_file);
    }
    
    return true;
}
/**
 * @file result_exporter.c
 * @brief 结果导出模块实现 - Part 1: 基础设施
 */

#include "result_exporter.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/stat.h>
#include <errno.h>

#ifdef _WIN32
#include <direct.h>
#define mkdir(path, mode) _mkdir(path)
#else
#include <sys/types.h>
#endif

// ============================================================================
// 全局状态
// ============================================================================

static char g_error_message[512] = {0};
static bool g_verbose = false;

// ============================================================================
// 错误处理
// ============================================================================

static void export_set_error(const char *format, ...)
{
    va_list args;
    va_start(args, format);
    vsnprintf(g_error_message, sizeof(g_error_message), format, args);
    va_end(args);
    
    if (g_verbose) {
        fprintf(stderr, "[Export Error] %s\n", g_error_message);
    }
}

const char* export_get_error(void)
{
    return g_error_message;
}

void export_clear_error(void)
{
    g_error_message[0] = '\0';
}

void export_set_verbose(bool verbose)
{
    g_verbose = verbose;
}

const char* export_get_version(void)
{
    return "1.0.0";
}

// ============================================================================
// 初始化和清理
// ============================================================================

bool exporter_init(void)
{
    export_clear_error();
    return true;
}

void exporter_cleanup(void)
{
    export_clear_error();
}

// ============================================================================
// 默认选项
// ============================================================================

ExportOptions exporter_default_options(void)
{
    ExportOptions options = {
        .include_metadata = true,
        .include_statistics = true,
        .include_confidence = true,
        .include_coordinates = true,
        .include_timestamp = true,
        .pretty_print = true,
        .use_relative_paths = false,
        .precision = 6,
        .encoding = "UTF-8"
    };
    return options;
}

ImageExportOptions exporter_default_image_options(void)
{
    ImageExportOptions options = {
        .draw_boxes = true,
        .draw_labels = true,
        .draw_confidence = true,
        .draw_grid = false,
        .box_thickness = 2,
        .font_size = 12,
        .box_color = {255, 0, 0, 255},
        .text_color = {255, 255, 255, 255},
        .bg_color = {0, 0, 0, 128},
        .opacity = 0.7f
    };
    return options;
}

ReportConfig exporter_default_report_config(void)
{
    ReportConfig config = {
        .title = "Detection Report",
        .author = "AI Detector",
        .description = "Automated detection results",
        .version = "1.0",
        .include_summary = true,
        .include_details = true,
        .include_images = true,
        .include_charts = false
    };
    return config;
}

// ============================================================================
// 路径和文件操作
// ============================================================================

static bool create_directory_recursive(const char *path)
{
    char tmp[512];
    char *p = NULL;
    size_t len;
    
    snprintf(tmp, sizeof(tmp), "%s", path);
    len = strlen(tmp);
    
    if (tmp[len - 1] == '/' || tmp[len - 1] == '\\') {
        tmp[len - 1] = 0;
    }
    
    for (p = tmp + 1; *p; p++) {
        if (*p == '/' || *p == '\\') {
            *p = 0;
            if (mkdir(tmp, 0755) != 0 && errno != EEXIST) {
                return false;
            }
            *p = '/';
        }
    }
    
    if (mkdir(tmp, 0755) != 0 && errno != EEXIST) {
        return false;
    }
    
    return true;
}

bool export_validate_path(const char *output_path, bool create_dirs)
{
    if (!output_path || strlen(output_path) == 0) {
        export_set_error("输出路径为空");
        return false;
    }
    
    char dir_path[512];
    const char *last_slash = strrchr(output_path, '/');
    const char *last_backslash = strrchr(output_path, '\\');
    const char *separator = (last_slash > last_backslash) ? last_slash : last_backslash;
    
    if (separator) {
        size_t dir_len = separator - output_path;
        if (dir_len >= sizeof(dir_path)) {
            export_set_error("路径太长");
            return false;
        }
        strncpy(dir_path, output_path, dir_len);
        dir_path[dir_len] = '\0';
        
        struct stat st;
        if (stat(dir_path, &st) != 0) {
            if (create_dirs) {
                if (!create_directory_recursive(dir_path)) {
                    export_set_error("无法创建目录: %s", dir_path);
                    return false;
                }
            } else {
                export_set_error("目录不存在: %s", dir_path);
                return false;
            }
        } else if (!S_ISDIR(st.st_mode)) {
            export_set_error("路径不是目录: %s", dir_path);
            return false;
        }
    }
    
    return true;
}

const char* export_get_extension(ExportFormat format)
{
    switch (format) {
        case EXPORT_FORMAT_JSON:     return "json";
        case EXPORT_FORMAT_XML:      return "xml";
        case EXPORT_FORMAT_CSV:      return "csv";
        case EXPORT_FORMAT_TXT:      return "txt";
        case EXPORT_FORMAT_HTML:     return "html";
        case EXPORT_FORMAT_MARKDOWN: return "md";
        case EXPORT_FORMAT_YAML:     return "yaml";
        case EXPORT_FORMAT_LATEX:    return "tex";
        default:                     return "txt";
    }
}

ExportFormat export_format_from_extension(const char *extension)
{
    if (!extension) return EXPORT_FORMAT_JSON;
    
    if (strcmp(extension, "json") == 0) return EXPORT_FORMAT_JSON;
    if (strcmp(extension, "xml") == 0) return EXPORT_FORMAT_XML;
    if (strcmp(extension, "csv") == 0) return EXPORT_FORMAT_CSV;
    if (strcmp(extension, "txt") == 0) return EXPORT_FORMAT_TXT;
    if (strcmp(extension, "html") == 0 || strcmp(extension, "htm") == 0) 
        return EXPORT_FORMAT_HTML;
    if (strcmp(extension, "md") == 0 || strcmp(extension, "markdown") == 0) 
        return EXPORT_FORMAT_MARKDOWN;
    if (strcmp(extension, "yaml") == 0 || strcmp(extension, "yml") == 0) 
        return EXPORT_FORMAT_YAML;
    if (strcmp(extension, "tex") == 0) return EXPORT_FORMAT_LATEX;
    
    return EXPORT_FORMAT_JSON;
}

// ============================================================================
// 辅助函数
// ============================================================================

static void write_indent(FILE *fp, int level, bool pretty)
{
    if (pretty) {
        for (int i = 0; i < level; i++) {
            fprintf(fp, "  ");
        }
    }
}

static void json_escape_string(const char *str, char *output, size_t output_size)
{
    size_t i = 0, j = 0;
    
    while (str[i] && j < output_size - 2) {
        switch (str[i]) {
            case '"':
                if (j < output_size - 3) {
                    output[j++] = '\\';
                    output[j++] = '"';
                }
                break;
            case '\\':
                if (j < output_size - 3) {
                    output[j++] = '\\';
                    output[j++] = '\\';
                }
                break;
            case '\n':
                if (j < output_size - 3) {
                    output[j++] = '\\';
                    output[j++] = 'n';
                }
                break;
            case '\r':
                if (j < output_size - 3) {
                    output[j++] = '\\';
                    output[j++] = 'r';
                }
                break;
            case '\t':
                if (j < output_size - 3) {
                    output[j++] = '\\';
                    output[j++] = 't';
                }
                break;
            default:
                output[j++] = str[i];
                break;
        }
        i++;
    }
    output[j] = '\0';
}

static void xml_escape_string(const char *str, char *output, size_t output_size)
{
    size_t i = 0, j = 0;
    
    while (str[i] && j < output_size - 6) {
        switch (str[i]) {
            case '<':
                strcpy(&output[j], "&lt;");
                j += 4;
                break;
            case '>':
                strcpy(&output[j], "&gt;");
                j += 4;
                break;
            case '&':
                strcpy(&output[j], "&amp;");
                j += 5;
                break;
            case '"':
                strcpy(&output[j], "&quot;");
                j += 6;
                break;
            case '\'':
                strcpy(&output[j], "&apos;");
                j += 6;
                break;
            default:
                output[j++] = str[i];
                break;
        }
        i++;
    }
    output[j] = '\0';
}
/**
 * @file result_exporter.c
 * @brief 结果导出模块实现 - Part 2: JSON导出
 */

// ============================================================================
// JSON导出实现
// ============================================================================

bool export_to_json(const DetectionResult *results, int count,
                   const char *output_file,
                   const ExportOptions *options)
{
    export_clear_error();
    
    if (!results || count <= 0 || !output_file) {
        export_set_error("无效的参数");
        return false;
    }
    
    if (!export_validate_path(output_file, true)) {
        return false;
    }
    
    ExportOptions default_opts = exporter_default_options();
    if (!options) {
        options = &default_opts;
    }
    
    FILE *fp = fopen(output_file, "w");
    if (!fp) {
        export_set_error("无法打开文件: %s", output_file);
        return false;
    }
    
    bool pretty = options->pretty_print;
    const char *nl = pretty ? "\n" : "";
    const char *sp = pretty ? " " : "";
    
    // 开始JSON对象
    fprintf(fp, "{%s", nl);
    
    // 元数据
    if (options->include_metadata) {
        write_indent(fp, 1, pretty);
        fprintf(fp, "\"metadata\":%s{%s", sp, nl);
        
        write_indent(fp, 2, pretty);
        fprintf(fp, "\"version\":%s\"%s\",%s", sp, export_get_version(), nl);
        
        if (options->include_timestamp) {
            time_t now = time(NULL);
            char time_str[64];
            strftime(time_str, sizeof(time_str), "%Y-%m-%d %H:%M:%S", 
                    localtime(&now));
            
            write_indent(fp, 2, pretty);
            fprintf(fp, "\"timestamp\":%s%ld,%s", sp, (long)now, nl);
            
            write_indent(fp, 2, pretty);
            fprintf(fp, "\"timestamp_str\":%s\"%s\",%s", sp, time_str, nl);
        }
        
        write_indent(fp, 2, pretty);
        fprintf(fp, "\"total_detections\":%s%d%s", sp, count, nl);
        
        write_indent(fp, 1, pretty);
        fprintf(fp, "},%s", nl);
    }
    
    // 检测结果数组
    write_indent(fp, 1, pretty);
    fprintf(fp, "\"detections\":%s[%s", sp, nl);
    
    for (int i = 0; i < count; i++) {
        const DetectionResult *result = &results[i];
        
        write_indent(fp, 2, pretty);
        fprintf(fp, "{%s", nl);
        
        // ID
        write_indent(fp, 3, pretty);
        fprintf(fp, "\"id\":%s%d,%s", sp, i, nl);
        
        // 类型
        write_indent(fp, 3, pretty);
        const char *type_str = (result->type == DETECTION_AI_GENERATED) ? 
                              "ai_generated" : "real";
        fprintf(fp, "\"type\":%s\"%s\",%s", sp, type_str, nl);
        
        // 置信度
        if (options->include_confidence) {
            write_indent(fp, 3, pretty);
            fprintf(fp, "\"confidence\":%s%.*f,%s", 
                   sp, options->precision, result->confidence, nl);
        }
        
        // 区域
        if (options->include_coordinates && result->has_region) {
            write_indent(fp, 3, pretty);
            fprintf(fp, "\"region\":%s{%s", sp, nl);
            
            write_indent(fp, 4, pretty);
            fprintf(fp, "\"x\":%s%d,%s", sp, result->region.x, nl);
            
            write_indent(fp, 4, pretty);
            fprintf(fp, "\"y\":%s%d,%s", sp, result->region.y, nl);
            
            write_indent(fp, 4, pretty);
            fprintf(fp, "\"width\":%s%d,%s", sp, result->region.width, nl);
            
            write_indent(fp, 4, pretty);
            fprintf(fp, "\"height\":%s%d%s", sp, result->region.height, nl);
            
            write_indent(fp, 3, pretty);
            fprintf(fp, "},%s", nl);
        }
        
        // 特征
        write_indent(fp, 3, pretty);
        fprintf(fp, "\"features\":%s{%s", sp, nl);
        
        write_indent(fp, 4, pretty);
        fprintf(fp, "\"noise_level\":%s%.*f,%s", 
               sp, options->precision, result->features.noise_level, nl);
        
        write_indent(fp, 4, pretty);
        fprintf(fp, "\"compression_artifacts\":%s%.*f,%s", 
               sp, options->precision, result->features.compression_artifacts, nl);
        
        write_indent(fp, 4, pretty);
        fprintf(fp, "\"edge_consistency\":%s%.*f,%s", 
               sp, options->precision, result->features.edge_consistency, nl);
        
        write_indent(fp, 4, pretty);
        fprintf(fp, "\"color_distribution\":%s%.*f,%s", 
               sp, options->precision, result->features.color_distribution, nl);
        
        write_indent(fp, 4, pretty);
        fprintf(fp, "\"texture_pattern\":%s%.*f%s", 
               sp, options->precision, result->features.texture_pattern, nl);
        
        write_indent(fp, 3, pretty);
        fprintf(fp, "}%s", nl);
        
        write_indent(fp, 2, pretty);
        fprintf(fp, "}");
        
        if (i < count - 1) {
            fprintf(fp, ",");
        }
        fprintf(fp, "%s", nl);
    }
    
    write_indent(fp, 1, pretty);
    fprintf(fp, "]");
    
    // 统计信息
    if (options->include_statistics) {
        fprintf(fp, ",%s", nl);
        
        ExportStatistics stats;
        if (export_calculate_statistics(results, count, &stats)) {
            write_indent(fp, 1, pretty);
            fprintf(fp, "\"statistics\":%s{%s", sp, nl);
            
            write_indent(fp, 2, pretty);
            fprintf(fp, "\"total_detections\":%s%d,%s", 
                   sp, stats.total_detections, nl);
            
            write_indent(fp, 2, pretty);
            fprintf(fp, "\"average_confidence\":%s%.*f%s", 
                   sp, options->precision, stats.average_confidence, nl);
            
            write_indent(fp, 1, pretty);
            fprintf(fp, "}%s", nl);
        }
    } else {
        fprintf(fp, "%s", nl);
    }
    
    // 结束JSON对象
    fprintf(fp, "}%s", nl);
    
    fclose(fp);
    
    if (g_verbose) {
        printf("成功导出JSON到: %s\n", output_file);
    }
    
    return true;
}

// ============================================================================
// JSON批量导出
// ============================================================================

bool export_batch_to_json(const DetectionResult **results_array,
                          const int *counts,
                          const char **source_files,
                          int batch_count,
                          const char *output_file,
                          const ExportOptions *options)
{
    export_clear_error();
    
    if (!results_array || !counts || batch_count <= 0 || !output_file) {
        export_set_error("无效的参数");
        return false;
    }
    
    if (!export_validate_path(output_file, true)) {
        return false;
    }
    
    ExportOptions default_opts = exporter_default_options();
    if (!options) {
        options = &default_opts;
    }
    
    FILE *fp = fopen(output_file, "w");
    if (!fp) {
        export_set_error("无法打开文件: %s", output_file);
        return false;
    }
    
    bool pretty = options->pretty_print;
    const char *nl = pretty ? "\n" : "";
    const char *sp = pretty ? " " : "";
    
    // 开始JSON对象
    fprintf(fp, "{%s", nl);
    
    // 元数据
    if (options->include_metadata) {
        write_indent(fp, 1, pretty);
        fprintf(fp, "\"metadata\":%s{%s", sp, nl);
        
        write_indent(fp, 2, pretty);
        fprintf(fp, "\"version\":%s\"%s\",%s", sp, export_get_version(), nl);
        
        if (options->include_timestamp) {
            time_t now = time(NULL);
            char time_str[64];
            strftime(time_str, sizeof(time_str), "%Y-%m-%d %H:%M:%S", 
                    localtime(&now));
            
            write_indent(fp, 2, pretty);
            fprintf(fp, "\"timestamp\":%s%ld,%s", sp, (long)now, nl);
            
            write_indent(fp, 2, pretty);
            fprintf(fp, "\"timestamp_str\":%s\"%s\",%s", sp, time_str, nl);
        }
        
        write_indent(fp, 2, pretty);
        fprintf(fp, "\"batch_count\":%s%d%s", sp, batch_count, nl);
        
        write_indent(fp, 1, pretty);
        fprintf(fp, "},%s", nl);
    }
    
    // 批次数组
    write_indent(fp, 1, pretty);
    fprintf(fp, "\"batches\":%s[%s", sp, nl);
    
    for (int b = 0; b < batch_count; b++) {
        const DetectionResult *results = results_array[b];
        int count = counts[b];
        
        write_indent(fp, 2, pretty);
        fprintf(fp, "{%s", nl);
        
        // 源文件
        if (source_files && source_files[b]) {
            write_indent(fp, 3, pretty);
            char escaped[512];
            json_escape_string(source_files[b], escaped, sizeof(escaped));
            fprintf(fp, "\"source_file\":%s\"%s\",%s", sp, escaped, nl);
        }
        
        // 检测数量
        write_indent(fp, 3, pretty);
        fprintf(fp, "\"detection_count\":%s%d,%s", sp, count, nl);
        
        // 检测结果
        write_indent(fp, 3, pretty);
        fprintf(fp, "\"detections\":%s[%s", sp, nl);
        
        for (int i = 0; i < count; i++) {
            const DetectionResult *result = &results[i];
            
            write_indent(fp, 4, pretty);
            fprintf(fp, "{%s", nl);
            
            write_indent(fp, 5, pretty);
            fprintf(fp, "\"id\":%s%d,%s", sp, i, nl);
            
            write_indent(fp, 5, pretty);
            const char *type_str = (result->type == DETECTION_AI_GENERATED) ? 
                                  "ai_generated" : "real";
            fprintf(fp, "\"type\":%s\"%s\",%s", sp, type_str, nl);
            
            if (options->include_confidence) {
                write_indent(fp, 5, pretty);
                fprintf(fp, "\"confidence\":%s%.*f,%s", 
                       sp, options->precision, result->confidence, nl);
            }
            
            // 特征（简化版）
            write_indent(fp, 5, pretty);
            fprintf(fp, "\"features\":%s{%s", sp, nl);
            
            write_indent(fp, 6, pretty);
            fprintf(fp, "\"noise_level\":%s%.*f,%s", 
                   sp, options->precision, result->features.noise_level, nl);
            
            write_indent(fp, 6, pretty);
            fprintf(fp, "\"compression_artifacts\":%s%.*f%s", 
                   sp, options->precision, result->features.compression_artifacts, nl);
            
            write_indent(fp, 5, pretty);
            fprintf(fp, "}%s", nl);
            
            write_indent(fp, 4, pretty);
            fprintf(fp, "}");
            
            if (i < count - 1) {
                fprintf(fp, ",");
            }
            fprintf(fp, "%s", nl);
        }
        
        write_indent(fp, 3, pretty);
        fprintf(fp, "]%s", nl);
        
        write_indent(fp, 2, pretty);
        fprintf(fp, "}");
        
        if (b < batch_count - 1) {
            fprintf(fp, ",");
        }
        fprintf(fp, "%s", nl);
    }
    
    write_indent(fp, 1, pretty);
    fprintf(fp, "]%s", nl);
    
    fprintf(fp, "}%s", nl);
    
    fclose(fp);
    
    if (g_verbose) {
        printf("成功导出批量JSON到: %s\n", output_file);
    }
    
    return true;
}

// ============================================================================
// JSON流式导出（用于大数据集）
// ============================================================================

typedef struct {
    FILE *fp;
    bool first_item;
    bool pretty_print;
    int indent_level;
} JsonStreamContext;

JsonStreamContext* export_json_stream_begin(const char *output_file,
                                           const ExportOptions *options)
{
    export_clear_error();
    
    if (!output_file) {
        export_set_error("无效的输出文件");
        return NULL;
    }
    
    if (!export_validate_path(output_file, true)) {
        return NULL;
    }
    
    ExportOptions default_opts = exporter_default_options();
    if (!options) {
        options = &default_opts;
    }
    
    JsonStreamContext *ctx = (JsonStreamContext*)malloc(sizeof(JsonStreamContext));
    if (!ctx) {
        export_set_error("内存分配失败");
        return NULL;
    }
    
    ctx->fp = fopen(output_file, "w");
    if (!ctx->fp) {
        export_set_error("无法打开文件: %s", output_file);
        free(ctx);
        return NULL;
    }
    
    ctx->first_item = true;
    ctx->pretty_print = options->pretty_print;
    ctx->indent_level = 1;
    
    const char *nl = ctx->pretty_print ? "\n" : "";
    const char *sp = ctx->pretty_print ? " " : "";
    
    // 开始JSON对象和数组
    fprintf(ctx->fp, "{%s", nl);
    write_indent(ctx->fp, 1, ctx->pretty_print);
    fprintf(ctx->fp, "\"detections\":%s[%s", sp, nl);
    
    return ctx;
}

bool export_json_stream_write(JsonStreamContext *ctx,
                              const DetectionResult *result,
                              const ExportOptions *options)
{
    if (!ctx || !ctx->fp || !result) {
        export_set_error("无效的上下文或结果");
        return false;
    }
    
    ExportOptions default_opts = exporter_default_options();
    if (!options) {
        options = &default_opts;
    }
    
    const char *nl = ctx->pretty_print ? "\n" : "";
    const char *sp = ctx->pretty_print ? " " : "";
    
    // 如果不是第一项，添加逗号
    if (!ctx->first_item) {
        fprintf(ctx->fp, ",%s", nl);
    }
    ctx->first_item = false;
    
    write_indent(ctx->fp, 2, ctx->pretty_print);
    fprintf(ctx->fp, "{%s", nl);
    
    write_indent(ctx->fp, 3, ctx->pretty_print);
    const char *type_str = (result->type == DETECTION_AI_GENERATED) ? 
                          "ai_generated" : "real";
    fprintf(ctx->fp, "\"type\":%s\"%s\",%s", sp, type_str, nl);
    
    if (options->include_confidence) {
        write_indent(ctx->fp, 3, ctx->pretty_print);
        fprintf(ctx->fp, "\"confidence\":%s%.*f,%s", 
               sp, options->precision, result->confidence, nl);
    }
    
    write_indent(ctx->fp, 3, ctx->pretty_print);
    fprintf(ctx->fp, "\"features\":%s{%s", sp, nl);
    
    write_indent(ctx->fp, 4, ctx->pretty_print);
    fprintf(ctx->fp, "\"noise_level\":%s%.*f,%s", 
           sp, options->precision, result->features.noise_level, nl);
    
    write_indent(ctx->fp, 4, ctx->pretty_print);
    fprintf(ctx->fp, "\"compression_artifacts\":%s%.*f%s", 
           sp, options->precision, result->features.compression_artifacts, nl);
    
    write_indent(ctx->fp, 3, ctx->pretty_print);
    fprintf(ctx->fp, "}%s", nl);
    
    write_indent(ctx->fp, 2, ctx->pretty_print);
    fprintf(ctx->fp, "}");
    
    return true;
}

bool export_json_stream_end(JsonStreamContext *ctx)
{
    if (!ctx || !ctx->fp) {
        export_set_error("无效的上下文");
        return false;
    }
    
    const char *nl = ctx->pretty_print ? "\n" : "";
    
    // 结束数组和对象
    fprintf(ctx->fp, "%s", nl);
    write_indent(ctx->fp, 1, ctx->pretty_print);
    fprintf(ctx->fp, "]%s", nl);
    fprintf(ctx->fp, "}%s", nl);
    
    fclose(ctx->fp);
    free(ctx);
    
    if (g_verbose) {
        printf("JSON流式导出完成\n");
    }
    
    return true;
}
/**
 * @file result_exporter.c
 * @brief 结果导出模块实现 - Part 3: XML导出
 */

// ============================================================================
// XML导出实现
// ============================================================================

bool export_to_xml(const DetectionResult *results, int count,
                  const char *output_file,
                  const ExportOptions *options)
{
    export_clear_error();
    
    if (!results || count <= 0 || !output_file) {
        export_set_error("无效的参数");
        return false;
    }
    
    if (!export_validate_path(output_file, true)) {
        return false;
    }
    
    ExportOptions default_opts = exporter_default_options();
    if (!options) {
        options = &default_opts;
    }
    
    FILE *fp = fopen(output_file, "w");
    if (!fp) {
        export_set_error("无法打开文件: %s", output_file);
        return false;
    }
    
    bool pretty = options->pretty_print;
    const char *nl = pretty ? "\n" : "";
    
    // XML声明
    fprintf(fp, "<?xml version=\"1.0\" encoding=\"%s\"?>%s", 
           options->encoding, nl);
    
    // 根元素
    fprintf(fp, "<detection_results>%s", nl);
    
    // 元数据
    if (options->include_metadata) {
        write_indent(fp, 1, pretty);
        fprintf(fp, "<metadata>%s", nl);
        
        write_indent(fp, 2, pretty);
        fprintf(fp, "<version>%s</version>%s", export_get_version(), nl);
        
        if (options->include_timestamp) {
            time_t now = time(NULL);
            char time_str[64];
            strftime(time_str, sizeof(time_str), "%Y-%m-%d %H:%M:%S", 
                    localtime(&now));
            
            write_indent(fp, 2, pretty);
            fprintf(fp, "<timestamp>%ld</timestamp>%s", (long)now, nl);
            
            write_indent(fp, 2, pretty);
            fprintf(fp, "<timestamp_str>%s</timestamp_str>%s", time_str, nl);
        }
        
        write_indent(fp, 2, pretty);
        fprintf(fp, "<total_detections>%d</total_detections>%s", count, nl);
        
        write_indent(fp, 1, pretty);
        fprintf(fp, "</metadata>%s", nl);
    }
    
    // 检测结果
    write_indent(fp, 1, pretty);
    fprintf(fp, "<detections>%s", nl);
    
    for (int i = 0; i < count; i++) {
        const DetectionResult *result = &results[i];
        
        write_indent(fp, 2, pretty);
        fprintf(fp, "<detection id=\"%d\">%s", i, nl);
        
        // 类型
        write_indent(fp, 3, pretty);
        const char *type_str = (result->type == DETECTION_AI_GENERATED) ? 
                              "ai_generated" : "real";
        fprintf(fp, "<type>%s</type>%s", type_str, nl);
        
        // 置信度
        if (options->include_confidence) {
            write_indent(fp, 3, pretty);
            fprintf(fp, "<confidence>%.*f</confidence>%s", 
                   options->precision, result->confidence, nl);
        }
        
        // 区域
        if (options->include_coordinates && result->has_region) {
            write_indent(fp, 3, pretty);
            fprintf(fp, "<region>%s", nl);
            
            write_indent(fp, 4, pretty);
            fprintf(fp, "<x>%d</x>%s", result->region.x, nl);
            
            write_indent(fp, 4, pretty);
            fprintf(fp, "<y>%d</y>%s", result->region.y, nl);
            
            write_indent(fp, 4, pretty);
            fprintf(fp, "<width>%d</width>%s", result->region.width, nl);
            
            write_indent(fp, 4, pretty);
            fprintf(fp, "<height>%d</height>%s", result->region.height, nl);
            
            write_indent(fp, 3, pretty);
            fprintf(fp, "</region>%s", nl);
        }
        
        // 特征
        write_indent(fp, 3, pretty);
        fprintf(fp, "<features>%s", nl);
        
        write_indent(fp, 4, pretty);
        fprintf(fp, "<noise_level>%.*f</noise_level>%s", 
               options->precision, result->features.noise_level, nl);
        
        write_indent(fp, 4, pretty);
        fprintf(fp, "<compression_artifacts>%.*f</compression_artifacts>%s", 
               options->precision, result->features.compression_artifacts, nl);
        
        write_indent(fp, 4, pretty);
        fprintf(fp, "<edge_consistency>%.*f</edge_consistency>%s", 
               options->precision, result->features.edge_consistency, nl);
        
        write_indent(fp, 4, pretty);
        fprintf(fp, "<color_distribution>%.*f</color_distribution>%s", 
               options->precision, result->features.color_distribution, nl);
        
        write_indent(fp, 4, pretty);
        fprintf(fp, "<texture_pattern>%.*f</texture_pattern>%s", 
               options->precision, result->features.texture_pattern, nl);
        
        write_indent(fp, 3, pretty);
        fprintf(fp, "</features>%s", nl);
        
        write_indent(fp, 2, pretty);
        fprintf(fp, "</detection>%s", nl);
    }
    
    write_indent(fp, 1, pretty);
    fprintf(fp, "</detections>%s", nl);
    
    // 统计信息
    if (options->include_statistics) {
        ExportStatistics stats;
        if (export_calculate_statistics(results, count, &stats)) {
            write_indent(fp, 1, pretty);
            fprintf(fp, "<statistics>%s", nl);
            
            write_indent(fp, 2, pretty);
            fprintf(fp, "<total_detections>%d</total_detections>%s", 
                   stats.total_detections, nl);
            
            write_indent(fp, 2, pretty);
            fprintf(fp, "<average_confidence>%.*f</average_confidence>%s", 
                   options->precision, stats.average_confidence, nl);
            
            write_indent(fp, 1, pretty);
            fprintf(fp, "</statistics>%s", nl);
        }
    }
    
    // 关闭根元素
    fprintf(fp, "</detection_results>%s", nl);
    
    fclose(fp);
    
    if (g_verbose) {
        printf("成功导出XML到: %s\n", output_file);
    }
    
    return true;
}

// ============================================================================
// CSV导出实现
// ============================================================================

bool export_to_csv(const DetectionResult *results, int count,
                  const char *output_file,
                  const ExportOptions *options)
{
    export_clear_error();
    
    if (!results || count <= 0 || !output_file) {
        export_set_error("无效的参数");
        return false;
    }
    
    if (!export_validate_path(output_file, true)) {
        return false;
    }
    
    ExportOptions default_opts = exporter_default_options();
    if (!options) {
        options = &default_opts;
    }
    
    FILE *fp = fopen(output_file, "w");
    if (!fp) {
        export_set_error("无法打开文件: %s", output_file);
        return false;
    }
    
    // CSV头部
    fprintf(fp, "ID,Type,Confidence");
    
    if (options->include_coordinates) {
        fprintf(fp, ",Region_X,Region_Y,Region_Width,Region_Height");
    }
    
    fprintf(fp, ",Noise_Level,Compression_Artifacts,Edge_Consistency,");
    fprintf(fp, "Color_Distribution,Texture_Pattern\n");
    
    // 数据行
    for (int i = 0; i < count; i++) {
        const DetectionResult *result = &results[i];
        
        // ID
        fprintf(fp, "%d,", i);
        
        // 类型
        const char *type_str = (result->type == DETECTION_AI_GENERATED) ? 
                              "AI_Generated" : "Real";
        fprintf(fp, "%s,", type_str);
        
        // 置信度
        if (options->include_confidence) {
            fprintf(fp, "%.*f", options->precision, result->confidence);
        }
        fprintf(fp, ",");
        
        // 区域
        if (options->include_coordinates) {
            if (result->has_region) {
                fprintf(fp, "%d,%d,%d,%d,", 
                       result->region.x, result->region.y,
                       result->region.width, result->region.height);
            } else {
                fprintf(fp, ",,,,");
            }
        }
        
        // 特征
        fprintf(fp, "%.*f,", options->precision, result->features.noise_level);
        fprintf(fp, "%.*f,", options->precision, result->features.compression_artifacts);
        fprintf(fp, "%.*f,", options->precision, result->features.edge_consistency);
        fprintf(fp, "%.*f,", options->precision, result->features.color_distribution);
        fprintf(fp, "%.*f", options->precision, result->features.texture_pattern);
        
        fprintf(fp, "\n");
    }
    
    fclose(fp);
    
    if (g_verbose) {
        printf("成功导出CSV到: %s\n", output_file);
    }
    
    return true;
}

// ============================================================================
// TXT导出实现
// ============================================================================

bool export_to_txt(const DetectionResult *results, int count,
                  const char *output_file,
                  const ExportOptions *options)
{
    export_clear_error();
    
    if (!results || count <= 0 || !output_file) {
        export_set_error("无效的参数");
        return false;
    }
    
    if (!export_validate_path(output_file, true)) {
        return false;
    }
    
    ExportOptions default_opts = exporter_default_options();
    if (!options) {
        options = &default_opts;
    }
    
    FILE *fp = fopen(output_file, "w");
    if (!fp) {
        export_set_error("无法打开文件: %s", output_file);
        return false;
    }
    
    // 标题
    fprintf(fp, "AI Detection Results\n");
    fprintf(fp, "====================\n\n");
    
    // 元数据
    if (options->include_metadata) {
        fprintf(fp, "Metadata:\n");
        fprintf(fp, "---------\n");
        fprintf(fp, "Version: %s\n", export_get_version());
        
        if (options->include_timestamp) {
            time_t now = time(NULL);
            char time_str[64];
            strftime(time_str, sizeof(time_str), "%Y-%m-%d %H:%M:%S", 
                    localtime(&now));
            fprintf(fp, "Generated: %s\n", time_str);
        }
        
        fprintf(fp, "Total Detections: %d\n\n", count);
    }
    
    // 检测结果
    for (int i = 0; i < count; i++) {
        const DetectionResult *result = &results[i];
        
        fprintf(fp, "Detection #%d\n", i + 1);
        fprintf(fp, "------------\n");
        
        // 类型
        const char *type_str = (result->type == DETECTION_AI_GENERATED) ? 
                              "AI Generated" : "Real";
        fprintf(fp, "Type: %s\n", type_str);
        
        // 置信度
        if (options->include_confidence) {
            fprintf(fp, "Confidence: %.*f%%\n", 
                   options->precision, result->confidence * 100.0);
        }
        
        // 区域
        if (options->include_coordinates && result->has_region) {
            fprintf(fp, "Region: (%d, %d) - %dx%d\n", 
                   result->region.x, result->region.y,
                   result->region.width, result->region.height);
        }
        
        // 特征
        fprintf(fp, "\nFeatures:\n");
        fprintf(fp, "  Noise Level:           %.*f\n", 
               options->precision, result->features.noise_level);
        fprintf(fp, "  Compression Artifacts: %.*f\n", 
               options->precision, result->features.compression_artifacts);
        fprintf(fp, "  Edge Consistency:      %.*f\n", 
               options->precision, result->features.edge_consistency);
        fprintf(fp, "  Color Distribution:    %.*f\n", 
               options->precision, result->features.color_distribution);
        fprintf(fp, "  Texture Pattern:       %.*f\n", 
               options->precision, result->features.texture_pattern);
        
        fprintf(fp, "\n");
    }
    
    // 统计信息
    if (options->include_statistics) {
        ExportStatistics stats;
        if (export_calculate_statistics(results, count, &stats)) {
            fprintf(fp, "Statistics Summary\n");
            fprintf(fp, "------------------\n");
            fprintf(fp, "Total Detections: %d\n", stats.total_detections);
            fprintf(fp, "Average Confidence: %.*f%%\n", 
                   options->precision, stats.average_confidence * 100.0);
            fprintf(fp, "\n");
        }
    }
    
    fclose(fp);
    
    if (g_verbose) {
        printf("成功导出TXT到: %s\n", output_file);
    }
    
    return true;
}

// ============================================================================
// YAML导出实现
// ============================================================================

bool export_to_yaml(const DetectionResult *results, int count,
                   const char *output_file,
                   const ExportOptions *options)
{
    export_clear_error();
    
    if (!results || count <= 0 || !output_file) {
        export_set_error("无效的参数");
        return false;
    }
    
    if (!export_validate_path(output_file, true)) {
        return false;
    }
    
    ExportOptions default_opts = exporter_default_options();
    if (!options) {
        options = &default_opts;
    }
    
    FILE *fp = fopen(output_file, "w");
    if (!fp) {
        export_set_error("无法打开文件: %s", output_file);
        return false;
    }
    
    // YAML文档开始
    fprintf(fp, "---\n");
    
    // 元数据
    if (options->include_metadata) {
        fprintf(fp, "metadata:\n");
        fprintf(fp, "  version: \"%s\"\n", export_get_version());
        
        if (options->include_timestamp) {
            time_t now = time(NULL);
            char time_str[64];
            strftime(time_str, sizeof(time_str), "%Y-%m-%d %H:%M:%S", 
                    localtime(&now));
            fprintf(fp, "  timestamp: %ld\n", (long)now);
            fprintf(fp, "  timestamp_str: \"%s\"\n", time_str);
        }
        
        fprintf(fp, "  total_detections: %d\n", count);
        fprintf(fp, "\n");
    }
    
    // 检测结果
    fprintf(fp, "detections:\n");
    
    for (int i = 0; i < count; i++) {
        const DetectionResult *result = &results[i];
        
        fprintf(fp, "  - id: %d\n", i);
        
        // 类型
        const char *type_str = (result->type == DETECTION_AI_GENERATED) ? 
                              "ai_generated" : "real";
        fprintf(fp, "    type: %s\n", type_str);
        
        // 置信度
        if (options->include_confidence) {
            fprintf(fp, "    confidence: %.*f\n", 
                   options->precision, result->confidence);
        }
        
        // 区域
        if (options->include_coordinates && result->has_region) {
            fprintf(fp, "    region:\n");
            fprintf(fp, "      x: %d\n", result->region.x);
            fprintf(fp, "      y: %d\n", result->region.y);
            fprintf(fp, "      width: %d\n", result->region.width);
            fprintf(fp, "      height: %d\n", result->region.height);
        }
        
        // 特征
        fprintf(fp, "    features:\n");
        fprintf(fp, "      noise_level: %.*f\n", 
               options->precision, result->features.noise_level);
        fprintf(fp, "      compression_artifacts: %.*f\n", 
               options->precision, result->features.compression_artifacts);
        fprintf(fp, "      edge_consistency: %.*f\n", 
               options->precision, result->features.edge_consistency);
        fprintf(fp, "      color_distribution: %.*f\n", 
               options->precision, result->features.color_distribution);
        fprintf(fp, "      texture_pattern: %.*f\n", 
               options->precision, result->features.texture_pattern);
        
        if (i < count - 1) {
            fprintf(fp, "\n");
        }
    }
    
    // 统计信息
    if (options->include_statistics) {
        ExportStatistics stats;
        if (export_calculate_statistics(results, count, &stats)) {
            fprintf(fp, "\nstatistics:\n");
            fprintf(fp, "  total_detections: %d\n", stats.total_detections);
            fprintf(fp, "  average_confidence: %.*f\n", 
                   options->precision, stats.average_confidence);
        }
    }
    
    fclose(fp);
    
    if (g_verbose) {
        printf("成功导出YAML到: %s\n", output_file);
    }
    
    return true;
}
/**
 * @file result_exporter.c
 * @brief 结果导出模块实现 - Part 4: HTML导出
 */

// ============================================================================
// HTML导出实现
// ============================================================================

bool export_to_html(const DetectionResult *results, int count,
                   const char *output_file,
                   const ExportOptions *options)
{
    export_clear_error();
    
    if (!results || count <= 0 || !output_file) {
        export_set_error("无效的参数");
        return false;
    }
    
    if (!export_validate_path(output_file, true)) {
        return false;
    }
    
    ExportOptions default_opts = exporter_default_options();
    if (!options) {
        options = &default_opts;
    }
    
    FILE *fp = fopen(output_file, "w");
    if (!fp) {
        export_set_error("无法打开文件: %s", output_file);
        return false;
    }
    
    // HTML头部
    fprintf(fp, "<!DOCTYPE html>\n");
    fprintf(fp, "<html lang=\"zh-CN\">\n");
    fprintf(fp, "<head>\n");
    fprintf(fp, "    <meta charset=\"%s\">\n", options->encoding);
    fprintf(fp, "    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n");
    fprintf(fp, "    <title>AI Detection Results</title>\n");
    
    // CSS样式
    fprintf(fp, "    <style>\n");
    fprintf(fp, "        * { margin: 0; padding: 0; box-sizing: border-box; }\n");
    fprintf(fp, "        body {\n");
    fprintf(fp, "            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;\n");
    fprintf(fp, "            background: linear-gradient(135deg, #667eea 0%%, #764ba2 100%%);\n");
    fprintf(fp, "            padding: 20px;\n");
    fprintf(fp, "            min-height: 100vh;\n");
    fprintf(fp, "        }\n");
    fprintf(fp, "        .container {\n");
    fprintf(fp, "            max-width: 1200px;\n");
    fprintf(fp, "            margin: 0 auto;\n");
    fprintf(fp, "            background: white;\n");
    fprintf(fp, "            border-radius: 10px;\n");
    fprintf(fp, "            box-shadow: 0 10px 40px rgba(0,0,0,0.2);\n");
    fprintf(fp, "            overflow: hidden;\n");
    fprintf(fp, "        }\n");
    fprintf(fp, "        .header {\n");
    fprintf(fp, "            background: linear-gradient(135deg, #667eea 0%%, #764ba2 100%%);\n");
    fprintf(fp, "            color: white;\n");
    fprintf(fp, "            padding: 30px;\n");
    fprintf(fp, "            text-align: center;\n");
    fprintf(fp, "        }\n");
    fprintf(fp, "        .header h1 {\n");
    fprintf(fp, "            font-size: 2.5em;\n");
    fprintf(fp, "            margin-bottom: 10px;\n");
    fprintf(fp, "        }\n");
    fprintf(fp, "        .metadata {\n");
    fprintf(fp, "            background: #f8f9fa;\n");
    fprintf(fp, "            padding: 20px 30px;\n");
    fprintf(fp, "            border-bottom: 2px solid #e9ecef;\n");
    fprintf(fp, "        }\n");
    fprintf(fp, "        .metadata-item {\n");
    fprintf(fp, "            display: inline-block;\n");
    fprintf(fp, "            margin-right: 30px;\n");
    fprintf(fp, "            color: #495057;\n");
    fprintf(fp, "        }\n");
    fprintf(fp, "        .metadata-label {\n");
    fprintf(fp, "            font-weight: bold;\n");
    fprintf(fp, "            color: #212529;\n");
    fprintf(fp, "        }\n");
    fprintf(fp, "        .content {\n");
    fprintf(fp, "            padding: 30px;\n");
    fprintf(fp, "        }\n");
    fprintf(fp, "        .detection-card {\n");
    fprintf(fp, "            background: white;\n");
    fprintf(fp, "            border: 1px solid #dee2e6;\n");
    fprintf(fp, "            border-radius: 8px;\n");
    fprintf(fp, "            padding: 20px;\n");
    fprintf(fp, "            margin-bottom: 20px;\n");
    fprintf(fp, "            transition: transform 0.2s, box-shadow 0.2s;\n");
    fprintf(fp, "        }\n");
    fprintf(fp, "        .detection-card:hover {\n");
    fprintf(fp, "            transform: translateY(-2px);\n");
    fprintf(fp, "            box-shadow: 0 4px 12px rgba(0,0,0,0.1);\n");
    fprintf(fp, "        }\n");
    fprintf(fp, "        .detection-header {\n");
    fprintf(fp, "            display: flex;\n");
    fprintf(fp, "            justify-content: space-between;\n");
    fprintf(fp, "            align-items: center;\n");
    fprintf(fp, "            margin-bottom: 15px;\n");
    fprintf(fp, "            padding-bottom: 15px;\n");
    fprintf(fp, "            border-bottom: 2px solid #e9ecef;\n");
    fprintf(fp, "        }\n");
    fprintf(fp, "        .detection-id {\n");
    fprintf(fp, "            font-size: 1.2em;\n");
    fprintf(fp, "            font-weight: bold;\n");
    fprintf(fp, "            color: #495057;\n");
    fprintf(fp, "        }\n");
    fprintf(fp, "        .badge {\n");
    fprintf(fp, "            display: inline-block;\n");
    fprintf(fp, "            padding: 5px 15px;\n");
    fprintf(fp, "            border-radius: 20px;\n");
    fprintf(fp, "            font-size: 0.9em;\n");
    fprintf(fp, "            font-weight: bold;\n");
    fprintf(fp, "        }\n");
    fprintf(fp, "        .badge-ai {\n");
    fprintf(fp, "            background: #dc3545;\n");
    fprintf(fp, "            color: white;\n");
    fprintf(fp, "        }\n");
    fprintf(fp, "        .badge-real {\n");
    fprintf(fp, "            background: #28a745;\n");
    fprintf(fp, "            color: white;\n");
    fprintf(fp, "        }\n");
    fprintf(fp, "        .confidence {\n");
    fprintf(fp, "            font-size: 1.5em;\n");
    fprintf(fp, "            font-weight: bold;\n");
    fprintf(fp, "            color: #667eea;\n");
    fprintf(fp, "        }\n");
    fprintf(fp, "        .features-grid {\n");
    fprintf(fp, "            display: grid;\n");
    fprintf(fp, "            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));\n");
    fprintf(fp, "            gap: 15px;\n");
    fprintf(fp, "            margin-top: 15px;\n");
    fprintf(fp, "        }\n");
    fprintf(fp, "        .feature-item {\n");
    fprintf(fp, "            background: #f8f9fa;\n");
    fprintf(fp, "            padding: 15px;\n");
    fprintf(fp, "            border-radius: 5px;\n");
    fprintf(fp, "            border-left: 4px solid #667eea;\n");
    fprintf(fp, "        }\n");
    fprintf(fp, "        .feature-label {\n");
    fprintf(fp, "            font-size: 0.85em;\n");
    fprintf(fp, "            color: #6c757d;\n");
    fprintf(fp, "            margin-bottom: 5px;\n");
    fprintf(fp, "        }\n");
    fprintf(fp, "        .feature-value {\n");
    fprintf(fp, "            font-size: 1.2em;\n");
    fprintf(fp, "            font-weight: bold;\n");
    fprintf(fp, "            color: #212529;\n");
    fprintf(fp, "        }\n");
    fprintf(fp, "        .progress-bar {\n");
    fprintf(fp, "            width: 100%%;\n");
    fprintf(fp, "            height: 8px;\n");
    fprintf(fp, "            background: #e9ecef;\n");
    fprintf(fp, "            border-radius: 4px;\n");
    fprintf(fp, "            overflow: hidden;\n");
    fprintf(fp, "            margin-top: 5px;\n");
    fprintf(fp, "        }\n");
    fprintf(fp, "        .progress-fill {\n");
    fprintf(fp, "            height: 100%%;\n");
    fprintf(fp, "            background: linear-gradient(90deg, #667eea 0%%, #764ba2 100%%);\n");
    fprintf(fp, "            transition: width 0.3s;\n");
    fprintf(fp, "        }\n");
    fprintf(fp, "        .statistics {\n");
    fprintf(fp, "            background: #f8f9fa;\n");
    fprintf(fp, "            padding: 20px;\n");
    fprintf(fp, "            border-radius: 8px;\n");
    fprintf(fp, "            margin-top: 30px;\n");
    fprintf(fp, "        }\n");
    fprintf(fp, "        .statistics h2 {\n");
    fprintf(fp, "            color: #495057;\n");
    fprintf(fp, "            margin-bottom: 15px;\n");
    fprintf(fp, "        }\n");
    fprintf(fp, "        .stat-grid {\n");
    fprintf(fp, "            display: grid;\n");
    fprintf(fp, "            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));\n");
    fprintf(fp, "            gap: 20px;\n");
    fprintf(fp, "        }\n");
    fprintf(fp, "        .stat-card {\n");
    fprintf(fp, "            background: white;\n");
    fprintf(fp, "            padding: 20px;\n");
    fprintf(fp, "            border-radius: 8px;\n");
    fprintf(fp, "            text-align: center;\n");
    fprintf(fp, "            box-shadow: 0 2px 4px rgba(0,0,0,0.1);\n");
    fprintf(fp, "        }\n");
    fprintf(fp, "        .stat-value {\n");
    fprintf(fp, "            font-size: 2em;\n");
    fprintf(fp, "            font-weight: bold;\n");
    fprintf(fp, "            color: #667eea;\n");
    fprintf(fp, "        }\n");
    fprintf(fp, "        .stat-label {\n");
    fprintf(fp, "            color: #6c757d;\n");
    fprintf(fp, "            margin-top: 5px;\n");
    fprintf(fp, "        }\n");
    fprintf(fp, "        .footer {\n");
    fprintf(fp, "            text-align: center;\n");
    fprintf(fp, "            padding: 20px;\n");
    fprintf(fp, "            color: #6c757d;\n");
    fprintf(fp, "            border-top: 1px solid #dee2e6;\n");
    fprintf(fp, "            margin-top: 30px;\n");
    fprintf(fp, "        }\n");
    fprintf(fp, "    </style>\n");
    fprintf(fp, "</head>\n");
    fprintf(fp, "<body>\n");
    fprintf(fp, "    <div class=\"container\">\n");
    
    // 页面头部
    fprintf(fp, "        <div class=\"header\">\n");
    fprintf(fp, "            <h1>🤖 AI Detection Results</h1>\n");
    fprintf(fp, "            <p>Automated Image Analysis Report</p>\n");
    fprintf(fp, "        </div>\n");
    
    // 元数据
    if (options->include_metadata) {
        fprintf(fp, "        <div class=\"metadata\">\n");
        
        fprintf(fp, "            <span class=\"metadata-item\">\n");
        fprintf(fp, "                <span class=\"metadata-label\">Version:</span> %s\n", 
               export_get_version());
        fprintf(fp, "            </span>\n");
        
        if (options->include_timestamp) {
            time_t now = time(NULL);
            char time_str[64];
            strftime(time_str, sizeof(time_str), "%Y-%m-%d %H:%M:%S", 
                    localtime(&now));
            
            fprintf(fp, "            <span class=\"metadata-item\">\n");
            fprintf(fp, "                <span class=\"metadata-label\">Generated:</span> %s\n", 
                   time_str);
            fprintf(fp, "            </span>\n");
        }
        
        fprintf(fp, "            <span class=\"metadata-item\">\n");
        fprintf(fp, "                <span class=\"metadata-label\">Total Detections:</span> %d\n", 
               count);
        fprintf(fp, "            </span>\n");
        
        fprintf(fp, "        </div>\n");
    }
    
    // 内容区域
    fprintf(fp, "        <div class=\"content\">\n");
    
    // 检测结果卡片
    for (int i = 0; i < count; i++) {
        const DetectionResult *result = &results[i];
        
        fprintf(fp, "            <div class=\"detection-card\">\n");
        
        // 卡片头部
        fprintf(fp, "                <div class=\"detection-header\">\n");
        fprintf(fp, "                    <div>\n");
        fprintf(fp, "                        <span class=\"detection-id\">Detection #%d</span>\n", 
               i + 1);
        
        // 类型标签
        if (result->type == DETECTION_AI_GENERATED) {
            fprintf(fp, "                        <span class=\"badge badge-ai\">AI Generated</span>\n");
        } else {
            fprintf(fp, "                        <span class=\"badge badge-real\">Real</span>\n");
        }
        
        fprintf(fp, "                    </div>\n");
        
        // 置信度
        if (options->include_confidence) {
            fprintf(fp, "                    <div class=\"confidence\">%.*f%%</div>\n", 
                   options->precision, result->confidence * 100.0);
        }
        
        fprintf(fp, "                </div>\n");
        
        // 区域信息
        if (options->include_coordinates && result->has_region) {
            fprintf(fp, "                <div style=\"margin-bottom: 15px; color: #6c757d;\">\n");
            fprintf(fp, "                    📍 Region: (%d, %d) - %dx%d pixels\n", 
                   result->region.x, result->region.y,
                   result->region.width, result->region.height);
            fprintf(fp, "                </div>\n");
        }
        
        // 特征网格
        fprintf(fp, "                <div class=\"features-grid\">\n");
        
        // 噪声水平
        fprintf(fp, "                    <div class=\"feature-item\">\n");
        fprintf(fp, "                        <div class=\"feature-label\">Noise Level</div>\n");
        fprintf(fp, "                        <div class=\"feature-value\">%.*f</div>\n", 
               options->precision, result->features.noise_level);
        fprintf(fp, "                        <div class=\"progress-bar\">\n");
        fprintf(fp, "                            <div class=\"progress-fill\" style=\"width: %.*f%%\"></div>\n", 
               0, result->features.noise_level * 100.0);
        fprintf(fp, "                        </div>\n");
        fprintf(fp, "                    </div>\n");
        
        // 压缩伪影
        fprintf(fp, "                    <div class=\"feature-item\">\n");
        fprintf(fp, "                        <div class=\"feature-label\">Compression Artifacts</div>\n");
        fprintf(fp, "                        <div class=\"feature-value\">%.*f</div>\n", 
               options->precision, result->features.compression_artifacts);
        fprintf(fp, "                        <div class=\"progress-bar\">\n");
        fprintf(fp, "                            <div class=\"progress-fill\" style=\"width: %.*f%%\"></div>\n", 
               0, result->features.compression_artifacts * 100.0);
        fprintf(fp, "                        </div>\n");
        fprintf(fp, "                    </div>\n");
        
        // 边缘一致性
        fprintf(fp, "                    <div class=\"feature-item\">\n");
        fprintf(fp, "                        <div class=\"feature-label\">Edge Consistency</div>\n");
        fprintf(fp, "                        <div class=\"feature-value\">%.*f</div>\n", 
               options->precision, result->features.edge_consistency);
        fprintf(fp, "                        <div class=\"progress-bar\">\n");
        fprintf(fp, "                            <div class=\"progress-fill\" style=\"width: %.*f%%\"></div>\n", 
               0, result->features.edge_consistency * 100.0);
        fprintf(fp, "                        </div>\n");
        fprintf(fp, "                    </div>\n");
        
        // 颜色分布
        fprintf(fp, "                    <div class=\"feature-item\">\n");
        fprintf(fp, "                        <div class=\"feature-label\">Color Distribution</div>\n");
        fprintf(fp, "                        <div class=\"feature-value\">%.*f</div>\n", 
               options->precision, result->features.color_distribution);
        fprintf(fp, "                        <div class=\"progress-bar\">\n");
        fprintf(fp, "                            <div class=\"progress-fill\" style=\"width: %.*f%%\"></div>\n", 
               0, result->features.color_distribution * 100.0);
        fprintf(fp, "                        </div>\n");
        fprintf(fp, "                    </div>\n");
        
        // 纹理模式
        fprintf(fp, "                    <div class=\"feature-item\">\n");
        fprintf(fp, "                        <div class=\"feature-label\">Texture Pattern</div>\n");
        fprintf(fp, "                        <div class=\"feature-value\">%.*f</div>\n", 
               options->precision, result->features.texture_pattern);
        fprintf(fp, "                        <div class=\"progress-bar\">\n");
        fprintf(fp, "                            <div class=\"progress-fill\" style=\"width: %.*f%%\"></div>\n", 
               0, result->features.texture_pattern * 100.0);
        fprintf(fp, "                        </div>\n");
        fprintf(fp, "                    </div>\n");
        
        fprintf(fp, "                </div>\n");
        fprintf(fp, "            </div>\n");
    }
    
    // 统计信息
    if (options->include_statistics) {
        ExportStatistics stats;
        if (export_calculate_statistics(results, count, &stats)) {
            fprintf(fp, "            <div class=\"statistics\">\n");
            fprintf(fp, "                <h2>📊 Statistics Summary</h2>\n");
            fprintf(fp, "                <div class=\"stat-grid\">\n");
            
            fprintf(fp, "                    <div class=\"stat-card\">\n");
            fprintf(fp, "                        <div class=\"stat-value\">%d</div>\n", 
                   stats.total_detections);
            fprintf(fp, "                        <div class=\"stat-label\">Total Detections</div>\n");
            fprintf(fp, "                    </div>\n");
            
            fprintf(fp, "                    <div class=\"stat-card\">\n");
            fprintf(fp, "                        <div class=\"stat-value\">%.*f%%</div>\n", 
                   options->precision, stats.average_confidence * 100.0);
            fprintf(fp, "                        <div class=\"stat-label\">Average Confidence</div>\n");
            fprintf(fp, "                    </div>\n");
            
            fprintf(fp, "                </div>\n");
            fprintf(fp, "            </div>\n");
        }
    }
    
    fprintf(fp, "        </div>\n");
    
    // 页脚
    fprintf(fp, "        <div class=\"footer\">\n");
    fprintf(fp, "            <p>Generated by AI Detection System v%s</p>\n", 
           export_get_version());
    fprintf(fp, "        </div>\n");
    
    fprintf(fp, "    </div>\n");
    fprintf(fp, "</body>\n");
    fprintf(fp, "</html>\n");
    
    fclose(fp);
    
    if (g_verbose) {
        printf("成功导出HTML到: %s\n", output_file);
    }
    
    return true;
}
/**
 * @file result_exporter.c
 * @brief 结果导出模块实现 - Part 5: Markdown导出和统计功能
 */

// ============================================================================
// Markdown导出实现
// ============================================================================

bool export_to_markdown(const DetectionResult *results, int count,
                       const char *output_file,
                       const ExportOptions *options)
{
    export_clear_error();
    
    if (!results || count <= 0 || !output_file) {
        export_set_error("无效的参数");
        return false;
    }
    
    if (!export_validate_path(output_file, true)) {
        return false;
    }
    
    ExportOptions default_opts = exporter_default_options();
    if (!options) {
        options = &default_opts;
    }
    
    FILE *fp = fopen(output_file, "w");
    if (!fp) {
        export_set_error("无法打开文件: %s", output_file);
        return false;
    }
    
    // 标题
    fprintf(fp, "# AI Detection Results\n\n");
    
    // 元数据
    if (options->include_metadata) {
        fprintf(fp, "## Metadata\n\n");
        fprintf(fp, "- **Version**: %s\n", export_get_version());
        
        if (options->include_timestamp) {
            time_t now = time(NULL);
            char time_str[64];
            strftime(time_str, sizeof(time_str), "%Y-%m-%d %H:%M:%S", 
                    localtime(&now));
            fprintf(fp, "- **Generated**: %s\n", time_str);
        }
        
        fprintf(fp, "- **Total Detections**: %d\n\n", count);
        fprintf(fp, "---\n\n");
    }
    
    // 检测结果
    fprintf(fp, "## Detection Results\n\n");
    
    for (int i = 0; i < count; i++) {
        const DetectionResult *result = &results[i];
        
        fprintf(fp, "### Detection #%d\n\n", i + 1);
        
        // 类型和置信度
        const char *type_str = (result->type == DETECTION_AI_GENERATED) ? 
                              "🤖 AI Generated" : "✅ Real";
        fprintf(fp, "**Type**: %s\n\n", type_str);
        
        if (options->include_confidence) {
            fprintf(fp, "**Confidence**: %.*f%%\n\n", 
                   options->precision, result->confidence * 100.0);
            
            // 置信度进度条（使用Unicode字符）
            int bar_length = 20;
            int filled = (int)(result->confidence * bar_length);
            fprintf(fp, "```\n[");
            for (int j = 0; j < bar_length; j++) {
                fprintf(fp, "%s", j < filled ? "█" : "░");
            }
            fprintf(fp, "] %.*f%%\n```\n\n", 
                   options->precision, result->confidence * 100.0);
        }
        
        // 区域信息
        if (options->include_coordinates && result->has_region) {
            fprintf(fp, "**Region**: `(%d, %d)` - `%dx%d` pixels\n\n", 
                   result->region.x, result->region.y,
                   result->region.width, result->region.height);
        }
        
        // 特征表格
        fprintf(fp, "#### Features\n\n");
        fprintf(fp, "| Feature | Value | Visualization |\n");
        fprintf(fp, "|---------|-------|---------------|\n");
        
        // 噪声水平
        fprintf(fp, "| Noise Level | %.*f | ", 
               options->precision, result->features.noise_level);
        int noise_bars = (int)(result->features.noise_level * 10);
        for (int j = 0; j < 10; j++) {
            fprintf(fp, "%s", j < noise_bars ? "▓" : "░");
        }
        fprintf(fp, " |\n");
        
        // 压缩伪影
        fprintf(fp, "| Compression Artifacts | %.*f | ", 
               options->precision, result->features.compression_artifacts);
        int comp_bars = (int)(result->features.compression_artifacts * 10);
        for (int j = 0; j < 10; j++) {
            fprintf(fp, "%s", j < comp_bars ? "▓" : "░");
        }
        fprintf(fp, " |\n");
        
        // 边缘一致性
        fprintf(fp, "| Edge Consistency | %.*f | ", 
               options->precision, result->features.edge_consistency);
        int edge_bars = (int)(result->features.edge_consistency * 10);
        for (int j = 0; j < 10; j++) {
            fprintf(fp, "%s", j < edge_bars ? "▓" : "░");
        }
        fprintf(fp, " |\n");
        
        // 颜色分布
        fprintf(fp, "| Color Distribution | %.*f | ", 
               options->precision, result->features.color_distribution);
        int color_bars = (int)(result->features.color_distribution * 10);
        for (int j = 0; j < 10; j++) {
            fprintf(fp, "%s", j < color_bars ? "▓" : "░");
        }
        fprintf(fp, " |\n");
        
        // 纹理模式
        fprintf(fp, "| Texture Pattern | %.*f | ", 
               options->precision, result->features.texture_pattern);
        int texture_bars = (int)(result->features.texture_pattern * 10);
        for (int j = 0; j < 10; j++) {
            fprintf(fp, "%s", j < texture_bars ? "▓" : "░");
        }
        fprintf(fp, " |\n\n");
        
        if (i < count - 1) {
            fprintf(fp, "---\n\n");
        }
    }
    
    // 统计信息
    if (options->include_statistics) {
        ExportStatistics stats;
        if (export_calculate_statistics(results, count, &stats)) {
            fprintf(fp, "## Statistics Summary\n\n");
            
            fprintf(fp, "| Metric | Value |\n");
            fprintf(fp, "|--------|-------|\n");
            fprintf(fp, "| Total Detections | %d |\n", stats.total_detections);
            fprintf(fp, "| AI Generated | %d |\n", stats.ai_generated_count);
            fprintf(fp, "| Real Images | %d |\n", stats.real_count);
            fprintf(fp, "| Average Confidence | %.*f%% |\n", 
                   options->precision, stats.average_confidence * 100.0);
            fprintf(fp, "| Min Confidence | %.*f%% |\n", 
                   options->precision, stats.min_confidence * 100.0);
            fprintf(fp, "| Max Confidence | %.*f%% |\n", 
                   options->precision, stats.max_confidence * 100.0);
            fprintf(fp, "\n");
            
            // 特征平均值
            fprintf(fp, "### Average Feature Values\n\n");
            fprintf(fp, "| Feature | Average |\n");
            fprintf(fp, "|---------|----------|\n");
            fprintf(fp, "| Noise Level | %.*f |\n", 
                   options->precision, stats.avg_noise_level);
            fprintf(fp, "| Compression Artifacts | %.*f |\n", 
                   options->precision, stats.avg_compression_artifacts);
            fprintf(fp, "| Edge Consistency | %.*f |\n", 
                   options->precision, stats.avg_edge_consistency);
            fprintf(fp, "| Color Distribution | %.*f |\n", 
                   options->precision, stats.avg_color_distribution);
            fprintf(fp, "| Texture Pattern | %.*f |\n", 
                   options->precision, stats.avg_texture_pattern);
            fprintf(fp, "\n");
        }
    }
    
    // 页脚
    fprintf(fp, "---\n\n");
    fprintf(fp, "*Generated by AI Detection System v%s*\n", export_get_version());
    
    fclose(fp);
    
    if (g_verbose) {
        printf("成功导出Markdown到: %s\n", output_file);
    }
    
    return true;
}

// ============================================================================
// 统计计算实现
// ============================================================================

bool export_calculate_statistics(const DetectionResult *results, int count,
                                 ExportStatistics *stats)
{
    if (!results || count <= 0 || !stats) {
        return false;
    }
    
    memset(stats, 0, sizeof(ExportStatistics));
    
    stats->total_detections = count;
    stats->min_confidence = 1.0;
    stats->max_confidence = 0.0;
    
    double sum_confidence = 0.0;
    double sum_noise = 0.0;
    double sum_compression = 0.0;
    double sum_edge = 0.0;
    double sum_color = 0.0;
    double sum_texture = 0.0;
    
    for (int i = 0; i < count; i++) {
        const DetectionResult *result = &results[i];
        
        // 计数
        if (result->type == DETECTION_AI_GENERATED) {
            stats->ai_generated_count++;
        } else {
            stats->real_count++;
        }
        
        // 置信度统计
        sum_confidence += result->confidence;
        if (result->confidence < stats->min_confidence) {
            stats->min_confidence = result->confidence;
        }
        if (result->confidence > stats->max_confidence) {
            stats->max_confidence = result->confidence;
        }
        
        // 特征统计
        sum_noise += result->features.noise_level;
        sum_compression += result->features.compression_artifacts;
        sum_edge += result->features.edge_consistency;
        sum_color += result->features.color_distribution;
        sum_texture += result->features.texture_pattern;
    }
    
    // 计算平均值
    stats->average_confidence = sum_confidence / count;
    stats->avg_noise_level = sum_noise / count;
    stats->avg_compression_artifacts = sum_compression / count;
    stats->avg_edge_consistency = sum_edge / count;
    stats->avg_color_distribution = sum_color / count;
    stats->avg_texture_pattern = sum_texture / count;
    
    // 计算标准差
    double var_confidence = 0.0;
    double var_noise = 0.0;
    double var_compression = 0.0;
    double var_edge = 0.0;
    double var_color = 0.0;
    double var_texture = 0.0;
    
    for (int i = 0; i < count; i++) {
        const DetectionResult *result = &results[i];
        
        double diff_conf = result->confidence - stats->average_confidence;
        var_confidence += diff_conf * diff_conf;
        
        double diff_noise = result->features.noise_level - stats->avg_noise_level;
        var_noise += diff_noise * diff_noise;
        
        double diff_comp = result->features.compression_artifacts - 
                          stats->avg_compression_artifacts;
        var_compression += diff_comp * diff_comp;
        
        double diff_edge = result->features.edge_consistency - 
                          stats->avg_edge_consistency;
        var_edge += diff_edge * diff_edge;
        
        double diff_color = result->features.color_distribution - 
                           stats->avg_color_distribution;
        var_color += diff_color * diff_color;
        
        double diff_texture = result->features.texture_pattern - 
                             stats->avg_texture_pattern;
        var_texture += diff_texture * diff_texture;
    }
    
    stats->std_dev_confidence = sqrt(var_confidence / count);
    stats->std_dev_noise_level = sqrt(var_noise / count);
    stats->std_dev_compression_artifacts = sqrt(var_compression / count);
    stats->std_dev_edge_consistency = sqrt(var_edge / count);
    stats->std_dev_color_distribution = sqrt(var_color / count);
    stats->std_dev_texture_pattern = sqrt(var_texture / count);
    
    return true;
}

// ============================================================================
// 统计报告生成
// ============================================================================

bool export_statistics_report(const DetectionResult *results, int count,
                              const char *output_file,
                              ExportFormat format)
{
    export_clear_error();
    
    if (!results || count <= 0 || !output_file) {
        export_set_error("无效的参数");
        return false;
    }
    
    ExportStatistics stats;
    if (!export_calculate_statistics(results, count, &stats)) {
        export_set_error("统计计算失败");
        return false;
    }
    
    FILE *fp = fopen(output_file, "w");
    if (!fp) {
        export_set_error("无法打开文件: %s", output_file);
        return false;
    }
    
    if (format == EXPORT_FORMAT_TXT) {
        // 文本格式统计报告
        fprintf(fp, "AI Detection Statistics Report\n");
        fprintf(fp, "==============================\n\n");
        
        fprintf(fp, "Detection Summary\n");
        fprintf(fp, "-----------------\n");
        fprintf(fp, "Total Detections:     %d\n", stats.total_detections);
        fprintf(fp, "AI Generated:         %d (%.1f%%)\n", 
               stats.ai_generated_count,
               (stats.ai_generated_count * 100.0) / stats.total_detections);
        fprintf(fp, "Real Images:          %d (%.1f%%)\n\n", 
               stats.real_count,
               (stats.real_count * 100.0) / stats.total_detections);
        
        fprintf(fp, "Confidence Statistics\n");
        fprintf(fp, "---------------------\n");
        fprintf(fp, "Average:              %.4f\n", stats.average_confidence);
        fprintf(fp, "Minimum:              %.4f\n", stats.min_confidence);
        fprintf(fp, "Maximum:              %.4f\n", stats.max_confidence);
        fprintf(fp, "Standard Deviation:   %.4f\n\n", stats.std_dev_confidence);
        
        fprintf(fp, "Feature Averages\n");
        fprintf(fp, "----------------\n");
        fprintf(fp, "Noise Level:          %.4f (±%.4f)\n", 
               stats.avg_noise_level, stats.std_dev_noise_level);
        fprintf(fp, "Compression:          %.4f (±%.4f)\n", 
               stats.avg_compression_artifacts, stats.std_dev_compression_artifacts);
        fprintf(fp, "Edge Consistency:     %.4f (±%.4f)\n", 
               stats.avg_edge_consistency, stats.std_dev_edge_consistency);
        fprintf(fp, "Color Distribution:   %.4f (±%.4f)\n", 
               stats.avg_color_distribution, stats.std_dev_color_distribution);
        fprintf(fp, "Texture Pattern:      %.4f (±%.4f)\n", 
               stats.avg_texture_pattern, stats.std_dev_texture_pattern);
        
    } else if (format == EXPORT_FORMAT_JSON) {
        // JSON格式统计报告
        fprintf(fp, "{\n");
        fprintf(fp, "  \"summary\": {\n");
        fprintf(fp, "    \"total_detections\": %d,\n", stats.total_detections);
        fprintf(fp, "    \"ai_generated_count\": %d,\n", stats.ai_generated_count);
        fprintf(fp, "    \"real_count\": %d\n", stats.real_count);
        fprintf(fp, "  },\n");
        fprintf(fp, "  \"confidence\": {\n");
        fprintf(fp, "    \"average\": %.4f,\n", stats.average_confidence);
        fprintf(fp, "    \"min\": %.4f,\n", stats.min_confidence);
        fprintf(fp, "    \"max\": %.4f,\n", stats.max_confidence);
        fprintf(fp, "    \"std_dev\": %.4f\n", stats.std_dev_confidence);
        fprintf(fp, "  },\n");
        fprintf(fp, "  \"features\": {\n");
        fprintf(fp, "    \"noise_level\": {\n");
        fprintf(fp, "      \"average\": %.4f,\n", stats.avg_noise_level);
        fprintf(fp, "      \"std_dev\": %.4f\n", stats.std_dev_noise_level);
        fprintf(fp, "    },\n");
        fprintf(fp, "    \"compression_artifacts\": {\n");
        fprintf(fp, "      \"average\": %.4f,\n", stats.avg_compression_artifacts);
        fprintf(fp, "      \"std_dev\": %.4f\n", stats.std_dev_compression_artifacts);
        fprintf(fp, "    },\n");
        fprintf(fp, "    \"edge_consistency\": {\n");
        fprintf(fp, "      \"average\": %.4f,\n", stats.avg_edge_consistency);
        fprintf(fp, "      \"std_dev\": %.4f\n", stats.std_dev_edge_consistency);
        fprintf(fp, "    },\n");
        fprintf(fp, "    \"color_distribution\": {\n");
        fprintf(fp, "      \"average\": %.4f,\n", stats.avg_color_distribution);
        fprintf(fp, "      \"std_dev\": %.4f\n", stats.std_dev_color_distribution);
        fprintf(fp, "    },\n");
        fprintf(fp, "    \"texture_pattern\": {\n");
        fprintf(fp, "      \"average\": %.4f,\n", stats.avg_texture_pattern);
        fprintf(fp, "      \"std_dev\": %.4f\n", stats.std_dev_texture_pattern);
        fprintf(fp, "    }\n");
        fprintf(fp, "  }\n");
        fprintf(fp, "}\n");
    }
    
    fclose(fp);
    
    if (g_verbose) {
        printf("成功导出统计报告到: %s\n", output_file);
    }
    
    return true;
}

// ============================================================================
// 比较报告生成
// ============================================================================

bool export_comparison_report(const DetectionResult *results1, int count1,
                              const DetectionResult *results2, int count2,
                              const char *output_file,
                              const char *label1,
                              const char *label2)
{
    export_clear_error();
    
    if (!results1 || !results2 || count1 <= 0 || count2 <= 0 || !output_file) {
        export_set_error("无效的参数");
        return false;
    }
    
    ExportStatistics stats1, stats2;
    if (!export_calculate_statistics(results1, count1, &stats1) ||
        !export_calculate_statistics(results2, count2, &stats2)) {
        export_set_error("统计计算失败");
        return false;
    }
    
    FILE *fp = fopen(output_file, "w");
    if (!fp) {
        export_set_error("无法打开文件: %s", output_file);
        return false;
    }
    
    const char *name1 = label1 ? label1 : "Dataset 1";
    const char *name2 = label2 ? label2 : "Dataset 2";
    
    fprintf(fp, "# Detection Results Comparison\n\n");
    fprintf(fp, "## Overview\n\n");
    fprintf(fp, "| Metric | %s | %s | Difference |\n", name1, name2);
    fprintf(fp, "|--------|%s|%s|------------|\n",
           "----------|", "----------|");
    
    fprintf(fp, "| Total Detections | %d | %d | %+d |\n",
           stats1.total_detections, stats2.total_detections,
           stats2.total_detections - stats1.total_detections);
    
    fprintf(fp, "| AI Generated | %d | %d | %+d |\n",
           stats1.ai_generated_count, stats2.ai_generated_count,
           stats2.ai_generated_count - stats1.ai_generated_count);
    
    fprintf(fp, "| Real Images | %d | %d | %+d |\n",
           stats1.real_count, stats2.real_count,
           stats2.real_count - stats1.real_count);
    
    fprintf(fp, "| Avg Confidence | %.4f | %.4f | %+.4f |\n\n",
           stats1.average_confidence, stats2.average_confidence,
           stats2.average_confidence - stats1.average_confidence);
    
    fprintf(fp, "## Feature Comparison\n\n");
    fprintf(fp, "| Feature | %s | %s | Difference |\n", name1, name2);
    fprintf(fp, "|---------|%s|%s|------------|\n",
           "----------|", "----------|");
    
    fprintf(fp, "| Noise Level | %.4f | %.4f | %+.4f |\n",
           stats1.avg_noise_level, stats2.avg_noise_level,
           stats2.avg_noise_level - stats1.avg_noise_level);
    
    fprintf(fp, "| Compression | %.4f | %.4f | %+.4f |\n",
           stats1.avg_compression_artifacts, stats2.avg_compression_artifacts,
           stats2.avg_compression_artifacts - stats1.avg_compression_artifacts);
    
    fprintf(fp, "| Edge Consistency | %.4f | %.4f | %+.4f |\n",
           stats1.avg_edge_consistency, stats2.avg_edge_consistency,
           stats2.avg_edge_consistency - stats1.avg_edge_consistency);
    
    fprintf(fp, "| Color Distribution | %.4f | %.4f | %+.4f |\n",
           stats1.avg_color_distribution, stats2.avg_color_distribution,
           stats2.avg_color_distribution - stats1.avg_color_distribution);
    
    fprintf(fp, "| Texture Pattern | %.4f | %.4f | %+.4f |\n",
           stats1.avg_texture_pattern, stats2.avg_texture_pattern,
           stats2.avg_texture_pattern - stats1.avg_texture_pattern);
    
    fclose(fp);
    
    if (g_verbose) {
        printf("成功导出比较报告到: %s\n", output_file);
    }
    
    return true;
}
/**
 * @file result_exporter.c
 * @brief 结果导出模块实现 - Part 6: 辅助函数和完成
 */

// ============================================================================
// 批量导出实现
// ============================================================================

bool export_batch(const DetectionResult *results, int count,
                 const char *base_path,
                 const ExportFormat *formats,
                 int format_count,
                 const ExportOptions *options)
{
    export_clear_error();
    
    if (!results || count <= 0 || !base_path || !formats || format_count <= 0) {
        export_set_error("无效的参数");
        return false;
    }
    
    ExportOptions default_opts = exporter_default_options();
    if (!options) {
        options = &default_opts;
    }
    
    bool all_success = true;
    int success_count = 0;
    
    for (int i = 0; i < format_count; i++) {
        char output_file[512];
        const char *ext = export_get_format_extension(formats[i]);
        
        snprintf(output_file, sizeof(output_file), "%s.%s", base_path, ext);
        
        bool result = false;
        switch (formats[i]) {
            case EXPORT_FORMAT_JSON:
                result = export_to_json(results, count, output_file, options);
                break;
            case EXPORT_FORMAT_XML:
                result = export_to_xml(results, count, output_file, options);
                break;
            case EXPORT_FORMAT_CSV:
                result = export_to_csv(results, count, output_file, options);
                break;
            case EXPORT_FORMAT_TXT:
                result = export_to_txt(results, count, output_file, options);
                break;
            case EXPORT_FORMAT_HTML:
                result = export_to_html(results, count, output_file, options);
                break;
            case EXPORT_FORMAT_YAML:
                result = export_to_yaml(results, count, output_file, options);
                break;
            case EXPORT_FORMAT_MARKDOWN:
                result = export_to_markdown(results, count, output_file, options);
                break;
            default:
                export_set_error("不支持的格式: %d", formats[i]);
                result = false;
                break;
        }
        
        if (result) {
            success_count++;
        } else {
            all_success = false;
            if (g_verbose) {
                fprintf(stderr, "导出失败 (%s): %s\n", 
                       ext, export_get_last_error());
            }
        }
    }
    
    if (g_verbose) {
        printf("批量导出完成: %d/%d 成功\n", success_count, format_count);
    }
    
    return all_success;
}

// ============================================================================
// 过滤导出实现
// ============================================================================

bool export_filtered(const DetectionResult *results, int count,
                    const char *output_file,
                    ExportFormat format,
                    const ExportFilter *filter,
                    const ExportOptions *options)
{
    export_clear_error();
    
    if (!results || count <= 0 || !output_file || !filter) {
        export_set_error("无效的参数");
        return false;
    }
    
    // 分配临时数组存储过滤后的结果
    DetectionResult *filtered = malloc(count * sizeof(DetectionResult));
    if (!filtered) {
        export_set_error("内存分配失败");
        return false;
    }
    
    int filtered_count = 0;
    
    // 应用过滤器
    for (int i = 0; i < count; i++) {
        const DetectionResult *result = &results[i];
        bool include = true;
        
        // 类型过滤
        if (filter->filter_by_type) {
            if (result->type != filter->type) {
                include = false;
            }
        }
        
        // 置信度过滤
        if (include && filter->filter_by_confidence) {
            if (result->confidence < filter->min_confidence ||
                result->confidence > filter->max_confidence) {
                include = false;
            }
        }
        
        // 区域过滤
        if (include && filter->filter_by_region && result->has_region) {
            if (result->region.width < filter->min_width ||
                result->region.width > filter->max_width ||
                result->region.height < filter->min_height ||
                result->region.height > filter->max_height) {
                include = false;
            }
        }
        
        // 特征过滤
        if (include && filter->filter_by_features) {
            if (result->features.noise_level < filter->min_noise_level ||
                result->features.noise_level > filter->max_noise_level) {
                include = false;
            }
        }
        
        if (include) {
            filtered[filtered_count++] = *result;
        }
    }
    
    if (filtered_count == 0) {
        free(filtered);
        export_set_error("没有符合过滤条件的结果");
        return false;
    }
    
    // 导出过滤后的结果
    bool result = export_results(filtered, filtered_count, output_file, 
                                 format, options);
    
    free(filtered);
    
    if (g_verbose && result) {
        printf("过滤导出完成: %d/%d 结果符合条件\n", filtered_count, count);
    }
    
    return result;
}

// ============================================================================
// 分页导出实现
// ============================================================================

bool export_paginated(const DetectionResult *results, int count,
                     const char *base_path,
                     ExportFormat format,
                     int page_size,
                     const ExportOptions *options)
{
    export_clear_error();
    
    if (!results || count <= 0 || !base_path || page_size <= 0) {
        export_set_error("无效的参数");
        return false;
    }
    
    int total_pages = (count + page_size - 1) / page_size;
    bool all_success = true;
    
    for (int page = 0; page < total_pages; page++) {
        int start_idx = page * page_size;
        int end_idx = start_idx + page_size;
        if (end_idx > count) {
            end_idx = count;
        }
        
        int page_count = end_idx - start_idx;
        
        // 生成分页文件名
        char output_file[512];
        const char *ext = export_get_format_extension(format);
        snprintf(output_file, sizeof(output_file), "%s_page%d.%s", 
                base_path, page + 1, ext);
        
        // 导出当前页
        bool result = export_results(&results[start_idx], page_count, 
                                     output_file, format, options);
        
        if (!result) {
            all_success = false;
            if (g_verbose) {
                fprintf(stderr, "导出第 %d 页失败: %s\n", 
                       page + 1, export_get_last_error());
            }
        }
    }
    
    if (g_verbose) {
        printf("分页导出完成: %d 页\n", total_pages);
    }
    
    return all_success;
}

// ============================================================================
// 增量导出实现
// ============================================================================

bool export_append(const DetectionResult *results, int count,
                  const char *output_file,
                  ExportFormat format)
{
    export_clear_error();
    
    if (!results || count <= 0 || !output_file) {
        export_set_error("无效的参数");
        return false;
    }
    
    // 只有某些格式支持追加
    if (format != EXPORT_FORMAT_CSV && 
        format != EXPORT_FORMAT_TXT &&
        format != EXPORT_FORMAT_JSON) {
        export_set_error("该格式不支持追加模式");
        return false;
    }
    
    FILE *fp = fopen(output_file, "a");
    if (!fp) {
        export_set_error("无法打开文件: %s", output_file);
        return false;
    }
    
    if (format == EXPORT_FORMAT_CSV) {
        // CSV追加（不写头部）
        for (int i = 0; i < count; i++) {
            const DetectionResult *result = &results[i];
            
            fprintf(fp, "%d,", i);
            fprintf(fp, "%s,", 
                   (result->type == DETECTION_AI_GENERATED) ? "AI_Generated" : "Real");
            fprintf(fp, "%.4f,", result->confidence);
            
            if (result->has_region) {
                fprintf(fp, "%d,%d,%d,%d,", 
                       result->region.x, result->region.y,
                       result->region.width, result->region.height);
            } else {
                fprintf(fp, ",,,,");
            }
            
            fprintf(fp, "%.4f,", result->features.noise_level);
            fprintf(fp, "%.4f,", result->features.compression_artifacts);
            fprintf(fp, "%.4f,", result->features.edge_consistency);
            fprintf(fp, "%.4f,", result->features.color_distribution);
            fprintf(fp, "%.4f\n", result->features.texture_pattern);
        }
    } else if (format == EXPORT_FORMAT_TXT) {
        // TXT追加
        for (int i = 0; i < count; i++) {
            const DetectionResult *result = &results[i];
            
            fprintf(fp, "\nDetection #%d\n", i + 1);
            fprintf(fp, "Type: %s\n", 
                   (result->type == DETECTION_AI_GENERATED) ? "AI Generated" : "Real");
            fprintf(fp, "Confidence: %.4f\n", result->confidence);
            
            if (result->has_region) {
                fprintf(fp, "Region: (%d, %d) - %dx%d\n", 
                       result->region.x, result->region.y,
                       result->region.width, result->region.height);
            }
            
            fprintf(fp, "Features:\n");
            fprintf(fp, "  Noise: %.4f\n", result->features.noise_level);
            fprintf(fp, "  Compression: %.4f\n", result->features.compression_artifacts);
            fprintf(fp, "  Edge: %.4f\n", result->features.edge_consistency);
            fprintf(fp, "  Color: %.4f\n", result->features.color_distribution);
            fprintf(fp, "  Texture: %.4f\n", result->features.texture_pattern);
        }
    }
    
    fclose(fp);
    
    if (g_verbose) {
        printf("成功追加 %d 条结果到: %s\n", count, output_file);
    }
    
    return true;
}

// ============================================================================
// 模板导出实现
// ============================================================================

bool export_with_template(const DetectionResult *results, int count,
                         const char *output_file,
                         const char *template_file,
                         const ExportOptions *options)
{
    export_clear_error();
    
    if (!results || count <= 0 || !output_file || !template_file) {
        export_set_error("无效的参数");
        return false;
    }
    
    // 读取模板文件
    FILE *template_fp = fopen(template_file, "r");
    if (!template_fp) {
        export_set_error("无法打开模板文件: %s", template_file);
        return false;
    }
    
    // 读取模板内容
    fseek(template_fp, 0, SEEK_END);
    long template_size = ftell(template_fp);
    fseek(template_fp, 0, SEEK_SET);
    
    char *template_content = malloc(template_size + 1);
    if (!template_content) {
        fclose(template_fp);
        export_set_error("内存分配失败");
        return false;
    }
    
    fread(template_content, 1, template_size, template_fp);
    template_content[template_size] = '\0';
    fclose(template_fp);
    
    // 打开输出文件
    FILE *output_fp = fopen(output_file, "w");
    if (!output_fp) {
        free(template_content);
        export_set_error("无法打开输出文件: %s", output_file);
        return false;
    }
    
    // 处理模板变量
    char *pos = template_content;
    while (*pos) {
        if (strncmp(pos, "{{", 2) == 0) {
            // 找到变量
            char *end = strstr(pos + 2, "}}");
            if (end) {
                char var_name[128];
                int var_len = end - pos - 2;
                if (var_len < sizeof(var_name)) {
                    strncpy(var_name, pos + 2, var_len);
                    var_name[var_len] = '\0';
                    
                    // 替换变量
                    if (strcmp(var_name, "VERSION") == 0) {
                        fprintf(output_fp, "%s", export_get_version());
                    } else if (strcmp(var_name, "COUNT") == 0) {
                        fprintf(output_fp, "%d", count);
                    } else if (strcmp(var_name, "TIMESTAMP") == 0) {
                        time_t now = time(NULL);
                        char time_str[64];
                        strftime(time_str, sizeof(time_str), "%Y-%m-%d %H:%M:%S", 
                                localtime(&now));
                        fprintf(output_fp, "%s", time_str);
                    } else if (strncmp(var_name, "RESULTS", 7) == 0) {
                        // 输出结果列表
                        for (int i = 0; i < count; i++) {
                            const DetectionResult *result = &results[i];
                            fprintf(output_fp, "Detection %d: %s (%.2f%%)\n",
                                   i + 1,
                                   (result->type == DETECTION_AI_GENERATED) ? 
                                   "AI" : "Real",
                                   result->confidence * 100.0);
                        }
                    }
                    
                    pos = end + 2;
                    continue;
                }
            }
        }
        
        fputc(*pos, output_fp);
        pos++;
    }
    
    free(template_content);
    fclose(output_fp);
    
    if (g_verbose) {
        printf("成功使用模板导出到: %s\n", output_file);
    }
    
    return true;
}

// ============================================================================
// 压缩导出实现
// ============================================================================

#ifdef HAVE_ZLIB
#include <zlib.h>

bool export_compressed(const DetectionResult *results, int count,
                      const char *output_file,
                      ExportFormat format,
                      const ExportOptions *options)
{
    export_clear_error();
    
    if (!results || count <= 0 || !output_file) {
        export_set_error("无效的参数");
        return false;
    }
    
    // 先导出到临时文件
    char temp_file[512];
    snprintf(temp_file, sizeof(temp_file), "%s.tmp", output_file);
    
    if (!export_results(results, count, temp_file, format, options)) {
        return false;
    }
    
    // 读取临时文件
    FILE *fp = fopen(temp_file, "rb");
    if (!fp) {
        export_set_error("无法打开临时文件");
        remove(temp_file);
        return false;
    }
    
    fseek(fp, 0, SEEK_END);
    long file_size = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    
    unsigned char *buffer = malloc(file_size);
    if (!buffer) {
        fclose(fp);
        remove(temp_file);
        export_set_error("内存分配失败");
        return false;
    }
    
    fread(buffer, 1, file_size, fp);
    fclose(fp);
    
    // 压缩数据
    uLongf compressed_size = compressBound(file_size);
    unsigned char *compressed = malloc(compressed_size);
    if (!compressed) {
        free(buffer);
        remove(temp_file);
        export_set_error("内存分配失败");
        return false;
    }
    
    int result = compress2(compressed, &compressed_size, buffer, file_size, 
                          Z_BEST_COMPRESSION);
    
    free(buffer);
    
    if (result != Z_OK) {
        free(compressed);
        remove(temp_file);
        export_set_error("压缩失败");
        return false;
    }
    
    // 写入压缩文件
    gzFile gz = gzopen(output_file, "wb");
    if (!gz) {
        free(compressed);
        remove(temp_file);
        export_set_error("无法创建压缩文件");
        return false;
    }
    
    gzwrite(gz, compressed, compressed_size);
    gzclose(gz);
    
    free(compressed);
    remove(temp_file);
    
    if (g_verbose) {
        printf("成功导出压缩文件: %s (压缩率: %.1f%%)\n", 
               output_file, (compressed_size * 100.0) / file_size);
    }
    
    return true;
}
#endif

// ============================================================================
// 加密导出实现
// ============================================================================

#ifdef HAVE_OPENSSL
#include <openssl/evp.h>
#include <openssl/aes.h>

bool export_encrypted(const DetectionResult *results, int count,
                     const char *output_file,
                     ExportFormat format,
                     const char *password,
                     const ExportOptions *options)
{
    export_clear_error();
    
    if (!results || count <= 0 || !output_file || !password) {
        export_set_error("无效的参数");
        return false;
    }
    
    // 先导出到临时文件
    char temp_file[512];
    snprintf(temp_file, sizeof(temp_file), "%s.tmp", output_file);
    
    if (!export_results(results, count, temp_file, format, options)) {
        return false;
    }
    
    // 读取临时文件
    FILE *fp = fopen(temp_file, "rb");
    if (!fp) {
        export_set_error("无法打开临时文件");
        remove(temp_file);
        return false;
    }
    
    fseek(fp, 0, SEEK_END);
    long file_size = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    
    unsigned char *plaintext = malloc(file_size);
    if (!plaintext) {
        fclose(fp);
        remove(temp_file);
        export_set_error("内存分配失败");
        return false;
    }
    
    fread(plaintext, 1, file_size, fp);
    fclose(fp);
    
    // 生成密钥和IV
    unsigned char key[32];
    unsigned char iv[16];
    
    EVP_BytesToKey(EVP_aes_256_cbc(), EVP_sha256(), NULL,
                   (unsigned char*)password, strlen(password),
                   1, key, iv);
    
    // 加密
    EVP_CIPHER_CTX *ctx = EVP_CIPHER_CTX_new();
    EVP_EncryptInit_ex(ctx, EVP_aes_256_cbc(), NULL, key, iv);
    
    int cipher_len = file_size + EVP_CIPHER_block_size(EVP_aes_256_cbc());
    unsigned char *ciphertext = malloc(cipher_len);
    if (!ciphertext) {
        EVP_CIPHER_CTX_free(ctx);
        free(plaintext);
        remove(temp_file);
        export_set_error("内存分配失败");
        return false;
    }
    
    int len;
    EVP_EncryptUpdate(ctx, ciphertext, &len, plaintext, file_size);
    int ciphertext_len = len;
    
    EVP_EncryptFinal_ex(ctx, ciphertext + len, &len);
    ciphertext_len += len;
    
    EVP_CIPHER_CTX_free(ctx);
    free(plaintext);
    
    // 写入加密文件
    FILE *out_fp = fopen(output_file, "wb");
    if (!out_fp) {
        free(ciphertext);
        remove(temp_file);
        export_set_error("无法创建输出文件");
        return false;
    }
    
    // 写入魔数和版本
    fwrite("AIENC", 1, 5, out_fp);
    uint8_t version = 1;
    fwrite(&version, 1, 1, out_fp);
    
    // 写入IV
    fwrite(iv, 1, 16, out_fp);
    
    // 写入加密数据
    fwrite(ciphertext, 1, ciphertext_len, out_fp);
    
    fclose(out_fp);
    free(ciphertext);
    remove(temp_file);
    
    if (g_verbose) {
        printf("成功导出加密文件: %s\n", output_file);
    }
    
    return true;
}
#endif

// ============================================================================
// 验证导出文件
// ============================================================================

bool export_verify(const char *file_path, ExportFormat format)
{
    export_clear_error();
    
    if (!file_path) {
        export_set_error("无效的文件路径");
        return false;
    }
    
    FILE *fp = fopen(file_path, "r");
    if (!fp) {
        export_set_error("无法打开文件: %s", file_path);
        return false;
    }
    
    bool valid = false;
    
    switch (format) {
        case EXPORT_FORMAT_JSON: {
            // 简单验证JSON格式
            char buffer[1024];
            if (fgets(buffer, sizeof(buffer), fp)) {
                valid = (strchr(buffer, '{') != NULL);
            }
            break;
        }
        
        case EXPORT_FORMAT_XML: {
            // 简单验证XML格式
            char buffer[1024];
            if (fgets(buffer, sizeof(buffer), fp)) {
                valid = (strstr(buffer, "<?xml") != NULL);
            }
            break;
        }
        
        case EXPORT_FORMAT_CSV: {
            // 验证CSV头部
            char buffer[1024];
            if (fgets(buffer, sizeof(buffer), fp)) {
                valid = (strstr(buffer, "ID,Type,Confidence") != NULL);
            }
            break;
        }
        
        case EXPORT_FORMAT_HTML: {
            // 验证HTML格式
            char buffer[1024];
            if (fgets(buffer, sizeof(buffer), fp)) {
                valid = (strstr(buffer, "<!DOCTYPE html>") != NULL ||
                        strstr(buffer, "<html") != NULL);
            }
            break;
        }
        
        case EXPORT_FORMAT_YAML: {
            // 验证YAML格式
            char buffer[1024];
            if (fgets(buffer, sizeof(buffer), fp)) {
                valid = (strncmp(buffer, "---", 3) == 0);
            }
            break;
        }
        
        case EXPORT_FORMAT_MARKDOWN: {
            // 验证Markdown格式
            char buffer[1024];
            if (fgets(buffer, sizeof(buffer), fp)) {
                valid = (buffer[0] == '#');
            }
            break;
        }
        
        default:
            valid = true;  // 其他格式默认有效
            break;
    }
    
    fclose(fp);
    
    if (!valid) {
        export_set_error("文件格式验证失败");
    }
    
    return valid;
}

// ============================================================================
// 获取文件信息
// ============================================================================

bool export_get_file_info(const char *file_path, ExportFileInfo *info)
{
    if (!file_path || !info) {
        return false;
    }
    
    memset(info, 0, sizeof(ExportFileInfo));
    
    struct stat st;
    if (stat(file_path, &st) != 0) {
        return false;
    }
    
    info->file_size = st.st_size;
    info->creation_time = st.st_ctime;
    info->modification_time = st.st_mtime;
    
    // 检测格式
    const char *ext = strrchr(file_path, '.');
    if (ext) {
        ext++;
        if (strcmp(ext, "json") == 0) {
            info->format = EXPORT_FORMAT_JSON;
        } else if (strcmp(ext, "xml") == 0) {
            info->format = EXPORT_FORMAT_XML;
        } else if (strcmp(ext, "csv") == 0) {
            info->format = EXPORT_FORMAT_CSV;
        } else if (strcmp(ext, "txt") == 0) {
            info->format = EXPORT_FORMAT_TXT;
        } else if (strcmp(ext, "html") == 0) {
            info->format = EXPORT_FORMAT_HTML;
        } else if (strcmp(ext, "yaml") == 0 || strcmp(ext, "yml") == 0) {
            info->format = EXPORT_FORMAT_YAML;
        } else if (strcmp(ext, "md") == 0) {
            info->format = EXPORT_FORMAT_MARKDOWN;
        }
    }
    
    strncpy(info->file_path, file_path, sizeof(info->file_path) - 1);
    
    return true;
}

// ============================================================================
// 清理和资源管理
// ============================================================================

void export_cleanup(void)
{
    // 清理全局资源
    export_clear_error();
    
    if (g_verbose) {
        printf("导出模块已清理\n");
    }
}

// ============================================================================
// 版本信息
// ============================================================================

void export_print_version(void)
{
    printf("Result Exporter Module v%s\n", export_get_version());
    printf("Supported formats:\n");
    printf("  - JSON\n");
    printf("  - XML\n");
    printf("  - CSV\n");
    printf("  - TXT\n");
    printf("  - HTML\n");
    printf("  - YAML\n");
    printf("  - Markdown\n");
    
#ifdef HAVE_ZLIB
    printf("  - Compression support: Yes\n");
#else
    printf("  - Compression support: No\n");
#endif

#ifdef HAVE_OPENSSL
    printf("  - Encryption support: Yes\n");
#else
    printf("  - Encryption support: No\n");
#endif
}

// ============================================================================
// 帮助信息
// ============================================================================

void export_print_help(void)
{
    printf("Result Exporter - Usage Guide\n");
    printf("==============================\n\n");
    
    printf("Basic Export:\n");
    printf("  export_results(results, count, \"output.json\", EXPORT_FORMAT_JSON, NULL);\n\n");
    
    printf("With Options:\n");
    printf("  ExportOptions opts = exporter_default_options();\n");
    printf("  opts.pretty_print = true;\n");
    printf("  opts.include_metadata = true;\n");
    printf("  export_results(results, count, \"output.json\", EXPORT_FORMAT_JSON, &opts);\n\n");
    
    printf("Batch Export:\n");
    printf("  ExportFormat formats[] = {EXPORT_FORMAT_JSON, EXPORT_FORMAT_CSV};\n");
    printf("  export_batch(results, count, \"output\", formats, 2, NULL);\n\n");
    
    printf("Filtered Export:\n");
    printf("  ExportFilter filter = {0};\n");
    printf("  filter.filter_by_confidence = true;\n");
    printf("  filter.min_confidence = 0.8;\n");
    printf("  export_filtered(results, count, \"high_conf.json\", EXPORT_FORMAT_JSON, &filter, NULL);\n\n");
    
    printf("Statistics Report:\n");
    printf("  export_statistics_report(results, count, \"stats.txt\", EXPORT_FORMAT_TXT);\n\n");
}

// ============================================================================
// 模块初始化（可选）
// ============================================================================

bool export_init(void)
{
    // 初始化全局状态
    export_clear_error();
    g_verbose = false;
    
    if (g_verbose) {
        printf("导出模块已初始化\n");
    }
    
    return true;
}

// ============================================================================
// 结束标记
// ============================================================================

/* End of result_exporter.c */

