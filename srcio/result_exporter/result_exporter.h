/**
 * @file result_exporter.h
 * @brief 结果导出模块 - 将检测结果导出为各种格式
 * @author hany
 * @date 2025
 */

#ifndef RESULT_EXPORTER_H
#define RESULT_EXPORTER_H

#include <stdio.h>
#include <stdbool.h>
#include <time.h>
#include "detector.h"
#include "png_handler.h"

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// 导出格式枚举
// ============================================================================

/**
 * @brief 导出格式类型
 */
typedef enum {
    EXPORT_FORMAT_JSON,      /**< JSON格式 */
    EXPORT_FORMAT_XML,       /**< XML格式 */
    EXPORT_FORMAT_CSV,       /**< CSV格式 */
    EXPORT_FORMAT_TXT,       /**< 纯文本格式 */
    EXPORT_FORMAT_HTML,      /**< HTML格式 */
    EXPORT_FORMAT_MARKDOWN,  /**< Markdown格式 */
    EXPORT_FORMAT_YAML,      /**< YAML格式 */
    EXPORT_FORMAT_LATEX      /**< LaTeX格式 */
} ExportFormat;

/**
 * @brief 图像导出格式
 */
typedef enum {
    IMAGE_EXPORT_ANNOTATED,  /**< 带标注的图像 */
    IMAGE_EXPORT_HEATMAP,    /**< 热力图 */
    IMAGE_EXPORT_OVERLAY,    /**< 叠加层 */
    IMAGE_EXPORT_SIDEBYSIDE, /**< 并排对比 */
    IMAGE_EXPORT_GRID        /**< 网格布局 */
} ImageExportFormat;

// ============================================================================
// 导出选项结构
// ============================================================================

/**
 * @brief 导出选项配置
 */
typedef struct {
    bool include_metadata;      /**< 包含元数据 */
    bool include_statistics;    /**< 包含统计信息 */
    bool include_confidence;    /**< 包含置信度 */
    bool include_coordinates;   /**< 包含坐标信息 */
    bool include_timestamp;     /**< 包含时间戳 */
    bool pretty_print;          /**< 格式化输出（用于JSON/XML） */
    bool use_relative_paths;    /**< 使用相对路径 */
    int precision;              /**< 浮点数精度 */
    const char *encoding;       /**< 字符编码（默认UTF-8） */
} ExportOptions;

/**
 * @brief 图像导出选项
 */
typedef struct {
    bool draw_boxes;            /**< 绘制边界框 */
    bool draw_labels;           /**< 绘制标签 */
    bool draw_confidence;       /**< 绘制置信度 */
    bool draw_grid;             /**< 绘制网格 */
    int box_thickness;          /**< 边界框线条粗细 */
    int font_size;              /**< 字体大小 */
    uint8_t box_color[4];       /**< 边界框颜色 (RGBA) */
    uint8_t text_color[4];      /**< 文本颜色 (RGBA) */
    uint8_t bg_color[4];        /**< 背景颜色 (RGBA) */
    float opacity;              /**< 透明度 */
} ImageExportOptions;

/**
 * @brief 报告配置
 */
typedef struct {
    const char *title;          /**< 报告标题 */
    const char *author;         /**< 作者 */
    const char *description;    /**< 描述 */
    const char *version;        /**< 版本 */
    bool include_summary;       /**< 包含摘要 */
    bool include_details;       /**< 包含详细信息 */
    bool include_images;        /**< 包含图像 */
    bool include_charts;        /**< 包含图表 */
} ReportConfig;

// ============================================================================
// 导出结果结构
// ============================================================================

/**
 * @brief 导出统计信息
 */
typedef struct {
    int total_detections;       /**< 总检测数 */
    int true_positives;         /**< 真阳性 */
    int false_positives;        /**< 假阳性 */
    int false_negatives;        /**< 假阴性 */
    double precision;           /**< 精确率 */
    double recall;              /**< 召回率 */
    double f1_score;            /**< F1分数 */
    double average_confidence;  /**< 平均置信度 */
    double processing_time;     /**< 处理时间（秒） */
} ExportStatistics;

/**
 * @brief 导出元数据
 */
typedef struct {
    const char *source_file;    /**< 源文件路径 */
    const char *detector_name;  /**< 检测器名称 */
    const char *detector_version; /**< 检测器版本 */
    time_t timestamp;           /**< 时间戳 */
    int image_width;            /**< 图像宽度 */
    int image_height;           /**< 图像高度 */
    const char *color_space;    /**< 颜色空间 */
    int bit_depth;              /**< 位深度 */
} ExportMetadata;

// ============================================================================
// 核心导出函数
// ============================================================================

/**
 * @brief 初始化导出器
 * @return 成功返回true，失败返回false
 */
bool exporter_init(void);

/**
 * @brief 清理导出器
 */
void exporter_cleanup(void);

/**
 * @brief 创建默认导出选项
 * @return 默认导出选项
 */
ExportOptions exporter_default_options(void);

/**
 * @brief 创建默认图像导出选项
 * @return 默认图像导出选项
 */
ImageExportOptions exporter_default_image_options(void);

/**
 * @brief 创建默认报告配置
 * @return 默认报告配置
 */
ReportConfig exporter_default_report_config(void);

// ============================================================================
// 文本格式导出
// ============================================================================

/**
 * @brief 导出检测结果为JSON格式
 * @param results 检测结果数组
 * @param count 结果数量
 * @param output_file 输出文件路径
 * @param options 导出选项（可为NULL使用默认值）
 * @return 成功返回true，失败返回false
 */
bool export_to_json(const DetectionResult *results, int count,
                   const char *output_file,
                   const ExportOptions *options);

/**
 * @brief 导出检测结果为XML格式
 * @param results 检测结果数组
 * @param count 结果数量
 * @param output_file 输出文件路径
 * @param options 导出选项
 * @return 成功返回true，失败返回false
 */
bool export_to_xml(const DetectionResult *results, int count,
                  const char *output_file,
                  const ExportOptions *options);

/**
 * @brief 导出检测结果为CSV格式
 * @param results 检测结果数组
 * @param count 结果数量
 * @param output_file 输出文件路径
 * @param options 导出选项
 * @return 成功返回true，失败返回false
 */
bool export_to_csv(const DetectionResult *results, int count,
                  const char *output_file,
                  const ExportOptions *options);

/**
 * @brief 导出检测结果为纯文本格式
 * @param results 检测结果数组
 * @param count 结果数量
 * @param output_file 输出文件路径
 * @param options 导出选项
 * @return 成功返回true，失败返回false
 */
bool export_to_text(const DetectionResult *results, int count,
                   const char *output_file,
                   const ExportOptions *options);

/**
 * @brief 导出检测结果为YAML格式
 * @param results 检测结果数组
 * @param count 结果数量
 * @param output_file 输出文件路径
 * @param options 导出选项
 * @return 成功返回true，失败返回false
 */
bool export_to_yaml(const DetectionResult *results, int count,
                   const char *output_file,
                   const ExportOptions *options);

// ============================================================================
// 报告生成
// ============================================================================

/**
 * @brief 生成HTML报告
 * @param results 检测结果数组
 * @param count 结果数量
 * @param output_file 输出文件路径
 * @param config 报告配置
 * @param options 导出选项
 * @return 成功返回true，失败返回false
 */
bool export_html_report(const DetectionResult *results, int count,
                       const char *output_file,
                       const ReportConfig *config,
                       const ExportOptions *options);

/**
 * @brief 生成Markdown报告
 * @param results 检测结果数组
 * @param count 结果数量
 * @param output_file 输出文件路径
 * @param config 报告配置
 * @param options 导出选项
 * @return 成功返回true，失败返回false
 */
bool export_markdown_report(const DetectionResult *results, int count,
                           const char *output_file,
                           const ReportConfig *config,
                           const ExportOptions *options);

/**
 * @brief 生成LaTeX报告
 * @param results 检测结果数组
 * @param count 结果数量
 * @param output_file 输出文件路径
 * @param config 报告配置
 * @param options 导出选项
 * @return 成功返回true，失败返回false
 */
bool export_latex_report(const DetectionResult *results, int count,
                        const char *output_file,
                        const ReportConfig *config,
                        const ExportOptions *options);

// ============================================================================
// 图像导出
// ============================================================================

/**
 * @brief 导出带标注的图像
 * @param image 原始图像
 * @param results 检测结果数组
 * @param count 结果数量
 * @param output_file 输出文件路径
 * @param options 图像导出选项
 * @return 成功返回true，失败返回false
 */
bool export_annotated_image(const Image *image,
                           const DetectionResult *results, int count,
                           const char *output_file,
                           const ImageExportOptions *options);

/**
 * @brief 导出热力图
 * @param image 原始图像
 * @param results 检测结果数组
 * @param count 结果数量
 * @param output_file 输出文件路径
 * @param options 图像导出选项
 * @return 成功返回true，失败返回false
 */
bool export_heatmap(const Image *image,
                   const DetectionResult *results, int count,
                   const char *output_file,
                   const ImageExportOptions *options);

/**
 * @brief 导出对比图（原图和标注图并排）
 * @param image 原始图像
 * @param results 检测结果数组
 * @param count 结果数量
 * @param output_file 输出文件路径
 * @param options 图像导出选项
 * @return 成功返回true，失败返回false
 */
bool export_comparison_image(const Image *image,
                            const DetectionResult *results, int count,
                            const char *output_file,
                            const ImageExportOptions *options);

/**
 * @brief 导出网格布局图像（多个检测区域）
 * @param image 原始图像
 * @param results 检测结果数组
 * @param count 结果数量
 * @param output_file 输出文件路径
 * @param grid_cols 网格列数
 * @param options 图像导出选项
 * @return 成功返回true，失败返回false
 */
bool export_grid_image(const Image *image,
                      const DetectionResult *results, int count,
                      const char *output_file,
                      int grid_cols,
                      const ImageExportOptions *options);

// ============================================================================
// 批量导出
// ============================================================================

/**
 * @brief 批量导出结果
 * @param results 检测结果数组
 * @param count 结果数量
 * @param output_dir 输出目录
 * @param formats 导出格式数组
 * @param format_count 格式数量
 * @param options 导出选项
 * @return 成功返回true，失败返回false
 */
bool export_batch(const DetectionResult *results, int count,
                 const char *output_dir,
                 const ExportFormat *formats, int format_count,
                 const ExportOptions *options);

/**
 * @brief 批量导出图像
 * @param images 图像数组
 * @param results 检测结果数组（每个图像对应一个结果数组）
 * @param counts 每个图像的结果数量
 * @param image_count 图像数量
 * @param output_dir 输出目录
 * @param format 图像导出格式
 * @param options 图像导出选项
 * @return 成功返回true，失败返回false
 */
bool export_batch_images(const Image **images,
                        const DetectionResult **results,
                        const int *counts,
                        int image_count,
                        const char *output_dir,
                        ImageExportFormat format,
                        const ImageExportOptions *options);

// ============================================================================
// 统计和元数据
// ============================================================================

/**
 * @brief 计算导出统计信息
 * @param results 检测结果数组
 * @param count 结果数量
 * @param stats 输出统计信息
 * @return 成功返回true，失败返回false
 */
bool export_calculate_statistics(const DetectionResult *results, int count,
                                 ExportStatistics *stats);

/**
 * @brief 创建导出元数据
 * @param image 图像
 * @param detector_name 检测器名称
 * @param detector_version 检测器版本
 * @param metadata 输出元数据
 * @return 成功返回true，失败返回false
 */
bool export_create_metadata(const Image *image,
                           const char *detector_name,
                           const char *detector_version,
                           ExportMetadata *metadata);

/**
 * @brief 导出统计信息到文件
 * @param stats 统计信息
 * @param output_file 输出文件路径
 * @param format 导出格式
 * @return 成功返回true，失败返回false
 */
bool export_statistics(const ExportStatistics *stats,
                      const char *output_file,
                      ExportFormat format);

/**
 * @brief 导出元数据到文件
 * @param metadata 元数据
 * @param output_file 输出文件路径
 * @param format 导出格式
 * @return 成功返回true，失败返回false
 */
bool export_metadata(const ExportMetadata *metadata,
                    const char *output_file,
                    ExportFormat format);

// ============================================================================
// 辅助函数
// ============================================================================

/**
 * @brief 获取导出格式的文件扩展名
 * @param format 导出格式
 * @return 文件扩展名（不含点）
 */
const char* export_get_extension(ExportFormat format);

/**
 * @brief 从文件扩展名获取导出格式
 * @param extension 文件扩展名
 * @return 导出格式，如果无法识别返回EXPORT_FORMAT_JSON
 */
ExportFormat export_format_from_extension(const char *extension);

/**
 * @brief 验证输出路径
 * @param output_path 输出路径
 * @param create_dirs 如果目录不存在是否创建
 * @return 成功返回true，失败返回false
 */
bool export_validate_path(const char *output_path, bool create_dirs);

/**
 * @brief 获取最后的错误信息
 * @return 错误信息字符串
 */
const char* export_get_error(void);

/**
 * @brief 清除错误信息
 */
void export_clear_error(void);

/**
 * @brief 设置详细输出模式
 * @param verbose 是否启用详细输出
 */
void export_set_verbose(bool verbose);

/**
 * @brief 获取导出器版本
 * @return 版本字符串
 */
const char* export_get_version(void);

#ifdef __cplusplus
}
#endif

#endif // RESULT_EXPORTER_H
