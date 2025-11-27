/**
 * @file design_calibration.h
 * @brief 设计校准模块 - 用于校准设计参数和系统响应
 * @author hany
 * @version 1.0
 * @date 2025
 * 
 * 本模块提供设计参数校准功能，包括：
 * - 几何参数校准（放大倍率、旋转、畸变）
 * - 光学参数校准（焦距、像差）
 * - 探测器响应校准
 * - 系统传递函数测量
 * - 标定板识别和分析
 */

#ifndef DESIGN_CALIBRATION_H
#define DESIGN_CALIBRATION_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// 类型定义和枚举
// ============================================================================

/**
 * @brief 错误码
 */
typedef enum {
    DESIGN_CALIB_SUCCESS = 0,
    DESIGN_CALIB_ERROR_NULL_POINTER,
    DESIGN_CALIB_ERROR_INVALID_PARAM,
    DESIGN_CALIB_ERROR_MEMORY_ALLOCATION,
    DESIGN_CALIB_ERROR_FILE_IO,
    DESIGN_CALIB_ERROR_INVALID_IMAGE,
    DESIGN_CALIB_ERROR_INSUFFICIENT_POINTS,
    DESIGN_CALIB_ERROR_OPTIMIZATION_FAILED,
    DESIGN_CALIB_ERROR_PATTERN_NOT_FOUND,
    DESIGN_CALIB_ERROR_OPERATION_CANCELLED,
    DESIGN_CALIB_ERROR_NOT_CONVERGED
} DesignCalibError;

/**
 * @brief 标定板类型
 */
typedef enum {
    DESIGN_CALIB_PATTERN_CHECKERBOARD,    // 棋盘格
    DESIGN_CALIB_PATTERN_CIRCLES,         // 圆点阵列
    DESIGN_CALIB_PATTERN_ASYMMETRIC_CIRCLES, // 非对称圆点
    DESIGN_CALIB_PATTERN_CHARUCO,         // ChArUco板
    DESIGN_CALIB_PATTERN_CUSTOM           // 自定义图案
} DesignCalibPatternType;

/**
 * @brief 畸变模型类型
 */
typedef enum {
    DESIGN_CALIB_DISTORTION_NONE,         // 无畸变
    DESIGN_CALIB_DISTORTION_RADIAL,       // 径向畸变
    DESIGN_CALIB_DISTORTION_TANGENTIAL,   // 切向畸变
    DESIGN_CALIB_DISTORTION_FULL,         // 完整畸变模型
    DESIGN_CALIB_DISTORTION_FISHEYE       // 鱼眼畸变
} DesignCalibDistortionModel;

/**
 * @brief 优化方法
 */
typedef enum {
    DESIGN_CALIB_OPTIM_LEVENBERG_MARQUARDT,
    DESIGN_CALIB_OPTIM_GAUSS_NEWTON,
    DESIGN_CALIB_OPTIM_GRADIENT_DESCENT,
    DESIGN_CALIB_OPTIM_RANSAC
} DesignCalibOptimMethod;

/**
 * @brief 校准状态
 */
typedef enum {
    DESIGN_CALIB_STATUS_IDLE,
    DESIGN_CALIB_STATUS_DETECTING_PATTERN,
    DESIGN_CALIB_STATUS_EXTRACTING_FEATURES,
    DESIGN_CALIB_STATUS_OPTIMIZING,
    DESIGN_CALIB_STATUS_VALIDATING,
    DESIGN_CALIB_STATUS_COMPLETED,
    DESIGN_CALIB_STATUS_ERROR
} DesignCalibStatus;

// ============================================================================
// 数据结构
// ============================================================================

/**
 * @brief 2D点
 */
typedef struct {
    float x;
    float y;
} DesignCalibPoint2D;

/**
 * @brief 3D点
 */
typedef struct {
    float x;
    float y;
    float z;
} DesignCalibPoint3D;

/**
 * @brief 标定板配置
 */
typedef struct {
    DesignCalibPatternType type;
    int pattern_width;              // 图案宽度（特征点数）
    int pattern_height;             // 图案高度（特征点数）
    float square_size;              // 方格大小（物理单位，如mm）
    float circle_spacing;           // 圆点间距
    bool use_subpixel_refinement;   // 是否使用亚像素精化
    float subpixel_window_size;     // 亚像素窗口大小
} DesignCalibPatternConfig;

/**
 * @brief 检测到的标定板
 */
typedef struct {
    DesignCalibPoint2D *image_points;  // 图像坐标点
    DesignCalibPoint3D *object_points; // 物体坐标点
    int num_points;                     // 点数
    bool is_valid;                      // 是否有效
    float detection_quality;            // 检测质量 [0-100]
    float reprojection_error;           // 重投影误差
} DesignCalibDetectedPattern;

/**
 * @brief 相机内参
 */
typedef struct {
    float fx;                    // 焦距 x
    float fy;                    // 焦距 y
    float cx;                    // 主点 x
    float cy;                    // 主点 y
    float skew;                  // 倾斜系数
    
    // 畸变参数
    float k1, k2, k3;           // 径向畸变系数
    float p1, p2;               // 切向畸变系数
    float k4, k5, k6;           // 高阶径向畸变
    
    int image_width;
    int image_height;
} DesignCalibIntrinsics;

/**
 * @brief 相机外参（位姿）
 */
typedef struct {
    float rotation[9];           // 旋转矩阵 (3x3)
    float translation[3];        // 平移向量
    float rodrigues[3];          // 罗德里格斯向量
    float quaternion[4];         // 四元数表示
} DesignCalibExtrinsics;

/**
 * @brief 立体校准参数
 */
typedef struct {
    DesignCalibIntrinsics left_intrinsics;
    DesignCalibIntrinsics right_intrinsics;
    
    float rotation[9];           // 左到右的旋转
    float translation[3];        // 左到右的平移
    float essential_matrix[9];   // 本质矩阵
    float fundamental_matrix[9]; // 基础矩阵
    
    float baseline;              // 基线距离
    float convergence_angle;     // 会聚角
} DesignCalibStereoParams;

/**
 * @brief 几何校准结果
 */
typedef struct {
    // 放大倍率
    float magnification_x;
    float magnification_y;
    float magnification_error;
    
    // 旋转
    float rotation_angle;        // 度
    float rotation_center_x;
    float rotation_center_y;
    
    // 畸变
    DesignCalibDistortionModel distortion_model;
    float distortion_coeffs[8];
    float max_distortion;        // 最大畸变量（像素）
    
    // 质量指标
    float rms_error;             // RMS误差
    float max_error;             // 最大误差
    int num_calibration_points;
    
    bool is_valid;
} DesignCalibGeometryResult;

/**
 * @brief 光学校准结果
 */
typedef struct {
    // 焦距
    float focal_length;          // mm
    float focal_length_error;
    
    // 工作距离
    float working_distance;      // mm
    float working_distance_error;
    
    // 像差
    float spherical_aberration;
    float chromatic_aberration;
    float astigmatism;
    float field_curvature;
    
    // MTF (Modulation Transfer Function)
    float *mtf_curve;            // MTF曲线
    int mtf_num_points;
    float mtf_cutoff_frequency;  // 截止频率
    
    bool is_valid;
} DesignCalibOpticalResult;

/**
 * @brief 探测器响应校准结果
 */
typedef struct {
    // 增益
    float *gain_map;             // 增益图
    float mean_gain;
    float gain_uniformity;       // 增益均匀性
    
    // 偏移
    float *offset_map;           // 偏移图
    float mean_offset;
    
    // 线性度
    float linearity_error;       // 线性度误差
    float *linearity_curve;
    int linearity_num_points;
    
    // 噪声
    float read_noise;            // 读出噪声
    float dark_current;          // 暗电流
    float *noise_map;            // 噪声图
    
    // 动态范围
    float dynamic_range;         // dB
    
    int width;
    int height;
    bool is_valid;
} DesignCalibDetectorResult;

/**
 * @brief 系统传递函数
 */
typedef struct {
    // 点扩散函数 (PSF)
    float *psf;
    int psf_size;
    float psf_fwhm;              // 半高全宽
    
    // 调制传递函数 (MTF)
    float *mtf_radial;           // 径向MTF
    float *mtf_tangential;       // 切向MTF
    int mtf_num_frequencies;
    float *frequencies;          // 频率点
    
    // 相位传递函数 (PTF)
    float *ptf;
    
    // 光学传递函数 (OTF)
    float *otf_real;
    float *otf_imag;
    
    bool is_valid;
} DesignCalibTransferFunction;

/**
 * @brief 校准配置
 */
typedef struct {
    // 标定板配置
    DesignCalibPatternConfig pattern_config;
    
    // 优化参数
    DesignCalibOptimMethod optim_method;
    int max_iterations;
    float convergence_threshold;
    bool use_robust_estimation;
    float outlier_threshold;
    
    // 畸变模型
    DesignCalibDistortionModel distortion_model;
    bool fix_principal_point;
    bool fix_aspect_ratio;
    bool zero_tangent_dist;
    
    // 质量控制
    float min_detection_quality;
    float max_reprojection_error;
    int min_num_images;
    
    // 多线程
    int num_threads;
    
    // 调试
    bool verbose;
    bool save_debug_images;
    const char *debug_output_dir;
} DesignCalibConfig;

/**
 * @brief 校准上下文
 */
typedef struct DesignCalibContext DesignCalibContext;

/**
 * @brief 进度回调函数
 */
typedef void (*DesignCalibProgressCallback)(
    const char *message,
    int current,
    int total,
    void *user_data);

/**
 * @brief 完整校准结果
 */
typedef struct {
    DesignCalibIntrinsics intrinsics;
    DesignCalibGeometryResult geometry;
    DesignCalibOpticalResult optical;
    DesignCalibDetectorResult detector;
    DesignCalibTransferFunction transfer_function;
    
    // 统计信息
    int num_images_used;
    int num_points_total;
    float overall_rms_error;
    float overall_quality_score;
    
    // 时间戳
    time_t calibration_time;
    
    bool is_valid;
} DesignCalibFullResult;

// ============================================================================
// 核心API函数
// ============================================================================

/**
 * @brief 创建校准上下文
 */
DesignCalibContext* design_calib_create_context(
    const DesignCalibConfig *config);

/**
 * @brief 销毁校准上下文
 */
void design_calib_destroy_context(DesignCalibContext *context);

/**
 * @brief 设置进度回调
 */
void design_calib_set_progress_callback(
    DesignCalibContext *context,
    DesignCalibProgressCallback callback,
    void *user_data);

/**
 * @brief 取消当前操作
 */
void design_calib_cancel(DesignCalibContext *context);

/**
 * @brief 获取当前状态
 */
DesignCalibStatus design_calib_get_status(
    const DesignCalibContext *context);

// ============================================================================
// 标定板检测
// ============================================================================

/**
 * @brief 检测标定板
 */
DesignCalibError design_calib_detect_pattern(
    DesignCalibContext *context,
    const float *image,
    int width,
    int height,
    DesignCalibDetectedPattern **pattern);

/**
 * @brief 批量检测标定板
 */
DesignCalibError design_calib_detect_patterns_batch(
    DesignCalibContext *context,
    const float **images,
    int num_images,
    int width,
    int height,
    DesignCalibDetectedPattern ***patterns,
    int *num_detected);

/**
 * @brief 亚像素精化
 */
DesignCalibError design_calib_refine_corners(
    const float *image,
    int width,
    int height,
    DesignCalibPoint2D *corners,
    int num_corners,
    float window_size);

/**
 * @brief 释放检测到的标定板
 */
void design_calib_free_detected_pattern(
    DesignCalibDetectedPattern *pattern);

// ============================================================================
// 相机校准
// ============================================================================

/**
 * @brief 单目相机校准
 */
DesignCalibError design_calib_calibrate_camera(
    DesignCalibContext *context,
    const DesignCalibDetectedPattern **patterns,
    int num_patterns,
    int image_width,
    int image_height,
    DesignCalibIntrinsics *intrinsics,
    DesignCalibExtrinsics **extrinsics);

/**
 * @brief 立体相机校准
 */
DesignCalibError design_calib_calibrate_stereo(
    DesignCalibContext *context,
    const DesignCalibDetectedPattern **left_patterns,
    const DesignCalibDetectedPattern **right_patterns,
    int num_patterns,
    int image_width,
    int image_height,
    DesignCalibStereoParams *stereo_params);

/**
 * @brief 计算重投影误差
 */
DesignCalibError design_calib_compute_reprojection_error(
    const DesignCalibIntrinsics *intrinsics,
    const DesignCalibExtrinsics *extrinsics,
    const DesignCalibDetectedPattern *pattern,
    float *mean_error,
    float *max_error,
    float **per_point_errors);

// ============================================================================
// 几何校准
// ============================================================================

/**
 * @brief 校准放大倍率
 */
DesignCalibError design_calib_calibrate_magnification(
    DesignCalibContext *context,
    const float *image,
    int width,
    int height,
    float known_distance,  // 已知物理距离
    DesignCalibGeometryResult *result);

/**
 * @brief 校准旋转
 */
DesignCalibError design_calib_calibrate_rotation(
    DesignCalibContext *context,
    const float *image,
    int width,
    int height,
    DesignCalibGeometryResult *result);

/**
 * @brief 校准畸变
 */
DesignCalibError design_calib_calibrate_distortion(
    DesignCalibContext *context,
    const DesignCalibDetectedPattern **patterns,
    int num_patterns,
    int image_width,
    int image_height,
    DesignCalibGeometryResult *result);

/**
 * @brief 完整几何校准
 */
DesignCalibError design_calib_calibrate_geometry(
    DesignCalibContext *context,
    const float **images,
    int num_images,
    int width,
    int height,
    DesignCalibGeometryResult *result);

// ============================================================================
// 光学校准
// ============================================================================

/**
 * @brief 测量焦距
 */
DesignCalibError design_calib_measure_focal_length(
    DesignCalibContext *context,
    const float *image,
    int width,
    int height,
    float known_object_size,
    DesignCalibOpticalResult *result);

/**
 * @brief 测量工作距离
 */
DesignCalibError design_calib_measure_working_distance(
    DesignCalibContext *context,
    const float **images,
    int num_images,
    int width,
    int height,
    const float *known_positions,
    DesignCalibOpticalResult *result);

/**
 * @brief 测量像差
 */
DesignCalibError design_calib_measure_aberrations(
    DesignCalibContext *context,
    const float *image,
    int width,
    int height,
    DesignCalibOpticalResult *result);

/**
 * @brief 测量MTF
 */
DesignCalibError design_calib_measure_mtf(
    DesignCalibContext *context,
    const float *image,
    int width,
    int height,
    DesignCalibOpticalResult *result);

// ============================================================================
// 探测器校准
// ============================================================================

/**
 * @brief 校准探测器增益
 */
DesignCalibError design_calib_calibrate_gain(
    DesignCalibContext *context,
    const float **images,
    const float *exposure_times,
    int num_images,
    int width,
    int height,
    DesignCalibDetectorResult *result);

/**
 * @brief 校准探测器偏移
 */
DesignCalibError design_calib_calibrate_offset(
    DesignCalibContext *context,
    const float **dark_images,
    int num_images,
    int width,
    int height,
    DesignCalibDetectorResult *result);

/**
 * @brief 测量线性度
 */
DesignCalibError design_calib_measure_linearity(
    DesignCalibContext *context,
    const float **images,
    const float *input_levels,
    int num_levels,
    int width,
    int height,
    DesignCalibDetectorResult *result);

/**
 * @brief 测量噪声特性
 */
DesignCalibError design_calib_measure_noise(
    DesignCalibContext *context,
    const float **images,
    int num_images,
    int width,
    int height,
    DesignCalibDetectorResult *result);

/**
 * @brief 完整探测器校准
 */
DesignCalibError design_calib_calibrate_detector(
    DesignCalibContext *context,
    const float **dark_images,
    int num_dark,
    const float **flat_images,
    int num_flat,
    const float **linearity_images,
    const float *linearity_levels,
    int num_linearity,
    int width,
    int height,
    DesignCalibDetectorResult *result);

// ============================================================================
// 系统传递函数
// ============================================================================

/**
 * @brief 测量PSF
 */
DesignCalibError design_calib_measure_psf(
    DesignCalibContext *context,
    const float *image,
    int width,
    int height,
    DesignCalibTransferFunction *result);

/**
 * @brief 计算MTF从PSF
 */
DesignCalibError design_calib_compute_mtf_from_psf(
    const float *psf,
    int psf_size,
    DesignCalibTransferFunction *result);

/**
 * @brief 测量边缘扩散函数 (ESF)
 */
DesignCalibError design_calib_measure_esf(
    DesignCalibContext *context,
    const float *image,
    int width,
    int height,
    float **esf,
    int *esf_length);

/**
 * @brief 从ESF计算LSF和MTF
 */
DesignCalibError design_calib_compute_lsf_mtf_from_esf(
    const float *esf,
    int esf_length,
    float **lsf,
    int *lsf_length,
    DesignCalibTransferFunction *mtf);

// ============================================================================
// 畸变校正
// ============================================================================

/**
 * @brief 应用畸变校正
 */
DesignCalibError design_calib_undistort_image(
    const float *distorted_image,
    int width,
    int height,
    const DesignCalibIntrinsics *intrinsics,
    float **undistorted_image);

/**
 * @brief 应用畸变校正（带插值方法）
 */
DesignCalibError design_calib_undistort_image_interp(
    const float *distorted_image,
    int width,
    int height,
    const DesignCalibIntrinsics *intrinsics,
    int interp_method,  // 0: nearest, 1: bilinear, 2: bicubic
    float **undistorted_image);

/**
 * @brief 计算畸变校正映射
 */
DesignCalibError design_calib_compute_undistort_map(
    int width,
    int height,
    const DesignCalibIntrinsics *intrinsics,
    float **map_x,
    float **map_y);

/**
 * @brief 应用预计算的映射
 */
DesignCalibError design_calib_remap(
    const float *src_image,
    int width,
    int height,
    const float *map_x,
    const float *map_y,
    float **dst_image);

// ============================================================================
// 坐标变换
// ============================================================================

/**
 * @brief 图像坐标到世界坐标
 */
DesignCalibError design_calib_image_to_world(
    const DesignCalibPoint2D *image_point,
    const DesignCalibIntrinsics *intrinsics,
    const DesignCalibExtrinsics *extrinsics,
    float z_world,
    DesignCalibPoint3D *world_point);

/**
 * @brief 世界坐标到图像坐标
 */
DesignCalibError design_calib_world_to_image(
    const DesignCalibPoint3D *world_point,
    const DesignCalibIntrinsics *intrinsics,
    const DesignCalibExtrinsics *extrinsics,
    DesignCalibPoint2D *image_point);

/**
 * @brief 批量坐标变换
 */
DesignCalibError design_calib_transform_points(
    const DesignCalibPoint3D *world_points,
    int num_points,
    const DesignCalibIntrinsics *intrinsics,
    const DesignCalibExtrinsics *extrinsics,
    DesignCalibPoint2D **image_points);

// ============================================================================
// 完整校准流程
// ============================================================================

/**
 * @brief 执行完整校准
 */
DesignCalibError design_calib_full_calibration(
    DesignCalibContext *context,
    const float **calibration_images,
    int num_calib_images,
    const float **dark_images,
    int num_dark_images,
    const float **flat_images,
    int num_flat_images,
    int width,
    int height,
    DesignCalibFullResult *result);

/**
 * @brief 验证校准结果
 */
DesignCalibError design_calib_validate_calibration(
    DesignCalibContext *context,
    const DesignCalibFullResult *result,
    const float **validation_images,
    int num_validation_images,
    int width,
    int height,
    float *validation_score);

// ============================================================================
// 保存和加载
// ============================================================================

/**
 * @brief 保存校准结果
 */
DesignCalibError design_calib_save_result(
    const char *filename,
    const DesignCalibFullResult *result);

/**
 * @brief 加载校准结果
 */
DesignCalibError design_calib_load_result(
    const char *filename,
    DesignCalibFullResult *result);

/**
 * @brief 导出为XML格式
 */
DesignCalibError design_calib_export_to_xml(
    const char *filename,
    const DesignCalibFullResult *result);

/**
 * @brief 导出为JSON格式
 */
DesignCalibError design_calib_export_to_json(
    const char *filename,
    const DesignCalibFullResult *result);

/**
 * @brief 从XML导入
 */
DesignCalibError design_calib_import_from_xml(
    const char *filename,
    DesignCalibFullResult *result);

// ============================================================================
// 内存管理
// ============================================================================

/**
 * @brief 释放几何校准结果
 */
void design_calib_free_geometry_result(
    DesignCalibGeometryResult *result);

/**
 * @brief 释放光学校准结果
 */
void design_calib_free_optical_result(
    DesignCalibOpticalResult *result);

/**
 * @brief 释放探测器校准结果
 */
void design_calib_free_detector_result(
    DesignCalibDetectorResult *result);

/**
 * @brief 释放传递函数结果
 */
void design_calib_free_transfer_function(
    DesignCalibTransferFunction *result);

/**
 * @brief 释放完整校准结果
 */
void design_calib_free_full_result(
    DesignCalibFullResult *result);

// ============================================================================
// 辅助函数
// ============================================================================

/**
 * @brief 获取错误描述
 */
const char* design_calib_error_string(DesignCalibError error);

/**
 * @brief 打印校准结果摘要
 */
void design_calib_print_result_summary(
    const DesignCalibFullResult *result);

/**
 * @brief 计算校准质量分数
 */
float design_calib_compute_quality_score(
    const DesignCalibFullResult *result);

/**
 * @brief 生成校准报告
 */
DesignCalibError design_calib_generate_report(
    const DesignCalibFullResult *result,
    const char *output_filename);

/**
 * @brief 可视化校准结果
 */
DesignCalibError design_calib_visualize_result(
    const DesignCalibFullResult *result,
    const float *sample_image,
    int width,
    int height,
    const char *output_filename);

// ============================================================================
// 高级功能
// ============================================================================

/**
 * @brief 自动选择最佳校准图像
 */
DesignCalibError design_calib_select_best_images(
    DesignCalibContext *context,
    const float **images,
    int num_images,
    int width,
    int height,
    int num_to_select,
    int **selected_indices);

/**
 * @brief 估计校准不确定度
 */
DesignCalibError design_calib_estimate_uncertainty(
    DesignCalibContext *context,
    const DesignCalibFullResult *result,
    const DesignCalibDetectedPattern **patterns,
    int num_patterns,
    float *parameter_uncertainties);

/**
 * @brief 在线校准更新
 */
DesignCalibError design_calib_online_update(
    DesignCalibContext *context,
    DesignCalibFullResult *result,
    const float *new_image,
    int width,
    int height);

/**
 * @brief 多相机联合校准
 */
DesignCalibError design_calib_multi_camera_calibration(
    DesignCalibContext *context,
    const float ***camera_images,  // [num_cameras][num_images]
    int num_cameras,
    int num_images,
    int width,
    int height,
    DesignCalibFullResult **results,
    float **relative_poses);

#ifdef __cplusplus
}
#endif

#endif // DESIGN_CALIBRATION_H
