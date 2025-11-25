/**
 * @file phase_retrieval.h
 * @brief 相位恢复算法头文件
 * @author hany
 * @date 2025
 * 
 * 本文件定义了各种相位恢复算法的接口，包括：
 * - Gerchberg-Saxton (GS) 算法
 * - Hybrid Input-Output (HIO) 算法
 * - Error Reduction (ER) 算法
 * - RAAR (Relaxed Averaged Alternating Reflections) 算法
 * - 混合算法
 * - 多平面相位恢复
 * - 全息重建算法
 */

#ifndef PHASE_RETRIEVAL_H
#define PHASE_RETRIEVAL_H

#include "image.h"
#include "diffraction.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// 常量定义
// ============================================================================

#define RECONSTRUCTION_MAX_ITERATIONS 10000
#define RECONSTRUCTION_MAX_PLANES 10
#define RECONSTRUCTION_DEFAULT_TOLERANCE 1e-6f
#define RECONSTRUCTION_DEFAULT_BETA 0.9f
#define RECONSTRUCTION_DEFAULT_RELAXATION 0.5f

// ============================================================================
// 错误码
// ============================================================================

typedef enum {
    RECONSTRUCTION_SUCCESS = 0,
    RECONSTRUCTION_ERROR_NULL_POINTER = -1,
    RECONSTRUCTION_ERROR_INVALID_PARAMS = -2,
    RECONSTRUCTION_ERROR_MEMORY_ALLOCATION = -3,
    RECONSTRUCTION_ERROR_FFT_FAILED = -4,
    RECONSTRUCTION_ERROR_DIFFRACTION_FAILED = -5,
    RECONSTRUCTION_ERROR_CONVERGENCE = -6,
    RECONSTRUCTION_ERROR_IO = -7,
    RECONSTRUCTION_ERROR_NOT_IMPLEMENTED = -8
} ReconstructionErrorCode;

// ============================================================================
// 算法类型
// ============================================================================

/**
 * @brief 相位恢复算法类型
 */
typedef enum {
    PHASE_RETRIEVAL_GS,          ///< Gerchberg-Saxton算法
    PHASE_RETRIEVAL_HIO,         ///< Hybrid Input-Output算法
    PHASE_RETRIEVAL_ER,          ///< Error Reduction算法
    PHASE_RETRIEVAL_RAAR,        ///< RAAR算法
    PHASE_RETRIEVAL_HYBRID,      ///< 混合算法（自适应）
    PHASE_RETRIEVAL_CUSTOM       ///< 自定义算法
} PhaseRetrievalAlgorithm;

/**
 * @brief 全息类型
 */
typedef enum {
    HOLOGRAPHY_OFF_AXIS,         ///< 离轴全息
    HOLOGRAPHY_INLINE,           ///< 同轴全息
    HOLOGRAPHY_PHASE_SHIFTING,   ///< 相移全息
    HOLOGRAPHY_DIGITAL           ///< 数字全息
} HolographyType;

/**
 * @brief 可视化类型
 */
typedef enum {
    VISUALIZATION_AMPLITUDE,     ///< 振幅
    VISUALIZATION_PHASE,         ///< 相位
    VISUALIZATION_INTENSITY,     ///< 强度
    VISUALIZATION_COMPLEX,       ///< 复数（实部+虚部）
    VISUALIZATION_REAL,          ///< 实部
    VISUALIZATION_IMAGINARY      ///< 虚部
} VisualizationType;

// ============================================================================
// 约束类型
// ============================================================================

/**
 * @brief 约束标志位
 */
typedef enum {
    CONSTRAINT_NONE = 0,
    CONSTRAINT_SUPPORT = 1 << 0,        ///< 支撑域约束
    CONSTRAINT_POSITIVITY = 1 << 1,     ///< 正值约束
    CONSTRAINT_AMPLITUDE = 1 << 2,      ///< 振幅约束
    CONSTRAINT_PHASE = 1 << 3,          ///< 相位约束
    CONSTRAINT_REAL = 1 << 4,           ///< 实值约束
    CONSTRAINT_RANGE = 1 << 5           ///< 值域约束
} ConstraintFlags;

/**
 * @brief 支撑域定义
 */
typedef struct {
    int width;                   ///< 宽度
    int height;                  ///< 高度
    bool *mask;                  ///< 支撑域掩模（true表示在支撑域内）
    float *weights;              ///< 权重（可选，NULL表示均匀权重）
} SupportDomain;

/**
 * @brief 约束集合
 */
typedef struct {
    ConstraintFlags constraint_flags;  ///< 约束标志
    SupportDomain *support;            ///< 支撑域（可选）
    float min_value;                   ///< 最小值（用于值域约束）
    float max_value;                   ///< 最大值（用于值域约束）
    RealImage *amplitude_constraint;   ///< 振幅约束图像（可选）
    RealImage *phase_constraint;       ///< 相位约束图像（可选）
} ConstraintSet;

// ============================================================================
// 参数结构
// ============================================================================

/**
 * @brief 相位恢复参数
 */
typedef struct {
    PhaseRetrievalAlgorithm algorithm;  ///< 算法类型
    int max_iterations;                 ///< 最大迭代次数
    float tolerance;                    ///< 收敛容差
    float beta;                         ///< HIO算法的β参数
    float relaxation;                   ///< RAAR算法的松弛参数
    
    // Shrinkwrap参数
    bool use_shrinkwrap;                ///< 是否使用Shrinkwrap
    int shrinkwrap_interval;            ///< Shrinkwrap更新间隔
    float shrinkwrap_sigma;             ///< 高斯模糊标准差
    float shrinkwrap_threshold;         ///< 阈值（相对于最大值）
    
    // 初始化参数
    bool use_random_phase;              ///< 使用随机相位初始化
    unsigned int random_seed;           ///< 随机种子
    ComplexImage *initial_guess;        ///< 初始猜测（可选）
    
    // 输出控制
    bool verbose;                       ///< 是否输出详细信息
    int print_interval;                 ///< 打印间隔
    bool save_intermediate;             ///< 是否保存中间结果
    int save_interval;                  ///< 保存间隔
    char *output_prefix;                ///< 输出文件前缀
} PhaseRetrievalParams;

/**
 * @brief 全息参数
 */
typedef struct {
    HolographyType type;                ///< 全息类型
    
    // 离轴全息参数
    float carrier_frequency_x;          ///< 载波频率x分量（归一化）
    float carrier_frequency_y;          ///< 载波频率y分量（归一化）
    float filter_size;                  ///< 滤波器大小（相对于图像尺寸）
    
    // 相移全息参数
    int num_phase_steps;                ///< 相移步数
    float *phase_shifts;                ///< 相移值数组
    
    // 参考波
    ComplexImage *reference_wave;       ///< 参考波（可选）
    
    // 重建参数
    bool numerical_refocus;             ///< 是否进行数值重聚焦
    DiffractionParams *diffraction_params;  ///< 衍射参数
    
    // 孪生像处理
    bool remove_twin_image;             ///< 是否去除孪生像
    int twin_removal_iterations;        ///< 孪生像去除迭代次数
} HolographyParams;

/**
 * @brief 多平面相位恢复参数
 */
typedef struct {
    int num_planes;                     ///< 平面数量
    RealImage **measurements;           ///< 各平面的测量强度
    double *distances;                  ///< 各平面的传播距离
    float *plane_weights;               ///< 各平面的权重
    DiffractionParams *diffraction_params;  ///< 衍射参数
    PhaseRetrievalParams base_params;   ///< 基础相位恢复参数
} MultiPlaneParams;

// ============================================================================
// 结果结构
// ============================================================================

/**
 * @brief 重建质量评估
 */
typedef struct {
    float rmse;                         ///< 均方根误差
    float psnr;                         ///< 峰值信噪比
    float ssim;                         ///< 结构相似性
    float phase_error;                  ///< 相位误差
    float amplitude_error;              ///< 振幅误差
} ReconstructionQuality;

/**
 * @brief 重建结果
 */
typedef struct {
    ComplexImage *reconstructed;        ///< 重建的复振幅
    int iterations_performed;           ///< 实际执行的迭代次数
    float final_error;                  ///< 最终误差
    bool converged;                     ///< 是否收敛
    double computation_time;            ///< 计算时间（秒）
    
    // 误差历史
    float *error_history;               ///< 误差历史数组
    int error_history_length;           ///< 误差历史长度
    
    // 质量评估（如果有真值）
    ReconstructionQuality *quality;     ///< 质量评估结果
} ReconstructionResult;

// ============================================================================
// 参数创建和销毁
// ============================================================================

/**
 * @brief 创建默认相位恢复参数
 * @return 参数指针，失败返回NULL
 */
PhaseRetrievalParams* phase_retrieval_params_create_default(void);

/**
 * @brief 销毁相位恢复参数
 * @param params 参数指针
 */
void phase_retrieval_params_destroy(PhaseRetrievalParams *params);

/**
 * @brief 创建默认全息参数
 * @param type 全息类型
 * @return 参数指针，失败返回NULL
 */
HolographyParams* holography_params_create_default(HolographyType type);

/**
 * @brief 销毁全息参数
 * @param params 参数指针
 */
void holography_params_destroy(HolographyParams *params);

/**
 * @brief 创建多平面参数
 * @param num_planes 平面数量
 * @return 参数指针，失败返回NULL
 */
MultiPlaneParams* multiplane_params_create(int num_planes);

/**
 * @brief 销毁多平面参数
 * @param params 参数指针
 */
void multiplane_params_destroy(MultiPlaneParams *params);

// ============================================================================
// 约束相关函数
// ============================================================================

/**
 * @brief 创建支撑域
 * @param width 宽度
 * @param height 高度
 * @return 支撑域指针，失败返回NULL
 */
SupportDomain* support_domain_create(int width, int height);

/**
 * @brief 销毁支撑域
 * @param support 支撑域指针
 */
void support_domain_destroy(SupportDomain *support);

/**
 * @brief 从图像创建支撑域
 * @param image 输入图像
 * @param threshold 阈值（相对于最大值）
 * @return 支撑域指针，失败返回NULL
 */
SupportDomain* support_domain_from_image(const RealImage *image, float threshold);

/**
 * @brief 创建圆形支撑域
 * @param width 宽度
 * @param height 高度
 * @param center_x 中心x坐标
 * @param center_y 中心y坐标
 * @param radius 半径
 * @return 支撑域指针，失败返回NULL
 */
SupportDomain* support_domain_create_circular(
    int width, int height,
    float center_x, float center_y,
    float radius);

/**
 * @brief 创建矩形支撑域
 * @param width 宽度
 * @param height 高度
 * @param x 左上角x坐标
 * @param y 左上角y坐标
 * @param rect_width 矩形宽度
 * @param rect_height 矩形高度
 * @return 支撑域指针，失败返回NULL
 */
SupportDomain* support_domain_create_rectangular(
    int width, int height,
    int x, int y,
    int rect_width, int rect_height);

/**
 * @brief 创建约束集合
 * @return 约束集合指针，失败返回NULL
 */
ConstraintSet* constraint_set_create(void);

/**
 * @brief 销毁约束集合
 * @param constraints 约束集合指针
 */
void constraint_set_destroy(ConstraintSet *constraints);

/**
 * @brief 应用约束到复数场
 * @param field 复数场
 * @param constraints 约束集合
 * @param in_object_plane 是否在物平面（true）或像平面（false）
 * @return 错误码
 */
int apply_constraints(
    ComplexImage *field,
    const ConstraintSet *constraints,
    bool in_object_plane);

/**
 * @brief 更新支撑域（Shrinkwrap算法）
 * @param field 当前复数场
 * @param support 支撑域
 * @param sigma 高斯模糊标准差
 * @param threshold 阈值（相对于最大值）
 * @return 错误码
 */
int update_support_shrinkwrap(
    const ComplexImage *field,
    SupportDomain *support,
    float sigma,
    float threshold);

// ============================================================================
// 结果相关函数
// ============================================================================

/**
 * @brief 创建重建结果
 * @param width 宽度
 * @param height 高度
 * @param max_iterations 最大迭代次数（用于分配误差历史）
 * @return 结果指针，失败返回NULL
 */
ReconstructionResult* reconstruction_result_create(
    int width, int height,
    int max_iterations);

/**
 * @brief 销毁重建结果
 * @param result 结果指针
 */
void reconstruction_result_destroy(ReconstructionResult *result);

// ============================================================================
// 相位恢复算法
// ============================================================================

/**
 * @brief Gerchberg-Saxton算法
 * @param intensity_object 物平面强度
 * @param intensity_image 像平面强度
 * @param params 算法参数
 * @param constraints 约束集合（可选）
 * @param diffraction_params 衍射参数
 * @param result 输出结果
 * @return 错误码
 */
int phase_retrieval_gs(
    const RealImage *intensity_object,
    const RealImage *intensity_image,
    const PhaseRetrievalParams *params,
    const ConstraintSet *constraints,
    const DiffractionParams *diffraction_params,
    ReconstructionResult *result);

/**
 * @brief Hybrid Input-Output算法
 * @param intensity_object 物平面强度
 * @param intensity_image 像平面强度
 * @param params 算法参数
 * @param constraints 约束集合（可选）
 * @param diffraction_params 衍射参数
 * @param result 输出结果
 * @return 错误码
 */
int phase_retrieval_hio(
    const RealImage *intensity_object,
    const RealImage *intensity_image,
    const PhaseRetrievalParams *params,
    const ConstraintSet *constraints,
    const DiffractionParams *diffraction_params,
    ReconstructionResult *result);

/**
 * @brief Error Reduction算法
 * @param intensity_object 物平面强度
 * @param intensity_image 像平面强度
 * @param params 算法参数
 * @param constraints 约束集合（可选）
 * @param diffraction_params 衍射参数
 * @param result 输出结果
 * @return 错误码
 */
int phase_retrieval_er(
    const RealImage *intensity_object,
    const RealImage *intensity_image,
    const PhaseRetrievalParams *params,
    const ConstraintSet *constraints,
    const DiffractionParams *diffraction_params,
    ReconstructionResult *result);

/**
 * @brief RAAR算法
 * @param intensity_object 物平面强度
 * @param intensity_image 像平面强度
 * @param params 算法参数
 * @param constraints 约束集合（可选）
 * @param diffraction_params 衍射参数
 * @param result 输出结果
 * @return 错误码
 */
int phase_retrieval_raar(
    const RealImage *intensity_object,
    const RealImage *intensity_image,
    const PhaseRetrievalParams *params,
    const ConstraintSet *constraints,
    const DiffractionParams *diffraction_params,
    ReconstructionResult *result);

/**
 * @brief 混合相位恢复算法（自适应选择）
 * @param intensity_object 物平面强度
 * @param intensity_image 像平面强度
 * @param params 算法参数
 * @param constraints 约束集合（可选）
 * @param diffraction_params 衍射参数
 * @param result 输出结果
 * @return 错误码
 */
int phase_retrieval_hybrid(
    const RealImage *intensity_object,
    const RealImage *intensity_image,
    const PhaseRetrievalParams *params,
    const ConstraintSet *constraints,
    const DiffractionParams *diffraction_params,
    ReconstructionResult *result);

/**
 * @brief 通用相位恢复接口
 * @param intensity_object 物平面强度
 * @param intensity_image 像平面强度
 * @param params 算法参数
 * @param constraints 约束集合（可选）
 * @param diffraction_params 衍射参数
 * @param result 输出结果
 * @return 错误码
 */
int phase_retrieval(
    const RealImage *intensity_object,
    const RealImage *intensity_image,
    const PhaseRetrievalParams *params,
    const ConstraintSet *constraints,
    const DiffractionParams *diffraction_params,
    ReconstructionResult *result);

/**
 * @brief 多平面相位恢复
 * @param params 多平面参数
 * @param constraints 约束集合（可选）
 * @param result 输出结果
 * @return 错误码
 */
int phase_retrieval_multiplane(
    const MultiPlaneParams *params,
    const ConstraintSet *constraints,
    ReconstructionResult *result);

// ============================================================================
// 全息重建算法
// ============================================================================

/**
 * @brief 离轴全息重建
 * @param hologram 全息图
 * @param params 全息参数
 * @param reconstructed 输出重建结果
 * @return 错误码
 */
int holography_reconstruct_off_axis(
    const RealImage *hologram,
    const HolographyParams *params,
    ComplexImage *reconstructed);

/**
 * @brief 相移全息重建
 * @param holograms 全息图数组
 * @param params 全息参数
 * @param reconstructed 输出重建结果
 * @return 错误码
 */
int holography_reconstruct_phase_shifting(
    const RealImage **holograms,
    const HolographyParams *params,
    ComplexImage *reconstructed);

/**
 * @brief 同轴全息重建
 * @param hologram 全息图
 * @param params 全息参数
 * @param reconstructed 输出重建结果
 * @return 错误码
 */
int holography_reconstruct_inline(
    const RealImage *hologram,
    const HolographyParams *params,
    ComplexImage *reconstructed);

/**
 * @brief 通用全息重建接口
 * @param hologram 全息图
 * @param params 全息参数
 * @param reconstructed 输出重建结果
 * @return 错误码
 */
int holography_reconstruct(
    const RealImage *hologram,
    const HolographyParams *params,
    ComplexImage *reconstructed);

/**
 * @brief 数值重聚焦
 * @param field 输入复数场
 * @param refocus_distance 重聚焦距离
 * @param base_params 基础衍射参数
 * @param refocused 输出重聚焦结果
 * @return 错误码
 */
int numerical_refocus(
    const ComplexImage *field,
    double refocus_distance,
    const DiffractionParams *base_params,
    ComplexImage *refocused);

/**
 * @brief 自动聚焦
 * @param field 输入复数场
 * @param base_params 基础衍射参数
 * @param distance_min 最小距离
 * @param distance_max 最大距离
 * @param num_steps 搜索步数
 * @param best_distance 输出最佳距离
 * @param focused 输出聚焦结果（可选）
 * @return 错误码
 */
int autofocus(
    const ComplexImage *field,
    const DiffractionParams *base_params,
    double distance_min,
    double distance_max,
    int num_steps,
    double *best_distance,
    ComplexImage *focused);

// ============================================================================
// 相位处理
// ============================================================================

/**
 * @brief 二维相位展开
 * @param wrapped_phase 包裹相位
 * @param unwrapped_phase 输出展开相位
 * @return 错误码
 */
int phase_unwrap_2d(
    const RealImage *wrapped_phase,
    RealImage *unwrapped_phase);

/**
 * @brief 从复数场展开相位
 * @param complex_field 复数场
 * @param unwrapped_phase 输出展开相位
 * @return 错误码
 */
int phase_unwrap_from_complex(
    const ComplexImage *complex_field,
    RealImage *unwrapped_phase);

/**
 * @brief 相位滤波
 * @param phase 输入相位
 * @param filtered_phase 输出滤波后的相位
 * @param filter_size 滤波器大小
 * @return 错误码
 */
int phase_filter(
    const RealImage *phase,
    RealImage *filtered_phase,
    int filter_size);

// ============================================================================
// 质量评估
// ============================================================================

/**
 * @brief 评估重建质量
 * @param reconstructed 重建结果
 * @param ground_truth 真值
 * @param quality 输出质量评估
 * @return 错误码
 */
int evaluate_reconstruction_quality(
    const ComplexImage *reconstructed,
    const ComplexImage *ground_truth,
    ReconstructionQuality *quality);

/**
 * @brief 计算聚焦度量
 * @param field 复数场
 * @param metric_type 度量类型（0=梯度方差，1=拉普拉斯方差）
 * @return 聚焦度量值
 */
float compute_focus_metric(
    const ComplexImage *field,
    int metric_type);

// ============================================================================
// 结果保存和可视化
// ============================================================================

/**
 * @brief 保存重建结果
 * @param result 重建结果
 * @param output_prefix 输出文件前缀
 * @return 错误码
 */
int save_reconstruction_result(
    const ReconstructionResult *result,
    const char *output_prefix);

/**
 * @brief 可视化重建结果
 * @param reconstructed 重建的复数场
 * @param visualization 输出可视化图像
 * @param type 可视化类型
 * @return 错误码
 */
int visualize_reconstruction(
    const ComplexImage *reconstructed,
    RealImage *visualization,
    VisualizationType type);

/**
 * @brief 保存误差历史曲线
 * @param error_history 误差历史数组
 * @param length 数组长度
 * @param filename 输出文件名
 * @return 错误码
 */
int save_error_history(
    const float *error_history,
    int length,
    const char *filename);

// ============================================================================
// 辅助函数
// ============================================================================

/**
 * @brief 验证相位恢复参数
 * @param params 参数
 * @return 错误码
 */
int reconstruction_validate_params(const PhaseRetrievalParams *params);

/**
 * @brief 初始化随机相位
 * @param field 复数场
 * @param amplitude 振幅图像（可选）
 * @return 错误码
 */
int initialize_random_phase(
    ComplexImage *field,
    const RealImage *amplitude);

/**
 * @brief 计算RMS误差
 * @param field1 复数场1
 * @param field2 复数场2
 * @return RMS误差
 */
float compute_rms_error(
    const ComplexImage *field1,
    const ComplexImage *field2);

/**
 * @brief 归一化复数场
 * @param field 复数场
 * @param method 归一化方法（0=最大值，1=能量）
 * @return 错误码
 */
int normalize_complex_field(
    ComplexImage *field,
    int method);

/**
 * @brief 获取错误信息字符串
 * @param error_code 错误码
 * @return 错误信息字符串
 */
const char* reconstruction_get_error_string(int error_code);

/**
 * @brief 打印重建统计信息
 * @param result 重建结果
 */
void print_reconstruction_statistics(const ReconstructionResult *result);

// ============================================================================
// 高级功能
// ============================================================================

/**
 * @brief 自适应参数调整
 * @param params 参数
 * @param current_error 当前误差
 * @param iteration 当前迭代次数
 * @return 错误码
 */
int adaptive_parameter_adjustment(
    PhaseRetrievalParams *params,
    float current_error,
    int iteration);

/**
 * @brief 多尺度相位恢复
 * @param intensity_object 物平面强度
 * @param intensity_image 像平面强度
 * @param params 算法参数
 * @param num_scales 尺度数量
 * @param constraints 约束集合（可选）
 * @param diffraction_params 衍射参数
 * @param result 输出结果
 * @return 错误码
 */
int phase_retrieval_multiscale(
    const RealImage *intensity_object,
    const RealImage *intensity_image,
    const PhaseRetrievalParams *params,
    int num_scales,
    const ConstraintSet *constraints,
    const DiffractionParams *diffraction_params,
    ReconstructionResult *result);

/**
 * @brief 并行相位恢复（多线程）
 * @param intensity_object 物平面强度
 * @param intensity_image 像平面强度
 * @param params 算法参数
 * @param constraints 约束集合（可选）
 * @param diffraction_params 衍射参数
 * @param num_threads 线程数
 * @param result 输出结果
 * @return 错误码
 */
int phase_retrieval_parallel(
    const RealImage *intensity_object,
    const RealImage *intensity_image,
    const PhaseRetrievalParams *params,
    const ConstraintSet *constraints,
    const DiffractionParams *diffraction_params,
    int num_threads,
    ReconstructionResult *result);

#ifdef __cplusplus
}
#endif

#endif // PHASE_RETRIEVAL_H
