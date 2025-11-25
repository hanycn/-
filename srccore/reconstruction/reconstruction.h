/**
 * @file reconstruction.h
 * @brief 全息重建和相位恢复算法库
 * @author hany
 * @version
 * @date 2025
 * 
 * 本头文件定义了全息重建和相位恢复的相关函数和数据结构。
 * 主要功能包括：
 * - 数字全息重建
 * - 相位恢复算法（GS, HIO, ER等）
 * - 多平面相位恢复
 * - 迭代重建算法
 * - 约束优化方法
 */

#ifndef RECONSTRUCTION_H
#define RECONSTRUCTION_H

#include "diffraction_model.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// 常量定义
// ============================================================================

#define RECONSTRUCTION_MAX_ITERATIONS 10000
#define RECONSTRUCTION_DEFAULT_TOLERANCE 1e-6
#define RECONSTRUCTION_MAX_PLANES 10

// ============================================================================
// 错误码定义
// ============================================================================

typedef enum {
    RECONSTRUCTION_SUCCESS = 0,
    RECONSTRUCTION_ERROR_NULL_POINTER = -1,
    RECONSTRUCTION_ERROR_INVALID_PARAMS = -2,
    RECONSTRUCTION_ERROR_MEMORY_ALLOCATION = -3,
    RECONSTRUCTION_ERROR_NOT_CONVERGED = -4,
    RECONSTRUCTION_ERROR_INVALID_CONSTRAINT = -5,
    RECONSTRUCTION_ERROR_DIFFRACTION_FAILED = -6,
    RECONSTRUCTION_ERROR_MAX_ITERATIONS = -7
} ReconstructionError;

// ============================================================================
// 重建算法类型
// ============================================================================

/**
 * @brief 相位恢复算法类型
 */
typedef enum {
    PHASE_RETRIEVAL_GS,          ///< Gerchberg-Saxton算法
    PHASE_RETRIEVAL_HIO,         ///< Hybrid Input-Output算法
    PHASE_RETRIEVAL_ER,          ///< Error Reduction算法
    PHASE_RETRIEVAL_RAAR,        ///< Relaxed Averaged Alternating Reflections
    PHASE_RETRIEVAL_OSS,         ///< Oversampling Smoothness算法
    PHASE_RETRIEVAL_HYBRID       ///< 混合算法（自适应切换）
} PhaseRetrievalAlgorithm;

/**
 * @brief 全息重建类型
 */
typedef enum {
    HOLOGRAPHY_OFF_AXIS,         ///< 离轴全息
    HOLOGRAPHY_IN_LINE,          ///< 同轴全息
    HOLOGRAPHY_PHASE_SHIFTING,   ///< 相移全息
    HOLOGRAPHY_DIGITAL           ///< 数字全息
} HolographyType;

/**
 * @brief 约束类型
 */
typedef enum {
    CONSTRAINT_NONE = 0,
    CONSTRAINT_SUPPORT = 1 << 0,      ///< 支撑域约束
    CONSTRAINT_POSITIVITY = 1 << 1,   ///< 正值约束
    CONSTRAINT_AMPLITUDE = 1 << 2,    ///< 振幅约束
    CONSTRAINT_PHASE = 1 << 3,        ///< 相位约束
    CONSTRAINT_TOTAL_VARIATION = 1 << 4  ///< 全变分约束
} ConstraintType;

// ============================================================================
// 数据结构定义
// ============================================================================

/**
 * @brief 支撑域定义
 */
typedef struct {
    int width;                    ///< 宽度
    int height;                   ///< 高度
    bool *mask;                   ///< 支撑域掩模（true表示在支撑域内）
    bool auto_update;             ///< 是否自动更新支撑域
    float threshold;              ///< 自动更新阈值
} SupportConstraint;

/**
 * @brief 振幅约束
 */
typedef struct {
    float *amplitude;             ///< 已知振幅
    float *weight;                ///< 约束权重（0-1）
    bool enforce_exact;           ///< 是否强制精确匹配
} AmplitudeConstraint;

/**
 * @brief 相位约束
 */
typedef struct {
    float *phase;                 ///< 已知相位
    float *weight;                ///< 约束权重（0-1）
    bool wrap_phase;              ///< 是否进行相位包裹
} PhaseConstraint;

/**
 * @brief 约束集合
 */
typedef struct {
    unsigned int constraint_flags;        ///< 约束类型标志位
    SupportConstraint *support;           ///< 支撑域约束
    AmplitudeConstraint *amplitude;       ///< 振幅约束
    PhaseConstraint *phase;               ///< 相位约束
    float positivity_weight;              ///< 正值约束权重
    float tv_weight;                      ///< 全变分约束权重
} ConstraintSet;

/**
 * @brief 相位恢复参数
 */
typedef struct {
    PhaseRetrievalAlgorithm algorithm;    ///< 算法类型
    int max_iterations;                   ///< 最大迭代次数
    float tolerance;                      ///< 收敛容差
    float beta;                           ///< HIO算法参数（通常0.5-0.9）
    float relaxation;                     ///< RAAR算法松弛参数
    
    bool use_shrinkwrap;                  ///< 是否使用shrinkwrap更新支撑域
    int shrinkwrap_interval;              ///< shrinkwrap更新间隔
    float shrinkwrap_sigma;               ///< shrinkwrap高斯模糊sigma
    float shrinkwrap_threshold;           ///< shrinkwrap阈值
    
    bool use_averaging;                   ///< 是否使用平均
    int averaging_window;                 ///< 平均窗口大小
    
    bool verbose;                         ///< 是否输出详细信息
    int print_interval;                   ///< 打印间隔
} PhaseRetrievalParams;

/**
 * @brief 多平面相位恢复参数
 */
typedef struct {
    int num_planes;                       ///< 平面数量
    double *distances;                    ///< 各平面距离
    ComplexImage **measurements;          ///< 各平面测量值
    float *plane_weights;                 ///< 各平面权重
    
    PhaseRetrievalParams base_params;     ///< 基础相位恢复参数
    DiffractionParams *diffraction_params; ///< 衍射参数
} MultiPlaneParams;

/**
 * @brief 全息重建参数
 */
typedef struct {
    HolographyType type;                  ///< 全息类型
    
    // 离轴全息参数
    float carrier_frequency_x;            ///< 载频x分量
    float carrier_frequency_y;            ///< 载频y分量
    float filter_size;                    ///< 滤波器尺寸
    
    // 相移全息参数
    int num_phase_steps;                  ///< 相移步数
    float *phase_shifts;                  ///< 相移值数组
    
    // 通用参数
    ComplexImage *reference_wave;         ///< 参考光波
    bool remove_twin_image;               ///< 是否去除孪生像
    bool numerical_refocus;               ///< 是否数值重聚焦
    
    DiffractionParams *diffraction_params; ///< 衍射参数
} HolographyParams;

/**
 * @brief 重建结果
 */
typedef struct {
    ComplexImage *reconstructed;          ///< 重建的复数场
    int iterations_performed;             ///< 实际迭代次数
    float final_error;                    ///< 最终误差
    float *error_history;                 ///< 误差历史
    int error_history_length;             ///< 误差历史长度
    bool converged;                       ///< 是否收敛
    double computation_time;              ///< 计算时间（秒）
} ReconstructionResult;

// ============================================================================
// 相位恢复算法
// ============================================================================

/**
 * @brief 创建相位恢复参数（使用默认值）
 * @return 相位恢复参数指针
 */
PhaseRetrievalParams* phase_retrieval_params_create_default(void);

/**
 * @brief 销毁相位恢复参数
 * @param params 参数指针
 */
void phase_retrieval_params_destroy(PhaseRetrievalParams *params);

/**
 * @brief Gerchberg-Saxton算法
 * @param intensity_object 物平面强度
 * @param intensity_image 像平面强度
 * @param params 相位恢复参数
 * @param constraints 约束集合
 * @param diffraction_params 衍射参数
 * @param result 重建结果
 * @return 错误码
 */
int phase_retrieval_gs(
    const RealImage *intensity_object,
    const RealImage *intensity_image,
    const PhaseRetrievalParams *params,
    const ConstraintSet *constraints,
    const DiffractionParams *diffraction_params,
    ReconstructionResult *result
);

/**
 * @brief Hybrid Input-Output算法
 * @param intensity_object 物平面强度
 * @param intensity_image 像平面强度
 * @param params 相位恢复参数
 * @param constraints 约束集合
 * @param diffraction_params 衍射参数
 * @param result 重建结果
 * @return 错误码
 */
int phase_retrieval_hio(
    const RealImage *intensity_object,
    const RealImage *intensity_image,
    const PhaseRetrievalParams *params,
    const ConstraintSet *constraints,
    const DiffractionParams *diffraction_params,
    ReconstructionResult *result
);

/**
 * @brief Error Reduction算法
 * @param intensity_object 物平面强度
 * @param intensity_image 像平面强度
 * @param params 相位恢复参数
 * @param constraints 约束集合
 * @param diffraction_params 衍射参数
 * @param result 重建结果
 * @return 错误码
 */
int phase_retrieval_er(
    const RealImage *intensity_object,
    const RealImage *intensity_image,
    const PhaseRetrievalParams *params,
    const ConstraintSet *constraints,
    const DiffractionParams *diffraction_params,
    ReconstructionResult *result
);

/**
 * @brief RAAR算法
 * @param intensity_object 物平面强度
 * @param intensity_image 像平面强度
 * @param params 相位恢复参数
 * @param constraints 约束集合
 * @param diffraction_params 衍射参数
 * @param result 重建结果
 * @return 错误码
 */
int phase_retrieval_raar(
    const RealImage *intensity_object,
    const RealImage *intensity_image,
    const PhaseRetrievalParams *params,
    const ConstraintSet *constraints,
    const DiffractionParams *diffraction_params,
    ReconstructionResult *result
);

/**
 * @brief 混合相位恢复算法（自适应选择）
 * @param intensity_object 物平面强度
 * @param intensity_image 像平面强度
 * @param params 相位恢复参数
 * @param constraints 约束集合
 * @param diffraction_params 衍射参数
 * @param result 重建结果
 * @return 错误码
 */
int phase_retrieval_hybrid(
    const RealImage *intensity_object,
    const RealImage *intensity_image,
    const PhaseRetrievalParams *params,
    const ConstraintSet *constraints,
    const DiffractionParams *diffraction_params,
    ReconstructionResult *result
);

/**
 * @brief 通用相位恢复接口
 * @param intensity_object 物平面强度
 * @param intensity_image 像平面强度
 * @param params 相位恢复参数
 * @param constraints 约束集合
 * @param diffraction_params 衍射参数
 * @param result 重建结果
 * @return 错误码
 */
int phase_retrieval(
    const RealImage *intensity_object,
    const RealImage *intensity_image,
    const PhaseRetrievalParams *params,
    const ConstraintSet *constraints,
    const DiffractionParams *diffraction_params,
    ReconstructionResult *result
);

// ============================================================================
// 多平面相位恢复
// ============================================================================

/**
 * @brief 创建多平面参数
 * @param num_planes 平面数量
 * @return 多平面参数指针
 */
MultiPlaneParams* multiplane_params_create(int num_planes);

/**
 * @brief 销毁多平面参数
 * @param params 参数指针
 */
void multiplane_params_destroy(MultiPlaneParams *params);

/**
 * @brief 多平面相位恢复
 * @param params 多平面参数
 * @param constraints 约束集合
 * @param result 重建结果
 * @return 错误码
 */
int phase_retrieval_multiplane(
    const MultiPlaneParams *params,
    const ConstraintSet *constraints,
    ReconstructionResult *result
);

// ============================================================================
// 全息重建
// ============================================================================

/**
 * @brief 创建全息重建参数
 * @param type 全息类型
 * @return 全息参数指针
 */
HolographyParams* holography_params_create(HolographyType type);

/**
 * @brief 销毁全息重建参数
 * @param params 参数指针
 */
void holography_params_destroy(HolographyParams *params);

/**
 * @brief 离轴全息重建
 * @param hologram 全息图
 * @param params 全息参数
 * @param result 重建结果
 * @return 错误码
 */
int holography_reconstruct_off_axis(
    const RealImage *hologram,
    const HolographyParams *params,
    ReconstructionResult *result
);

/**
 * @brief 同轴全息重建
 * @param hologram 全息图
 * @param params 全息参数
 * @param result 重建结果
 * @return 错误码
 */
int holography_reconstruct_in_line(
    const RealImage *hologram,
    const HolographyParams *params,
    ReconstructionResult *result
);

/**
 * @brief 相移全息重建
 * @param holograms 多幅相移全息图
 * @param params 全息参数
 * @param result 重建结果
 * @return 错误码
 */
int holography_reconstruct_phase_shifting(
    const RealImage **holograms,
    const HolographyParams *params,
    ReconstructionResult *result
);

/**
 * @brief 通用全息重建接口
 * @param holograms 全息图数组
 * @param num_holograms 全息图数量
 * @param params 全息参数
 * @param result 重建结果
 * @return 错误码
 */
int holography_reconstruct(
    const RealImage **holograms,
    int num_holograms,
    const HolographyParams *params,
    ReconstructionResult *result
);

/**
 * @brief 数值重聚焦
 * @param hologram 全息图
 * @param focus_distance 聚焦距离
 * @param params 全息参数
 * @param focused_image 聚焦后的图像
 * @return 错误码
 */
int holography_numerical_refocus(
    const ComplexImage *hologram,
    double focus_distance,
    const HolographyParams *params,
    ComplexImage *focused_image
);

// ============================================================================
// 约束管理
// ============================================================================

/**
 * @brief 创建约束集合
 * @return 约束集合指针
 */
ConstraintSet* constraint_set_create(void);

/**
 * @brief 销毁约束集合
 * @param constraints 约束集合指针
 */
void constraint_set_destroy(ConstraintSet *constraints);

/**
 * @brief 创建支撑域约束
 * @param width 宽度
 * @param height 高度
 * @return 支撑域约束指针
 */
SupportConstraint* support_constraint_create(int width, int height);

/**
 * @brief 从图像创建支撑域
 * @param image 输入图像
 * @param threshold 阈值
 * @return 支撑域约束指针
 */
SupportConstraint* support_constraint_from_image(
    const RealImage *image,
    float threshold
);

/**
 * @brief 销毁支撑域约束
 * @param support 支撑域约束指针
 */
void support_constraint_destroy(SupportConstraint *support);

/**
 * @brief 创建振幅约束
 * @param width 宽度
 * @param height 高度
 * @return 振幅约束指针
 */
AmplitudeConstraint* amplitude_constraint_create(int width, int height);

/**
 * @brief 销毁振幅约束
 * @param amplitude 振幅约束指针
 */
void amplitude_constraint_destroy(AmplitudeConstraint *amplitude);

/**
 * @brief 创建相位约束
 * @param width 宽度
 * @param height 高度
 * @return 相位约束指针
 */
PhaseConstraint* phase_constraint_create(int width, int height);

/**
 * @brief 销毁相位约束
 * @param phase 相位约束指针
 */
void phase_constraint_destroy(PhaseConstraint *phase);

/**
 * @brief 应用约束到复数场
 * @param field 复数场
 * @param constraints 约束集合
 * @param in_object_plane 是否在物平面
 * @return 错误码
 */
int apply_constraints(
    ComplexImage *field,
    const ConstraintSet *constraints,
    bool in_object_plane
);

// ============================================================================
// 重建结果管理
// ============================================================================

/**
 * @brief 创建重建结果
 * @param width 宽度
 * @param height 高度
 * @param max_iterations 最大迭代次数
 * @return 重建结果指针
 */
ReconstructionResult* reconstruction_result_create(
    int width,
    int height,
    int max_iterations
);

/**
 * @brief 销毁重建结果
 * @param result 重建结果指针
 */
void reconstruction_result_destroy(ReconstructionResult *result);

// ============================================================================
// 误差计算
// ============================================================================

/**
 * @brief 计算重建误差
 * @param measured 测量值
 * @param reconstructed 重建值
 * @param error 输出误差
 * @return 错误码
 */
int compute_reconstruction_error(
    const ComplexImage *measured,
    const ComplexImage *reconstructed,
    float *error
);

/**
 * @brief 计算强度误差
 * @param measured_intensity 测量强度
 * @param reconstructed 重建复数场
 * @param error 输出误差
 * @return 错误码
 */
int compute_intensity_error(
    const RealImage *measured_intensity,
    const ComplexImage *reconstructed,
    float *error
);

// ============================================================================
// 辅助函数
// ============================================================================

/**
 * @brief 更新支撑域（shrinkwrap）
 * @param field 当前复数场
 * @param support 支撑域约束
 * @param sigma 高斯模糊参数
 * @param threshold 阈值
 * @return 错误码
 */
int update_support_shrinkwrap(
    const ComplexImage *field,
    SupportConstraint *support,
    float sigma,
    float threshold
);

/**
 * @brief 初始化随机相位
 * @param field 复数场
 * @param amplitude 振幅（可选）
 * @return 错误码
 */
int initialize_random_phase(
    ComplexImage *field,
    const RealImage *amplitude
);

/**
 * @brief 获取错误信息字符串
 * @param error_code 错误码
 * @return 错误信息字符串
 */
const char* reconstruction_get_error_string(int error_code);

/**
 * @brief 验证重建参数
 * @param params 相位恢复参数
 * @return 错误码
 */
int reconstruction_validate_params(const PhaseRetrievalParams *params);

#ifdef __cplusplus
}
#endif

#endif // RECONSTRUCTION_H
