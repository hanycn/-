/**
 * @file deconvolution.h
 * @brief Image deconvolution algorithms for optical microscopy and imaging
 * @author Your Name
 * @date 2024
 * 
 * This header provides various deconvolution algorithms including:
 * - Richardson-Lucy deconvolution
 * - Wiener filtering
 * - Blind deconvolution
 * - Total Variation deconvolution
 * - Regularized deconvolution methods
 */

#ifndef DECONVOLUTION_H
#define DECONVOLUTION_H

#include <stdint.h>
#include <stdbool.h>
#include <complex.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// 类型定义
// ============================================================================

// 复数类型
typedef float complex ComplexF;
typedef double complex ComplexD;

// ============================================================================
// 常量定义
// ============================================================================

#define DECONVOLUTION_MAX_ITERATIONS 10000
#define DECONVOLUTION_DEFAULT_TOLERANCE 1e-6f
#define DECONVOLUTION_MAX_PSF_SIZE 512

// 错误码
#define DECONVOLUTION_SUCCESS 0
#define DECONVOLUTION_ERROR_NULL_POINTER -1
#define DECONVOLUTION_ERROR_INVALID_PARAMS -2
#define DECONVOLUTION_ERROR_MEMORY_ALLOCATION -3
#define DECONVOLUTION_ERROR_FFT_FAILED -4
#define DECONVOLUTION_ERROR_NOT_CONVERGED -5
#define DECONVOLUTION_ERROR_DIMENSION_MISMATCH -6
#define DECONVOLUTION_ERROR_FILE_IO -7
#define DECONVOLUTION_ERROR_INVALID_PSF -8
#define DECONVOLUTION_ERROR_NUMERICAL_INSTABILITY -9

// ============================================================================
// 数据结构
// ============================================================================

/**
 * @brief 实数图像结构
 */
typedef struct {
    int width;              // 图像宽度
    int height;             // 图像高度
    int channels;           // 通道数（1=灰度，3=RGB）
    float *data;            // 图像数据（行优先）
} RealImage;

/**
 * @brief 复数图像结构
 */
typedef struct {
    int width;              // 图像宽度
    int height;             // 图像高度
    ComplexF *data;         // 复数数据
} ComplexImage;

/**
 * @brief 3D实数图像结构
 */
typedef struct {
    int width;              // X维度
    int height;             // Y维度
    int depth;              // Z维度
    int channels;           // 通道数
    float *data;            // 3D数据
} RealImage3D;

/**
 * @brief 3D复数图像结构
 */
typedef struct {
    int width;
    int height;
    int depth;
    ComplexF *data;
} ComplexImage3D;

// ============================================================================
// 反卷积算法类型
// ============================================================================

/**
 * @brief 反卷积算法枚举
 */
typedef enum {
    DECONV_RICHARDSON_LUCY,      // Richardson-Lucy算法
    DECONV_WIENER,               // Wiener滤波
    DECONV_REGULARIZED_INVERSE,  // 正则化逆滤波
    DECONV_BLIND,                // 盲反卷积
    DECONV_TOTAL_VARIATION,      // 全变分反卷积
    DECONV_LUCY_RICHARDSON_TV,   // RL + TV正则化
    DECONV_GOLD,                 // Gold反卷积
    DECONV_VAN_CITTERT,          // Van Cittert迭代
    DECONV_LANDWEBER,            // Landweber迭代
    DECONV_CONJUGATE_GRADIENT    // 共轭梯度法
} DeconvolutionAlgorithm;

/**
 * @brief 边界处理方式
 */
typedef enum {
    BOUNDARY_ZERO,          // 零填充
    BOUNDARY_PERIODIC,      // 周期边界
    BOUNDARY_MIRROR,        // 镜像边界
    BOUNDARY_REPLICATE      // 复制边界
} BoundaryCondition;

/**
 * @brief 正则化类型
 */
typedef enum {
    REGULARIZATION_NONE,           // 无正则化
    REGULARIZATION_TIKHONOV,       // Tikhonov正则化
    REGULARIZATION_TV,             // 全变分
    REGULARIZATION_L1,             // L1范数
    REGULARIZATION_GOOD_ROUGHNESS, // Good's roughness
    REGULARIZATION_ENTROPY         // 最大熵
} RegularizationType;

// ============================================================================
// 参数结构
// ============================================================================

/**
 * @brief Richardson-Lucy反卷积参数
 */
typedef struct {
    int max_iterations;           // 最大迭代次数
    float tolerance;              // 收敛容差
    float damping_factor;         // 阻尼因子（0-1）
    bool use_acceleration;        // 是否使用加速
    int acceleration_order;       // 加速阶数
    bool enforce_positivity;      // 强制正值约束
    float background_level;       // 背景水平
    bool verbose;                 // 是否输出详细信息
    int print_interval;           // 打印间隔
} RichardsonLucyParams;

/**
 * @brief Wiener滤波参数
 */
typedef struct {
    float noise_variance;         // 噪声方差
    float signal_variance;        // 信号方差
    float regularization;         // 正则化参数（SNR估计）
    bool estimate_noise;          // 是否自动估计噪声
    bool use_adaptive;            // 是否使用自适应Wiener
    int window_size;              // 自适应窗口大小
} WienerParams;

/**
 * @brief 盲反卷积参数
 */
typedef struct {
    int max_iterations;           // 最大迭代次数
    float tolerance;              // 收敛容差
    int psf_size;                 // PSF尺寸
    float psf_regularization;     // PSF正则化参数
    float image_regularization;   // 图像正则化参数
    bool initialize_psf;          // 是否初始化PSF
    float *initial_psf;           // 初始PSF（可选）
    bool enforce_psf_positivity;  // PSF正值约束
    bool enforce_psf_sum;         // PSF归一化约束
    bool verbose;
    int print_interval;
} BlindDeconvParams;

/**
 * @brief 全变分反卷积参数
 */
typedef struct {
    int max_iterations;           // 最大迭代次数
    float tolerance;              // 收敛容差
    float lambda;                 // TV正则化参数
    float dt;                     // 时间步长
    bool use_anisotropic;         // 各向异性TV
    bool enforce_positivity;      // 正值约束
    int inner_iterations;         // 内部迭代次数
    bool verbose;
    int print_interval;
} TVDeconvParams;

/**
 * @brief 通用反卷积参数
 */
typedef struct {
    DeconvolutionAlgorithm algorithm;  // 算法类型
    BoundaryCondition boundary;        // 边界条件
    RegularizationType regularization; // 正则化类型
    float regularization_param;        // 正则化参数
    int max_iterations;                // 最大迭代次数
    float tolerance;                   // 收敛容差
    bool enforce_positivity;           // 正值约束
    bool verbose;                      // 详细输出
    int print_interval;                // 打印间隔
    
    // 算法特定参数（联合体）
    union {
        RichardsonLucyParams rl;
        WienerParams wiener;
        BlindDeconvParams blind;
        TVDeconvParams tv;
    } params;
} DeconvolutionParams;

/**
 * @brief PSF（点扩散函数）结构
 */
typedef struct {
    int width;                    // PSF宽度
    int height;                   // PSF高度
    int depth;                    // PSF深度（3D）
    float *data;                  // PSF数据
    bool is_normalized;           // 是否已归一化
    float center_x;               // 中心X坐标
    float center_y;               // 中心Y坐标
    float center_z;               // 中心Z坐标（3D）
    
    // PSF元数据
    float wavelength;             // 波长（nm）
    float numerical_aperture;     // 数值孔径
    float refractive_index;       // 折射率
    float pixel_size;             // 像素大小（μm）
    float z_spacing;              // Z轴间距（μm）
} PSF;

/**
 * @brief 反卷积结果结构
 */
typedef struct {
    RealImage *deconvolved;       // 反卷积结果
    PSF *estimated_psf;           // 估计的PSF（盲反卷积）
    
    int iterations_performed;     // 执行的迭代次数
    bool converged;               // 是否收敛
    float final_error;            // 最终误差
    float *error_history;         // 误差历史
    int error_history_length;     // 误差历史长度
    
    double computation_time;      // 计算时间（秒）
    
    // 质量指标
    float snr_improvement;        // SNR改善（dB）
    float contrast_improvement;   // 对比度改善
    float resolution_improvement; // 分辨率改善
} DeconvolutionResult;

/**
 * @brief 3D反卷积结果
 */
typedef struct {
    RealImage3D *deconvolved;     // 3D反卷积结果
    PSF *estimated_psf;           // 估计的PSF
    
    int iterations_performed;
    bool converged;
    float final_error;
    float *error_history;
    int error_history_length;
    
    double computation_time;
    
    float snr_improvement;
    float contrast_improvement;
    float resolution_improvement;
} DeconvolutionResult3D;

// ============================================================================
// PSF生成和操作
// ============================================================================

/**
 * @brief 创建PSF结构
 */
PSF* psf_create(int width, int height, int depth);

/**
 * @brief 销毁PSF结构
 */
void psf_destroy(PSF *psf);

/**
 * @brief 从文件加载PSF
 */
PSF* psf_load_from_file(const char *filename);

/**
 * @brief 保存PSF到文件
 */
int psf_save_to_file(const PSF *psf, const char *filename);

/**
 * @brief 归一化PSF
 */
int psf_normalize(PSF *psf);

/**
 * @brief 生成高斯PSF
 */
PSF* psf_generate_gaussian(int width, int height, float sigma_x, float sigma_y);

/**
 * @brief 生成3D高斯PSF
 */
PSF* psf_generate_gaussian_3d(int width, int height, int depth,
                               float sigma_x, float sigma_y, float sigma_z);

/**
 * @brief 生成Airy盘PSF（衍射受限）
 */
PSF* psf_generate_airy(int width, int height, float wavelength,
                        float numerical_aperture, float pixel_size);

/**
 * @brief 生成Born-Wolf 3D PSF（理论显微镜PSF）
 */
PSF* psf_generate_born_wolf(int width, int height, int depth,
                             float wavelength, float numerical_aperture,
                             float refractive_index, float pixel_size,
                             float z_spacing);

/**
 * @brief 生成Gibson-Lanni PSF（考虑球差）
 */
PSF* psf_generate_gibson_lanni(int width, int height, int depth,
                                float wavelength, float numerical_aperture,
                                float refractive_index_immersion,
                                float refractive_index_sample,
                                float working_distance,
                                float pixel_size, float z_spacing);

/**
 * @brief 估计PSF中心位置
 */
int psf_estimate_center(const PSF *psf, float *center_x, float *center_y, float *center_z);

/**
 * @brief 裁剪PSF到指定大小
 */
PSF* psf_crop(const PSF *psf, int new_width, int new_height, int new_depth);

/**
 * @brief 填充PSF到指定大小
 */
PSF* psf_pad(const PSF *psf, int new_width, int new_height, int new_depth);

// ============================================================================
// 参数创建和销毁
// ============================================================================

/**
 * @brief 创建默认Richardson-Lucy参数
 */
RichardsonLucyParams richardson_lucy_params_default(void);

/**
 * @brief 创建默认Wiener参数
 */
WienerParams wiener_params_default(void);

/**
 * @brief 创建默认盲反卷积参数
 */
BlindDeconvParams blind_deconv_params_default(void);

/**
 * @brief 创建默认TV反卷积参数
 */
TVDeconvParams tv_deconv_params_default(void);

/**
 * @brief 创建默认通用反卷积参数
 */
DeconvolutionParams deconvolution_params_default(DeconvolutionAlgorithm algorithm);

// ============================================================================
// 结果创建和销毁
// ============================================================================

/**
 * @brief 创建反卷积结果结构
 */
DeconvolutionResult* deconvolution_result_create(int width, int height,
                                                  int max_iterations);

/**
 * @brief 销毁反卷积结果
 */
void deconvolution_result_destroy(DeconvolutionResult *result);

/**
 * @brief 创建3D反卷积结果结构
 */
DeconvolutionResult3D* deconvolution_result_3d_create(int width, int height,
                                                       int depth, int max_iterations);

/**
 * @brief 销毁3D反卷积结果
 */
void deconvolution_result_3d_destroy(DeconvolutionResult3D *result);

// ============================================================================
// 2D反卷积算法
// ============================================================================

/**
 * @brief Richardson-Lucy反卷积
 * 
 * @param blurred 模糊图像
 * @param psf 点扩散函数
 * @param params 算法参数
 * @param result 输出结果
 * @return 错误码
 */
int deconvolve_richardson_lucy(const RealImage *blurred,
                               const PSF *psf,
                               const RichardsonLucyParams *params,
                               DeconvolutionResult *result);

/**
 * @brief 加速Richardson-Lucy反卷积
 */
int deconvolve_richardson_lucy_accelerated(const RealImage *blurred,
                                          const PSF *psf,
                                          const RichardsonLucyParams *params,
                                          DeconvolutionResult *result);

/**
 * @brief Wiener滤波反卷积
 */
int deconvolve_wiener(const RealImage *blurred,
                     const PSF *psf,
                     const WienerParams *params,
                     DeconvolutionResult *result);

/**
 * @brief 自适应Wiener滤波
 */
int deconvolve_wiener_adaptive(const RealImage *blurred,
                              const PSF *psf,
                              const WienerParams *params,
                              DeconvolutionResult *result);

/**
 * @brief 盲反卷积
 */
int deconvolve_blind(const RealImage *blurred,
                    const BlindDeconvParams *params,
                    DeconvolutionResult *result);

/**
 * @brief 全变分反卷积
 */
int deconvolve_total_variation(const RealImage *blurred,
                               const PSF *psf,
                               const TVDeconvParams *params,
                               DeconvolutionResult *result);

/**
 * @brief Richardson-Lucy + TV正则化
 */
int deconvolve_rl_tv(const RealImage *blurred,
                    const PSF *psf,
                    const RichardsonLucyParams *rl_params,
                    const TVDeconvParams *tv_params,
                    DeconvolutionResult *result);

/**
 * @brief Gold反卷积
 */
int deconvolve_gold(const RealImage *blurred,
                   const PSF *psf,
                   int max_iterations,
                   float tolerance,
                   DeconvolutionResult *result);

/**
 * @brief Van Cittert迭代反卷积
 */
int deconvolve_van_cittert(const RealImage *blurred,
                          const PSF *psf,
                          int max_iterations,
                          float relaxation,
                          DeconvolutionResult *result);

/**
 * @brief Landweber迭代反卷积
 */
int deconvolve_landweber(const RealImage *blurred,
                        const PSF *psf,
                        int max_iterations,
                        float step_size,
                        DeconvolutionResult *result);

/**
 * @brief 共轭梯度反卷积
 */
int deconvolve_conjugate_gradient(const RealImage *blurred,
                                 const PSF *psf,
                                 int max_iterations,
                                 float tolerance,
                                 DeconvolutionResult *result);

/**
 * @brief 通用反卷积接口
 */
int deconvolve(const RealImage *blurred,
              const PSF *psf,
              const DeconvolutionParams *params,
              DeconvolutionResult *result);

// ============================================================================
// 3D反卷积算法
// ============================================================================

/**
 * @brief 3D Richardson-Lucy反卷积
 */
int deconvolve_3d_richardson_lucy(const RealImage3D *blurred,
                                 const PSF *psf,
                                 const RichardsonLucyParams *params,
                                 DeconvolutionResult3D *result);

/**
 * @brief 3D Wiener滤波
 */
int deconvolve_3d_wiener(const RealImage3D *blurred,
                        const PSF *psf,
                        const WienerParams *params,
                        DeconvolutionResult3D *result);

/**
 * @brief 3D盲反卷积
 */
int deconvolve_3d_blind(const RealImage3D *blurred,
                       const BlindDeconvParams *params,
                       DeconvolutionResult3D *result);

/**
 * @brief 3D全变分反卷积
 */
int deconvolve_3d_total_variation(const RealImage3D *blurred,
                                 const PSF *psf,
                                 const TVDeconvParams *params,
                                 DeconvolutionResult3D *result);

/**
 * @brief 通用3D反卷积接口
 */
int deconvolve_3d(const RealImage3D *blurred,
                 const PSF *psf,
                 const DeconvolutionParams *params,
                 DeconvolutionResult3D *result);

// ============================================================================
// 噪声估计和预处理
// ============================================================================

/**
 * @brief 估计图像噪声方差
 */
float estimate_noise_variance(const RealImage *image);

/**
 * @brief 估计信号方差
 */
float estimate_signal_variance(const RealImage *image);

/**
 * @brief 估计信噪比
 */
float estimate_snr(const RealImage *image);

/**
 * @brief 自动估计Wiener滤波参数
 */
int estimate_wiener_parameters(const RealImage *blurred,
                               const PSF *psf,
                               WienerParams *params);

/**
 * @brief 图像预处理（去噪、归一化等）
 */
int preprocess_image(RealImage *image, bool denoise, bool normalize);

// ============================================================================
// 质量评估
// ============================================================================

/**
 * @brief 计算反卷积质量指标
 */
int compute_deconvolution_quality(const DeconvolutionResult *result,
                                 const RealImage *reference,
                                 const RealImage *original_blurred);

/**
 * @brief 计算图像清晰度
 */
float compute_image_sharpness(const RealImage *image);

/**
 * @brief 计算图像对比度
 */
float compute_image_contrast(const RealImage *image);

/**
 * @brief 计算SNR改善
 */
float compute_snr_improvement(const RealImage *original,
                             const RealImage *deconvolved);

// ============================================================================
// 结果保存和可视化
// ============================================================================

/**
 * @brief 保存反卷积结果
 */
int save_deconvolution_result(const DeconvolutionResult *result,
                              const char *filename_prefix);

/**
 * @brief 保存3D反卷积结果
 */
int save_deconvolution_result_3d(const DeconvolutionResult3D *result,
                                 const char *filename_prefix);

/**
 * @brief 创建对比图像（原图vs反卷积）
 */
RealImage* create_comparison_image(const RealImage *original,
                                   const RealImage *deconvolved);

// ============================================================================
// 辅助函数
// ============================================================================

/**
 * @brief 验证参数有效性
 */
int validate_deconvolution_params(const DeconvolutionParams *params);

/**
 * @brief 打印算法信息
 */
void print_algorithm_info(DeconvolutionAlgorithm algorithm);

/**
 * @brief 打印反卷积摘要
 */
void print_deconvolution_summary(const DeconvolutionResult *result);

/**
 * @brief 获取错误信息字符串
 */
const char* deconvolution_error_string(int error_code);

/**
 * @brief 打印错误信息
 */
void print_error_message(int error_code, const char *context);

#ifdef __cplusplus
}
#endif

#endif // DECONVOLUTION_H
