#ifndef DIFFRACTION_MODEL_H
#define DIFFRACTION_MODEL_H

#include <complex.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// 类型定义
// ============================================================================

// 复数类型定义（使用C99标准）
typedef float complex ComplexF;
typedef double complex ComplexD;

// 衍射类型枚举
typedef enum {
    DIFFRACTION_FRESNEL,        // 菲涅尔衍射（近场）
    DIFFRACTION_FRAUNHOFER,     // 夫琅禾费衍射（远场）
    DIFFRACTION_ANGULAR_SPECTRUM // 角谱法（精确，推荐）
} DiffractionType;

// 波长信息结构
typedef struct {
    double wavelength;          // 波长 (米)，例如: 550e-9 (绿光)
    double weight;              // 权重（用于多色光），归一化后总和为1
} WavelengthInfo;

// 衍射参数结构
typedef struct {
    // 图像尺寸
    int width;                  // 图像宽度（像素）
    int height;                 // 图像高度（像素）
    
    // 物理参数
    double pixel_size;          // 像素尺寸（米），例如: 5e-6 (5微米)
    double propagation_distance; // 传播距离（米），例如: 0.01 (1厘米)
    
    // 波长信息
    WavelengthInfo *wavelengths; // 波长数组
    int num_wavelengths;        // 波长数量
    
    // 算法参数
    DiffractionType type;       // 衍射类型
    bool use_padding;           // 是否使用零填充（减少边界效应）
    int padding_factor;         // 填充因子（2的幂，通常为2）
} DiffractionParams;

// 复数图像结构
typedef struct {
    ComplexF *data;             // 复数数据（行优先存储）
    int width;                  // 宽度
    int height;                 // 高度
    int stride;                 // 行跨度（通常等于width）
} ComplexImage;

// 实数图像结构
typedef struct {
    float *data;                // 实数数据（行优先存储）
    int width;                  // 宽度
    int height;                 // 高度
    int channels;               // 通道数（1=灰度，3=RGB）
    int stride;                 // 行跨度（width * channels）
} RealImage;

// 错误码
typedef enum {
    DIFFRACTION_SUCCESS = 0,
    DIFFRACTION_ERROR_NULL_POINTER = -1,
    DIFFRACTION_ERROR_INVALID_PARAMS = -2,
    DIFFRACTION_ERROR_MEMORY_ALLOCATION = -3,
    DIFFRACTION_ERROR_FFT_FAILED = -4,
    DIFFRACTION_ERROR_INVALID_WAVELENGTH = -5
} DiffractionError;

// ============================================================================
// 参数管理函数
// ============================================================================

/**
 * 创建衍射参数结构
 * 
 * @param width 图像宽度（像素）
 * @param height 图像高度（像素）
 * @param pixel_size 像素尺寸（米），例如: 5e-6
 * @param distance 传播距离（米），例如: 0.01
 * @param wavelength 波长（米），例如: 550e-9 (绿光)
 * @param type 衍射类型
 * @return 衍射参数指针，失败返回NULL
 * 
 * 示例:
 *   DiffractionParams *params = diffraction_params_create(
 *       1024, 1024,           // 1024x1024图像
 *       5e-6,                 // 5微米像素
 *       0.01,                 // 1厘米传播距离
 *       550e-9,               // 550纳米波长（绿光）
 *       DIFFRACTION_ANGULAR_SPECTRUM
 *   );
 */
DiffractionParams* diffraction_params_create(
    int width, int height,
    double pixel_size, double distance,
    double wavelength, DiffractionType type
);

/**
 * 添加波长（用于多色光模拟）
 * 
 * @param params 衍射参数
 * @param wavelength 波长（米）
 * @param weight 权重（相对强度）
 * @return 成功返回0，失败返回负数错误码
 * 
 * 示例（模拟白光）:
 *   diffraction_params_add_wavelength(params, 450e-9, 0.3); // 蓝光
 *   diffraction_params_add_wavelength(params, 550e-9, 0.4); // 绿光
 *   diffraction_params_add_wavelength(params, 650e-9, 0.3); // 红光
 */
int diffraction_params_add_wavelength(
    DiffractionParams *params,
    double wavelength, double weight
);

/**
 * 设置填充参数
 * 
 * @param params 衍射参数
 * @param use_padding 是否使用填充
 * @param padding_factor 填充因子（2的幂）
 */
void diffraction_params_set_padding(
    DiffractionParams *params,
    bool use_padding, int padding_factor
);

/**
 * 销毁衍射参数
 * 
 * @param params 衍射参数指针
 */
void diffraction_params_destroy(DiffractionParams *params);

// ============================================================================
// 图像管理函数
// ============================================================================

/**
 * 创建复数图像
 * 
 * @param width 宽度
 * @param height 高度
 * @return 复数图像指针，失败返回NULL
 */
ComplexImage* complex_image_create(int width, int height);

/**
 * 复制复数图像
 * 
 * @param src 源图像
 * @return 新图像指针，失败返回NULL
 */
ComplexImage* complex_image_clone(const ComplexImage *src);

/**
 * 销毁复数图像
 * 
 * @param img 图像指针
 */
void complex_image_destroy(ComplexImage *img);

/**
 * 创建实数图像
 * 
 * @param width 宽度
 * @param height 高度
 * @param channels 通道数（1或3）
 * @return 实数图像指针，失败返回NULL
 */
RealImage* real_image_create(int width, int height, int channels);

/**
 * 复制实数图像
 * 
 * @param src 源图像
 * @return 新图像指针，失败返回NULL
 */
RealImage* real_image_clone(const RealImage *src);

/**
 * 销毁实数图像
 * 
 * @param img 图像指针
 */
void real_image_destroy(RealImage *img);

// ============================================================================
// 图像转换函数
// ============================================================================

/**
 * 从实数图像创建复数图像（实部=振幅，虚部=0）
 * 
 * @param real_img 实数图像
 * @return 复数图像指针，失败返回NULL
 */
ComplexImage* complex_image_from_real(const RealImage *real_img);

/**
 * 从振幅和相位创建复数图像
 * 
 * @para
