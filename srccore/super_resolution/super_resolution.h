/**
 * @file super_resolution.h
 * @brief 超分辨率图像重建库
 * 
 * 提供多种超分辨率算法，包括：
 * - 插值方法（双三次、Lanczos等）
 * - 基于重建的方法（IBSR）
 * - 基于学习的方法（稀疏编码、字典学习）
 * - 深度学习方法接口
 * - 多帧超分辨率
 * 
 * @author Image Processing Library
 * @version 1.0.0
 * @date 2024
 */

#ifndef SUPER_RESOLUTION_H
#define SUPER_RESOLUTION_H

#include <stdint.h>
#include <stdbool.h>
#include <complex.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// 类型定义
// ============================================================================

/** 复数类型 */
typedef float complex ComplexF;

/** 图像结构 */
typedef struct {
    int width;          /**< 图像宽度 */
    int height;         /**< 图像高度 */
    int channels;       /**< 通道数 */
    float *data;        /**< 图像数据 (行优先，通道交错) */
} Image;

/** 3D图像结构 */
typedef struct {
    int width;          /**< 宽度 */
    int height;         /**< 高度 */
    int depth;          /**< 深度 */
    int channels;       /**< 通道数 */
    float *data;        /**< 图像数据 */
} Image3D;

/** 图像块结构 */
typedef struct {
    int width;          /**< 块宽度 */
    int height;         /**< 块高度 */
    int channels;       /**< 通道数 */
    float *data;        /**< 块数据 */
    int x;              /**< 块在原图中的x坐标 */
    int y;              /**< 块在原图中的y坐标 */
} ImagePatch;

// ============================================================================
// 枚举类型
// ============================================================================

/** 超分辨率算法类型 */
typedef enum {
    SR_NEAREST_NEIGHBOR,        /**< 最近邻插值 */
    SR_BILINEAR,               /**< 双线性插值 */
    SR_BICUBIC,                /**< 双三次插值 */
    SR_LANCZOS,                /**< Lanczos插值 */
    SR_IBSR,                   /**< 迭代反向投影 */
    SR_SPARSE_CODING,          /**< 稀疏编码 */
    SR_DICTIONARY_LEARNING,    /**< 字典学习 */
    SR_EXAMPLE_BASED,          /**< 基于样例 */
    SR_MULTI_FRAME,            /**< 多帧超分辨率 */
    SR_DEEP_LEARNING,          /**< 深度学习方法 */
    SR_EDGE_DIRECTED,          /**< 边缘引导 */
    SR_GRADIENT_PROFILE        /**< 梯度轮廓 */
} SRAlgorithm;

/** 插值方法 */
typedef enum {
    INTERP_NEAREST,            /**< 最近邻 */
    INTERP_LINEAR,             /**< 线性 */
    INTERP_CUBIC,              /**< 三次 */
    INTERP_LANCZOS2,           /**< Lanczos-2 */
    INTERP_LANCZOS3,           /**< Lanczos-3 */
    INTERP_LANCZOS4            /**< Lanczos-4 */
} InterpolationMethod;

/** 边界处理方式 */
typedef enum {
    BOUNDARY_ZERO,             /**< 零填充 */
    BOUNDARY_REPLICATE,        /**< 边界复制 */
    BOUNDARY_REFLECT,          /**< 镜像反射 */
    BOUNDARY_WRAP              /**< 周期包裹 */
} BoundaryMode;

/** 下采样方法 */
typedef enum {
    DOWNSAMPLE_NEAREST,        /**< 最近邻 */
    DOWNSAMPLE_AVERAGE,        /**< 平均 */
    DOWNSAMPLE_GAUSSIAN,       /**< 高斯 */
    DOWNSAMPLE_LANCZOS         /**< Lanczos */
} DownsampleMethod;

/** 配准方法 */
typedef enum {
    REGISTRATION_NONE,         /**< 无配准 */
    REGISTRATION_TRANSLATION,  /**< 平移 */
    REGISTRATION_AFFINE,       /**< 仿射变换 */
    REGISTRATION_OPTICAL_FLOW  /**< 光流 */
} RegistrationMethod;

/** 错误代码 */
typedef enum {
    SR_SUCCESS = 0,                    /**< 成功 */
    SR_ERROR_NULL_POINTER = -1,        /**< 空指针 */
    SR_ERROR_INVALID_DIMENSIONS = -2,  /**< 无效尺寸 */
    SR_ERROR_MEMORY_ALLOCATION = -3,   /**< 内存分配失败 */
    SR_ERROR_INVALID_SCALE = -4,       /**< 无效缩放因子 */
    SR_ERROR_INVALID_ALGORITHM = -5,   /**< 无效算法 */
    SR_ERROR_CONVERGENCE = -6,         /**< 收敛失败 */
    SR_ERROR_FILE_IO = -7,             /**< 文件IO错误 */
    SR_ERROR_NOT_IMPLEMENTED = -8,     /**< 未实现 */
    SR_ERROR_INVALID_PARAMETER = -9    /**< 无效参数 */
} SRErrorCode;

// ============================================================================
// 参数结构
// ============================================================================

/** 插值参数 */
typedef struct {
    InterpolationMethod method;  /**< 插值方法 */
    int lanczos_a;              /**< Lanczos参数a */
    float sharpness;            /**< 锐化程度 [0, 1] */
    bool anti_aliasing;         /**< 是否抗锯齿 */
    BoundaryMode boundary;      /**< 边界处理 */
} InterpolationParams;

/** 迭代反向投影参数 */
typedef struct {
    int max_iterations;         /**< 最大迭代次数 */
    float tolerance;            /**< 收敛容差 */
    float lambda;               /**< 正则化参数 */
    float alpha;                /**< 步长 */
    DownsampleMethod downsample; /**< 下采样方法 */
    bool use_edge_prior;        /**< 使用边缘先验 */
    bool verbose;               /**< 详细输出 */
} IBSRParams;

/** 稀疏编码参数 */
typedef struct {
    int dict_size;              /**< 字典大小 */
    int patch_size;             /**< 图像块大小 */
    int overlap;                /**< 重叠像素数 */
    float sparsity;             /**< 稀疏度 */
    int max_iterations;         /**< 最大迭代次数 */
    float tolerance;            /**< 收敛容差 */
    bool use_pca;               /**< 使用PCA降维 */
    int pca_components;         /**< PCA成分数 */
} SparseCodingParams;

/** 字典学习参数 */
typedef struct {
    int dict_size;              /**< 字典大小 */
    int patch_size;             /**< 图像块大小 */
    int overlap;                /**< 重叠像素数 */
    int training_iterations;    /**< 训练迭代次数 */
    int sparse_iterations;      /**< 稀疏编码迭代次数 */
    float sparsity;             /**< 稀疏度 */
    float learning_rate;        /**< 学习率 */
    bool use_online_learning;   /**< 使用在线学习 */
    int batch_size;             /**< 批大小 */
} DictionaryLearningParams;

/** 多帧超分辨率参数 */
typedef struct {
    int num_frames;             /**< 帧数 */
    RegistrationMethod registration; /**< 配准方法 */
    float registration_threshold; /**< 配准阈值 */
    bool use_robust_estimation; /**< 使用鲁棒估计 */
    float outlier_threshold;    /**< 离群点阈值 */
    int max_iterations;         /**< 最大迭代次数 */
    float tolerance;            /**< 收敛容差 */
    float regularization;       /**< 正则化参数 */
} MultiFrameParams;

/** 边缘引导参数 */
typedef struct {
    float edge_threshold;       /**< 边缘阈值 */
    float edge_strength;        /**< 边缘强度 */
    int gradient_window;        /**< 梯度窗口大小 */
    bool use_bilateral;         /**< 使用双边滤波 */
    float spatial_sigma;        /**< 空间sigma */
    float range_sigma;          /**< 范围sigma */
} EdgeDirectedParams;

/** 深度学习参数 */
typedef struct {
    char *model_path;           /**< 模型路径 */
    char *model_type;           /**< 模型类型 (SRCNN, ESPCN, etc.) */
    int num_layers;             /**< 层数 */
    int num_filters;            /**< 滤波器数量 */
    bool use_residual;          /**< 使用残差学习 */
    bool use_batch_norm;        /**< 使用批归一化 */
    float dropout_rate;         /**< Dropout率 */
} DeepLearningParams;

/** 统一超分辨率参数 */
typedef struct {
    SRAlgorithm algorithm;      /**< 算法类型 */
    float scale_factor;         /**< 缩放因子 */
    int output_width;           /**< 输出宽度（可选） */
    int output_height;          /**< 输出高度（可选） */
    
    union {
        InterpolationParams interp;
        IBSRParams ibsr;
        SparseCodingParams sparse;
        DictionaryLearningParams dict;
        MultiFrameParams multi_frame;
        EdgeDirectedParams edge;
        DeepLearningParams deep;
    } params;
    
    bool preserve_aspect_ratio; /**< 保持宽高比 */
    bool post_process;          /**< 后处理 */
    float sharpening;           /**< 锐化强度 */
    float denoising;            /**< 去噪强度 */
} SRParams;

/** 超分辨率结果 */
typedef struct {
    Image *output;              /**< 输出图像 */
    float psnr;                 /**< PSNR（如果有参考） */
    float ssim;                 /**< SSIM（如果有参考） */
    float computation_time;     /**< 计算时间（秒） */
    int iterations;             /**< 迭代次数 */
    bool converged;             /**< 是否收敛 */
    float *error_history;       /**< 误差历史 */
    int error_history_length;   /**< 误差历史长度 */
} SRResult;

/** 字典结构 */
typedef struct {
    int size;                   /**< 字典大小 */
    int atom_dim;               /**< 原子维度 */
    float *atoms;               /**< 字典原子 */
    float *lr_atoms;            /**< 低分辨率字典 */
    float *hr_atoms;            /**< 高分辨率字典 */
} Dictionary;

/** 配准结果 */
typedef struct {
    float tx;                   /**< x方向平移 */
    float ty;                   /**< y方向平移 */
    float rotation;             /**< 旋转角度 */
    float scale_x;              /**< x方向缩放 */
    float scale_y;              /**< y方向缩放 */
    float shear;                /**< 剪切 */
    float *transform_matrix;    /**< 变换矩阵 (3x3) */
    float error;                /**< 配准误差 */
} RegistrationResult;

/** 质量评估指标 */
typedef struct {
    float psnr;                 /**< 峰值信噪比 */
    float ssim;                 /**< 结构相似性 */
    float mse;                  /**< 均方误差 */
    float mae;                  /**< 平均绝对误差 */
    float sharpness;            /**< 锐度 */
    float edge_strength;        /**< 边缘强度 */
    float contrast;             /**< 对比度 */
    float entropy;              /**< 熵 */
} QualityMetrics;

// ============================================================================
// 图像管理函数
// ============================================================================

/**
 * @brief 创建图像
 * @param width 宽度
 * @param height 高度
 * @param channels 通道数
 * @return 图像指针，失败返回NULL
 */
Image* image_create(int width, int height, int channels);

/**
 * @brief 创建3D图像
 * @param width 宽度
 * @param height 高度
 * @param depth 深度
 * @param channels 通道数
 * @return 3D图像指针，失败返回NULL
 */
Image3D* image_3d_create(int width, int height, int depth, int channels);

/**
 * @brief 销毁图像
 * @param image 图像指针
 */
void image_destroy(Image *image);

/**
 * @brief 销毁3D图像
 * @param image 3D图像指针
 */
void image_3d_destroy(Image3D *image);

/**
 * @brief 复制图像
 * @param src 源图像
 * @return 新图像指针，失败返回NULL
 */
Image* image_clone(const Image *src);

/**
 * @brief 从文件加载图像
 * @param filename 文件名
 * @return 图像指针，失败返回NULL
 */
Image* image_load(const char *filename);

/**
 * @brief 保存图像到文件
 * @param image 图像指针
 * @param filename 文件名
 * @return 错误代码
 */
int image_save(const Image *image, const char *filename);

/**
 * @brief 转换图像到灰度
 * @param image 输入图像
 * @return 灰度图像，失败返回NULL
 */
Image* image_to_grayscale(const Image *image);

/**
 * @brief 归一化图像到[0, 1]
 * @param image 图像指针
 * @return 错误代码
 */
int image_normalize(Image *image);

/**
 * @brief 裁剪图像
 * @param image 输入图像
 * @param x 起始x坐标
 * @param y 起始y坐标
 * @param width 宽度
 * @param height 高度
 * @return 裁剪后的图像，失败返回NULL
 */
Image* image_crop(const Image *image, int x, int y, int width, int height);

/**
 * @brief 填充图像
 * @param image 输入图像
 * @param top 顶部填充
 * @param bottom 底部填充
 * @param left 左侧填充
 * @param right 右侧填充
 * @param mode 边界模式
 * @return 填充后的图像，失败返回NULL
 */
Image* image_pad(const Image *image, int top, int bottom, int left, int right,
                 BoundaryMode mode);

// ============================================================================
// 插值方法
// ============================================================================

/**
 * @brief 最近邻插值
 * @param input 输入图像
 * @param scale_factor 缩放因子
 * @return 输出图像，失败返回NULL
 */
Image* interpolate_nearest(const Image *input, float scale_factor);

/**
 * @brief 双线性插值
 * @param input 输入图像
 * @param scale_factor 缩放因子
 * @return 输出图像，失败返回NULL
 */
Image* interpolate_bilinear(const Image *input, float scale_factor);

/**
 * @brief 双三次插值
 * @param input 输入图像
 * @param scale_factor 缩放因子
 * @param a 三次插值参数（通常为-0.5或-0.75）
 * @return 输出图像，失败返回NULL
 */
Image* interpolate_bicubic(const Image *input, float scale_factor, float a);

/**
 * @brief Lanczos插值
 * @param input 输入图像
 * @param scale_factor 缩放因子
 * @param a Lanczos参数（通常为2或3）
 * @return 输出图像，失败返回NULL
 */
Image* interpolate_lanczos(const Image *input, float scale_factor, int a);

/**
 * @brief 通用插值函数
 * @param input 输入图像
 * @param scale_factor 缩放因子
 * @param params 插值参数
 * @return 输出图像，失败返回NULL
 */
Image* interpolate(const Image *input, float scale_factor, 
                   const InterpolationParams *params);

// ============================================================================
// 迭代反向投影超分辨率
// ============================================================================

/**
 * @brief 迭代反向投影超分辨率
 * @param input 输入低分辨率图像
 * @param scale_factor 缩放因子
 * @param params IBSR参数
 * @param result 结果结构
 * @return 错误代码
 */
int super_resolve_ibsr(const Image *input, float scale_factor,
                       const IBSRParams *params, SRResult *result);

/**
 * @brief 带边缘先验的IBSR
 * @param input 输入低分辨率图像
 * @param scale_factor 缩放因子
 * @param params IBSR参数
 * @param result 结果结构
 * @return 错误代码
 */
int super_resolve_ibsr_edge_prior(const Image *input, float scale_factor,
                                   const IBSRParams *params, SRResult *result);

// ============================================================================
// 稀疏编码超分辨率
// ============================================================================

/**
 * @brief 稀疏编码超分辨率
 * @param input 输入低分辨率图像
 * @param scale_factor 缩放因子
 * @param params 稀疏编码参数
 * @param result 结果结构
 * @return 错误代码
 */
int super_resolve_sparse_coding(const Image *input, float scale_factor,
                                const SparseCodingParams *params, 
                                SRResult *result);

/**
 * @brief 使用预训练字典的稀疏编码超分辨率
 * @param input 输入低分辨率图像
 * @param scale_factor 缩放因子
 * @param dictionary 预训练字典
 * @param params 稀疏编码参数
 * @param result 结果结构
 * @return 错误代码
 */
int super_resolve_sparse_coding_with_dict(const Image *input, float scale_factor,
                                          const Dictionary *dictionary,
                                          const SparseCodingParams *params,
                                          SRResult *result);

// ============================================================================
// 字典学习
// ============================================================================

/**
 * @brief 创建字典
 * @param size 字典大小
 * @param atom_dim 原子维度
 * @return 字典指针，失败返回NULL
 */
Dictionary* dictionary_create(int size, int atom_dim);

/**
 * @brief 销毁字典
 * @param dict 字典指针
 */
void dictionary_destroy(Dictionary *dict);

/**
 * @brief 训练超分辨率字典
 * @param training_images 训练图像数组
 * @param num_images 图像数量
 * @param scale_factor 缩放因子
 * @param params 字典学习参数
 * @return 字典指针，失败返回NULL
 */
Dictionary* train_sr_dictionary(const Image **training_images, int num_images,
                               float scale_factor,
                               const DictionaryLearningParams *params);

/**
 * @brief 保存字典到文件
 * @param dict 字典指针
 * @param filename 文件名
 * @return 错误代码
 */
int dictionary_save(const Dictionary *dict, const char *filename);

/**
 * @brief 从文件加载字典
 * @param filename 文件名
 * @return 字典指针，失败返回NULL
 */
Dictionary* dictionary_load(const char *filename);

// ============================================================================
// 多帧超分辨率
// ============================================================================

/**
 * @brief 多帧超分辨率
 * @param frames 输入帧数组
 * @param num_frames 帧数
 * @param scale_factor 缩放因子
 * @param params 多帧参数
 * @param result 结果结构
 * @return 错误代码
 */
int super_resolve_multi_frame(const Image **frames, int num_frames,
                              float scale_factor,
                              const MultiFrameParams *params,
                              SRResult *result);

/**
 * @brief 图像配准
 * @param reference 参考图像
 * @param target 目标图像
 * @param method 配准方法
 * @param result 配准结果
 * @return 错误代码
 */
int register_images(const Image *reference, const Image *target,
                   RegistrationMethod method, RegistrationResult *result);

/**
 * @brief 应用配准变换
 * @param image 输入图像
 * @param registration 配准结果
 * @return 变换后的图像，失败返回NULL
 */
Image* apply_registration(const Image *image, 
                         const RegistrationResult *registration);

// ============================================================================
// 边缘引导超分辨率
// ============================================================================

/**
 * @brief 边缘引导超分辨率
 * @param input 输入低分辨率图像
 * @param scale_factor 缩放因子
 * @param params 边缘引导参数
 * @param result 结果结构
 * @return 错误代码
 */
int super_resolve_edge_directed(const Image *input, float scale_factor,
                                const EdgeDirectedParams *params,
                                SRResult *result);

/**
 * @brief 梯度轮廓先验超分辨率
 * @param input 输入低分辨率图像
 * @param scale_factor 缩放因子
 * @param params 边缘引导参数
 * @param result 结果结构
 * @return 错误代码
 */
int super_resolve_gradient_profile(const Image *input, float scale_factor,
                                   const EdgeDirectedParams *params,
                                   SRResult *result);

// ============================================================================
// 深度学习超分辨率接口
// ============================================================================

/**
 * @brief 深度学习超分辨率
 * @param input 输入低分辨率图像
 * @param scale_factor 缩放因子
 * @param params 深度学习参数
 * @param result 结果结构
 * @return 错误代码
 */
int super_resolve_deep_learning(const Image *input, float scale_factor,
                                const DeepLearningParams *params,
                                SRResult *result);

// ============================================================================
// 统一接口
// ============================================================================

/**
 * @brief 统一超分辨率接口
 * @param input 输入低分辨率图像
 * @param params 超分辨率参数
 * @param result 结果结构
 * @return 错误代码
 */
int super_resolve(const Image *input, const SRParams *params, SRResult *result);

/**
 * @brief 3D超分辨率
 * @param input 输入低分辨率3D图像
 * @param params 超分辨率参数
 * @param result 结果结构
 * @return 错误代码
 */
int super_resolve_3d(const Image3D *input, const SRParams *params, 
                     SRResult *result);

// ============================================================================
// 质量评估
// ============================================================================

/**
 * @brief 计算PSNR
 * @param reference 参考图像
 * @param test 测试图像
 * @return PSNR值
 */
float compute_psnr(const Image *reference, const Image *test);

/**
 * @brief 计算SSIM
 * @param reference 参考图像
 * @param test 测试图像
 * @return SSIM值
 */
float compute_ssim(const Image *reference, const Image *test);

/**
 * @brief 评估超分辨率质量
 * @param reference 参考高分辨率图像（可选）
 * @param low_res 低分辨率图像
 * @param super_res 超分辨率图像
 * @param metrics 质量指标
 * @return 错误代码
 */
int evaluate_sr_quality(const Image *reference, const Image *low_res,
                       const Image *super_res, QualityMetrics *metrics);

/**
 * @brief 计算图像锐度
 * @param image 输入图像
 * @return 锐度值
 */
float compute_sharpness(const Image *image);

/**
 * @brief 计算边缘强度
 * @param image 输入图像
 * @return 边缘强度值
 */
float compute_edge_strength(const Image *image);

// ============================================================================
// 辅助函数
// ============================================================================

/**
 * @brief 下采样图像
 * @param input 输入图像
 * @param scale_factor 缩放因子
 * @param method 下采样方法
 * @return 下采样图像，失败返回NULL
 */
Image* downsample_image(const Image *input, float scale_factor,
                       DownsampleMethod method);

/**
 * @brief 提取图像块
 * @param image 输入图像
 * @param x 起始x坐标
 * @param y 起始y坐标
 * @param patch_size 块大小
 * @return 图像块，失败返回NULL
 */
ImagePatch* extract_patch(const Image *image, int x, int y, int patch_size);

/**
 * @brief 放置图像块
 * @param image 目标图像
 * @param patch 图像块
 * @param x 起始x坐标
 * @param y 起始y坐标
 * @return 错误代码
 */
int place_patch(Image *image, const ImagePatch *patch, int x, int y);

/**
 * @brief 销毁图像块
 * @param patch 图像块指针
 */
void patch_destroy(ImagePatch *patch);

/**
 * @brief 应用后处理
 * @param image 输入图像
 * @param sharpening 锐化强度
 * @param denoising 去噪强度
 * @return 错误代码
 */
int apply_post_processing(Image *image, float sharpening, float denoising);

// ============================================================================
// 参数管理
// ============================================================================

/**
 * @brief 获取默认超分辨率参数
 * @param algorithm 算法类型
 * @return 默认参数
 */
SRParams sr_get_default_params(SRAlgorithm algorithm);

/**
 * @brief 获取默认插值参数
 * @return 默认插值参数
 */
InterpolationParams interp_get_default_params(void);

/**
 * @brief 获取默认IBSR参数
 * @return 默认IBSR参数
 */
IBSRParams ibsr_get_default_params(void);

/**
 * @brief 获取默认稀疏编码参数
 * @return 默认稀疏编码参数
 */
SparseCodingParams sparse_get_default_params(void);

/**
 * @brief 获取默认字典学习参数
 * @return 默认字典学习参数
 */
DictionaryLearningParams dict_get_default_params(void);

/**
 * @brief 获取默认多帧参数
 * @return 默认多帧参数
 */
MultiFrameParams multi_frame_get_default_params(void);

/**
 * @brief 获取默认边缘引导参数
 * @return 默认边缘引导参数
 */
EdgeDirectedParams edge_get_default_params(void);

// ============================================================================
// 结果管理
// ============================================================================

/**
 * @brief 创建超分辨率结果
 * @param width 输出宽度
 * @param height 输出高度
 * @param channels 通道数
 * @return 结果指针，失败返回NULL
 */
SRResult* sr_result_create(int width, int height, int channels);

/**
 * @brief 销毁超分辨率结果
 * @param result 结果指针
 */
void sr_result_destroy(SRResult *result);

// ============================================================================
// 错误处理
// ============================================================================

/**
 * @brief 获取错误描述
 * @param error
