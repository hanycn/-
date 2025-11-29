/**
 * @file design_calibration.c
 * @brief 设计校准模块实现
 * @author hany
 * @version 1.0
 * @date 2025
 */

#include "design_calibration.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <time.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ============================================================================
// 内存管理宏
// ============================================================================

#define DESIGN_CALIB_MALLOC(type, count) \
    ((type*)malloc((count) * sizeof(type)))

#define DESIGN_CALIB_CALLOC(type, count) \
    ((type*)calloc((count), sizeof(type)))

#define DESIGN_CALIB_FREE(ptr) \
    do { if (ptr) { free(ptr); (ptr) = NULL; } } while(0)

#define DESIGN_CALIB_CHECK_NULL(ptr) \
    do { if (!(ptr)) return DESIGN_CALIB_ERROR_NULL_POINTER; } while(0)

// ============================================================================
// 内部数据结构
// ============================================================================

/**
 * @brief 校准上下文内部结构
 */
struct DesignCalibContext {
    DesignCalibConfig config;
    DesignCalibStatus status;
    
    // 进度回调
    DesignCalibProgressCallback progress_callback;
    void *progress_user_data;
    
    // 取消标志
    volatile bool cancel_requested;
    
    // 统计信息
    int num_patterns_detected;
    int num_patterns_used;
    float total_reprojection_error;
    
    // 时间戳
    time_t start_time;
    time_t end_time;
    
    // 调试信息
    char last_error_message[256];
};

/**
 * @brief 优化参数结构
 */
typedef struct {
    // 相机参数
    double fx, fy;           // 焦距
    double cx, cy;           // 主点
    double skew;             // 倾斜
    
    // 畸变参数
    double k1, k2, k3;       // 径向畸变
    double p1, p2;           // 切向畸变
    double k4, k5, k6;       // 高阶径向畸变
    
    // 外参（每个图像）
    double *rotations;       // 罗德里格斯向量 [3 * num_images]
    double *translations;    // 平移向量 [3 * num_images]
    
    int num_images;
} OptimizationParams;

/**
 * @brief 优化问题结构
 */
typedef struct {
    const DesignCalibDetectedPattern **patterns;
    int num_patterns;
    int image_width;
    int image_height;
    DesignCalibDistortionModel distortion_model;
    bool fix_principal_point;
    bool fix_aspect_ratio;
    bool zero_tangent_dist;
} OptimizationProblem;

// ============================================================================
// 内部函数声明
// ============================================================================

// 进度更新
static void update_progress(
    DesignCalibContext *context,
    const char *message,
    int current,
    int total);

// 标定板检测
static DesignCalibError detect_checkerboard(
    const float *image,
    int width,
    int height,
    const DesignCalibPatternConfig *config,
    DesignCalibDetectedPattern *pattern);

static DesignCalibError detect_circles(
    const float *image,
    int width,
    int height,
    const DesignCalibPatternConfig *config,
    DesignCalibDetectedPattern *pattern);

// 角点精化
static DesignCalibError refine_corners_subpixel(
    const float *image,
    int width,
    int height,
    DesignCalibPoint2D *corners,
    int num_corners,
    float window_size);

// 优化
static DesignCalibError optimize_camera_parameters(
    const OptimizationProblem *problem,
    OptimizationParams *params);

static void compute_jacobian(
    const OptimizationProblem *problem,
    const OptimizationParams *params,
    double *jacobian);

static void compute_residuals(
    const OptimizationProblem *problem,
    const OptimizationParams *params,
    double *residuals);

// 投影和反投影
static void project_points(
    const DesignCalibPoint3D *object_points,
    int num_points,
    const double *rotation,
    const double *translation,
    const DesignCalibIntrinsics *intrinsics,
    DesignCalibPoint2D *image_points);

static void apply_distortion(
    float x,
    float y,
    const DesignCalibIntrinsics *intrinsics,
    float *x_distorted,
    float *y_distorted);

static void remove_distortion(
    float x_distorted,
    float y_distorted,
    const DesignCalibIntrinsics *intrinsics,
    float *x,
    float *y);

// 矩阵运算
static void rodrigues_to_matrix(
    const double *rodrigues,
    double *matrix);

static void matrix_to_rodrigues(
    const double *matrix,
    double *rodrigues);

static void matrix_multiply_3x3(
    const double *A,
    const double *B,
    double *C);

static void matrix_vector_multiply_3x3(
    const double *A,
    const double *v,
    double *result);

// 线性代数
static DesignCalibError solve_linear_system(
    const double *A,
    const double *b,
    int n,
    double *x);

static DesignCalibError svd_decomposition(
    const double *A,
    int m,
    int n,
    double *U,
    double *S,
    double *Vt);

// ============================================================================
// 上下文管理
// ============================================================================

DesignCalibContext* design_calib_create_context(
    const DesignCalibConfig *config)
{
    if (!config) {
        return NULL;
    }
    
    DesignCalibContext *context = DESIGN_CALIB_CALLOC(DesignCalibContext, 1);
    if (!context) {
        return NULL;
    }
    
    // 复制配置
    memcpy(&context->config, config, sizeof(DesignCalibConfig));
    
    // 初始化状态
    context->status = DESIGN_CALIB_STATUS_IDLE;
    context->cancel_requested = false;
    context->num_patterns_detected = 0;
    context->num_patterns_used = 0;
    context->total_reprojection_error = 0.0f;
    
    // 设置默认值
    if (context->config.max_iterations <= 0) {
        context->config.max_iterations = 100;
    }
    if (context->config.convergence_threshold <= 0.0f) {
        context->config.convergence_threshold = 1e-6f;
    }
    if (context->config.outlier_threshold <= 0.0f) {
        context->config.outlier_threshold = 3.0f;
    }
    if (context->config.min_detection_quality <= 0.0f) {
        context->config.min_detection_quality = 50.0f;
    }
    if (context->config.max_reprojection_error <= 0.0f) {
        context->config.max_reprojection_error = 2.0f;
    }
    if (context->config.min_num_images <= 0) {
        context->config.min_num_images = 3;
    }
    if (context->config.num_threads <= 0) {
        context->config.num_threads = 1;
    }
    
    return context;
}

void design_calib_destroy_context(DesignCalibContext *context)
{
    DESIGN_CALIB_FREE(context);
}

void design_calib_set_progress_callback(
    DesignCalibContext *context,
    DesignCalibProgressCallback callback,
    void *user_data)
{
    if (!context) return;
    
    context->progress_callback = callback;
    context->progress_user_data = user_data;
}

void design_calib_cancel(DesignCalibContext *context)
{
    if (!context) return;
    
    context->cancel_requested = true;
}

DesignCalibStatus design_calib_get_status(
    const DesignCalibContext *context)
{
    if (!context) {
        return DESIGN_CALIB_STATUS_ERROR;
    }
    
    return context->status;
}

/**
 * @brief 更新进度
 */
static void update_progress(
    DesignCalibContext *context,
    const char *message,
    int current,
    int total)
{
    if (!context || !context->progress_callback) {
        return;
    }
    
    context->progress_callback(
        message,
        current,
        total,
        context->progress_user_data);
}

// ============================================================================
// 标定板检测
// ============================================================================

DesignCalibError design_calib_detect_pattern(
    DesignCalibContext *context,
    const float *image,
    int width,
    int height,
    DesignCalibDetectedPattern **pattern)
{
    DESIGN_CALIB_CHECK_NULL(context);
    DESIGN_CALIB_CHECK_NULL(image);
    DESIGN_CALIB_CHECK_NULL(pattern);
    
    if (width <= 0 || height <= 0) {
        return DESIGN_CALIB_ERROR_INVALID_PARAM;
    }
    
    context->status = DESIGN_CALIB_STATUS_DETECTING_PATTERN;
    
    // 分配结果结构
    DesignCalibDetectedPattern *result = 
        DESIGN_CALIB_CALLOC(DesignCalibDetectedPattern, 1);
    if (!result) {
        return DESIGN_CALIB_ERROR_MEMORY_ALLOCATION;
    }
    
    DesignCalibError err;
    
    // 根据标定板类型检测
    switch (context->config.pattern_config.type) {
        case DESIGN_CALIB_PATTERN_CHECKERBOARD:
            err = detect_checkerboard(
                image, width, height,
                &context->config.pattern_config,
                result);
            break;
            
        case DESIGN_CALIB_PATTERN_CIRCLES:
        case DESIGN_CALIB_PATTERN_ASYMMETRIC_CIRCLES:
            err = detect_circles(
                image, width, height,
                &context->config.pattern_config,
                result);
            break;
            
        default:
            design_calib_free_detected_pattern(result);
            return DESIGN_CALIB_ERROR_INVALID_PARAM;
    }
    
    if (err != DESIGN_CALIB_SUCCESS) {
        design_calib_free_detected_pattern(result);
        return err;
    }
    
    // 亚像素精化
    if (context->config.pattern_config.use_subpixel_refinement) {
        context->status = DESIGN_CALIB_STATUS_EXTRACTING_FEATURES;
        
        err = refine_corners_subpixel(
            image, width, height,
            result->image_points,
            result->num_points,
            context->config.pattern_config.subpixel_window_size);
        
        if (err != DESIGN_CALIB_SUCCESS) {
            design_calib_free_detected_pattern(result);
            return err;
        }
    }
    
    // 评估检测质量
    result->detection_quality = evaluate_pattern_quality(result);
    
    if (result->detection_quality < context->config.min_detection_quality) {
        result->is_valid = false;
    } else {
        result->is_valid = true;
        context->num_patterns_detected++;
    }
    
    *pattern = result;
    context->status = DESIGN_CALIB_STATUS_IDLE;
    
    return DESIGN_CALIB_SUCCESS;
}

/**
 * @brief 检测棋盘格
 */
static DesignCalibError detect_checkerboard(
    const float *image,
    int width,
    int height,
    const DesignCalibPatternConfig *config,
    DesignCalibDetectedPattern *pattern)
{
    int pattern_width = config->pattern_width;
    int pattern_height = config->pattern_height;
    int num_corners = pattern_width * pattern_height;
    
    // 分配内存
    pattern->image_points = DESIGN_CALIB_MALLOC(DesignCalibPoint2D, num_corners);
    pattern->object_points = DESIGN_CALIB_MALLOC(DesignCalibPoint3D, num_corners);
    
    if (!pattern->image_points || !pattern->object_points) {
        DESIGN_CALIB_FREE(pattern->image_points);
        DESIGN_CALIB_FREE(pattern->object_points);
        return DESIGN_CALIB_ERROR_MEMORY_ALLOCATION;
    }
    
    pattern->num_points = num_corners;
    
    // 生成物体坐标点（假设Z=0平面）
    for (int y = 0; y < pattern_height; y++) {
        for (int x = 0; x < pattern_width; x++) {
            int idx = y * pattern_width + x;
            pattern->object_points[idx].x = x * config->square_size;
            pattern->object_points[idx].y = y * config->square_size;
            pattern->object_points[idx].z = 0.0f;
        }
    }
    
    // 检测角点
    // 1. 计算图像梯度
    float *grad_x = DESIGN_CALIB_MALLOC(float, width * height);
    float *grad_y = DESIGN_CALIB_MALLOC(float, width * height);
    
    if (!grad_x || !grad_y) {
        DESIGN_CALIB_FREE(grad_x);
        DESIGN_CALIB_FREE(grad_y);
        return DESIGN_CALIB_ERROR_MEMORY_ALLOCATION;
    }
    
    // Sobel算子计算梯度
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            int idx = y * width + x;
            
            // Sobel X
            grad_x[idx] = 
                -image[(y-1)*width + (x-1)] + image[(y-1)*width + (x+1)] +
                -2*image[y*width + (x-1)] + 2*image[y*width + (x+1)] +
                -image[(y+1)*width + (x-1)] + image[(y+1)*width + (x+1)];
            
            // Sobel Y
            grad_y[idx] = 
                -image[(y-1)*width + (x-1)] - 2*image[(y-1)*width + x] - image[(y-1)*width + (x+1)] +
                image[(y+1)*width + (x-1)] + 2*image[(y+1)*width + x] + image[(y+1)*width + (x+1)];
        }
    }
    
    // 2. 计算Harris角点响应
    float *harris_response = DESIGN_CALIB_CALLOC(float, width * height);
    if (!harris_response) {
        DESIGN_CALIB_FREE(grad_x);
        DESIGN_CALIB_FREE(grad_y);
        return DESIGN_CALIB_ERROR_MEMORY_ALLOCATION;
    }
    
    float k = 0.04f;  // Harris参数
    int window_size = 3;
    int radius = window_size / 2;
    
    for (int y = radius; y < height - radius; y++) {
        for (int x = radius; x < width - radius; x++) {
            double Ixx = 0.0, Iyy = 0.0, Ixy = 0.0;
            
            // 在窗口内累积
            for (int dy = -radius; dy <= radius; dy++) {
                for (int dx = -radius; dx <= radius; dx++) {
                    int idx = (y + dy) * width + (x + dx);
                    float gx = grad_x[idx];
                    float gy = grad_y[idx];
                    
                    Ixx += gx * gx;
                    Iyy += gy * gy;
                    Ixy += gx * gy;
                }
            }
            
            // 计算响应 R = det(M) - k * trace(M)^2
            double det = Ixx * Iyy - Ixy * Ixy;
            double trace = Ixx + Iyy;
            harris_response[y * width + x] = det - k * trace * trace;
        }
    }
    
    DESIGN_CALIB_FREE(grad_x);
    DESIGN_CALIB_FREE(grad_y);
    
    // 3. 非极大值抑制
    float threshold = 0.0f;
    
    // 计算阈值（使用响应的百分位数）
    float *sorted_response = DESIGN_CALIB_MALLOC(float, width * height);
    if (!sorted_response) {
        DESIGN_CALIB_FREE(harris_response);
        return DESIGN_CALIB_ERROR_MEMORY_ALLOCATION;
    }
    
    memcpy(sorted_response, harris_response, width * height * sizeof(float));
    qsort(sorted_response, width * height, sizeof(float), 
          (int(*)(const void*, const void*))compare_float);
    
    threshold = sorted_response[(int)(width * height * 0.99)];  // 99百分位
    DESIGN_CALIB_FREE(sorted_response);
    
    // 提取局部最大值
    typedef struct {
        float x, y;
        float response;
    } Corner;
    
    Corner *corners = DESIGN_CALIB_MALLOC(Corner, num_corners * 2);
    int num_detected = 0;
    
    if (!corners) {
        DESIGN_CALIB_FREE(harris_response);
        return DESIGN_CALIB_ERROR_MEMORY_ALLOCATION;
    }
    
    int nms_radius = 5;
    for (int y = nms_radius; y < height - nms_radius; y++) {
        for (int x = nms_radius; x < width - nms_radius; x++) {
            int idx = y * width + x;
            float response = harris_response[idx];
            
            if (response < threshold) continue;
            
            // 检查是否为局部最大值
            bool is_maximum = true;
            for (int dy = -nms_radius; dy <= nms_radius && is_maximum; dy++) {
                for (int dx = -nms_radius; dx <= nms_radius && is_maximum; dx++) {
                    if (dx == 0 && dy == 0) continue;
                    int neighbor_idx = (y + dy) * width + (x + dx);
                    if (harris_response[neighbor_idx] > response) {
                        is_maximum = false;
                    }
                }
            }
            
            if (is_maximum && num_detected < num_corners * 2) {
                corners[num_detected].x = x;
                corners[num_detected].y = y;
                corners[num_detected].response = response;
                num_detected++;
            }
        }
    }
    
    DESIGN_CALIB_FREE(harris_response);
    
    // 4. 排序并选择最强的角点
    qsort(corners, num_detected, sizeof(Corner),
          (int(*)(const void*, const void*))compare_corners);
    
    if (num_detected < num_corners) {
        DESIGN_CALIB_FREE(corners);
        return DESIGN_CALIB_ERROR_PATTERN_NOT_FOUND;
    }
    
    // 5. 尝试将角点组织成网格
    // 这是一个简化版本，实际应该使用更复杂的图匹配算法
    DesignCalibError err = organize_corners_to_grid(
        corners, num_detected,
        pattern_width, pattern_height,
        pattern->image_points);
    
    DESIGN_CALIB_FREE(corners);
    
    if (err != DESIGN_CALIB_SUCCESS) {
        return err;
    }
    
    return DESIGN_CALIB_SUCCESS;
}

/**
 * @brief 比较浮点数（用于qsort）
 */
static int compare_float(const void *a, const void *b)
{
    float fa = *(const float*)a;
    float fb = *(const float*)b;
    return (fa > fb) - (fa < fb);
}

/**
 * @brief 比较角点（按响应强度降序）
 */
static int compare_corners(const void *a, const void *b)
{
    const Corner *ca = (const Corner*)a;
    const Corner *cb = (const Corner*)b;
    return (cb->response > ca->response) - (cb->response < ca->response);
}

/**
 * @brief 将角点组织成网格
 */
static DesignCalibError organize_corners_to_grid(
    const Corner *corners,
    int num_corners,
    int grid_width,
    int grid_height,
    DesignCalibPoint2D *grid_points)
{
    int num_grid_points = grid_width * grid_height;
    
    if (num_corners < num_grid_points) {
        return DESIGN_CALIB_ERROR_INSUFFICIENT_POINTS;
    }
    
    // 简化版本：假设角点已经大致按网格排列
    // 实际应该使用更复杂的图匹配算法
    
    // 1. 找到最左上角的点作为起点
    int start_idx = 0;
    float min_dist = FLT_MAX;
    
    for (int i = 0; i < num_corners; i++) {
        float dist = corners[i].x * corners[i].x + corners[i].y * corners[i].y;
        if (dist < min_dist) {
            min_dist = dist;
            start_idx = i;
        }
    }
    
    // 2. 使用最近邻搜索构建网格
    bool *used = DESIGN_CALIB_CALLOC(bool, num_corners);
    if (!used) {
        return DESIGN_CALIB_ERROR_MEMORY_ALLOCATION;
    }
    
    grid_points[0].x = corners[start_idx].x;
    grid_points[0].y = corners[start_idx].y;
    used[start_idx] = true;
    
    // 估计网格间距
    float avg_spacing = estimate_grid_spacing(corners, num_corners);
    
    // 逐行构建网格
    for (int row = 0; row < grid_height; row++) {
        for (int col = 0; col < grid_width; col++) {
            if (row == 0 && col == 0) continue;
            
            int grid_idx = row * grid_width + col;
            
            // 预期位置
            float expected_x, expected_y;
            if (col > 0) {
                // 基于同行前一个点
                int prev_idx = row * grid_width + (col - 1);
                expected_x = grid_points[prev_idx].x + avg_spacing;
                expected_y = grid_points[prev_idx].y;
            } else {
                // 基于上一行同列的点
                int above_idx = (row - 1) * grid_width + col;
                expected_x = grid_points[above_idx].x;
                expected_y = grid_points[above_idx].y + avg_spacing;
            }
            
            // 找最近的未使用点
            int best_idx = -1;
            float best_dist = avg_spacing * 0.5f;  // 搜索半径
            
            for (int i = 0; i < num_corners; i++) {
                if (used[i]) continue;
                
                float dx = corners[i].x - expected_x;
                float dy = corners[i].y - expected_y;
                float dist = sqrtf(dx*dx + dy*dy);
                
                if (dist < best_dist) {
                    best_dist = dist;
                    best_idx = i;
                }
            }
            
            if (best_idx < 0) {
                DESIGN_CALIB_FREE(used);
                return DESIGN_CALIB_ERROR_PATTERN_NOT_FOUND;
            }
            
            grid_points[grid_idx].x = corners[best_idx].x;
            grid_points[grid_idx].y = corners[best_idx].y;
            used[best_idx] = true;
        }
    }
    
    DESIGN_CALIB_FREE(used);
    
    return DESIGN_CALIB_SUCCESS;
}

/**
 * @brief 估计网格间距
 */
static float estimate_grid_spacing(const Corner *corners, int num_corners)
{
    if (num_corners < 2) return 10.0f;
    
    // 计算最近邻距离的中位数
    float *distances = DESIGN_CALIB_MALLOC(float, num_corners);
    if (!distances) return 10.0f;
    
    for (int i = 0; i < num_corners; i++) {
        float min_dist = FLT_MAX;
        
        for (int j = 0; j < num_corners; j++) {
            if (i == j) continue;
            
            float dx = corners[i].x - corners[j].x;
            float dy = corners[i].y - corners[j].y;
            float dist = sqrtf(dx*dx + dy*dy);
            
            if (dist < min_dist) {
                min_dist = dist;
            }
        }
        
        distances[i] = min_dist;
    }
    
    // 排序并取中位数
    qsort(distances, num_corners, sizeof(float),
          (int(*)(const void*, const void*))compare_float);
    
    float median = distances[num_corners / 2];
    DESIGN_CALIB_FREE(distances);
    
    return median;
}

/**
 * @brief 评估标定板检测质量
 */
static float evaluate_pattern_quality(const DesignCalibDetectedPattern *pattern)
{
    if (!pattern || !pattern->is_valid || pattern->num_points == 0) {
        return 0.0f;
    }
    
    float quality = 100.0f;
    
    // 1. 检查点的分布均匀性
    float mean_x = 0.0f, mean_y = 0.0f;
    for (int i = 0; i < pattern->num_points; i++) {
        mean_x += pattern->image_points[i].x;
        mean_y += pattern->image_points[i].y;
    }
    mean_x /= pattern->num_points;
    mean_y /= pattern->num_points;
    
    float variance_x = 0.0f, variance_y = 0.0f;
    for (int i = 0; i < pattern->num_points; i++) {
        float dx = pattern->image_points[i].x - mean_x;
        float dy = pattern->image_points[i].y - mean_y;
        variance_x += dx * dx;
        variance_y += dy * dy;
    }
    variance_x /= pattern->num_points;
    variance_y /= pattern->num_points;
    
    float distribution_score = fminf(sqrtf(variance_x * variance_y) / 1000.0f, 1.0f);
    quality *= distribution_score;
    
    // 2. 检查网格规则性（如果是网格图案）
    // 这里简化处理
    
    // 3. 考虑重投影误差（如果已计算）
    if (pattern->reprojection_error > 0.0f) {
        float error_score = 1.0f / (1.0f + pattern->reprojection_error);
        quality *= error_score;
    }
    
    return quality;
}
// ============================================================================
// 圆点检测
// ============================================================================

/**
 * @brief 检测圆点阵列
 */
static DesignCalibError detect_circles(
    const float *image,
    int width,
    int height,
    const DesignCalibPatternConfig *config,
    DesignCalibDetectedPattern *pattern)
{
    int pattern_width = config->pattern_width;
    int pattern_height = config->pattern_height;
    int num_circles = pattern_width * pattern_height;
    
    // 分配内存
    pattern->image_points = DESIGN_CALIB_MALLOC(DesignCalibPoint2D, num_circles);
    pattern->object_points = DESIGN_CALIB_MALLOC(DesignCalibPoint3D, num_circles);
    
    if (!pattern->image_points || !pattern->object_points) {
        DESIGN_CALIB_FREE(pattern->image_points);
        DESIGN_CALIB_FREE(pattern->object_points);
        return DESIGN_CALIB_ERROR_MEMORY_ALLOCATION;
    }
    
    pattern->num_points = num_circles;
    
    // 生成物体坐标点
    float spacing = config->circle_spacing;
    bool asymmetric = (config->type == DESIGN_CALIB_PATTERN_ASYMMETRIC_CIRCLES);
    
    for (int y = 0; y < pattern_height; y++) {
        for (int x = 0; x < pattern_width; x++) {
            int idx = y * pattern_width + x;
            
            if (asymmetric && (y % 2 == 1)) {
                // 非对称模式：奇数行偏移半个间距
                pattern->object_points[idx].x = (x + 0.5f) * spacing;
            } else {
                pattern->object_points[idx].x = x * spacing;
            }
            
            pattern->object_points[idx].y = y * spacing;
            pattern->object_points[idx].z = 0.0f;
        }
    }
    
    // 检测圆点
    typedef struct {
        float x, y;
        float radius;
        float score;
    } Circle;
    
    Circle *circles = DESIGN_CALIB_MALLOC(Circle, num_circles * 2);
    int num_detected = 0;
    
    if (!circles) {
        return DESIGN_CALIB_ERROR_MEMORY_ALLOCATION;
    }
    
    // 使用Hough圆检测
    DesignCalibError err = detect_circles_hough(
        image, width, height,
        circles, &num_detected,
        num_circles * 2);
    
    if (err != DESIGN_CALIB_SUCCESS) {
        DESIGN_CALIB_FREE(circles);
        return err;
    }
    
    if (num_detected < num_circles) {
        DESIGN_CALIB_FREE(circles);
        return DESIGN_CALIB_ERROR_PATTERN_NOT_FOUND;
    }
    
    // 组织圆点成网格
    err = organize_circles_to_grid(
        circles, num_detected,
        pattern_width, pattern_height,
        asymmetric,
        pattern->image_points);
    
    DESIGN_CALIB_FREE(circles);
    
    return err;
}

/**
 * @brief Hough圆检测
 */
static DesignCalibError detect_circles_hough(
    const float *image,
    int width,
    int height,
    Circle *circles,
    int *num_detected,
    int max_circles)
{
    *num_detected = 0;
    
    // 1. 边缘检测（Canny）
    float *edges = DESIGN_CALIB_CALLOC(float, width * height);
    if (!edges) {
        return DESIGN_CALIB_ERROR_MEMORY_ALLOCATION;
    }
    
    DesignCalibError err = canny_edge_detection(
        image, width, height,
        1.0f,  // sigma
        0.1f,  // low threshold
        0.3f,  // high threshold
        edges);
    
    if (err != DESIGN_CALIB_SUCCESS) {
        DESIGN_CALIB_FREE(edges);
        return err;
    }
    
    // 2. Hough变换
    int min_radius = 5;
    int max_radius = 50;
    int num_radii = max_radius - min_radius + 1;
    
    // 累加器
    int *accumulator = DESIGN_CALIB_CALLOC(int, width * height * num_radii);
    if (!accumulator) {
        DESIGN_CALIB_FREE(edges);
        return DESIGN_CALIB_ERROR_MEMORY_ALLOCATION;
    }
    
    // 对每个边缘点投票
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            if (edges[y * width + x] < 0.5f) continue;
            
            // 对每个可能的半径
            for (int r = min_radius; r <= max_radius; r++) {
                int r_idx = r - min_radius;
                
                // 在圆周上采样
                int num_samples = (int)(2 * M_PI * r);
                for (int i = 0; i < num_samples; i++) {
                    float angle = 2.0f * M_PI * i / num_samples;
                    int cx = (int)(x + r * cosf(angle));
                    int cy = (int)(y + r * sinf(angle));
                    
                    if (cx >= 0 && cx < width && cy >= 0 && cy < height) {
                        int acc_idx = (cy * width + cx) * num_radii + r_idx;
                        accumulator[acc_idx]++;
                    }
                }
            }
        }
    }
    
    DESIGN_CALIB_FREE(edges);
    
    // 3. 找到局部最大值
    int threshold = 20;
    
    for (int y = max_radius; y < height - max_radius; y++) {
        for (int x = max_radius; x < width - max_radius; x++) {
            for (int r_idx = 0; r_idx < num_radii; r_idx++) {
                int acc_idx = (y * width + x) * num_radii + r_idx;
                int votes = accumulator[acc_idx];
                
                if (votes < threshold) continue;
                
                // 检查是否为局部最大值
                bool is_maximum = true;
                int search_radius = 5;
                
                for (int dy = -search_radius; dy <= search_radius && is_maximum; dy++) {
                    for (int dx = -search_radius; dx <= search_radius && is_maximum; dx++) {
                        for (int dr = -2; dr <= 2 && is_maximum; dr++) {
                            if (dx == 0 && dy == 0 && dr == 0) continue;
                            
                            int ny = y + dy;
                            int nx = x + dx;
                            int nr_idx = r_idx + dr;
                            
                            if (ny < 0 || ny >= height || nx < 0 || nx >= width ||
                                nr_idx < 0 || nr_idx >= num_radii) continue;
                            
                            int neighbor_idx = (ny * width + nx) * num_radii + nr_idx;
                            if (accumulator[neighbor_idx] > votes) {
                                is_maximum = false;
                            }
                        }
                    }
                }
                
                if (is_maximum && *num_detected < max_circles) {
                    circles[*num_detected].x = x;
                    circles[*num_detected].y = y;
                    circles[*num_detected].radius = min_radius + r_idx;
                    circles[*num_detected].score = votes;
                    (*num_detected)++;
                }
            }
        }
    }
    
    DESIGN_CALIB_FREE(accumulator);
    
    // 4. 精化圆心位置
    for (int i = 0; i < *num_detected; i++) {
        refine_circle_center(
            image, width, height,
            &circles[i].x,
            &circles[i].y,
            circles[i].radius);
    }
    
    return DESIGN_CALIB_SUCCESS;
}

/**
 * @brief Canny边缘检测
 */
static DesignCalibError canny_edge_detection(
    const float *image,
    int width,
    int height,
    float sigma,
    float low_threshold,
    float high_threshold,
    float *edges)
{
    // 1. 高斯平滑
    int kernel_size = (int)(6 * sigma + 1);
    if (kernel_size % 2 == 0) kernel_size++;
    
    float *gaussian_kernel = DESIGN_CALIB_MALLOC(float, kernel_size);
    if (!gaussian_kernel) {
        return DESIGN_CALIB_ERROR_MEMORY_ALLOCATION;
    }
    
    // 生成高斯核
    int radius = kernel_size / 2;
    float sum = 0.0f;
    for (int i = 0; i < kernel_size; i++) {
        int x = i - radius;
        gaussian_kernel[i] = expf(-(x * x) / (2 * sigma * sigma));
        sum += gaussian_kernel[i];
    }
    for (int i = 0; i < kernel_size; i++) {
        gaussian_kernel[i] /= sum;
    }
    
    // 应用高斯滤波
    float *smoothed = DESIGN_CALIB_MALLOC(float, width * height);
    float *temp = DESIGN_CALIB_MALLOC(float, width * height);
    
    if (!smoothed || !temp) {
        DESIGN_CALIB_FREE(gaussian_kernel);
        DESIGN_CALIB_FREE(smoothed);
        DESIGN_CALIB_FREE(temp);
        return DESIGN_CALIB_ERROR_MEMORY_ALLOCATION;
    }
    
    // X方向
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float sum = 0.0f;
            for (int k = 0; k < kernel_size; k++) {
                int xx = x + k - radius;
                if (xx >= 0 && xx < width) {
                    sum += image[y * width + xx] * gaussian_kernel[k];
                }
            }
            temp[y * width + x] = sum;
        }
    }
    
    // Y方向
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float sum = 0.0f;
            for (int k = 0; k < kernel_size; k++) {
                int yy = y + k - radius;
                if (yy >= 0 && yy < height) {
                    sum += temp[yy * width + x] * gaussian_kernel[k];
                }
            }
            smoothed[y * width + x] = sum;
        }
    }
    
    DESIGN_CALIB_FREE(gaussian_kernel);
    DESIGN_CALIB_FREE(temp);
    
    // 2. 计算梯度
    float *grad_mag = DESIGN_CALIB_MALLOC(float, width * height);
    float *grad_dir = DESIGN_CALIB_MALLOC(float, width * height);
    
    if (!grad_mag || !grad_dir) {
        DESIGN_CALIB_FREE(smoothed);
        DESIGN_CALIB_FREE(grad_mag);
        DESIGN_CALIB_FREE(grad_dir);
        return DESIGN_CALIB_ERROR_MEMORY_ALLOCATION;
    }
    
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            int idx = y * width + x;
            
            // Sobel算子
            float gx = 
                -smoothed[(y-1)*width + (x-1)] + smoothed[(y-1)*width + (x+1)] +
                -2*smoothed[y*width + (x-1)] + 2*smoothed[y*width + (x+1)] +
                -smoothed[(y+1)*width + (x-1)] + smoothed[(y+1)*width + (x+1)];
            
            float gy = 
                -smoothed[(y-1)*width + (x-1)] - 2*smoothed[(y-1)*width + x] - 
                smoothed[(y-1)*width + (x+1)] +
                smoothed[(y+1)*width + (x-1)] + 2*smoothed[(y+1)*width + x] + 
                smoothed[(y+1)*width + (x+1)];
            
            grad_mag[idx] = sqrtf(gx * gx + gy * gy);
            grad_dir[idx] = atan2f(gy, gx);
        }
    }
    
    DESIGN_CALIB_FREE(smoothed);
    
    // 3. 非极大值抑制
    float *nms = DESIGN_CALIB_CALLOC(float, width * height);
    if (!nms) {
        DESIGN_CALIB_FREE(grad_mag);
        DESIGN_CALIB_FREE(grad_dir);
        return DESIGN_CALIB_ERROR_MEMORY_ALLOCATION;
    }
    
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            int idx = y * width + x;
            float mag = grad_mag[idx];
            float dir = grad_dir[idx];
            
            // 量化方向到4个主方向
            float angle = dir * 180.0f / M_PI;
            if (angle < 0) angle += 180.0f;
            
            float mag1, mag2;
            
            if ((angle >= 0 && angle < 22.5f) || (angle >= 157.5f && angle <= 180.0f)) {
                mag1 = grad_mag[y * width + (x-1)];
                mag2 = grad_mag[y * width + (x+1)];
            } else if (angle >= 22.5f && angle < 67.5f) {
                mag1 = grad_mag[(y-1) * width + (x+1)];
                mag2 = grad_mag[(y+1) * width + (x-1)];
            } else if (angle >= 67.5f && angle < 112.5f) {
                mag1 = grad_mag[(y-1) * width + x];
                mag2 = grad_mag[(y+1) * width + x];
            } else {
                mag1 = grad_mag[(y-1) * width + (x-1)];
                mag2 = grad_mag[(y+1) * width + (x+1)];
            }
            
            if (mag >= mag1 && mag >= mag2) {
                nms[idx] = mag;
            }
        }
    }
    
    DESIGN_CALIB_FREE(grad_mag);
    DESIGN_CALIB_FREE(grad_dir);
    
    // 4. 双阈值和边缘跟踪
    float max_mag = 0.0f;
    for (int i = 0; i < width * height; i++) {
        if (nms[i] > max_mag) max_mag = nms[i];
    }
    
    if (max_mag > 0.0f) {
        for (int i = 0; i < width * height; i++) {
            nms[i] /= max_mag;
        }
    }
    
    memset(edges, 0, width * height * sizeof(float));
    
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            int idx = y * width + x;
            
            if (nms[idx] >= high_threshold) {
                edges[idx] = 1.0f;
            } else if (nms[idx] >= low_threshold) {
                // 检查8邻域
                bool has_strong = false;
                for (int dy = -1; dy <= 1 && !has_strong; dy++) {
                    for (int dx = -1; dx <= 1 && !has_strong; dx++) {
                        if (dx == 0 && dy == 0) continue;
                        if (nms[(y+dy)*width + (x+dx)] >= high_threshold) {
                            has_strong = true;
                        }
                    }
                }
                if (has_strong) edges[idx] = 1.0f;
            }
        }
    }
    
    DESIGN_CALIB_FREE(nms);
    
    return DESIGN_CALIB_SUCCESS;
}

/**
 * @brief 精化圆心位置
 */
static void refine_circle_center(
    const float *image,
    int width,
    int height,
    float *cx,
    float *cy,
    float radius)
{
    int search_radius = (int)(radius * 1.5f);
    int x0 = (int)*cx;
    int y0 = (int)*cy;
    
    float sum_x = 0.0f, sum_y = 0.0f, sum_w = 0.0f;
    
    for (int dy = -search_radius; dy <= search_radius; dy++) {
        for (int dx = -search_radius; dx <= search_radius; dx++) {
            int x = x0 + dx;
            int y = y0 + dy;
            
            if (x < 0 || x >= width || y < 0 || y >= height) continue;
            
            float dist = sqrtf(dx * dx + dy * dy);
            if (dist > search_radius) continue;
            
            float weight = expf(-fabsf(dist - radius) / (radius * 0.2f));
            weight *= image[y * width + x];
            
            sum_x += x * weight;
            sum_y += y * weight;
            sum_w += weight;
        }
    }
    
    if (sum_w > 0.0f) {
        *cx = sum_x / sum_w;
        *cy = sum_y / sum_w;
    }
}
/**
 * @brief 将圆点组织成网格
 */
static DesignCalibError organize_circles_to_grid(
    const Circle *circles,
    int num_circles,
    int grid_width,
    int grid_height,
    bool asymmetric,
    DesignCalibPoint2D *grid_points)
{
    int num_grid_points = grid_width * grid_height;
    
    if (num_circles < num_grid_points) {
        return DESIGN_CALIB_ERROR_INSUFFICIENT_POINTS;
    }
    
    // 1. 找到最左上角的圆
    int start_idx = 0;
    float min_dist = FLT_MAX;
    
    for (int i = 0; i < num_circles; i++) {
        float dist = circles[i].x * circles[i].x + circles[i].y * circles[i].y;
        if (dist < min_dist) {
            min_dist = dist;
            start_idx = i;
        }
    }
    
    // 2. 估计网格间距
    float avg_spacing = 0.0f;
    int spacing_count = 0;
    
    for (int i = 0; i < num_circles; i++) {
        float min_neighbor_dist = FLT_MAX;
        
        for (int j = 0; j < num_circles; j++) {
            if (i == j) continue;
            
            float dx = circles[i].x - circles[j].x;
            float dy = circles[i].y - circles[j].y;
            float dist = sqrtf(dx*dx + dy*dy);
            
            if (dist < min_neighbor_dist) {
                min_neighbor_dist = dist;
            }
        }
        
        if (min_neighbor_dist < FLT_MAX) {
            avg_spacing += min_neighbor_dist;
            spacing_count++;
        }
    }
    
    if (spacing_count > 0) {
        avg_spacing /= spacing_count;
    } else {
        return DESIGN_CALIB_ERROR_PATTERN_NOT_FOUND;
    }
    
    // 3. 构建网格
    bool *used = DESIGN_CALIB_CALLOC(bool, num_circles);
    if (!used) {
        return DESIGN_CALIB_ERROR_MEMORY_ALLOCATION;
    }
    
    grid_points[0].x = circles[start_idx].x;
    grid_points[0].y = circles[start_idx].y;
    used[start_idx] = true;
    
    for (int row = 0; row < grid_height; row++) {
        for (int col = 0; col < grid_width; col++) {
            if (row == 0 && col == 0) continue;
            
            int grid_idx = row * grid_width + col;
            
            // 预期位置
            float expected_x, expected_y;
            
            if (col > 0) {
                int prev_idx = row * grid_width + (col - 1);
                expected_x = grid_points[prev_idx].x + avg_spacing;
                expected_y = grid_points[prev_idx].y;
            } else {
                int above_idx = (row - 1) * grid_width + col;
                expected_x = grid_points[above_idx].x;
                expected_y = grid_points[above_idx].y + avg_spacing;
                
                // 非对称模式：奇数行偏移
                if (asymmetric && (row % 2 == 1)) {
                    expected_x += avg_spacing * 0.5f;
                }
            }
            
            // 找最近的未使用圆
            int best_idx = -1;
            float best_dist = avg_spacing * 0.5f;
            
            for (int i = 0; i < num_circles; i++) {
                if (used[i]) continue;
                
                float dx = circles[i].x - expected_x;
                float dy = circles[i].y - expected_y;
                float dist = sqrtf(dx*dx + dy*dy);
                
                if (dist < best_dist) {
                    best_dist = dist;
                    best_idx = i;
                }
            }
            
            if (best_idx < 0) {
                DESIGN_CALIB_FREE(used);
                return DESIGN_CALIB_ERROR_PATTERN_NOT_FOUND;
            }
            
            grid_points[grid_idx].x = circles[best_idx].x;
            grid_points[grid_idx].y = circles[best_idx].y;
            used[best_idx] = true;
        }
    }
    
    DESIGN_CALIB_FREE(used);
    
    return DESIGN_CALIB_SUCCESS;
}

// ============================================================================
// 亚像素精化
// ============================================================================

DesignCalibError design_calib_refine_corners(
    const float *image,
    int width,
    int height,
    DesignCalibPoint2D *corners,
    int num_corners,
    float window_size)
{
    DESIGN_CALIB_CHECK_NULL(image);
    DESIGN_CALIB_CHECK_NULL(corners);
    
    if (width <= 0 || height <= 0 || num_corners <= 0) {
        return DESIGN_CALIB_ERROR_INVALID_PARAM;
    }
    
    return refine_corners_subpixel(
        image, width, height,
        corners, num_corners,
        window_size);
}

/**
 * @brief 亚像素角点精化
 */
static DesignCalibError refine_corners_subpixel(
    const float *image,
    int width,
    int height,
    DesignCalibPoint2D *corners,
    int num_corners,
    float window_size)
{
    int half_win = (int)(window_size / 2);
    int max_iterations = 30;
    float epsilon = 0.01f;
    
    // 计算图像梯度
    float *grad_x = DESIGN_CALIB_MALLOC(float, width * height);
    float *grad_y = DESIGN_CALIB_MALLOC(float, width * height);
    
    if (!grad_x || !grad_y) {
        DESIGN_CALIB_FREE(grad_x);
        DESIGN_CALIB_FREE(grad_y);
        return DESIGN_CALIB_ERROR_MEMORY_ALLOCATION;
    }
    
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            int idx = y * width + x;
            grad_x[idx] = (image[idx + 1] - image[idx - 1]) * 0.5f;
            grad_y[idx] = (image[idx + width] - image[idx - width]) * 0.5f;
        }
    }
    
    // 对每个角点进行精化
    for (int i = 0; i < num_corners; i++) {
        float cx = corners[i].x;
        float cy = corners[i].y;
        
        for (int iter = 0; iter < max_iterations; iter++) {
            int icx = (int)cx;
            int icy = (int)cy;
            
            if (icx < half_win || icx >= width - half_win ||
                icy < half_win || icy >= height - half_win) {
                break;
            }
            
            // 计算窗口内的加权质心
            double sum_x = 0.0, sum_y = 0.0;
            double sum_gx2 = 0.0, sum_gy2 = 0.0, sum_gxgy = 0.0;
            
            for (int dy = -half_win; dy <= half_win; dy++) {
                for (int dx = -half_win; dx <= half_win; dx++) {
                    int x = icx + dx;
                    int y = icy + dy;
                    int idx = y * width + x;
                    
                    float gx = grad_x[idx];
                    float gy = grad_y[idx];
                    
                    sum_gx2 += gx * gx;
                    sum_gy2 += gy * gy;
                    sum_gxgy += gx * gy;
                    sum_x += gx * gx * dx + gx * gy * dy;
                    sum_y += gx * gy * dx + gy * gy * dy;
                }
            }
            
            // 求解线性系统
            double det = sum_gx2 * sum_gy2 - sum_gxgy * sum_gxgy;
            if (fabs(det) < 1e-10) break;
            
            double dx = (sum_gy2 * sum_x - sum_gxgy * sum_y) / det;
            double dy = (sum_gx2 * sum_y - sum_gxgy * sum_x) / det;
            
            cx += dx;
            cy += dy;
            
            if (fabs(dx) < epsilon && fabs(dy) < epsilon) {
                break;
            }
        }
        
        corners[i].x = cx;
        corners[i].y = cy;
    }
    
    DESIGN_CALIB_FREE(grad_x);
    DESIGN_CALIB_FREE(grad_y);
    
    return DESIGN_CALIB_SUCCESS;
}

// ============================================================================
// 批量检测
// ============================================================================

DesignCalibError design_calib_detect_patterns_batch(
    DesignCalibContext *context,
    const float **images,
    int num_images,
    int width,
    int height,
    DesignCalibDetectedPattern ***patterns,
    int *num_detected)
{
    DESIGN_CALIB_CHECK_NULL(context);
    DESIGN_CALIB_CHECK_NULL(images);
    DESIGN_CALIB_CHECK_NULL(patterns);
    DESIGN_CALIB_CHECK_NULL(num_detected);
    
    if (num_images <= 0 || width <= 0 || height <= 0) {
        return DESIGN_CALIB_ERROR_INVALID_PARAM;
    }
    
    // 分配结果数组
    DesignCalibDetectedPattern **results = 
        DESIGN_CALIB_MALLOC(DesignCalibDetectedPattern*, num_images);
    if (!results) {
        return DESIGN_CALIB_ERROR_MEMORY_ALLOCATION;
    }
    
    *num_detected = 0;
    
    // 检测每张图像
    for (int i = 0; i < num_images; i++) {
        if (context->cancel_requested) {
            // 清理已检测的
            for (int j = 0; j < *num_detected; j++) {
                design_calib_free_detected_pattern(results[j]);
            }
            DESIGN_CALIB_FREE(results);
            return DESIGN_CALIB_ERROR_OPERATION_CANCELLED;
        }
        
        update_progress(context, "Detecting patterns", i, num_images);
        
        DesignCalibDetectedPattern *pattern = NULL;
        DesignCalibError err = design_calib_detect_pattern(
            context,
            images[i],
            width,
            height,
            &pattern);
        
        if (err == DESIGN_CALIB_SUCCESS && pattern && pattern->is_valid) {
            results[*num_detected] = pattern;
            (*num_detected)++;
        } else {
            if (pattern) {
                design_calib_free_detected_pattern(pattern);
            }
        }
    }
    
    // 检查是否有足够的有效图像
    if (*num_detected < context->config.min_num_images) {
        for (int i = 0; i < *num_detected; i++) {
            design_calib_free_detected_pattern(results[i]);
        }
        DESIGN_CALIB_FREE(results);
        return DESIGN_CALIB_ERROR_INSUFFICIENT_IMAGES;
    }
    
    *patterns = results;
    
    return DESIGN_CALIB_SUCCESS;
}

void design_calib_free_detected_pattern(DesignCalibDetectedPattern *pattern)
{
    if (!pattern) return;
    
    DESIGN_CALIB_FREE(pattern->image_points);
    DESIGN_CALIB_FREE(pattern->object_points);
    DESIGN_CALIB_FREE(pattern);
}

// ============================================================================
// 相机校准主函数
// ============================================================================

DesignCalibError design_calib_calibrate_camera(
    DesignCalibContext *context,
    const DesignCalibDetectedPattern **patterns,
    int num_patterns,
    int image_width,
    int image_height,
    DesignCalibResult **result)
{
    DESIGN_CALIB_CHECK_NULL(context);
    DESIGN_CALIB_CHECK_NULL(patterns);
    DESIGN_CALIB_CHECK_NULL(result);
    
    if (num_patterns < context->config.min_num_images) {
        return DESIGN_CALIB_ERROR_INSUFFICIENT_IMAGES;
    }
    
    if (image_width <= 0 || image_height <= 0) {
        return DESIGN_CALIB_ERROR_INVALID_PARAM;
    }
    
    context->status = DESIGN_CALIB_STATUS_CALIBRATING;
    context->start_time = time(NULL);
    
    // 分配结果结构
    DesignCalibResult *calib_result = DESIGN_CALIB_CALLOC(DesignCalibResult, 1);
    if (!calib_result) {
        return DESIGN_CALIB_ERROR_MEMORY_ALLOCATION;
    }
    
    calib_result->image_width = image_width;
    calib_result->image_height = image_height;
    calib_result->num_images_used = num_patterns;
    
    // 分配每图像误差数组
    calib_result->per_image_errors = 
        DESIGN_CALIB_MALLOC(float, num_patterns);
    if (!calib_result->per_image_errors) {
        design_calib_free_result(calib_result);
        return DESIGN_CALIB_ERROR_MEMORY_ALLOCATION;
    }
    
    // 1. 初始化相机参数
    update_progress(context, "Initializing camera parameters", 0, 100);
    
    DesignCalibError err = initialize_camera_parameters(
        patterns,
        num_patterns,
        image_width,
        image_height,
        &calib_result->intrinsics);
    
    if (err != DESIGN_CALIB_SUCCESS) {
        design_calib_free_result(calib_result);
        return err;
    }
    
    // 2. 初始化外参（每个图像的位姿）
    update_progress(context, "Computing initial poses", 10, 100);
    
    OptimizationParams opt_params;
    opt_params.num_images = num_patterns;
    
    // 从内参复制
    opt_params.fx = calib_result->intrinsics.fx;
    opt_params.fy = calib_result->intrinsics.fy;
    opt_params.cx = calib_result->intrinsics.cx;
    opt_params.cy = calib_result->intrinsics.cy;
    opt_params.skew = calib_result->intrinsics.skew;
    opt_params.k1 = calib_result->intrinsics.k1;
    opt_params.k2 = calib_result->intrinsics.k2;
    opt_params.k3 = calib_result->intrinsics.k3;
    opt_params.p1 = calib_result->intrinsics.p1;
    opt_params.p2 = calib_result->intrinsics.p2;
    opt_params.k4 = calib_result->intrinsics.k4;
    opt_params.k5 = calib_result->intrinsics.k5;
    opt_params.k6 = calib_result->intrinsics.k6;
    
    // 分配外参
    opt_params.rotations = DESIGN_CALIB_CALLOC(double, 3 * num_patterns);
    opt_params.translations = DESIGN_CALIB_CALLOC(double, 3 * num_patterns);
    
    if (!opt_params.rotations || !opt_params.translations) {
        DESIGN_CALIB_FREE(opt_params.rotations);
        DESIGN_CALIB_FREE(opt_params.translations);
        design_calib_free_result(calib_result);
        return DESIGN_CALIB_ERROR_MEMORY_ALLOCATION;
    }
    
    // 计算初始外参
    for (int i = 0; i < num_patterns; i++) {
        err = estimate_pose_pnp(
            patterns[i],
            &calib_result->intrinsics,
            &opt_params.rotations[i * 3],
            &opt_params.translations[i * 3]);
        
        if (err != DESIGN_CALIB_SUCCESS) {
            DESIGN_CALIB_FREE(opt_params.rotations);
            DESIGN_CALIB_FREE(opt_params.translations);
            design_calib_free_result(calib_result);
            return err;
        }
    }
    
    // 3. 非线性优化
    update_progress(context, "Optimizing parameters", 30, 100);
    
    OptimizationProblem problem;
    problem.patterns = patterns;
    problem.num_patterns = num_patterns;
    problem.image_width = image_width;
    problem.image_height = image_height;
    problem.distortion_model = context->config.distortion_model;
    problem.fix_principal_point = context->config.fix_principal_point;
    problem.fix_aspect_ratio = context->config.fix_aspect_ratio;
    problem.zero_tangent_dist = context->config.zero_tangent_dist;
    
    err = optimize_camera_parameters(&problem, &opt_params);
    
    if (err != DESIGN_CALIB_SUCCESS) {
        DESIGN_CALIB_FREE(opt_params.rotations);
        DESIGN_CALIB_FREE(opt_params.translations);
        design_calib_free_result(calib_result);
        return err;
    }
    
    // 4. 更新结果
    calib_result->intrinsics.fx = opt_params.fx;
    calib_result->intrinsics.fy = opt_params.fy;
    calib_result->intrinsics.cx = opt_params.cx;
    calib_result->intrinsics.cy = opt_params.cy;
    calib_result->intrinsics.skew = opt_params.skew;
    calib_result->intrinsics.k1 = opt_params.k1;
    calib_result->intrinsics.k2 = opt_params.k2;
    calib_result->intrinsics.k3 = opt_params.k3;
    calib_result->intrinsics.p1 = opt_params.p1;
    calib_result->intrinsics.p2 = opt_params.p2;
    calib_result->intrinsics.k4 = opt_params.k4;
    calib_result->intrinsics.k5 = opt_params.k5;
    calib_result->intrinsics.k6 = opt_params.k6;
    
    // 5. 计算重投影误差
    update_progress(context, "Computing reprojection errors", 90, 100);
    
    double total_error = 0.0;
    int total_points = 0;
    
    for (int i = 0; i < num_patterns; i++) {
        double image_error = 0.0;
        
        for (int j = 0; j < patterns[i]->num_points; j++) {
            // 投影3D点
            DesignCalibPoint2D projected;
            project_point_3d_to_2d(
                &patterns[i]->object_points[j],
                &opt_params.rotations[i * 3],
                &opt_params.translations[i * 3],
                &calib_result->intrinsics,
                &projected);
            
            // 计算误差
            float dx = projected.x - patterns[i]->image_points[j].x;
            float dy = projected.y - patterns[i]->image_points[j].y;
            float error = sqrtf(dx*dx + dy*dy);
            
            image_error += error;
            total_error += error;
            total_points++;
        }
        
        calib_result->per_image_errors[i] = image_error / patterns[i]->num_points;
    }
    
    calib_result->mean_reprojection_error = total_error / total_points;
    
    // 计算标准差
    double variance = 0.0;
    for (int i = 0; i < num_patterns; i++) {
        for (int j = 0; j < patterns[i]->num_points; j++) {
            DesignCalibPoint2D projected;
            project_point_3d_to_2d(
                &patterns[i]->object_points[j],
                &opt_params.rotations[i * 3],
                &opt_params.translations[i * 3],
                &calib_result->intrinsics,
                &projected);
            
            float dx = projected.x - patterns[i]->image_points[j].x;
            float dy = projected.y - patterns[i]->image_points[j].y;
            float error = sqrtf(dx*dx + dy*dy);
            
            double diff = error - calib_result->mean_reprojection_error;
            variance += diff * diff;
        }
    }
    
    calib_result->std_reprojection_error = sqrt(variance / total_points);
    
    // 清理
    DESIGN_CALIB_FREE(opt_params.rotations);
    DESIGN_CALIB_FREE(opt_params.translations);
    
    context->end_time = time(NULL);
    context->status = DESIGN_CALIB_STATUS_IDLE;
    
    update_progress(context, "Calibration complete", 100, 100);
    
    *result = calib_result;
    
    return DESIGN_CALIB_SUCCESS;
}

void design_calib_free_result(DesignCalibResult *result)
{
    if (!result) return;
    
    DESIGN_CALIB_FREE(result->per_image_errors);
    DESIGN_CALIB_FREE(result);
}
// ============================================================================
// 相机参数初始化
// ============================================================================

/**
 * @brief 初始化相机内参
 */
static DesignCalibError initialize_camera_parameters(
    const DesignCalibDetectedPattern **patterns,
    int num_patterns,
    int image_width,
    int image_height,
    DesignCalibIntrinsics *intrinsics)
{
    // 1. 初始化主点为图像中心
    intrinsics->cx = image_width / 2.0;
    intrinsics->cy = image_height / 2.0;
    intrinsics->skew = 0.0;
    
    // 2. 使用单应性矩阵估计焦距
    // 收集所有单应性矩阵
    int max_homographies = num_patterns;
    double *homographies = DESIGN_CALIB_MALLOC(double, 9 * max_homographies);
    int num_homographies = 0;
    
    if (!homographies) {
        return DESIGN_CALIB_ERROR_MEMORY_ALLOCATION;
    }
    
    for (int i = 0; i < num_patterns; i++) {
        double H[9];
        DesignCalibError err = compute_homography(
            patterns[i]->object_points,
            patterns[i]->image_points,
            patterns[i]->num_points,
            H);
        
        if (err == DESIGN_CALIB_SUCCESS) {
            memcpy(&homographies[num_homographies * 9], H, 9 * sizeof(double));
            num_homographies++;
        }
    }
    
    if (num_homographies < 2) {
        DESIGN_CALIB_FREE(homographies);
        return DESIGN_CALIB_ERROR_INSUFFICIENT_IMAGES;
    }
    
    // 3. 从单应性矩阵估计内参
    DesignCalibError err = estimate_intrinsics_from_homographies(
        homographies,
        num_homographies,
        intrinsics);
    
    DESIGN_CALIB_FREE(homographies);
    
    if (err != DESIGN_CALIB_SUCCESS) {
        // 如果失败，使用默认值
        double focal_length = (image_width + image_height) / 2.0;
        intrinsics->fx = focal_length;
        intrinsics->fy = focal_length;
    }
    
    // 4. 初始化畸变参数为0
    intrinsics->k1 = 0.0;
    intrinsics->k2 = 0.0;
    intrinsics->k3 = 0.0;
    intrinsics->p1 = 0.0;
    intrinsics->p2 = 0.0;
    intrinsics->k4 = 0.0;
    intrinsics->k5 = 0.0;
    intrinsics->k6 = 0.0;
    
    return DESIGN_CALIB_SUCCESS;
}

/**
 * @brief 计算单应性矩阵 (DLT算法)
 */
static DesignCalibError compute_homography(
    const DesignCalibPoint3D *object_points,
    const DesignCalibPoint2D *image_points,
    int num_points,
    double H[9])
{
    if (num_points < 4) {
        return DESIGN_CALIB_ERROR_INSUFFICIENT_POINTS;
    }
    
    // 构建 A 矩阵 (2n x 9)
    double *A = DESIGN_CALIB_CALLOC(double, 2 * num_points * 9);
    if (!A) {
        return DESIGN_CALIB_ERROR_MEMORY_ALLOCATION;
    }
    
    for (int i = 0; i < num_points; i++) {
        double X = object_points[i].x;
        double Y = object_points[i].y;
        double x = image_points[i].x;
        double y = image_points[i].y;
        
        // 第一行
        int row1 = 2 * i;
        A[row1 * 9 + 0] = X;
        A[row1 * 9 + 1] = Y;
        A[row1 * 9 + 2] = 1.0;
        A[row1 * 9 + 3] = 0.0;
        A[row1 * 9 + 4] = 0.0;
        A[row1 * 9 + 5] = 0.0;
        A[row1 * 9 + 6] = -x * X;
        A[row1 * 9 + 7] = -x * Y;
        A[row1 * 9 + 8] = -x;
        
        // 第二行
        int row2 = 2 * i + 1;
        A[row2 * 9 + 0] = 0.0;
        A[row2 * 9 + 1] = 0.0;
        A[row2 * 9 + 2] = 0.0;
        A[row2 * 9 + 3] = X;
        A[row2 * 9 + 4] = Y;
        A[row2 * 9 + 5] = 1.0;
        A[row2 * 9 + 6] = -y * X;
        A[row2 * 9 + 7] = -y * Y;
        A[row2 * 9 + 8] = -y;
    }
    
    // SVD分解求解
    double *U = DESIGN_CALIB_MALLOC(double, 2 * num_points * 2 * num_points);
    double *S = DESIGN_CALIB_MALLOC(double, 9);
    double *Vt = DESIGN_CALIB_MALLOC(double, 9 * 9);
    
    if (!U || !S || !Vt) {
        DESIGN_CALIB_FREE(A);
        DESIGN_CALIB_FREE(U);
        DESIGN_CALIB_FREE(S);
        DESIGN_CALIB_FREE(Vt);
        return DESIGN_CALIB_ERROR_MEMORY_ALLOCATION;
    }
    
    DesignCalibError err = svd_decomposition(
        A, 2 * num_points, 9,
        U, S, Vt);
    
    DESIGN_CALIB_FREE(A);
    DESIGN_CALIB_FREE(U);
    DESIGN_CALIB_FREE(S);
    
    if (err != DESIGN_CALIB_SUCCESS) {
        DESIGN_CALIB_FREE(Vt);
        return err;
    }
    
    // H是V的最后一列（Vt的最后一行）
    for (int i = 0; i < 9; i++) {
        H[i] = Vt[8 * 9 + i];
    }
    
    DESIGN_CALIB_FREE(Vt);
    
    // 归一化使H[8] = 1
    if (fabs(H[8]) > 1e-10) {
        for (int i = 0; i < 9; i++) {
            H[i] /= H[8];
        }
    }
    
    return DESIGN_CALIB_SUCCESS;
}

/**
 * @brief 从单应性矩阵估计内参
 * 使用Zhang的方法
 */
static DesignCalibError estimate_intrinsics_from_homographies(
    const double *homographies,
    int num_homographies,
    DesignCalibIntrinsics *intrinsics)
{
    // 构建约束矩阵 V (2n x 6)
    double *V = DESIGN_CALIB_CALLOC(double, 2 * num_homographies * 6);
    if (!V) {
        return DESIGN_CALIB_ERROR_MEMORY_ALLOCATION;
    }
    
    for (int i = 0; i < num_homographies; i++) {
        const double *H = &homographies[i * 9];
        
        // v12
        double v12[6];
        compute_v_ij(H, 0, 1, v12);
        
        // v11
        double v11[6];
        compute_v_ij(H, 0, 0, v11);
        
        // v22
        double v22[6];
        compute_v_ij(H, 1, 1, v22);
        
        // 第一个约束: v12
        int row1 = 2 * i;
        for (int j = 0; j < 6; j++) {
            V[row1 * 6 + j] = v12[j];
        }
        
        // 第二个约束: v11 - v22
        int row2 = 2 * i + 1;
        for (int j = 0; j < 6; j++) {
            V[row2 * 6 + j] = v11[j] - v22[j];
        }
    }
    
    // SVD求解 Vb = 0
    double *U = DESIGN_CALIB_MALLOC(double, 2 * num_homographies * 2 * num_homographies);
    double *S = DESIGN_CALIB_MALLOC(double, 6);
    double *Vt = DESIGN_CALIB_MALLOC(double, 6 * 6);
    
    if (!U || !S || !Vt) {
        DESIGN_CALIB_FREE(V);
        DESIGN_CALIB_FREE(U);
        DESIGN_CALIB_FREE(S);
        DESIGN_CALIB_FREE(Vt);
        return DESIGN_CALIB_ERROR_MEMORY_ALLOCATION;
    }
    
    DesignCalibError err = svd_decomposition(
        V, 2 * num_homographies, 6,
        U, S, Vt);
    
    DESIGN_CALIB_FREE(V);
    DESIGN_CALIB_FREE(U);
    DESIGN_CALIB_FREE(S);
    
    if (err != DESIGN_CALIB_SUCCESS) {
        DESIGN_CALIB_FREE(Vt);
        return err;
    }
    
    // b是Vt的最后一行
    double b[6];
    for (int i = 0; i < 6; i++) {
        b[i] = Vt[5 * 6 + i];
    }
    
    DESIGN_CALIB_FREE(Vt);
    
    // 从b恢复内参
    // b = [B11, B12, B22, B13, B23, B33]
    double B11 = b[0];
    double B12 = b[1];
    double B22 = b[2];
    double B13 = b[3];
    double B23 = b[4];
    double B33 = b[5];
    
    // 计算内参
    double v0 = (B12 * B13 - B11 * B23) / (B11 * B22 - B12 * B12);
    double lambda = B33 - (B13 * B13 + v0 * (B12 * B13 - B11 * B23)) / B11;
    
    if (lambda <= 0 || B11 <= 0 || B22 <= 0) {
        return DESIGN_CALIB_ERROR_OPTIMIZATION_FAILED;
    }
    
    double alpha = sqrt(lambda / B11);
    double beta = sqrt(lambda * B11 / (B11 * B22 - B12 * B12));
    double gamma = -B12 * alpha * alpha * beta / lambda;
    double u0 = gamma * v0 / beta - B13 * alpha * alpha / lambda;
    
    intrinsics->fx = alpha;
    intrinsics->fy = beta;
    intrinsics->cx = u0;
    intrinsics->cy = v0;
    intrinsics->skew = gamma;
    
    return DESIGN_CALIB_SUCCESS;
}

/**
 * @brief 计算v_ij向量
 */
static void compute_v_ij(const double H[9], int i, int j, double v[6])
{
    double h_i0 = H[i * 3 + 0];
    double h_i1 = H[i * 3 + 1];
    double h_i2 = H[i * 3 + 2];
    double h_j0 = H[j * 3 + 0];
    double h_j1 = H[j * 3 + 1];
    double h_j2 = H[j * 3 + 2];
    
    v[0] = h_i0 * h_j0;
    v[1] = h_i0 * h_j1 + h_i1 * h_j0;
    v[2] = h_i1 * h_j1;
    v[3] = h_i2 * h_j0 + h_i0 * h_j2;
    v[4] = h_i2 * h_j1 + h_i1 * h_j2;
    v[5] = h_i2 * h_j2;
}

// ============================================================================
// PnP位姿估计
// ============================================================================

/**
 * @brief 使用PnP算法估计相机位姿
 */
static DesignCalibError estimate_pose_pnp(
    const DesignCalibDetectedPattern *pattern,
    const DesignCalibIntrinsics *intrinsics,
    double rotation[3],
    double translation[3])
{
    if (pattern->num_points < 4) {
        return DESIGN_CALIB_ERROR_INSUFFICIENT_POINTS;
    }
    
    // 1. 使用DLT获得初始估计
    DesignCalibError err = estimate_pose_dlt(
        pattern,
        intrinsics,
        rotation,
        translation);
    
    if (err != DESIGN_CALIB_SUCCESS) {
        return err;
    }
    
    // 2. 使用Levenberg-Marquardt优化
    err = refine_pose_lm(
        pattern,
        intrinsics,
        rotation,
        translation);
    
    return err;
}

/**
 * @brief DLT位姿估计
 */
static DesignCalibError estimate_pose_dlt(
    const DesignCalibDetectedPattern *pattern,
    const DesignCalibIntrinsics *intrinsics,
    double rotation[3],
    double translation[3])
{
    int n = pattern->num_points;
    
    // 构建投影矩阵 P = K[R|t]
    // 首先通过单应性估计
    double H[9];
    DesignCalibError err = compute_homography(
        pattern->object_points,
        pattern->image_points,
        n,
        H);
    
    if (err != DESIGN_CALIB_SUCCESS) {
        return err;
    }
    
    // 从H恢复R和t
    // H = K[r1 r2 t]
    // 其中r1, r2是旋转矩阵的前两列
    
    // 计算K的逆
    double K[9] = {
        intrinsics->fx, intrinsics->skew, intrinsics->cx,
        0.0, intrinsics->fy, intrinsics->cy,
        0.0, 0.0, 1.0
    };
    
    double K_inv[9];
    if (!invert_3x3_matrix(K, K_inv)) {
        return DESIGN_CALIB_ERROR_SINGULAR_MATRIX;
    }
    
    // H_norm = K_inv * H
    double H_norm[9];
    matrix_multiply_3x3(K_inv, H, H_norm);
    
    // 提取r1, r2, t
    double r1[3] = {H_norm[0], H_norm[3], H_norm[6]};
    double r2[3] = {H_norm[1], H_norm[4], H_norm[7]};
    double t[3] = {H_norm[2], H_norm[5], H_norm[8]};
    
    // 归一化
    double norm1 = sqrt(r1[0]*r1[0] + r1[1]*r1[1] + r1[2]*r1[2]);
    double norm2 = sqrt(r2[0]*r2[0] + r2[1]*r2[1] + r2[2]*r2[2]);
    double scale = (norm1 + norm2) / 2.0;
    
    if (scale < 1e-10) {
        return DESIGN_CALIB_ERROR_INVALID_PARAM;
    }
    
    for (int i = 0; i < 3; i++) {
        r1[i] /= scale;
        r2[i] /= scale;
        t[i] /= scale;
    }
    
    // 计算r3 = r1 x r2
    double r3[3];
    cross_product(r1, r2, r3);
    
    // 构建旋转矩阵
    double R[9] = {
        r1[0], r2[0], r3[0],
        r1[1], r2[1], r3[1],
        r1[2], r2[2], r3[2]
    };
    
    // 确保R是正交矩阵（通过SVD）
    double U[9], S[3], Vt[9];
    err = svd_decomposition_3x3(R, U, S, Vt);
    
    if (err != DESIGN_CALIB_SUCCESS) {
        return err;
    }
    
    // R = U * Vt
    matrix_multiply_3x3(U, Vt, R);
    
    // 转换为旋转向量
    rotation_matrix_to_rodrigues(R, rotation);
    
    // 设置平移
    translation[0] = t[0];
    translation[1] = t[1];
    translation[2] = t[2];
    
    return DESIGN_CALIB_SUCCESS;
}

/**
 * @brief 使用LM算法精化位姿
 */
static DesignCalibError refine_pose_lm(
    const DesignCalibDetectedPattern *pattern,
    const DesignCalibIntrinsics *intrinsics,
    double rotation[3],
    double translation[3])
{
    int n = pattern->num_points;
    int num_params = 6;  // 3 for rotation, 3 for translation
    int num_residuals = 2 * n;
    
    // LM参数
    double lambda = 0.01;
    double lambda_factor = 10.0;
    int max_iterations = 100;
    double epsilon = 1e-6;
    
    double *params = DESIGN_CALIB_MALLOC(double, num_params);
    double *residuals = DESIGN_CALIB_MALLOC(double, num_residuals);
    double *jacobian = DESIGN_CALIB_MALLOC(double, num_residuals * num_params);
    double *JtJ = DESIGN_CALIB_MALLOC(double, num_params * num_params);
    double *Jtr = DESIGN_CALIB_MALLOC(double, num_params);
    double *delta = DESIGN_CALIB_MALLOC(double, num_params);
    
    if (!params || !residuals || !jacobian || !JtJ || !Jtr || !delta) {
        DESIGN_CALIB_FREE(params);
        DESIGN_CALIB_FREE(residuals);
        DESIGN_CALIB_FREE(jacobian);
        DESIGN_CALIB_FREE(JtJ);
        DESIGN_CALIB_FREE(Jtr);
        DESIGN_CALIB_FREE(delta);
        return DESIGN_CALIB_ERROR_MEMORY_ALLOCATION;
    }
    
    // 初始化参数
    memcpy(params, rotation, 3 * sizeof(double));
    memcpy(params + 3, translation, 3 * sizeof(double));
    
    // 计算初始误差
    compute_pose_residuals(
        pattern, intrinsics,
        params, params + 3,
        residuals);
    
    double prev_error = compute_residual_norm(residuals, num_residuals);
    
    for (int iter = 0; iter < max_iterations; iter++) {
        // 计算Jacobian
        compute_pose_jacobian(
            pattern, intrinsics,
            params, params + 3,
            jacobian);
        
        // 计算 JtJ 和 Jtr
        compute_JtJ_and_Jtr(
            jacobian, residuals,
            num_residuals, num_params,
            JtJ, Jtr);
        
        // 添加阻尼: JtJ + lambda * diag(JtJ)
        for (int i = 0; i < num_params; i++) {
            JtJ[i * num_params + i] *= (1.0 + lambda);
        }
        
        // 求解 (JtJ + lambda*I) * delta = -Jtr
        if (!solve_linear_system(JtJ, Jtr, delta, num_params)) {
            lambda *= lambda_factor;
            continue;
        }
        
        // 更新参数
        double new_params[6];
        for (int i = 0; i < num_params; i++) {
            new_params[i] = params[i] - delta[i];
        }
        
        // 计算新的误差
        double new_residuals[2 * n];
        compute_pose_residuals(
            pattern, intrinsics,
            new_params, new_params + 3,
            new_residuals);
        
        double new_error = compute_residual_norm(new_residuals, num_residuals);
        
        if (new_error < prev_error) {
            // 接受更新
            memcpy(params, new_params, num_params * sizeof(double));
            memcpy(residuals, new_residuals, num_residuals * sizeof(double));
            
            lambda /= lambda_factor;
            
            if (fabs(prev_error - new_error) < epsilon) {
                break;
            }
            
            prev_error = new_error;
        } else {
            // 拒绝更新
            lambda *= lambda_factor;
        }
    }
    
    // 更新结果
    memcpy(rotation, params, 3 * sizeof(double));
    memcpy(translation, params + 3, 3 * sizeof(double));
    
    DESIGN_CALIB_FREE(params);
    DESIGN_CALIB_FREE(residuals);
    DESIGN_CALIB_FREE(jacobian);
    DESIGN_CALIB_FREE(JtJ);
    DESIGN_CALIB_FREE(Jtr);
    DESIGN_CALIB_FREE(delta);
    
    return DESIGN_CALIB_SUCCESS;
}

/**
 * @brief 计算位姿残差
 */
static void compute_pose_residuals(
    const DesignCalibDetectedPattern *pattern,
    const DesignCalibIntrinsics *intrinsics,
    const double rotation[3],
    const double translation[3],
    double *residuals)
{
    for (int i = 0; i < pattern->num_points; i++) {
        DesignCalibPoint2D projected;
        project_point_3d_to_2d(
            &pattern->object_points[i],
            rotation,
            translation,
            intrinsics,
            &projected);
        
        residuals[2 * i + 0] = projected.x - pattern->image_points[i].x;
        residuals[2 * i + 1] = projected.y - pattern->image_points[i].y;
    }
}

/**
 * @brief 计算位姿Jacobian
 */
static void compute_pose_jacobian(
    const DesignCalibDetectedPattern *pattern,
    const DesignCalibIntrinsics *intrinsics,
    const double rotation[3],
    const double translation[3],
    double *jacobian)
{
    double delta = 1e-6;
    int n = pattern->num_points;
    
    double residuals[2 * n];
    compute_pose_residuals(pattern, intrinsics, rotation, translation, residuals);
    
    // 对每个参数计算数值导数
    for (int p = 0; p < 6; p++) {
        double params_plus[6];
        memcpy(params_plus, rotation, 3 * sizeof(double));
        memcpy(params_plus + 3, translation, 3 * sizeof(double));
        
        params_plus[p] += delta;
        
        double residuals_plus[2 * n];
        compute_pose_residuals(
            pattern, intrinsics,
            params_plus, params_plus + 3,
            residuals_plus);
        
        for (int i = 0; i < 2 * n; i++) {
            jacobian[i * 6 + p] = (residuals_plus[i] - residuals[i]) / delta;
        }
    }
}
// ============================================================================
// 非线性优化 (Bundle Adjustment)
// ============================================================================

/**
 * @brief 优化相机参数
 */
static DesignCalibError optimize_camera_parameters(
    const OptimizationProblem *problem,
    OptimizationParams *params)
{
    // 计算参数数量
    int num_intrinsic_params = 13;  // fx, fy, cx, cy, skew, k1-k6, p1, p2
    int num_extrinsic_params = 6 * problem->num_patterns;  // 每个图像6个参数
    int num_params = num_intrinsic_params + num_extrinsic_params;
    
    // 计算残差数量
    int num_residuals = 0;
    for (int i = 0; i < problem->num_patterns; i++) {
        num_residuals += 2 * problem->patterns[i]->num_points;
    }
    
    // 分配内存
    double *x = DESIGN_CALIB_MALLOC(double, num_params);
    double *residuals = DESIGN_CALIB_MALLOC(double, num_residuals);
    double *jacobian = DESIGN_CALIB_MALLOC(double, num_residuals * num_params);
    
    if (!x || !residuals || !jacobian) {
        DESIGN_CALIB_FREE(x);
        DESIGN_CALIB_FREE(residuals);
        DESIGN_CALIB_FREE(jacobian);
        return DESIGN_CALIB_ERROR_MEMORY_ALLOCATION;
    }
    
    // 初始化参数向量
    pack_optimization_params(params, x);
    
    // Levenberg-Marquardt参数
    double lambda = 0.001;
    double lambda_factor = 10.0;
    int max_iterations = 200;
    double epsilon = 1e-8;
    
    // 计算初始误差
    compute_calibration_residuals(problem, params, residuals);
    double prev_error = compute_residual_norm(residuals, num_residuals);
    
    printf("Initial reprojection error: %.6f\n", prev_error / sqrt(num_residuals));
    
    for (int iter = 0; iter < max_iterations; iter++) {
        // 计算Jacobian
        compute_calibration_jacobian(problem, params, jacobian);
        
        // 构建正规方程
        double *JtJ = DESIGN_CALIB_CALLOC(double, num_params * num_params);
        double *Jtr = DESIGN_CALIB_CALLOC(double, num_params);
        double *delta = DESIGN_CALIB_MALLOC(double, num_params);
        
        if (!JtJ || !Jtr || !delta) {
            DESIGN_CALIB_FREE(x);
            DESIGN_CALIB_FREE(residuals);
            DESIGN_CALIB_FREE(jacobian);
            DESIGN_CALIB_FREE(JtJ);
            DESIGN_CALIB_FREE(Jtr);
            DESIGN_CALIB_FREE(delta);
            return DESIGN_CALIB_ERROR_MEMORY_ALLOCATION;
        }
        
        compute_JtJ_and_Jtr(jacobian, residuals, num_residuals, num_params, JtJ, Jtr);
        
        // 添加阻尼
        for (int i = 0; i < num_params; i++) {
            JtJ[i * num_params + i] *= (1.0 + lambda);
        }
        
        // 求解线性系统
        bool solved = solve_linear_system_cholesky(JtJ, Jtr, delta, num_params);
        
        if (!solved) {
            DESIGN_CALIB_FREE(JtJ);
            DESIGN_CALIB_FREE(Jtr);
            DESIGN_CALIB_FREE(delta);
            lambda *= lambda_factor;
            continue;
        }
        
        // 更新参数
        OptimizationParams new_params = *params;
        double *new_x = DESIGN_CALIB_MALLOC(double, num_params);
        
        if (!new_x) {
            DESIGN_CALIB_FREE(x);
            DESIGN_CALIB_FREE(residuals);
            DESIGN_CALIB_FREE(jacobian);
            DESIGN_CALIB_FREE(JtJ);
            DESIGN_CALIB_FREE(Jtr);
            DESIGN_CALIB_FREE(delta);
            return DESIGN_CALIB_ERROR_MEMORY_ALLOCATION;
        }
        
        for (int i = 0; i < num_params; i++) {
            new_x[i] = x[i] - delta[i];
        }
        
        unpack_optimization_params(new_x, &new_params);
        
        // 应用约束
        apply_parameter_constraints(problem, &new_params);
        
        // 计算新的误差
        double *new_residuals = DESIGN_CALIB_MALLOC(double, num_residuals);
        if (!new_residuals) {
            DESIGN_CALIB_FREE(x);
            DESIGN_CALIB_FREE(residuals);
            DESIGN_CALIB_FREE(jacobian);
            DESIGN_CALIB_FREE(JtJ);
            DESIGN_CALIB_FREE(Jtr);
            DESIGN_CALIB_FREE(delta);
            DESIGN_CALIB_FREE(new_x);
            return DESIGN_CALIB_ERROR_MEMORY_ALLOCATION;
        }
        
        compute_calibration_residuals(problem, &new_params, new_residuals);
        double new_error = compute_residual_norm(new_residuals, num_residuals);
        
        if (new_error < prev_error) {
            // 接受更新
            *params = new_params;
            memcpy(x, new_x, num_params * sizeof(double));
            memcpy(residuals, new_residuals, num_residuals * sizeof(double));
            
            lambda /= lambda_factor;
            
            double improvement = (prev_error - new_error) / prev_error;
            
            if (iter % 10 == 0) {
                printf("Iteration %d: error = %.6f, lambda = %.6e\n",
                       iter, new_error / sqrt(num_residuals), lambda);
            }
            
            if (improvement < epsilon) {
                DESIGN_CALIB_FREE(JtJ);
                DESIGN_CALIB_FREE(Jtr);
                DESIGN_CALIB_FREE(delta);
                DESIGN_CALIB_FREE(new_x);
                DESIGN_CALIB_FREE(new_residuals);
                break;
            }
            
            prev_error = new_error;
        } else {
            // 拒绝更新
            lambda *= lambda_factor;
        }
        
        DESIGN_CALIB_FREE(JtJ);
        DESIGN_CALIB_FREE(Jtr);
        DESIGN_CALIB_FREE(delta);
        DESIGN_CALIB_FREE(new_x);
        DESIGN_CALIB_FREE(new_residuals);
    }
    
    printf("Final reprojection error: %.6f\n", prev_error / sqrt(num_residuals));
    
    DESIGN_CALIB_FREE(x);
    DESIGN_CALIB_FREE(residuals);
    DESIGN_CALIB_FREE(jacobian);
    
    return DESIGN_CALIB_SUCCESS;
}

/**
 * @brief 打包优化参数
 */
static void pack_optimization_params(
    const OptimizationParams *params,
    double *x)
{
    int idx = 0;
    
    // 内参
    x[idx++] = params->fx;
    x[idx++] = params->fy;
    x[idx++] = params->cx;
    x[idx++] = params->cy;
    x[idx++] = params->skew;
    x[idx++] = params->k1;
    x[idx++] = params->k2;
    x[idx++] = params->k3;
    x[idx++] = params->p1;
    x[idx++] = params->p2;
    x[idx++] = params->k4;
    x[idx++] = params->k5;
    x[idx++] = params->k6;
    
    // 外参
    for (int i = 0; i < params->num_images; i++) {
        x[idx++] = params->rotations[i * 3 + 0];
        x[idx++] = params->rotations[i * 3 + 1];
        x[idx++] = params->rotations[i * 3 + 2];
        x[idx++] = params->translations[i * 3 + 0];
        x[idx++] = params->translations[i * 3 + 1];
        x[idx++] = params->translations[i * 3 + 2];
    }
}

/**
 * @brief 解包优化参数
 */
static void unpack_optimization_params(
    const double *x,
    OptimizationParams *params)
{
    int idx = 0;
    
    // 内参
    params->fx = x[idx++];
    params->fy = x[idx++];
    params->cx = x[idx++];
    params->cy = x[idx++];
    params->skew = x[idx++];
    params->k1 = x[idx++];
    params->k2 = x[idx++];
    params->k3 = x[idx++];
    params->p1 = x[idx++];
    params->p2 = x[idx++];
    params->k4 = x[idx++];
    params->k5 = x[idx++];
    params->k6 = x[idx++];
    
    // 外参
    for (int i = 0; i < params->num_images; i++) {
        params->rotations[i * 3 + 0] = x[idx++];
        params->rotations[i * 3 + 1] = x[idx++];
        params->rotations[i * 3 + 2] = x[idx++];
        params->translations[i * 3 + 0] = x[idx++];
        params->translations[i * 3 + 1] = x[idx++];
        params->translations[i * 3 + 2] = x[idx++];
    }
}

/**
 * @brief 应用参数约束
 */
static void apply_parameter_constraints(
    const OptimizationProblem *problem,
    OptimizationParams *params)
{
    // 固定主点
    if (problem->fix_principal_point) {
        params->cx = problem->image_width / 2.0;
        params->cy = problem->image_height / 2.0;
    }
    
    // 固定纵横比
    if (problem->fix_aspect_ratio) {
        double aspect_ratio = params->fx / params->fy;
        params->fy = params->fx / aspect_ratio;
    }
    
    // 零切向畸变
    if (problem->zero_tangent_dist) {
        params->p1 = 0.0;
        params->p2 = 0.0;
    }
    
    // 根据畸变模型限制参数
    switch (problem->distortion_model) {
        case DESIGN_CALIB_DISTORTION_NONE:
            params->k1 = params->k2 = params->k3 = 0.0;
            params->p1 = params->p2 = 0.0;
            params->k4 = params->k5 = params->k6 = 0.0;
            break;
            
        case DESIGN_CALIB_DISTORTION_RADIAL_2:
            params->k3 = 0.0;
            params->p1 = params->p2 = 0.0;
            params->k4 = params->k5 = params->k6 = 0.0;
            break;
            
        case DESIGN_CALIB_DISTORTION_RADIAL_3:
            params->p1 = params->p2 = 0.0;
            params->k4 = params->k5 = params->k6 = 0.0;
            break;
            
        case DESIGN_CALIB_DISTORTION_RADIAL_TANGENTIAL:
            params->k4 = params->k5 = params->k6 = 0.0;
            break;
            
        case DESIGN_CALIB_DISTORTION_RATIONAL:
            // 使用所有参数
            break;
    }
    
    // 确保焦距为正
    if (params->fx < 1.0) params->fx = 1.0;
    if (params->fy < 1.0) params->fy = 1.0;
}

/**
 * @brief 计算校准残差
 */
static void compute_calibration_residuals(
    const OptimizationProblem *problem,
    const OptimizationParams *params,
    double *residuals)
{
    DesignCalibIntrinsics intrinsics;
    intrinsics.fx = params->fx;
    intrinsics.fy = params->fy;
    intrinsics.cx = params->cx;
    intrinsics.cy = params->cy;
    intrinsics.skew = params->skew;
    intrinsics.k1 = params->k1;
    intrinsics.k2 = params->k2;
    intrinsics.k3 = params->k3;
    intrinsics.p1 = params->p1;
    intrinsics.p2 = params->p2;
    intrinsics.k4 = params->k4;
    intrinsics.k5 = params->k5;
    intrinsics.k6 = params->k6;
    
    int residual_idx = 0;
    
    for (int img = 0; img < problem->num_patterns; img++) {
        const DesignCalibDetectedPattern *pattern = problem->patterns[img];
        const double *rotation = &params->rotations[img * 3];
        const double *translation = &params->translations[img * 3];
        
        for (int pt = 0; pt < pattern->num_points; pt++) {
            DesignCalibPoint2D projected;
            project_point_3d_to_2d(
                &pattern->object_points[pt],
                rotation,
                translation,
                &intrinsics,
                &projected);
            
            residuals[residual_idx++] = projected.x - pattern->image_points[pt].x;
            residuals[residual_idx++] = projected.y - pattern->image_points[pt].y;
        }
    }
}

/**
 * @brief 计算校准Jacobian
 */
static void compute_calibration_jacobian(
    const OptimizationProblem *problem,
    const OptimizationParams *params,
    double *jacobian)
{
    double delta = 1e-7;
    
    int num_intrinsic_params = 13;
    int num_extrinsic_params = 6 * problem->num_patterns;
    int num_params = num_intrinsic_params + num_extrinsic_params;
    
    // 计算残差数量
    int num_residuals = 0;
    for (int i = 0; i < problem->num_patterns; i++) {
        num_residuals += 2 * problem->patterns[i]->num_points;
    }
    
    // 计算基准残差
    double *residuals = DESIGN_CALIB_MALLOC(double, num_residuals);
    compute_calibration_residuals(problem, params, residuals);
    
    // 对每个参数计算数值导数
    double *x = DESIGN_CALIB_MALLOC(double, num_params);
    pack_optimization_params(params, x);
    
    for (int p = 0; p < num_params; p++) {
        // 扰动参数
        double x_plus[num_params];
        memcpy(x_plus, x, num_params * sizeof(double));
        x_plus[p] += delta;
        
        // 解包参数
        OptimizationParams params_plus = *params;
        unpack_optimization_params(x_plus, &params_plus);
        
        // 计算扰动后的残差
        double *residuals_plus = DESIGN_CALIB_MALLOC(double, num_residuals);
        compute_calibration_residuals(problem, &params_plus, residuals_plus);
        
        // 计算导数
        for (int r = 0; r < num_residuals; r++) {
            jacobian[r * num_params + p] = (residuals_plus[r] - residuals[r]) / delta;
        }
        
        DESIGN_CALIB_FREE(residuals_plus);
    }
    
    DESIGN_CALIB_FREE(residuals);
    DESIGN_CALIB_FREE(x);
}

/**
 * @brief 3D点投影到2D
 */
static void project_point_3d_to_2d(
    const DesignCalibPoint3D *object_point,
    const double rotation[3],
    const double translation[3],
    const DesignCalibIntrinsics *intrinsics,
    DesignCalibPoint2D *image_point)
{
    // 1. 旋转向量转旋转矩阵
    double R[9];
    rodrigues_to_rotation_matrix(rotation, R);
    
    // 2. 变换到相机坐标系
    double X = object_point->x;
    double Y = object_point->y;
    double Z = object_point->z;
    
    double Xc = R[0]*X + R[1]*Y + R[2]*Z + translation[0];
    double Yc = R[3]*X + R[4]*Y + R[5]*Z + translation[1];
    double Zc = R[6]*X + R[7]*Y + R[8]*Z + translation[2];
    
    // 3. 投影到归一化平面
    if (fabs(Zc) < 1e-10) {
        image_point->x = 0.0;
        image_point->y = 0.0;
        return;
    }
    
    double xn = Xc / Zc;
    double yn = Yc / Zc;
    
    // 4. 应用畸变
    double r2 = xn*xn + yn*yn;
    double r4 = r2 * r2;
    double r6 = r4 * r2;
    
    // 径向畸变
    double radial = 1.0 + intrinsics->k1*r2 + intrinsics->k2*r4 + intrinsics->k3*r6;
    
    // 有理模型
    if (intrinsics->k4 != 0.0 || intrinsics->k5 != 0.0 || intrinsics->k6 != 0.0) {
        double radial_denom = 1.0 + intrinsics->k4*r2 + intrinsics->k5*r4 + intrinsics->k6*r6;
        if (fabs(radial_denom) > 1e-10) {
            radial /= radial_denom;
        }
    }
    
    // 切向畸变
    double dx = 2.0*intrinsics->p1*xn*yn + intrinsics->p2*(r2 + 2.0*xn*xn);
    double dy = intrinsics->p1*(r2 + 2.0*yn*yn) + 2.0*intrinsics->p2*xn*yn;
    
    double xd = xn * radial + dx;
    double yd = yn * radial + dy;
    
    // 5. 应用内参
    image_point->x = intrinsics->fx * xd + intrinsics->skew * yd + intrinsics->cx;
    image_point->y = intrinsics->fy * yd + intrinsics->cy;
}

// ============================================================================
// 线性代数工具函数
// ============================================================================

/**
 * @brief 计算 JtJ 和 Jtr
 */
static void compute_JtJ_and_Jtr(
    const double *jacobian,
    const double *residuals,
    int num_residuals,
    int num_params,
    double *JtJ,
    double *Jtr)
{
    // JtJ = Jt * J
    memset(JtJ, 0, num_params * num_params * sizeof(double));
    
    for (int i = 0; i < num_params; i++) {
        for (int j = 0; j < num_params; j++) {
            double sum = 0.0;
            for (int k = 0; k < num_residuals; k++) {
                sum += jacobian[k * num_params + i] * jacobian[k * num_params + j];
            }
            JtJ[i * num_params + j] = sum;
        }
    }
    
    // Jtr = Jt * r
    memset(Jtr, 0, num_params * sizeof(double));
    
    for (int i = 0; i < num_params; i++) {
        double sum = 0.0;
        for (int k = 0; k < num_residuals; k++) {
            sum += jacobian[k * num_params + i] * residuals[k];
        }
        Jtr[i] = sum;
    }
}

/**
 * @brief 使用Cholesky分解求解线性系统
 */
static bool solve_linear_system_cholesky(
    const double *A,
    const double *b,
    double *x,
    int n)
{
    // 复制A（因为Cholesky会修改它）
    double *L = DESIGN_CALIB_MALLOC(double, n * n);
    if (!L) return false;
    
    memcpy(L, A, n * n * sizeof(double));
    
    // Cholesky分解: A = L * Lt
    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= i; j++) {
            double sum = L[i * n + j];
            
            for (int k = 0; k < j; k++) {
                sum -= L[i * n + k] * L[j * n + k];
            }
            
            if (i == j) {
                if (sum <= 0.0) {
                    DESIGN_CALIB_FREE(L);
                    return false;
                }
                L[i * n + j] = sqrt(sum);
            } else {
                if (fabs(L[j * n + j]) < 1e-10) {
                    DESIGN_CALIB_FREE(L);
                    return false;
                }
                L[i * n + j] = sum / L[j * n + j];
            }
        }
    }
    
    // 前向替换: L * y = b
    double *y = DESIGN_CALIB_MALLOC(double, n);
    if (!y) {
        DESIGN_CALIB_FREE(L);
        return false;
    }
    
    for (int i = 0; i < n; i++) {
        double sum = b[i];
        for (int j = 0; j < i; j++) {
            sum -= L[i * n + j] * y[j];
        }
        y[i] = sum / L[i * n + i];
    }
    
    // 后向替换: Lt * x = y
    for (int i = n - 1; i >= 0; i--) {
        double sum = y[i];
        for (int j = i + 1; j < n; j++) {
            sum -= L[j * n + i] * x[j];
        }
        x[i] = sum / L[i * n + i];
    }
    
    DESIGN_CALIB_FREE(L);
    DESIGN_CALIB_FREE(y);
    
    return true;
}

/**
 * @brief 计算残差范数
 */
static double compute_residual_norm(const double *residuals, int n)
{
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += residuals[i] * residuals[i];
    }
    return sqrt(sum);
}

/**
 * @brief 求解线性系统 (简单高斯消元)
 */
static bool solve_linear_system(
    const double *A,
    const double *b,
    double *x,
    int n)
{
    // 复制矩阵
    double *Aug = DESIGN_CALIB_MALLOC(double, n * (n + 1));
    if (!Aug) return false;
    
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            Aug[i * (n + 1) + j] = A[i * n + j];
        }
        Aug[i * (n + 1) + n] = b[i];
    }
    
    // 高斯消元
    for (int i = 0; i < n; i++) {
        // 找主元
        int max_row = i;
        double max_val = fabs(Aug[i * (n + 1) + i]);
        
        for (int k = i + 1; k < n; k++) {
            double val = fabs(Aug[k * (n + 1) + i]);
            if (val > max_val) {
                max_val = val;
                max_row = k;
            }
        }
        
        if (max_val < 1e-10) {
            DESIGN_CALIB_FREE(Aug);
            return false;
        }
        
        // 交换行
        if (max_row != i) {
            for (int j = 0; j <= n; j++) {
                double temp = Aug[i * (n + 1) + j];
                Aug[i * (n + 1) + j] = Aug[max_row * (n + 1) + j];
                Aug[max_row * (n + 1) + j] = temp;
            }
        }
        
        // 消元
        for (int k = i + 1; k < n; k++) {
            double factor = Aug[k * (n + 1) + i] / Aug[i * (n + 1) + i];
            for (int j = i; j <= n; j++) {
                Aug[k * (n + 1) + j] -= factor * Aug[i * (n + 1) + j];
            }
        }
    }
    
    // 回代
    for (int i = n - 1; i >= 0; i--) {
        double sum = Aug[i * (n + 1) + n];
        for (int j = i + 1; j < n; j++) {
            sum -= Aug[i * (n + 1) + j] * x[j];
        }
        x[i] = sum / Aug[i * (n + 1) + i];
    }
    
    DESIGN_CALIB_FREE(Aug);
    return true;
}
// ============================================================================
// 旋转变换
// ============================================================================

/**
 * @brief Rodrigues旋转向量转旋转矩阵
 */
static void rodrigues_to_rotation_matrix(const double rvec[3], double R[9])
{
    double theta = sqrt(rvec[0]*rvec[0] + rvec[1]*rvec[1] + rvec[2]*rvec[2]);
    
    if (theta < 1e-10) {
        // 接近零，使用一阶近似
        R[0] = 1.0;  R[1] = -rvec[2];  R[2] = rvec[1];
        R[3] = rvec[2];  R[4] = 1.0;  R[5] = -rvec[0];
        R[6] = -rvec[1];  R[7] = rvec[0];  R[8] = 1.0;
        return;
    }
    
    // 归一化旋转轴
    double k[3] = {
        rvec[0] / theta,
        rvec[1] / theta,
        rvec[2] / theta
    };
    
    double c = cos(theta);
    double s = sin(theta);
    double c1 = 1.0 - c;
    
    // Rodrigues公式: R = I + sin(θ)K + (1-cos(θ))K²
    // 其中K是反对称矩阵
    R[0] = c + k[0]*k[0]*c1;
    R[1] = k[0]*k[1]*c1 - k[2]*s;
    R[2] = k[0]*k[2]*c1 + k[1]*s;
    
    R[3] = k[1]*k[0]*c1 + k[2]*s;
    R[4] = c + k[1]*k[1]*c1;
    R[5] = k[1]*k[2]*c1 - k[0]*s;
    
    R[6] = k[2]*k[0]*c1 - k[1]*s;
    R[7] = k[2]*k[1]*c1 + k[0]*s;
    R[8] = c + k[2]*k[2]*c1;
}

/**
 * @brief 旋转矩阵转Rodrigues旋转向量
 */
static void rotation_matrix_to_rodrigues(const double R[9], double rvec[3])
{
    double trace = R[0] + R[4] + R[8];
    
    if (trace >= 3.0 - 1e-6) {
        // 接近单位矩阵
        rvec[0] = (R[7] - R[5]) * 0.5;
        rvec[1] = (R[2] - R[6]) * 0.5;
        rvec[2] = (R[3] - R[1]) * 0.5;
        return;
    }
    
    if (trace <= -1.0 + 1e-6) {
        // 旋转180度
        int i = 0;
        if (R[4] > R[0]) i = 1;
        if (R[8] > R[i*3+i]) i = 2;
        
        int j = (i + 1) % 3;
        int k = (i + 2) % 3;
        
        double s = sqrt(R[i*3+i] - R[j*3+j] - R[k*3+k] + 1.0);
        double v[3] = {0, 0, 0};
        v[i] = s * 0.5;
        
        if (s > 1e-10) {
            v[j] = (R[i*3+j] + R[j*3+i]) / (2.0 * s);
            v[k] = (R[i*3+k] + R[k*3+i]) / (2.0 * s);
        }
        
        double theta = M_PI;
        rvec[0] = v[0] * theta;
        rvec[1] = v[1] * theta;
        rvec[2] = v[2] * theta;
        return;
    }
    
    // 一般情况
    double theta = acos((trace - 1.0) * 0.5);
    double k[3] = {
        (R[7] - R[5]) / (2.0 * sin(theta)),
        (R[2] - R[6]) / (2.0 * sin(theta)),
        (R[3] - R[1]) / (2.0 * sin(theta))
    };
    
    rvec[0] = k[0] * theta;
    rvec[1] = k[1] * theta;
    rvec[2] = k[2] * theta;
}

/**
 * @brief 四元数转旋转矩阵
 */
static void quaternion_to_rotation_matrix(const double q[4], double R[9])
{
    double w = q[0], x = q[1], y = q[2], z = q[3];
    
    double xx = x * x;
    double yy = y * y;
    double zz = z * z;
    double xy = x * y;
    double xz = x * z;
    double yz = y * z;
    double wx = w * x;
    double wy = w * y;
    double wz = w * z;
    
    R[0] = 1.0 - 2.0 * (yy + zz);
    R[1] = 2.0 * (xy - wz);
    R[2] = 2.0 * (xz + wy);
    
    R[3] = 2.0 * (xy + wz);
    R[4] = 1.0 - 2.0 * (xx + zz);
    R[5] = 2.0 * (yz - wx);
    
    R[6] = 2.0 * (xz - wy);
    R[7] = 2.0 * (yz + wx);
    R[8] = 1.0 - 2.0 * (xx + yy);
}

/**
 * @brief 旋转矩阵转四元数
 */
static void rotation_matrix_to_quaternion(const double R[9], double q[4])
{
    double trace = R[0] + R[4] + R[8];
    
    if (trace > 0.0) {
        double s = sqrt(trace + 1.0) * 2.0;
        q[0] = 0.25 * s;
        q[1] = (R[7] - R[5]) / s;
        q[2] = (R[2] - R[6]) / s;
        q[3] = (R[3] - R[1]) / s;
    } else if (R[0] > R[4] && R[0] > R[8]) {
        double s = sqrt(1.0 + R[0] - R[4] - R[8]) * 2.0;
        q[0] = (R[7] - R[5]) / s;
        q[1] = 0.25 * s;
        q[2] = (R[1] + R[3]) / s;
        q[3] = (R[2] + R[6]) / s;
    } else if (R[4] > R[8]) {
        double s = sqrt(1.0 + R[4] - R[0] - R[8]) * 2.0;
        q[0] = (R[2] - R[6]) / s;
        q[1] = (R[1] + R[3]) / s;
        q[2] = 0.25 * s;
        q[3] = (R[5] + R[7]) / s;
    } else {
        double s = sqrt(1.0 + R[8] - R[0] - R[4]) * 2.0;
        q[0] = (R[3] - R[1]) / s;
        q[1] = (R[2] + R[6]) / s;
        q[2] = (R[5] + R[7]) / s;
        q[3] = 0.25 * s;
    }
    
    // 归一化
    double norm = sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3]);
    if (norm > 1e-10) {
        q[0] /= norm;
        q[1] /= norm;
        q[2] /= norm;
        q[3] /= norm;
    }
}

/**
 * @brief 欧拉角转旋转矩阵 (ZYX顺序)
 */
static void euler_to_rotation_matrix(
    double roll,
    double pitch,
    double yaw,
    double R[9])
{
    double cr = cos(roll);
    double sr = sin(roll);
    double cp = cos(pitch);
    double sp = sin(pitch);
    double cy = cos(yaw);
    double sy = sin(yaw);
    
    R[0] = cy * cp;
    R[1] = cy * sp * sr - sy * cr;
    R[2] = cy * sp * cr + sy * sr;
    
    R[3] = sy * cp;
    R[4] = sy * sp * sr + cy * cr;
    R[5] = sy * sp * cr - cy * sr;
    
    R[6] = -sp;
    R[7] = cp * sr;
    R[8] = cp * cr;
}

/**
 * @brief 旋转矩阵转欧拉角 (ZYX顺序)
 */
static void rotation_matrix_to_euler(
    const double R[9],
    double *roll,
    double *pitch,
    double *yaw)
{
    *pitch = asin(-R[6]);
    
    if (fabs(cos(*pitch)) > 1e-6) {
        *roll = atan2(R[7], R[8]);
        *yaw = atan2(R[3], R[0]);
    } else {
        // 万向锁情况
        *roll = 0.0;
        *yaw = atan2(-R[1], R[4]);
    }
}

// ============================================================================
// 矩阵运算
// ============================================================================

/**
 * @brief 3x3矩阵乘法
 */
static void matrix_multiply_3x3(const double A[9], const double B[9], double C[9])
{
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            C[i*3 + j] = 0.0;
            for (int k = 0; k < 3; k++) {
                C[i*3 + j] += A[i*3 + k] * B[k*3 + j];
            }
        }
    }
}

/**
 * @brief 3x3矩阵转置
 */
static void matrix_transpose_3x3(const double A[9], double At[9])
{
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            At[j*3 + i] = A[i*3 + j];
        }
    }
}

/**
 * @brief 3x3矩阵求逆
 */
static bool invert_3x3_matrix(const double A[9], double A_inv[9])
{
    // 计算行列式
    double det = A[0] * (A[4]*A[8] - A[5]*A[7])
               - A[1] * (A[3]*A[8] - A[5]*A[6])
               + A[2] * (A[3]*A[7] - A[4]*A[6]);
    
    if (fabs(det) < 1e-10) {
        return false;
    }
    
    double inv_det = 1.0 / det;
    
    // 计算伴随矩阵
    A_inv[0] = (A[4]*A[8] - A[5]*A[7]) * inv_det;
    A_inv[1] = (A[2]*A[7] - A[1]*A[8]) * inv_det;
    A_inv[2] = (A[1]*A[5] - A[2]*A[4]) * inv_det;
    
    A_inv[3] = (A[5]*A[6] - A[3]*A[8]) * inv_det;
    A_inv[4] = (A[0]*A[8] - A[2]*A[6]) * inv_det;
    A_inv[5] = (A[2]*A[3] - A[0]*A[5]) * inv_det;
    
    A_inv[6] = (A[3]*A[7] - A[4]*A[6]) * inv_det;
    A_inv[7] = (A[1]*A[6] - A[0]*A[7]) * inv_det;
    A_inv[8] = (A[0]*A[4] - A[1]*A[3]) * inv_det;
    
    return true;
}

/**
 * @brief 3x3矩阵行列式
 */
static double matrix_determinant_3x3(const double A[9])
{
    return A[0] * (A[4]*A[8] - A[5]*A[7])
         - A[1] * (A[3]*A[8] - A[5]*A[6])
         + A[2] * (A[3]*A[7] - A[4]*A[6]);
}

/**
 * @brief 向量叉乘
 */
static void cross_product(const double a[3], const double b[3], double c[3])
{
    c[0] = a[1]*b[2] - a[2]*b[1];
    c[1] = a[2]*b[0] - a[0]*b[2];
    c[2] = a[0]*b[1] - a[1]*b[0];
}

/**
 * @brief 向量点乘
 */
static double dot_product(const double a[3], const double b[3])
{
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}

/**
 * @brief 向量归一化
 */
static void normalize_vector(double v[3])
{
    double norm = sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
    if (norm > 1e-10) {
        v[0] /= norm;
        v[1] /= norm;
        v[2] /= norm;
    }
}

/**
 * @brief 向量长度
 */
static double vector_norm(const double v[3])
{
    return sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
}

// ============================================================================
// SVD分解
// ============================================================================

/**
 * @brief 3x3矩阵SVD分解
 */
static DesignCalibError svd_decomposition_3x3(
    const double A[9],
    double U[9],
    double S[3],
    double Vt[9])
{
    // 使用Jacobi方法
    // 这里使用简化实现，实际应用中建议使用LAPACK等库
    
    // 计算 A^T * A
    double AtA[9];
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            AtA[i*3 + j] = 0.0;
            for (int k = 0; k < 3; k++) {
                AtA[i*3 + j] += A[k*3 + i] * A[k*3 + j];
            }
        }
    }
    
    // 特征值分解 AtA = V * S^2 * V^T
    double V[9];
    double eigenvalues[3];
    
    if (!eigen_decomposition_3x3(AtA, V, eigenvalues)) {
        return DESIGN_CALIB_ERROR_SVD_FAILED;
    }
    
    // 奇异值是特征值的平方根
    for (int i = 0; i < 3; i++) {
        S[i] = sqrt(fmax(eigenvalues[i], 0.0));
    }
    
    // V^T
    matrix_transpose_3x3(V, Vt);
    
    // U = A * V * S^-1
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            U[i*3 + j] = 0.0;
            for (int k = 0; k < 3; k++) {
                if (S[k] > 1e-10) {
                    U[i*3 + j] += A[i*3 + k] * V[k*3 + j] / S[k];
                }
            }
        }
    }
    
    return DESIGN_CALIB_SUCCESS;
}

/**
 * @brief 一般矩阵SVD分解 (简化版本)
 */
static DesignCalibError svd_decomposition(
    const double *A,
    int m,
    int n,
    double *U,
    double *S,
    double *Vt)
{
    // 这里应该使用专业的SVD库，如LAPACK
    // 为了演示，这里只实现了基本框架
    
    // 对于小矩阵，可以使用Jacobi方法
    // 对于大矩阵，应该使用更高效的算法
    
    if (m < n) {
        return DESIGN_CALIB_ERROR_INVALID_PARAM;
    }
    
    // 简化实现：只处理 m >= n 的情况
    // 计算 A^T * A
    double *AtA = DESIGN_CALIB_CALLOC(double, n * n);
    if (!AtA) {
        return DESIGN_CALIB_ERROR_MEMORY_ALLOCATION;
    }
    
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int k = 0; k < m; k++) {
                sum += A[k*n + i] * A[k*n + j];
            }
            AtA[i*n + j] = sum;
        }
    }
    
    // 特征值分解
    double *V = DESIGN_CALIB_MALLOC(double, n * n);
    double *eigenvalues = DESIGN_CALIB_MALLOC(double, n);
    
    if (!V || !eigenvalues) {
        DESIGN_CALIB_FREE(AtA);
        DESIGN_CALIB_FREE(V);
        DESIGN_CALIB_FREE(eigenvalues);
        return DESIGN_CALIB_ERROR_MEMORY_ALLOCATION;
    }
    
    // 使用幂迭代法求特征值和特征向量
    for (int i = 0; i < n; i++) {
        // 初始化特征向量
        for (int j = 0; j < n; j++) {
            V[j*n + i] = (i == j) ? 1.0 : 0.0;
        }
        
        // 幂迭代
        for (int iter = 0; iter < 100; iter++) {
            double *v = &V[i];
            double *Av = DESIGN_CALIB_MALLOC(double, n);
            
            if (!Av) {
                DESIGN_CALIB_FREE(AtA);
                DESIGN_CALIB_FREE(V);
                DESIGN_CALIB_FREE(eigenvalues);
                return DESIGN_CALIB_ERROR_MEMORY_ALLOCATION;
            }
            
            // Av = AtA * v
            for (int j = 0; j < n; j++) {
                Av[j] = 0.0;
                for (int k = 0; k < n; k++) {
                    Av[j] += AtA[j*n + k] * V[k*n + i];
                }
            }
            
            // 正交化
            for (int j = 0; j < i; j++) {
                double dot = 0.0;
                for (int k = 0; k < n; k++) {
                    dot += Av[k] * V[k*n + j];
                }
                for (int k = 0; k < n; k++) {
                    Av[k] -= dot * V[k*n + j];
                }
            }
            
            // 归一化
            double norm = 0.0;
            for (int j = 0; j < n; j++) {
                norm += Av[j] * Av[j];
            }
            norm = sqrt(norm);
            
            if (norm < 1e-10) {
                DESIGN_CALIB_FREE(Av);
                break;
            }
            
            for (int j = 0; j < n; j++) {
                V[j*n + i] = Av[j] / norm;
            }
            
            eigenvalues[i] = norm;
            DESIGN_CALIB_FREE(Av);
        }
    }
    
    // 奇异值
    for (int i = 0; i < n; i++) {
        S[i] = sqrt(fmax(eigenvalues[i], 0.0));
    }
    
    // Vt
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            Vt[i*n + j] = V[j*n + i];
        }
    }
    
    // U = A * V * S^-1
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            U[i*n + j] = 0.0;
            for (int k = 0; k < n; k++) {
                if (S[k] > 1e-10) {
                    U[i*n + j] += A[i*n + k] * V[k*n + j] / S[k];
                }
            }
        }
    }
    
    DESIGN_CALIB_FREE(AtA);
    DESIGN_CALIB_FREE(V);
    DESIGN_CALIB_FREE(eigenvalues);
    
    return DESIGN_CALIB_SUCCESS;
}

/**
 * @brief 3x3对称矩阵特征值分解
 */
static bool eigen_decomposition_3x3(
    const double A[9],
    double V[9],
    double eigenvalues[3])
{
    // 使用Jacobi方法
    double B[9];
    memcpy(B, A, 9 * sizeof(double));
    
    // 初始化V为单位矩阵
    memset(V, 0, 9 * sizeof(double));
    V[0] = V[4] = V[8] = 1.0;
    
    int max_iterations = 50;
    double epsilon = 1e-10;
    
    for (int iter = 0; iter < max_iterations; iter++) {
        // 找最大非对角元素
        int p = 0, q = 1;
        double max_val = fabs(B[0*3 + 1]);
        
        for (int i = 0; i < 3; i++) {
            for (int j = i + 1; j < 3; j++) {
                double val = fabs(B[i*3 + j]);
                if (val > max_val) {
                    max_val = val;
                    p = i;
                    q = j;
                }
            }
        }
        
        if (max_val < epsilon) {
            break;
        }
        
        // 计算旋转角度
        double theta;
        if (fabs(B[p*3 + p] - B[q*3 + q]) < epsilon) {
            theta = M_PI / 4.0;
        } else {
            theta = 0.5 * atan2(2.0 * B[p*3 + q], B[p*3 + p] - B[q*3 + q]);
        }
        
        double c = cos(theta);
        double s = sin(theta);
        
        // 应用Givens旋转
        double Bp[3], Bq[3];
        for (int i = 0; i < 3; i++) {
            Bp[i] = B[p*3 + i];
            Bq[i] = B[q*3 + i];
        }
        
        for (int i = 0; i < 3; i++) {
            B[p*3 + i] = c * Bp[i] - s * Bq[i];
            B[q*3 + i] = s * Bp[i] + c * Bq[i];
        }
        
        for (int i = 0; i < 3; i++) {
            Bp[i] = B[i*3 + p];
            Bq[i] = B[i*3 + q];
            B[i*3 + p] = c * Bp[i] - s * Bq[i];
            B[i*3 + q] = s * Bp[i] + c * Bq[i];
        }
        
        // 更新特征向量
        for (int i = 0; i < 3; i++) {
            double Vp = V[i*3 + p];
            double Vq = V[i*3 + q];
            V[i*3 + p] = c * Vp - s * Vq;
            V[i*3 + q] = s * Vp + c * Vq;
        }
    }
    
    // 提取特征值
    eigenvalues[0] = B[0];
    eigenvalues[1] = B[4];
    eigenvalues[2] = B[8];
    
    // 排序（降序）
    for (int i = 0; i < 2; i++) {
        for (int j = i + 1; j < 3; j++) {
            if (eigenvalues[j] > eigenvalues[i]) {
                // 交换特征值
                double temp = eigenvalues[i];
                eigenvalues[i] = eigenvalues[j];
                eigenvalues[j] = temp;
                
                // 交换特征向量
                for (int k = 0; k < 3; k++) {
                    temp = V[k*3 + i];
                    V[k*3 + i] = V[k*3 + j];
                    V[k*3 + j] = temp;
                }
            }
        }
    }
    
    return true;
}
/**
 * @brief 获取错误信息字符串
 */
DESIGN_CALIB_API const char* design_calib_get_error_string(DesignCalibError error)
{
    switch (error) {
        case DESIGN_CALIB_SUCCESS:
            return "Success";
        case DESIGN_CALIB_ERROR_NULL_POINTER:
            return "Null pointer error";
        case DESIGN_CALIB_ERROR_INVALID_PARAM:
            return "Invalid parameter";
        case DESIGN_CALIB_ERROR_MEMORY_ALLOCATION:
            return "Memory allocation failed";
        case DESIGN_CALIB_ERROR_PATTERN_NOT_FOUND:
            return "Calibration pattern not found";
        case DESIGN_CALIB_ERROR_INSUFFICIENT_IMAGES:
            return "Insufficient calibration images";
        case DESIGN_CALIB_ERROR_CALIBRATION_FAILED:
            return "Calibration failed";
        case DESIGN_CALIB_ERROR_OPTIMIZATION_FAILED:
            return "Optimization failed";
        case DESIGN_CALIB_ERROR_SVD_FAILED:
            return "SVD decomposition failed";
        case DESIGN_CALIB_ERROR_FILE_IO:
            return "File I/O error";
        case DESIGN_CALIB_ERROR_INVALID_IMAGE:
            return "Invalid image data";
        case DESIGN_CALIB_ERROR_UNSUPPORTED_FORMAT:
            return "Unsupported format";
        default:
            return "Unknown error";
    }
}

/**
 * @brief 获取库版本信息
 */
DESIGN_CALIB_API void design_calib_get_version(
    int *major,
    int *minor,
    int *patch)
{
    if (major) *major = DESIGN_CALIB_VERSION_MAJOR;
    if (minor) *minor = DESIGN_CALIB_VERSION_MINOR;
    if (patch) *patch = DESIGN_CALIB_VERSION_PATCH;
}

/**
 * @brief 获取版本字符串
 */
DESIGN_CALIB_API const char* design_calib_get_version_string(void)
{
    return DESIGN_CALIB_VERSION_STRING;
}

/**
 * @brief 打印库信息
 */
DESIGN_CALIB_API void design_calib_print_info(void)
{
    printf("Design Calibration Library\n");
    printf("Version: %s\n", DESIGN_CALIB_VERSION_STRING);
    printf("Build date: %s %s\n", __DATE__, __TIME__);
    printf("\n");
    printf("Features:\n");
    printf("  - Chessboard detection\n");
    printf("  - Circle grid detection\n");
    printf("  - ArUco marker detection\n");
    printf("  - Camera calibration\n");
    printf("  - Stereo calibration\n");
    printf("  - Image undistortion\n");
    printf("  - Multiple distortion models\n");
}

// ============================================================================
// 调试和可视化辅助函数
// ============================================================================

/**
 * @brief 打印内参矩阵
 */
DESIGN_CALIB_API void design_calib_print_intrinsics(
    const DesignCalibIntrinsics *intrinsics)
{
    if (!intrinsics) return;
    
    printf("Camera Intrinsics:\n");
    printf("  Focal length: fx=%.4f, fy=%.4f\n", intrinsics->fx, intrinsics->fy);
    printf("  Principal point: cx=%.4f, cy=%.4f\n", intrinsics->cx, intrinsics->cy);
    printf("  Skew: %.6f\n", intrinsics->skew);
    printf("\n");
    
    printf("Distortion Coefficients:\n");
    printf("  Radial: k1=%.6f, k2=%.6f, k3=%.6f\n", 
           intrinsics->k1, intrinsics->k2, intrinsics->k3);
    printf("  Tangential: p1=%.6f, p2=%.6f\n", 
           intrinsics->p1, intrinsics->p2);
    printf("  Rational: k4=%.6f, k5=%.6f, k6=%.6f\n", 
           intrinsics->k4, intrinsics->k5, intrinsics->k6);
    printf("\n");
    
    printf("Camera Matrix K:\n");
    printf("  [%.4f  %.4f  %.4f]\n", intrinsics->fx, intrinsics->skew, intrinsics->cx);
    printf("  [%.4f  %.4f  %.4f]\n", 0.0, intrinsics->fy, intrinsics->cy);
    printf("  [%.4f  %.4f  %.4f]\n", 0.0, 0.0, 1.0);
}

/**
 * @brief 打印外参
 */
DESIGN_CALIB_API void design_calib_print_extrinsics(
    const double rotation[3],
    const double translation[3])
{
    if (!rotation || !translation) return;
    
    printf("Camera Extrinsics:\n");
    printf("  Rotation vector: [%.6f, %.6f, %.6f]\n", 
           rotation[0], rotation[1], rotation[2]);
    printf("  Translation: [%.6f, %.6f, %.6f]\n", 
           translation[0], translation[1], translation[2]);
    
    // 转换为旋转矩阵
    double R[9];
    rodrigues_to_rotation_matrix(rotation, R);
    
    printf("  Rotation matrix:\n");
    printf("    [%.6f  %.6f  %.6f]\n", R[0], R[1], R[2]);
    printf("    [%.6f  %.6f  %.6f]\n", R[3], R[4], R[5]);
    printf("    [%.6f  %.6f  %.6f]\n", R[6], R[7], R[8]);
    
    // 转换为欧拉角
    double roll, pitch, yaw;
    rotation_matrix_to_euler(R, &roll, &pitch, &yaw);
    
    printf("  Euler angles (deg): roll=%.2f, pitch=%.2f, yaw=%.2f\n",
           roll * 180.0 / M_PI,
           pitch * 180.0 / M_PI,
           yaw * 180.0 / M_PI);
}

/**
 * @brief 打印质量指标
 */
DESIGN_CALIB_API void design_calib_print_quality_metrics(
    const DesignCalibQualityMetrics *metrics)
{
    if (!metrics) return;
    
    printf("Calibration Quality Metrics:\n");
    printf("  Mean reprojection error: %.4f pixels\n", 
           metrics->mean_reprojection_error);
    printf("  Max reprojection error: %.4f pixels\n", 
           metrics->max_reprojection_error);
    printf("  Std reprojection error: %.4f pixels\n", 
           metrics->std_reprojection_error);
    printf("  Coverage score: %.2f%%\n", 
           metrics->coverage_score * 100.0);
    printf("  Parameter confidence: %.2f%%\n", 
           metrics->parameter_confidence * 100.0);
    
    // 评估校准质量
    printf("\n");
    if (metrics->mean_reprojection_error < 0.5) {
        printf("  Quality: EXCELLENT\n");
    } else if (metrics->mean_reprojection_error < 1.0) {
        printf("  Quality: GOOD\n");
    } else if (metrics->mean_reprojection_error < 2.0) {
        printf("  Quality: ACCEPTABLE\n");
    } else {
        printf("  Quality: POOR - Consider recalibration\n");
    }
}

/**
 * @brief 绘制检测到的角点（简单ASCII可视化）
 */
DESIGN_CALIB_API void design_calib_visualize_pattern(
    const DesignCalibDetectedPattern *pattern,
    int console_width,
    int console_height)
{
    if (!pattern) return;
    
    // 创建ASCII画布
    char *canvas = DESIGN_CALIB_CALLOC(char, console_width * console_height);
    if (!canvas) return;
    
    // 初始化为空格
    memset(canvas, ' ', console_width * console_height);
    
    // 计算缩放比例
    double scale_x = (double)console_width / pattern->image_width;
    double scale_y = (double)console_height / pattern->image_height;
    
    // 绘制角点
    for (int i = 0; i < pattern->num_points; i++) {
        int x = (int)(pattern->image_points[i].x * scale_x);
        int y = (int)(pattern->image_points[i].y * scale_y);
        
        if (x >= 0 && x < console_width && y >= 0 && y < console_height) {
            canvas[y * console_width + x] = '*';
        }
    }
    
    // 打印画布
    printf("Pattern visualization (%dx%d):\n", pattern->image_width, pattern->image_height);
    for (int y = 0; y < console_height; y++) {
        for (int x = 0; x < console_width; x++) {
            putchar(canvas[y * console_width + x]);
        }
        putchar('\n');
    }
    
    DESIGN_CALIB_FREE(canvas);
}

// ============================================================================
// 高级功能
// ============================================================================

/**
 * @brief 计算视场角
 */
DESIGN_CALIB_API void design_calib_compute_fov(
    const DesignCalibIntrinsics *intrinsics,
    int image_width,
    int image_height,
    double *fov_x,
    double *fov_y,
    double *fov_diagonal)
{
    if (!intrinsics) return;
    
    // 水平视场角
    if (fov_x) {
        *fov_x = 2.0 * atan(image_width / (2.0 * intrinsics->fx)) * 180.0 / M_PI;
    }
    
    // 垂直视场角
    if (fov_y) {
        *fov_y = 2.0 * atan(image_height / (2.0 * intrinsics->fy)) * 180.0 / M_PI;
    }
    
    // 对角线视场角
    if (fov_diagonal) {
        double diagonal = sqrt(image_width * image_width + image_height * image_height);
        double f_avg = (intrinsics->fx + intrinsics->fy) / 2.0;
        *fov_diagonal = 2.0 * atan(diagonal / (2.0 * f_avg)) * 180.0 / M_PI;
    }
}

/**
 * @brief 计算像素尺寸
 */
DESIGN_CALIB_API void design_calib_compute_pixel_size(
    const DesignCalibIntrinsics *intrinsics,
    double sensor_width_mm,
    double sensor_height_mm,
    int image_width,
    int image_height,
    double *pixel_width_mm,
    double *pixel_height_mm)
{
    if (!intrinsics) return;
    
    if (pixel_width_mm) {
        *pixel_width_mm = sensor_width_mm / image_width;
    }
    
    if (pixel_height_mm) {
        *pixel_height_mm = sensor_height_mm / image_height;
    }
}

/**
 * @brief 估计相机到标定板的距离
 */
DESIGN_CALIB_API double design_calib_estimate_distance(
    const DesignCalibDetectedPattern *pattern,
    const DesignCalibIntrinsics *intrinsics,
    const double rotation[3],
    const double translation[3])
{
    if (!pattern || !intrinsics || !translation) {
        return -1.0;
    }
    
    // 距离就是平移向量的模
    return sqrt(translation[0]*translation[0] + 
                translation[1]*translation[1] + 
                translation[2]*translation[2]);
}

/**
 * @brief 检查标定板姿态是否合适
 */
DESIGN_CALIB_API bool design_calib_check_pattern_pose(
    const double rotation[3],
    const double translation[3],
    double min_angle_deg,
    double max_angle_deg,
    double min_distance,
    double max_distance)
{
    if (!rotation || !translation) return false;
    
    // 检查距离
    double distance = sqrt(translation[0]*translation[0] + 
                          translation[1]*translation[1] + 
                          translation[2]*translation[2]);
    
    if (distance < min_distance || distance > max_distance) {
        return false;
    }
    
    // 检查角度
    double R[9];
    rodrigues_to_rotation_matrix(rotation, R);
    
    // 计算标定板法向量与相机光轴的夹角
    double normal[3] = {R[6], R[7], R[8]};  // 第三列
    double camera_axis[3] = {0, 0, 1};
    
    double dot = fabs(dot_product(normal, camera_axis));
    double angle = acos(dot) * 180.0 / M_PI;
    
    if (angle < min_angle_deg || angle > max_angle_deg) {
        return false;
    }
    
    return true;
}

/**
 * @brief 生成校准报告
 */
DESIGN_CALIB_API DesignCalibError design_calib_generate_report(
    const char *filename,
    const DesignCalibDetectedPattern **patterns,
    int num_patterns,
    const DesignCalibIntrinsics *intrinsics,
    const double *rotations,
    const double *translations,
    const DesignCalibQualityMetrics *metrics)
{
    if (!filename || !patterns || !intrinsics || !metrics) {
        return DESIGN_CALIB_ERROR_NULL_POINTER;
    }
    
    FILE *fp = fopen(filename, "w");
    if (!fp) {
        return DESIGN_CALIB_ERROR_FILE_IO;
    }
    
    fprintf(fp, "Camera Calibration Report\n");
    fprintf(fp, "=========================\n\n");
    
    // 基本信息
    fprintf(fp, "Date: %s\n", __DATE__);
    fprintf(fp, "Library Version: %s\n\n", DESIGN_CALIB_VERSION_STRING);
    
    // 图像信息
    fprintf(fp, "Calibration Images: %d\n", num_patterns);
    fprintf(fp, "Image Size: %dx%d\n\n", 
            patterns[0]->image_width, patterns[0]->image_height);
    
    // 内参
    fprintf(fp, "Intrinsic Parameters:\n");
    fprintf(fp, "  fx = %.6f\n", intrinsics->fx);
    fprintf(fp, "  fy = %.6f\n", intrinsics->fy);
    fprintf(fp, "  cx = %.6f\n", intrinsics->cx);
    fprintf(fp, "  cy = %.6f\n", intrinsics->cy);
    fprintf(fp, "  skew = %.6f\n\n", intrinsics->skew);
    
    // 畸变系数
    fprintf(fp, "Distortion Coefficients:\n");
    fprintf(fp, "  k1 = %.6f\n", intrinsics->k1);
    fprintf(fp, "  k2 = %.6f\n", intrinsics->k2);
    fprintf(fp, "  k3 = %.6f\n", intrinsics->k3);
    fprintf(fp, "  p1 = %.6f\n", intrinsics->p1);
    fprintf(fp, "  p2 = %.6f\n", intrinsics->p2);
    fprintf(fp, "  k4 = %.6f\n", intrinsics->k4);
    fprintf(fp, "  k5 = %.6f\n", intrinsics->k5);
    fprintf(fp, "  k6 = %.6f\n\n", intrinsics->k6);
    
    // 质量指标
    fprintf(fp, "Quality Metrics:\n");
    fprintf(fp, "  Mean Reprojection Error: %.4f pixels\n", 
            metrics->mean_reprojection_error);
    fprintf(fp, "  Max Reprojection Error: %.4f pixels\n", 
            metrics->max_reprojection_error);
    fprintf(fp, "  Std Reprojection Error: %.4f pixels\n", 
            metrics->std_reprojection_error);
    fprintf(fp, "  Coverage Score: %.2f%%\n", 
            metrics->coverage_score * 100.0);
    fprintf(fp, "  Parameter Confidence: %.2f%%\n\n", 
            metrics->parameter_confidence * 100.0);
    
    // 视场角
    double fov_x, fov_y, fov_diag;
    design_calib_compute_fov(intrinsics, patterns[0]->image_width, 
                            patterns[0]->image_height,
                            &fov_x, &fov_y, &fov_diag);
    
    fprintf(fp, "Field of View:\n");
    fprintf(fp, "  Horizontal: %.2f degrees\n", fov_x);
    fprintf(fp, "  Vertical: %.2f degrees\n", fov_y);
    fprintf(fp, "  Diagonal: %.2f degrees\n\n", fov_diag);
    
    // 每个图像的详细信息
    if (rotations && translations) {
        fprintf(fp, "Per-Image Statistics:\n");
        fprintf(fp, "Image | Points | Distance | Angle | Error\n");
        fprintf(fp, "------|--------|----------|-------|-------\n");
        
        for (int i = 0; i < num_patterns; i++) {
            double distance = design_calib_estimate_distance(
                patterns[i], intrinsics, 
                &rotations[i*3], &translations[i*3]);
            
            double R[9];
            rodrigues_to_rotation_matrix(&rotations[i*3], R);
            double normal[3] = {R[6], R[7], R[8]};
            double camera_axis[3] = {0, 0, 1};
            double angle = acos(fabs(dot_product(normal, camera_axis))) * 180.0 / M_PI;
            
            // 计算该图像的重投影误差
            double error_sum = 0.0;
            for (int j = 0; j < patterns[i]->num_points; j++) {
                DesignCalibPoint2D projected;
                project_point_3d_to_2d(
                    &patterns[i]->object_points[j],
                    &rotations[i*3],
                    &translations[i*3],
                    intrinsics,
                    &projected);
                
                double dx = projected.x - patterns[i]->image_points[j].x;
                double dy = projected.y - patterns[i]->image_points[j].y;
                error_sum += sqrt(dx*dx + dy*dy);
            }
            double avg_error = error_sum / patterns[i]->num_points;
            
            fprintf(fp, "%5d | %6d | %8.2f | %5.1f | %.4f\n",
                    i+1, patterns[i]->num_points, distance, angle, avg_error);
        }
    }
    
    fclose(fp);
    
    return DESIGN_CALIB_SUCCESS;
}
// ============================================================================
// 自动校准辅助功能
// ============================================================================

/**
 * @brief 自动选择最佳校准图像
 */
DESIGN_CALIB_API DesignCalibError design_calib_select_best_images(
    const DesignCalibDetectedPattern **patterns,
    int num_patterns,
    int max_images,
    int *selected_indices,
    int *num_selected)
{
    if (!patterns || !selected_indices || !num_selected) {
        return DESIGN_CALIB_ERROR_NULL_POINTER;
    }
    
    if (num_patterns <= max_images) {
        // 所有图像都选择
        for (int i = 0; i < num_patterns; i++) {
            selected_indices[i] = i;
        }
        *num_selected = num_patterns;
        return DESIGN_CALIB_SUCCESS;
    }
    
    // 评分系统：基于覆盖范围、角度多样性、点数量
    double *scores = DESIGN_CALIB_MALLOC(double, num_patterns);
    if (!scores) {
        return DESIGN_CALIB_ERROR_MEMORY_ALLOCATION;
    }
    
    // 计算每个图像的得分
    for (int i = 0; i < num_patterns; i++) {
        double coverage_score = 0.0;
        double point_score = (double)patterns[i]->num_points / 100.0;
        
        // 计算图像中点的分布
        double min_x = INFINITY, max_x = -INFINITY;
        double min_y = INFINITY, max_y = -INFINITY;
        
        for (int j = 0; j < patterns[i]->num_points; j++) {
            double x = patterns[i]->image_points[j].x;
            double y = patterns[i]->image_points[j].y;
            
            if (x < min_x) min_x = x;
            if (x > max_x) max_x = x;
            if (y < min_y) min_y = y;
            if (y > max_y) max_y = y;
        }
        
        coverage_score = ((max_x - min_x) * (max_y - min_y)) / 
                        (patterns[i]->image_width * patterns[i]->image_height);
        
        scores[i] = coverage_score * 0.6 + point_score * 0.4;
    }
    
    // 选择得分最高的图像
    *num_selected = 0;
    for (int i = 0; i < max_images; i++) {
        int best_idx = -1;
        double best_score = -INFINITY;
        
        for (int j = 0; j < num_patterns; j++) {
            // 检查是否已被选择
            bool already_selected = false;
            for (int k = 0; k < *num_selected; k++) {
                if (selected_indices[k] == j) {
                    already_selected = true;
                    break;
                }
            }
            
            if (!already_selected && scores[j] > best_score) {
                best_score = scores[j];
                best_idx = j;
            }
        }
        
        if (best_idx >= 0) {
            selected_indices[*num_selected] = best_idx;
            (*num_selected)++;
        }
    }
    
    DESIGN_CALIB_FREE(scores);
    
    return DESIGN_CALIB_SUCCESS;
}

/**
 * @brief 检测校准图像质量
 */
DESIGN_CALIB_API DesignCalibError design_calib_check_image_quality(
    const DesignCalibDetectedPattern *pattern,
    double *quality_score,
    char *quality_message,
    int message_size)
{
    if (!pattern || !quality_score) {
        return DESIGN_CALIB_ERROR_NULL_POINTER;
    }
    
    *quality_score = 0.0;
    
    // 1. 检查点数量
    double point_score = fmin(1.0, (double)pattern->num_points / 50.0);
    
    // 2. 检查覆盖范围
    double min_x = INFINITY, max_x = -INFINITY;
    double min_y = INFINITY, max_y = -INFINITY;
    
    for (int i = 0; i < pattern->num_points; i++) {
        double x = pattern->image_points[i].x;
        double y = pattern->image_points[i].y;
        
        if (x < min_x) min_x = x;
        if (x > max_x) max_x = x;
        if (y < min_y) min_y = y;
        if (y > max_y) max_y = y;
    }
    
    double coverage = ((max_x - min_x) * (max_y - min_y)) / 
                     (pattern->image_width * pattern->image_height);
    double coverage_score = fmin(1.0, coverage / 0.3);
    
    // 3. 检查点的分布均匀性
    double uniformity_score = 1.0;  // 简化实现
    
    // 综合得分
    *quality_score = point_score * 0.3 + coverage_score * 0.5 + uniformity_score * 0.2;
    
    // 生成质量消息
    if (quality_message && message_size > 0) {
        if (*quality_score >= 0.8) {
            snprintf(quality_message, message_size, "Excellent quality");
        } else if (*quality_score >= 0.6) {
            snprintf(quality_message, message_size, "Good quality");
        } else if (*quality_score >= 0.4) {
            snprintf(quality_message, message_size, "Acceptable quality");
        } else {
            snprintf(quality_message, message_size, 
                    "Poor quality - consider retaking image");
        }
    }
    
    return DESIGN_CALIB_SUCCESS;
}

/**
 * @brief 推荐下一个校准图像的位置
 */
DESIGN_CALIB_API DesignCalibError design_calib_suggest_next_pose(
    const DesignCalibDetectedPattern **patterns,
    int num_patterns,
    char *suggestion,
    int suggestion_size)
{
    if (!patterns || !suggestion || suggestion_size <= 0) {
        return DESIGN_CALIB_ERROR_NULL_POINTER;
    }
    
    if (num_patterns == 0) {
        snprintf(suggestion, suggestion_size, 
                "Place calibration pattern in center of image");
        return DESIGN_CALIB_SUCCESS;
    }
    
    // 分析已有图像的覆盖情况
    bool has_center = false;
    bool has_corners[4] = {false, false, false, false};  // TL, TR, BL, BR
    bool has_edges[4] = {false, false, false, false};    // T, R, B, L
    bool has_tilted = false;
    
    for (int i = 0; i < num_patterns; i++) {
        // 计算中心位置
        double center_x = 0.0, center_y = 0.0;
        for (int j = 0; j < patterns[i]->num_points; j++) {
            center_x += patterns[i]->image_points[j].x;
            center_y += patterns[i]->image_points[j].y;
        }
        center_x /= patterns[i]->num_points;
        center_y /= patterns[i]->num_points;
        
        double img_center_x = patterns[i]->image_width / 2.0;
        double img_center_y = patterns[i]->image_height / 2.0;
        
        // 检查位置
        double dx = center_x - img_center_x;
        double dy = center_y - img_center_y;
        double dist = sqrt(dx*dx + dy*dy);
        
        if (dist < patterns[i]->image_width * 0.2) {
            has_center = true;
        }
        
        // 检查角落
        if (center_x < img_center_x && center_y < img_center_y) {
            has_corners[0] = true;  // Top-left
        }
        if (center_x > img_center_x && center_y < img_center_y) {
            has_corners[1] = true;  // Top-right
        }
        if (center_x < img_center_x && center_y > img_center_y) {
            has_corners[2] = true;  // Bottom-left
        }
        if (center_x > img_center_x && center_y > img_center_y) {
            has_corners[3] = true;  // Bottom-right
        }
    }
    
    // 生成建议
    if (!has_center) {
        snprintf(suggestion, suggestion_size, 
                "Place pattern in center of image");
    } else if (!has_corners[0]) {
        snprintf(suggestion, suggestion_size, 
                "Place pattern in top-left corner");
    } else if (!has_corners[1]) {
        snprintf(suggestion, suggestion_size, 
                "Place pattern in top-right corner");
    } else if (!has_corners[2]) {
        snprintf(suggestion, suggestion_size, 
                "Place pattern in bottom-left corner");
    } else if (!has_corners[3]) {
        snprintf(suggestion, suggestion_size, 
                "Place pattern in bottom-right corner");
    } else if (!has_tilted) {
        snprintf(suggestion, suggestion_size, 
                "Tilt pattern at various angles");
    } else {
        snprintf(suggestion, suggestion_size, 
                "Good coverage - ready to calibrate");
    }
    
    return DESIGN_CALIB_SUCCESS;
}

// ============================================================================
// 在线校准功能
// ============================================================================

/**
 * @brief 创建在线校准上下文
 */
DESIGN_CALIB_API DesignCalibError design_calib_create_online_context(
    DesignCalibOnlineContext **context,
    int image_width,
    int image_height,
    int max_images)
{
    if (!context) {
        return DESIGN_CALIB_ERROR_NULL_POINTER;
    }
    
    DesignCalibOnlineContext *ctx = DESIGN_CALIB_MALLOC(
        DesignCalibOnlineContext, 1);
    if (!ctx) {
        return DESIGN_CALIB_ERROR_MEMORY_ALLOCATION;
    }
    
    ctx->image_width = image_width;
    ctx->image_height = image_height;
    ctx->max_images = max_images;
    ctx->num_images = 0;
    ctx->is_calibrated = false;
    
    ctx->patterns = DESIGN_CALIB_CALLOC(
        DesignCalibDetectedPattern*, max_images);
    if (!ctx->patterns) {
        DESIGN_CALIB_FREE(ctx);
        return DESIGN_CALIB_ERROR_MEMORY_ALLOCATION;
    }
    
    // 初始化内参为默认值
    ctx->intrinsics.fx = image_width;
    ctx->intrinsics.fy = image_width;
    ctx->intrinsics.cx = image_width / 2.0;
    ctx->intrinsics.cy = image_height / 2.0;
    ctx->intrinsics.skew = 0.0;
    ctx->intrinsics.k1 = 0.0;
    ctx->intrinsics.k2 = 0.0;
    ctx->intrinsics.k3 = 0.0;
    ctx->intrinsics.p1 = 0.0;
    ctx->intrinsics.p2 = 0.0;
    ctx->intrinsics.k4 = 0.0;
    ctx->intrinsics.k5 = 0.0;
    ctx->intrinsics.k6 = 0.0;
    
    *context = ctx;
    
    return DESIGN_CALIB_SUCCESS;
}

/**
 * @brief 添加校准图像到在线上下文
 */
DESIGN_CALIB_API DesignCalibError design_calib_add_calibration_image(
    DesignCalibOnlineContext *context,
    const DesignCalibDetectedPattern *pattern)
{
    if (!context || !pattern) {
        return DESIGN_CALIB_ERROR_NULL_POINTER;
    }
    
    if (context->num_images >= context->max_images) {
        return DESIGN_CALIB_ERROR_INVALID_PARAM;
    }
    
    // 复制pattern
    DesignCalibDetectedPattern *new_pattern = DESIGN_CALIB_MALLOC(
        DesignCalibDetectedPattern, 1);
    if (!new_pattern) {
        return DESIGN_CALIB_ERROR_MEMORY_ALLOCATION;
    }
    
    memcpy(new_pattern, pattern, sizeof(DesignCalibDetectedPattern));
    
    // 复制点数据
    new_pattern->image_points = DESIGN_CALIB_MALLOC(
        DesignCalibPoint2D, pattern->num_points);
    new_pattern->object_points = DESIGN_CALIB_MALLOC(
        DesignCalibPoint3D, pattern->num_points);
    
    if (!new_pattern->image_points || !new_pattern->object_points) {
        DESIGN_CALIB_FREE(new_pattern->image_points);
        DESIGN_CALIB_FREE(new_pattern->object_points);
        DESIGN_CALIB_FREE(new_pattern);
        return DESIGN_CALIB_ERROR_MEMORY_ALLOCATION;
    }
    
    memcpy(new_pattern->image_points, pattern->image_points,
           pattern->num_points * sizeof(DesignCalibPoint2D));
    memcpy(new_pattern->object_points, pattern->object_points,
           pattern->num_points * sizeof(DesignCalibPoint3D));
    
    context->patterns[context->num_images] = new_pattern;
    context->num_images++;
    
    // 如果有足够的图像，尝试校准
    if (context->num_images >= 3) {
        DesignCalibError err = design_calib_calibrate_camera(
            (const DesignCalibDetectedPattern**)context->patterns,
            context->num_images,
            context->image_width,
            context->image_height,
            &context->intrinsics,
            DESIGN_CALIB_FLAG_FIX_ASPECT_RATIO);
        
        if (err == DESIGN_CALIB_SUCCESS) {
            context->is_calibrated = true;
        }
    }
    
    return DESIGN_CALIB_SUCCESS;
}

/**
 * @brief 获取在线校准状态
 */
DESIGN_CALIB_API DesignCalibError design_calib_get_online_status(
    const DesignCalibOnlineContext *context,
    int *num_images,
    bool *is_calibrated,
    double *current_error)
{
    if (!context) {
        return DESIGN_CALIB_ERROR_NULL_POINTER;
    }
    
    if (num_images) {
        *num_images = context->num_images;
    }
    
    if (is_calibrated) {
        *is_calibrated = context->is_calibrated;
    }
    
    if (current_error && context->is_calibrated) {
        // 计算当前重投影误差
        *current_error = design_calib_compute_reprojection_error(
            (const DesignCalibDetectedPattern**)context->patterns,
            context->num_images,
            &context->intrinsics,
            NULL,  // 需要存储旋转
            NULL); // 需要存储平移
    }
    
    return DESIGN_CALIB_SUCCESS;
}

/**
 * @brief 销毁在线校准上下文
 */
DESIGN_CALIB_API void design_calib_destroy_online_context(
    DesignCalibOnlineContext *context)
{
    if (!context) return;
    
    if (context->patterns) {
        for (int i = 0; i < context->num_images; i++) {
            if (context->patterns[i]) {
                DESIGN_CALIB_FREE(context->patterns[i]->image_points);
                DESIGN_CALIB_FREE(context->patterns[i]->object_points);
                DESIGN_CALIB_FREE(context->patterns[i]);
            }
        }
        DESIGN_CALIB_FREE(context->patterns);
    }
    
    DESIGN_CALIB_FREE(context);
}

// ============================================================================
// 多相机校准
// ============================================================================

/**
 * @brief 多相机系统校准
 */
DESIGN_CALIB_API DesignCalibError design_calib_multi_camera_calibrate(
    const DesignCalibDetectedPattern ***camera_patterns,
    int num_cameras,
    int num_patterns,
    int image_width,
    int image_height,
    DesignCalibIntrinsics *intrinsics_array,
    double *relative_rotations,
    double *relative_translations,
    DesignCalibFlags flags)
{
    if (!camera_patterns || !intrinsics_array || 
        !relative_rotations || !relative_translations) {
        return DESIGN_CALIB_ERROR_NULL_POINTER;
    }
    
    if (num_cameras < 2 || num_patterns < 3) {
        return DESIGN_CALIB_ERROR_INVALID_PARAM;
    }
    
    // 1. 分别校准每个相机
    for (int i = 0; i < num_cameras; i++) {
        DesignCalibError err = design_calib_calibrate_camera(
            camera_patterns[i],
            num_patterns,
            image_width,
            image_height,
            &intrinsics_array[i],
            flags);
        
        if (err != DESIGN_CALIB_SUCCESS) {
            return err;
        }
    }
    
    // 2. 估计相对位姿（以第一个相机为参考）
    for (int i = 1; i < num_cameras; i++) {
        DesignCalibError err = estimate_stereo_extrinsics(
            camera_patterns[0],
            camera_patterns[i],
            num_patterns,
            &intrinsics_array[0],
            &intrinsics_array[i],
            &relative_rotations[(i-1)*3],
            &relative_translations[(i-1)*3]);
        
        if (err != DESIGN_CALIB_SUCCESS) {
            return err;
        }
    }
    
    // 3. 全局优化（Bundle Adjustment）
    // 这里应该实现完整的多相机BA
    // 简化实现省略
    
    return DESIGN_CALIB_SUCCESS;
}

// ============================================================================
// 实用工具函数
// ============================================================================

/**
 * @brief 转换畸变模型
 */
DESIGN_CALIB_API DesignCalibError design_calib_convert_distortion_model(
    const DesignCalibIntrinsics *src_intrinsics,
    DesignCalibDistortionModel src_model,
    DesignCalibIntrinsics *dst_intrinsics,
    DesignCalibDistortionModel dst_model)
{
    if (!src_intrinsics || !dst_intrinsics) {
        return DESIGN_CALIB_ERROR_NULL_POINTER;
    }
    
    // 复制基本参数
    memcpy(dst_intrinsics, src_intrinsics, sizeof(DesignCalibIntrinsics));
    
    // 根据目标模型调整畸变系数
    switch (dst_model) {
        case DESIGN_CALIB_DISTORTION_NONE:
            dst_intrinsics->k1 = 0.0;
            dst_intrinsics->k2 = 0.0;
            dst_intrinsics->k3 = 0.0;
            dst_intrinsics->p1 = 0.0;
            dst_intrinsics->p2 = 0.0;
            dst_intrinsics->k4 = 0.0;
            dst_intrinsics->k5 = 0.0;
            dst_intrinsics->k6 = 0.0;
            break;
            
        case DESIGN_CALIB_DISTORTION_RADIAL:
            dst_intrinsics->p1 = 0.0;
            dst_intrinsics->p2 = 0.0;
            dst_intrinsics->k4 = 0.0;
            dst_intrinsics->k5 = 0.0;
            dst_intrinsics->k6 = 0.0;
            break;
            
        case DESIGN_CALIB_DISTORTION_RADIAL_TANGENTIAL:
            dst_intrinsics->k4 = 0.0;
            dst_intrinsics->k5 = 0.0;
            dst_intrinsics->k6 = 0.0;
            break;
            
        case DESIGN_CALIB_DISTORTION_RATIONAL:
            // 保留所有系数
            break;
            
        default:
            return DESIGN_CALIB_ERROR_UNSUPPORTED_FORMAT;
    }
    
    return DESIGN_CALIB_SUCCESS;
}

/**
 * @brief 比较两组内参
 */
DESIGN_CALIB_API double design_calib_compare_intrinsics(
    const DesignCalibIntrinsics *intrinsics1,
    const DesignCalibIntrinsics *intrinsics2)
{
    if (!intrinsics1 || !intrinsics2) {
        return -1.0;
    }
    
    // 计算归一化差异
    double diff = 0.0;
    
    diff += fabs(intrinsics1->fx - intrinsics2->fx) / intrinsics1->fx;
    diff += fabs(intrinsics1->fy - intrinsics2->fy) / intrinsics1->fy;
    diff += fabs(intrinsics1->cx - intrinsics2->cx) / intrinsics1->fx;
    diff += fabs(intrinsics1->cy - intrinsics2->cy) / intrinsics1->fy;
    
    diff += fabs(intrinsics1->k1 - intrinsics2->k1);
    diff += fabs(intrinsics1->k2 - intrinsics2->k2);
    diff += fabs(intrinsics1->p1 - intrinsics2->p1);
    diff += fabs(intrinsics1->p2 - intrinsics2->p2);
    
    return diff / 8.0;  // 平均差异
}

/**
 * @brief 验证内参合理性
 */
DESIGN_CALIB_API bool design_calib_validate_intrinsics(
    const DesignCalibIntrinsics *intrinsics,
    int image_width,
    int image_height)
{
    if (!intrinsics) return false;
    
    // 检查焦距
    if (intrinsics->fx <= 0 || intrinsics->fy <= 0) return false;
    if (intrinsics->fx > image_width * 3 || intrinsics->fy > image_height * 3) return false;
    
    // 检查主点
    if (intrinsics->cx < 0 || intrinsics->cx > image_width) return false;
    if (intrinsics->cy < 0 || intrinsics->cy > image_height) return false;
    
    // 检查畸变系数范围
    if (fabs(intrinsics->k1) > 10.0) return false;
    if (fabs(intrinsics->k2) > 10.0) return false;
    if (fabs(intrinsics->p1) > 1.0) return false;
    if (fabs(intrinsics->p2) > 1.0) return false;
    
    return true;
}

// ============================================================================
// 库清理
// ============================================================================

/**
 * @brief 清理库资源
 */
DESIGN_CALIB_API void design_calib_cleanup(void)
{
    // 清理全局资源（如果有）
    // 当前实现中没有全局资源需要清理
}

#ifdef __cplusplus
}
#endif

// ============================================================================
// 文件结束
// ============================================================================

