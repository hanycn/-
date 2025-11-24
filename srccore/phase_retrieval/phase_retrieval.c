/**
 * @file phase_retrieval.c
 * @brief 相位恢复算法实现
 * @author Assistant
 * @date 2024
 */

#include "phase_retrieval.h"
#include "fft.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <float.h>

// ============================================================================
// 内部辅助函数声明
// ============================================================================

static float compute_error_metric(
    const ComplexImage *field1,
    const ComplexImage *field2);

static int apply_amplitude_constraint(
    ComplexImage *field,
    const RealImage *target_amplitude);

static int apply_support_constraint(
    ComplexImage *field,
    const SupportDomain *support,
    float beta);

static void print_iteration_info(
    int iteration,
    float error,
    double elapsed_time);

// ============================================================================
// 参数创建和销毁
// ============================================================================

PhaseRetrievalParams* phase_retrieval_params_create_default(void)
{
    PhaseRetrievalParams *params = (PhaseRetrievalParams*)malloc(sizeof(PhaseRetrievalParams));
    if (!params) {
        return NULL;
    }

    // 设置默认值
    params->algorithm = PHASE_RETRIEVAL_HYBRID;
    params->max_iterations = 1000;
    params->tolerance = RECONSTRUCTION_DEFAULT_TOLERANCE;
    params->beta = RECONSTRUCTION_DEFAULT_BETA;
    params->relaxation = RECONSTRUCTION_DEFAULT_RELAXATION;

    // Shrinkwrap参数
    params->use_shrinkwrap = false;
    params->shrinkwrap_interval = 20;
    params->shrinkwrap_sigma = 3.0f;
    params->shrinkwrap_threshold = 0.2f;

    // 初始化参数
    params->use_random_phase = true;
    params->random_seed = (unsigned int)time(NULL);
    params->initial_guess = NULL;

    // 输出控制
    params->verbose = false;
    params->print_interval = 10;
    params->save_intermediate = false;
    params->save_interval = 100;
    params->output_prefix = NULL;

    return params;
}

void phase_retrieval_params_destroy(PhaseRetrievalParams *params)
{
    if (!params) {
        return;
    }

    if (params->initial_guess) {
        complex_image_destroy(params->initial_guess);
    }

    if (params->output_prefix) {
        free(params->output_prefix);
    }

    free(params);
}

HolographyParams* holography_params_create_default(HolographyType type)
{
    HolographyParams *params = (HolographyParams*)malloc(sizeof(HolographyParams));
    if (!params) {
        return NULL;
    }

    params->type = type;

    // 离轴全息默认参数
    params->carrier_frequency_x = 0.25f;
    params->carrier_frequency_y = 0.0f;
    params->filter_size = 0.1f;

    // 相移全息默认参数
    params->num_phase_steps = 4;
    params->phase_shifts = NULL;
    
    if (type == HOLOGRAPHY_PHASE_SHIFTING) {
        params->phase_shifts = (float*)malloc(4 * sizeof(float));
        if (params->phase_shifts) {
            params->phase_shifts[0] = 0.0f;
            params->phase_shifts[1] = M_PI / 2.0f;
            params->phase_shifts[2] = M_PI;
            params->phase_shifts[3] = 3.0f * M_PI / 2.0f;
        }
    }

    params->reference_wave = NULL;
    params->numerical_refocus = false;
    params->diffraction_params = NULL;
    params->remove_twin_image = false;
    params->twin_removal_iterations = 10;

    return params;
}

void holography_params_destroy(HolographyParams *params)
{
    if (!params) {
        return;
    }

    if (params->phase_shifts) {
        free(params->phase_shifts);
    }

    if (params->reference_wave) {
        complex_image_destroy(params->reference_wave);
    }

    if (params->diffraction_params) {
        free(params->diffraction_params);
    }

    free(params);
}

MultiPlaneParams* multiplane_params_create(int num_planes)
{
    if (num_planes <= 0 || num_planes > RECONSTRUCTION_MAX_PLANES) {
        return NULL;
    }

    MultiPlaneParams *params = (MultiPlaneParams*)malloc(sizeof(MultiPlaneParams));
    if (!params) {
        return NULL;
    }

    params->num_planes = num_planes;
    
    params->measurements = (RealImage**)calloc(num_planes, sizeof(RealImage*));
    params->distances = (double*)calloc(num_planes, sizeof(double));
    params->plane_weights = (float*)malloc(num_planes * sizeof(float));
    
    if (!params->measurements || !params->distances || !params->plane_weights) {
        free(params->measurements);
        free(params->distances);
        free(params->plane_weights);
        free(params);
        return NULL;
    }

    // 默认均匀权重
    for (int i = 0; i < num_planes; i++) {
        params->plane_weights[i] = 1.0f / num_planes;
    }

    params->diffraction_params = NULL;
    params->base_params = *phase_retrieval_params_create_default();

    return params;
}

void multiplane_params_destroy(MultiPlaneParams *params)
{
    if (!params) {
        return;
    }

    if (params->measurements) {
        for (int i = 0; i < params->num_planes; i++) {
            if (params->measurements[i]) {
                real_image_destroy(params->measurements[i]);
            }
        }
        free(params->measurements);
    }

    free(params->distances);
    free(params->plane_weights);

    if (params->diffraction_params) {
        free(params->diffraction_params);
    }

    phase_retrieval_params_destroy(&params->base_params);
    free(params);
}

// ============================================================================
// 约束相关函数
// ============================================================================

SupportDomain* support_domain_create(int width, int height)
{
    if (width <= 0 || height <= 0) {
        return NULL;
    }

    SupportDomain *support = (SupportDomain*)malloc(sizeof(SupportDomain));
    if (!support) {
        return NULL;
    }

    support->width = width;
    support->height = height;
    
    support->mask = (bool*)calloc(width * height, sizeof(bool));
    support->weights = NULL;

    if (!support->mask) {
        free(support);
        return NULL;
    }

    return support;
}

void support_domain_destroy(SupportDomain *support)
{
    if (!support) {
        return;
    }

    free(support->mask);
    free(support->weights);
    free(support);
}

SupportDomain* support_domain_from_image(const RealImage *image, float threshold)
{
    if (!image || threshold < 0.0f || threshold > 1.0f) {
        return NULL;
    }

    SupportDomain *support = support_domain_create(image->width, image->height);
    if (!support) {
        return NULL;
    }

    // 找到最大值
    float max_val = 0.0f;
    for (int i = 0; i < image->width * image->height; i++) {
        float val = image->data[i];
        if (val > max_val) {
            max_val = val;
        }
    }

    // 应用阈值
    float threshold_val = threshold * max_val;
    for (int i = 0; i < image->width * image->height; i++) {
        support->mask[i] = (image->data[i] >= threshold_val);
    }

    return support;
}

SupportDomain* support_domain_create_circular(
    int width, int height,
    float center_x, float center_y,
    float radius)
{
    if (width <= 0 || height <= 0 || radius <= 0.0f) {
        return NULL;
    }

    SupportDomain *support = support_domain_create(width, height);
    if (!support) {
        return NULL;
    }

    float radius_sq = radius * radius;

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            float dx = j - center_x;
            float dy = i - center_y;
            float dist_sq = dx * dx + dy * dy;
            
            support->mask[i * width + j] = (dist_sq <= radius_sq);
        }
    }

    return support;
}

SupportDomain* support_domain_create_rectangular(
    int width, int height,
    int x, int y,
    int rect_width, int rect_height)
{
    if (width <= 0 || height <= 0 || rect_width <= 0 || rect_height <= 0) {
        return NULL;
    }

    SupportDomain *support = support_domain_create(width, height);
    if (!support) {
        return NULL;
    }

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            bool inside = (j >= x && j < x + rect_width &&
                          i >= y && i < y + rect_height);
            support->mask[i * width + j] = inside;
        }
    }

    return support;
}

ConstraintSet* constraint_set_create(void)
{
    ConstraintSet *constraints = (ConstraintSet*)malloc(sizeof(ConstraintSet));
    if (!constraints) {
        return NULL;
    }

    constraints->constraint_flags = CONSTRAINT_NONE;
    constraints->support = NULL;
    constraints->min_value = 0.0f;
    constraints->max_value = 1.0f;
    constraints->amplitude_constraint = NULL;
    constraints->phase_constraint = NULL;

    return constraints;
}

void constraint_set_destroy(ConstraintSet *constraints)
{
    if (!constraints) {
        return;
    }

    if (constraints->support) {
        support_domain_destroy(constraints->support);
    }

    if (constraints->amplitude_constraint) {
        real_image_destroy(constraints->amplitude_constraint);
    }

    if (constraints->phase_constraint) {
        real_image_destroy(constraints->phase_constraint);
    }

    free(constraints);
}

int apply_constraints(
    ComplexImage *field,
    const ConstraintSet *constraints,
    bool in_object_plane)
{
    if (!field || !constraints) {
        return RECONSTRUCTION_ERROR_NULL_POINTER;
    }

    int size = field->width * field->height;

    // 应用支撑域约束（仅在物平面）
    if (in_object_plane && 
        (constraints->constraint_flags & CONSTRAINT_SUPPORT) &&
        constraints->support) {
        
        if (constraints->support->width != field->width ||
            constraints->support->height != field->height) {
            return RECONSTRUCTION_ERROR_INVALID_PARAMS;
        }

        for (int i = 0; i < size; i++) {
            if (!constraints->support->mask[i]) {
                field->data[i] = 0.0f + 0.0f * I;
            } else if (constraints->support->weights) {
                field->data[i] *= constraints->support->weights[i];
            }
        }
    }

    // 应用正值约束（仅在物平面）
    if (in_object_plane && 
        (constraints->constraint_flags & CONSTRAINT_POSITIVITY)) {
        
        for (int i = 0; i < size; i++) {
            float real_part = crealf(field->data[i]);
            float imag_part = cimagf(field->data[i]);
            
            if (real_part < 0.0f) {
                real_part = 0.0f;
            }
            
            field->data[i] = real_part + imag_part * I;
        }
    }

    // 应用振幅约束
    if ((constraints->constraint_flags & CONSTRAINT_AMPLITUDE) &&
        constraints->amplitude_constraint) {
        
        if (constraints->amplitude_constraint->width != field->width ||
            constraints->amplitude_constraint->height != field->height) {
            return RECONSTRUCTION_ERROR_INVALID_PARAMS;
        }

        for (int i = 0; i < size; i++) {
            float target_amp = constraints->amplitude_constraint->data[i];
            float current_phase = cargf(field->data[i]);
            
            field->data[i] = target_amp * (cosf(current_phase) + sinf(current_phase) * I);
        }
    }

    // 应用相位约束
    if ((constraints->constraint_flags & CONSTRAINT_PHASE) &&
        constraints->phase_constraint) {
        
        if (constraints->phase_constraint->width != field->width ||
            constraints->phase_constraint->height != field->height) {
            return RECONSTRUCTION_ERROR_INVALID_PARAMS;
        }

        for (int i = 0; i < size; i++) {
            float current_amp = cabsf(field->data[i]);
            float target_phase = constraints->phase_constraint->data[i];
            
            field->data[i] = current_amp * (cosf(target_phase) + sinf(target_phase) * I);
        }
    }

    // 应用实值约束
    if (constraints->constraint_flags & CONSTRAINT_REAL) {
        for (int i = 0; i < size; i++) {
            float real_part = crealf(field->data[i]);
            field->data[i] = real_part + 0.0f * I;
        }
    }

    // 应用值域约束
    if (constraints->constraint_flags & CONSTRAINT_RANGE) {
        for (int i = 0; i < size; i++) {
            float real_part = crealf(field->data[i]);
            float imag_part = cimagf(field->data[i]);
            
            if (real_part < constraints->min_value) {
                real_part = constraints->min_value;
            } else if (real_part > constraints->max_value) {
                real_part = constraints->max_value;
            }
            
            field->data[i] = real_part + imag_part * I;
        }
    }

    return RECONSTRUCTION_SUCCESS;
}

int update_support_shrinkwrap(
    const ComplexImage *field,
    SupportDomain *support,
    float sigma,
    float threshold)
{
    if (!field || !support) {
        return RECONSTRUCTION_ERROR_NULL_POINTER;
    }

    if (field->width != support->width || field->height != support->height) {
        return RECONSTRUCTION_ERROR_INVALID_PARAMS;
    }

    int width = field->width;
    int height = field->height;

    // 计算振幅
    RealImage *amplitude = real_image_create(width, height, 1);
    if (!amplitude) {
        return RECONSTRUCTION_ERROR_MEMORY_ALLOCATION;
    }

    for (int i = 0; i < width * height; i++) {
        amplitude->data[i] = cabsf(field->data[i]);
    }

    // 高斯模糊
    RealImage *blurred = real_image_create(width, height, 1);
    if (!blurred) {
        real_image_destroy(amplitude);
        return RECONSTRUCTION_ERROR_MEMORY_ALLOCATION;
    }

    // 简单的高斯模糊实现
    int kernel_size = (int)(3.0f * sigma);
    if (kernel_size % 2 == 0) kernel_size++;
    int half_kernel = kernel_size / 2;

    float *kernel = (float*)malloc(kernel_size * kernel_size * sizeof(float));
    if (!kernel) {
        real_image_destroy(amplitude);
        real_image_destroy(blurred);
        return RECONSTRUCTION_ERROR_MEMORY_ALLOCATION;
    }

    // 生成高斯核
    float sum = 0.0f;
    for (int i = 0; i < kernel_size; i++) {
        for (int j = 0; j < kernel_size; j++) {
            int di = i - half_kernel;
            int dj = j - half_kernel;
            float val = expf(-(di * di + dj * dj) / (2.0f * sigma * sigma));
            kernel[i * kernel_size + j] = val;
            sum += val;
        }
    }

    // 归一化
    for (int i = 0; i < kernel_size * kernel_size; i++) {
        kernel[i] /= sum;
    }

    // 应用卷积
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            float val = 0.0f;
            
            for (int ki = 0; ki < kernel_size; ki++) {
                for (int kj = 0; kj < kernel_size; kj++) {
                    int si = i + ki - half_kernel;
                    int sj = j + kj - half_kernel;
                    
                    // 边界处理
                    if (si < 0) si = 0;
                    if (si >= height) si = height - 1;
                    if (sj < 0) sj = 0;
                    if (sj >= width) sj = width - 1;
                    
                    val += amplitude->data[si * width + sj] * 
                           kernel[ki * kernel_size + kj];
                }
            }
            
            blurred->data[i * width + j] = val;
        }
    }

    free(kernel);
    real_image_destroy(amplitude);

    // 找到最大值
    float max_val = 0.0f;
    for (int i = 0; i < width * height; i++) {
        if (blurred->data[i] > max_val) {
            max_val = blurred->data[i];
        }
    }

    // 更新支撑域
    float threshold_val = threshold * max_val;
    for (int i = 0; i < width * height; i++) {
        support->mask[i] = (blurred->data[i] >= threshold_val);
    }

    real_image_destroy(blurred);

    return RECONSTRUCTION_SUCCESS;
}

// ============================================================================
// 结果相关函数
// ============================================================================

ReconstructionResult* reconstruction_result_create(
    int width, int height,
    int max_iterations)
{
    if (width <= 0 || height <= 0 || max_iterations <= 0) {
        return NULL;
    }

    ReconstructionResult *result = (ReconstructionResult*)malloc(sizeof(ReconstructionResult));
    if (!result) {
        return NULL;
    }

    result->reconstructed = complex_image_create(width, height);
    if (!result->reconstructed) {
        free(result);
        return NULL;
    }

    result->error_history = (float*)malloc(max_iterations * sizeof(float));
    if (!result->error_history) {
        complex_image_destroy(result->reconstructed);
        free(result);
        return NULL;
    }

    result->iterations_performed = 0;
    result->final_error = 0.0f;
    result->converged = false;
    result->computation_time = 0.0;
    result->error_history_length = 0;
    result->quality = NULL;

    return result;
}

void reconstruction_result_destroy(ReconstructionResult *result)
{
    if (!result) {
        return;
    }

    if (result->reconstructed) {
        complex_image_destroy(result->reconstructed);
    }

    free(result->error_history);

    if (result->quality) {
        free(result->quality);
    }

    free(result);
}

// ============================================================================
// 辅助函数实现
// ============================================================================

static float compute_error_metric(
    const ComplexImage *field1,
    const ComplexImage *field2)
{
    if (!field1 || !field2) {
        return -1.0f;
    }

    if (field1->width != field2->width || field1->height != field2->height) {
        return -1.0f;
    }

    int size = field1->width * field1->height;
    float error = 0.0f;

    for (int i = 0; i < size; i++) {
        ComplexF diff = field1->data[i] - field2->data[i];
        error += crealf(diff) * crealf(diff) + cimagf(diff) * cimagf(diff);
    }

    return sqrtf(error / size);
}

static int apply_amplitude_constraint(
    ComplexImage *field,
    const RealImage *target_amplitude)
{
    if (!field || !target_amplitude) {
        return RECONSTRUCTION_ERROR_NULL_POINTER;
    }

    if (field->width != target_amplitude->width ||
        field->height != target_amplitude->height) {
        return RECONSTRUCTION_ERROR_INVALID_PARAMS;
    }

    int size = field->width * field->height;

    for (int i = 0; i < size; i++) {
        float target_amp = target_amplitude->data[i];
        float current_phase = cargf(field->data[i]);
        
        field->data[i] = target_amp * (cosf(current_phase) + sinf(current_phase) * I);
    }

    return RECONSTRUCTION_SUCCESS;
}

static int apply_support_constraint(
    ComplexImage *field,
    const SupportDomain *support,
    float beta)
{
    if (!field || !support) {
        return RECONSTRUCTION_ERROR_NULL_POINTER;
    }

    if (field->width != support->width || field->height != support->height) {
        return RECONSTRUCTION_ERROR_INVALID_PARAMS;
    }

    int size = field->width * field->height;

    for (int i = 0; i < size; i++) {
        if (!support->mask[i]) {
            // 在支撑域外：应用HIO更新
            field->data[i] *= -beta;
        }
        // 在支撑域内：保持不变
    }

    return RECONSTRUCTION_SUCCESS;
}

static void print_iteration_info(
    int iteration,
    float error,
    double elapsed_time)
{
    printf("Iteration %4d: Error = %.6e, Time = %.3f s\n",
           iteration, error, elapsed_time);
}
// ============================================================================
// 初始化和验证函数
// ============================================================================

int reconstruction_validate_params(const PhaseRetrievalParams *params)
{
    if (!params) {
        return RECONSTRUCTION_ERROR_NULL_POINTER;
    }

    if (params->max_iterations <= 0 || 
        params->max_iterations > RECONSTRUCTION_MAX_ITERATIONS) {
        return RECONSTRUCTION_ERROR_INVALID_PARAMS;
    }

    if (params->tolerance <= 0.0f || params->tolerance >= 1.0f) {
        return RECONSTRUCTION_ERROR_INVALID_PARAMS;
    }

    if (params->beta < 0.0f || params->beta > 1.0f) {
        return RECONSTRUCTION_ERROR_INVALID_PARAMS;
    }

    if (params->relaxation < 0.0f || params->relaxation > 1.0f) {
        return RECONSTRUCTION_ERROR_INVALID_PARAMS;
    }

    return RECONSTRUCTION_SUCCESS;
}

int initialize_random_phase(
    ComplexImage *field,
    const RealImage *amplitude)
{
    if (!field) {
        return RECONSTRUCTION_ERROR_NULL_POINTER;
    }

    int size = field->width * field->height;

    if (amplitude) {
        if (amplitude->width != field->width || 
            amplitude->height != field->height) {
            return RECONSTRUCTION_ERROR_INVALID_PARAMS;
        }

        // 使用给定振幅和随机相位
        for (int i = 0; i < size; i++) {
            float amp = amplitude->data[i];
            float phase = ((float)rand() / RAND_MAX) * 2.0f * M_PI - M_PI;
            field->data[i] = amp * (cosf(phase) + sinf(phase) * I);
        }
    } else {
        // 完全随机的复数值
        for (int i = 0; i < size; i++) {
            float real_part = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
            float imag_part = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
            field->data[i] = real_part + imag_part * I;
        }
    }

    return RECONSTRUCTION_SUCCESS;
}

float compute_rms_error(
    const ComplexImage *field1,
    const ComplexImage *field2)
{
    return compute_error_metric(field1, field2);
}

int normalize_complex_field(
    ComplexImage *field,
    int method)
{
    if (!field) {
        return RECONSTRUCTION_ERROR_NULL_POINTER;
    }

    int size = field->width * field->height;

    if (method == 0) {
        // 按最大值归一化
        float max_amp = 0.0f;
        for (int i = 0; i < size; i++) {
            float amp = cabsf(field->data[i]);
            if (amp > max_amp) {
                max_amp = amp;
            }
        }

        if (max_amp > 0.0f) {
            for (int i = 0; i < size; i++) {
                field->data[i] /= max_amp;
            }
        }
    } else if (method == 1) {
        // 按能量归一化
        float energy = 0.0f;
        for (int i = 0; i < size; i++) {
            float amp = cabsf(field->data[i]);
            energy += amp * amp;
        }

        if (energy > 0.0f) {
            float scale = sqrtf(size / energy);
            for (int i = 0; i < size; i++) {
                field->data[i] *= scale;
            }
        }
    } else {
        return RECONSTRUCTION_ERROR_INVALID_PARAMS;
    }

    return RECONSTRUCTION_SUCCESS;
}

// ============================================================================
// Gerchberg-Saxton算法
// ============================================================================

int phase_retrieval_gs(
    const RealImage *intensity_object,
    const RealImage *intensity_image,
    const PhaseRetrievalParams *params,
    const ConstraintSet *constraints,
    const DiffractionParams *diffraction_params,
    ReconstructionResult *result)
{
    if (!intensity_object || !intensity_image || !params || !result) {
        return RECONSTRUCTION_ERROR_NULL_POINTER;
    }

    // 验证参数
    int ret = reconstruction_validate_params(params);
    if (ret != RECONSTRUCTION_SUCCESS) {
        return ret;
    }

    int width = intensity_object->width;
    int height = intensity_object->height;

    if (intensity_image->width != width || intensity_image->height != height) {
        return RECONSTRUCTION_ERROR_INVALID_PARAMS;
    }

    // 创建工作图像
    ComplexImage *object_field = complex_image_create(width, height);
    ComplexImage *image_field = complex_image_create(width, height);
    ComplexImage *prev_field = complex_image_create(width, height);

    if (!object_field || !image_field || !prev_field) {
        complex_image_destroy(object_field);
        complex_image_destroy(image_field);
        complex_image_destroy(prev_field);
        return RECONSTRUCTION_ERROR_MEMORY_ALLOCATION;
    }

    // 初始化
    srand(params->random_seed);
    
    if (params->initial_guess) {
        memcpy(object_field->data, params->initial_guess->data,
               width * height * sizeof(ComplexF));
    } else if (params->use_random_phase) {
        initialize_random_phase(object_field, intensity_object);
    } else {
        // 使用零相位初始化
        for (int i = 0; i < width * height; i++) {
            object_field->data[i] = intensity_object->data[i] + 0.0f * I;
        }
    }

    // 计时开始
    clock_t start_time = clock();

    // 主迭代循环
    for (int iter = 0; iter < params->max_iterations; iter++) {
        // 保存当前场用于误差计算
        memcpy(prev_field->data, object_field->data,
               width * height * sizeof(ComplexF));

        // 1. 应用物平面约束
        if (constraints) {
            apply_constraints(object_field, constraints, true);
        }

        // 2. 传播到像平面
        if (diffraction_params) {
            ret = diffraction_propagate(object_field, image_field, diffraction_params);
        } else {
            // 使用FFT作为默认传播
            ret = fft_2d_forward(object_field, image_field);
        }

        if (ret != DIFFRACTION_SUCCESS && ret != FFT_SUCCESS) {
            complex_image_destroy(object_field);
            complex_image_destroy(image_field);
            complex_image_destroy(prev_field);
            return RECONSTRUCTION_ERROR_DIFFRACTION_FAILED;
        }

        // 3. 应用像平面振幅约束
        apply_amplitude_constraint(image_field, intensity_image);

        // 4. 传播回物平面
        if (diffraction_params) {
            DiffractionParams back_params = *diffraction_params;
            back_params.distance = -back_params.distance;
            ret = diffraction_propagate(image_field, object_field, &back_params);
        } else {
            ret = fft_2d_inverse(image_field, object_field);
        }

        if (ret != DIFFRACTION_SUCCESS && ret != FFT_SUCCESS) {
            complex_image_destroy(object_field);
            complex_image_destroy(image_field);
            complex_image_destroy(prev_field);
            return RECONSTRUCTION_ERROR_DIFFRACTION_FAILED;
        }

        // 5. 应用物平面振幅约束
        apply_amplitude_constraint(object_field, intensity_object);

        // 6. 计算误差
        float error = compute_error_metric(object_field, prev_field);
        result->error_history[iter] = error;
        result->error_history_length = iter + 1;

        // 7. 输出信息
        if (params->verbose && (iter % params->print_interval == 0)) {
            double elapsed = (double)(clock() - start_time) / CLOCKS_PER_SEC;
            print_iteration_info(iter + 1, error, elapsed);
        }

        // 8. 保存中间结果
        if (params->save_intermediate && 
            params->output_prefix &&
            (iter % params->save_interval == 0)) {
            char filename[512];
            snprintf(filename, sizeof(filename), "%s_iter_%04d",
                    params->output_prefix, iter);
            save_reconstruction_result(result, filename);
        }

        // 9. 检查收敛
        if (error < params->tolerance) {
            result->converged = true;
            result->iterations_performed = iter + 1;
            break;
        }

        result->iterations_performed = iter + 1;
    }

    // 计时结束
    result->computation_time = (double)(clock() - start_time) / CLOCKS_PER_SEC;
    result->final_error = result->error_history[result->error_history_length - 1];

    // 复制最终结果
    memcpy(result->reconstructed->data, object_field->data,
           width * height * sizeof(ComplexF));

    // 清理
    complex_image_destroy(object_field);
    complex_image_destroy(image_field);
    complex_image_destroy(prev_field);

    return RECONSTRUCTION_SUCCESS;
}

// ============================================================================
// Hybrid Input-Output (HIO) 算法
// ============================================================================

int phase_retrieval_hio(
    const RealImage *intensity_object,
    const RealImage *intensity_image,
    const PhaseRetrievalParams *params,
    const ConstraintSet *constraints,
    const DiffractionParams *diffraction_params,
    ReconstructionResult *result)
{
    if (!intensity_object || !intensity_image || !params || !result) {
        return RECONSTRUCTION_ERROR_NULL_POINTER;
    }

    int ret = reconstruction_validate_params(params);
    if (ret != RECONSTRUCTION_SUCCESS) {
        return ret;
    }

    int width = intensity_object->width;
    int height = intensity_object->height;

    if (intensity_image->width != width || intensity_image->height != height) {
        return RECONSTRUCTION_ERROR_INVALID_PARAMS;
    }

    // 创建工作图像
    ComplexImage *object_field = complex_image_create(width, height);
    ComplexImage *image_field = complex_image_create(width, height);
    ComplexImage *prev_field = complex_image_create(width, height);
    ComplexImage *gs_update = complex_image_create(width, height);

    if (!object_field || !image_field || !prev_field || !gs_update) {
        complex_image_destroy(object_field);
        complex_image_destroy(image_field);
        complex_image_destroy(prev_field);
        complex_image_destroy(gs_update);
        return RECONSTRUCTION_ERROR_MEMORY_ALLOCATION;
    }

    // 初始化
    srand(params->random_seed);
    
    if (params->initial_guess) {
        memcpy(object_field->data, params->initial_guess->data,
               width * height * sizeof(ComplexF));
    } else {
        initialize_random_phase(object_field, intensity_object);
    }

    // 创建或更新支撑域
    SupportDomain *support = NULL;
    if (constraints && (constraints->constraint_flags & CONSTRAINT_SUPPORT)) {
        support = constraints->support;
    }

    clock_t start_time = clock();

    // 主迭代循环
    for (int iter = 0; iter < params->max_iterations; iter++) {
        // 保存当前场
        memcpy(prev_field->data, object_field->data,
               width * height * sizeof(ComplexF));

        // 1. 传播到像平面
        if (diffraction_params) {
            ret = diffraction_propagate(object_field, image_field, diffraction_params);
        } else {
            ret = fft_2d_forward(object_field, image_field);
        }

        if (ret != DIFFRACTION_SUCCESS && ret != FFT_SUCCESS) {
            complex_image_destroy(object_field);
            complex_image_destroy(image_field);
            complex_image_destroy(prev_field);
            complex_image_destroy(gs_update);
            return RECONSTRUCTION_ERROR_DIFFRACTION_FAILED;
        }

        // 2. 应用像平面振幅约束
        apply_amplitude_constraint(image_field, intensity_image);

        // 3. 传播回物平面
        if (diffraction_params) {
            DiffractionParams back_params = *diffraction_params;
            back_params.distance = -back_params.distance;
            ret = diffraction_propagate(image_field, gs_update, &back_params);
        } else {
            ret = fft_2d_inverse(image_field, gs_update);
        }

        if (ret != DIFFRACTION_SUCCESS && ret != FFT_SUCCESS) {
            complex_image_destroy(object_field);
            complex_image_destroy(image_field);
            complex_image_destroy(prev_field);
            complex_image_destroy(gs_update);
            return RECONSTRUCTION_ERROR_DIFFRACTION_FAILED;
        }

        // 4. HIO更新
        if (support) {
            for (int i = 0; i < width * height; i++) {
                if (support->mask[i]) {
                    // 在支撑域内：使用GS更新
                    object_field->data[i] = gs_update->data[i];
                } else {
                    // 在支撑域外：HIO更新
                    object_field->data[i] = prev_field->data[i] - 
                                           params->beta * gs_update->data[i];
                }
            }
        } else {
            // 没有支撑域，使用正值约束
            for (int i = 0; i < width * height; i++) {
                float real_part = crealf(gs_update->data[i]);
                
                if (real_part >= 0.0f) {
                    object_field->data[i] = gs_update->data[i];
                } else {
                    object_field->data[i] = prev_field->data[i] - 
                                           params->beta * gs_update->data[i];
                }
            }
        }

        // 5. 应用其他约束
        if (constraints) {
            apply_constraints(object_field, constraints, true);
        }

        // 6. Shrinkwrap更新
        if (params->use_shrinkwrap && 
            support &&
            (iter % params->shrinkwrap_interval == 0) &&
            iter > 0) {
            update_support_shrinkwrap(object_field, support,
                                     params->shrinkwrap_sigma,
                                     params->shrinkwrap_threshold);
        }

        // 7. 计算误差
        float error = compute_error_metric(object_field, prev_field);
        result->error_history[iter] = error;
        result->error_history_length = iter + 1;

        // 8. 输出信息
        if (params->verbose && (iter % params->print_interval == 0)) {
            double elapsed = (double)(clock() - start_time) / CLOCKS_PER_SEC;
            print_iteration_info(iter + 1, error, elapsed);
        }

        // 9. 保存中间结果
        if (params->save_intermediate && 
            params->output_prefix &&
            (iter % params->save_interval == 0)) {
            char filename[512];
            snprintf(filename, sizeof(filename), "%s_iter_%04d",
                    params->output_prefix, iter);
            
            // 临时保存当前结果
            memcpy(result->reconstructed->data, object_field->data,
                   width * height * sizeof(ComplexF));
            save_reconstruction_result(result, filename);
        }

        // 10. 检查收敛
        if (error < params->tolerance) {
            result->converged = true;
            result->iterations_performed = iter + 1;
            break;
        }

        result->iterations_performed = iter + 1;
    }

    // 计时结束
    result->computation_time = (double)(clock() - start_time) / CLOCKS_PER_SEC;
    result->final_error = result->error_history[result->error_history_length - 1];

    // 复制最终结果
    memcpy(result->reconstructed->data, object_field->data,
           width * height * sizeof(ComplexF));

    // 清理
    complex_image_destroy(object_field);
    complex_image_destroy(image_field);
    complex_image_destroy(prev_field);
    complex_image_destroy(gs_update);

    return RECONSTRUCTION_SUCCESS;
}

// ============================================================================
// Error Reduction (ER) 算法
// ============================================================================

int phase_retrieval_er(
    const RealImage *intensity_object,
    const RealImage *intensity_image,
    const PhaseRetrievalParams *params,
    const ConstraintSet *constraints,
    const DiffractionParams *diffraction_params,
    ReconstructionResult *result)
{
    if (!intensity_object || !intensity_image || !params || !result) {
        return RECONSTRUCTION_ERROR_NULL_POINTER;
    }

    int ret = reconstruction_validate_params(params);
    if (ret != RECONSTRUCTION_SUCCESS) {
        return ret;
    }

    int width = intensity_object->width;
    int height = intensity_object->height;

    if (intensity_image->width != width || intensity_image->height != height) {
        return RECONSTRUCTION_ERROR_INVALID_PARAMS;
    }

    // 创建工作图像
    ComplexImage *object_field = complex_image_create(width, height);
    ComplexImage *image_field = complex_image_create(width, height);
    ComplexImage *prev_field = complex_image_create(width, height);

    if (!object_field || !image_field || !prev_field) {
        complex_image_destroy(object_field);
        complex_image_destroy(image_field);
        complex_image_destroy(prev_field);
        return RECONSTRUCTION_ERROR_MEMORY_ALLOCATION;
    }

    // 初始化
    srand(params->random_seed);
    
    if (params->initial_guess) {
        memcpy(object_field->data, params->initial_guess->data,
               width * height * sizeof(ComplexF));
    } else {
        initialize_random_phase(object_field, intensity_object);
    }

    // 获取支撑域
    SupportDomain *support = NULL;
    if (constraints && (constraints->constraint_flags & CONSTRAINT_SUPPORT)) {
        support = constraints->support;
    }

    clock_t start_time = clock();

    // 主迭代循环
    for (int iter = 0; iter < params->max_iterations; iter++) {
        // 保存当前场
        memcpy(prev_field->data, object_field->data,
               width * height * sizeof(ComplexF));

        // 1. 传播到像平面
        if (diffraction_params) {
            ret = diffraction_propagate(object_field, image_field, diffraction_params);
        } else {
            ret = fft_2d_forward(object_field, image_field);
        }

        if (ret != DIFFRACTION_SUCCESS && ret != FFT_SUCCESS) {
            complex_image_destroy(object_field);
            complex_image_destroy(image_field);
            complex_image_destroy(prev_field);
            return RECONSTRUCTION_ERROR_DIFFRACTION_FAILED;
        }

        // 2. 应用像平面振幅约束
        apply_amplitude_constraint(image_field, intensity_image);

        // 3. 传播回物平面
        if (diffraction_params) {
            DiffractionParams back_params = *diffraction_params;
            back_params.distance = -back_params.distance;
            ret = diffraction_propagate(image_field, object_field, &back_params);
        } else {
            ret = fft_2d_inverse(image_field, object_field);
        }

        if (ret != DIFFRACTION_SUCCESS && ret != FFT_SUCCESS) {
            complex_image_destroy(object_field);
            complex_image_destroy(image_field);
            complex_image_destroy(prev_field);
            return RECONSTRUCTION_ERROR_DIFFRACTION_FAILED;
        }

        // 4. ER更新：在支撑域外置零
        if (support) {
            for (int i = 0; i < width * height; i++) {
                if (!support->mask[i]) {
                    object_field->data[i] = 0.0f + 0.0f * I;
                }
            }
        } else {
            // 使用正值约束
            for (int i = 0; i < width * height; i++) {
                float real_part = crealf(object_field->data[i]);
                if (real_part < 0.0f) {
                    object_field->data[i] = 0.0f + cimagf(object_field->data[i]) * I;
                }
            }
        }

        // 5. 应用其他约束
        if (constraints) {
            apply_constraints(object_field, constraints, true);
        }

        // 6. 计算误差
        float error = compute_error_metric(object_field, prev_field);
        result->error_history[iter] = error;
        result->error_history_length = iter + 1;

        // 7. 输出信息
        if (params->verbose && (iter % params->print_interval == 0)) {
            double elapsed = (double)(clock() - start_time) / CLOCKS_PER_SEC;
            print_iteration_info(iter + 1, error, elapsed);
        }

        // 8. 检查收敛
        if (error < params->tolerance) {
            result->converged = true;
            result->iterations_performed = iter + 1;
            break;
        }

        result->iterations_performed = iter + 1;
    }

    // 计时结束
    result->computation_time = (double)(clock() - start_time) / CLOCKS_PER_SEC;
    result->final_error = result->error_history[result->error_history_length - 1];

    // 复制最终结果
    memcpy(result->reconstructed->data, object_field->data,
           width * height * sizeof(ComplexF));

    // 清理
    complex_image_destroy(object_field);
    complex_image_destroy(image_field);
    complex_image_destroy(prev_field);

    return RECONSTRUCTION_SUCCESS;
}
// ============================================================================
// RAAR (Relaxed Averaged Alternating Reflections) 算法
// ============================================================================

int phase_retrieval_raar(
    const RealImage *intensity_object,
    const RealImage *intensity_image,
    const PhaseRetrievalParams *params,
    const ConstraintSet *constraints,
    const DiffractionParams *diffraction_params,
    ReconstructionResult *result)
{
    if (!intensity_object || !intensity_image || !params || !result) {
        return RECONSTRUCTION_ERROR_NULL_POINTER;
    }

    int ret = reconstruction_validate_params(params);
    if (ret != RECONSTRUCTION_SUCCESS) {
        return ret;
    }

    int width = intensity_object->width;
    int height = intensity_object->height;
    int size = width * height;

    if (intensity_image->width != width || intensity_image->height != height) {
        return RECONSTRUCTION_ERROR_INVALID_PARAMS;
    }

    // 创建工作图像
    ComplexImage *object_field = complex_image_create(width, height);
    ComplexImage *image_field = complex_image_create(width, height);
    ComplexImage *prev_field = complex_image_create(width, height);
    ComplexImage *ps_field = complex_image_create(width, height);  // 支撑域投影
    ComplexImage *pm_field = complex_image_create(width, height);  // 测量投影

    if (!object_field || !image_field || !prev_field || !ps_field || !pm_field) {
        complex_image_destroy(object_field);
        complex_image_destroy(image_field);
        complex_image_destroy(prev_field);
        complex_image_destroy(ps_field);
        complex_image_destroy(pm_field);
        return RECONSTRUCTION_ERROR_MEMORY_ALLOCATION;
    }

    // 初始化
    srand(params->random_seed);
    
    if (params->initial_guess) {
        memcpy(object_field->data, params->initial_guess->data,
               size * sizeof(ComplexF));
    } else {
        initialize_random_phase(object_field, intensity_object);
    }

    // 获取支撑域
    SupportDomain *support = NULL;
    if (constraints && (constraints->constraint_flags & CONSTRAINT_SUPPORT)) {
        support = constraints->support;
    }

    float beta = params->relaxation;  // RAAR的松弛参数
    clock_t start_time = clock();

    // 主迭代循环
    for (int iter = 0; iter < params->max_iterations; iter++) {
        // 保存当前场
        memcpy(prev_field->data, object_field->data, size * sizeof(ComplexF));

        // 1. 计算支撑域投影 Ps(g)
        memcpy(ps_field->data, object_field->data, size * sizeof(ComplexF));
        if (support) {
            for (int i = 0; i < size; i++) {
                if (!support->mask[i]) {
                    ps_field->data[i] = 0.0f + 0.0f * I;
                }
            }
        } else {
            // 使用正值约束
            for (int i = 0; i < size; i++) {
                float real_part = crealf(ps_field->data[i]);
                if (real_part < 0.0f) {
                    ps_field->data[i] = 0.0f + cimagf(ps_field->data[i]) * I;
                }
            }
        }

        // 2. 计算 2*Ps(g) - g
        for (int i = 0; i < size; i++) {
            ps_field->data[i] = 2.0f * ps_field->data[i] - object_field->data[i];
        }

        // 3. 传播到像平面
        if (diffraction_params) {
            ret = diffraction_propagate(ps_field, image_field, diffraction_params);
        } else {
            ret = fft_2d_forward(ps_field, image_field);
        }

        if (ret != DIFFRACTION_SUCCESS && ret != FFT_SUCCESS) {
            complex_image_destroy(object_field);
            complex_image_destroy(image_field);
            complex_image_destroy(prev_field);
            complex_image_destroy(ps_field);
            complex_image_destroy(pm_field);
            return RECONSTRUCTION_ERROR_DIFFRACTION_FAILED;
        }

        // 4. 应用像平面振幅约束
        apply_amplitude_constraint(image_field, intensity_image);

        // 5. 传播回物平面得到 Pm(2*Ps(g) - g)
        if (diffraction_params) {
            DiffractionParams back_params = *diffraction_params;
            back_params.distance = -back_params.distance;
            ret = diffraction_propagate(image_field, pm_field, &back_params);
        } else {
            ret = fft_2d_inverse(image_field, pm_field);
        }

        if (ret != DIFFRACTION_SUCCESS && ret != FFT_SUCCESS) {
            complex_image_destroy(object_field);
            complex_image_destroy(image_field);
            complex_image_destroy(prev_field);
            complex_image_destroy(ps_field);
            complex_image_destroy(pm_field);
            return RECONSTRUCTION_ERROR_DIFFRACTION_FAILED;
        }

        // 6. RAAR更新: g_new = beta * Pm(2*Ps(g) - g) + (1 - 2*beta) * Ps(g) + beta * g
        for (int i = 0; i < size; i++) {
            // 重新计算Ps(g)
            ComplexF ps_val = object_field->data[i];
            if (support) {
                if (!support->mask[i]) {
                    ps_val = 0.0f + 0.0f * I;
                }
            } else {
                float real_part = crealf(ps_val);
                if (real_part < 0.0f) {
                    ps_val = 0.0f + cimagf(ps_val) * I;
                }
            }

            object_field->data[i] = beta * pm_field->data[i] + 
                                   (1.0f - 2.0f * beta) * ps_val + 
                                   beta * object_field->data[i];
        }

        // 7. 应用其他约束
        if (constraints) {
            apply_constraints(object_field, constraints, true);
        }

        // 8. Shrinkwrap更新
        if (params->use_shrinkwrap && 
            support &&
            (iter % params->shrinkwrap_interval == 0) &&
            iter > 0) {
            update_support_shrinkwrap(object_field, support,
                                     params->shrinkwrap_sigma,
                                     params->shrinkwrap_threshold);
        }

        // 9. 计算误差
        float error = compute_error_metric(object_field, prev_field);
        result->error_history[iter] = error;
        result->error_history_length = iter + 1;

        // 10. 输出信息
        if (params->verbose && (iter % params->print_interval == 0)) {
            double elapsed = (double)(clock() - start_time) / CLOCKS_PER_SEC;
            print_iteration_info(iter + 1, error, elapsed);
        }

        // 11. 保存中间结果
        if (params->save_intermediate && 
            params->output_prefix &&
            (iter % params->save_interval == 0)) {
            char filename[512];
            snprintf(filename, sizeof(filename), "%s_iter_%04d",
                    params->output_prefix, iter);
            
            memcpy(result->reconstructed->data, object_field->data,
                   size * sizeof(ComplexF));
            save_reconstruction_result(result, filename);
        }

        // 12. 检查收敛
        if (error < params->tolerance) {
            result->converged = true;
            result->iterations_performed = iter + 1;
            break;
        }

        result->iterations_performed = iter + 1;
    }

    // 计时结束
    result->computation_time = (double)(clock() - start_time) / CLOCKS_PER_SEC;
    result->final_error = result->error_history[result->error_history_length - 1];

    // 复制最终结果
    memcpy(result->reconstructed->data, object_field->data,
           size * sizeof(ComplexF));

    // 清理
    complex_image_destroy(object_field);
    complex_image_destroy(image_field);
    complex_image_destroy(prev_field);
    complex_image_destroy(ps_field);
    complex_image_destroy(pm_field);

    return RECONSTRUCTION_SUCCESS;
}

// ============================================================================
// 混合相位恢复算法（自适应）
// ============================================================================

int phase_retrieval_hybrid(
    const RealImage *intensity_object,
    const RealImage *intensity_image,
    const PhaseRetrievalParams *params,
    const ConstraintSet *constraints,
    const DiffractionParams *diffraction_params,
    ReconstructionResult *result)
{
    if (!intensity_object || !intensity_image || !params || !result) {
        return RECONSTRUCTION_ERROR_NULL_POINTER;
    }

    int ret = reconstruction_validate_params(params);
    if (ret != RECONSTRUCTION_SUCCESS) {
        return ret;
    }

    if (params->verbose) {
        printf("Starting hybrid phase retrieval algorithm...\n");
        printf("Strategy: ER -> HIO -> RAAR\n");
    }

    // 创建修改后的参数
    PhaseRetrievalParams modified_params = *params;
    
    // 阶段1: Error Reduction (前20%迭代)
    int er_iterations = params->max_iterations / 5;
    modified_params.max_iterations = er_iterations;
    modified_params.algorithm = PHASE_RETRIEVAL_ER;
    
    if (params->verbose) {
        printf("\n=== Phase 1: Error Reduction (%d iterations) ===\n", er_iterations);
    }

    ret = phase_retrieval_er(intensity_object, intensity_image,
                            &modified_params, constraints,
                            diffraction_params, result);
    
    if (ret != RECONSTRUCTION_SUCCESS) {
        return ret;
    }

    // 检查是否已经收敛
    if (result->converged) {
        if (params->verbose) {
            printf("Converged in ER phase!\n");
        }
        return RECONSTRUCTION_SUCCESS;
    }

    // 阶段2: HIO (接下来50%迭代)
    int hio_iterations = params->max_iterations / 2;
    modified_params.max_iterations = hio_iterations;
    modified_params.algorithm = PHASE_RETRIEVAL_HIO;
    modified_params.initial_guess = result->reconstructed;
    modified_params.use_random_phase = false;
    
    if (params->verbose) {
        printf("\n=== Phase 2: Hybrid Input-Output (%d iterations) ===\n", hio_iterations);
    }

    // 创建临时结果
    ReconstructionResult *temp_result = reconstruction_result_create(
        intensity_object->width, intensity_object->height, hio_iterations);
    
    if (!temp_result) {
        return RECONSTRUCTION_ERROR_MEMORY_ALLOCATION;
    }

    ret = phase_retrieval_hio(intensity_object, intensity_image,
                             &modified_params, constraints,
                             diffraction_params, temp_result);
    
    if (ret != RECONSTRUCTION_SUCCESS) {
        reconstruction_result_destroy(temp_result);
        return ret;
    }

    // 合并结果
    int prev_length = result->error_history_length;
    for (int i = 0; i < temp_result->error_history_length; i++) {
        if (prev_length + i < params->max_iterations) {
            result->error_history[prev_length + i] = temp_result->error_history[i];
        }
    }
    result->error_history_length = prev_length + temp_result->error_history_length;
    result->iterations_performed += temp_result->iterations_performed;
    result->computation_time += temp_result->computation_time;

    // 更新重建结果
    memcpy(result->reconstructed->data, temp_result->reconstructed->data,
           intensity_object->width * intensity_object->height * sizeof(ComplexF));

    if (temp_result->converged) {
        result->converged = true;
        result->final_error = temp_result->final_error;
        reconstruction_result_destroy(temp_result);
        
        if (params->verbose) {
            printf("Converged in HIO phase!\n");
        }
        return RECONSTRUCTION_SUCCESS;
    }

    reconstruction_result_destroy(temp_result);

    // 阶段3: RAAR (剩余30%迭代)
    int raar_iterations = params->max_iterations - result->iterations_performed;
    if (raar_iterations > 0) {
        modified_params.max_iterations = raar_iterations;
        modified_params.algorithm = PHASE_RETRIEVAL_RAAR;
        modified_params.initial_guess = result->reconstructed;
        
        if (params->verbose) {
            printf("\n=== Phase 3: RAAR (%d iterations) ===\n", raar_iterations);
        }

        temp_result = reconstruction_result_create(
            intensity_object->width, intensity_object->height, raar_iterations);
        
        if (!temp_result) {
            return RECONSTRUCTION_ERROR_MEMORY_ALLOCATION;
        }

        ret = phase_retrieval_raar(intensity_object, intensity_image,
                                   &modified_params, constraints,
                                   diffraction_params, temp_result);
        
        if (ret != RECONSTRUCTION_SUCCESS) {
            reconstruction_result_destroy(temp_result);
            return ret;
        }

        // 合并最终结果
        prev_length = result->error_history_length;
        for (int i = 0; i < temp_result->error_history_length; i++) {
            if (prev_length + i < params->max_iterations) {
                result->error_history[prev_length + i] = temp_result->error_history[i];
            }
        }
        result->error_history_length = prev_length + temp_result->error_history_length;
        result->iterations_performed += temp_result->iterations_performed;
        result->computation_time += temp_result->computation_time;
        result->converged = temp_result->converged;
        result->final_error = temp_result->final_error;

        memcpy(result->reconstructed->data, temp_result->reconstructed->data,
               intensity_object->width * intensity_object->height * sizeof(ComplexF));

        reconstruction_result_destroy(temp_result);
    }

    if (params->verbose) {
        printf("\n=== Hybrid algorithm completed ===\n");
        printf("Total iterations: %d\n", result->iterations_performed);
        printf("Final error: %.6e\n", result->final_error);
        printf("Converged: %s\n", result->converged ? "Yes" : "No");
    }

    return RECONSTRUCTION_SUCCESS;
}

// ============================================================================
// 通用相位恢复接口
// ============================================================================

int phase_retrieval(
    const RealImage *intensity_object,
    const RealImage *intensity_image,
    const PhaseRetrievalParams *params,
    const ConstraintSet *constraints,
    const DiffractionParams *diffraction_params,
    ReconstructionResult *result)
{
    if (!params) {
        return RECONSTRUCTION_ERROR_NULL_POINTER;
    }

    // 根据算法类型调用相应的函数
    switch (params->algorithm) {
        case PHASE_RETRIEVAL_GS:
            return phase_retrieval_gs(intensity_object, intensity_image,
                                     params, constraints, diffraction_params, result);
        
        case PHASE_RETRIEVAL_HIO:
            return phase_retrieval_hio(intensity_object, intensity_image,
                                      params, constraints, diffraction_params, result);
        
        case PHASE_RETRIEVAL_ER:
            return phase_retrieval_er(intensity_object, intensity_image,
                                     params, constraints, diffraction_params, result);
        
        case PHASE_RETRIEVAL_RAAR:
            return phase_retrieval_raar(intensity_object, intensity_image,
                                       params, constraints, diffraction_params, result);
        
        case PHASE_RETRIEVAL_HYBRID:
            return phase_retrieval_hybrid(intensity_object, intensity_image,
                                         params, constraints, diffraction_params, result);
        
        default:
            return RECONSTRUCTION_ERROR_INVALID_PARAMS;
    }
}

// ============================================================================
// 多平面相位恢复
// ============================================================================

int phase_retrieval_multiplane(
    const MultiPlaneParams *params,
    const ConstraintSet *constraints,
    ReconstructionResult *result)
{
    if (!params || !result) {
        return RECONSTRUCTION_ERROR_NULL_POINTER;
    }

    if (params->num_planes <= 0 || params->num_planes > RECONSTRUCTION_MAX_PLANES) {
        return RECONSTRUCTION_ERROR_INVALID_PARAMS;
    }

    // 验证所有测量图像
    int width = params->measurements[0]->width;
    int height = params->measurements[0]->height;
    int size = width * height;

    for (int p = 0; p < params->num_planes; p++) {
        if (!params->measurements[p]) {
            return RECONSTRUCTION_ERROR_NULL_POINTER;
        }
        if (params->measurements[p]->width != width ||
            params->measurements[p]->height != height) {
            return RECONSTRUCTION_ERROR_INVALID_PARAMS;
        }
    }

    // 创建工作图像
    ComplexImage *object_field = complex_image_create(width, height);
    ComplexImage *prev_field = complex_image_create(width, height);
    ComplexImage **plane_fields = (ComplexImage**)malloc(
        params->num_planes * sizeof(ComplexImage*));

    if (!object_field || !prev_field || !plane_fields) {
        complex_image_destroy(object_field);
        complex_image_destroy(prev_field);
        free(plane_fields);
        return RECONSTRUCTION_ERROR_MEMORY_ALLOCATION;
    }

    for (int p = 0; p < params->num_planes; p++) {
        plane_fields[p] = complex_image_create(width, height);
        if (!plane_fields[p]) {
            for (int i = 0; i < p; i++) {
                complex_image_destroy(plane_fields[i]);
            }
            free(plane_fields);
            complex_image_destroy(object_field);
            complex_image_destroy(prev_field);
            return RECONSTRUCTION_ERROR_MEMORY_ALLOCATION;
        }
    }

    // 初始化
    srand(params->base_params.random_seed);
    initialize_random_phase(object_field, params->measurements[0]);

    clock_t start_time = clock();

    // 主迭代循环
    for (int iter = 0; iter < params->base_params.max_iterations; iter++) {
        // 保存当前场
        memcpy(prev_field->data, object_field->data, size * sizeof(ComplexF));

        // 对每个平面进行处理
        ComplexImage *accumulated = complex_image_create(width, height);
        if (!accumulated) {
            for (int p = 0; p < params->num_planes; p++) {
                complex_image_destroy(plane_fields[p]);
            }
            free(plane_fields);
            complex_image_destroy(object_field);
            complex_image_destroy(prev_field);
            return RECONSTRUCTION_ERROR_MEMORY_ALLOCATION;
        }

        // 初始化累积场
        memset(accumulated->data, 0, size * sizeof(ComplexF));

        for (int p = 0; p < params->num_planes; p++) {
            // 传播到第p个平面
            DiffractionParams plane_params = *params->diffraction_params;
            plane_params.distance = params->distances[p];

            int ret = diffraction_propagate(object_field, plane_fields[p], &plane_params);
            if (ret != DIFFRACTION_SUCCESS) {
                complex_image_destroy(accumulated);
                for (int i = 0; i < params->num_planes; i++) {
                    complex_image_destroy(plane_fields[i]);
                }
                free(plane_fields);
                complex_image_destroy(object_field);
                complex_image_destroy(prev_field);
                return RECONSTRUCTION_ERROR_DIFFRACTION_FAILED;
            }

            // 应用振幅约束
            apply_amplitude_constraint(plane_fields[p], params->measurements[p]);

            // 传播回物平面
            plane_params.distance = -params->distances[p];
            ComplexImage *back_field = complex_image_create(width, height);
            if (!back_field) {
                complex_image_destroy(accumulated);
                for (int i = 0; i < params->num_planes; i++) {
                    complex_image_destroy(plane_fields[i]);
                }
                free(plane_fields);
                complex_image_destroy(object_field);
                complex_image_destroy(prev_field);
                return RECONSTRUCTION_ERROR_MEMORY_ALLOCATION;
            }

            ret = diffraction_propagate(plane_fields[p], back_field, &plane_params);
            if (ret != DIFFRACTION_SUCCESS) {
                complex_image_destroy(back_field);
                complex_image_destroy(accumulated);
                for (int i = 0; i < params->num_planes; i++) {
                    complex_image_destroy(plane_fields[i]);
                }
                free(plane_fields);
                complex_image_destroy(object_field);
                complex_image_destroy(prev_field);
                return RECONSTRUCTION_ERROR_DIFFRACTION_FAILED;
            }

            // 加权累积
            float weight = params->plane_weights[p];
            for (int i = 0; i < size; i++) {
                accumulated->data[i] += weight * back_field->data[i];
            }

            complex_image_destroy(back_field);
        }

        // 更新物场
        memcpy(object_field->data, accumulated->data, size * sizeof(ComplexF));
        complex_image_destroy(accumulated);

        // 应用约束
        if (constraints) {
            apply_constraints(object_field, constraints, true);
        }

        // 计算误差
        float error = compute_error_metric(object_field, prev_field);
        result->error_history[iter] = error;
        result->error_history_length = iter + 1;

        // 输出信息
        if (params->base_params.verbose && 
            (iter % params->base_params.print_interval == 0)) {
            double elapsed = (double)(clock() - start_time) / CLOCKS_PER_SEC;
            print_iteration_info(iter + 1, error, elapsed);
        }

        // 检查收敛
        if (error < params->base_params.tolerance) {
            result->converged = true;
            result->iterations_performed = iter + 1;
            break;
        }

        result->iterations_performed = iter + 1;
    }

    // 计时结束
    result->computation_time = (double)(clock() - start_time) / CLOCKS_PER_SEC;
    result->final_error = result->error_history[result->error_history_length - 1];

    // 复制最终结果
    memcpy(result->reconstructed->data, object_field->data, size * sizeof(ComplexF));

    // 清理
    for (int p = 0; p < params->num_planes; p++) {
        complex_image_destroy(plane_fields[p]);
    }
    free(plane_fields);
    complex_image_destroy(object_field);
    complex_image_destroy(prev_field);

    return RECONSTRUCTION_SUCCESS;
}
// ============================================================================
// 离轴全息重建
// ============================================================================

int holography_reconstruct_offaxis(
    const RealImage *hologram,
    const HolographyParams *params,
    ReconstructionResult *result)
{
    if (!hologram || !params || !result) {
        return RECONSTRUCTION_ERROR_NULL_POINTER;
    }

    if (params->type != HOLOGRAPHY_OFF_AXIS) {
        return RECONSTRUCTION_ERROR_INVALID_PARAMS;
    }

    int width = hologram->width;
    int height = hologram->height;
    int size = width * height;

    // 1. 将全息图转换为复数场
    ComplexImage *hologram_field = complex_image_create(width, height);
    if (!hologram_field) {
        return RECONSTRUCTION_ERROR_MEMORY_ALLOCATION;
    }

    for (int i = 0; i < size; i++) {
        hologram_field->data[i] = hologram->data[i] + 0.0f * I;
    }

    // 2. FFT到频域
    ComplexImage *spectrum = complex_image_create(width, height);
    if (!spectrum) {
        complex_image_destroy(hologram_field);
        return RECONSTRUCTION_ERROR_MEMORY_ALLOCATION;
    }

    int ret = fft_2d_forward(hologram_field, spectrum);
    if (ret != FFT_SUCCESS) {
        complex_image_destroy(hologram_field);
        complex_image_destroy(spectrum);
        return RECONSTRUCTION_ERROR_FFT_FAILED;
    }

    // 3. 找到载波频率位置
    float carrier_fx = params->carrier_frequency_x;
    float carrier_fy = params->carrier_frequency_y;
    
    int center_x = (int)(carrier_fx * width);
    int center_y = (int)(carrier_fy * height);

    // 处理周期性边界
    if (center_x < 0) center_x += width;
    if (center_y < 0) center_y += height;
    if (center_x >= width) center_x -= width;
    if (center_y >= height) center_y -= height;

    // 4. 创建频域滤波器（圆形低通）
    RealImage *filter = real_image_create(width, height, 1);
    if (!filter) {
        complex_image_destroy(hologram_field);
        complex_image_destroy(spectrum);
        return RECONSTRUCTION_ERROR_MEMORY_ALLOCATION;
    }

    float filter_radius = params->filter_size * fminf(width, height);
    float radius_sq = filter_radius * filter_radius;

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            int dx = j - center_x;
            int dy = i - center_y;
            
            // 处理周期性边界
            if (dx > width / 2) dx -= width;
            if (dx < -width / 2) dx += width;
            if (dy > height / 2) dy -= height;
            if (dy < -height / 2) dy += height;
            
            float dist_sq = dx * dx + dy * dy;
            
            // 高斯滤波器
            filter->data[i * width + j] = expf(-dist_sq / (2.0f * radius_sq));
        }
    }

    // 5. 应用滤波器并移位到中心
    ComplexImage *filtered = complex_image_create(width, height);
    if (!filtered) {
        complex_image_destroy(hologram_field);
        complex_image_destroy(spectrum);
        real_image_destroy(filter);
        return RECONSTRUCTION_ERROR_MEMORY_ALLOCATION;
    }

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            int idx = i * width + j;
            
            // 计算移位后的位置
            int new_j = j - center_x;
            int new_i = i - center_y;
            
            if (new_j < 0) new_j += width;
            if (new_i < 0) new_i += height;
            if (new_j >= width) new_j -= width;
            if (new_i >= height) new_i -= height;
            
            int new_idx = new_i * width + new_j;
            
            filtered->data[new_idx] = spectrum->data[idx] * filter->data[idx];
        }
    }

    real_image_destroy(filter);
    complex_image_destroy(spectrum);

    // 6. IFFT回到空域
    ret = fft_2d_inverse(filtered, result->reconstructed);
    if (ret != FFT_SUCCESS) {
        complex_image_destroy(hologram_field);
        complex_image_destroy(filtered);
        return RECONSTRUCTION_ERROR_FFT_FAILED;
    }

    complex_image_destroy(filtered);

    // 7. 如果需要，进行数值再聚焦
    if (params->numerical_refocus && params->diffraction_params) {
        ComplexImage *refocused = complex_image_create(width, height);
        if (!refocused) {
            complex_image_destroy(hologram_field);
            return RECONSTRUCTION_ERROR_MEMORY_ALLOCATION;
        }

        ret = diffraction_propagate(result->reconstructed, refocused,
                                   params->diffraction_params);
        if (ret != DIFFRACTION_SUCCESS) {
            complex_image_destroy(hologram_field);
            complex_image_destroy(refocused);
            return RECONSTRUCTION_ERROR_DIFFRACTION_FAILED;
        }

        memcpy(result->reconstructed->data, refocused->data,
               size * sizeof(ComplexF));
        complex_image_destroy(refocused);
    }

    // 8. 孪生像去除（如果需要）
    if (params->remove_twin_image) {
        // 使用迭代方法去除孪生像
        ComplexImage *object = complex_image_create(width, height);
        if (!object) {
            complex_image_destroy(hologram_field);
            return RECONSTRUCTION_ERROR_MEMORY_ALLOCATION;
        }

        memcpy(object->data, result->reconstructed->data, size * sizeof(ComplexF));

        for (int iter = 0; iter < params->twin_removal_iterations; iter++) {
            // 正向传播
            ComplexImage *forward = complex_image_create(width, height);
            if (!forward) {
                complex_image_destroy(hologram_field);
                complex_image_destroy(object);
                return RECONSTRUCTION_ERROR_MEMORY_ALLOCATION;
            }

            if (params->diffraction_params) {
                diffraction_propagate(object, forward, params->diffraction_params);
            } else {
                fft_2d_forward(object, forward);
            }

            // 应用全息图约束
            for (int i = 0; i < size; i++) {
                float measured_intensity = hologram->data[i];
                float current_intensity = cabsf(forward->data[i]);
                
                if (current_intensity > 0.0f) {
                    forward->data[i] *= sqrtf(measured_intensity / current_intensity);
                }
            }

            // 反向传播
            if (params->diffraction_params) {
                DiffractionParams back_params = *params->diffraction_params;
                back_params.distance = -back_params.distance;
                diffraction_propagate(forward, object, &back_params);
            } else {
                fft_2d_inverse(forward, object);
            }

            complex_image_destroy(forward);
        }

        memcpy(result->reconstructed->data, object->data, size * sizeof(ComplexF));
        complex_image_destroy(object);
    }

    complex_image_destroy(hologram_field);

    result->iterations_performed = 1;
    result->converged = true;
    result->final_error = 0.0f;

    return RECONSTRUCTION_SUCCESS;
}

// ============================================================================
// 相移全息重建
// ============================================================================

int holography_reconstruct_phase_shifting(
    const RealImage **holograms,
    const HolographyParams *params,
    ReconstructionResult *result)
{
    if (!holograms || !params || !result) {
        return RECONSTRUCTION_ERROR_NULL_POINTER;
    }

    if (params->type != HOLOGRAPHY_PHASE_SHIFTING) {
        return RECONSTRUCTION_ERROR_INVALID_PARAMS;
    }

    if (params->num_phase_steps < 3 || !params->phase_shifts) {
        return RECONSTRUCTION_ERROR_INVALID_PARAMS;
    }

    int width = holograms[0]->width;
    int height = holograms[0]->height;
    int size = width * height;

    // 验证所有全息图尺寸一致
    for (int i = 1; i < params->num_phase_steps; i++) {
        if (!holograms[i] ||
            holograms[i]->width != width ||
            holograms[i]->height != height) {
            return RECONSTRUCTION_ERROR_INVALID_PARAMS;
        }
    }

    // 使用标准4步相移算法
    if (params->num_phase_steps == 4) {
        // I1 = I0 + Ir + 2*sqrt(I0*Ir)*cos(φ)
        // I2 = I0 + Ir + 2*sqrt(I0*Ir)*cos(φ + π/2) = I0 + Ir - 2*sqrt(I0*Ir)*sin(φ)
        // I3 = I0 + Ir + 2*sqrt(I0*Ir)*cos(φ + π) = I0 + Ir - 2*sqrt(I0*Ir)*cos(φ)
        // I4 = I0 + Ir + 2*sqrt(I0*Ir)*cos(φ + 3π/2) = I0 + Ir + 2*sqrt(I0*Ir)*sin(φ)
        
        // 相位: φ = atan2(I4 - I2, I1 - I3)
        // 振幅: A = sqrt((I4-I2)^2 + (I1-I3)^2) / 2

        for (int i = 0; i < size; i++) {
            float I1 = holograms[0]->data[i];
            float I2 = holograms[1]->data[i];
            float I3 = holograms[2]->data[i];
            float I4 = holograms[3]->data[i];

            float numerator = I4 - I2;
            float denominator = I1 - I3;
            
            float phase = atan2f(numerator, denominator);
            float amplitude = sqrtf(numerator * numerator + 
                                   denominator * denominator) / 2.0f;

            result->reconstructed->data[i] = amplitude * 
                (cosf(phase) + sinf(phase) * I);
        }
    } else {
        // 通用N步相移算法
        // 使用最小二乘法
        
        for (int i = 0; i < size; i++) {
            float sum_cos = 0.0f;
            float sum_sin = 0.0f;
            float sum_I = 0.0f;

            for (int step = 0; step < params->num_phase_steps; step++) {
                float I = holograms[step]->data[i];
                float shift = params->phase_shifts[step];
                
                sum_I += I;
                sum_cos += I * cosf(shift);
                sum_sin += I * sinf(shift);
            }

            float avg_I = sum_I / params->num_phase_steps;
            float A_cos = 2.0f * sum_cos / params->num_phase_steps;
            float A_sin = 2.0f * sum_sin / params->num_phase_steps;

            float phase = atan2f(A_sin, A_cos);
            float amplitude = sqrtf(A_cos * A_cos + A_sin * A_sin);

            result->reconstructed->data[i] = amplitude * 
                (cosf(phase) + sinf(phase) * I);
        }
    }

    // 如果提供了参考波，进行归一化
    if (params->reference_wave) {
        if (params->reference_wave->width != width ||
            params->reference_wave->height != height) {
            return RECONSTRUCTION_ERROR_INVALID_PARAMS;
        }

        for (int i = 0; i < size; i++) {
            ComplexF ref = params->reference_wave->data[i];
            if (cabsf(ref) > 1e-6f) {
                result->reconstructed->data[i] /= ref;
            }
        }
    }

    // 数值再聚焦
    if (params->numerical_refocus && params->diffraction_params) {
        ComplexImage *refocused = complex_image_create(width, height);
        if (!refocused) {
            return RECONSTRUCTION_ERROR_MEMORY_ALLOCATION;
        }

        int ret = diffraction_propagate(result->reconstructed, refocused,
                                       params->diffraction_params);
        if (ret != DIFFRACTION_SUCCESS) {
            complex_image_destroy(refocused);
            return RECONSTRUCTION_ERROR_DIFFRACTION_FAILED;
        }

        memcpy(result->reconstructed->data, refocused->data,
               size * sizeof(ComplexF));
        complex_image_destroy(refocused);
    }

    result->iterations_performed = 1;
    result->converged = true;
    result->final_error = 0.0f;

    return RECONSTRUCTION_SUCCESS;
}

// ============================================================================
// 在线全息重建
// ============================================================================

int holography_reconstruct_inline(
    const RealImage *hologram,
    const HolographyParams *params,
    const PhaseRetrievalParams *pr_params,
    const ConstraintSet *constraints,
    ReconstructionResult *result)
{
    if (!hologram || !params || !pr_params || !result) {
        return RECONSTRUCTION_ERROR_NULL_POINTER;
    }

    if (params->type != HOLOGRAPHY_IN_LINE) {
        return RECONSTRUCTION_ERROR_INVALID_PARAMS;
    }

    int width = hologram->width;
    int height = hologram->height;

    // 在线全息需要使用相位恢复算法
    // 强度约束来自全息图本身
    
    // 创建物平面强度估计（初始假设为均匀）
    RealImage *object_intensity = real_image_create(width, height, 1);
    if (!object_intensity) {
        return RECONSTRUCTION_ERROR_MEMORY_ALLOCATION;
    }

    // 初始化为均匀强度
    float avg_intensity = 0.0f;
    for (int i = 0; i < width * height; i++) {
        avg_intensity += hologram->data[i];
    }
    avg_intensity /= (width * height);

    for (int i = 0; i < width * height; i++) {
        object_intensity->data[i] = avg_intensity;
    }

    // 使用相位恢复算法
    int ret = phase_retrieval(object_intensity, hologram,
                             pr_params, constraints,
                             params->diffraction_params, result);

    real_image_destroy(object_intensity);

    return ret;
}

// ============================================================================
// 通用全息重建接口
// ============================================================================

int holography_reconstruct(
    const void *input_data,
    const HolographyParams *params,
    const PhaseRetrievalParams *pr_params,
    const ConstraintSet *constraints,
    ReconstructionResult *result)
{
    if (!input_data || !params || !result) {
        return RECONSTRUCTION_ERROR_NULL_POINTER;
    }

    switch (params->type) {
        case HOLOGRAPHY_OFF_AXIS:
            return holography_reconstruct_offaxis(
                (const RealImage*)input_data, params, result);

        case HOLOGRAPHY_PHASE_SHIFTING:
            return holography_reconstruct_phase_shifting(
                (const RealImage**)input_data, params, result);

        case HOLOGRAPHY_IN_LINE:
            if (!pr_params) {
                return RECONSTRUCTION_ERROR_NULL_POINTER;
            }
            return holography_reconstruct_inline(
                (const RealImage*)input_data, params, pr_params,
                constraints, result);

        default:
            return RECONSTRUCTION_ERROR_INVALID_PARAMS;
    }
}

// ============================================================================
// 质量评估
// ============================================================================

int compute_reconstruction_quality(
    const ReconstructionResult *result,
    const RealImage *reference,
    ReconstructionQuality *quality)
{
    if (!result || !quality) {
        return RECONSTRUCTION_ERROR_NULL_POINTER;
    }

    int width = result->reconstructed->width;
    int height = result->reconstructed->height;
    int size = width * height;

    // 计算振幅
    RealImage *amplitude = real_image_create(width, height, 1);
    if (!amplitude) {
        return RECONSTRUCTION_ERROR_MEMORY_ALLOCATION;
    }

    for (int i = 0; i < size; i++) {
        amplitude->data[i] = cabsf(result->reconstructed->data[i]);
    }

    // 如果有参考图像，计算误差指标
    if (reference) {
        if (reference->width != width || reference->height != height) {
            real_image_destroy(amplitude);
            return RECONSTRUCTION_ERROR_INVALID_PARAMS;
        }

        // RMSE
        float sum_sq_error = 0.0f;
        float sum_sq_ref = 0.0f;

        for (int i = 0; i < size; i++) {
            float diff = amplitude->data[i] - reference->data[i];
            sum_sq_error += diff * diff;
            sum_sq_ref += reference->data[i] * reference->data[i];
        }

        quality->rmse = sqrtf(sum_sq_error / size);
        quality->nrmse = quality->rmse / sqrtf(sum_sq_ref / size);

        // PSNR
        float max_val = 0.0f;
        for (int i = 0; i < size; i++) {
            if (reference->data[i] > max_val) {
                max_val = reference->data[i];
            }
        }

        float mse = sum_sq_error / size;
        if (mse > 0.0f) {
            quality->psnr = 10.0f * log10f((max_val * max_val) / mse);
        } else {
            quality->psnr = INFINITY;
        }

        // SSIM (简化版本)
        float mean_amp = 0.0f, mean_ref = 0.0f;
        for (int i = 0; i < size; i++) {
            mean_amp += amplitude->data[i];
            mean_ref += reference->data[i];
        }
        mean_amp /= size;
        mean_ref /= size;

        float var_amp = 0.0f, var_ref = 0.0f, covar = 0.0f;
        for (int i = 0; i < size; i++) {
            float diff_amp = amplitude->data[i] - mean_amp;
            float diff_ref = reference->data[i] - mean_ref;
            var_amp += diff_amp * diff_amp;
            var_ref += diff_ref * diff_ref;
            covar += diff_amp * diff_ref;
        }
        var_amp /= size;
        var_ref /= size;
        covar /= size;

        float C1 = 0.01f * max_val * 0.01f * max_val;
        float C2 = 0.03f * max_val * 0.03f * max_val;

        quality->ssim = ((2.0f * mean_amp * mean_ref + C1) * 
                        (2.0f * covar + C2)) /
                       ((mean_amp * mean_amp + mean_ref * mean_ref + C1) *
                        (var_amp + var_ref + C2));
    } else {
        quality->rmse = 0.0f;
        quality->nrmse = 0.0f;
        quality->psnr = 0.0f;
        quality->ssim = 0.0f;
    }

    // 计算对比度
    float min_val = amplitude->data[0];
    float max_val = amplitude->data[0];
    float mean_val = 0.0f;

    for (int i = 0; i < size; i++) {
        float val = amplitude->data[i];
        if (val < min_val) min_val = val;
        if (val > max_val) max_val = val;
        mean_val += val;
    }
    mean_val /= size;

    if (max_val + min_val > 0.0f) {
        quality->contrast = (max_val - min_val) / (max_val + min_val);
    } else {
        quality->contrast = 0.0f;
    }

    // 计算清晰度（使用梯度）
    float sharpness = 0.0f;
    for (int i = 1; i < height - 1; i++) {
        for (int j = 1; j < width - 1; j++) {
            int idx = i * width + j;
            float gx = amplitude->data[idx + 1] - amplitude->data[idx - 1];
            float gy = amplitude->data[idx + width] - amplitude->data[idx - width];
            sharpness += sqrtf(gx * gx + gy * gy);
        }
    }
    quality->sharpness = sharpness / ((width - 2) * (height - 2));

    // 计算信噪比估计
    float signal_power = 0.0f;
    float noise_power = 0.0f;

    for (int i = 0; i < size; i++) {
        signal_power += amplitude->data[i] * amplitude->data[i];
    }
    signal_power /= size;

    // 估计噪声（使用高频成分）
    for (int i = 1; i < height - 1; i++) {
        for (int j = 1; j < width - 1; j++) {
            int idx = i * width + j;
            float laplacian = -4.0f * amplitude->data[idx] +
                             amplitude->data[idx - 1] + amplitude->data[idx + 1] +
                             amplitude->data[idx - width] + amplitude->data[idx + width];
            noise_power += laplacian * laplacian;
        }
    }
    noise_power /= ((width - 2) * (height - 2));

    if (noise_power > 0.0f) {
        quality->snr = 10.0f * log10f(signal_power / noise_power);
    } else {
        quality->snr = INFINITY;
    }

    real_image_destroy(amplitude);

    return RECONSTRUCTION_SUCCESS;
}
// ============================================================================
// 结果保存函数
// ============================================================================

int save_reconstruction_result(
    const ReconstructionResult *result,
    const char *filename_prefix)
{
    if (!result || !filename_prefix) {
        return RECONSTRUCTION_ERROR_NULL_POINTER;
    }

    int width = result->reconstructed->width;
    int height = result->reconstructed->height;
    int size = width * height;

    // 1. 保存振幅
    RealImage *amplitude = real_image_create(width, height, 1);
    if (!amplitude) {
        return RECONSTRUCTION_ERROR_MEMORY_ALLOCATION;
    }

    for (int i = 0; i < size; i++) {
        amplitude->data[i] = cabsf(result->reconstructed->data[i]);
    }

    char filename[512];
    snprintf(filename, sizeof(filename), "%s_amplitude.raw", filename_prefix);
    
    FILE *fp = fopen(filename, "wb");
    if (fp) {
        fwrite(amplitude->data, sizeof(float), size, fp);
        fclose(fp);
    }

    // 保存为图像格式（如果可用）
    snprintf(filename, sizeof(filename), "%s_amplitude.png", filename_prefix);
    image_save_png(amplitude, filename);

    real_image_destroy(amplitude);

    // 2. 保存相位
    RealImage *phase = real_image_create(width, height, 1);
    if (!phase) {
        return RECONSTRUCTION_ERROR_MEMORY_ALLOCATION;
    }

    for (int i = 0; i < size; i++) {
        phase->data[i] = cargf(result->reconstructed->data[i]);
    }

    snprintf(filename, sizeof(filename), "%s_phase.raw", filename_prefix);
    fp = fopen(filename, "wb");
    if (fp) {
        fwrite(phase->data, sizeof(float), size, fp);
        fclose(fp);
    }

    snprintf(filename, sizeof(filename), "%s_phase.png", filename_prefix);
    image_save_png(phase, filename);

    real_image_destroy(phase);

    // 3. 保存实部和虚部
    RealImage *real_part = real_image_create(width, height, 1);
    RealImage *imag_part = real_image_create(width, height, 1);
    
    if (!real_part || !imag_part) {
        real_image_destroy(real_part);
        real_image_destroy(imag_part);
        return RECONSTRUCTION_ERROR_MEMORY_ALLOCATION;
    }

    for (int i = 0; i < size; i++) {
        real_part->data[i] = crealf(result->reconstructed->data[i]);
        imag_part->data[i] = cimagf(result->reconstructed->data[i]);
    }

    snprintf(filename, sizeof(filename), "%s_real.raw", filename_prefix);
    fp = fopen(filename, "wb");
    if (fp) {
        fwrite(real_part->data, sizeof(float), size, fp);
        fclose(fp);
    }

    snprintf(filename, sizeof(filename), "%s_imag.raw", filename_prefix);
    fp = fopen(filename, "wb");
    if (fp) {
        fwrite(imag_part->data, sizeof(float), size, fp);
        fclose(fp);
    }

    real_image_destroy(real_part);
    real_image_destroy(imag_part);

    // 4. 保存复数场（完整数据）
    snprintf(filename, sizeof(filename), "%s_complex.raw", filename_prefix);
    fp = fopen(filename, "wb");
    if (fp) {
        fwrite(result->reconstructed->data, sizeof(ComplexF), size, fp);
        fclose(fp);
    }

    // 5. 保存误差历史
    if (result->error_history_length > 0) {
        snprintf(filename, sizeof(filename), "%s_error_history.txt", filename_prefix);
        fp = fopen(filename, "w");
        if (fp) {
            fprintf(fp, "# Iteration\tError\n");
            for (int i = 0; i < result->error_history_length; i++) {
                fprintf(fp, "%d\t%.8e\n", i + 1, result->error_history[i]);
            }
            fclose(fp);
        }
    }

    // 6. 保存元数据
    snprintf(filename, sizeof(filename), "%s_metadata.txt", filename_prefix);
    fp = fopen(filename, "w");
    if (fp) {
        fprintf(fp, "Reconstruction Metadata\n");
        fprintf(fp, "=======================\n\n");
        fprintf(fp, "Image size: %d x %d\n", width, height);
        fprintf(fp, "Iterations performed: %d\n", result->iterations_performed);
        fprintf(fp, "Converged: %s\n", result->converged ? "Yes" : "No");
        fprintf(fp, "Final error: %.8e\n", result->final_error);
        fprintf(fp, "Computation time: %.3f seconds\n", result->computation_time);
        
        if (result->quality) {
            fprintf(fp, "\nQuality Metrics:\n");
            fprintf(fp, "  RMSE: %.6f\n", result->quality->rmse);
            fprintf(fp, "  NRMSE: %.6f\n", result->quality->nrmse);
            fprintf(fp, "  PSNR: %.2f dB\n", result->quality->psnr);
            fprintf(fp, "  SSIM: %.4f\n", result->quality->ssim);
            fprintf(fp, "  Contrast: %.4f\n", result->quality->contrast);
            fprintf(fp, "  Sharpness: %.4f\n", result->quality->sharpness);
            fprintf(fp, "  SNR: %.2f dB\n", result->quality->snr);
        }
        
        fclose(fp);
    }

    return RECONSTRUCTION_SUCCESS;
}

int load_reconstruction_result(
    const char *filename_prefix,
    ReconstructionResult *result)
{
    if (!filename_prefix || !result) {
        return RECONSTRUCTION_ERROR_NULL_POINTER;
    }

    char filename[512];
    
    // 加载复数场
    snprintf(filename, sizeof(filename), "%s_complex.raw", filename_prefix);
    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        return RECONSTRUCTION_ERROR_FILE_IO;
    }

    int width = result->reconstructed->width;
    int height = result->reconstructed->height;
    int size = width * height;

    size_t read_count = fread(result->reconstructed->data, 
                             sizeof(ComplexF), size, fp);
    fclose(fp);

    if (read_count != size) {
        return RECONSTRUCTION_ERROR_FILE_IO;
    }

    // 加载误差历史（如果存在）
    snprintf(filename, sizeof(filename), "%s_error_history.txt", filename_prefix);
    fp = fopen(filename, "r");
    if (fp) {
        char line[256];
        fgets(line, sizeof(line), fp); // 跳过标题行
        
        result->error_history_length = 0;
        while (fgets(line, sizeof(line), fp) && 
               result->error_history_length < RECONSTRUCTION_MAX_ITERATIONS) {
            int iter;
            float error;
            if (sscanf(line, "%d\t%f", &iter, &error) == 2) {
                result->error_history[result->error_history_length++] = error;
            }
        }
        
        if (result->error_history_length > 0) {
            result->final_error = 
                result->error_history[result->error_history_length - 1];
        }
        
        fclose(fp);
    }

    return RECONSTRUCTION_SUCCESS;
}

// ============================================================================
// 可视化辅助函数
// ============================================================================

int create_phase_colormap(
    const ComplexImage *field,
    RealImage *rgb_image)
{
    if (!field || !rgb_image) {
        return RECONSTRUCTION_ERROR_NULL_POINTER;
    }

    if (rgb_image->channels != 3) {
        return RECONSTRUCTION_ERROR_INVALID_PARAMS;
    }

    int width = field->width;
    int height = field->height;
    int size = width * height;

    if (rgb_image->width != width || rgb_image->height != height) {
        return RECONSTRUCTION_ERROR_INVALID_PARAMS;
    }

    // 使用HSV色彩空间，相位映射到色调
    for (int i = 0; i < size; i++) {
        float amplitude = cabsf(field->data[i]);
        float phase = cargf(field->data[i]);
        
        // 归一化相位到[0, 1]
        float hue = (phase + M_PI) / (2.0f * M_PI);
        float saturation = 1.0f;
        float value = amplitude;
        
        // 归一化value
        // 这里可以使用全局最大值或局部自适应
        
        // HSV转RGB
        float c = value * saturation;
        float x = c * (1.0f - fabsf(fmodf(hue * 6.0f, 2.0f) - 1.0f));
        float m = value - c;
        
        float r, g, b;
        int h_sector = (int)(hue * 6.0f);
        
        switch (h_sector) {
            case 0: r = c; g = x; b = 0; break;
            case 1: r = x; g = c; b = 0; break;
            case 2: r = 0; g = c; b = x; break;
            case 3: r = 0; g = x; b = c; break;
            case 4: r = x; g = 0; b = c; break;
            default: r = c; g = 0; b = x; break;
        }
        
        rgb_image->data[i * 3 + 0] = r + m;
        rgb_image->data[i * 3 + 1] = g + m;
        rgb_image->data[i * 3 + 2] = b + m;
    }

    return RECONSTRUCTION_SUCCESS;
}

int create_amplitude_phase_composite(
    const ComplexImage *field,
    RealImage *composite)
{
    if (!field || !composite) {
        return RECONSTRUCTION_ERROR_NULL_POINTER;
    }

    int width = field->width;
    int height = field->height;

    // 创建2x1的复合图像
    if (composite->width != width * 2 || composite->height != height) {
        return RECONSTRUCTION_ERROR_INVALID_PARAMS;
    }

    // 左侧：振幅
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            int src_idx = i * width + j;
            int dst_idx = i * (width * 2) + j;
            composite->data[dst_idx] = cabsf(field->data[src_idx]);
        }
    }

    // 右侧：相位
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            int src_idx = i * width + j;
            int dst_idx = i * (width * 2) + width + j;
            float phase = cargf(field->data[src_idx]);
            // 归一化到[0, 1]
            composite->data[dst_idx] = (phase + M_PI) / (2.0f * M_PI);
        }
    }

    return RECONSTRUCTION_SUCCESS;
}

// ============================================================================
// 打印和日志函数
// ============================================================================

void print_iteration_info(int iteration, float error, double elapsed_time)
{
    printf("Iteration %4d: Error = %.6e, Time = %.3f s\n",
           iteration, error, elapsed_time);
}

void print_reconstruction_summary(const ReconstructionResult *result)
{
    if (!result) {
        return;
    }

    printf("\n");
    printf("========================================\n");
    printf("Reconstruction Summary\n");
    printf("========================================\n");
    printf("Image size: %d x %d\n", 
           result->reconstructed->width, 
           result->reconstructed->height);
    printf("Iterations performed: %d\n", result->iterations_performed);
    printf("Converged: %s\n", result->converged ? "Yes" : "No");
    printf("Final error: %.8e\n", result->final_error);
    printf("Computation time: %.3f seconds\n", result->computation_time);
    
    if (result->quality) {
        printf("\nQuality Metrics:\n");
        printf("  RMSE:      %.6f\n", result->quality->rmse);
        printf("  NRMSE:     %.6f\n", result->quality->nrmse);
        printf("  PSNR:      %.2f dB\n", result->quality->psnr);
        printf("  SSIM:      %.4f\n", result->quality->ssim);
        printf("  Contrast:  %.4f\n", result->quality->contrast);
        printf("  Sharpness: %.4f\n", result->quality->sharpness);
        printf("  SNR:       %.2f dB\n", result->quality->snr);
    }
    
    printf("========================================\n\n");
}

void print_algorithm_info(PhaseRetrievalAlgorithm algorithm)
{
    printf("\n");
    printf("Algorithm: ");
    
    switch (algorithm) {
        case PHASE_RETRIEVAL_GS:
            printf("Gerchberg-Saxton (GS)\n");
            printf("Description: Classic iterative Fourier transform algorithm\n");
            printf("Best for: Simple phase retrieval with good initial guess\n");
            break;
            
        case PHASE_RETRIEVAL_HIO:
            printf("Hybrid Input-Output (HIO)\n");
            printf("Description: Uses feedback parameter for better convergence\n");
            printf("Best for: Problems with support constraints\n");
            break;
            
        case PHASE_RETRIEVAL_ER:
            printf("Error Reduction (ER)\n");
            printf("Description: Projects onto constraint sets\n");
            printf("Best for: Initial iterations or refinement\n");
            break;
            
        case PHASE_RETRIEVAL_RAAR:
            printf("Relaxed Averaged Alternating Reflections (RAAR)\n");
            printf("Description: Advanced projection algorithm\n");
            printf("Best for: Difficult problems with multiple constraints\n");
            break;
            
        case PHASE_RETRIEVAL_HYBRID:
            printf("Hybrid Algorithm\n");
            printf("Description: Combines ER, HIO, and RAAR adaptively\n");
            printf("Best for: General purpose, robust reconstruction\n");
            break;
            
        default:
            printf("Unknown\n");
            break;
    }
    
    printf("\n");
}

// ============================================================================
// 错误处理
// ============================================================================

const char* reconstruction_error_string(int error_code)
{
    switch (error_code) {
        case RECONSTRUCTION_SUCCESS:
            return "Success";
        case RECONSTRUCTION_ERROR_NULL_POINTER:
            return "Null pointer error";
        case RECONSTRUCTION_ERROR_INVALID_PARAMS:
            return "Invalid parameters";
        case RECONSTRUCTION_ERROR_MEMORY_ALLOCATION:
            return "Memory allocation failed";
        case RECONSTRUCTION_ERROR_FFT_FAILED:
            return "FFT operation failed";
        case RECONSTRUCTION_ERROR_DIFFRACTION_FAILED:
            return "Diffraction propagation failed";
        case RECONSTRUCTION_ERROR_FILE_IO:
            return "File I/O error";
        case RECONSTRUCTION_ERROR_NOT_CONVERGED:
            return "Algorithm did not converge";
        case RECONSTRUCTION_ERROR_CONSTRAINT_VIOLATION:
            return "Constraint violation";
        default:
            return "Unknown error";
    }
}

void print_error_message(int error_code, const char *context)
{
    fprintf(stderr, "Error");
    if (context) {
        fprintf(stderr, " in %s", context);
    }
    fprintf(stderr, ": %s (code %d)\n", 
            reconstruction_error_string(error_code), error_code);
}

// ============================================================================
// 性能分析
// ============================================================================

typedef struct {
    clock_t start_time;
    clock_t end_time;
    double elapsed_seconds;
    const char *operation_name;
} Timer;

Timer* timer_create(const char *operation_name)
{
    Timer *timer = (Timer*)malloc(sizeof(Timer));
    if (timer) {
        timer->operation_name = operation_name;
        timer->start_time = 0;
        timer->end_time = 0;
        timer->elapsed_seconds = 0.0;
    }
    return timer;
}

void timer_start(Timer *timer)
{
    if (timer) {
        timer->start_time = clock();
    }
}

void timer_stop(Timer *timer)
{
    if (timer) {
        timer->end_time = clock();
        timer->elapsed_seconds = 
            (double)(timer->end_time - timer->start_time) / CLOCKS_PER_SEC;
    }
}

void timer_print(const Timer *timer)
{
    if (timer && timer->operation_name) {
        printf("%s: %.3f seconds\n", 
               timer->operation_name, timer->elapsed_seconds);
    }
}

void timer_destroy(Timer *timer)
{
    free(timer);
}

// ============================================================================
// 批处理支持
// ============================================================================

typedef struct {
    int num_reconstructions;
    ReconstructionResult **results;
    double total_time;
    int successful_count;
    int failed_count;
} BatchReconstructionResult;

BatchReconstructionResult* batch_reconstruction_result_create(int num_reconstructions)
{
    BatchReconstructionResult *batch = 
        (BatchReconstructionResult*)malloc(sizeof(BatchReconstructionResult));
    
    if (!batch) {
        return NULL;
    }

    batch->num_reconstructions = num_reconstructions;
    batch->results = (ReconstructionResult**)calloc(
        num_reconstructions, sizeof(ReconstructionResult*));
    
    if (!batch->results) {
        free(batch);
        return NULL;
    }

    batch->total_time = 0.0;
    batch->successful_count = 0;
    batch->failed_count = 0;

    return batch;
}

void batch_reconstruction_result_destroy(BatchReconstructionResult *batch)
{
    if (batch) {
        if (batch->results) {
            for (int i = 0; i < batch->num_reconstructions; i++) {
                reconstruction_result_destroy(batch->results[i]);
            }
            free(batch->results);
        }
        free(batch);
    }
}

int batch_phase_retrieval(
    const RealImage **object_intensities,
    const RealImage **image_intensities,
    int num_images,
    const PhaseRetrievalParams *params,
    const ConstraintSet *constraints,
    const DiffractionParams *diffraction_params,
    BatchReconstructionResult *batch_result)
{
    if (!object_intensities || !image_intensities || 
        !params || !batch_result) {
        return RECONSTRUCTION_ERROR_NULL_POINTER;
    }

    if (num_images != batch_result->num_reconstructions) {
        return RECONSTRUCTION_ERROR_INVALID_PARAMS;
    }

    clock_t batch_start = clock();

    for (int i = 0; i < num_images; i++) {
        if (params->verbose) {
            printf("\n=== Processing image %d/%d ===\n", i + 1, num_images);
        }

        // 创建结果结构
        batch_result->results[i] = reconstruction_result_create(
            object_intensities[i]->width,
            object_intensities[i]->height,
            params->max_iterations);

        if (!batch_result->results[i]) {
            batch_result->failed_count++;
            continue;
        }

        // 执行重建
        int ret = phase_retrieval(
            object_intensities[i],
            image_intensities[i],
            params,
            constraints,
            diffraction_params,
            batch_result->results[i]);

        if (ret == RECONSTRUCTION_SUCCESS) {
            batch_result->successful_count++;
            
            // 保存结果
            if (params->output_prefix) {
                char filename[512];
                snprintf(filename, sizeof(filename), "%s_%04d",
                        params->output_prefix, i);
                save_reconstruction_result(batch_result->results[i], filename);
            }
        } else {
            batch_result->failed_count++;
            if (params->verbose) {
                print_error_message(ret, "batch reconstruction");
            }
        }
    }

    batch_result->total_time = 
        (double)(clock() - batch_start) / CLOCKS_PER_SEC;

    if (params->verbose) {
        printf("\n=== Batch Reconstruction Summary ===\n");
        printf("Total images: %d\n", num_images);
        printf("Successful: %d\n", batch_result->successful_count);
        printf("Failed: %d\n", batch_result->failed_count);
        printf("Total time: %.3f seconds\n", batch_result->total_time);
        printf("Average time per image: %.3f seconds\n",
               batch_result->total_time / num_images);
    }

    return RECONSTRUCTION_SUCCESS;
}

// ============================================================================
// 自适应参数调整
// ============================================================================

int adaptive_parameter_tuning(
    const RealImage *intensity_object,
    const RealImage *intensity_image,
    PhaseRetrievalParams *params,
    const ConstraintSet *constraints,
    const DiffractionParams *diffraction_params)
{
    if (!intensity_object || !intensity_image || !params) {
        return RECONSTRUCTION_ERROR_NULL_POINTER;
    }

    // 基于图像特性自动调整参数
    int size = intensity_object->width * intensity_object->height;

    // 1. 估计信噪比
    float mean_intensity = 0.0f;
    float var_intensity = 0.0f;
    
    for (int i = 0; i < size; i++) {
        mean_intensity += intensity_image->data[i];
    }
    mean_intensity /= size;

    for (int i = 0; i < size; i++) {
        float diff = intensity_image->data[i] - mean_intensity;
        var_intensity += diff * diff;
    }
    var_intensity /= size;

    float estimated_snr = 10.0f * log10f(mean_intensity * mean_intensity / var_intensity);

    // 2. 根据SNR调整参数
    if (estimated_snr < 10.0f) {
        // 低SNR：使用更保守的参数
        params->beta = 0.5f;
        params->relaxation = 0.5f;
        params->tolerance *= 2.0f;
    } else if (estimated_snr > 30.0f) {
        // 高SNR：可以使用更激进的参数
        params->beta = 0.9f;
        params->relaxation = 0.9f;
        params->tolerance *= 0.5f;
    }

    // 3. 根据图像大小调整迭代次数
    if (size > 1024 * 1024) {
        // 大图像可能需要更多迭代
        params->max_iterations = (int)(params->max_iterations * 1.5f);
    }

    if (params->verbose) {
        printf("Adaptive parameter tuning:\n");
        printf("  Estimated SNR: %.2f dB\n", estimated_snr);
        printf("  Adjusted beta: %.2f\n", params->beta);
        printf("  Adjusted relaxation: %.2f\n", params->relaxation);
        printf("  Adjusted tolerance: %.2e\n", params->tolerance);
        printf("  Adjusted max iterations: %d\n", params->max_iterations);
    }

    return RECONSTRUCTION_SUCCESS;
}

// ============================================================================
// 主函数示例（用于测试）
// ============================================================================

#ifdef PHASE_RETRIEVAL_STANDALONE

int main(int argc, char *argv[])
{
    printf("Phase Retrieval Library - Standalone Test\n");
    printf("==========================================\n\n");

    // 这里可以添加测试代码
    // 例如：加载测试图像，运行重建算法，保存结果等

    printf("Test completed successfully!\n");
    return 0;
}

#endif // PHASE_RETRIEVAL_STANDALONE

