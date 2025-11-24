/**
 * @file reconstruction.c
 * @brief 全息重建和相位恢复算法实现
 * @author Diffraction Simulation Library Team
 * @version 1.0.0
 */

#include "reconstruction.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdio.h>

// ============================================================================
// 内部辅助函数声明
// ============================================================================

static float compute_rms_error(const ComplexImage *a, const ComplexImage *b);
static void apply_gaussian_blur(float *data, int width, int height, float sigma);
static float rand_uniform(void);
static void complex_multiply_inplace(ComplexImage *a, const ComplexImage *b);
static void complex_add_scaled(ComplexImage *dest, const ComplexImage *src, float scale);

// ============================================================================
// 参数创建和销毁函数
// ============================================================================

PhaseRetrievalParams* phase_retrieval_params_create_default(void)
{
    PhaseRetrievalParams *params = (PhaseRetrievalParams*)malloc(sizeof(PhaseRetrievalParams));
    if (!params) {
        return NULL;
    }

    // 设置默认值
    params->algorithm = PHASE_RETRIEVAL_HIO;
    params->max_iterations = 1000;
    params->tolerance = 1e-6f;
    params->beta = 0.7f;
    params->relaxation = 0.9f;
    
    params->use_shrinkwrap = false;
    params->shrinkwrap_interval = 20;
    params->shrinkwrap_sigma = 3.0f;
    params->shrinkwrap_threshold = 0.2f;
    
    params->use_averaging = false;
    params->averaging_window = 10;
    
    params->verbose = false;
    params->print_interval = 100;

    return params;
}

void phase_retrieval_params_destroy(PhaseRetrievalParams *params)
{
    if (params) {
        free(params);
    }
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
    
    params->distances = (double*)calloc(num_planes, sizeof(double));
    params->measurements = (ComplexImage**)calloc(num_planes, sizeof(ComplexImage*));
    params->plane_weights = (float*)malloc(num_planes * sizeof(float));
    
    if (!params->distances || !params->measurements || !params->plane_weights) {
        multiplane_params_destroy(params);
        return NULL;
    }

    // 默认权重相等
    for (int i = 0; i < num_planes; i++) {
        params->plane_weights[i] = 1.0f / num_planes;
    }

    params->diffraction_params = NULL;

    return params;
}

void multiplane_params_destroy(MultiPlaneParams *params)
{
    if (params) {
        if (params->distances) {
            free(params->distances);
        }
        if (params->measurements) {
            // 注意：不释放measurements中的图像，因为它们可能是外部管理的
            free(params->measurements);
        }
        if (params->plane_weights) {
            free(params->plane_weights);
        }
        free(params);
    }
}

HolographyParams* holography_params_create(HolographyType type)
{
    HolographyParams *params = (HolographyParams*)malloc(sizeof(HolographyParams));
    if (!params) {
        return NULL;
    }

    memset(params, 0, sizeof(HolographyParams));
    params->type = type;

    // 根据类型设置默认值
    switch (type) {
        case HOLOGRAPHY_OFF_AXIS:
            params->carrier_frequency_x = 0.0f;
            params->carrier_frequency_y = 0.0f;
            params->filter_size = 0.3f;
            break;
            
        case HOLOGRAPHY_PHASE_SHIFTING:
            params->num_phase_steps = 4;
            params->phase_shifts = (float*)malloc(4 * sizeof(float));
            if (params->phase_shifts) {
                params->phase_shifts[0] = 0.0f;
                params->phase_shifts[1] = M_PI / 2.0f;
                params->phase_shifts[2] = M_PI;
                params->phase_shifts[3] = 3.0f * M_PI / 2.0f;
            }
            break;
            
        default:
            break;
    }

    params->reference_wave = NULL;
    params->remove_twin_image = true;
    params->numerical_refocus = false;
    params->diffraction_params = NULL;

    return params;
}

void holography_params_destroy(HolographyParams *params)
{
    if (params) {
        if (params->phase_shifts) {
            free(params->phase_shifts);
        }
        // reference_wave和diffraction_params由外部管理
        free(params);
    }
}

// ============================================================================
// 约束集合管理
// ============================================================================

ConstraintSet* constraint_set_create(void)
{
    ConstraintSet *constraints = (ConstraintSet*)malloc(sizeof(ConstraintSet));
    if (!constraints) {
        return NULL;
    }

    memset(constraints, 0, sizeof(ConstraintSet));
    constraints->constraint_flags = CONSTRAINT_NONE;
    constraints->positivity_weight = 1.0f;
    constraints->tv_weight = 0.0f;

    return constraints;
}

void constraint_set_destroy(ConstraintSet *constraints)
{
    if (constraints) {
        if (constraints->support) {
            support_constraint_destroy(constraints->support);
        }
        if (constraints->amplitude) {
            amplitude_constraint_destroy(constraints->amplitude);
        }
        if (constraints->phase) {
            phase_constraint_destroy(constraints->phase);
        }
        free(constraints);
    }
}

// ============================================================================
// 支撑域约束
// ============================================================================

SupportConstraint* support_constraint_create(int width, int height)
{
    if (width <= 0 || height <= 0) {
        return NULL;
    }

    SupportConstraint *support = (SupportConstraint*)malloc(sizeof(SupportConstraint));
    if (!support) {
        return NULL;
    }

    support->width = width;
    support->height = height;
    support->mask = (bool*)calloc(width * height, sizeof(bool));
    
    if (!support->mask) {
        free(support);
        return NULL;
    }

    support->auto_update = false;
    support->threshold = 0.2f;

    return support;
}

SupportConstraint* support_constraint_from_image(
    const RealImage *image,
    float threshold)
{
    if (!image || threshold < 0.0f || threshold > 1.0f) {
        return NULL;
    }

    SupportConstraint *support = support_constraint_create(image->width, image->height);
    if (!support) {
        return NULL;
    }

    // 计算图像的最大值用于归一化
    float max_val = 0.0f;
    for (int i = 0; i < image->height; i++) {
        for (int j = 0; j < image->width; j++) {
            int idx = i * image->width + j;
            float val;
            
            if (image->channels == 1) {
                val = image->data[idx];
            } else {
                // 多通道取平均
                val = 0.0f;
                for (int c = 0; c < image->channels; c++) {
                    val += image->data[i * image->stride + j * image->channels + c];
                }
                val /= image->channels;
            }
            
            if (val > max_val) {
                max_val = val;
            }
        }
    }

    // 根据阈值创建掩模
    float threshold_val = threshold * max_val;
    for (int i = 0; i < image->height; i++) {
        for (int j = 0; j < image->width; j++) {
            int idx = i * image->width + j;
            float val;
            
            if (image->channels == 1) {
                val = image->data[idx];
            } else {
                val = 0.0f;
                for (int c = 0; c < image->channels; c++) {
                    val += image->data[i * image->stride + j * image->channels + c];
                }
                val /= image->channels;
            }
            
            support->mask[idx] = (val >= threshold_val);
        }
    }

    support->threshold = threshold;

    return support;
}

void support_constraint_destroy(SupportConstraint *support)
{
    if (support) {
        if (support->mask) {
            free(support->mask);
        }
        free(support);
    }
}

// ============================================================================
// 振幅约束
// ============================================================================

AmplitudeConstraint* amplitude_constraint_create(int width, int height)
{
    if (width <= 0 || height <= 0) {
        return NULL;
    }

    AmplitudeConstraint *amplitude = (AmplitudeConstraint*)malloc(sizeof(AmplitudeConstraint));
    if (!amplitude) {
        return NULL;
    }

    int size = width * height;
    amplitude->amplitude = (float*)calloc(size, sizeof(float));
    amplitude->weight = (float*)malloc(size * sizeof(float));
    
    if (!amplitude->amplitude || !amplitude->weight) {
        amplitude_constraint_destroy(amplitude);
        return NULL;
    }

    // 默认权重为1
    for (int i = 0; i < size; i++) {
        amplitude->weight[i] = 1.0f;
    }

    amplitude->enforce_exact = true;

    return amplitude;
}

void amplitude_constraint_destroy(AmplitudeConstraint *amplitude)
{
    if (amplitude) {
        if (amplitude->amplitude) {
            free(amplitude->amplitude);
        }
        if (amplitude->weight) {
            free(amplitude->weight);
        }
        free(amplitude);
    }
}

// ============================================================================
// 相位约束
// ============================================================================

PhaseConstraint* phase_constraint_create(int width, int height)
{
    if (width <= 0 || height <= 0) {
        return NULL;
    }

    PhaseConstraint *phase = (PhaseConstraint*)malloc(sizeof(PhaseConstraint));
    if (!phase) {
        return NULL;
    }

    int size = width * height;
    phase->phase = (float*)calloc(size, sizeof(float));
    phase->weight = (float*)malloc(size * sizeof(float));
    
    if (!phase->phase || !phase->weight) {
        phase_constraint_destroy(phase);
        return NULL;
    }

    // 默认权重为1
    for (int i = 0; i < size; i++) {
        phase->weight[i] = 1.0f;
    }

    phase->wrap_phase = true;

    return phase;
}

void phase_constraint_destroy(PhaseConstraint *phase)
{
    if (phase) {
        if (phase->phase) {
            free(phase->phase);
        }
        if (phase->weight) {
            free(phase->weight);
        }
        free(phase);
    }
}

// ============================================================================
// 约束应用函数
// ============================================================================

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
        
        SupportConstraint *support = constraints->support;
        
        if (support->width != field->width || support->height != field->height) {
            return RECONSTRUCTION_ERROR_INVALID_CONSTRAINT;
        }

        for (int i = 0; i < size; i++) {
            if (!support->mask[i]) {
                field->data[i] = 0.0f + 0.0f * I;
            }
        }
    }

    // 应用正值约束（仅在物平面）
    if (in_object_plane && 
        (constraints->constraint_flags & CONSTRAINT_POSITIVITY)) {
        
        float weight = constraints->positivity_weight;
        
        for (int i = 0; i < size; i++) {
            float real_part = crealf(field->data[i]);
            float imag_part = cimagf(field->data[i]);
            
            // 如果实部为负，将其设为0或减小
            if (real_part < 0.0f) {
                real_part *= (1.0f - weight);
            }
            
            field->data[i] = real_part + imag_part * I;
        }
    }

    // 应用振幅约束
    if ((constraints->constraint_flags & CONSTRAINT_AMPLITUDE) &&
        constraints->amplitude) {
        
        AmplitudeConstraint *amp_constraint = constraints->amplitude;
        
        for (int i = 0; i < size; i++) {
            float current_amp = cabsf(field->data[i]);
            float current_phase = cargf(field->data[i]);
            float target_amp = amp_constraint->amplitude[i];
            float weight = amp_constraint->weight[i];
            
            float new_amp;
            if (amp_constraint->enforce_exact) {
                new_amp = target_amp;
            } else {
                new_amp = current_amp * (1.0f - weight) + target_amp * weight;
            }
            
            field->data[i] = new_amp * (cosf(current_phase) + sinf(current_phase) * I);
        }
    }

    // 应用相位约束
    if ((constraints->constraint_flags & CONSTRAINT_PHASE) &&
        constraints->phase) {
        
        PhaseConstraint *phase_constraint = constraints->phase;
        
        for (int i = 0; i < size; i++) {
            float current_amp = cabsf(field->data[i]);
            float current_phase = cargf(field->data[i]);
            float target_phase = phase_constraint->phase[i];
            float weight = phase_constraint->weight[i];
            
            float new_phase = current_phase * (1.0f - weight) + target_phase * weight;
            
            if (phase_constraint->wrap_phase) {
                // 相位包裹到[-π, π]
                while (new_phase > M_PI) new_phase -= 2.0f * M_PI;
                while (new_phase < -M_PI) new_phase += 2.0f * M_PI;
            }
            
            field->data[i] = current_amp * (cosf(new_phase) + sinf(new_phase) * I);
        }
    }

    // 应用全变分约束（简化版本）
    if ((constraints->constraint_flags & CONSTRAINT_TOTAL_VARIATION) &&
        constraints->tv_weight > 0.0f) {
        
        float tv_weight = constraints->tv_weight;
        ComplexImage *smoothed = complex_image_create(field->width, field->height);
        
        if (!smoothed) {
            return RECONSTRUCTION_ERROR_MEMORY_ALLOCATION;
        }

        // 简单的平滑滤波
        for (int i = 1; i < field->height - 1; i++) {
            for (int j = 1; j < field->width - 1; j++) {
                int idx = i * field->width + j;
                
                ComplexF sum = 0.0f + 0.0f * I;
                sum += field->data[(i-1) * field->width + j];
                sum += field->data[(i+1) * field->width + j];
                sum += field->data[i * field->width + (j-1)];
                sum += field->data[i * field->width + (j+1)];
                sum += field->data[idx] * 4.0f;
                
                smoothed->data[idx] = sum / 8.0f;
            }
        }

        // 混合原始和平滑版本
        for (int i = 0; i < size; i++) {
            field->data[i] = field->data[i] * (1.0f - tv_weight) + 
                           smoothed->data[i] * tv_weight;
        }

        complex_image_destroy(smoothed);
    }

    return RECONSTRUCTION_SUCCESS;
}

// ============================================================================
// Shrinkwrap支撑域更新
// ============================================================================

int update_support_shrinkwrap(
    const ComplexImage *field,
    SupportConstraint *support,
    float sigma,
    float threshold)
{
    if (!field || !support) {
        return RECONSTRUCTION_ERROR_NULL_POINTER;
    }

    if (field->width != support->width || field->height != support->height) {
        return RECONSTRUCTION_ERROR_INVALID_PARAMS;
    }

    int size = field->width * field->height;

    // 计算振幅
    float *amplitude = (float*)malloc(size * sizeof(float));
    if (!amplitude) {
        return RECONSTRUCTION_ERROR_MEMORY_ALLOCATION;
    }

    float max_amp = 0.0f;
    for (int i = 0; i < size; i++) {
        amplitude[i] = cabsf(field->data[i]);
        if (amplitude[i] > max_amp) {
            max_amp = amplitude[i];
        }
    }

    // 应用高斯模糊
    if (sigma > 0.0f) {
        apply_gaussian_blur(amplitude, field->width, field->height, sigma);
    }

    // 更新支撑域掩模
    float threshold_val = threshold * max_amp;
    for (int i = 0; i < size; i++) {
        support->mask[i] = (amplitude[i] >= threshold_val);
    }

    free(amplitude);

    return RECONSTRUCTION_SUCCESS;
}

// ============================================================================
// 重建结果管理
// ============================================================================

ReconstructionResult* reconstruction_result_create(
    int width,
    int height,
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
    result->error_history = (float*)malloc(max_iterations * sizeof(float));
    
    if (!result->reconstructed || !result->error_history) {
        reconstruction_result_destroy(result);
        return NULL;
    }

    result->iterations_performed = 0;
    result->final_error = 0.0f;
    result->error_history_length = 0;
    result->converged = false;
    result->computation_time = 0.0;

    return result;
}

void reconstruction_result_destroy(ReconstructionResult *result)
{
    if (result) {
        if (result->reconstructed) {
            complex_image_destroy(result->reconstructed);
        }
        if (result->error_history) {
            free(result->error_history);
        }
        free(result);
    }
}

// ============================================================================
// 初始化函数
// ============================================================================

int initialize_random_phase(
    ComplexImage *field,
    const RealImage *amplitude)
{
    if (!field) {
        return RECONSTRUCTION_ERROR_NULL_POINTER;
    }

    // 初始化随机数生成器
    srand((unsigned int)time(NULL));

    for (int i = 0; i < field->height; i++) {
        for (int j = 0; j < field->width; j++) {
            int idx = i * field->width + j;
            
            float amp = 1.0f;
            if (amplitude) {
                if (amplitude->channels == 1) {
                    amp = amplitude->data[idx];
                } else {
                    amp = amplitude->data[i * amplitude->stride + j * amplitude->channels];
                }
            }
            
            // 随机相位 [0, 2π]
            float phase = rand_uniform() * 2.0f * M_PI;
            
            field->data[idx] = amp * (cosf(phase) + sinf(phase) * I);
        }
    }

    return RECONSTRUCTION_SUCCESS;
}

// ============================================================================
// 内部辅助函数实现
// ============================================================================

static float rand_uniform(void)
{
    return (float)rand() / (float)RAND_MAX;
}

static float compute_rms_error(const ComplexImage *a, const ComplexImage *b)
{
    if (!a || !b || a->width != b->width || a->height != b->height) {
        return -1.0f;
    }

    int size = a->width * a->height;
    float sum = 0.0f;

    for (int i = 0; i < size; i++) {
        ComplexF diff = a->data[i] - b->data[i];
        float real_part = crealf(diff);
        float imag_part = cimagf(diff);
        sum += real_part * real_part + imag_part * imag_part;
    }

    return sqrtf(sum / size);
}

static void apply_gaussian_blur(float *data, int width, int height, float sigma)
{
    if (!data || sigma <= 0.0f) {
        return;
    }

    // 创建高斯核
    int kernel_size = (int)(6.0f * sigma + 1.0f);
    if (kernel_size % 2 == 0) kernel_size++;
    int half_size = kernel_size / 2;

    float *kernel = (float*)malloc(kernel_size * sizeof(float));
    if (!kernel) {
        return;
    }

    // 计算1D高斯核
    float sum = 0.0f;
    for (int i = 0; i < kernel_size; i++) {
        int x = i - half_size;
        kernel[i] = expf(-(x * x) / (2.0f * sigma * sigma));
        sum += kernel[i];
    }

    // 归一化
    for (int i = 0; i < kernel_size; i++) {
        kernel[i] /= sum;
    }

    // 临时缓冲区
    float *temp = (float*)malloc(width * height * sizeof(float));
    if (!temp) {
        free(kernel);
        return;
    }

    // 水平方向模糊
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            float val = 0.0f;
            for (int k = 0; k < kernel_size; k++) {
                int jj = j + k - half_size;
                if (jj >= 0 && jj < width) {
                    val += data[i * width + jj] * kernel[k];
                }
            }
            temp[i * width + j] = val;
        }
    }

    // 垂直方向模糊
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            float val = 0.0f;
            for (int k = 0; k < kernel_size; k++) {
                int ii = i + k - half_size;
                if (ii >= 0 && ii < height) {
                    val += temp[ii * width + j] * kernel[k];
                }
            }
            data[i * width + j] = val;
        }
    }

    free(kernel);
    free(temp);
}
// ============================================================================
// 误差计算函数
// ============================================================================

int compute_reconstruction_error(
    const ComplexImage *measured,
    const ComplexImage *reconstructed,
    float *error)
{
    if (!measured || !reconstructed || !error) {
        return RECONSTRUCTION_ERROR_NULL_POINTER;
    }

    if (measured->width != reconstructed->width || 
        measured->height != reconstructed->height) {
        return RECONSTRUCTION_ERROR_INVALID_PARAMS;
    }

    *error = compute_rms_error(measured, reconstructed);
    
    if (*error < 0.0f) {
        return RECONSTRUCTION_ERROR_INVALID_PARAMS;
    }

    return RECONSTRUCTION_SUCCESS;
}

int compute_intensity_error(
    const RealImage *measured_intensity,
    const ComplexImage *reconstructed,
    float *error)
{
    if (!measured_intensity || !reconstructed || !error) {
        return RECONSTRUCTION_ERROR_NULL_POINTER;
    }

    if (measured_intensity->width != reconstructed->width || 
        measured_intensity->height != reconstructed->height) {
        return RECONSTRUCTION_ERROR_INVALID_PARAMS;
    }

    int size = reconstructed->width * reconstructed->height;
    float sum = 0.0f;
    float measured_sum = 0.0f;

    for (int i = 0; i < reconstructed->height; i++) {
        for (int j = 0; j < reconstructed->width; j++) {
            int idx = i * reconstructed->width + j;
            
            float measured_val;
            if (measured_intensity->channels == 1) {
                measured_val = measured_intensity->data[idx];
            } else {
                measured_val = measured_intensity->data[i * measured_intensity->stride + 
                                                       j * measured_intensity->channels];
            }
            
            float reconstructed_intensity = cabsf(reconstructed->data[idx]);
            reconstructed_intensity *= reconstructed_intensity;
            
            float diff = measured_val - reconstructed_intensity;
            sum += diff * diff;
            measured_sum += measured_val * measured_val;
        }
    }

    // 归一化误差
    if (measured_sum > 0.0f) {
        *error = sqrtf(sum / measured_sum);
    } else {
        *error = sqrtf(sum / size);
    }

    return RECONSTRUCTION_SUCCESS;
}

// ============================================================================
// Gerchberg-Saxton (GS) 算法
// ============================================================================

int phase_retrieval_gs(
    const RealImage *intensity_object,
    const RealImage *intensity_image,
    const PhaseRetrievalParams *params,
    const ConstraintSet *constraints,
    const DiffractionParams *diffraction_params,
    ReconstructionResult *result)
{
    if (!intensity_object || !intensity_image || !params || 
        !diffraction_params || !result) {
        return RECONSTRUCTION_ERROR_NULL_POINTER;
    }

    if (intensity_object->width != intensity_image->width ||
        intensity_object->height != intensity_image->height) {
        return RECONSTRUCTION_ERROR_INVALID_PARAMS;
    }

    clock_t start_time = clock();

    int width = intensity_object->width;
    int height = intensity_object->height;

    // 创建工作图像
    ComplexImage *object_field = complex_image_create(width, height);
    ComplexImage *image_field = complex_image_create(width, height);
    ComplexImage *prev_object = complex_image_create(width, height);
    
    if (!object_field || !image_field || !prev_object) {
        complex_image_destroy(object_field);
        complex_image_destroy(image_field);
        complex_image_destroy(prev_object);
        return RECONSTRUCTION_ERROR_MEMORY_ALLOCATION;
    }

    // 初始化：随机相位
    int ret = initialize_random_phase(object_field, intensity_object);
    if (ret != RECONSTRUCTION_SUCCESS) {
        complex_image_destroy(object_field);
        complex_image_destroy(image_field);
        complex_image_destroy(prev_object);
        return ret;
    }

    // 迭代
    bool converged = false;
    int iter;
    
    for (iter = 0; iter < params->max_iterations; iter++) {
        // 保存当前物场
        memcpy(prev_object->data, object_field->data, 
               width * height * sizeof(ComplexF));

        // 1. 应用物平面约束
        if (constraints) {
            ret = apply_constraints(object_field, constraints, true);
            if (ret != RECONSTRUCTION_SUCCESS) {
                break;
            }
        }

        // 2. 传播到像平面
        ret = diffraction_propagate(object_field, image_field, diffraction_params);
        if (ret != DIFFRACTION_SUCCESS) {
            ret = RECONSTRUCTION_ERROR_DIFFRACTION_FAILED;
            break;
        }

        // 3. 应用像平面振幅约束
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                int idx = i * width + j;
                
                float measured_intensity;
                if (intensity_image->channels == 1) {
                    measured_intensity = intensity_image->data[idx];
                } else {
                    measured_intensity = intensity_image->data[i * intensity_image->stride + 
                                                              j * intensity_image->channels];
                }
                
                float measured_amplitude = sqrtf(measured_intensity);
                float current_phase = cargf(image_field->data[idx]);
                
                image_field->data[idx] = measured_amplitude * 
                    (cosf(current_phase) + sinf(current_phase) * I);
            }
        }

        // 4. 反向传播到物平面
        DiffractionParams backward_params = *diffraction_params;
        backward_params.direction = DIRECTION_BACKWARD;
        
        ret = diffraction_propagate(image_field, object_field, &backward_params);
        if (ret != DIFFRACTION_SUCCESS) {
            ret = RECONSTRUCTION_ERROR_DIFFRACTION_FAILED;
            break;
        }

        // 5. 应用物平面振幅约束
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                int idx = i * width + j;
                
                float measured_intensity;
                if (intensity_object->channels == 1) {
                    measured_intensity = intensity_object->data[idx];
                } else {
                    measured_intensity = intensity_object->data[i * intensity_object->stride + 
                                                               j * intensity_object->channels];
                }
                
                float measured_amplitude = sqrtf(measured_intensity);
                float current_phase = cargf(object_field->data[idx]);
                
                object_field->data[idx] = measured_amplitude * 
                    (cosf(current_phase) + sinf(current_phase) * I);
            }
        }

        // 6. 计算误差
        float error = compute_rms_error(object_field, prev_object);
        result->error_history[iter] = error;
        result->error_history_length = iter + 1;

        // 7. 检查收敛
        if (error < params->tolerance) {
            converged = true;
            if (params->verbose) {
                printf("GS converged at iteration %d, error = %e\n", iter + 1, error);
            }
            break;
        }

        // 8. Shrinkwrap更新
        if (params->use_shrinkwrap && constraints && constraints->support &&
            (iter + 1) % params->shrinkwrap_interval == 0) {
            update_support_shrinkwrap(object_field, constraints->support,
                                     params->shrinkwrap_sigma,
                                     params->shrinkwrap_threshold);
        }

        // 9. 打印进度
        if (params->verbose && (iter + 1) % params->print_interval == 0) {
            printf("GS iteration %d/%d, error = %e\n", 
                   iter + 1, params->max_iterations, error);
        }
    }

    // 保存结果
    memcpy(result->reconstructed->data, object_field->data,
           width * height * sizeof(ComplexF));
    result->iterations_performed = iter + 1;
    result->final_error = result->error_history[iter];
    result->converged = converged;
    result->computation_time = (double)(clock() - start_time) / CLOCKS_PER_SEC;

    // 清理
    complex_image_destroy(object_field);
    complex_image_destroy(image_field);
    complex_image_destroy(prev_object);

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
    if (!intensity_object || !intensity_image || !params || 
        !diffraction_params || !result) {
        return RECONSTRUCTION_ERROR_NULL_POINTER;
    }

    if (intensity_object->width != intensity_image->width ||
        intensity_object->height != intensity_image->height) {
        return RECONSTRUCTION_ERROR_INVALID_PARAMS;
    }

    clock_t start_time = clock();

    int width = intensity_object->width;
    int height = intensity_object->height;
    float beta = params->beta;

    // 创建工作图像
    ComplexImage *object_field = complex_image_create(width, height);
    ComplexImage *image_field = complex_image_create(width, height);
    ComplexImage *prev_object = complex_image_create(width, height);
    ComplexImage *gs_result = complex_image_create(width, height);
    
    if (!object_field || !image_field || !prev_object || !gs_result) {
        complex_image_destroy(object_field);
        complex_image_destroy(image_field);
        complex_image_destroy(prev_object);
        complex_image_destroy(gs_result);
        return RECONSTRUCTION_ERROR_MEMORY_ALLOCATION;
    }

    // 初始化：随机相位
    int ret = initialize_random_phase(object_field, intensity_object);
    if (ret != RECONSTRUCTION_SUCCESS) {
        complex_image_destroy(object_field);
        complex_image_destroy(image_field);
        complex_image_destroy(prev_object);
        complex_image_destroy(gs_result);
        return ret;
    }

    // 迭代
    bool converged = false;
    int iter;
    
    for (iter = 0; iter < params->max_iterations; iter++) {
        // 保存当前物场
        memcpy(prev_object->data, object_field->data, 
               width * height * sizeof(ComplexF));

        // 1. 传播到像平面
        ret = diffraction_propagate(object_field, image_field, diffraction_params);
        if (ret != DIFFRACTION_SUCCESS) {
            ret = RECONSTRUCTION_ERROR_DIFFRACTION_FAILED;
            break;
        }

        // 2. 应用像平面振幅约束
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                int idx = i * width + j;
                
                float measured_intensity;
                if (intensity_image->channels == 1) {
                    measured_intensity = intensity_image->data[idx];
                } else {
                    measured_intensity = intensity_image->data[i * intensity_image->stride + 
                                                              j * intensity_image->channels];
                }
                
                float measured_amplitude = sqrtf(measured_intensity);
                float current_phase = cargf(image_field->data[idx]);
                
                image_field->data[idx] = measured_amplitude * 
                    (cosf(current_phase) + sinf(current_phase) * I);
            }
        }

        // 3. 反向传播到物平面
        DiffractionParams backward_params = *diffraction_params;
        backward_params.direction = DIRECTION_BACKWARD;
        
        ret = diffraction_propagate(image_field, gs_result, &backward_params);
        if (ret != DIFFRACTION_SUCCESS) {
            ret = RECONSTRUCTION_ERROR_DIFFRACTION_FAILED;
            break;
        }

        // 4. HIO更新
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                int idx = i * width + j;
                
                // 检查是否在支撑域内
                bool in_support = true;
                if (constraints && (constraints->constraint_flags & CONSTRAINT_SUPPORT) &&
                    constraints->support) {
                    in_support = constraints->support->mask[idx];
                }

                // 检查正值约束
                bool satisfies_positivity = true;
                if (constraints && (constraints->constraint_flags & CONSTRAINT_POSITIVITY)) {
                    satisfies_positivity = (crealf(gs_result->data[idx]) >= 0.0f);
                }

                if (in_support && satisfies_positivity) {
                    // 在约束域内：使用GS结果
                    object_field->data[idx] = gs_result->data[idx];
                } else {
                    // 在约束域外：HIO更新
                    object_field->data[idx] = prev_object->data[idx] - 
                                             beta * gs_result->data[idx];
                }
            }
        }

        // 5. 应用物平面振幅约束（如果有）
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                int idx = i * width + j;
                
                float measured_intensity;
                if (intensity_object->channels == 1) {
                    measured_intensity = intensity_object->data[idx];
                } else {
                    measured_intensity = intensity_object->data[i * intensity_object->stride + 
                                                               j * intensity_object->channels];
                }
                
                if (measured_intensity > 0.0f) {
                    float measured_amplitude = sqrtf(measured_intensity);
                    float current_phase = cargf(object_field->data[idx]);
                    
                    object_field->data[idx] = measured_amplitude * 
                        (cosf(current_phase) + sinf(current_phase) * I);
                }
            }
        }

        // 6. 计算误差
        float error = compute_rms_error(object_field, prev_object);
        result->error_history[iter] = error;
        result->error_history_length = iter + 1;

        // 7. 检查收敛
        if (error < params->tolerance) {
            converged = true;
            if (params->verbose) {
                printf("HIO converged at iteration %d, error = %e\n", iter + 1, error);
            }
            break;
        }

        // 8. Shrinkwrap更新
        if (params->use_shrinkwrap && constraints && constraints->support &&
            (iter + 1) % params->shrinkwrap_interval == 0) {
            update_support_shrinkwrap(object_field, constraints->support,
                                     params->shrinkwrap_sigma,
                                     params->shrinkwrap_threshold);
        }

        // 9. 打印进度
        if (params->verbose && (iter + 1) % params->print_interval == 0) {
            printf("HIO iteration %d/%d, error = %e\n", 
                   iter + 1, params->max_iterations, error);
        }
    }

    // 保存结果
    memcpy(result->reconstructed->data, object_field->data,
           width * height * sizeof(ComplexF));
    result->iterations_performed = iter + 1;
    result->final_error = result->error_history[iter];
    result->converged = converged;
    result->computation_time = (double)(clock() - start_time) / CLOCKS_PER_SEC;

    // 清理
    complex_image_destroy(object_field);
    complex_image_destroy(image_field);
    complex_image_destroy(prev_object);
    complex_image_destroy(gs_result);

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
    if (!intensity_object || !intensity_image || !params || 
        !diffraction_params || !result) {
        return RECONSTRUCTION_ERROR_NULL_POINTER;
    }

    if (intensity_object->width != intensity_image->width ||
        intensity_object->height != intensity_image->height) {
        return RECONSTRUCTION_ERROR_INVALID_PARAMS;
    }

    clock_t start_time = clock();

    int width = intensity_object->width;
    int height = intensity_object->height;

    // 创建工作图像
    ComplexImage *object_field = complex_image_create(width, height);
    ComplexImage *image_field = complex_image_create(width, height);
    ComplexImage *prev_object = complex_image_create(width, height);
    
    if (!object_field || !image_field || !prev_object) {
        complex_image_destroy(object_field);
        complex_image_destroy(image_field);
        complex_image_destroy(prev_object);
        return RECONSTRUCTION_ERROR_MEMORY_ALLOCATION;
    }

    // 初始化：随机相位
    int ret = initialize_random_phase(object_field, intensity_object);
    if (ret != RECONSTRUCTION_SUCCESS) {
        complex_image_destroy(object_field);
        complex_image_destroy(image_field);
        complex_image_destroy(prev_object);
        return ret;
    }

    // 迭代
    bool converged = false;
    int iter;
    
    for (iter = 0; iter < params->max_iterations; iter++) {
        // 保存当前物场
        memcpy(prev_object->data, object_field->data, 
               width * height * sizeof(ComplexF));

        // 1. 传播到像平面
        ret = diffraction_propagate(object_field, image_field, diffraction_params);
        if (ret != DIFFRACTION_SUCCESS) {
            ret = RECONSTRUCTION_ERROR_DIFFRACTION_FAILED;
            break;
        }

        // 2. 应用像平面振幅约束
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                int idx = i * width + j;
                
                float measured_intensity;
                if (intensity_image->channels == 1) {
                    measured_intensity = intensity_image->data[idx];
                } else {
                    measured_intensity = intensity_image->data[i * intensity_image->stride + 
                                                              j * intensity_image->channels];
                }
                
                float measured_amplitude = sqrtf(measured_intensity);
                float current_phase = cargf(image_field->data[idx]);
                
                image_field->data[idx] = measured_amplitude * 
                    (cosf(current_phase) + sinf(current_phase) * I);
            }
        }

        // 3. 反向传播到物平面
        DiffractionParams backward_params = *diffraction_params;
        backward_params.direction = DIRECTION_BACKWARD;
        
        ret = diffraction_propagate(image_field, object_field, &backward_params);
        if (ret != DIFFRACTION_SUCCESS) {
            ret = RECONSTRUCTION_ERROR_DIFFRACTION_FAILED;
            break;
        }

        // 4. ER更新：在约束域外设为0
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                int idx = i * width + j;
                
                // 检查是否在支撑域内
                bool in_support = true;
                if (constraints && (constraints->constraint_flags & CONSTRAINT_SUPPORT) &&
                    constraints->support) {
                    in_support = constraints->support->mask[idx];
                }

                // 检查正值约束
                bool satisfies_positivity = true;
                if (constraints && (constraints->constraint_flags & CONSTRAINT_POSITIVITY)) {
                    satisfies_positivity = (crealf(object_field->data[idx]) >= 0.0f);
                }

                if (!in_support || !satisfies_positivity) {
                    // 在约束域外：设为0
                    object_field->data[idx] = 0.0f + 0.0f * I;
                }
            }
        }

        // 5. 应用物平面振幅约束（如果有）
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                int idx = i * width + j;
                
                float measured_intensity;
                if (intensity_object->channels == 1) {
                    measured_intensity = intensity_object->data[idx];
                } else {
                    measured_intensity = intensity_object->data[i * intensity_object->stride + 
                                                               j * intensity_object->channels];
                }
                
                if (measured_intensity > 0.0f) {
                    float measured_amplitude = sqrtf(measured_intensity);
                    float current_phase = cargf(object_field->data[idx]);
                    
                    object_field->data[idx] = measured_amplitude * 
                        (cosf(current_phase) + sinf(current_phase) * I);
                }
            }
        }

        // 6. 计算误差
        float error = compute_rms_error(object_field, prev_object);
        result->error_history[iter] = error;
        result->error_history_length = iter + 1;

        // 7. 检查收敛
        if (error < params->tolerance) {
            converged = true;
            if (params->verbose) {
                printf("ER converged at iteration %d, error = %e\n", iter + 1, error);
            }
            break;
        }

        // 8. Shrinkwrap更新
        if (params->use_shrinkwrap && constraints && constraints->support &&
            (iter + 1) % params->shrinkwrap_interval == 0) {
            update_support_shrinkwrap(object_field, constraints->support,
                                     params->shrinkwrap_sigma,
                                     params->shrinkwrap_threshold);
        }

        // 9. 打印进度
        if (params->verbose && (iter + 1) % params->print_interval == 0) {
            printf("ER iteration %d/%d, error = %e\n", 
                   iter + 1, params->max_iterations, error);
        }
    }

    // 保存结果
    memcpy(result->reconstructed->data, object_field->data,
           width * height * sizeof(ComplexF));
    result->iterations_performed = iter + 1;
    result->final_error = result->error_history[iter];
    result->converged = converged;
    result->computation_time = (double)(clock() - start_time) / CLOCKS_PER_SEC;

    // 清理
    complex_image_destroy(object_field);
    complex_image_destroy(image_field);
    complex_image_destroy(prev_object);

    return RECONSTRUCTION_SUCCESS;
}
// ============================================================================
// Relaxed Averaged Alternating Reflections (RAAR) 算法
// ============================================================================

int phase_retrieval_raar(
    const RealImage *intensity_object,
    const RealImage *intensity_image,
    const PhaseRetrievalParams *params,
    const ConstraintSet *constraints,
    const DiffractionParams *diffraction_params,
    ReconstructionResult *result)
{
    if (!intensity_object || !intensity_image || !params || 
        !diffraction_params || !result) {
        return RECONSTRUCTION_ERROR_NULL_POINTER;
    }

    if (intensity_object->width != intensity_image->width ||
        intensity_object->height != intensity_image->height) {
        return RECONSTRUCTION_ERROR_INVALID_PARAMS;
    }

    clock_t start_time = clock();

    int width = intensity_object->width;
    int height = intensity_object->height;
    float beta = params->relaxation;  // RAAR松弛参数

    // 创建工作图像
    ComplexImage *object_field = complex_image_create(width, height);
    ComplexImage *image_field = complex_image_create(width, height);
    ComplexImage *prev_object = complex_image_create(width, height);
    ComplexImage *ps = complex_image_create(width, height);  // 支撑域投影
    ComplexImage *pm = complex_image_create(width, height);  // 测量域投影
    
    if (!object_field || !image_field || !prev_object || !ps || !pm) {
        complex_image_destroy(object_field);
        complex_image_destroy(image_field);
        complex_image_destroy(prev_object);
        complex_image_destroy(ps);
        complex_image_destroy(pm);
        return RECONSTRUCTION_ERROR_MEMORY_ALLOCATION;
    }

    // 初始化：随机相位
    int ret = initialize_random_phase(object_field, intensity_object);
    if (ret != RECONSTRUCTION_SUCCESS) {
        complex_image_destroy(object_field);
        complex_image_destroy(image_field);
        complex_image_destroy(prev_object);
        complex_image_destroy(ps);
        complex_image_destroy(pm);
        return ret;
    }

    // 迭代
    bool converged = false;
    int iter;
    
    for (iter = 0; iter < params->max_iterations; iter++) {
        // 保存当前物场
        memcpy(prev_object->data, object_field->data, 
               width * height * sizeof(ComplexF));

        // 1. 传播到像平面
        ret = diffraction_propagate(object_field, image_field, diffraction_params);
        if (ret != DIFFRACTION_SUCCESS) {
            ret = RECONSTRUCTION_ERROR_DIFFRACTION_FAILED;
            break;
        }

        // 2. 应用像平面振幅约束（测量域投影 Pm）
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                int idx = i * width + j;
                
                float measured_intensity;
                if (intensity_image->channels == 1) {
                    measured_intensity = intensity_image->data[idx];
                } else {
                    measured_intensity = intensity_image->data[i * intensity_image->stride + 
                                                              j * intensity_image->channels];
                }
                
                float measured_amplitude = sqrtf(measured_intensity);
                float current_phase = cargf(image_field->data[idx]);
                
                image_field->data[idx] = measured_amplitude * 
                    (cosf(current_phase) + sinf(current_phase) * I);
            }
        }

        // 3. 反向传播到物平面
        DiffractionParams backward_params = *diffraction_params;
        backward_params.direction = DIRECTION_BACKWARD;
        
        ret = diffraction_propagate(image_field, pm, &backward_params);
        if (ret != DIFFRACTION_SUCCESS) {
            ret = RECONSTRUCTION_ERROR_DIFFRACTION_FAILED;
            break;
        }

        // 4. 应用支撑域投影 Ps
        memcpy(ps->data, pm->data, width * height * sizeof(ComplexF));
        
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                int idx = i * width + j;
                
                // 检查是否在支撑域内
                bool in_support = true;
                if (constraints && (constraints->constraint_flags & CONSTRAINT_SUPPORT) &&
                    constraints->support) {
                    in_support = constraints->support->mask[idx];
                }

                // 检查正值约束
                bool satisfies_positivity = true;
                if (constraints && (constraints->constraint_flags & CONSTRAINT_POSITIVITY)) {
                    satisfies_positivity = (crealf(ps->data[idx]) >= 0.0f);
                }

                if (!in_support || !satisfies_positivity) {
                    ps->data[idx] = 0.0f + 0.0f * I;
                }
            }
        }

        // 5. RAAR更新：x_{n+1} = β*Ps*Pm*x_n + (1-2β)*Pm*x_n + β*x_n
        for (int i = 0; i < width * height; i++) {
            ComplexF ps_pm = ps->data[i];           // Ps*Pm*x_n
            ComplexF pm_val = pm->data[i];          // Pm*x_n
            ComplexF x_n = prev_object->data[i];    // x_n
            
            object_field->data[i] = beta * (2.0f * ps_pm - pm_val) + 
                                   (1.0f - 2.0f * beta) * pm_val;
        }

        // 6. 应用物平面振幅约束（如果有）
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                int idx = i * width + j;
                
                float measured_intensity;
                if (intensity_object->channels == 1) {
                    measured_intensity = intensity_object->data[idx];
                } else {
                    measured_intensity = intensity_object->data[i * intensity_object->stride + 
                                                               j * intensity_object->channels];
                }
                
                if (measured_intensity > 0.0f) {
                    float measured_amplitude = sqrtf(measured_intensity);
                    float current_phase = cargf(object_field->data[idx]);
                    
                    object_field->data[idx] = measured_amplitude * 
                        (cosf(current_phase) + sinf(current_phase) * I);
                }
            }
        }

        // 7. 计算误差
        float error = compute_rms_error(object_field, prev_object);
        result->error_history[iter] = error;
        result->error_history_length = iter + 1;

        // 8. 检查收敛
        if (error < params->tolerance) {
            converged = true;
            if (params->verbose) {
                printf("RAAR converged at iteration %d, error = %e\n", iter + 1, error);
            }
            break;
        }

        // 9. Shrinkwrap更新
        if (params->use_shrinkwrap && constraints && constraints->support &&
            (iter + 1) % params->shrinkwrap_interval == 0) {
            update_support_shrinkwrap(object_field, constraints->support,
                                     params->shrinkwrap_sigma,
                                     params->shrinkwrap_threshold);
        }

        // 10. 打印进度
        if (params->verbose && (iter + 1) % params->print_interval == 0) {
            printf("RAAR iteration %d/%d, error = %e\n", 
                   iter + 1, params->max_iterations, error);
        }
    }

    // 保存结果
    memcpy(result->reconstructed->data, object_field->data,
           width * height * sizeof(ComplexF));
    result->iterations_performed = iter + 1;
    result->final_error = result->error_history[iter];
    result->converged = converged;
    result->computation_time = (double)(clock() - start_time) / CLOCKS_PER_SEC;

    // 清理
    complex_image_destroy(object_field);
    complex_image_destroy(image_field);
    complex_image_destroy(prev_object);
    complex_image_destroy(ps);
    complex_image_destroy(pm);

    return RECONSTRUCTION_SUCCESS;
}

// ============================================================================
// 混合相位恢复算法（自适应选择）
// ============================================================================

int phase_retrieval_hybrid(
    const RealImage *intensity_object,
    const RealImage *intensity_image,
    const PhaseRetrievalParams *params,
    const ConstraintSet *constraints,
    const DiffractionParams *diffraction_params,
    ReconstructionResult *result)
{
    if (!intensity_object || !intensity_image || !params || 
        !diffraction_params || !result) {
        return RECONSTRUCTION_ERROR_NULL_POINTER;
    }

    if (params->verbose) {
        printf("Starting hybrid phase retrieval algorithm\n");
    }

    // 策略：
    // 1. 前20%迭代使用ER（快速收敛到大致解）
    // 2. 中间60%迭代使用HIO（精细化）
    // 3. 最后20%迭代使用ER（稳定收敛）

    int total_iterations = params->max_iterations;
    int er_phase1 = (int)(total_iterations * 0.2);
    int hio_phase = (int)(total_iterations * 0.6);
    int er_phase2 = total_iterations - er_phase1 - hio_phase;

    PhaseRetrievalParams phase_params = *params;
    int ret;

    // Phase 1: ER
    if (params->verbose) {
        printf("Phase 1: ER for %d iterations\n", er_phase1);
    }
    
    phase_params.max_iterations = er_phase1;
    phase_params.algorithm = PHASE_RETRIEVAL_ER;
    
    ret = phase_retrieval_er(intensity_object, intensity_image, 
                            &phase_params, constraints, 
                            diffraction_params, result);
    
    if (ret != RECONSTRUCTION_SUCCESS) {
        return ret;
    }

    // 检查是否已经收敛
    if (result->converged) {
        if (params->verbose) {
            printf("Converged in Phase 1\n");
        }
        return RECONSTRUCTION_SUCCESS;
    }

    // Phase 2: HIO
    if (params->verbose) {
        printf("Phase 2: HIO for %d iterations\n", hio_phase);
    }

    // 创建临时结果用于HIO
    ReconstructionResult *hio_result = reconstruction_result_create(
        intensity_object->width, intensity_object->height, hio_phase);
    
    if (!hio_result) {
        return RECONSTRUCTION_ERROR_MEMORY_ALLOCATION;
    }

    // 使用Phase 1的结果作为初始值
    memcpy(hio_result->reconstructed->data, result->reconstructed->data,
           intensity_object->width * intensity_object->height * sizeof(ComplexF));

    phase_params.max_iterations = hio_phase;
    phase_params.algorithm = PHASE_RETRIEVAL_HIO;
    
    // 临时修改：从已有结果继续
    ComplexImage *temp_init = complex_image_create(
        intensity_object->width, intensity_object->height);
    
    if (!temp_init) {
        reconstruction_result_destroy(hio_result);
        return RECONSTRUCTION_ERROR_MEMORY_ALLOCATION;
    }

    memcpy(temp_init->data, result->reconstructed->data,
           intensity_object->width * intensity_object->height * sizeof(ComplexF));

    ret = phase_retrieval_hio(intensity_object, intensity_image, 
                             &phase_params, constraints, 
                             diffraction_params, hio_result);

    complex_image_destroy(temp_init);

    if (ret != RECONSTRUCTION_SUCCESS) {
        reconstruction_result_destroy(hio_result);
        return ret;
    }

    // 合并误差历史
    int offset = result->error_history_length;
    for (int i = 0; i < hio_result->error_history_length; i++) {
        if (offset + i < total_iterations) {
            result->error_history[offset + i] = hio_result->error_history[i];
        }
    }
    result->error_history_length = offset + hio_result->error_history_length;
    result->iterations_performed += hio_result->iterations_performed;

    // 更新重建结果
    memcpy(result->reconstructed->data, hio_result->reconstructed->data,
           intensity_object->width * intensity_object->height * sizeof(ComplexF));

    // 检查是否已经收敛
    if (hio_result->converged) {
        if (params->verbose) {
            printf("Converged in Phase 2\n");
        }
        result->converged = true;
        result->final_error = hio_result->final_error;
        reconstruction_result_destroy(hio_result);
        return RECONSTRUCTION_SUCCESS;
    }

    reconstruction_result_destroy(hio_result);

    // Phase 3: ER (refinement)
    if (params->verbose) {
        printf("Phase 3: ER refinement for %d iterations\n", er_phase2);
    }

    ReconstructionResult *er_result = reconstruction_result_create(
        intensity_object->width, intensity_object->height, er_phase2);
    
    if (!er_result) {
        return RECONSTRUCTION_ERROR_MEMORY_ALLOCATION;
    }

    phase_params.max_iterations = er_phase2;
    phase_params.algorithm = PHASE_RETRIEVAL_ER;
    
    ret = phase_retrieval_er(intensity_object, intensity_image, 
                            &phase_params, constraints, 
                            diffraction_params, er_result);

    if (ret != RECONSTRUCTION_SUCCESS) {
        reconstruction_result_destroy(er_result);
        return ret;
    }

    // 合并最终结果
    offset = result->error_history_length;
    for (int i = 0; i < er_result->error_history_length; i++) {
        if (offset + i < total_iterations) {
            result->error_history[offset + i] = er_result->error_history[i];
        }
    }
    result->error_history_length = offset + er_result->error_history_length;
    result->iterations_performed += er_result->iterations_performed;

    memcpy(result->reconstructed->data, er_result->reconstructed->data,
           intensity_object->width * intensity_object->height * sizeof(ComplexF));

    result->converged = er_result->converged;
    result->final_error = er_result->final_error;

    reconstruction_result_destroy(er_result);

    if (params->verbose) {
        printf("Hybrid algorithm completed: %d total iterations, final error = %e\n",
               result->iterations_performed, result->final_error);
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

    // 验证参数
    int ret = reconstruction_validate_params(params);
    if (ret != RECONSTRUCTION_SUCCESS) {
        return ret;
    }

    // 根据算法类型调用相应函数
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

    if (!params->diffraction_params) {
        return RECONSTRUCTION_ERROR_NULL_POINTER;
    }

    clock_t start_time = clock();

    // 获取图像尺寸（从第一个测量平面）
    int width = params->measurements[0]->width;
    int height = params->measurements[0]->height;

    // 创建工作图像
    ComplexImage *object_field = complex_image_create(width, height);
    ComplexImage *prev_object = complex_image_create(width, height);
    ComplexImage **plane_fields = (ComplexImage**)malloc(
        params->num_planes * sizeof(ComplexImage*));
    
    if (!object_field || !prev_object || !plane_fields) {
        complex_image_destroy(object_field);
        complex_image_destroy(prev_object);
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
            complex_image_destroy(prev_object);
            return RECONSTRUCTION_ERROR_MEMORY_ALLOCATION;
        }
    }

    // 初始化：随机相位
    int ret = initialize_random_phase(object_field, NULL);
    if (ret != RECONSTRUCTION_SUCCESS) {
        goto cleanup;
    }

    // 迭代
    bool converged = false;
    int iter;
    PhaseRetrievalParams *pr_params = &params->base_params;
    
    for (iter = 0; iter < pr_params->max_iterations; iter++) {
        // 保存当前物场
        memcpy(prev_object->data, object_field->data, 
               width * height * sizeof(ComplexF));

        // 应用物平面约束
        if (constraints) {
            ret = apply_constraints(object_field, constraints, true);
            if (ret != RECONSTRUCTION_SUCCESS) {
                goto cleanup;
            }
        }

        // 对每个平面进行传播和约束
        ComplexImage *accumulated = complex_image_create(width, height);
        if (!accumulated) {
            ret = RECONSTRUCTION_ERROR_MEMORY_ALLOCATION;
            goto cleanup;
        }

        // 初始化累积场为0
        memset(accumulated->data, 0, width * height * sizeof(ComplexF));

        for (int p = 0; p < params->num_planes; p++) {
            // 创建该平面的衍射参数
            DiffractionParams plane_params = *params->diffraction_params;
            plane_params.propagation_distance = params->distances[p];

            // 前向传播到该平面
            ret = diffraction_propagate(object_field, plane_fields[p], &plane_params);
            if (ret != DIFFRACTION_SUCCESS) {
                complex_image_destroy(accumulated);
                ret = RECONSTRUCTION_ERROR_DIFFRACTION_FAILED;
                goto cleanup;
            }

            // 应用该平面的振幅约束
            for (int i = 0; i < width * height; i++) {
                float measured_amplitude = cabsf(params->measurements[p]->data[i]);
                float current_phase = cargf(plane_fields[p]->data[i]);
                
                plane_fields[p]->data[i] = measured_amplitude * 
                    (cosf(current_phase) + sinf(current_phase) * I);
            }

            // 反向传播到物平面
            plane_params.direction = DIRECTION_BACKWARD;
            ComplexImage *back_prop = complex_image_create(width, height);
            
            if (!back_prop) {
                complex_image_destroy(accumulated);
                ret = RECONSTRUCTION_ERROR_MEMORY_ALLOCATION;
                goto cleanup;
            }

            ret = diffraction_propagate(plane_fields[p], back_prop, &plane_params);
            if (ret != DIFFRACTION_SUCCESS) {
                complex_image_destroy(back_prop);
                complex_image_destroy(accumulated);
                ret = RECONSTRUCTION_ERROR_DIFFRACTION_FAILED;
                goto cleanup;
            }

            // 加权累积
            float weight = params->plane_weights[p];
            for (int i = 0; i < width * height; i++) {
                accumulated->data[i] += weight * back_prop->data[i];
            }

            complex_image_destroy(back_prop);
        }

        // 更新物场为加权平均
        memcpy(object_field->data, accumulated->data, 
               width * height * sizeof(ComplexF));
        complex_image_destroy(accumulated);

        // 计算误差
        float error = compute_rms_error(object_field, prev_object);
        result->error_history[iter] = error;
        result->error_history_length = iter + 1;

        // 检查收敛
        if (error < pr_params->tolerance) {
            converged = true;
            if (pr_params->verbose) {
                printf("Multiplane converged at iteration %d, error = %e\n", 
                       iter + 1, error);
            }
            break;
        }

        // Shrinkwrap更新
        if (pr_params->use_shrinkwrap && constraints && constraints->support &&
            (iter + 1) % pr_params->shrinkwrap_interval == 0) {
            update_support_shrinkwrap(object_field, constraints->support,
                                     pr_params->shrinkwrap_sigma,
                                     pr_params->shrinkwrap_threshold);
        }

        // 打印进度
        if (pr_params->verbose && (iter + 1) % pr_params->print_interval == 0) {
            printf("Multiplane iteration %d/%d, error = %e\n", 
                   iter + 1, pr_params->max_iterations, error);
        }
    }

    // 保存结果
    memcpy(result->reconstructed->data, object_field->data,
           width * height * sizeof(ComplexF));
    result->iterations_performed = iter + 1;
    result->final_error = result->error_history[iter];
    result->converged = converged;
    result->computation_time = (double)(clock() - start_time) / CLOCKS_PER_SEC;

cleanup:
    // 清理
    complex_image_destroy(object_field);
    complex_image_destroy(prev_object);
    for (int p = 0; p < params->num_planes; p++) {
        complex_image_destroy(plane_fields[p]);
    }
    free(plane_fields);

    return ret;
}
// ============================================================================
// 离轴全息重建
// ============================================================================

int holography_reconstruct_off_axis(
    const RealImage *hologram,
    const HolographyParams *params,
    ComplexImage *reconstructed)
{
    if (!hologram || !params || !reconstructed) {
        return RECONSTRUCTION_ERROR_NULL_POINTER;
    }

    if (params->type != HOLOGRAPHY_OFF_AXIS) {
        return RECONSTRUCTION_ERROR_INVALID_PARAMS;
    }

    int width = hologram->width;
    int height = hologram->height;

    // 创建复数全息图
    ComplexImage *hologram_complex = complex_image_create(width, height);
    if (!hologram_complex) {
        return RECONSTRUCTION_ERROR_MEMORY_ALLOCATION;
    }

    // 转换为复数（实部为全息图，虚部为0）
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            int idx = i * width + j;
            float val;
            
            if (hologram->channels == 1) {
                val = hologram->data[idx];
            } else {
                // 多通道取平均
                val = 0.0f;
                for (int c = 0; c < hologram->channels; c++) {
                    val += hologram->data[i * hologram->stride + j * hologram->channels + c];
                }
                val /= hologram->channels;
            }
            
            hologram_complex->data[idx] = val + 0.0f * I;
        }
    }

    // 执行FFT到频域
    ComplexImage *spectrum = complex_image_create(width, height);
    if (!spectrum) {
        complex_image_destroy(hologram_complex);
        return RECONSTRUCTION_ERROR_MEMORY_ALLOCATION;
    }

    int ret = fft_2d_forward(hologram_complex, spectrum);
    if (ret != FFT_SUCCESS) {
        complex_image_destroy(hologram_complex);
        complex_image_destroy(spectrum);
        return RECONSTRUCTION_ERROR_FFT_FAILED;
    }

    // 创建频域滤波器
    float *filter = (float*)malloc(width * height * sizeof(float));
    if (!filter) {
        complex_image_destroy(hologram_complex);
        complex_image_destroy(spectrum);
        return RECONSTRUCTION_ERROR_MEMORY_ALLOCATION;
    }

    // 计算滤波器中心位置（载波频率位置）
    float center_u = params->carrier_frequency_x * width;
    float center_v = params->carrier_frequency_y * height;
    float filter_radius = params->filter_size * sqrtf(width * width + height * height);

    // 创建带通滤波器（高斯型）
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            int idx = i * width + j;
            
            // 频域坐标（中心化）
            float u = (j < width/2) ? j : j - width;
            float v = (i < height/2) ? i : i - height;
            
            // 到载波频率的距离
            float du = u - center_u;
            float dv = v - center_v;
            float dist = sqrtf(du * du + dv * dv);
            
            // 高斯滤波器
            filter[idx] = expf(-(dist * dist) / (2.0f * filter_radius * filter_radius));
        }
    }

    // 应用滤波器
    for (int i = 0; i < width * height; i++) {
        spectrum->data[i] *= filter[i];
    }

    free(filter);

    // 频域平移（将+1级移到中心）
    ComplexImage *shifted_spectrum = complex_image_create(width, height);
    if (!shifted_spectrum) {
        complex_image_destroy(hologram_complex);
        complex_image_destroy(spectrum);
        return RECONSTRUCTION_ERROR_MEMORY_ALLOCATION;
    }

    // 应用相位斜坡进行频域平移
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            int idx = i * width + j;
            
            // 相位斜坡
            float phase_shift = -2.0f * M_PI * 
                (params->carrier_frequency_x * j + params->carrier_frequency_y * i);
            
            ComplexF shift_factor = cosf(phase_shift) + sinf(phase_shift) * I;
            shifted_spectrum->data[idx] = spectrum->data[idx] * shift_factor;
        }
    }

    complex_image_destroy(spectrum);

    // 逆FFT回到空域
    ret = fft_2d_inverse(shifted_spectrum, reconstructed);
    if (ret != FFT_SUCCESS) {
        complex_image_destroy(hologram_complex);
        complex_image_destroy(shifted_spectrum);
        return RECONSTRUCTION_ERROR_FFT_FAILED;
    }

    complex_image_destroy(hologram_complex);
    complex_image_destroy(shifted_spectrum);

    // 如果需要数值重聚焦
    if (params->numerical_refocus && params->diffraction_params) {
        ComplexImage *refocused = complex_image_create(width, height);
        if (!refocused) {
            return RECONSTRUCTION_ERROR_MEMORY_ALLOCATION;
        }

        ret = diffraction_propagate(reconstructed, refocused, params->diffraction_params);
        if (ret != DIFFRACTION_SUCCESS) {
            complex_image_destroy(refocused);
            return RECONSTRUCTION_ERROR_DIFFRACTION_FAILED;
        }

        memcpy(reconstructed->data, refocused->data, width * height * sizeof(ComplexF));
        complex_image_destroy(refocused);
    }

    // 如果需要去除孪生像
    if (params->remove_twin_image) {
        // 简单方法：应用支撑域约束
        // 更复杂的方法需要迭代算法
        for (int i = 0; i < width * height; i++) {
            float amplitude = cabsf(reconstructed->data[i]);
            float phase = cargf(reconstructed->data[i]);
            
            // 保留主要信息，抑制弱信号
            float threshold = 0.1f;  // 可调参数
            if (amplitude < threshold) {
                reconstructed->data[i] = 0.0f + 0.0f * I;
            }
        }
    }

    return RECONSTRUCTION_SUCCESS;
}

// ============================================================================
// 相移全息重建
// ============================================================================

int holography_reconstruct_phase_shifting(
    const RealImage **holograms,
    const HolographyParams *params,
    ComplexImage *reconstructed)
{
    if (!holograms || !params || !reconstructed) {
        return RECONSTRUCTION_ERROR_NULL_POINTER;
    }

    if (params->type != HOLOGRAPHY_PHASE_SHIFTING) {
        return RECONSTRUCTION_ERROR_INVALID_PARAMS;
    }

    if (params->num_phase_steps < 3) {
        return RECONSTRUCTION_ERROR_INVALID_PARAMS;
    }

    int width = holograms[0]->width;
    int height = holograms[0]->height;

    // 验证所有全息图尺寸一致
    for (int p = 1; p < params->num_phase_steps; p++) {
        if (holograms[p]->width != width || holograms[p]->height != height) {
            return RECONSTRUCTION_ERROR_INVALID_PARAMS;
        }
    }

    // 使用标准4步相移算法
    if (params->num_phase_steps == 4) {
        // I0, I1, I2, I3 对应相移 0, π/2, π, 3π/2
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                int idx = i * width + j;
                
                float I0, I1, I2, I3;
                
                // 提取强度值
                if (holograms[0]->channels == 1) {
                    I0 = holograms[0]->data[idx];
                    I1 = holograms[1]->data[idx];
                    I2 = holograms[2]->data[idx];
                    I3 = holograms[3]->data[idx];
                } else {
                    I0 = holograms[0]->data[i * holograms[0]->stride + j * holograms[0]->channels];
                    I1 = holograms[1]->data[i * holograms[1]->stride + j * holograms[1]->channels];
                    I2 = holograms[2]->data[i * holograms[2]->stride + j * holograms[2]->channels];
                    I3 = holograms[3]->data[i * holograms[3]->stride + j * holograms[3]->channels];
                }
                
                // 计算复振幅
                // Real = (I0 - I2) / 2
                // Imag = (I1 - I3) / 2
                float real_part = (I0 - I2) * 0.5f;
                float imag_part = (I1 - I3) * 0.5f;
                
                reconstructed->data[idx] = real_part + imag_part * I;
            }
        }
    } else {
        // 通用N步相移算法
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                int idx = i * width + j;
                
                float real_sum = 0.0f;
                float imag_sum = 0.0f;
                
                for (int p = 0; p < params->num_phase_steps; p++) {
                    float intensity;
                    
                    if (holograms[p]->channels == 1) {
                        intensity = holograms[p]->data[idx];
                    } else {
                        intensity = holograms[p]->data[i * holograms[p]->stride + 
                                                       j * holograms[p]->channels];
                    }
                    
                    float phase_shift = params->phase_shifts[p];
                    
                    real_sum += intensity * cosf(phase_shift);
                    imag_sum += intensity * sinf(phase_shift);
                }
                
                // 归一化
                real_sum *= 2.0f / params->num_phase_steps;
                imag_sum *= 2.0f / params->num_phase_steps;
                
                reconstructed->data[idx] = real_sum + imag_sum * I;
            }
        }
    }

    // 如果提供了参考波，进行归一化
    if (params->reference_wave) {
        for (int i = 0; i < width * height; i++) {
            // 除以参考波的共轭
            ComplexF ref_conj = conjf(params->reference_wave->data[i]);
            float ref_intensity = cabsf(ref_conj);
            
            if (ref_intensity > 1e-6f) {
                reconstructed->data[i] /= ref_conj;
            }
        }
    }

    // 如果需要数值重聚焦
    if (params->numerical_refocus && params->diffraction_params) {
        ComplexImage *refocused = complex_image_create(width, height);
        if (!refocused) {
            return RECONSTRUCTION_ERROR_MEMORY_ALLOCATION;
        }

        int ret = diffraction_propagate(reconstructed, refocused, params->diffraction_params);
        if (ret != DIFFRACTION_SUCCESS) {
            complex_image_destroy(refocused);
            return RECONSTRUCTION_ERROR_DIFFRACTION_FAILED;
        }

        memcpy(reconstructed->data, refocused->data, width * height * sizeof(ComplexF));
        complex_image_destroy(refocused);
    }

    return RECONSTRUCTION_SUCCESS;
}

// ============================================================================
// 同轴全息重建
// ============================================================================

int holography_reconstruct_inline(
    const RealImage *hologram,
    const HolographyParams *params,
    ComplexImage *reconstructed)
{
    if (!hologram || !params || !reconstructed) {
        return RECONSTRUCTION_ERROR_NULL_POINTER;
    }

    if (params->type != HOLOGRAPHY_INLINE) {
        return RECONSTRUCTION_ERROR_INVALID_PARAMS;
    }

    if (!params->diffraction_params) {
        return RECONSTRUCTION_ERROR_NULL_POINTER;
    }

    int width = hologram->width;
    int height = hologram->height;

    // 创建复数全息图
    ComplexImage *hologram_complex = complex_image_create(width, height);
    if (!hologram_complex) {
        return RECONSTRUCTION_ERROR_MEMORY_ALLOCATION;
    }

    // 转换为复数
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            int idx = i * width + j;
            float val;
            
            if (hologram->channels == 1) {
                val = hologram->data[idx];
            } else {
                val = 0.0f;
                for (int c = 0; c < hologram->channels; c++) {
                    val += hologram->data[i * hologram->stride + j * hologram->channels + c];
                }
                val /= hologram->channels;
            }
            
            hologram_complex->data[idx] = val + 0.0f * I;
        }
    }

    // 如果有参考波，减去参考波强度
    if (params->reference_wave) {
        for (int i = 0; i < width * height; i++) {
            float ref_intensity = cabsf(params->reference_wave->data[i]);
            ref_intensity *= ref_intensity;
            
            float hologram_val = crealf(hologram_complex->data[i]);
            hologram_complex->data[i] = (hologram_val - ref_intensity) + 0.0f * I;
        }
    }

    // 数值重建（反向传播）
    DiffractionParams backward_params = *params->diffraction_params;
    backward_params.direction = DIRECTION_BACKWARD;

    int ret = diffraction_propagate(hologram_complex, reconstructed, &backward_params);
    if (ret != DIFFRACTION_SUCCESS) {
        complex_image_destroy(hologram_complex);
        return RECONSTRUCTION_ERROR_DIFFRACTION_FAILED;
    }

    complex_image_destroy(hologram_complex);

    // 如果需要去除孪生像，使用迭代方法
    if (params->remove_twin_image) {
        // 使用简单的迭代方法去除孪生像
        ComplexImage *object_estimate = complex_image_create(width, height);
        ComplexImage *forward_prop = complex_image_create(width, height);
        
        if (!object_estimate || !forward_prop) {
            complex_image_destroy(object_estimate);
            complex_image_destroy(forward_prop);
            return RECONSTRUCTION_ERROR_MEMORY_ALLOCATION;
        }

        // 初始估计
        memcpy(object_estimate->data, reconstructed->data, width * height * sizeof(ComplexF));

        // 迭代几次
        int num_iterations = 10;
        for (int iter = 0; iter < num_iterations; iter++) {
            // 前向传播
            DiffractionParams forward_params = *params->diffraction_params;
            forward_params.direction = DIRECTION_FORWARD;
            
            ret = diffraction_propagate(object_estimate, forward_prop, &forward_params);
            if (ret != DIFFRACTION_SUCCESS) {
                break;
            }

            // 应用全息图约束
            for (int i = 0; i < width * height; i++) {
                float measured_intensity = crealf(hologram_complex->data[i]);
                float current_phase = cargf(forward_prop->data[i]);
                
                forward_prop->data[i] = sqrtf(fabsf(measured_intensity)) * 
                    (cosf(current_phase) + sinf(current_phase) * I);
            }

            // 反向传播
            ret = diffraction_propagate(forward_prop, object_estimate, &backward_params);
            if (ret != DIFFRACTION_SUCCESS) {
                break;
            }

            // 应用物体约束（正值）
            for (int i = 0; i < width * height; i++) {
                float real_part = crealf(object_estimate->data[i]);
                float imag_part = cimagf(object_estimate->data[i]);
                
                if (real_part < 0.0f) {
                    real_part = 0.0f;
                }
                
                object_estimate->data[i] = real_part + imag_part * I;
            }
        }

        // 使用改进的估计
        memcpy(reconstructed->data, object_estimate->data, width * height * sizeof(ComplexF));

        complex_image_destroy(object_estimate);
        complex_image_destroy(forward_prop);
    }

    return RECONSTRUCTION_SUCCESS;
}

// ============================================================================
// 通用全息重建接口
// ============================================================================

int holography_reconstruct(
    const RealImage *hologram,
    const HolographyParams *params,
    ComplexImage *reconstructed)
{
    if (!hologram || !params || !reconstructed) {
        return RECONSTRUCTION_ERROR_NULL_POINTER;
    }

    // 根据全息类型调用相应函数
    switch (params->type) {
        case HOLOGRAPHY_OFF_AXIS:
            return holography_reconstruct_off_axis(hologram, params, reconstructed);
        
        case HOLOGRAPHY_INLINE:
            return holography_reconstruct_inline(hologram, params, reconstructed);
        
        case HOLOGRAPHY_PHASE_SHIFTING:
            // 对于相移全息，需要多个全息图
            // 这里只是占位，实际应该使用专门的接口
            return RECONSTRUCTION_ERROR_INVALID_PARAMS;
        
        default:
            return RECONSTRUCTION_ERROR_INVALID_PARAMS;
    }
}

// ============================================================================
// 数值重聚焦
// ============================================================================

int numerical_refocus(
    const ComplexImage *field,
    double refocus_distance,
    const DiffractionParams *base_params,
    ComplexImage *refocused)
{
    if (!field || !base_params || !refocused) {
        return RECONSTRUCTION_ERROR_NULL_POINTER;
    }

    if (field->width != refocused->width || field->height != refocused->height) {
        return RECONSTRUCTION_ERROR_INVALID_PARAMS;
    }

    // 创建重聚焦参数
    DiffractionParams refocus_params = *base_params;
    refocus_params.propagation_distance = refocus_distance;

    // 执行衍射传播
    int ret = diffraction_propagate(field, refocused, &refocus_params);
    if (ret != DIFFRACTION_SUCCESS) {
        return RECONSTRUCTION_ERROR_DIFFRACTION_FAILED;
    }

    return RECONSTRUCTION_SUCCESS;
}

// ============================================================================
// 自动聚焦
// ============================================================================

int autofocus(
    const ComplexImage *field,
    const DiffractionParams *base_params,
    double distance_min,
    double distance_max,
    int num_steps,
    double *best_distance,
    ComplexImage *focused)
{
    if (!field || !base_params || !best_distance) {
        return RECONSTRUCTION_ERROR_NULL_POINTER;
    }

    if (num_steps <= 0 || distance_min >= distance_max) {
        return RECONSTRUCTION_ERROR_INVALID_PARAMS;
    }

    int width = field->width;
    int height = field->height;

    ComplexImage *test_field = complex_image_create(width, height);
    if (!test_field) {
        return RECONSTRUCTION_ERROR_MEMORY_ALLOCATION;
    }

    double step_size = (distance_max - distance_min) / (num_steps - 1);
    double best_focus_metric = -1.0;
    *best_distance = distance_min;

    // 遍历不同距离
    for (int step = 0; step < num_steps; step++) {
        double distance = distance_min + step * step_size;

        // 重聚焦到该距离
        int ret = numerical_refocus(field, distance, base_params, test_field);
        if (ret != RECONSTRUCTION_SUCCESS) {
            complex_image_destroy(test_field);
            return ret;
        }

        // 计算聚焦度量（使用梯度方差）
        float focus_metric = 0.0f;
        
        for (int i = 1; i < height - 1; i++) {
            for (int j = 1; j < width - 1; j++) {
                int idx = i * width + j;
                
                float intensity = cabsf(test_field->data[idx]);
                intensity *= intensity;
                
                // 计算梯度
                float intensity_right = cabsf(test_field->data[idx + 1]);
                intensity_right *= intensity_right;
                float intensity_down = cabsf(test_field->data[idx + width]);
                intensity_down *= intensity_down;
                
                float grad_x = intensity_right - intensity;
                float grad_y = intensity_down - intensity;
                
                focus_metric += grad_x * grad_x + grad_y * grad_y;
            }
        }

        // 更新最佳距离
        if (focus_metric > best_focus_metric) {
            best_focus_metric = focus_metric;
            *best_distance = distance;
        }
    }

    // 重聚焦到最佳距离
    if (focused) {
        int ret = numerical_refocus(field, *best_distance, base_params, focused);
        if (ret != RECONSTRUCTION_SUCCESS) {
            complex_image_destroy(test_field);
            return ret;
        }
    }

    complex_image_destroy(test_field);

    return RECONSTRUCTION_SUCCESS;
}
// ============================================================================
// 相位展开
// ============================================================================

int phase_unwrap_2d(
    const RealImage *wrapped_phase,
    RealImage *unwrapped_phase)
{
    if (!wrapped_phase || !unwrapped_phase) {
        return RECONSTRUCTION_ERROR_NULL_POINTER;
    }

    if (wrapped_phase->width != unwrapped_phase->width ||
        wrapped_phase->height != unwrapped_phase->height) {
        return RECONSTRUCTION_ERROR_INVALID_PARAMS;
    }

    int width = wrapped_phase->width;
    int height = wrapped_phase->height;

    // 使用质量引导路径跟踪算法
    // 1. 计算相位质量图
    float *quality = (float*)malloc(width * height * sizeof(float));
    bool *unwrapped = (bool*)calloc(width * height, sizeof(bool));
    
    if (!quality || !unwrapped) {
        free(quality);
        free(unwrapped);
        return RECONSTRUCTION_ERROR_MEMORY_ALLOCATION;
    }

    // 计算质量图（使用相位导数方差）
    for (int i = 1; i < height - 1; i++) {
        for (int j = 1; j < width - 1; j++) {
            int idx = i * width + j;
            
            float phase_center = wrapped_phase->data[idx];
            float phase_right = wrapped_phase->data[idx + 1];
            float phase_left = wrapped_phase->data[idx - 1];
            float phase_up = wrapped_phase->data[idx - width];
            float phase_down = wrapped_phase->data[idx + width];
            
            // 计算二阶导数
            float d2x = phase_right + phase_left - 2.0f * phase_center;
            float d2y = phase_up + phase_down - 2.0f * phase_center;
            
            quality[idx] = sqrtf(d2x * d2x + d2y * d2y);
        }
    }

    // 边界质量设为最低
    for (int i = 0; i < height; i++) {
        quality[i * width] = 1e6f;
        quality[i * width + width - 1] = 1e6f;
    }
    for (int j = 0; j < width; j++) {
        quality[j] = 1e6f;
        quality[(height - 1) * width + j] = 1e6f;
    }

    // 2. 创建优先队列（使用简单的排序数组）
    typedef struct {
        int index;
        float quality;
    } QueueElement;

    QueueElement *queue = (QueueElement*)malloc(width * height * sizeof(QueueElement));
    if (!queue) {
        free(quality);
        free(unwrapped);
        return RECONSTRUCTION_ERROR_MEMORY_ALLOCATION;
    }

    int queue_size = 0;

    // 找到质量最好的起始点
    int start_idx = 0;
    float best_quality = quality[0];
    for (int i = 0; i < width * height; i++) {
        if (quality[i] < best_quality) {
            best_quality = quality[i];
            start_idx = i;
        }
    }

    // 初始化起始点
    unwrapped_phase->data[start_idx] = wrapped_phase->data[start_idx];
    unwrapped[start_idx] = true;

    // 将邻居加入队列
    int start_i = start_idx / width;
    int start_j = start_idx % width;

    int neighbors[4][2] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
    
    for (int n = 0; n < 4; n++) {
        int ni = start_i + neighbors[n][0];
        int nj = start_j + neighbors[n][1];
        
        if (ni >= 0 && ni < height && nj >= 0 && nj < width) {
            int nidx = ni * width + nj;
            if (!unwrapped[nidx]) {
                queue[queue_size].index = nidx;
                queue[queue_size].quality = quality[nidx];
                queue_size++;
            }
        }
    }

    // 3. 路径跟踪展开
    while (queue_size > 0) {
        // 找到质量最好的像素
        int best_idx = 0;
        for (int i = 1; i < queue_size; i++) {
            if (queue[i].quality < queue[best_idx].quality) {
                best_idx = i;
            }
        }

        int current_idx = queue[best_idx].index;
        
        // 从队列中移除
        queue[best_idx] = queue[queue_size - 1];
        queue_size--;

        if (unwrapped[current_idx]) {
            continue;
        }

        // 展开当前像素
        int ci = current_idx / width;
        int cj = current_idx % width;
        
        float wrapped_value = wrapped_phase->data[current_idx];
        float unwrapped_value = wrapped_value;
        
        // 找到已展开的邻居
        int num_unwrapped_neighbors = 0;
        float neighbor_sum = 0.0f;
        
        for (int n = 0; n < 4; n++) {
            int ni = ci + neighbors[n][0];
            int nj = cj + neighbors[n][1];
            
            if (ni >= 0 && ni < height && nj >= 0 && nj < width) {
                int nidx = ni * width + nj;
                if (unwrapped[nidx]) {
                    neighbor_sum += unwrapped_phase->data[nidx];
                    num_unwrapped_neighbors++;
                }
            }
        }

        if (num_unwrapped_neighbors > 0) {
            float neighbor_avg = neighbor_sum / num_unwrapped_neighbors;
            
            // 计算需要的2π倍数
            float diff = wrapped_value - neighbor_avg;
            
            // 将diff归一化到[-π, π]
            while (diff > M_PI) diff -= 2.0f * M_PI;
            while (diff < -M_PI) diff += 2.0f * M_PI;
            
            unwrapped_value = neighbor_avg + diff;
        }

        unwrapped_phase->data[current_idx] = unwrapped_value;
        unwrapped[current_idx] = true;

        // 将未展开的邻居加入队列
        for (int n = 0; n < 4; n++) {
            int ni = ci + neighbors[n][0];
            int nj = cj + neighbors[n][1];
            
            if (ni >= 0 && ni < height && nj >= 0 && nj < width) {
                int nidx = ni * width + nj;
                if (!unwrapped[nidx]) {
                    // 检查是否已在队列中
                    bool in_queue = false;
                    for (int q = 0; q < queue_size; q++) {
                        if (queue[q].index == nidx) {
                            in_queue = true;
                            break;
                        }
                    }
                    
                    if (!in_queue) {
                        queue[queue_size].index = nidx;
                        queue[queue_size].quality = quality[nidx];
                        queue_size++;
                    }
                }
            }
        }
    }

    // 清理
    free(quality);
    free(unwrapped);
    free(queue);

    return RECONSTRUCTION_SUCCESS;
}

// ============================================================================
// 相位展开（从复数场）
// ============================================================================

int phase_unwrap_from_complex(
    const ComplexImage *complex_field,
    RealImage *unwrapped_phase)
{
    if (!complex_field || !unwrapped_phase) {
        return RECONSTRUCTION_ERROR_NULL_POINTER;
    }

    if (complex_field->width != unwrapped_phase->width ||
        complex_field->height != unwrapped_phase->height) {
        return RECONSTRUCTION_ERROR_INVALID_PARAMS;
    }

    int width = complex_field->width;
    int height = complex_field->height;

    // 提取包裹相位
    RealImage *wrapped_phase = real_image_create(width, height, 1);
    if (!wrapped_phase) {
        return RECONSTRUCTION_ERROR_MEMORY_ALLOCATION;
    }

    for (int i = 0; i < width * height; i++) {
        wrapped_phase->data[i] = cargf(complex_field->data[i]);
    }

    // 执行相位展开
    int ret = phase_unwrap_2d(wrapped_phase, unwrapped_phase);
    
    real_image_destroy(wrapped_phase);

    return ret;
}

// ============================================================================
// 结果质量评估
// ============================================================================

int evaluate_reconstruction_quality(
    const ComplexImage *reconstructed,
    const ComplexImage *ground_truth,
    ReconstructionQuality *quality)
{
    if (!reconstructed || !quality) {
        return RECONSTRUCTION_ERROR_NULL_POINTER;
    }

    if (reconstructed->width != ground_truth->width ||
        reconstructed->height != ground_truth->height) {
        return RECONSTRUCTION_ERROR_INVALID_PARAMS;
    }

    int size = reconstructed->width * reconstructed->height;

    // 1. 计算RMSE
    float mse = 0.0f;
    for (int i = 0; i < size; i++) {
        ComplexF diff = reconstructed->data[i] - ground_truth->data[i];
        mse += crealf(diff) * crealf(diff) + cimagf(diff) * cimagf(diff);
    }
    quality->rmse = sqrtf(mse / size);

    // 2. 计算PSNR
    float max_val = 0.0f;
    for (int i = 0; i < size; i++) {
        float val = cabsf(ground_truth->data[i]);
        if (val > max_val) {
            max_val = val;
        }
    }
    
    if (max_val > 0.0f && quality->rmse > 0.0f) {
        quality->psnr = 20.0f * log10f(max_val / quality->rmse);
    } else {
        quality->psnr = INFINITY;
    }

    // 3. 计算SSIM（结构相似性）
    // 简化版本：使用局部窗口
    int window_size = 11;
    int half_window = window_size / 2;
    
    float ssim_sum = 0.0f;
    int num_windows = 0;
    
    float C1 = 0.01f * 0.01f;  // 常数
    float C2 = 0.03f * 0.03f;
    
    for (int i = half_window; i < reconstructed->height - half_window; i += window_size) {
        for (int j = half_window; j < reconstructed->width - half_window; j += window_size) {
            // 计算局部均值和方差
            float mean_rec = 0.0f, mean_gt = 0.0f;
            float var_rec = 0.0f, var_gt = 0.0f;
            float covar = 0.0f;
            int window_count = 0;
            
            for (int wi = -half_window; wi <= half_window; wi++) {
                for (int wj = -half_window; wj <= half_window; wj++) {
                    int idx = (i + wi) * reconstructed->width + (j + wj);
                    
                    float val_rec = cabsf(reconstructed->data[idx]);
                    float val_gt = cabsf(ground_truth->data[idx]);
                    
                    mean_rec += val_rec;
                    mean_gt += val_gt;
                    window_count++;
                }
            }
            
            mean_rec /= window_count;
            mean_gt /= window_count;
            
            for (int wi = -half_window; wi <= half_window; wi++) {
                for (int wj = -half_window; wj <= half_window; wj++) {
                    int idx = (i + wi) * reconstructed->width + (j + wj);
                    
                    float val_rec = cabsf(reconstructed->data[idx]);
                    float val_gt = cabsf(ground_truth->data[idx]);
                    
                    float diff_rec = val_rec - mean_rec;
                    float diff_gt = val_gt - mean_gt;
                    
                    var_rec += diff_rec * diff_rec;
                    var_gt += diff_gt * diff_gt;
                    covar += diff_rec * diff_gt;
                }
            }
            
            var_rec /= window_count;
            var_gt /= window_count;
            covar /= window_count;
            
            // SSIM公式
            float numerator = (2.0f * mean_rec * mean_gt + C1) * (2.0f * covar + C2);
            float denominator = (mean_rec * mean_rec + mean_gt * mean_gt + C1) * 
                               (var_rec + var_gt + C2);
            
            if (denominator > 0.0f) {
                ssim_sum += numerator / denominator;
                num_windows++;
            }
        }
    }
    
    quality->ssim = (num_windows > 0) ? (ssim_sum / num_windows) : 0.0f;

    // 4. 计算相位误差
    float phase_error_sum = 0.0f;
    for (int i = 0; i < size; i++) {
        float phase_rec = cargf(reconstructed->data[i]);
        float phase_gt = cargf(ground_truth->data[i]);
        
        float phase_diff = phase_rec - phase_gt;
        
        // 归一化到[-π, π]
        while (phase_diff > M_PI) phase_diff -= 2.0f * M_PI;
        while (phase_diff < -M_PI) phase_diff += 2.0f * M_PI;
        
        phase_error_sum += phase_diff * phase_diff;
    }
    quality->phase_error = sqrtf(phase_error_sum / size);

    // 5. 计算振幅误差
    float amplitude_error_sum = 0.0f;
    for (int i = 0; i < size; i++) {
        float amp_rec = cabsf(reconstructed->data[i]);
        float amp_gt = cabsf(ground_truth->data[i]);
        
        float amp_diff = amp_rec - amp_gt;
        amplitude_error_sum += amp_diff * amp_diff;
    }
    quality->amplitude_error = sqrtf(amplitude_error_sum / size);

    return RECONSTRUCTION_SUCCESS;
}

// ============================================================================
// 保存重建结果
// ============================================================================

int save_reconstruction_result(
    const ReconstructionResult *result,
    const char *output_prefix)
{
    if (!result || !output_prefix) {
        return RECONSTRUCTION_ERROR_NULL_POINTER;
    }

    char filename[512];
    int ret;

    // 1. 保存振幅图像
    snprintf(filename, sizeof(filename), "%s_amplitude.png", output_prefix);
    RealImage *amplitude = real_image_create(
        result->reconstructed->width,
        result->reconstructed->height, 1);
    
    if (!amplitude) {
        return RECONSTRUCTION_ERROR_MEMORY_ALLOCATION;
    }

    for (int i = 0; i < result->reconstructed->width * result->reconstructed->height; i++) {
        amplitude->data[i] = cabsf(result->reconstructed->data[i]);
    }

    ret = image_save(filename, amplitude);
    real_image_destroy(amplitude);
    
    if (ret != IMAGE_SUCCESS) {
        return RECONSTRUCTION_ERROR_IO;
    }

    // 2. 保存相位图像
    snprintf(filename, sizeof(filename), "%s_phase.png", output_prefix);
    RealImage *phase = real_image_create(
        result->reconstructed->width,
        result->reconstructed->height, 1);
    
    if (!phase) {
        return RECONSTRUCTION_ERROR_MEMORY_ALLOCATION;
    }

    for (int i = 0; i < result->reconstructed->width * result->reconstructed->height; i++) {
        float phase_val = cargf(result->reconstructed->data[i]);
        // 归一化到[0, 1]
        phase->data[i] = (phase_val + M_PI) / (2.0f * M_PI);
    }

    ret = image_save(filename, phase);
    real_image_destroy(phase);
    
    if (ret != IMAGE_SUCCESS) {
        return RECONSTRUCTION_ERROR_IO;
    }

    // 3. 保存误差历史
    snprintf(filename, sizeof(filename), "%s_error_history.txt", output_prefix);
    FILE *fp = fopen(filename, "w");
    if (!fp) {
        return RECONSTRUCTION_ERROR_IO;
    }

    fprintf(fp, "# Iteration\tError\n");
    for (int i = 0; i < result->error_history_length; i++) {
        fprintf(fp, "%d\t%e\n", i + 1, result->error_history[i]);
    }
    fclose(fp);

    // 4. 保存元数据
    snprintf(filename, sizeof(filename), "%s_metadata.txt", output_prefix);
    fp = fopen(filename, "w");
    if (!fp) {
        return RECONSTRUCTION_ERROR_IO;
    }

    fprintf(fp, "Reconstruction Metadata\n");
    fprintf(fp, "=======================\n");
    fprintf(fp, "Image size: %d x %d\n", 
            result->reconstructed->width, result->reconstructed->height);
    fprintf(fp, "Iterations performed: %d\n", result->iterations_performed);
    fprintf(fp, "Final error: %e\n", result->final_error);
    fprintf(fp, "Converged: %s\n", result->converged ? "Yes" : "No");
    fprintf(fp, "Computation time: %.3f seconds\n", result->computation_time);
    
    fclose(fp);

    return RECONSTRUCTION_SUCCESS;
}

// ============================================================================
// 可视化重建结果
// ============================================================================

int visualize_reconstruction(
    const ComplexImage *reconstructed,
    RealImage *visualization,
    VisualizationType type)
{
    if (!reconstructed || !visualization) {
        return RECONSTRUCTION_ERROR_NULL_POINTER;
    }

    if (reconstructed->width != visualization->width ||
        reconstructed->height != visualization->height) {
        return RECONSTRUCTION_ERROR_INVALID_PARAMS;
    }

    int size = reconstructed->width * reconstructed->height;

    switch (type) {
        case VISUALIZATION_AMPLITUDE:
            // 显示振幅
            for (int i = 0; i < size; i++) {
                float amp = cabsf(reconstructed->data[i]);
                
                // 归一化
                if (visualization->channels == 1) {
                    visualization->data[i] = amp;
                } else {
                    for (int c = 0; c < visualization->channels; c++) {
                        visualization->data[i * visualization->channels + c] = amp;
                    }
                }
            }
            break;

        case VISUALIZATION_PHASE:
            // 显示相位（彩色编码）
            for (int i = 0; i < size; i++) {
                float phase = cargf(reconstructed->data[i]);
                
                // 归一化到[0, 1]
                float normalized_phase = (phase + M_PI) / (2.0f * M_PI);
                
                if (visualization->channels >= 3) {
                    // HSV到RGB转换（H = phase, S = 1, V = 1）
                    float h = normalized_phase * 6.0f;
                    int hi = (int)h;
                    float f = h - hi;
                    
                    float r, g, b;
                    switch (hi % 6) {
                        case 0: r = 1.0f; g = f; b = 0.0f; break;
                        case 1: r = 1.0f - f; g = 1.0f; b = 0.0f; break;
                        case 2: r = 0.0f; g = 1.0f; b = f; break;
                        case 3: r = 0.0f; g = 1.0f - f; b = 1.0f; break;
                        case 4: r = f; g = 0.0f; b = 1.0f; break;
                        case 5: r = 1.0f; g = 0.0f; b = 1.0f - f; break;
                        default: r = g = b = 0.0f;
                    }
                    
                    visualization->data[i * visualization->channels + 0] = r;
                    visualization->data[i * visualization->channels + 1] = g;
                    visualization->data[i * visualization->channels + 2] = b;
                } else {
                    visualization->data[i] = normalized_phase;
                }
            }
            break;

        case VISUALIZATION_INTENSITY:
            // 显示强度
            for (int i = 0; i < size; i++) {
                float amp = cabsf(reconstructed->data[i]);
                float intensity = amp * amp;
                
                if (visualization->channels == 1) {
                    visualization->data[i] = intensity;
                } else {
                    for (int c = 0; c < visualization->channels; c++) {
                        visualization->data[i * visualization->channels + c] = intensity;
                    }
                }
            }
            break;

        case VISUALIZATION_COMPLEX:
            // 显示复数（实部和虚部）
            for (int i = 0; i < size; i++) {
                float real_part = crealf(reconstructed->data[i]);
                float imag_part = cimagf(reconstructed->data[i]);
                
                if (visualization->channels >= 2) {
                    visualization->data[i * visualization->channels + 0] = 
                        (real_part + 1.0f) * 0.5f;  // 归一化到[0, 1]
                    visualization->data[i * visualization->channels + 1] = 
                        (imag_part + 1.0f) * 0.5f;
                    
                    if (visualization->channels >= 3) {
                        visualization->data[i * visualization->channels + 2] = 0.0f;
                    }
                } else {
                    visualization->data[i] = (real_part + 1.0f) * 0.5f;
                }
            }
            break;

        default:
            return RECONSTRUCTION_ERROR_INVALID_PARAMS;
    }

    return RECONSTRUCTION_SUCCESS;
}

// ============================================================================
// 获取错误信息
// ============================================================================

const char* reconstruction_get_error_string(int error_code)
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
        case RECONSTRUCTION_ERROR_CONVERGENCE:
            return "Algorithm did not converge";
        case RECONSTRUCTION_ERROR_IO:
            return "Input/output error";
        default:
            return "Unknown error";
    }
}

