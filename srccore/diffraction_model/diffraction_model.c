#include "diffraction_model.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <fftw3.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ============================================================================
// 内部辅助函数声明
// ============================================================================

static int fresnel_propagate(
    const ComplexImage *input,
    const DiffractionParams *params,
    ComplexImage *output,
    bool forward
);

static int fraunhofer_propagate(
    const ComplexImage *input,
    const DiffractionParams *params,
    ComplexImage *output,
    bool forward
);

static int angular_spectrum_propagate(
    const ComplexImage *input,
    const DiffractionParams *params,
    ComplexImage *output,
    bool forward
);

static ComplexImage* apply_padding(const ComplexImage *img, int factor);
static ComplexImage* remove_padding(const ComplexImage *img, int orig_width, int orig_height);
static void fftshift_complex(ComplexF *data, int width, int height);

// ============================================================================
// 参数管理函数实现
// ============================================================================

DiffractionParams* diffraction_params_create(
    int width, int height,
    double pixel_size, double distance,
    double wavelength, DiffractionType type)
{
    if (width <= 0 || height <= 0 || pixel_size <= 0 || 
        distance <= 0 || wavelength <= 0) {
        return NULL;
    }

    DiffractionParams *params = (DiffractionParams*)malloc(sizeof(DiffractionParams));
    if (!params) return NULL;

    params->width = width;
    params->height = height;
    params->pixel_size = pixel_size;
    params->propagation_distance = distance;
    params->type = type;
    params->use_padding = true;
    params->padding_factor = 2;

    // 初始化单波长
    params->wavelengths = (WavelengthInfo*)malloc(sizeof(WavelengthInfo));
    if (!params->wavelengths) {
        free(params);
        return NULL;
    }
    params->wavelengths[0].wavelength = wavelength;
    params->wavelengths[0].weight = 1.0;
    params->num_wavelengths = 1;

    return params;
}

int diffraction_params_add_wavelength(
    DiffractionParams *params,
    double wavelength, double weight)
{
    if (!params || wavelength <= 0 || weight < 0) {
        return DIFFRACTION_ERROR_INVALID_PARAMS;
    }

    int new_count = params->num_wavelengths + 1;
    WavelengthInfo *new_wavelengths = (WavelengthInfo*)realloc(
        params->wavelengths,
        new_count * sizeof(WavelengthInfo)
    );
    
    if (!new_wavelengths) {
        return DIFFRACTION_ERROR_MEMORY_ALLOCATION;
    }

    new_wavelengths[new_count - 1].wavelength = wavelength;
    new_wavelengths[new_count - 1].weight = weight;
    
    params->wavelengths = new_wavelengths;
    params->num_wavelengths = new_count;

    return DIFFRACTION_SUCCESS;
}

void diffraction_params_set_padding(
    DiffractionParams *params,
    bool use_padding, int padding_factor)
{
    if (params) {
        params->use_padding = use_padding;
        params->padding_factor = padding_factor;
    }
}

void diffraction_params_destroy(DiffractionParams *params)
{
    if (params) {
        free(params->wavelengths);
        free(params);
    }
}

// ============================================================================
// 图像管理函数实现
// ============================================================================

ComplexImage* complex_image_create(int width, int height)
{
    if (width <= 0 || height <= 0) return NULL;

    ComplexImage *img = (ComplexImage*)malloc(sizeof(ComplexImage));
    if (!img) return NULL;

    img->width = width;
    img->height = height;
    img->stride = width;
    
    // 使用FFTW的内存分配以确保对齐
    img->data = (ComplexF*)fftwf_malloc(width * height * sizeof(ComplexF));
    if (!img->data) {
        free(img);
        return NULL;
    }

    memset(img->data, 0, width * height * sizeof(ComplexF));
    return img;
}

ComplexImage* complex_image_clone(const ComplexImage *src)
{
    if (!src || !src->data) return NULL;

    ComplexImage *dst = complex_image_create(src->width, src->height);
    if (!dst) return NULL;

    memcpy(dst->data, src->data, src->width * src->height * sizeof(ComplexF));
    return dst;
}

void complex_image_destroy(ComplexImage *img)
{
    if (img) {
        if (img->data) fftwf_free(img->data);
        free(img);
    }
}

RealImage* real_image_create(int width, int height, int channels)
{
    if (width <= 0 || height <= 0 || channels <= 0) return NULL;

    RealImage *img = (RealImage*)malloc(sizeof(RealImage));
    if (!img) return NULL;

    img->width = width;
    img->height = height;
    img->channels = channels;
    img->stride = width * channels;

    img->data = (float*)malloc(width * height * channels * sizeof(float));
    if (!img->data) {
        free(img);
        return NULL;
    }

    memset(img->data, 0, width * height * channels * sizeof(float));
    return img;
}
// ============================================================================
// 核心衍射传播函数实现
// ============================================================================

int diffraction_propagate_forward(
    const ComplexImage *object,
    const DiffractionParams *params,
    ComplexImage *image)
{
    if (!object || !params || !image) {
        return DIFFRACTION_ERROR_NULL_POINTER;
    }

    if (object->width != params->width || object->height != params->height ||
        image->width != params->width || image->height != params->height) {
        return DIFFRACTION_ERROR_INVALID_PARAMS;
    }

    switch (params->type) {
        case DIFFRACTION_FRESNEL:
            return fresnel_propagate(object, params, image, true);
        case DIFFRACTION_FRAUNHOFER:
            return fraunhofer_propagate(object, params, image, true);
        case DIFFRACTION_ANGULAR_SPECTRUM:
            return angular_spectrum_propagate(object, params, image, true);
        default:
            return DIFFRACTION_ERROR_INVALID_PARAMS;
    }
}

int diffraction_propagate_backward(
    const ComplexImage *image,
    const DiffractionParams *params,
    ComplexImage *object)
{
    if (!image || !params || !object) {
        return DIFFRACTION_ERROR_NULL_POINTER;
    }

    if (image->width != params->width || image->height != params->height ||
        object->width != params->width || object->height != params->height) {
        return DIFFRACTION_ERROR_INVALID_PARAMS;
    }

    switch (params->type) {
        case DIFFRACTION_FRESNEL:
            return fresnel_propagate(image, params, object, false);
        case DIFFRACTION_FRAUNHOFER:
            return fraunhofer_propagate(image, params, object, false);
        case DIFFRACTION_ANGULAR_SPECTRUM:
            return angular_spectrum_propagate(image, params, object, false);
        default:
            return DIFFRACTION_ERROR_INVALID_PARAMS;
    }
}

int diffraction_propagate_multispectral(
    const ComplexImage *object,
    const DiffractionParams *params,
    RealImage *image)
{
    if (!object || !params || !image) {
        return DIFFRACTION_ERROR_NULL_POINTER;
    }
    
    if (params->num_wavelengths <= 0) {
        return DIFFRACTION_ERROR_INVALID_WAVELENGTH;
    }

    // 创建临时图像存储每个波长的结果
    ComplexImage *temp_image = complex_image_create(params->width, params->height);
    if (!temp_image) {
        return DIFFRACTION_ERROR_MEMORY_ALLOCATION;
    }

    // 清零输出图像
    memset(image->data, 0, image->width * image->height * image->channels * sizeof(float));

    // 归一化权重
    double total_weight = 0.0;
    for (int w = 0; w < params->num_wavelengths; w++) {
        total_weight += params->wavelengths[w].weight;
    }

    // 对每个波长进行传播并累加
    for (int w = 0; w < params->num_wavelengths; w++) {
        // 创建单波长参数
        DiffractionParams single_params = *params;
        single_params.num_wavelengths = 1;
        WavelengthInfo single_wavelength = params->wavelengths[w];
        single_params.wavelengths = &single_wavelength;

        // 传播
        int result = diffraction_propagate_forward(object, &single_params, temp_image);
        if (result != DIFFRACTION_SUCCESS) {
            complex_image_destroy(temp_image);
            return result;
        }

        // 计算强度并累加
        double weight = params->wavelengths[w].weight / total_weight;
        for (int i = 0; i < params->height; i++) {
            for (int j = 0; j < params->width; j++) {
                int idx = i * params->width + j;
                ComplexF val = temp_image->data[idx];
                float real_part = crealf(val);
                float imag_part = cimagf(val);
                float intensity = real_part * real_part + imag_part * imag_part;
                
                // 如果是RGB图像，根据波长分配到不同通道
                if (image->channels == 3) {
                    double lambda = params->wavelengths[w].wavelength * 1e9; // 转换为纳米
                    int channel = 0;
                    if (lambda < 500) channel = 2;      // 蓝色
                    else if (lambda < 600) channel = 1; // 绿色
                    else channel = 0;                   // 红色
                    image->data[i * image->stride + j * image->channels + channel] += 
                        intensity * weight;
                } else {
                    // 灰度图像
                    image->data[idx] += intensity * weight;
                }
            }
        }
    }

    complex_image_destroy(temp_image);
    return DIFFRACTION_SUCCESS;
}

// ============================================================================
// 传递函数相关实现
// ============================================================================

int diffraction_compute_transfer_function(
    const DiffractionParams *params,
    ComplexImage *transfer_func)
{
    if (!params || !transfer_func) {
        return DIFFRACTION_ERROR_NULL_POINTER;
    }

    if (transfer_func->width != params->width || 
        transfer_func->height != params->height) {
        return DIFFRACTION_ERROR_INVALID_PARAMS;
    }

    double lambda = params->wavelengths[0].wavelength;
    double z = params->propagation_distance;
    double dx = params->pixel_size;
    int width = params->width;
    int height = params->height;

    double k = 2.0 * M_PI / lambda;
    double df_x = 1.0 / (width * dx);
    double df_y = 1.0 / (height * dx);

    // 根据衍射类型计算传递函数
    switch (params->type) {
        case DIFFRACTION_ANGULAR_SPECTRUM:
            // 角谱法传递函数: H(fx, fy) = exp(i * 2π * z * sqrt(1/λ² - fx² - fy²))
            for (int i = 0; i < height; i++) {
                for (int j = 0; j < width; j++) {
                    int idx = i * width + j;
                    
                    int fi = (i < height / 2) ? i : i - height;
                    int fj = (j < width / 2) ? j : j - width;
                    
                    double fx = fj * df_x;
                    double fy = fi * df_y;
                    
                    double f_squared = fx * fx + fy * fy;
                    double lambda_squared = lambda * lambda;
                    
                    if (f_squared < 1.0 / lambda_squared) {
                        // 传播波
                        double kz = sqrt(1.0 / lambda_squared - f_squared);
                        double phase = 2.0 * M_PI * kz * z;
                        transfer_func->data[idx] = cosf(phase) + I * sinf(phase);
                    } else {
                        // 倏逝波（衰减）
                        double kz = sqrt(f_squared - 1.0 / lambda_squared);
                        double decay = exp(-2.0 * M_PI * kz * z);
                        transfer_func->data[idx] = decay;
                    }
                }
            }
            break;

        case DIFFRACTION_FRESNEL:
            // 菲涅尔传递函数: H(fx, fy) = exp(-i * π * λ * z * (fx² + fy²))
            for (int i = 0; i < height; i++) {
                for (int j = 0; j < width; j++) {
                    int idx = i * width + j;
                    
                    int fi = (i < height / 2) ? i : i - height;
                    int fj = (j < width / 2) ? j : j - width;
                    
                    double fx = fj * df_x;
                    double fy = fi * df_y;
                    
                    double phase = -M_PI * lambda * z * (fx * fx + fy * fy);
                    transfer_func->data[idx] = cosf(phase) + I * sinf(phase);
                }
            }
            break;

        case DIFFRACTION_FRAUNHOFER:
            // 夫琅禾费近似（简化为FFT）
            for (int i = 0; i < height * width; i++) {
                transfer_func->data[i] = 1.0f + 0.0f * I;
            }
            break;

        default:
            return DIFFRACTION_ERROR_INVALID_PARAMS;
    }

    return DIFFRACTION_SUCCESS;
}

int diffraction_propagate_with_transfer_function(
    const ComplexImage *input,
    const ComplexImage *transfer_func,
    ComplexImage *output)
{
    if (!input || !transfer_func || !output) {
        return DIFFRACTION_ERROR_NULL_POINTER;
    }

    if (input->width != transfer_func->width || 
        input->height != transfer_func->height ||
        input->width != output->width || 
        input->height != output->height) {
        return DIFFRACTION_ERROR_INVALID_PARAMS;
    }

    int width = input->width;
    int height = input->height;

    // 创建FFT缓冲区
    ComplexF *fft_buffer = (ComplexF*)fftwf_malloc(width * height * sizeof(ComplexF));
    if (!fft_buffer) {
        return DIFFRACTION_ERROR_MEMORY_ALLOCATION;
    }

    // 复制输入数据
    ComplexImage *temp_input = complex_image_clone(input);
    if (!temp_input) {
        fftwf_free(fft_buffer);
        return DIFFRACTION_ERROR_MEMORY_ALLOCATION;
    }

    // FFT shift
    fftshift_complex(temp_input->data, width, height);

    // 正向FFT
    fftwf_plan plan_forward = fftwf_plan_dft_2d(
        height, width,
        (fftwf_complex*)temp_input->data,
        (fftwf_complex*)fft_buffer,
        FFTW_FORWARD, FFTW_ESTIMATE
    );

    if (!plan_forward) {
        fftwf_free(fft_buffer);
        complex_image_destroy(temp_input);
        return DIFFRACTION_ERROR_FFT_FAILED;
    }

    fftwf_execute(plan_forward);

    // 乘以传递函数
    for (int i = 0; i < width * height; i++) {
        fft_buffer[i] *= transfer_func->data[i];
    }

    // 反向FFT
    fftwf_plan plan_backward = fftwf_plan_dft_2d(
        height, width,
        (fftwf_complex*)fft_buffer,
        (fftwf_complex*)output->data,
        FFTW_BACKWARD, FFTW_ESTIMATE
    );

    if (!plan_backward) {
        fftwf_destroy_plan(plan_forward);
        fftwf_free(fft_buffer);
        complex_image_destroy(temp_input);
        return DIFFRACTION_ERROR_FFT_FAILED;
    }

    fftwf_execute(plan_backward);

    // IFFT shift
    fftshift_complex(output->data, width, height);

    // 归一化
    float norm = 1.0f / (width * height);
    for (int i = 0; i < width * height; i++) {
        output->data[i] *= norm;
    }

    // 清理
    fftwf_destroy_plan(plan_forward);
    fftwf_destroy_plan(plan_backward);
    fftwf_free(fft_buffer);
    complex_image_destroy(temp_input);

    return DIFFRACTION_SUCCESS;
}
// ============================================================================
// 菲涅尔衍射实现（近场衍射）
// ============================================================================

static int fresnel_propagate(
    const ComplexImage *input,
    const DiffractionParams *params,
    ComplexImage *output,
    bool forward)
{
    if (!input || !params || !output) {
        return DIFFRACTION_ERROR_NULL_POINTER;
    }

    double lambda = params->wavelengths[0].wavelength;
    double z = forward ? params->propagation_distance : -params->propagation_distance;
    double dx = params->pixel_size;
    int width = params->width;
    int height = params->height;

    // 应用零填充
    ComplexImage *padded_input = NULL;
    if (params->use_padding) {
        padded_input = apply_padding(input, params->padding_factor);
        if (!padded_input) return DIFFRACTION_ERROR_MEMORY_ALLOCATION;
    } else {
        padded_input = complex_image_clone(input);
        if (!padded_input) return DIFFRACTION_ERROR_MEMORY_ALLOCATION;
    }
    
    int padded_width = padded_input->width;
    int padded_height = padded_input->height;

    // 创建FFT缓冲区
    ComplexF *fft_buffer = (ComplexF*)fftwf_malloc(padded_width * padded_height * sizeof(ComplexF));
    if (!fft_buffer) {
        complex_image_destroy(padded_input);
        return DIFFRACTION_ERROR_MEMORY_ALLOCATION;
    }

    // 创建FFT计划
    fftwf_plan plan_forward = fftwf_plan_dft_2d(
        padded_height, padded_width,
        (fftwf_complex*)padded_input->data,
        (fftwf_complex*)fft_buffer,
        FFTW_FORWARD, FFTW_ESTIMATE
    );

    if (!plan_forward) {
        fftwf_free(fft_buffer);
        complex_image_destroy(padded_input);
        return DIFFRACTION_ERROR_FFT_FAILED;
    }

    // 步骤1: 乘以输入平面的二次相位因子
    double k = 2.0 * M_PI / lambda;
    for (int i = 0; i < padded_height; i++) {
        for (int j = 0; j < padded_width; j++) {
            int idx = i * padded_width + j;
            double x = (j - padded_width / 2.0) * dx;
            double y = (i - padded_height / 2.0) * dx;
            double phase = k / (2.0 * z) * (x * x + y * y);
            ComplexF factor = cosf(phase) + I * sinf(phase);
            padded_input->data[idx] *= factor;
        }
    }

    // FFT shift (将零频移到中心)
    fftshift_complex(padded_input->data, padded_width, padded_height);

    // 步骤2: FFT
    fftwf_execute(plan_forward);

    // 步骤3: 乘以传递函数
    double df_x = 1.0 / (padded_width * dx);
    double df_y = 1.0 / (padded_height * dx);
    
    for (int i = 0; i < padded_height; i++) {
        for (int j = 0; j < padded_width; j++) {
            int idx = i * padded_width + j;
            
            // FFT频率坐标
            int fi = (i < padded_height / 2) ? i : i - padded_height;
            int fj = (j < padded_width / 2) ? j : j - padded_width;
            
            double fx = fj * df_x;
            double fy = fi * df_y;
            
            // 传递函数 H(fx, fy) = exp(-i * pi * lambda * z * (fx^2 + fy^2))
            double phase = -M_PI * lambda * z * (fx * fx + fy * fy);
            ComplexF H = cosf(phase) + I * sinf(phase);
            
            fft_buffer[idx] *= H;
        }
    }

    // 步骤4: IFFT
    ComplexF *result_buffer = (ComplexF*)fftwf_malloc(padded_width * padded_height * sizeof(ComplexF));
    if (!result_buffer) {
        fftwf_destroy_plan(plan_forward);
        fftwf_free(fft_buffer);
        complex_image_destroy(padded_input);
        return DIFFRACTION_ERROR_MEMORY_ALLOCATION;
    }

    fftwf_plan plan_backward = fftwf_plan_dft_2d(
        padded_height, padded_width,
        (fftwf_complex*)fft_buffer,
        (fftwf_complex*)result_buffer,
        FFTW_BACKWARD, FFTW_ESTIMATE
    );

    if (!plan_backward) {
        fftwf_destroy_plan(plan_forward);
        fftwf_free(fft_buffer);
        fftwf_free(result_buffer);
        complex_image_destroy(padded_input);
        return DIFFRACTION_ERROR_FFT_FAILED;
    }
    
    fftwf_execute(plan_backward);

    // IFFT shift
    fftshift_complex(result_buffer, padded_width, padded_height);

    // 归一化
    float norm = 1.0f / (padded_width * padded_height);
    for (int i = 0; i < padded_height * padded_width; i++) {
        result_buffer[i] *= norm;
    }

    // 步骤5: 乘以输出平面的二次相位因子
    for (int i = 0; i < padded_height; i++) {
        for (int j = 0; j < padded_width; j++) {
            int idx = i * padded_width + j;
            double x = (j - padded_width / 2.0) * dx;
            double y = (i - padded_height / 2.0) * dx;
            double phase = k / (2.0 * z) * (x * x + y * y);
            ComplexF factor = cosf(phase) + I * sinf(phase);
            result_buffer[idx] *= factor;
        }
    }

    // 移除填充
    if (params->use_padding) {
        ComplexImage temp = {result_buffer, padded_width, padded_height, padded_width};
        ComplexImage *unpadded = remove_padding(&temp, width, height);
        if (!unpadded) {
            fftwf_destroy_plan(plan_forward);
            fftwf_destroy_plan(plan_backward);
            fftwf_free(fft_buffer);
            fftwf_free(result_buffer);
            complex_image_destroy(padded_input);
            return DIFFRACTION_ERROR_MEMORY_ALLOCATION;
        }
        memcpy(output->data, unpadded->data, width * height * sizeof(ComplexF));
        complex_image_destroy(unpadded);
    } else {
        memcpy(output->data, result_buffer, width * height * sizeof(ComplexF));
    }

    // 清理
    fftwf_destroy_plan(plan_forward);
    fftwf_destroy_plan(plan_backward);
    fftwf_free(fft_buffer);
    fftwf_free(result_buffer);
    complex_image_destroy(padded_input);

    return DIFFRACTION_SUCCESS;
}

// ============================================================================
// 菲涅尔衍射的卷积实现（备选方法）
// ============================================================================

int diffraction_fresnel_convolution(
    const ComplexImage *input,
    const DiffractionParams *params,
    ComplexImage *output,
    bool forward)
{
    if (!input || !params || !output) {
        return DIFFRACTION_ERROR_NULL_POINTER;
    }

    double lambda = params->wavelengths[0].wavelength;
    double z = forward ? params->propagation_distance : -params->propagation_distance;
    double dx = params->pixel_size;
    int width = params->width;
    int height = params->height;

    // 应用零填充
    ComplexImage *padded_input = NULL;
    if (params->use_padding) {
        padded_input = apply_padding(input, params->padding_factor);
    } else {
        padded_input = complex_image_clone(input);
    }
    
    if (!padded_input) {
        return DIFFRACTION_ERROR_MEMORY_ALLOCATION;
    }
    
    int padded_width = padded_input->width;
    int padded_height = padded_input->height;

    // 创建菲涅尔核
    ComplexImage *kernel = complex_image_create(padded_width, padded_height);
    if (!kernel) {
        complex_image_destroy(padded_input);
        return DIFFRACTION_ERROR_MEMORY_ALLOCATION;
    }

    // 计算菲涅尔核: h(x,y) = exp(i*k*z) / (i*lambda*z) * exp(i*k/(2*z)*(x^2+y^2))
    double k = 2.0 * M_PI / lambda;
    ComplexF prefactor = cexpf(I * k * z) / (I * lambda * z);
    
    for (int i = 0; i < padded_height; i++) {
        for (int j = 0; j < padded_width; j++) {
            int idx = i * padded_width + j;
            double x = (j - padded_width / 2.0) * dx;
            double y = (i - padded_height / 2.0) * dx;
            double phase = k / (2.0 * z) * (x * x + y * y);
            kernel->data[idx] = prefactor * (cosf(phase) + I * sinf(phase));
        }
    }

    // FFT shift输入和核
    fftshift_complex(padded_input->data, padded_width, padded_height);
    fftshift_complex(kernel->data, padded_width, padded_height);

    // 创建FFT缓冲区
    ComplexF *input_fft = (ComplexF*)fftwf_malloc(padded_width * padded_height * sizeof(ComplexF));
    ComplexF *kernel_fft = (ComplexF*)fftwf_malloc(padded_width * padded_height * sizeof(ComplexF));
    
    if (!input_fft || !kernel_fft) {
        if (input_fft) fftwf_free(input_fft);
        if (kernel_fft) fftwf_free(kernel_fft);
        complex_image_destroy(padded_input);
        complex_image_destroy(kernel);
        return DIFFRACTION_ERROR_MEMORY_ALLOCATION;
    }

    // 创建FFT计划
    fftwf_plan plan_input = fftwf_plan_dft_2d(
        padded_height, padded_width,
        (fftwf_complex*)padded_input->data,
        (fftwf_complex*)input_fft,
        FFTW_FORWARD, FFTW_ESTIMATE
    );

    fftwf_plan plan_kernel = fftwf_plan_dft_2d(
        padded_height, padded_width,
        (fftwf_complex*)kernel->data,
        (fftwf_complex*)kernel_fft,
        FFTW_FORWARD, FFTW_ESTIMATE
    );

    if (!plan_input || !plan_kernel) {
        if (plan_input) fftwf_destroy_plan(plan_input);
        if (plan_kernel) fftwf_destroy_plan(plan_kernel);
        fftwf_free(input_fft);
        fftwf_free(kernel_fft);
        complex_image_destroy(padded_input);
        complex_image_destroy(kernel);
        return DIFFRACTION_ERROR_FFT_FAILED;
    }

    // 执行FFT
    fftwf_execute(plan_input);
    fftwf_execute(plan_kernel);

    // 频域相乘
    for (int i = 0; i < padded_width * padded_height; i++) {
        input_fft[i] *= kernel_fft[i];
    }

    // IFFT
    ComplexF *result_buffer = (ComplexF*)fftwf_malloc(padded_width * padded_height * sizeof(ComplexF));
    if (!result_buffer) {
        fftwf_destroy_plan(plan_input);
        fftwf_destroy_plan(plan_kernel);
        fftwf_free(input_fft);
        fftwf_free(kernel_fft);
        complex_image_destroy(padded_input);
        complex_image_destroy(kernel);
        return DIFFRACTION_ERROR_MEMORY_ALLOCATION;
    }

    fftwf_plan plan_backward = fftwf_plan_dft_2d(
        padded_height, padded_width,
        (fftwf_complex*)input_fft,
        (fftwf_complex*)result_buffer,
        FFTW_BACKWARD, FFTW_ESTIMATE
    );

    if (!plan_backward) {
        fftwf_destroy_plan(plan_input);
        fftwf_destroy_plan(plan_kernel);
        fftwf_free(input_fft);
        fftwf_free(kernel_fft);
        fftwf_free(result_buffer);
        complex_image_destroy(padded_input);
        complex_image_destroy(kernel);
        return DIFFRACTION_ERROR_FFT_FAILED;
    }

    fftwf_execute(plan_backward);

    // IFFT shift
    fftshift_complex(result_buffer, padded_width, padded_height);

    // 归一化
    float norm = dx * dx / (padded_width * padded_height);
    for (int i = 0; i < padded_width * padded_height; i++) {
        result_buffer[i] *= norm;
    }

    // 移除填充
    if (params->use_padding) {
        ComplexImage temp = {result_buffer, padded_width, padded_height, padded_width};
        ComplexImage *unpadded = remove_padding(&temp, width, height);
        if (!unpadded) {
            fftwf_destroy_plan(plan_input);
            fftwf_destroy_plan(plan_kernel);
            fftwf_destroy_plan(plan_backward);
            fftwf_free(input_fft);
            fftwf_free(kernel_fft);
            fftwf_free(result_buffer);
            complex_image_destroy(padded_input);
            complex_image_destroy(kernel);
            return DIFFRACTION_ERROR_MEMORY_ALLOCATION;
        }
        memcpy(output->data, unpadded->data, width * height * sizeof(ComplexF));
        complex_image_destroy(unpadded);
    } else {
        memcpy(output->data, result_buffer, width * height * sizeof(ComplexF));
    }

    // 清理
    fftwf_destroy_plan(plan_input);
    fftwf_destroy_plan(plan_kernel);
    fftwf_destroy_plan(plan_backward);
    fftwf_free(input_fft);
    fftwf_free(kernel_fft);
    fftwf_free(result_buffer);
    complex_image_destroy(padded_input);
    complex_image_destroy(kernel);

    return DIFFRACTION_SUCCESS;
}
// ============================================================================
// 夫琅禾费衍射实现（远场衍射）
// ============================================================================

static int fraunhofer_propagate(
    const ComplexImage *input,
    const DiffractionParams *params,
    ComplexImage *output,
    bool forward)
{
    if (!input || !params || !output) {
        return DIFFRACTION_ERROR_NULL_POINTER;
    }

    double lambda = params->wavelengths[0].wavelength;
    double z = forward ? params->propagation_distance : -params->propagation_distance;
    double dx = params->pixel_size;
    int width = params->width;
    int height = params->height;

    // 应用零填充
    ComplexImage *padded_input = NULL;
    if (params->use_padding) {
        padded_input = apply_padding(input, params->padding_factor);
    } else {
        padded_input = complex_image_clone(input);
    }
    
    if (!padded_input) {
        return DIFFRACTION_ERROR_MEMORY_ALLOCATION;
    }
    
    int padded_width = padded_input->width;
    int padded_height = padded_input->height;

    // 创建FFT缓冲区
    ComplexF *fft_buffer = (ComplexF*)fftwf_malloc(padded_width * padded_height * sizeof(ComplexF));
    if (!fft_buffer) {
        complex_image_destroy(padded_input);
        return DIFFRACTION_ERROR_MEMORY_ALLOCATION;
    }

    double k = 2.0 * M_PI / lambda;

    if (forward) {
        // 正向传播：物平面 -> 像平面
        
        // 步骤1: 乘以二次相位因子
        for (int i = 0; i < padded_height; i++) {
            for (int j = 0; j < padded_width; j++) {
                int idx = i * padded_width + j;
                double x = (j - padded_width / 2.0) * dx;
                double y = (i - padded_height / 2.0) * dx;
                double phase = k / (2.0 * z) * (x * x + y * y);
                ComplexF factor = cosf(phase) + I * sinf(phase);
                padded_input->data[idx] *= factor;
            }
        }

        // FFT shift
        fftshift_complex(padded_input->data, padded_width, padded_height);

        // 步骤2: FFT
        fftwf_plan plan = fftwf_plan_dft_2d(
            padded_height, padded_width,
            (fftwf_complex*)padded_input->data,
            (fftwf_complex*)fft_buffer,
            FFTW_FORWARD, FFTW_ESTIMATE
        );

        if (!plan) {
            fftwf_free(fft_buffer);
            complex_image_destroy(padded_input);
            return DIFFRACTION_ERROR_FFT_FAILED;
        }

        fftwf_execute(plan);
        fftwf_destroy_plan(plan);

        // IFFT shift
        fftshift_complex(fft_buffer, padded_width, padded_height);

        // 步骤3: 乘以常数因子和二次相位
        ComplexF prefactor = cexpf(I * k * z) / (I * lambda * z);
        double df_x = 1.0 / (padded_width * dx);
        double df_y = 1.0 / (padded_height * dx);

        for (int i = 0; i < padded_height; i++) {
            for (int j = 0; j < padded_width; j++) {
                int idx = i * padded_width + j;
                
                double fx = (j - padded_width / 2.0) * df_x;
                double fy = (i - padded_height / 2.0) * df_y;
                
                double x_out = fx * lambda * z;
                double y_out = fy * lambda * z;
                
                double phase = k / (2.0 * z) * (x_out * x_out + y_out * y_out);
                ComplexF factor = prefactor * (cosf(phase) + I * sinf(phase));
                
                fft_buffer[idx] *= factor * dx * dx;
            }
        }

    } else {
        // 反向传播：像平面 -> 物平面
        
        // 步骤1: 除以常数因子和二次相位
        ComplexF prefactor = 1.0f / (cexpf(I * k * z) / (I * lambda * z));
        double df_x = 1.0 / (padded_width * dx);
        double df_y = 1.0 / (padded_height * dx);

        for (int i = 0; i < padded_height; i++) {
            for (int j = 0; j < padded_width; j++) {
                int idx = i * padded_width + j;
                
                double fx = (j - padded_width / 2.0) * df_x;
                double fy = (i - padded_height / 2.0) * df_y;
                
                double x_in = fx * lambda * z;
                double y_in = fy * lambda * z;
                
                double phase = -k / (2.0 * z) * (x_in * x_in + y_in * y_in);
                ComplexF factor = prefactor * (cosf(phase) + I * sinf(phase));
                
                padded_input->data[idx] *= factor / (dx * dx);
            }
        }

        // FFT shift
        fftshift_complex(padded_input->data, padded_width, padded_height);

        // 步骤2: IFFT
        fftwf_plan plan = fftwf_plan_dft_2d(
            padded_height, padded_width,
            (fftwf_complex*)padded_input->data,
            (fftwf_complex*)fft_buffer,
            FFTW_BACKWARD, FFTW_ESTIMATE
        );

        if (!plan) {
            fftwf_free(fft_buffer);
            complex_image_destroy(padded_input);
            return DIFFRACTION_ERROR_FFT_FAILED;
        }

        fftwf_execute(plan);
        fftwf_destroy_plan(plan);

        // IFFT shift
        fftshift_complex(fft_buffer, padded_width, padded_height);

        // 归一化
        float norm = 1.0f / (padded_width * padded_height);
        for (int i = 0; i < padded_width * padded_height; i++) {
            fft_buffer[i] *= norm;
        }

        // 步骤3: 除以二次相位因子
        for (int i = 0; i < padded_height; i++) {
            for (int j = 0; j < padded_width; j++) {
                int idx = i * padded_width + j;
                double x = (j - padded_width / 2.0) * dx;
                double y = (i - padded_height / 2.0) * dx;
                double phase = -k / (2.0 * z) * (x * x + y * y);
                ComplexF factor = cosf(phase) + I * sinf(phase);
                fft_buffer[idx] *= factor;
            }
        }
    }

    // 移除填充
    if (params->use_padding) {
        ComplexImage temp = {fft_buffer, padded_width, padded_height, padded_width};
        ComplexImage *unpadded = remove_padding(&temp, width, height);
        if (!unpadded) {
            fftwf_free(fft_buffer);
            complex_image_destroy(padded_input);
            return DIFFRACTION_ERROR_MEMORY_ALLOCATION;
        }
        memcpy(output->data, unpadded->data, width * height * sizeof(ComplexF));
        complex_image_destroy(unpadded);
    } else {
        memcpy(output->data, fft_buffer, width * height * sizeof(ComplexF));
    }

    // 清理
    fftwf_free(fft_buffer);
    complex_image_destroy(padded_input);

    return DIFFRACTION_SUCCESS;
}

// ============================================================================
// 角谱法实现（最精确的方法）
// ============================================================================

static int angular_spectrum_propagate(
    const ComplexImage *input,
    const DiffractionParams *params,
    ComplexImage *output,
    bool forward)
{
    if (!input || !params || !output) {
        return DIFFRACTION_ERROR_NULL_POINTER;
    }

    double lambda = params->wavelengths[0].wavelength;
    double z = forward ? params->propagation_distance : -params->propagation_distance;
    double dx = params->pixel_size;
    int width = params->width;
    int height = params->height;

    // 应用零填充
    ComplexImage *padded_input = NULL;
    if (params->use_padding) {
        padded_input = apply_padding(input, params->padding_factor);
    } else {
        padded_input = complex_image_clone(input);
    }
    
    if (!padded_input) {
        return DIFFRACTION_ERROR_MEMORY_ALLOCATION;
    }
    
    int padded_width = padded_input->width;
    int padded_height = padded_input->height;

    // 创建FFT缓冲区
    ComplexF *fft_buffer = (ComplexF*)fftwf_malloc(padded_width * padded_height * sizeof(ComplexF));
    if (!fft_buffer) {
        complex_image_destroy(padded_input);
        return DIFFRACTION_ERROR_MEMORY_ALLOCATION;
    }

    // FFT shift
    fftshift_complex(padded_input->data, padded_width, padded_height);

    // 步骤1: 正向FFT
    fftwf_plan plan_forward = fftwf_plan_dft_2d(
        padded_height, padded_width,
        (fftwf_complex*)padded_input->data,
        (fftwf_complex*)fft_buffer,
        FFTW_FORWARD, FFTW_ESTIMATE
    );

    if (!plan_forward) {
        fftwf_free(fft_buffer);
        complex_image_destroy(padded_input);
        return DIFFRACTION_ERROR_FFT_FAILED;
    }

    fftwf_execute(plan_forward);

    // 步骤2: 乘以角谱传递函数
    double df_x = 1.0 / (padded_width * dx);
    double df_y = 1.0 / (padded_height * dx);
    double k = 2.0 * M_PI / lambda;
    
    for (int i = 0; i < padded_height; i++) {
        for (int j = 0; j < padded_width; j++) {
            int idx = i * padded_width + j;
            
            // FFT频率坐标
            int fi = (i < padded_height / 2) ? i : i - padded_height;
            int fj = (j < padded_width / 2) ? j : j - padded_width;
            
            double fx = fj * df_x;
            double fy = fi * df_y;
            
            double f_squared = fx * fx + fy * fy;
            double lambda_squared = lambda * lambda;
            
            ComplexF H;
            if (f_squared < 1.0 / lambda_squared) {
                // 传播波：H(fx, fy) = exp(i * 2π * z * sqrt(1/λ² - fx² - fy²))
                double kz = sqrt(1.0 / lambda_squared - f_squared);
                double phase = 2.0 * M_PI * kz * z;
                H = cosf(phase) + I * sinf(phase);
            } else {
                // 倏逝波（衰减）：H(fx, fy) = exp(-2π * z * sqrt(fx² + fy² - 1/λ²))
                double kz = sqrt(f_squared - 1.0 / lambda_squared);
                double decay = exp(-2.0 * M_PI * kz * fabs(z));
                H = decay;
            }
            
            fft_buffer[idx] *= H;
        }
    }

    // 步骤3: 反向FFT
    ComplexF *result_buffer = (ComplexF*)fftwf_malloc(padded_width * padded_height * sizeof(ComplexF));
    if (!result_buffer) {
        fftwf_destroy_plan(plan_forward);
        fftwf_free(fft_buffer);
        complex_image_destroy(padded_input);
        return DIFFRACTION_ERROR_MEMORY_ALLOCATION;
    }

    fftwf_plan plan_backward = fftwf_plan_dft_2d(
        padded_height, padded_width,
        (fftwf_complex*)fft_buffer,
        (fftwf_complex*)result_buffer,
        FFTW_BACKWARD, FFTW_ESTIMATE
    );

    if (!plan_backward) {
        fftwf_destroy_plan(plan_forward);
        fftwf_free(fft_buffer);
        fftwf_free(result_buffer);
        complex_image_destroy(padded_input);
        return DIFFRACTION_ERROR_FFT_FAILED;
    }

    fftwf_execute(plan_backward);

    // IFFT shift
    fftshift_complex(result_buffer, padded_width, padded_height);

    // 归一化
    float norm = 1.0f / (padded_width * padded_height);
    for (int i = 0; i < padded_width * padded_height; i++) {
        result_buffer[i] *= norm;
    }

    // 移除填充
    if (params->use_padding) {
        ComplexImage temp = {result_buffer, padded_width, padded_height, padded_width};
        ComplexImage *unpadded = remove_padding(&temp, width, height);
        if (!unpadded) {
            fftwf_destroy_plan(plan_forward);
            fftwf_destroy_plan(plan_backward);
            fftwf_free(fft_buffer);
            fftwf_free(result_buffer);
            complex_image_destroy(padded_input);
            return DIFFRACTION_ERROR_MEMORY_ALLOCATION;
        }
        memcpy(output->data, unpadded->data, width * height * sizeof(ComplexF));
        complex_image_destroy(unpadded);
    } else {
        memcpy(output->data, result_buffer, width * height * sizeof(ComplexF));
    }

    // 清理
    fftwf_destroy_plan(plan_forward);
    fftwf_destroy_plan(plan_backward);
    fftwf_free(fft_buffer);
    fftwf_free(result_buffer);
    complex_image_destroy(padded_input);

    return DIFFRACTION_SUCCESS;
}
// ============================================================================
// 辅助函数实现
// ============================================================================

// FFT shift实现（将零频移到中心）
static void fftshift_complex(ComplexF *data, int width, int height)
{
    int half_width = width / 2;
    int half_height = height / 2;
    
    // 交换四个象限
    for (int i = 0; i < half_height; i++) {
        for (int j = 0; j < half_width; j++) {
            // 第一象限 <-> 第三象限
            ComplexF temp = data[i * width + j];
            data[i * width + j] = data[(i + half_height) * width + (j + half_width)];
            data[(i + half_height) * width + (j + half_width)] = temp;
            
            // 第二象限 <-> 第四象限
            temp = data[i * width + (j + half_width)];
            data[i * width + (j + half_width)] = data[(i + half_height) * width + j];
            data[(i + half_height) * width + j] = temp;
        }
    }
}

// 零填充实现
static ComplexImage* apply_padding(const ComplexImage *input, int padding_factor)
{
    if (!input || padding_factor < 1) {
        return NULL;
    }

    int new_width = input->width * padding_factor;
    int new_height = input->height * padding_factor;
    
    ComplexImage *padded = complex_image_create(new_width, new_height);
    if (!padded) {
        return NULL;
    }

    // 初始化为零
    memset(padded->data, 0, new_width * new_height * sizeof(ComplexF));

    // 计算偏移量（居中）
    int offset_x = (new_width - input->width) / 2;
    int offset_y = (new_height - input->height) / 2;

    // 复制原始数据到中心
    for (int i = 0; i < input->height; i++) {
        for (int j = 0; j < input->width; j++) {
            int src_idx = i * input->width + j;
            int dst_idx = (i + offset_y) * new_width + (j + offset_x);
            padded->data[dst_idx] = input->data[src_idx];
        }
    }

    return padded;
}

// 移除填充实现
static ComplexImage* remove_padding(const ComplexImage *padded, int original_width, int original_height)
{
    if (!padded) {
        return NULL;
    }

    ComplexImage *output = complex_image_create(original_width, original_height);
    if (!output) {
        return NULL;
    }

    // 计算偏移量
    int offset_x = (padded->width - original_width) / 2;
    int offset_y = (padded->height - original_height) / 2;

    // 提取中心区域
    for (int i = 0; i < original_height; i++) {
        for (int j = 0; j < original_width; j++) {
            int src_idx = (i + offset_y) * padded->width + (j + offset_x);
            int dst_idx = i * original_width + j;
            output->data[dst_idx] = padded->data[src_idx];
        }
    }

    return output;
}

// ============================================================================
// 图像转换函数
// ============================================================================

int diffraction_complex_to_intensity(
    const ComplexImage *complex_img,
    RealImage *intensity_img)
{
    if (!complex_img || !intensity_img) {
        return DIFFRACTION_ERROR_NULL_POINTER;
    }

    if (complex_img->width != intensity_img->width ||
        complex_img->height != intensity_img->height) {
        return DIFFRACTION_ERROR_INVALID_PARAMS;
    }

    for (int i = 0; i < complex_img->height; i++) {
        for (int j = 0; j < complex_img->width; j++) {
            int idx = i * complex_img->width + j;
            ComplexF val = complex_img->data[idx];
            float real_part = crealf(val);
            float imag_part = cimagf(val);
            float intensity = real_part * real_part + imag_part * imag_part;
            
            if (intensity_img->channels == 1) {
                intensity_img->data[idx] = intensity;
            } else {
                // 对于多通道图像，复制到所有通道
                for (int c = 0; c < intensity_img->channels; c++) {
                    intensity_img->data[i * intensity_img->stride + j * intensity_img->channels + c] = intensity;
                }
            }
        }
    }

    return DIFFRACTION_SUCCESS;
}

int diffraction_complex_to_phase(
    const ComplexImage *complex_img,
    RealImage *phase_img)
{
    if (!complex_img || !phase_img) {
        return DIFFRACTION_ERROR_NULL_POINTER;
    }

    if (complex_img->width != phase_img->width ||
        complex_img->height != phase_img->height) {
        return DIFFRACTION_ERROR_INVALID_PARAMS;
    }

    for (int i = 0; i < complex_img->height; i++) {
        for (int j = 0; j < complex_img->width; j++) {
            int idx = i * complex_img->width + j;
            ComplexF val = complex_img->data[idx];
            float phase = atan2f(cimagf(val), crealf(val));
            
            if (phase_img->channels == 1) {
                phase_img->data[idx] = phase;
            } else {
                // 对于多通道图像，复制到所有通道
                for (int c = 0; c < phase_img->channels; c++) {
                    phase_img->data[i * phase_img->stride + j * phase_img->channels + c] = phase;
                }
            }
        }
    }

    return DIFFRACTION_SUCCESS;
}

int diffraction_complex_to_amplitude(
    const ComplexImage *complex_img,
    RealImage *amplitude_img)
{
    if (!complex_img || !amplitude_img) {
        return DIFFRACTION_ERROR_NULL_POINTER;
    }

    if (complex_img->width != amplitude_img->width ||
        complex_img->height != amplitude_img->height) {
        return DIFFRACTION_ERROR_INVALID_PARAMS;
    }

    for (int i = 0; i < complex_img->height; i++) {
        for (int j = 0; j < complex_img->width; j++) {
            int idx = i * complex_img->width + j;
            ComplexF val = complex_img->data[idx];
            float amplitude = cabsf(val);
            
            if (amplitude_img->channels == 1) {
                amplitude_img->data[idx] = amplitude;
            } else {
                // 对于多通道图像，复制到所有通道
                for (int c = 0; c < amplitude_img->channels; c++) {
                    amplitude_img->data[i * amplitude_img->stride + j * amplitude_img->channels + c] = amplitude;
                }
            }
        }
    }

    return DIFFRACTION_SUCCESS;
}

int diffraction_real_to_complex(
    const RealImage *real_img,
    ComplexImage *complex_img,
    bool is_amplitude)
{
    if (!real_img || !complex_img) {
        return DIFFRACTION_ERROR_NULL_POINTER;
    }

    if (real_img->width != complex_img->width ||
        real_img->height != complex_img->height) {
        return DIFFRACTION_ERROR_INVALID_PARAMS;
    }

    for (int i = 0; i < real_img->height; i++) {
        for (int j = 0; j < real_img->width; j++) {
            int idx = i * real_img->width + j;
            float value;
            
            if (real_img->channels == 1) {
                value = real_img->data[idx];
            } else {
                // 对于多通道图像，取平均值或第一个通道
                value = real_img->data[i * real_img->stride + j * real_img->channels];
            }
            
            if (is_amplitude) {
                // 振幅图像：复数 = 振幅 * exp(i*0) = 振幅
                complex_img->data[idx] = value + 0.0f * I;
            } else {
                // 强度图像：复数 = sqrt(强度) * exp(i*0)
                complex_img->data[idx] = sqrtf(fmaxf(value, 0.0f)) + 0.0f * I;
            }
        }
    }

    return DIFFRACTION_SUCCESS;
}

// ============================================================================
// 验证和诊断函数
// ============================================================================

int diffraction_validate_params(const DiffractionParams *params)
{
    if (!params) {
        return DIFFRACTION_ERROR_NULL_POINTER;
    }

    // 检查尺寸
    if (params->width <= 0 || params->height <= 0) {
        return DIFFRACTION_ERROR_INVALID_PARAMS;
    }

    // 检查像素尺寸
    if (params->pixel_size <= 0.0) {
        return DIFFRACTION_ERROR_INVALID_PARAMS;
    }

    // 检查传播距离
    if (params->propagation_distance == 0.0) {
        return DIFFRACTION_ERROR_INVALID_PARAMS;
    }

    // 检查波长
    if (params->num_wavelengths <= 0 || !params->wavelengths) {
        return DIFFRACTION_ERROR_INVALID_WAVELENGTH;
    }

    for (int i = 0; i < params->num_wavelengths; i++) {
        if (params->wavelengths[i].wavelength <= 0.0) {
            return DIFFRACTION_ERROR_INVALID_WAVELENGTH;
        }
        if (params->wavelengths[i].weight < 0.0) {
            return DIFFRACTION_ERROR_INVALID_PARAMS;
        }
    }

    // 检查衍射类型
    if (params->type != DIFFRACTION_FRESNEL &&
        params->type != DIFFRACTION_FRAUNHOFER &&
        params->type != DIFFRACTION_ANGULAR_SPECTRUM) {
        return DIFFRACTION_ERROR_INVALID_PARAMS;
    }

    // 检查填充因子
    if (params->use_padding && params->padding_factor < 1) {
        return DIFFRACTION_ERROR_INVALID_PARAMS;
    }

    return DIFFRACTION_SUCCESS;
}

int diffraction_compute_fresnel_number(
    const DiffractionParams *params,
    double *fresnel_number)
{
    if (!params || !fresnel_number) {
        return DIFFRACTION_ERROR_NULL_POINTER;
    }

    double lambda = params->wavelengths[0].wavelength;
    double z = fabs(params->propagation_distance);
    double a = fmin(params->width, params->height) * params->pixel_size / 2.0;

    // 菲涅尔数: F = a² / (λ * z)
    *fresnel_number = (a * a) / (lambda * z);

    return DIFFRACTION_SUCCESS;
}

int diffraction_suggest_method(
    const DiffractionParams *params,
    DiffractionType *suggested_type)
{
    if (!params || !suggested_type) {
        return DIFFRACTION_ERROR_NULL_POINTER;
    }

    double fresnel_number;
    int result = diffraction_compute_fresnel_number(params, &fresnel_number);
    if (result != DIFFRACTION_SUCCESS) {
        return result;
    }

    // 根据菲涅尔数建议方法
    if (fresnel_number > 10.0) {
        // 近场：使用菲涅尔衍射
        *suggested_type = DIFFRACTION_FRESNEL;
    } else if (fresnel_number < 0.1) {
        // 远场：使用夫琅禾费衍射
        *suggested_type = DIFFRACTION_FRAUNHOFER;
    } else {
        // 中间区域：使用角谱法（最精确）
        *suggested_type = DIFFRACTION_ANGULAR_SPECTRUM;
    }

    return DIFFRACTION_SUCCESS;
}

const char* diffraction_get_error_string(int error_code)
{
    switch (error_code) {
        case DIFFRACTION_SUCCESS:
            return "Success";
        case DIFFRACTION_ERROR_NULL_POINTER:
            return "Null pointer error";
        case DIFFRACTION_ERROR_INVALID_PARAMS:
            return "Invalid parameters";
        case DIFFRACTION_ERROR_MEMORY_ALLOCATION:
            return "Memory allocation failed";
        case DIFFRACTION_ERROR_FFT_FAILED:
            return "FFT operation failed";
        case DIFFRACTION_ERROR_INVALID_WAVELENGTH:
            return "Invalid wavelength";
        case DIFFRACTION_ERROR_INVALID_IMAGE_SIZE:
            return "Invalid image size";
        default:
            return "Unknown error";
    }
}

// ============================================================================
// 性能优化函数
// ============================================================================

int diffraction_set_fft_wisdom(const char *wisdom_file)
{
    if (!wisdom_file) {
        return DIFFRACTION_ERROR_NULL_POINTER;
    }

    if (fftwf_import_wisdom_from_filename(wisdom_file)) {
        return DIFFRACTION_SUCCESS;
    } else {
        return DIFFRACTION_ERROR_INVALID_PARAMS;
    }
}

int diffraction_save_fft_wisdom(const char *wisdom_file)
{
    if (!wisdom_file) {
        return DIFFRACTION_ERROR_NULL_POINTER;
    }

    if (fftwf_export_wisdom_to_filename(wisdom_file)) {
        return DIFFRACTION_SUCCESS;
    } else {
        return DIFFRACTION_ERROR_INVALID_PARAMS;
    }
}

void diffraction_cleanup_fft()
{
    fftwf_cleanup();
}

// ============================================================================
// 批处理函数
// ============================================================================

int diffraction_propagate_batch(
    const ComplexImage **inputs,
    int num_images,
    const DiffractionParams *params,
    ComplexImage **outputs)
{
    if (!inputs || !outputs || !params || num_images <= 0) {
        return DIFFRACTION_ERROR_NULL_POINTER;
    }

    for (int i = 0; i < num_images; i++) {
        int result = diffraction_propagate_forward(inputs[i], params, outputs[i]);
        if (result != DIFFRACTION_SUCCESS) {
            return result;
        }
    }

    return DIFFRACTION_SUCCESS;
}

// ============================================================================
// 多步传播函数
// ============================================================================

int diffraction_propagate_multistep(
    const ComplexImage *input,
    const DiffractionParams *params,
    int num_steps,
    ComplexImage **intermediate_results)
{
    if (!input || !params || !intermediate_results || num_steps <= 0) {
        return DIFFRACTION_ERROR_NULL_POINTER;
    }

    // 计算每步的传播距离
    double step_distance = params->propagation_distance / num_steps;
    
    // 创建临时参数
    DiffractionParams step_params = *params;
    step_params.propagation_distance = step_distance;

    // 第一步
    int result = diffraction_propagate_forward(input, &step_params, intermediate_results[0]);
    if (result != DIFFRACTION_SUCCESS) {
        return result;
    }

    // 后续步骤
    for (int i = 1; i < num_steps; i++) {
        result = diffraction_propagate_forward(
            intermediate_results[i-1],
            &step_params,
            intermediate_results[i]
        );
        if (result != DIFFRACTION_SUCCESS) {
            return result;
        }
    }

    return DIFFRACTION_SUCCESS;
}

// ============================================================================
// 统计和分析函数
// ============================================================================

int diffraction_compute_statistics(
    const ComplexImage *image,
    DiffractionStatistics *stats)
{
    if (!image || !stats) {
        return DIFFRACTION_ERROR_NULL_POINTER;
    }

    int size = image->width * image->height;
    
    // 初始化
    stats->mean_intensity = 0.0;
    stats->max_intensity = 0.0;
    stats->min_intensity = FLT_MAX;
    stats->total_power = 0.0;

    // 计算统计量
    for (int i = 0; i < size; i++) {
        ComplexF val = image->data[i];
        float real_part = crealf(val);
        float imag_part = cimagf(val);
        float intensity = real_part * real_part + imag_part * imag_part;
        
        stats->mean_intensity += intensity;
        stats->total_power += intensity;
        
        if (intensity > stats->max_intensity) {
            stats->max_intensity = intensity;
        }
        if (intensity < stats->min_intensity) {
            stats->min_intensity = intensity;
        }
    }

    stats->mean_intensity /= size;

    return DIFFRACTION_SUCCESS;
}

// ============================================================================
// 库版本信息
// ============================================================================

const char* diffraction_get_version(void)
{
    return "1.0.0";
}

void diffraction_get_build_info(char *buffer, size_t buffer_size)
{
    if (!buffer || buffer_size == 0) {
        return;
    }

    snprintf(buffer, buffer_size,
             "Diffraction Library v%s\n"
             "Built: %s %s\n"
             "FFTW version: %s\n"
             "Features: Fresnel, Fraunhofer, Angular Spectrum\n",
             diffraction_get_version(),
             __DATE__, __TIME__,
             fftwf_version);
}


