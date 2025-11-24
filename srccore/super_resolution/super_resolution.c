/**
 * @file super_resolution.c
 * @brief 超分辨率图像重建库实现
 */

#include "super_resolution.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <float.h>

// 如果可用，使用FFTW
#ifdef USE_FFTW
#include <fftw3.h>
#endif

// ============================================================================
// 常量定义
// ============================================================================

#define PI 3.14159265358979323846
#define EPSILON 1e-8f
#define MAX_ERROR_HISTORY 1000

// ============================================================================
// 内部辅助宏
// ============================================================================

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define CLAMP(x, min, max) (MIN(MAX((x), (min)), (max)))
#define SQUARE(x) ((x) * (x))

// ============================================================================
// 图像管理函数
// ============================================================================

Image* image_create(int width, int height, int channels)
{
    if (width <= 0 || height <= 0 || channels <= 0) {
        return NULL;
    }

    Image *image = (Image*)malloc(sizeof(Image));
    if (!image) {
        return NULL;
    }

    image->width = width;
    image->height = height;
    image->channels = channels;

    size_t size = (size_t)width * height * channels;
    image->data = (float*)calloc(size, sizeof(float));
    
    if (!image->data) {
        free(image);
        return NULL;
    }

    return image;
}

Image3D* image_3d_create(int width, int height, int depth, int channels)
{
    if (width <= 0 || height <= 0 || depth <= 0 || channels <= 0) {
        return NULL;
    }

    Image3D *image = (Image3D*)malloc(sizeof(Image3D));
    if (!image) {
        return NULL;
    }

    image->width = width;
    image->height = height;
    image->depth = depth;
    image->channels = channels;

    size_t size = (size_t)width * height * depth * channels;
    image->data = (float*)calloc(size, sizeof(float));
    
    if (!image->data) {
        free(image);
        return NULL;
    }

    return image;
}

void image_destroy(Image *image)
{
    if (image) {
        if (image->data) {
            free(image->data);
        }
        free(image);
    }
}

void image_3d_destroy(Image3D *image)
{
    if (image) {
        if (image->data) {
            free(image->data);
        }
        free(image);
    }
}

Image* image_clone(const Image *src)
{
    if (!src) {
        return NULL;
    }

    Image *dst = image_create(src->width, src->height, src->channels);
    if (!dst) {
        return NULL;
    }

    size_t size = (size_t)src->width * src->height * src->channels;
    memcpy(dst->data, src->data, size * sizeof(float));

    return dst;
}

Image* image_to_grayscale(const Image *image)
{
    if (!image) {
        return NULL;
    }

    if (image->channels == 1) {
        return image_clone(image);
    }

    Image *gray = image_create(image->width, image->height, 1);
    if (!gray) {
        return NULL;
    }

    int size = image->width * image->height;

    if (image->channels == 3) {
        // RGB to grayscale: 0.299*R + 0.587*G + 0.114*B
        for (int i = 0; i < size; i++) {
            int idx = i * 3;
            gray->data[i] = 0.299f * image->data[idx] +
                           0.587f * image->data[idx + 1] +
                           0.114f * image->data[idx + 2];
        }
    } else if (image->channels == 4) {
        // RGBA to grayscale (忽略alpha通道)
        for (int i = 0; i < size; i++) {
            int idx = i * 4;
            gray->data[i] = 0.299f * image->data[idx] +
                           0.587f * image->data[idx + 1] +
                           0.114f * image->data[idx + 2];
        }
    } else {
        // 其他情况：取第一个通道
        for (int i = 0; i < size; i++) {
            gray->data[i] = image->data[i * image->channels];
        }
    }

    return gray;
}

int image_normalize(Image *image)
{
    if (!image) {
        return SR_ERROR_NULL_POINTER;
    }

    size_t size = (size_t)image->width * image->height * image->channels;

    // 找到最小值和最大值
    float min_val = image->data[0];
    float max_val = image->data[0];

    for (size_t i = 1; i < size; i++) {
        if (image->data[i] < min_val) min_val = image->data[i];
        if (image->data[i] > max_val) max_val = image->data[i];
    }

    // 归一化到[0, 1]
    float range = max_val - min_val;
    if (range < EPSILON) {
        // 图像是常数，设为0
        memset(image->data, 0, size * sizeof(float));
        return SR_SUCCESS;
    }

    for (size_t i = 0; i < size; i++) {
        image->data[i] = (image->data[i] - min_val) / range;
    }

    return SR_SUCCESS;
}

Image* image_crop(const Image *image, int x, int y, int width, int height)
{
    if (!image) {
        return NULL;
    }

    if (x < 0 || y < 0 || x + width > image->width || y + height > image->height) {
        return NULL;
    }

    Image *cropped = image_create(width, height, image->channels);
    if (!cropped) {
        return NULL;
    }

    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            int src_idx = ((y + row) * image->width + (x + col)) * image->channels;
            int dst_idx = (row * width + col) * image->channels;
            
            for (int c = 0; c < image->channels; c++) {
                cropped->data[dst_idx + c] = image->data[src_idx + c];
            }
        }
    }

    return cropped;
}

// 边界处理辅助函数
static inline int handle_boundary(int coord, int size, BoundaryMode mode)
{
    switch (mode) {
        case BOUNDARY_ZERO:
            return (coord < 0 || coord >= size) ? -1 : coord;
            
        case BOUNDARY_REPLICATE:
            return CLAMP(coord, 0, size - 1);
            
        case BOUNDARY_REFLECT:
            if (coord < 0) {
                coord = -coord - 1;
            }
            if (coord >= size) {
                coord = 2 * size - coord - 1;
            }
            return CLAMP(coord, 0, size - 1);
            
        case BOUNDARY_WRAP:
            coord = coord % size;
            if (coord < 0) coord += size;
            return coord;
            
        default:
            return CLAMP(coord, 0, size - 1);
    }
}

Image* image_pad(const Image *image, int top, int bottom, int left, int right,
                 BoundaryMode mode)
{
    if (!image) {
        return NULL;
    }

    int new_width = image->width + left + right;
    int new_height = image->height + top + bottom;

    Image *padded = image_create(new_width, new_height, image->channels);
    if (!padded) {
        return NULL;
    }

    for (int y = 0; y < new_height; y++) {
        for (int x = 0; x < new_width; x++) {
            int src_y = y - top;
            int src_x = x - left;

            src_y = handle_boundary(src_y, image->height, mode);
            src_x = handle_boundary(src_x, image->width, mode);

            int dst_idx = (y * new_width + x) * image->channels;

            if (src_y == -1 || src_x == -1) {
                // 零填充
                for (int c = 0; c < image->channels; c++) {
                    padded->data[dst_idx + c] = 0.0f;
                }
            } else {
                int src_idx = (src_y * image->width + src_x) * image->channels;
                for (int c = 0; c < image->channels; c++) {
                    padded->data[dst_idx + c] = image->data[src_idx + c];
                }
            }
        }
    }

    return padded;
}

// ============================================================================
// 图像块管理
// ============================================================================

ImagePatch* extract_patch(const Image *image, int x, int y, int patch_size)
{
    if (!image || patch_size <= 0) {
        return NULL;
    }

    if (x < 0 || y < 0 || 
        x + patch_size > image->width || 
        y + patch_size > image->height) {
        return NULL;
    }

    ImagePatch *patch = (ImagePatch*)malloc(sizeof(ImagePatch));
    if (!patch) {
        return NULL;
    }

    patch->width = patch_size;
    patch->height = patch_size;
    patch->channels = image->channels;
    patch->x = x;
    patch->y = y;

    size_t size = (size_t)patch_size * patch_size * image->channels;
    patch->data = (float*)malloc(size * sizeof(float));
    
    if (!patch->data) {
        free(patch);
        return NULL;
    }

    // 提取块数据
    for (int row = 0; row < patch_size; row++) {
        for (int col = 0; col < patch_size; col++) {
            int src_idx = ((y + row) * image->width + (x + col)) * image->channels;
            int dst_idx = (row * patch_size + col) * image->channels;
            
            for (int c = 0; c < image->channels; c++) {
                patch->data[dst_idx + c] = image->data[src_idx + c];
            }
        }
    }

    return patch;
}

int place_patch(Image *image, const ImagePatch *patch, int x, int y)
{
    if (!image || !patch) {
        return SR_ERROR_NULL_POINTER;
    }

    if (x < 0 || y < 0 || 
        x + patch->width > image->width || 
        y + patch->height > image->height) {
        return SR_ERROR_INVALID_DIMENSIONS;
    }

    if (patch->channels != image->channels) {
        return SR_ERROR_INVALID_DIMENSIONS;
    }

    // 放置块数据
    for (int row = 0; row < patch->height; row++) {
        for (int col = 0; col < patch->width; col++) {
            int src_idx = (row * patch->width + col) * patch->channels;
            int dst_idx = ((y + row) * image->width + (x + col)) * image->channels;
            
            for (int c = 0; c < image->channels; c++) {
                image->data[dst_idx + c] = patch->data[src_idx + c];
            }
        }
    }

    return SR_SUCCESS;
}

void patch_destroy(ImagePatch *patch)
{
    if (patch) {
        if (patch->data) {
            free(patch->data);
        }
        free(patch);
    }
}

// ============================================================================
// 插值核函数
// ============================================================================

// 三次插值核
static inline float cubic_kernel(float x, float a)
{
    x = fabsf(x);
    
    if (x <= 1.0f) {
        return (a + 2.0f) * x * x * x - (a + 3.0f) * x * x + 1.0f;
    } else if (x < 2.0f) {
        return a * x * x * x - 5.0f * a * x * x + 8.0f * a * x - 4.0f * a;
    }
    
    return 0.0f;
}

// Lanczos核
static inline float lanczos_kernel(float x, int a)
{
    if (x == 0.0f) {
        return 1.0f;
    }
    
    if (fabsf(x) < a) {
        float pi_x = PI * x;
        return (a * sinf(pi_x) * sinf(pi_x / a)) / (pi_x * pi_x);
    }
    
    return 0.0f;
}

// ============================================================================
// 最近邻插值
// ============================================================================

Image* interpolate_nearest(const Image *input, float scale_factor)
{
    if (!input || scale_factor <= 0.0f) {
        return NULL;
    }

    int new_width = (int)(input->width * scale_factor + 0.5f);
    int new_height = (int)(input->height * scale_factor + 0.5f);

    Image *output = image_create(new_width, new_height, input->channels);
    if (!output) {
        return NULL;
    }

    float inv_scale = 1.0f / scale_factor;

    for (int y = 0; y < new_height; y++) {
        for (int x = 0; x < new_width; x++) {
            // 映射到原图坐标
            int src_x = (int)(x * inv_scale + 0.5f);
            int src_y = (int)(y * inv_scale + 0.5f);

            // 边界检查
            src_x = CLAMP(src_x, 0, input->width - 1);
            src_y = CLAMP(src_y, 0, input->height - 1);

            int src_idx = (src_y * input->width + src_x) * input->channels;
            int dst_idx = (y * new_width + x) * input->channels;

            for (int c = 0; c < input->channels; c++) {
                output->data[dst_idx + c] = input->data[src_idx + c];
            }
        }
    }

    return output;
}

// ============================================================================
// 双线性插值
// ============================================================================

Image* interpolate_bilinear(const Image *input, float scale_factor)
{
    if (!input || scale_factor <= 0.0f) {
        return NULL;
    }

    int new_width = (int)(input->width * scale_factor + 0.5f);
    int new_height = (int)(input->height * scale_factor + 0.5f);

    Image *output = image_create(new_width, new_height, input->channels);
    if (!output) {
        return NULL;
    }

    float inv_scale = 1.0f / scale_factor;

    for (int y = 0; y < new_height; y++) {
        for (int x = 0; x < new_width; x++) {
            // 映射到原图坐标
            float src_x = (x + 0.5f) * inv_scale - 0.5f;
            float src_y = (y + 0.5f) * inv_scale - 0.5f;

            // 获取四个邻近像素的坐标
            int x0 = (int)floorf(src_x);
            int y0 = (int)floorf(src_y);
            int x1 = x0 + 1;
            int y1 = y0 + 1;

            // 计算插值权重
            float wx = src_x - x0;
            float wy = src_y - y0;

            // 边界检查
            x0 = CLAMP(x0, 0, input->width - 1);
            x1 = CLAMP(x1, 0, input->width - 1);
            y0 = CLAMP(y0, 0, input->height - 1);
            y1 = CLAMP(y1, 0, input->height - 1);

            int dst_idx = (y * new_width + x) * input->channels;

            // 双线性插值
            for (int c = 0; c < input->channels; c++) {
                float v00 = input->data[(y0 * input->width + x0) * input->channels + c];
                float v01 = input->data[(y0 * input->width + x1) * input->channels + c];
                float v10 = input->data[(y1 * input->width + x0) * input->channels + c];
                float v11 = input->data[(y1 * input->width + x1) * input->channels + c];

                float v0 = v00 * (1.0f - wx) + v01 * wx;
                float v1 = v10 * (1.0f - wx) + v11 * wx;
                
                output->data[dst_idx + c] = v0 * (1.0f - wy) + v1 * wy;
            }
        }
    }

    return output;
}

// ============================================================================
// 双三次插值
// ============================================================================

Image* interpolate_bicubic(const Image *input, float scale_factor, float a)
{
    if (!input || scale_factor <= 0.0f) {
        return NULL;
    }

    int new_width = (int)(input->width * scale_factor + 0.5f);
    int new_height = (int)(input->height * scale_factor + 0.5f);

    Image *output = image_create(new_width, new_height, input->channels);
    if (!output) {
        return NULL;
    }

    float inv_scale = 1.0f / scale_factor;

    for (int y = 0; y < new_height; y++) {
        for (int x = 0; x < new_width; x++) {
            // 映射到原图坐标
            float src_x = (x + 0.5f) * inv_scale - 0.5f;
            float src_y = (y + 0.5f) * inv_scale - 0.5f;

            int x_int = (int)floorf(src_x);
            int y_int = (int)floorf(src_y);

            int dst_idx = (y * new_width + x) * input->channels;

            // 对每个通道进行插值
            for (int c = 0; c < input->channels; c++) {
                float sum = 0.0f;
                float weight_sum = 0.0f;

                // 4x4邻域
                for (int dy = -1; dy <= 2; dy++) {
                    for (int dx = -1; dx <= 2; dx++) {
                        int sx = x_int + dx;
                        int sy = y_int + dy;

                        // 边界检查
                        if (sx < 0 || sx >= input->width || 
                            sy < 0 || sy >= input->height) {
                            continue;
                        }

                        float wx = cubic_kernel(src_x - sx, a);
                        float wy = cubic_kernel(src_y - sy, a);
                        float weight = wx * wy;

                        int src_idx = (sy * input->width + sx) * input->channels + c;
                        sum += input->data[src_idx] * weight;
                        weight_sum += weight;
                    }
                }

                if (weight_sum > EPSILON) {
                    output->data[dst_idx + c] = sum / weight_sum;
                } else {
                    output->data[dst_idx + c] = 0.0f;
                }
            }
        }
    }

    return output;
}

// ============================================================================
// Lanczos插值
// ============================================================================

Image* interpolate_lanczos(const Image *input, float scale_factor, int a)
{
    if (!input || scale_factor <= 0.0f || a <= 0) {
        return NULL;
    }

    int new_width = (int)(input->width * scale_factor + 0.5f);
    int new_height = (int)(input->height * scale_factor + 0.5f);

    Image *output = image_create(new_width, new_height, input->channels);
    if (!output) {
        return NULL;
    }

    float inv_scale = 1.0f / scale_factor;

    for (int y = 0; y < new_height; y++) {
        for (int x = 0; x < new_width; x++) {
            // 映射到原图坐标
            float src_x = (x + 0.5f) * inv_scale - 0.5f;
            float src_y = (y + 0.5f) * inv_scale - 0.5f;

            int x_int = (int)floorf(src_x);
            int y_int = (int)floorf(src_y);

            int dst_idx = (y * new_width + x) * input->channels;

            // 对每个通道进行插值
            for (int c = 0; c < input->channels; c++) {
                float sum = 0.0f;
                float weight_sum = 0.0f;

                // Lanczos邻域
                for (int dy = -a + 1; dy <= a; dy++) {
                    for (int dx = -a + 1; dx <= a; dx++) {
                        int sx = x_int + dx;
                        int sy = y_int + dy;

                        // 边界检查
                        if (sx < 0 || sx >= input->width || 
                            sy < 0 || sy >= input->height) {
                            continue;
                        }

                        float wx = lanczos_kernel(src_x - sx, a);
                        float wy = lanczos_kernel(src_y - sy, a);
                        float weight = wx * wy;

                        int src_idx = (sy * input->width + sx) * input->channels + c;
                        sum += input->data[src_idx] * weight;
                        weight_sum += weight;
                    }
                }

                if (weight_sum > EPSILON) {
                    output->data[dst_idx + c] = sum / weight_sum;
                } else {
                    output->data[dst_idx + c] = 0.0f;
                }
            }
        }
    }

    return output;
}
// ============================================================================
// 通用插值函数
// ============================================================================

Image* interpolate(const Image *input, float scale_factor, 
                   const InterpolationParams *params)
{
    if (!input || !params || scale_factor <= 0.0f) {
        return NULL;
    }

    Image *output = NULL;

    // 根据插值方法选择
    switch (params->method) {
        case INTERP_NEAREST:
            output = interpolate_nearest(input, scale_factor);
            break;

        case INTERP_LINEAR:
            output = interpolate_bilinear(input, scale_factor);
            break;

        case INTERP_CUBIC:
            output = interpolate_bicubic(input, scale_factor, -0.5f);
            break;

        case INTERP_LANCZOS2:
            output = interpolate_lanczos(input, scale_factor, 2);
            break;

        case INTERP_LANCZOS3:
            output = interpolate_lanczos(input, scale_factor, 3);
            break;

        case INTERP_LANCZOS4:
            output = interpolate_lanczos(input, scale_factor, 4);
            break;

        default:
            return NULL;
    }

    if (!output) {
        return NULL;
    }

    // 应用锐化
    if (params->sharpness > 0.0f) {
        apply_sharpening(output, params->sharpness);
    }

    // 应用抗锯齿
    if (params->anti_aliasing && scale_factor < 1.0f) {
        apply_anti_aliasing(output);
    }

    return output;
}

// ============================================================================
// 下采样函数
// ============================================================================

Image* downsample_image(const Image *input, float scale_factor,
                       DownsampleMethod method)
{
    if (!input || scale_factor <= 0.0f || scale_factor >= 1.0f) {
        return NULL;
    }

    int new_width = (int)(input->width * scale_factor + 0.5f);
    int new_height = (int)(input->height * scale_factor + 0.5f);

    if (new_width <= 0 || new_height <= 0) {
        return NULL;
    }

    Image *output = image_create(new_width, new_height, input->channels);
    if (!output) {
        return NULL;
    }

    float inv_scale = 1.0f / scale_factor;

    switch (method) {
        case DOWNSAMPLE_NEAREST:
            // 最近邻下采样
            for (int y = 0; y < new_height; y++) {
                for (int x = 0; x < new_width; x++) {
                    int src_x = (int)(x * inv_scale + 0.5f);
                    int src_y = (int)(y * inv_scale + 0.5f);

                    src_x = CLAMP(src_x, 0, input->width - 1);
                    src_y = CLAMP(src_y, 0, input->height - 1);

                    int src_idx = (src_y * input->width + src_x) * input->channels;
                    int dst_idx = (y * new_width + x) * input->channels;

                    for (int c = 0; c < input->channels; c++) {
                        output->data[dst_idx + c] = input->data[src_idx + c];
                    }
                }
            }
            break;

        case DOWNSAMPLE_AVERAGE:
            // 区域平均下采样
            for (int y = 0; y < new_height; y++) {
                for (int x = 0; x < new_width; x++) {
                    float src_x_start = x * inv_scale;
                    float src_y_start = y * inv_scale;
                    float src_x_end = (x + 1) * inv_scale;
                    float src_y_end = (y + 1) * inv_scale;

                    int x0 = (int)floorf(src_x_start);
                    int y0 = (int)floorf(src_y_start);
                    int x1 = (int)ceilf(src_x_end);
                    int y1 = (int)ceilf(src_y_end);

                    x0 = CLAMP(x0, 0, input->width - 1);
                    y0 = CLAMP(y0, 0, input->height - 1);
                    x1 = CLAMP(x1, 0, input->width);
                    y1 = CLAMP(y1, 0, input->height);

                    int dst_idx = (y * new_width + x) * input->channels;
                    float count = (x1 - x0) * (y1 - y0);

                    for (int c = 0; c < input->channels; c++) {
                        float sum = 0.0f;
                        for (int sy = y0; sy < y1; sy++) {
                            for (int sx = x0; sx < x1; sx++) {
                                int src_idx = (sy * input->width + sx) * input->channels + c;
                                sum += input->data[src_idx];
                            }
                        }
                        output->data[dst_idx + c] = sum / count;
                    }
                }
            }
            break;

        case DOWNSAMPLE_GAUSSIAN:
            // 高斯模糊后下采样
            {
                // 先应用高斯模糊
                float sigma = 0.5f / scale_factor;
                Image *blurred = apply_gaussian_blur(input, sigma);
                if (!blurred) {
                    image_destroy(output);
                    return NULL;
                }

                // 然后双线性下采样
                for (int y = 0; y < new_height; y++) {
                    for (int x = 0; x < new_width; x++) {
                        float src_x = (x + 0.5f) * inv_scale - 0.5f;
                        float src_y = (y + 0.5f) * inv_scale - 0.5f;

                        int x0 = (int)floorf(src_x);
                        int y0 = (int)floorf(src_y);
                        int x1 = x0 + 1;
                        int y1 = y0 + 1;

                        float wx = src_x - x0;
                        float wy = src_y - y0;

                        x0 = CLAMP(x0, 0, blurred->width - 1);
                        x1 = CLAMP(x1, 0, blurred->width - 1);
                        y0 = CLAMP(y0, 0, blurred->height - 1);
                        y1 = CLAMP(y1, 0, blurred->height - 1);

                        int dst_idx = (y * new_width + x) * input->channels;

                        for (int c = 0; c < input->channels; c++) {
                            float v00 = blurred->data[(y0 * blurred->width + x0) * input->channels + c];
                            float v01 = blurred->data[(y0 * blurred->width + x1) * input->channels + c];
                            float v10 = blurred->data[(y1 * blurred->width + x0) * input->channels + c];
                            float v11 = blurred->data[(y1 * blurred->width + x1) * input->channels + c];

                            float v0 = v00 * (1.0f - wx) + v01 * wx;
                            float v1 = v10 * (1.0f - wx) + v11 * wx;
                            
                            output->data[dst_idx + c] = v0 * (1.0f - wy) + v1 * wy;
                        }
                    }
                }

                image_destroy(blurred);
            }
            break;

        case DOWNSAMPLE_LANCZOS:
            // Lanczos下采样
            {
                Image *temp = interpolate_lanczos(input, scale_factor, 3);
                if (!temp) {
                    image_destroy(output);
                    return NULL;
                }
                image_destroy(output);
                output = temp;
            }
            break;

        default:
            image_destroy(output);
            return NULL;
    }

    return output;
}

// ============================================================================
// 辅助图像处理函数
// ============================================================================

// 高斯模糊
static Image* apply_gaussian_blur(const Image *input, float sigma)
{
    if (!input || sigma <= 0.0f) {
        return NULL;
    }

    // 计算高斯核大小
    int kernel_size = (int)(6.0f * sigma + 1.0f);
    if (kernel_size % 2 == 0) kernel_size++;
    int radius = kernel_size / 2;

    // 创建高斯核
    float *kernel = (float*)malloc(kernel_size * sizeof(float));
    if (!kernel) {
        return NULL;
    }

    float sum = 0.0f;
    for (int i = 0; i < kernel_size; i++) {
        int x = i - radius;
        kernel[i] = expf(-(x * x) / (2.0f * sigma * sigma));
        sum += kernel[i];
    }

    // 归一化
    for (int i = 0; i < kernel_size; i++) {
        kernel[i] /= sum;
    }

    // 创建临时图像
    Image *temp = image_create(input->width, input->height, input->channels);
    Image *output = image_create(input->width, input->height, input->channels);

    if (!temp || !output) {
        free(kernel);
        image_destroy(temp);
        image_destroy(output);
        return NULL;
    }

    // 水平方向模糊
    for (int y = 0; y < input->height; y++) {
        for (int x = 0; x < input->width; x++) {
            int dst_idx = (y * input->width + x) * input->channels;

            for (int c = 0; c < input->channels; c++) {
                float sum_val = 0.0f;
                float sum_weight = 0.0f;

                for (int k = 0; k < kernel_size; k++) {
                    int sx = x + k - radius;
                    if (sx >= 0 && sx < input->width) {
                        int src_idx = (y * input->width + sx) * input->channels + c;
                        sum_val += input->data[src_idx] * kernel[k];
                        sum_weight += kernel[k];
                    }
                }

                temp->data[dst_idx + c] = sum_val / sum_weight;
            }
        }
    }

    // 垂直方向模糊
    for (int y = 0; y < input->height; y++) {
        for (int x = 0; x < input->width; x++) {
            int dst_idx = (y * input->width + x) * input->channels;

            for (int c = 0; c < input->channels; c++) {
                float sum_val = 0.0f;
                float sum_weight = 0.0f;

                for (int k = 0; k < kernel_size; k++) {
                    int sy = y + k - radius;
                    if (sy >= 0 && sy < input->height) {
                        int src_idx = (sy * input->width + x) * input->channels + c;
                        sum_val += temp->data[src_idx] * kernel[k];
                        sum_weight += kernel[k];
                    }
                }

                output->data[dst_idx + c] = sum_val / sum_weight;
            }
        }
    }

    free(kernel);
    image_destroy(temp);

    return output;
}

// 锐化
static int apply_sharpening(Image *image, float strength)
{
    if (!image || strength <= 0.0f) {
        return SR_ERROR_INVALID_PARAMETER;
    }

    // 创建拉普拉斯核
    float laplacian[9] = {
        0.0f, -1.0f, 0.0f,
        -1.0f, 4.0f, -1.0f,
        0.0f, -1.0f, 0.0f
    };

    Image *temp = image_clone(image);
    if (!temp) {
        return SR_ERROR_MEMORY_ALLOCATION;
    }

    // 应用拉普拉斯算子
    for (int y = 1; y < image->height - 1; y++) {
        for (int x = 1; x < image->width - 1; x++) {
            int center_idx = (y * image->width + x) * image->channels;

            for (int c = 0; c < image->channels; c++) {
                float sum = 0.0f;

                for (int ky = -1; ky <= 1; ky++) {
                    for (int kx = -1; kx <= 1; kx++) {
                        int idx = ((y + ky) * image->width + (x + kx)) * image->channels + c;
                        int k_idx = (ky + 1) * 3 + (kx + 1);
                        sum += temp->data[idx] * laplacian[k_idx];
                    }
                }

                // 添加锐化
                float sharpened = temp->data[center_idx + c] + strength * sum;
                image->data[center_idx + c] = CLAMP(sharpened, 0.0f, 1.0f);
            }
        }
    }

    image_destroy(temp);
    return SR_SUCCESS;
}

// 抗锯齿
static int apply_anti_aliasing(Image *image)
{
    if (!image) {
        return SR_ERROR_NULL_POINTER;
    }

    // 简单的3x3平均滤波
    Image *temp = image_clone(image);
    if (!temp) {
        return SR_ERROR_MEMORY_ALLOCATION;
    }

    for (int y = 1; y < image->height - 1; y++) {
        for (int x = 1; x < image->width - 1; x++) {
            int center_idx = (y * image->width + x) * image->channels;

            for (int c = 0; c < image->channels; c++) {
                float sum = 0.0f;
                int count = 0;

                for (int ky = -1; ky <= 1; ky++) {
                    for (int kx = -1; kx <= 1; kx++) {
                        int idx = ((y + ky) * image->width + (x + kx)) * image->channels + c;
                        sum += temp->data[idx];
                        count++;
                    }
                }

                image->data[center_idx + c] = sum / count;
            }
        }
    }

    image_destroy(temp);
    return SR_SUCCESS;
}

// ============================================================================
// 迭代反向投影超分辨率 (IBSR)
// ============================================================================

int super_resolve_ibsr(const Image *input, float scale_factor,
                       const IBSRParams *params, SRResult *result)
{
    if (!input || !params || !result || scale_factor <= 1.0f) {
        return SR_ERROR_NULL_POINTER;
    }

    clock_t start_time = clock();

    // 初始上采样（使用双三次插值）
    Image *estimate = interpolate_bicubic(input, scale_factor, -0.5f);
    if (!estimate) {
        return SR_ERROR_MEMORY_ALLOCATION;
    }

    Image *estimate_prev = image_clone(estimate);
    if (!estimate_prev) {
        image_destroy(estimate);
        return SR_ERROR_MEMORY_ALLOCATION;
    }

    // 迭代反向投影
    int iter;
    bool converged = false;

    for (iter = 0; iter < params->max_iterations; iter++) {
        // 1. 模拟下采样
        Image *simulated_lr = downsample_image(estimate, 1.0f / scale_factor,
                                               params->downsample);
        if (!simulated_lr) {
            image_destroy(estimate);
            image_destroy(estimate_prev);
            return SR_ERROR_MEMORY_ALLOCATION;
        }

        // 2. 计算误差
        Image *error_lr = image_create(input->width, input->height, input->channels);
        if (!error_lr) {
            image_destroy(estimate);
            image_destroy(estimate_prev);
            image_destroy(simulated_lr);
            return SR_ERROR_MEMORY_ALLOCATION;
        }

        int lr_size = input->width * input->height * input->channels;
        for (int i = 0; i < lr_size; i++) {
            error_lr->data[i] = input->data[i] - simulated_lr->data[i];
        }

        image_destroy(simulated_lr);

        // 3. 上采样误差
        Image *error_hr = interpolate_bicubic(error_lr, scale_factor, -0.5f);
        image_destroy(error_lr);

        if (!error_hr) {
            image_destroy(estimate);
            image_destroy(estimate_prev);
            return SR_ERROR_MEMORY_ALLOCATION;
        }

        // 4. 更新估计
        int hr_size = estimate->width * estimate->height * estimate->channels;
        for (int i = 0; i < hr_size; i++) {
            estimate->data[i] += params->alpha * error_hr->data[i];
            
            // 正则化
            if (params->lambda > 0.0f) {
                float diff = estimate->data[i] - estimate_prev->data[i];
                estimate->data[i] -= params->lambda * diff;
            }

            // 确保非负
            if (estimate->data[i] < 0.0f) {
                estimate->data[i] = 0.0f;
            }
        }

        image_destroy(error_hr);

        // 5. 检查收敛
        float error = 0.0f;
        for (int i = 0; i < hr_size; i++) {
            float diff = estimate->data[i] - estimate_prev->data[i];
            error += diff * diff;
        }
        error = sqrtf(error / hr_size);

        if (iter < MAX_ERROR_HISTORY) {
            result->error_history[iter] = error;
            result->error_history_length = iter + 1;
        }

        if (params->verbose) {
            printf("IBSR Iteration %d: error = %.6e\n", iter, error);
        }

        if (error < params->tolerance) {
            converged = true;
            if (params->verbose) {
                printf("Converged at iteration %d\n", iter);
            }
            break;
        }

        // 保存当前估计
        memcpy(estimate_prev->data, estimate->data, hr_size * sizeof(float));
    }

    // 复制结果
    memcpy(result->output->data, estimate->data,
           estimate->width * estimate->height * estimate->channels * sizeof(float));

    result->iterations = iter;
    result->converged = converged;

    clock_t end_time = clock();
    result->computation_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;

    image_destroy(estimate);
    image_destroy(estimate_prev);

    return SR_SUCCESS;
}

// ============================================================================
// 带边缘先验的IBSR
// ============================================================================

// 计算边缘图
static Image* compute_edge_map(const Image *image)
{
    if (!image) {
        return NULL;
    }

    Image *gray = image_to_grayscale(image);
    if (!gray) {
        return NULL;
    }

    Image *edges = image_create(gray->width, gray->height, 1);
    if (!edges) {
        image_destroy(gray);
        return NULL;
    }

    // Sobel算子
    float sobel_x[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
    float sobel_y[9] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};

    for (int y = 1; y < gray->height - 1; y++) {
        for (int x = 1; x < gray->width - 1; x++) {
            float gx = 0.0f, gy = 0.0f;

            for (int ky = -1; ky <= 1; ky++) {
                for (int kx = -1; kx <= 1; kx++) {
                    int idx = (y + ky) * gray->width + (x + kx);
                    int k_idx = (ky + 1) * 3 + (kx + 1);
                    
                    gx += gray->data[idx] * sobel_x[k_idx];
                    gy += gray->data[idx] * sobel_y[k_idx];
                }
            }

            int edge_idx = y * gray->width + x;
            edges->data[edge_idx] = sqrtf(gx * gx + gy * gy);
        }
    }

    image_destroy(gray);
    return edges;
}

int super_resolve_ibsr_edge_prior(const Image *input, float scale_factor,
                                   const IBSRParams *params, SRResult *result)
{
    if (!input || !params || !result || scale_factor <= 1.0f) {
        return SR_ERROR_NULL_POINTER;
    }

    if (!params->use_edge_prior) {
        return super_resolve_ibsr(input, scale_factor, params, result);
    }

    clock_t start_time = clock();

    // 计算输入图像的边缘图
    Image *edge_map_lr = compute_edge_map(input);
    if (!edge_map_lr) {
        return SR_ERROR_MEMORY_ALLOCATION;
    }

    // 上采样边缘图
    Image *edge_map_hr = interpolate_bicubic(edge_map_lr, scale_factor, -0.5f);
    image_destroy(edge_map_lr);

    if (!edge_map_hr) {
        return SR_ERROR_MEMORY_ALLOCATION;
    }

    // 初始上采样
    Image *estimate = interpolate_bicubic(input, scale_factor, -0.5f);
    if (!estimate) {
        image_destroy(edge_map_hr);
        return SR_ERROR_MEMORY_ALLOCATION;
    }

    Image *estimate_prev = image_clone(estimate);
    if (!estimate_prev) {
        image_destroy(estimate);
        image_destroy(edge_map_hr);
        return SR_ERROR_MEMORY_ALLOCATION;
    }

    // 迭代
    int iter;
    bool converged = false;

    for (iter = 0; iter < params->max_iterations; iter++) {
        // 标准IBSR步骤
        Image *simulated_lr = downsample_image(estimate, 1.0f / scale_factor,
                                               params->downsample);
        if (!simulated_lr) {
            break;
        }

        Image *error_lr = image_create(input->width, input->height, input->channels);
        if (!error_lr) {
            image_destroy(simulated_lr);
            break;
        }

        int lr_size = input->width * input->height * input->channels;
        for (int i = 0; i < lr_size; i++) {
            error_lr->data[i] = input->data[i] - simulated_lr->data[i];
        }
        image_destroy(simulated_lr);

        Image *error_hr = interpolate_bicubic(error_lr, scale_factor, -0.5f);
        image_destroy(error_lr);

        if (!error_hr) {
            break;
        }

        // 使用边缘先验更新
        int hr_size = estimate->width * estimate->height * estimate->channels;
        for (int y = 0; y < estimate->height; y++) {
            for (int x = 0; x < estimate->width; x++) {
                int edge_idx = y * estimate->width + x;
                float edge_weight = 1.0f + edge_map_hr->data[edge_idx];

                for (int c = 0; c < estimate->channels; c++) {
                    int idx = (y * estimate->width + x) * estimate->channels + c;
                    
                    estimate->data[idx] += params->alpha * error_hr->data[idx] * edge_weight;
                    
                    if (params->lambda > 0.0f) {
                        float diff = estimate->data[idx] - estimate_prev->data[idx];
                        estimate->data[idx] -= params->lambda * diff / edge_weight;
                    }

                    if (estimate->data[idx] < 0.0f) {
                        estimate->data[idx] = 0.0f;
                    }
                }
            }
        }

        image_destroy(error_hr);

        // 检查收敛
        float error = 0.0f;
        for (int i = 0; i < hr_size; i++) {
            float diff = estimate->data[i] - estimate_prev->data[i];
            error += diff * diff;
        }
        error = sqrtf(error / hr_size);

        if (iter < MAX_ERROR_HISTORY) {
            result->error_history[iter] = error;
            result->error_history_length = iter + 1;
        }

        if (params->verbose) {
            printf("IBSR-Edge Iteration %d: error = %.6e\n", iter, error);
        }

        if (error < params->tolerance) {
            converged = true;
            break;
        }

        memcpy(estimate_prev->data, estimate->data, hr_size * sizeof(float));
    }

    // 复制结果
    memcpy(result->output->data, estimate->data,
           estimate->width * estimate->height * estimate->channels * sizeof(float));

    result->iterations = iter;
    result->converged = converged;

    clock_t end_time = clock();
    result->computation_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;

    image_destroy(estimate);
    image_destroy(estimate_prev);
    image_destroy(edge_map_hr);

    return SR_SUCCESS;
}
// ============================================================================
// 稀疏编码超分辨率
// ============================================================================

// 正交匹配追踪 (OMP) 算法
static int orthogonal_matching_pursuit(const float *signal, int signal_dim,
                                      const float *dictionary, int dict_size,
                                      float sparsity, int max_iterations,
                                      float *coefficients)
{
    if (!signal || !dictionary || !coefficients) {
        return SR_ERROR_NULL_POINTER;
    }

    // 初始化
    memset(coefficients, 0, dict_size * sizeof(float));
    
    float *residual = (float*)malloc(signal_dim * sizeof(float));
    int *support = (int*)malloc(max_iterations * sizeof(int));
    float *projection = (float*)malloc(signal_dim * sizeof(float));
    
    if (!residual || !support || !projection) {
        free(residual);
        free(support);
        free(projection);
        return SR_ERROR_MEMORY_ALLOCATION;
    }

    // 初始残差
    memcpy(residual, signal, signal_dim * sizeof(float));
    
    int support_size = 0;
    int max_support = (int)(dict_size * sparsity);
    if (max_support > max_iterations) {
        max_support = max_iterations;
    }

    // OMP迭代
    for (int iter = 0; iter < max_support; iter++) {
        // 找到与残差最相关的原子
        float max_correlation = 0.0f;
        int best_atom = -1;

        for (int i = 0; i < dict_size; i++) {
            // 检查是否已在支持集中
            bool in_support = false;
            for (int j = 0; j < support_size; j++) {
                if (support[j] == i) {
                    in_support = true;
                    break;
                }
            }
            if (in_support) continue;

            // 计算相关性
            float correlation = 0.0f;
            for (int j = 0; j < signal_dim; j++) {
                correlation += residual[j] * dictionary[i * signal_dim + j];
            }
            correlation = fabsf(correlation);

            if (correlation > max_correlation) {
                max_correlation = correlation;
                best_atom = i;
            }
        }

        if (best_atom == -1) {
            break;
        }

        // 添加到支持集
        support[support_size++] = best_atom;

        // 最小二乘求解
        // 构建子字典
        float *sub_dict = (float*)malloc(signal_dim * support_size * sizeof(float));
        if (!sub_dict) {
            free(residual);
            free(support);
            free(projection);
            return SR_ERROR_MEMORY_ALLOCATION;
        }

        for (int i = 0; i < support_size; i++) {
            int atom_idx = support[i];
            for (int j = 0; j < signal_dim; j++) {
                sub_dict[i * signal_dim + j] = dictionary[atom_idx * signal_dim + j];
            }
        }

        // 求解 sub_dict * x = signal (使用伪逆)
        float *sub_coef = (float*)calloc(support_size, sizeof(float));
        if (!sub_coef) {
            free(sub_dict);
            free(residual);
            free(support);
            free(projection);
            return SR_ERROR_MEMORY_ALLOCATION;
        }

        // 简化的最小二乘求解 (正规方程)
        for (int i = 0; i < support_size; i++) {
            float numerator = 0.0f;
            float denominator = 0.0f;
            
            for (int j = 0; j < signal_dim; j++) {
                numerator += sub_dict[i * signal_dim + j] * signal[j];
                denominator += sub_dict[i * signal_dim + j] * sub_dict[i * signal_dim + j];
            }
            
            if (denominator > EPSILON) {
                sub_coef[i] = numerator / denominator;
            }
        }

        // 更新系数
        for (int i = 0; i < support_size; i++) {
            coefficients[support[i]] = sub_coef[i];
        }

        // 计算投影
        memset(projection, 0, signal_dim * sizeof(float));
        for (int i = 0; i < support_size; i++) {
            int atom_idx = support[i];
            for (int j = 0; j < signal_dim; j++) {
                projection[j] += coefficients[atom_idx] * dictionary[atom_idx * signal_dim + j];
            }
        }

        // 更新残差
        for (int i = 0; i < signal_dim; i++) {
            residual[i] = signal[i] - projection[i];
        }

        // 检查残差
        float residual_norm = 0.0f;
        for (int i = 0; i < signal_dim; i++) {
            residual_norm += residual[i] * residual[i];
        }
        residual_norm = sqrtf(residual_norm);

        if (residual_norm < EPSILON) {
            break;
        }

        free(sub_dict);
        free(sub_coef);
    }

    free(residual);
    free(support);
    free(projection);

    return SR_SUCCESS;
}

// 从图像块提取特征
static void extract_patch_features(const ImagePatch *patch, float *features)
{
    if (!patch || !features) {
        return;
    }

    int size = patch->width * patch->height * patch->channels;
    
    // 简单地展平图像块
    memcpy(features, patch->data, size * sizeof(float));
    
    // 归一化
    float mean = 0.0f;
    for (int i = 0; i < size; i++) {
        mean += features[i];
    }
    mean /= size;

    float std = 0.0f;
    for (int i = 0; i < size; i++) {
        features[i] -= mean;
        std += features[i] * features[i];
    }
    std = sqrtf(std / size);

    if (std > EPSILON) {
        for (int i = 0; i < size; i++) {
            features[i] /= std;
        }
    }
}

int super_resolve_sparse_coding(const Image *input, float scale_factor,
                                const SparseCodingParams *params, 
                                SRResult *result)
{
    if (!input || !params || !result || scale_factor <= 1.0f) {
        return SR_ERROR_NULL_POINTER;
    }

    clock_t start_time = clock();

    // 初始上采样
    Image *initial_hr = interpolate_bicubic(input, scale_factor, -0.5f);
    if (!initial_hr) {
        return SR_ERROR_MEMORY_ALLOCATION;
    }

    // 训练字典
    DictionaryLearningParams dict_params = dict_get_default_params();
    dict_params.dict_size = params->dict_size;
    dict_params.patch_size = params->patch_size;
    dict_params.overlap = params->overlap;
    dict_params.sparsity = params->sparsity;
    dict_params.training_iterations = params->max_iterations;

    const Image *training_images[] = {input};
    Dictionary *dict = train_sr_dictionary(training_images, 1, scale_factor, &dict_params);
    
    if (!dict) {
        image_destroy(initial_hr);
        return SR_ERROR_MEMORY_ALLOCATION;
    }

    // 使用字典进行超分辨率
    int ret = super_resolve_sparse_coding_with_dict(input, scale_factor, dict, params, result);

    dictionary_destroy(dict);
    image_destroy(initial_hr);

    clock_t end_time = clock();
    result->computation_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;

    return ret;
}

int super_resolve_sparse_coding_with_dict(const Image *input, float scale_factor,
                                          const Dictionary *dictionary,
                                          const SparseCodingParams *params,
                                          SRResult *result)
{
    if (!input || !dictionary || !params || !result || scale_factor <= 1.0f) {
        return SR_ERROR_NULL_POINTER;
    }

    clock_t start_time = clock();

    int output_width = (int)(input->width * scale_factor + 0.5f);
    int output_height = (int)(input->height * scale_factor + 0.5f);

    // 创建输出图像
    Image *output = image_create(output_width, output_height, input->channels);
    if (!output) {
        return SR_ERROR_MEMORY_ALLOCATION;
    }

    // 创建权重图（用于重叠区域的加权平均）
    float *weights = (float*)calloc(output_width * output_height, sizeof(float));
    if (!weights) {
        image_destroy(output);
        return SR_ERROR_MEMORY_ALLOCATION;
    }

    int patch_size = params->patch_size;
    int overlap = params->overlap;
    int stride = patch_size - overlap;

    int feature_dim = patch_size * patch_size * input->channels;
    float *lr_features = (float*)malloc(feature_dim * sizeof(float));
    float *coefficients = (float*)malloc(dictionary->size * sizeof(float));
    float *hr_patch_data = (float*)malloc(feature_dim * sizeof(float));

    if (!lr_features || !coefficients || !hr_patch_data) {
        free(weights);
        free(lr_features);
        free(coefficients);
        free(hr_patch_data);
        image_destroy(output);
        return SR_ERROR_MEMORY_ALLOCATION;
    }

    // 处理每个图像块
    int num_patches_y = (input->height - patch_size) / stride + 1;
    int num_patches_x = (input->width - patch_size) / stride + 1;

    for (int py = 0; py < num_patches_y; py++) {
        for (int px = 0; px < num_patches_x; px++) {
            int y = py * stride;
            int x = px * stride;

            // 确保不超出边界
            if (y + patch_size > input->height) {
                y = input->height - patch_size;
            }
            if (x + patch_size > input->width) {
                x = input->width - patch_size;
            }

            // 提取低分辨率块
            ImagePatch *lr_patch = extract_patch(input, x, y, patch_size);
            if (!lr_patch) {
                continue;
            }

            // 提取特征
            extract_patch_features(lr_patch, lr_features);

            // 稀疏编码
            int omp_ret = orthogonal_matching_pursuit(
                lr_features, feature_dim,
                dictionary->lr_atoms, dictionary->size,
                params->sparsity, params->max_iterations,
                coefficients
            );

            if (omp_ret != SR_SUCCESS) {
                patch_destroy(lr_patch);
                continue;
            }

            // 使用高分辨率字典重建
            memset(hr_patch_data, 0, feature_dim * sizeof(float));
            for (int i = 0; i < dictionary->size; i++) {
                if (fabsf(coefficients[i]) > EPSILON) {
                    for (int j = 0; j < feature_dim; j++) {
                        hr_patch_data[j] += coefficients[i] * dictionary->hr_atoms[i * feature_dim + j];
                    }
                }
            }

            // 将高分辨率块放置到输出图像
            int hr_x = (int)(x * scale_factor + 0.5f);
            int hr_y = (int)(y * scale_factor + 0.5f);
            int hr_patch_size = (int)(patch_size * scale_factor + 0.5f);

            for (int dy = 0; dy < hr_patch_size && hr_y + dy < output_height; dy++) {
                for (int dx = 0; dx < hr_patch_size && hr_x + dx < output_width; dx++) {
                    int out_idx = ((hr_y + dy) * output_width + (hr_x + dx)) * input->channels;
                    int patch_idx = (dy * hr_patch_size + dx) * input->channels;

                    // 加权累加
                    float weight = 1.0f;
                    for (int c = 0; c < input->channels; c++) {
                        output->data[out_idx + c] += hr_patch_data[patch_idx + c] * weight;
                    }

                    int weight_idx = (hr_y + dy) * output_width + (hr_x + dx);
                    weights[weight_idx] += weight;
                }
            }

            patch_destroy(lr_patch);
        }
    }

    // 归一化（加权平均）
    for (int i = 0; i < output_width * output_height; i++) {
        if (weights[i] > EPSILON) {
            for (int c = 0; c < input->channels; c++) {
                output->data[i * input->channels + c] /= weights[i];
            }
        }
    }

    // 复制到结果
    memcpy(result->output->data, output->data,
           output_width * output_height * input->channels * sizeof(float));

    free(weights);
    free(lr_features);
    free(coefficients);
    free(hr_patch_data);
    image_destroy(output);

    clock_t end_time = clock();
    result->computation_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;

    return SR_SUCCESS;
}

// ============================================================================
// 字典学习
// ============================================================================

Dictionary* dictionary_create(int size, int atom_dim)
{
    if (size <= 0 || atom_dim <= 0) {
        return NULL;
    }

    Dictionary *dict = (Dictionary*)malloc(sizeof(Dictionary));
    if (!dict) {
        return NULL;
    }

    dict->size = size;
    dict->atom_dim = atom_dim;

    dict->atoms = (float*)malloc(size * atom_dim * sizeof(float));
    dict->lr_atoms = (float*)malloc(size * atom_dim * sizeof(float));
    dict->hr_atoms = (float*)malloc(size * atom_dim * sizeof(float));

    if (!dict->atoms || !dict->lr_atoms || !dict->hr_atoms) {
        dictionary_destroy(dict);
        return NULL;
    }

    // 随机初始化
    for (int i = 0; i < size * atom_dim; i++) {
        dict->atoms[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        dict->lr_atoms[i] = dict->atoms[i];
        dict->hr_atoms[i] = dict->atoms[i];
    }

    // 归一化原子
    for (int i = 0; i < size; i++) {
        float norm = 0.0f;
        for (int j = 0; j < atom_dim; j++) {
            norm += dict->atoms[i * atom_dim + j] * dict->atoms[i * atom_dim + j];
        }
        norm = sqrtf(norm);

        if (norm > EPSILON) {
            for (int j = 0; j < atom_dim; j++) {
                dict->atoms[i * atom_dim + j] /= norm;
                dict->lr_atoms[i * atom_dim + j] /= norm;
                dict->hr_atoms[i * atom_dim + j] /= norm;
            }
        }
    }

    return dict;
}

void dictionary_destroy(Dictionary *dict)
{
    if (dict) {
        if (dict->atoms) free(dict->atoms);
        if (dict->lr_atoms) free(dict->lr_atoms);
        if (dict->hr_atoms) free(dict->hr_atoms);
        free(dict);
    }
}

// K-SVD字典更新
static int update_dictionary_ksvd(Dictionary *dict, float **training_samples,
                                  float **sparse_codes, int num_samples)
{
    if (!dict || !training_samples || !sparse_codes) {
        return SR_ERROR_NULL_POINTER;
    }

    int atom_dim = dict->atom_dim;
    int dict_size = dict->size;

    // 对每个原子进行更新
    for (int k = 0; k < dict_size; k++) {
        // 找到使用该原子的样本
        int *using_samples = (int*)malloc(num_samples * sizeof(int));
        int num_using = 0;

        for (int i = 0; i < num_samples; i++) {
            if (fabsf(sparse_codes[i][k]) > EPSILON) {
                using_samples[num_using++] = i;
            }
        }

        if (num_using == 0) {
            free(using_samples);
            continue;
        }

        // 计算误差矩阵
        float *error_matrix = (float*)malloc(atom_dim * num_using * sizeof(float));
        if (!error_matrix) {
            free(using_samples);
            return SR_ERROR_MEMORY_ALLOCATION;
        }

        for (int i = 0; i < num_using; i++) {
            int sample_idx = using_samples[i];
            
            for (int j = 0; j < atom_dim; j++) {
                error_matrix[i * atom_dim + j] = training_samples[sample_idx][j];
                
                // 减去其他原子的贡献
                for (int l = 0; l < dict_size; l++) {
                    if (l != k) {
                        error_matrix[i * atom_dim + j] -= 
                            sparse_codes[sample_idx][l] * dict->lr_atoms[l * atom_dim + j];
                    }
                }
            }
        }

        // 简化的SVD更新：使用主成分
        float *new_atom = (float*)calloc(atom_dim, sizeof(float));
        if (!new_atom) {
            free(error_matrix);
            free(using_samples);
            return SR_ERROR_MEMORY_ALLOCATION;
        }

        // 计算误差矩阵的第一主成分
        for (int i = 0; i < num_using; i++) {
            for (int j = 0; j < atom_dim; j++) {
                new_atom[j] += error_matrix[i * atom_dim + j];
            }
        }

        // 归一化
        float norm = 0.0f;
        for (int j = 0; j < atom_dim; j++) {
            norm += new_atom[j] * new_atom[j];
        }
        norm = sqrtf(norm);

        if (norm > EPSILON) {
            for (int j = 0; j < atom_dim; j++) {
                dict->lr_atoms[k * atom_dim + j] = new_atom[j] / norm;
            }

            // 更新稀疏系数
            for (int i = 0; i < num_using; i++) {
                int sample_idx = using_samples[i];
                float coef = 0.0f;
                
                for (int j = 0; j < atom_dim; j++) {
                    coef += error_matrix[i * atom_dim + j] * dict->lr_atoms[k * atom_dim + j];
                }
                
                sparse_codes[sample_idx][k] = coef;
            }
        }

        free(new_atom);
        free(error_matrix);
        free(using_samples);
    }

    return SR_SUCCESS;
}

Dictionary* train_sr_dictionary(const Image **training_images, int num_images,
                               float scale_factor,
                               const DictionaryLearningParams *params)
{
    if (!training_images || num_images <= 0 || !params || scale_factor <= 1.0f) {
        return NULL;
    }

    int patch_size = params->patch_size;
    int dict_size = params->dict_size;
    int atom_dim = patch_size * patch_size * training_images[0]->channels;

    // 创建字典
    Dictionary *dict = dictionary_create(dict_size, atom_dim);
    if (!dict) {
        return NULL;
    }

    // 收集训练样本
    int max_patches = num_images * 1000; // 每张图像最多1000个块
    float **lr_samples = (float**)malloc(max_patches * sizeof(float*));
    float **hr_samples = (float**)malloc(max_patches * sizeof(float*));
    
    if (!lr_samples || !hr_samples) {
        dictionary_destroy(dict);
        free(lr_samples);
        free(hr_samples);
        return NULL;
    }

    int total_samples = 0;

    // 从训练图像中提取样本
    for (int img_idx = 0; img_idx < num_images; img_idx++) {
        const Image *img = training_images[img_idx];
        
        // 创建高分辨率版本
        Image *hr_img = interpolate_bicubic(img, scale_factor, -0.5f);
        if (!hr_img) {
            continue;
        }

        int stride = patch_size / 2;
        int num_patches_y = (img->height - patch_size) / stride + 1;
        int num_patches_x = (img->width - patch_size) / stride + 1;

        for (int py = 0; py < num_patches_y && total_samples < max_patches; py++) {
            for (int px = 0; px < num_patches_x && total_samples < max_patches; px++) {
                int y = py * stride;
                int x = px * stride;

                if (y + patch_size > img->height) y = img->height - patch_size;
                if (x + patch_size > img->width) x = img->width - patch_size;

                // 提取低分辨率块
                ImagePatch *lr_patch = extract_patch(img, x, y, patch_size);
                if (!lr_patch) continue;

                // 提取对应的高分辨率块
                int hr_x = (int)(x * scale_factor + 0.5f);
                int hr_y = (int)(y * scale_factor + 0.5f);
                int hr_patch_size = (int)(patch_size * scale_factor + 0.5f);
                
                ImagePatch *hr_patch = extract_patch(hr_img, hr_x, hr_y, hr_patch_size);
                if (!hr_patch) {
                    patch_destroy(lr_patch);
                    continue;
                }

                // 分配并提取特征
                lr_samples[total_samples] = (float*)malloc(atom_dim * sizeof(float));
                hr_samples[total_samples] = (float*)malloc(atom_dim * sizeof(float));

                if (lr_samples[total_samples] && hr_samples[total_samples]) {
                    extract_patch_features(lr_patch, lr_samples[total_samples]);
                    extract_patch_features(hr_patch, hr_samples[total_samples]);
                    total_samples++;
                }

                patch_destroy(lr_patch);
                patch_destroy(hr_patch);
            }
        }

        image_destroy(hr_img);
    }

    if (total_samples == 0) {
        dictionary_destroy(dict);
        free(lr_samples);
        free(hr_samples);
        return NULL;
    }

    // 字典学习迭代
    float **sparse_codes = (float**)malloc(total_samples * sizeof(float*));
    for (int i = 0; i < total_samples; i++) {
        sparse_codes[i] = (float*)calloc(dict_size, sizeof(float));
    }

    for (int iter = 0; iter < params->training_iterations; iter++) {
        // 稀疏编码阶段
        for (int i = 0; i < total_samples; i++) {
            orthogonal_matching_pursuit(
                lr_samples[i], atom_dim,
                dict->lr_atoms, dict_size,
                params->sparsity, params->sparse_iterations,
                sparse_codes[i]
            );
        }

        // 字典更新阶段
        update_dictionary_ksvd(dict, lr_samples, sparse_codes, total_samples);

        // 同时更新高分辨率字典
        update_dictionary_ksvd(dict, hr_samples, sparse_codes, total_samples);
        memcpy(dict->hr_atoms, dict->atoms, dict_size * atom_dim * sizeof(float));

        printf("Dictionary learning iteration %d/%d\n", iter + 1, params->training_iterations);
    }

    // 清理
    for (int i = 0; i < total_samples; i++) {
        free(lr_samples[i]);
        free(hr_samples[i]);
        free(sparse_codes[i]);
    }
    free(lr_samples);
    free(hr_samples);
    free(sparse_codes);

    return dict;
}

int dictionary_save(const Dictionary *dict, const char *filename)
{
    if (!dict || !filename) {
        return SR_ERROR_NULL_POINTER;
    }

    FILE *fp = fopen(filename, "wb");
    if (!fp) {
        return SR_ERROR_FILE_IO;
    }

    // 写入头信息
    fwrite(&dict->size, sizeof(int), 1, fp);
    fwrite(&dict->atom_dim, sizeof(int), 1, fp);

    // 写入字典数据
    fwrite(dict->atoms, sizeof(float), dict->size * dict->atom_dim, fp);
    fwrite(dict->lr_atoms, sizeof(float), dict->size * dict->atom_dim, fp);
    fwrite(dict->hr_atoms, sizeof(float), dict->size * dict->atom_dim, fp);

    fclose(fp);
    return SR_SUCCESS;
}

Dictionary* dictionary_load(const char *filename)
{
    if (!filename) {
        return NULL;
    }

    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        return NULL;
    }

    int size, atom_dim;
    if (fread(&size, sizeof(int), 1, fp) != 1 ||
        fread(&atom_dim, sizeof(int), 1, fp) != 1) {
        fclose(fp);
        return NULL;
    }

    Dictionary *dict = dictionary_create(size, atom_dim);
    if (!dict) {
        fclose(fp);
        return NULL;
    }

    // 读取字典数据
    if (fread(dict->atoms, sizeof(float), size * atom_dim, fp) != (size_t)(size * atom_dim) ||
        fread(dict->lr_atoms, sizeof(float), size * atom_dim, fp) != (size_t)(size * atom_dim) ||
        fread(dict->hr_atoms, sizeof(float), size * atom_dim, fp) != (size_t)(size * atom_dim)) {
        dictionary_destroy(dict);
        fclose(fp);
        return NULL;
    }

    fclose(fp);
    return dict;
}
// ============================================================================
// 卷积神经网络辅助函数
// ============================================================================

// 卷积层
static int convolve_2d(const float *input, int in_width, int in_height, int in_channels,
                      const float *kernel, int kernel_size, int out_channels,
                      const float *bias, float *output, ActivationType activation)
{
    if (!input || !kernel || !output) {
        return SR_ERROR_NULL_POINTER;
    }

    int out_width = in_width;
    int out_height = in_height;
    int pad = kernel_size / 2;

    for (int oc = 0; oc < out_channels; oc++) {
        for (int y = 0; y < out_height; y++) {
            for (int x = 0; x < out_width; x++) {
                float sum = 0.0f;

                // 卷积操作
                for (int ic = 0; ic < in_channels; ic++) {
                    for (int ky = 0; ky < kernel_size; ky++) {
                        for (int kx = 0; kx < kernel_size; kx++) {
                            int in_y = y + ky - pad;
                            int in_x = x + kx - pad;

                            // 边界处理（零填充）
                            if (in_y >= 0 && in_y < in_height && 
                                in_x >= 0 && in_x < in_width) {
                                int in_idx = (in_y * in_width + in_x) * in_channels + ic;
                                int k_idx = ((oc * in_channels + ic) * kernel_size + ky) * kernel_size + kx;
                                sum += input[in_idx] * kernel[k_idx];
                            }
                        }
                    }
                }

                // 添加偏置
                if (bias) {
                    sum += bias[oc];
                }

                // 激活函数
                switch (activation) {
                    case ACTIVATION_RELU:
                        sum = (sum > 0.0f) ? sum : 0.0f;
                        break;
                    case ACTIVATION_LEAKY_RELU:
                        sum = (sum > 0.0f) ? sum : 0.01f * sum;
                        break;
                    case ACTIVATION_SIGMOID:
                        sum = 1.0f / (1.0f + expf(-sum));
                        break;
                    case ACTIVATION_TANH:
                        sum = tanhf(sum);
                        break;
                    case ACTIVATION_PRELU:
                        sum = (sum > 0.0f) ? sum : 0.25f * sum;
                        break;
                    case ACTIVATION_LINEAR:
                    default:
                        break;
                }

                int out_idx = (y * out_width + x) * out_channels + oc;
                output[out_idx] = sum;
            }
        }
    }

    return SR_SUCCESS;
}

// 转置卷积（上采样）
static int conv_transpose_2d(const float *input, int in_width, int in_height, int in_channels,
                            const float *kernel, int kernel_size, int stride,
                            int out_channels, const float *bias, float *output,
                            ActivationType activation)
{
    if (!input || !kernel || !output) {
        return SR_ERROR_NULL_POINTER;
    }

    int out_width = in_width * stride;
    int out_height = in_height * stride;

    // 初始化输出
    memset(output, 0, out_width * out_height * out_channels * sizeof(float));

    // 转置卷积
    for (int ic = 0; ic < in_channels; ic++) {
        for (int y = 0; y < in_height; y++) {
            for (int x = 0; x < in_width; x++) {
                int in_idx = (y * in_width + x) * in_channels + ic;
                float in_val = input[in_idx];

                for (int oc = 0; oc < out_channels; oc++) {
                    for (int ky = 0; ky < kernel_size; ky++) {
                        for (int kx = 0; kx < kernel_size; kx++) {
                            int out_y = y * stride + ky;
                            int out_x = x * stride + kx;

                            if (out_y < out_height && out_x < out_width) {
                                int out_idx = (out_y * out_width + out_x) * out_channels + oc;
                                int k_idx = ((oc * in_channels + ic) * kernel_size + ky) * kernel_size + kx;
                                output[out_idx] += in_val * kernel[k_idx];
                            }
                        }
                    }
                }
            }
        }
    }

    // 添加偏置和激活
    for (int i = 0; i < out_width * out_height; i++) {
        for (int oc = 0; oc < out_channels; oc++) {
            int idx = i * out_channels + oc;
            
            if (bias) {
                output[idx] += bias[oc];
            }

            // 激活函数
            switch (activation) {
                case ACTIVATION_RELU:
                    output[idx] = (output[idx] > 0.0f) ? output[idx] : 0.0f;
                    break;
                case ACTIVATION_LEAKY_RELU:
                    output[idx] = (output[idx] > 0.0f) ? output[idx] : 0.01f * output[idx];
                    break;
                case ACTIVATION_SIGMOID:
                    output[idx] = 1.0f / (1.0f + expf(-output[idx]));
                    break;
                case ACTIVATION_TANH:
                    output[idx] = tanhf(output[idx]);
                    break;
                case ACTIVATION_PRELU:
                    output[idx] = (output[idx] > 0.0f) ? output[idx] : 0.25f * output[idx];
                    break;
                case ACTIVATION_LINEAR:
                default:
                    break;
            }
        }
    }

    return SR_SUCCESS;
}

// 批归一化
static int batch_normalize(float *data, int width, int height, int channels,
                          const float *gamma, const float *beta,
                          const float *mean, const float *variance, float epsilon)
{
    if (!data || !gamma || !beta || !mean || !variance) {
        return SR_ERROR_NULL_POINTER;
    }

    int spatial_size = width * height;

    for (int c = 0; c < channels; c++) {
        float std = sqrtf(variance[c] + epsilon);
        
        for (int i = 0; i < spatial_size; i++) {
            int idx = i * channels + c;
            // 归一化
            data[idx] = (data[idx] - mean[c]) / std;
            // 缩放和平移
            data[idx] = gamma[c] * data[idx] + beta[c];
        }
    }

    return SR_SUCCESS;
}

// ============================================================================
// SRCNN (Super-Resolution Convolutional Neural Network)
// ============================================================================

SRCNNModel* srcnn_create(int scale_factor)
{
    if (scale_factor <= 1) {
        return NULL;
    }

    SRCNNModel *model = (SRCNNModel*)malloc(sizeof(SRCNNModel));
    if (!model) {
        return NULL;
    }

    model->scale_factor = scale_factor;
    model->num_layers = 3;

    // 层配置：特征提取 -> 非线性映射 -> 重建
    model->layer_configs = (LayerConfig*)malloc(3 * sizeof(LayerConfig));
    if (!model->layer_configs) {
        free(model);
        return NULL;
    }

    // 第一层：特征提取 (9x9, 64 filters)
    model->layer_configs[0].kernel_size = 9;
    model->layer_configs[0].num_filters = 64;
    model->layer_configs[0].activation = ACTIVATION_RELU;

    // 第二层：非线性映射 (1x1, 32 filters)
    model->layer_configs[1].kernel_size = 1;
    model->layer_configs[1].num_filters = 32;
    model->layer_configs[1].activation = ACTIVATION_RELU;

    // 第三层：重建 (5x5, channels filters)
    model->layer_configs[2].kernel_size = 5;
    model->layer_configs[2].num_filters = 3; // RGB
    model->layer_configs[2].activation = ACTIVATION_LINEAR;

    // 分配权重和偏置
    model->weights = (float**)malloc(3 * sizeof(float*));
    model->biases = (float**)malloc(3 * sizeof(float*));

    if (!model->weights || !model->biases) {
        srcnn_destroy(model);
        return NULL;
    }

    // 初始化权重（简化的随机初始化）
    int in_channels = 3; // RGB输入
    for (int i = 0; i < 3; i++) {
        int k_size = model->layer_configs[i].kernel_size;
        int n_filters = model->layer_configs[i].num_filters;
        int weight_size = k_size * k_size * in_channels * n_filters;

        model->weights[i] = (float*)malloc(weight_size * sizeof(float));
        model->biases[i] = (float*)calloc(n_filters, sizeof(float));

        if (!model->weights[i] || !model->biases[i]) {
            srcnn_destroy(model);
            return NULL;
        }

        // He初始化
        float std = sqrtf(2.0f / (k_size * k_size * in_channels));
        for (int j = 0; j < weight_size; j++) {
            model->weights[i][j] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * std;
        }

        in_channels = n_filters;
    }

    model->is_trained = false;

    return model;
}

void srcnn_destroy(SRCNNModel *model)
{
    if (model) {
        if (model->layer_configs) {
            free(model->layer_configs);
        }
        if (model->weights) {
            for (int i = 0; i < model->num_layers; i++) {
                if (model->weights[i]) {
                    free(model->weights[i]);
                }
            }
            free(model->weights);
        }
        if (model->biases) {
            for (int i = 0; i < model->num_layers; i++) {
                if (model->biases[i]) {
                    free(model->biases[i]);
                }
            }
            free(model->biases);
        }
        free(model);
    }
}

int srcnn_forward(const SRCNNModel *model, const Image *input, Image *output)
{
    if (!model || !input || !output) {
        return SR_ERROR_NULL_POINTER;
    }

    if (!model->is_trained) {
        return SR_ERROR_MODEL_NOT_TRAINED;
    }

    // 先用双三次插值上采样到目标大小
    Image *upsampled = interpolate_bicubic(input, (float)model->scale_factor, -0.5f);
    if (!upsampled) {
        return SR_ERROR_MEMORY_ALLOCATION;
    }

    int width = upsampled->width;
    int height = upsampled->height;
    int channels = upsampled->channels;

    // 分配中间层缓冲区
    float **layer_outputs = (float**)malloc((model->num_layers + 1) * sizeof(float*));
    if (!layer_outputs) {
        image_destroy(upsampled);
        return SR_ERROR_MEMORY_ALLOCATION;
    }

    // 输入层
    layer_outputs[0] = upsampled->data;

    // 前向传播
    int current_channels = channels;
    for (int i = 0; i < model->num_layers; i++) {
        int out_channels = model->layer_configs[i].num_filters;
        int size = width * height * out_channels;

        layer_outputs[i + 1] = (float*)malloc(size * sizeof(float));
        if (!layer_outputs[i + 1]) {
            for (int j = 1; j <= i; j++) {
                free(layer_outputs[j]);
            }
            free(layer_outputs);
            image_destroy(upsampled);
            return SR_ERROR_MEMORY_ALLOCATION;
        }

        // 卷积
        int ret = convolve_2d(
            layer_outputs[i], width, height, current_channels,
            model->weights[i], model->layer_configs[i].kernel_size,
            out_channels, model->biases[i],
            layer_outputs[i + 1], model->layer_configs[i].activation
        );

        if (ret != SR_SUCCESS) {
            for (int j = 1; j <= i + 1; j++) {
                free(layer_outputs[j]);
            }
            free(layer_outputs);
            image_destroy(upsampled);
            return ret;
        }

        current_channels = out_channels;
    }

    // 复制输出
    memcpy(output->data, layer_outputs[model->num_layers],
           width * height * current_channels * sizeof(float));

    // 清理
    for (int i = 1; i <= model->num_layers; i++) {
        free(layer_outputs[i]);
    }
    free(layer_outputs);
    image_destroy(upsampled);

    return SR_SUCCESS;
}

int super_resolve_srcnn(const Image *input, const SRCNNModel *model, SRResult *result)
{
    if (!input || !model || !result) {
        return SR_ERROR_NULL_POINTER;
    }

    clock_t start_time = clock();

    int ret = srcnn_forward(model, input, result->output);

    clock_t end_time = clock();
    result->computation_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;

    return ret;
}

// ============================================================================
// ESPCN (Efficient Sub-Pixel Convolutional Neural Network)
// ============================================================================

ESPCNModel* espcn_create(int scale_factor)
{
    if (scale_factor <= 1) {
        return NULL;
    }

    ESPCNModel *model = (ESPCNModel*)malloc(sizeof(ESPCNModel));
    if (!model) {
        return NULL;
    }

    model->scale_factor = scale_factor;
    model->num_layers = 3;

    // 层配置
    model->layer_configs = (LayerConfig*)malloc(3 * sizeof(LayerConfig));
    if (!model->layer_configs) {
        free(model);
        return NULL;
    }

    // 第一层：特征提取 (5x5, 64 filters)
    model->layer_configs[0].kernel_size = 5;
    model->layer_configs[0].num_filters = 64;
    model->layer_configs[0].activation = ACTIVATION_RELU;

    // 第二层：非线性映射 (3x3, 32 filters)
    model->layer_configs[1].kernel_size = 3;
    model->layer_configs[1].num_filters = 32;
    model->layer_configs[1].activation = ACTIVATION_RELU;

    // 第三层：子像素卷积 (3x3, channels * scale^2 filters)
    model->layer_configs[2].kernel_size = 3;
    model->layer_configs[2].num_filters = 3 * scale_factor * scale_factor;
    model->layer_configs[2].activation = ACTIVATION_LINEAR;

    // 分配权重和偏置
    model->weights = (float**)malloc(3 * sizeof(float*));
    model->biases = (float**)malloc(3 * sizeof(float*));

    if (!model->weights || !model->biases) {
        espcn_destroy(model);
        return NULL;
    }

    // 初始化权重
    int in_channels = 3;
    for (int i = 0; i < 3; i++) {
        int k_size = model->layer_configs[i].kernel_size;
        int n_filters = model->layer_configs[i].num_filters;
        int weight_size = k_size * k_size * in_channels * n_filters;

        model->weights[i] = (float*)malloc(weight_size * sizeof(float));
        model->biases[i] = (float*)calloc(n_filters, sizeof(float));

        if (!model->weights[i] || !model->biases[i]) {
            espcn_destroy(model);
            return NULL;
        }

        float std = sqrtf(2.0f / (k_size * k_size * in_channels));
        for (int j = 0; j < weight_size; j++) {
            model->weights[i][j] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * std;
        }

        in_channels = n_filters;
    }

    model->is_trained = false;

    return model;
}

void espcn_destroy(ESPCNModel *model)
{
    if (model) {
        if (model->layer_configs) {
            free(model->layer_configs);
        }
        if (model->weights) {
            for (int i = 0; i < model->num_layers; i++) {
                if (model->weights[i]) {
                    free(model->weights[i]);
                }
            }
            free(model->weights);
        }
        if (model->biases) {
            for (int i = 0; i < model->num_layers; i++) {
                if (model->biases[i]) {
                    free(model->biases[i]);
                }
            }
            free(model->biases);
        }
        free(model);
    }
}

// 子像素重排（Pixel Shuffle）
static int pixel_shuffle(const float *input, int width, int height, int channels,
                        int scale_factor, float *output)
{
    if (!input || !output || scale_factor <= 1) {
        return SR_ERROR_INVALID_PARAMETER;
    }

    int out_width = width * scale_factor;
    int out_height = height * scale_factor;
    int out_channels = channels / (scale_factor * scale_factor);

    if (channels % (scale_factor * scale_factor) != 0) {
        return SR_ERROR_INVALID_DIMENSIONS;
    }

    for (int c = 0; c < out_channels; c++) {
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                for (int dy = 0; dy < scale_factor; dy++) {
                    for (int dx = 0; dx < scale_factor; dx++) {
                        int in_c = c * scale_factor * scale_factor + dy * scale_factor + dx;
                        int in_idx = (y * width + x) * channels + in_c;

                        int out_y = y * scale_factor + dy;
                        int out_x = x * scale_factor + dx;
                        int out_idx = (out_y * out_width + out_x) * out_channels + c;

                        output[out_idx] = input[in_idx];
                    }
                }
            }
        }
    }

    return SR_SUCCESS;
}

int espcn_forward(const ESPCNModel *model, const Image *input, Image *output)
{
    if (!model || !input || !output) {
        return SR_ERROR_NULL_POINTER;
    }

    if (!model->is_trained) {
        return SR_ERROR_MODEL_NOT_TRAINED;
    }

    int width = input->width;
    int height = input->height;
    int channels = input->channels;

    // 分配中间层缓冲区
    float **layer_outputs = (float**)malloc((model->num_layers + 1) * sizeof(float*));
    if (!layer_outputs) {
        return SR_ERROR_MEMORY_ALLOCATION;
    }

    layer_outputs[0] = input->data;

    // 前向传播（在低分辨率空间）
    int current_channels = channels;
    for (int i = 0; i < model->num_layers; i++) {
        int out_channels = model->layer_configs[i].num_filters;
        int size = width * height * out_channels;

        layer_outputs[i + 1] = (float*)malloc(size * sizeof(float));
        if (!layer_outputs[i + 1]) {
            for (int j = 1; j <= i; j++) {
                free(layer_outputs[j]);
            }
            free(layer_outputs);
            return SR_ERROR_MEMORY_ALLOCATION;
        }

        int ret = convolve_2d(
            layer_outputs[i], width, height, current_channels,
            model->weights[i], model->layer_configs[i].kernel_size,
            out_channels, model->biases[i],
            layer_outputs[i + 1], model->layer_configs[i].activation
        );

        if (ret != SR_SUCCESS) {
            for (int j = 1; j <= i + 1; j++) {
                free(layer_outputs[j]);
            }
            free(layer_outputs);
            return ret;
        }

        current_channels = out_channels;
    }

    // 子像素重排
    int ret = pixel_shuffle(
        layer_outputs[model->num_layers],
        width, height, current_channels,
        model->scale_factor, output->data
    );

    // 清理
    for (int i = 1; i <= model->num_layers; i++) {
        free(layer_outputs[i]);
    }
    free(layer_outputs);

    return ret;
}

int super_resolve_espcn(const Image *input, const ESPCNModel *model, SRResult *result)
{
    if (!input || !model || !result) {
        return SR_ERROR_NULL_POINTER;
    }

    clock_t start_time = clock();

    int ret = espcn_forward(model, input, result->output);

    clock_t end_time = clock();
    result->computation_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;

    return ret;
}
// ============================================================================
// EDSR (Enhanced Deep Super-Resolution)
// ============================================================================

EDSRModel* edsr_create(int scale_factor, int num_res_blocks)
{
    if (scale_factor <= 1 || num_res_blocks <= 0) {
        return NULL;
    }

    EDSRModel *model = (EDSRModel*)malloc(sizeof(EDSRModel));
    if (!model) {
        return NULL;
    }

    model->scale_factor = scale_factor;
    model->num_res_blocks = num_res_blocks;
    model->num_features = 64;

    // 分配残差块
    model->res_blocks = (ResidualBlock*)malloc(num_res_blocks * sizeof(ResidualBlock));
    if (!model->res_blocks) {
        free(model);
        return NULL;
    }

    // 初始化每个残差块
    for (int i = 0; i < num_res_blocks; i++) {
        ResidualBlock *block = &model->res_blocks[i];
        
        // 第一个卷积层 (3x3, 64 filters)
        int weight_size1 = 3 * 3 * model->num_features * model->num_features;
        block->conv1_weights = (float*)malloc(weight_size1 * sizeof(float));
        block->conv1_bias = (float*)calloc(model->num_features, sizeof(float));

        // 第二个卷积层 (3x3, 64 filters)
        int weight_size2 = 3 * 3 * model->num_features * model->num_features;
        block->conv2_weights = (float*)malloc(weight_size2 * sizeof(float));
        block->conv2_bias = (float*)calloc(model->num_features, sizeof(float));

        if (!block->conv1_weights || !block->conv1_bias ||
            !block->conv2_weights || !block->conv2_bias) {
            edsr_destroy(model);
            return NULL;
        }

        // He初始化
        float std = sqrtf(2.0f / (3 * 3 * model->num_features));
        for (int j = 0; j < weight_size1; j++) {
            block->conv1_weights[j] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * std;
        }
        for (int j = 0; j < weight_size2; j++) {
            block->conv2_weights[j] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * std;
        }

        block->res_scale = 0.1f; // 残差缩放因子
    }

    // 初始卷积层
    int init_weight_size = 3 * 3 * 3 * model->num_features; // RGB -> 64 features
    model->init_conv_weights = (float*)malloc(init_weight_size * sizeof(float));
    model->init_conv_bias = (float*)calloc(model->num_features, sizeof(float));

    // 上采样层
    int upsample_weight_size = 3 * 3 * model->num_features * (model->num_features * scale_factor * scale_factor);
    model->upsample_weights = (float*)malloc(upsample_weight_size * sizeof(float));
    model->upsample_bias = (float*)calloc(model->num_features * scale_factor * scale_factor, sizeof(float));

    // 最终卷积层
    int final_weight_size = 3 * 3 * model->num_features * 3; // 64 features -> RGB
    model->final_conv_weights = (float*)malloc(final_weight_size * sizeof(float));
    model->final_conv_bias = (float*)calloc(3, sizeof(float));

    if (!model->init_conv_weights || !model->init_conv_bias ||
        !model->upsample_weights || !model->upsample_bias ||
        !model->final_conv_weights || !model->final_conv_bias) {
        edsr_destroy(model);
        return NULL;
    }

    // 初始化权重
    float std = sqrtf(2.0f / (3 * 3 * 3));
    for (int i = 0; i < init_weight_size; i++) {
        model->init_conv_weights[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * std;
    }

    std = sqrtf(2.0f / (3 * 3 * model->num_features));
    for (int i = 0; i < upsample_weight_size; i++) {
        model->upsample_weights[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * std;
    }
    for (int i = 0; i < final_weight_size; i++) {
        model->final_conv_weights[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * std;
    }

    model->is_trained = false;

    return model;
}

void edsr_destroy(EDSRModel *model)
{
    if (model) {
        if (model->res_blocks) {
            for (int i = 0; i < model->num_res_blocks; i++) {
                ResidualBlock *block = &model->res_blocks[i];
                if (block->conv1_weights) free(block->conv1_weights);
                if (block->conv1_bias) free(block->conv1_bias);
                if (block->conv2_weights) free(block->conv2_weights);
                if (block->conv2_bias) free(block->conv2_bias);
            }
            free(model->res_blocks);
        }
        if (model->init_conv_weights) free(model->init_conv_weights);
        if (model->init_conv_bias) free(model->init_conv_bias);
        if (model->upsample_weights) free(model->upsample_weights);
        if (model->upsample_bias) free(model->upsample_bias);
        if (model->final_conv_weights) free(model->final_conv_weights);
        if (model->final_conv_bias) free(model->final_conv_bias);
        free(model);
    }
}

// 残差块前向传播
static int residual_block_forward(const ResidualBlock *block, const float *input,
                                  int width, int height, int channels,
                                  float *output)
{
    if (!block || !input || !output) {
        return SR_ERROR_NULL_POINTER;
    }

    int size = width * height * channels;
    
    // 分配临时缓冲区
    float *temp1 = (float*)malloc(size * sizeof(float));
    float *temp2 = (float*)malloc(size * sizeof(float));
    
    if (!temp1 || !temp2) {
        free(temp1);
        free(temp2);
        return SR_ERROR_MEMORY_ALLOCATION;
    }

    // 第一个卷积 + ReLU
    int ret = convolve_2d(input, width, height, channels,
                         block->conv1_weights, 3, channels,
                         block->conv1_bias, temp1, ACTIVATION_RELU);
    if (ret != SR_SUCCESS) {
        free(temp1);
        free(temp2);
        return ret;
    }

    // 第二个卷积（无激活）
    ret = convolve_2d(temp1, width, height, channels,
                     block->conv2_weights, 3, channels,
                     block->conv2_bias, temp2, ACTIVATION_LINEAR);
    if (ret != SR_SUCCESS) {
        free(temp1);
        free(temp2);
        return ret;
    }

    // 残差连接：output = input + res_scale * conv2(conv1(input))
    for (int i = 0; i < size; i++) {
        output[i] = input[i] + block->res_scale * temp2[i];
    }

    free(temp1);
    free(temp2);

    return SR_SUCCESS;
}

int edsr_forward(const EDSRModel *model, const Image *input, Image *output)
{
    if (!model || !input || !output) {
        return SR_ERROR_NULL_POINTER;
    }

    if (!model->is_trained) {
        return SR_ERROR_MODEL_NOT_TRAINED;
    }

    int width = input->width;
    int height = input->height;
    int channels = input->channels;

    // 初始卷积
    int feature_size = width * height * model->num_features;
    float *features = (float*)malloc(feature_size * sizeof(float));
    if (!features) {
        return SR_ERROR_MEMORY_ALLOCATION;
    }

    int ret = convolve_2d(input->data, width, height, channels,
                         model->init_conv_weights, 3, model->num_features,
                         model->init_conv_bias, features, ACTIVATION_LINEAR);
    if (ret != SR_SUCCESS) {
        free(features);
        return ret;
    }

    // 保存初始特征用于全局残差连接
    float *init_features = (float*)malloc(feature_size * sizeof(float));
    if (!init_features) {
        free(features);
        return SR_ERROR_MEMORY_ALLOCATION;
    }
    memcpy(init_features, features, feature_size * sizeof(float));

    // 残差块
    float *temp_features = (float*)malloc(feature_size * sizeof(float));
    if (!temp_features) {
        free(features);
        free(init_features);
        return SR_ERROR_MEMORY_ALLOCATION;
    }

    for (int i = 0; i < model->num_res_blocks; i++) {
        ret = residual_block_forward(&model->res_blocks[i], features,
                                     width, height, model->num_features,
                                     temp_features);
        if (ret != SR_SUCCESS) {
            free(features);
            free(init_features);
            free(temp_features);
            return ret;
        }
        
        // 交换缓冲区
        float *swap = features;
        features = temp_features;
        temp_features = swap;
    }

    // 全局残差连接
    for (int i = 0; i < feature_size; i++) {
        features[i] += init_features[i];
    }

    free(init_features);
    free(temp_features);

    // 上采样
    int upsampled_channels = model->num_features * model->scale_factor * model->scale_factor;
    int upsampled_size = width * height * upsampled_channels;
    float *upsampled = (float*)malloc(upsampled_size * sizeof(float));
    if (!upsampled) {
        free(features);
        return SR_ERROR_MEMORY_ALLOCATION;
    }

    ret = convolve_2d(features, width, height, model->num_features,
                     model->upsample_weights, 3, upsampled_channels,
                     model->upsample_bias, upsampled, ACTIVATION_LINEAR);
    free(features);

    if (ret != SR_SUCCESS) {
        free(upsampled);
        return ret;
    }

    // 子像素重排
    int out_width = width * model->scale_factor;
    int out_height = height * model->scale_factor;
    float *shuffled = (float*)malloc(out_width * out_height * model->num_features * sizeof(float));
    if (!shuffled) {
        free(upsampled);
        return SR_ERROR_MEMORY_ALLOCATION;
    }

    ret = pixel_shuffle(upsampled, width, height, upsampled_channels,
                       model->scale_factor, shuffled);
    free(upsampled);

    if (ret != SR_SUCCESS) {
        free(shuffled);
        return ret;
    }

    // 最终卷积
    ret = convolve_2d(shuffled, out_width, out_height, model->num_features,
                     model->final_conv_weights, 3, 3,
                     model->final_conv_bias, output->data, ACTIVATION_LINEAR);
    free(shuffled);

    return ret;
}

int super_resolve_edsr(const Image *input, const EDSRModel *model, SRResult *result)
{
    if (!input || !model || !result) {
        return SR_ERROR_NULL_POINTER;
    }

    clock_t start_time = clock();

    int ret = edsr_forward(model, input, result->output);

    clock_t end_time = clock();
    result->computation_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;

    return ret;
}

// ============================================================================
// 图像质量评估
// ============================================================================

// 计算PSNR (Peak Signal-to-Noise Ratio)
float calculate_psnr(const Image *img1, const Image *img2)
{
    if (!img1 || !img2) {
        return -1.0f;
    }

    if (img1->width != img2->width || img1->height != img2->height ||
        img1->channels != img2->channels) {
        return -1.0f;
    }

    int size = img1->width * img1->height * img1->channels;
    double mse = 0.0;

    for (int i = 0; i < size; i++) {
        double diff = img1->data[i] - img2->data[i];
        mse += diff * diff;
    }

    mse /= size;

    if (mse < EPSILON) {
        return 100.0f; // 图像完全相同
    }

    // 假设像素值范围是[0, 1]
    double psnr = 10.0 * log10(1.0 / mse);

    return (float)psnr;
}

// 计算SSIM (Structural Similarity Index)
float calculate_ssim(const Image *img1, const Image *img2)
{
    if (!img1 || !img2) {
        return -1.0f;
    }

    if (img1->width != img2->width || img1->height != img2->height ||
        img1->channels != img2->channels) {
        return -1.0f;
    }

    // 转换为灰度图
    Image *gray1 = image_to_grayscale(img1);
    Image *gray2 = image_to_grayscale(img2);

    if (!gray1 || !gray2) {
        image_destroy(gray1);
        image_destroy(gray2);
        return -1.0f;
    }

    // SSIM参数
    const float C1 = 0.01f * 0.01f;
    const float C2 = 0.03f * 0.03f;
    const int window_size = 11;
    const int radius = window_size / 2;

    double ssim_sum = 0.0;
    int count = 0;

    // 使用滑动窗口计算局部SSIM
    for (int y = radius; y < gray1->height - radius; y++) {
        for (int x = radius; x < gray1->width - radius; x++) {
            // 计算窗口内的统计量
            double mean1 = 0.0, mean2 = 0.0;
            double var1 = 0.0, var2 = 0.0, covar = 0.0;
            int window_count = 0;

            for (int wy = -radius; wy <= radius; wy++) {
                for (int wx = -radius; wx <= radius; wx++) {
                    int idx = ((y + wy) * gray1->width + (x + wx));
                    float val1 = gray1->data[idx];
                    float val2 = gray2->data[idx];

                    mean1 += val1;
                    mean2 += val2;
                    window_count++;
                }
            }

            mean1 /= window_count;
            mean2 /= window_count;

            for (int wy = -radius; wy <= radius; wy++) {
                for (int wx = -radius; wx <= radius; wx++) {
                    int idx = ((y + wy) * gray1->width + (x + wx));
                    float val1 = gray1->data[idx];
                    float val2 = gray2->data[idx];

                    double diff1 = val1 - mean1;
                    double diff2 = val2 - mean2;

                    var1 += diff1 * diff1;
                    var2 += diff2 * diff2;
                    covar += diff1 * diff2;
                }
            }

            var1 /= (window_count - 1);
            var2 /= (window_count - 1);
            covar /= (window_count - 1);

            // 计算SSIM
            double numerator = (2.0 * mean1 * mean2 + C1) * (2.0 * covar + C2);
            double denominator = (mean1 * mean1 + mean2 * mean2 + C1) * (var1 + var2 + C2);

            double ssim = numerator / denominator;
            ssim_sum += ssim;
            count++;
        }
    }

    image_destroy(gray1);
    image_destroy(gray2);

    return (float)(ssim_sum / count);
}

// 计算MSE (Mean Squared Error)
float calculate_mse(const Image *img1, const Image *img2)
{
    if (!img1 || !img2) {
        return -1.0f;
    }

    if (img1->width != img2->width || img1->height != img2->height ||
        img1->channels != img2->channels) {
        return -1.0f;
    }

    int size = img1->width * img1->height * img1->channels;
    double mse = 0.0;

    for (int i = 0; i < size; i++) {
        double diff = img1->data[i] - img2->data[i];
        mse += diff * diff;
    }

    return (float)(mse / size);
}

// 计算MAE (Mean Absolute Error)
float calculate_mae(const Image *img1, const Image *img2)
{
    if (!img1 || !img2) {
        return -1.0f;
    }

    if (img1->width != img2->width || img1->height != img2->height ||
        img1->channels != img2->channels) {
        return -1.0f;
    }

    int size = img1->width * img1->height * img1->channels;
    double mae = 0.0;

    for (int i = 0; i < size; i++) {
        mae += fabs(img1->data[i] - img2->data[i]);
    }

    return (float)(mae / size);
}

// 综合质量评估
int evaluate_sr_quality(const Image *sr_image, const Image *ground_truth,
                       QualityMetrics *metrics)
{
    if (!sr_image || !ground_truth || !metrics) {
        return SR_ERROR_NULL_POINTER;
    }

    metrics->psnr = calculate_psnr(sr_image, ground_truth);
    metrics->ssim = calculate_ssim(sr_image, ground_truth);
    metrics->mse = calculate_mse(sr_image, ground_truth);
    metrics->mae = calculate_mae(sr_image, ground_truth);

    return SR_SUCCESS;
}

// ============================================================================
// 模型保存和加载
// ============================================================================

int srcnn_save(const SRCNNModel *model, const char *filename)
{
    if (!model || !filename) {
        return SR_ERROR_NULL_POINTER;
    }

    FILE *fp = fopen(filename, "wb");
    if (!fp) {
        return SR_ERROR_FILE_IO;
    }

    // 写入模型配置
    fwrite(&model->scale_factor, sizeof(int), 1, fp);
    fwrite(&model->num_layers, sizeof(int), 1, fp);
    fwrite(&model->is_trained, sizeof(bool), 1, fp);

    // 写入层配置
    for (int i = 0; i < model->num_layers; i++) {
        fwrite(&model->layer_configs[i].kernel_size, sizeof(int), 1, fp);
        fwrite(&model->layer_configs[i].num_filters, sizeof(int), 1, fp);
        fwrite(&model->layer_configs[i].activation, sizeof(ActivationType), 1, fp);
    }

    // 写入权重和偏置
    int in_channels = 3;
    for (int i = 0; i < model->num_layers; i++) {
        int k_size = model->layer_configs[i].kernel_size;
        int n_filters = model->layer_configs[i].num_filters;
        int weight_size = k_size * k_size * in_channels * n_filters;

        fwrite(model->weights[i], sizeof(float), weight_size, fp);
        fwrite(model->biases[i], sizeof(float), n_filters, fp);

        in_channels = n_filters;
    }

    fclose(fp);
    return SR_SUCCESS;
}

SRCNNModel* srcnn_load(const char *filename)
{
    if (!filename) {
        return NULL;
    }

    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        return NULL;
    }

    // 读取模型配置
    int scale_factor, num_layers;
    bool is_trained;

    if (fread(&scale_factor, sizeof(int), 1, fp) != 1 ||
        fread(&num_layers, sizeof(int), 1, fp) != 1 ||
        fread(&is_trained, sizeof(bool), 1, fp) != 1) {
        fclose(fp);
        return NULL;
    }

    // 创建模型
    SRCNNModel *model = srcnn_create(scale_factor);
    if (!model) {
        fclose(fp);
        return NULL;
    }

    model->is_trained = is_trained;

    // 读取层配置
    for (int i = 0; i < num_layers; i++) {
        if (fread(&model->layer_configs[i].kernel_size, sizeof(int), 1, fp) != 1 ||
            fread(&model->layer_configs[i].num_filters, sizeof(int), 1, fp) != 1 ||
            fread(&model->layer_configs[i].activation, sizeof(ActivationType), 1, fp) != 1) {
            srcnn_destroy(model);
            fclose(fp);
            return NULL;
        }
    }

    // 读取权重和偏置
    int in_channels = 3;
    for (int i = 0; i < num_layers; i++) {
        int k_size = model->layer_configs[i].kernel_size;
        int n_filters = model->layer_configs[i].num_filters;
        int weight_size = k_size * k_size * in_channels * n_filters;

        if (fread(model->weights[i], sizeof(float), weight_size, fp) != (size_t)weight_size ||
            fread(model->biases[i], sizeof(float), n_filters, fp) != (size_t)n_filters) {
            srcnn_destroy(model);
            fclose(fp);
            return NULL;
        }

        in_channels = n_filters;
    }

    fclose(fp);
    return model;
}
// ============================================================================
// 模型保存和加载（续）
// ============================================================================

int espcn_save(const ESPCNModel *model, const char *filename)
{
    if (!model || !filename) {
        return SR_ERROR_NULL_POINTER;
    }

    FILE *fp = fopen(filename, "wb");
    if (!fp) {
        return SR_ERROR_FILE_IO;
    }

    // 写入模型配置
    fwrite(&model->scale_factor, sizeof(int), 1, fp);
    fwrite(&model->num_layers, sizeof(int), 1, fp);
    fwrite(&model->is_trained, sizeof(bool), 1, fp);

    // 写入层配置
    for (int i = 0; i < model->num_layers; i++) {
        fwrite(&model->layer_configs[i].kernel_size, sizeof(int), 1, fp);
        fwrite(&model->layer_configs[i].num_filters, sizeof(int), 1, fp);
        fwrite(&model->layer_configs[i].activation, sizeof(ActivationType), 1, fp);
    }

    // 写入权重和偏置
    int in_channels = 3;
    for (int i = 0; i < model->num_layers; i++) {
        int k_size = model->layer_configs[i].kernel_size;
        int n_filters = model->layer_configs[i].num_filters;
        int weight_size = k_size * k_size * in_channels * n_filters;

        fwrite(model->weights[i], sizeof(float), weight_size, fp);
        fwrite(model->biases[i], sizeof(float), n_filters, fp);

        in_channels = n_filters;
    }

    fclose(fp);
    return SR_SUCCESS;
}

ESPCNModel* espcn_load(const char *filename)
{
    if (!filename) {
        return NULL;
    }

    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        return NULL;
    }

    // 读取模型配置
    int scale_factor, num_layers;
    bool is_trained;

    if (fread(&scale_factor, sizeof(int), 1, fp) != 1 ||
        fread(&num_layers, sizeof(int), 1, fp) != 1 ||
        fread(&is_trained, sizeof(bool), 1, fp) != 1) {
        fclose(fp);
        return NULL;
    }

    // 创建模型
    ESPCNModel *model = espcn_create(scale_factor);
    if (!model) {
        fclose(fp);
        return NULL;
    }

    model->is_trained = is_trained;

    // 读取层配置
    for (int i = 0; i < num_layers; i++) {
        if (fread(&model->layer_configs[i].kernel_size, sizeof(int), 1, fp) != 1 ||
            fread(&model->layer_configs[i].num_filters, sizeof(int), 1, fp) != 1 ||
            fread(&model->layer_configs[i].activation, sizeof(ActivationType), 1, fp) != 1) {
            espcn_destroy(model);
            fclose(fp);
            return NULL;
        }
    }

    // 读取权重和偏置
    int in_channels = 3;
    for (int i = 0; i < num_layers; i++) {
        int k_size = model->layer_configs[i].kernel_size;
        int n_filters = model->layer_configs[i].num_filters;
        int weight_size = k_size * k_size * in_channels * n_filters;

        if (fread(model->weights[i], sizeof(float), weight_size, fp) != (size_t)weight_size ||
            fread(model->biases[i], sizeof(float), n_filters, fp) != (size_t)n_filters) {
            espcn_destroy(model);
            fclose(fp);
            return NULL;
        }

        in_channels = n_filters;
    }

    fclose(fp);
    return model;
}

int edsr_save(const EDSRModel *model, const char *filename)
{
    if (!model || !filename) {
        return SR_ERROR_NULL_POINTER;
    }

    FILE *fp = fopen(filename, "wb");
    if (!fp) {
        return SR_ERROR_FILE_IO;
    }

    // 写入模型配置
    fwrite(&model->scale_factor, sizeof(int), 1, fp);
    fwrite(&model->num_res_blocks, sizeof(int), 1, fp);
    fwrite(&model->num_features, sizeof(int), 1, fp);
    fwrite(&model->is_trained, sizeof(bool), 1, fp);

    // 写入残差块
    for (int i = 0; i < model->num_res_blocks; i++) {
        ResidualBlock *block = &model->res_blocks[i];
        int weight_size = 3 * 3 * model->num_features * model->num_features;

        fwrite(block->conv1_weights, sizeof(float), weight_size, fp);
        fwrite(block->conv1_bias, sizeof(float), model->num_features, fp);
        fwrite(block->conv2_weights, sizeof(float), weight_size, fp);
        fwrite(block->conv2_bias, sizeof(float), model->num_features, fp);
        fwrite(&block->res_scale, sizeof(float), 1, fp);
    }

    // 写入其他层
    int init_weight_size = 3 * 3 * 3 * model->num_features;
    fwrite(model->init_conv_weights, sizeof(float), init_weight_size, fp);
    fwrite(model->init_conv_bias, sizeof(float), model->num_features, fp);

    int upsample_weight_size = 3 * 3 * model->num_features * 
                               (model->num_features * model->scale_factor * model->scale_factor);
    fwrite(model->upsample_weights, sizeof(float), upsample_weight_size, fp);
    fwrite(model->upsample_bias, sizeof(float), 
           model->num_features * model->scale_factor * model->scale_factor, fp);

    int final_weight_size = 3 * 3 * model->num_features * 3;
    fwrite(model->final_conv_weights, sizeof(float), final_weight_size, fp);
    fwrite(model->final_conv_bias, sizeof(float), 3, fp);

    fclose(fp);
    return SR_SUCCESS;
}

EDSRModel* edsr_load(const char *filename)
{
    if (!filename) {
        return NULL;
    }

    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        return NULL;
    }

    // 读取模型配置
    int scale_factor, num_res_blocks, num_features;
    bool is_trained;

    if (fread(&scale_factor, sizeof(int), 1, fp) != 1 ||
        fread(&num_res_blocks, sizeof(int), 1, fp) != 1 ||
        fread(&num_features, sizeof(int), 1, fp) != 1 ||
        fread(&is_trained, sizeof(bool), 1, fp) != 1) {
        fclose(fp);
        return NULL;
    }

    // 创建模型
    EDSRModel *model = edsr_create(scale_factor, num_res_blocks);
    if (!model) {
        fclose(fp);
        return NULL;
    }

    model->is_trained = is_trained;

    // 读取残差块
    for (int i = 0; i < num_res_blocks; i++) {
        ResidualBlock *block = &model->res_blocks[i];
        int weight_size = 3 * 3 * num_features * num_features;

        if (fread(block->conv1_weights, sizeof(float), weight_size, fp) != (size_t)weight_size ||
            fread(block->conv1_bias, sizeof(float), num_features, fp) != (size_t)num_features ||
            fread(block->conv2_weights, sizeof(float), weight_size, fp) != (size_t)weight_size ||
            fread(block->conv2_bias, sizeof(float), num_features, fp) != (size_t)num_features ||
            fread(&block->res_scale, sizeof(float), 1, fp) != 1) {
            edsr_destroy(model);
            fclose(fp);
            return NULL;
        }
    }

    // 读取其他层
    int init_weight_size = 3 * 3 * 3 * num_features;
    if (fread(model->init_conv_weights, sizeof(float), init_weight_size, fp) != (size_t)init_weight_size ||
        fread(model->init_conv_bias, sizeof(float), num_features, fp) != (size_t)num_features) {
        edsr_destroy(model);
        fclose(fp);
        return NULL;
    }

    int upsample_weight_size = 3 * 3 * num_features * (num_features * scale_factor * scale_factor);
    int upsample_bias_size = num_features * scale_factor * scale_factor;
    if (fread(model->upsample_weights, sizeof(float), upsample_weight_size, fp) != (size_t)upsample_weight_size ||
        fread(model->upsample_bias, sizeof(float), upsample_bias_size, fp) != (size_t)upsample_bias_size) {
        edsr_destroy(model);
        fclose(fp);
        return NULL;
    }

    int final_weight_size = 3 * 3 * num_features * 3;
    if (fread(model->final_conv_weights, sizeof(float), final_weight_size, fp) != (size_t)final_weight_size ||
        fread(model->final_conv_bias, sizeof(float), 3, fp) != 3) {
        edsr_destroy(model);
        fclose(fp);
        return NULL;
    }

    fclose(fp);
    return model;
}

// ============================================================================
// 批处理和实用工具
// ============================================================================

// 批量处理图像
int batch_super_resolve(const char **input_files, int num_files,
                       const char *output_dir, SRMethod method,
                       float scale_factor, void *params)
{
    if (!input_files || !output_dir || num_files <= 0) {
        return SR_ERROR_NULL_POINTER;
    }

    int success_count = 0;
    int fail_count = 0;

    for (int i = 0; i < num_files; i++) {
        printf("Processing %d/%d: %s\n", i + 1, num_files, input_files[i]);

        // 加载图像
        Image *input = image_load(input_files[i]);
        if (!input) {
            fprintf(stderr, "Failed to load image: %s\n", input_files[i]);
            fail_count++;
            continue;
        }

        // 创建输出图像
        int out_width = (int)(input->width * scale_factor + 0.5f);
        int out_height = (int)(input->height * scale_factor + 0.5f);
        Image *output = image_create(out_width, out_height, input->channels);

        if (!output) {
            fprintf(stderr, "Failed to create output image\n");
            image_destroy(input);
            fail_count++;
            continue;
        }

        // 创建结果结构
        SRResult result;
        result.output = output;

        // 执行超分辨率
        int ret = SR_ERROR_INVALID_PARAMETER;
        
        switch (method) {
            case SR_METHOD_BICUBIC:
                ret = super_resolve_bicubic(input, scale_factor, &result);
                break;
            case SR_METHOD_LANCZOS:
                ret = super_resolve_lanczos(input, scale_factor, 3, &result);
                break;
            case SR_METHOD_EDGE_DIRECTED:
                ret = super_resolve_edge_directed(input, scale_factor, 
                                                  (EdgeDirectedParams*)params, &result);
                break;
            case SR_METHOD_SPARSE_CODING:
                ret = super_resolve_sparse_coding(input, scale_factor,
                                                  (SparseCodingParams*)params, &result);
                break;
            default:
                fprintf(stderr, "Unsupported method\n");
                break;
        }

        if (ret != SR_SUCCESS) {
            fprintf(stderr, "Super-resolution failed for: %s\n", input_files[i]);
            image_destroy(input);
            image_destroy(output);
            fail_count++;
            continue;
        }

        // 构建输出文件名
        char output_path[512];
        const char *filename = strrchr(input_files[i], '/');
        if (!filename) {
            filename = strrchr(input_files[i], '\\');
        }
        if (!filename) {
            filename = input_files[i];
        } else {
            filename++;
        }

        snprintf(output_path, sizeof(output_path), "%s/%s", output_dir, filename);

        // 保存结果
        if (image_save(output, output_path) != SR_SUCCESS) {
            fprintf(stderr, "Failed to save output: %s\n", output_path);
            fail_count++;
        } else {
            printf("Saved: %s (%.2f seconds)\n", output_path, result.computation_time);
            success_count++;
        }

        image_destroy(input);
        image_destroy(output);
    }

    printf("\nBatch processing complete:\n");
    printf("  Success: %d\n", success_count);
    printf("  Failed: %d\n", fail_count);

    return (fail_count == 0) ? SR_SUCCESS : SR_ERROR_PROCESSING;
}

// 自动选择最佳方法
SRMethod auto_select_method(const Image *input, float scale_factor, 
                           float quality_preference)
{
    if (!input) {
        return SR_METHOD_BICUBIC;
    }

    int image_size = input->width * input->height;

    // 根据图像大小和质量偏好选择方法
    if (quality_preference < 0.3f) {
        // 快速模式
        return SR_METHOD_BILINEAR;
    } else if (quality_preference < 0.6f) {
        // 平衡模式
        return SR_METHOD_BICUBIC;
    } else if (quality_preference < 0.8f) {
        // 高质量模式
        if (image_size < 1024 * 1024) {
            return SR_METHOD_LANCZOS;
        } else {
            return SR_METHOD_BICUBIC;
        }
    } else {
        // 最高质量模式
        if (image_size < 512 * 512) {
            return SR_METHOD_EDGE_DIRECTED;
        } else if (image_size < 1024 * 1024) {
            return SR_METHOD_LANCZOS;
        } else {
            return SR_METHOD_BICUBIC;
        }
    }
}

// 渐进式超分辨率（多次小倍数放大）
int progressive_super_resolve(const Image *input, float total_scale,
                             SRMethod method, void *params, SRResult *result)
{
    if (!input || !result || total_scale <= 1.0f) {
        return SR_ERROR_NULL_POINTER;
    }

    // 确定渐进步骤
    float step_scale = 2.0f;
    int num_steps = (int)(log(total_scale) / log(step_scale) + 0.5f);
    
    if (num_steps < 1) {
        num_steps = 1;
        step_scale = total_scale;
    }

    Image *current = image_clone(input);
    if (!current) {
        return SR_ERROR_MEMORY_ALLOCATION;
    }

    clock_t start_time = clock();

    // 渐进式放大
    for (int step = 0; step < num_steps; step++) {
        float scale = (step == num_steps - 1) ? 
                     (total_scale / powf(step_scale, step)) : step_scale;

        int out_width = (int)(current->width * scale + 0.5f);
        int out_height = (int)(current->height * scale + 0.5f);

        Image *temp_output = image_create(out_width, out_height, current->channels);
        if (!temp_output) {
            image_destroy(current);
            return SR_ERROR_MEMORY_ALLOCATION;
        }

        SRResult temp_result;
        temp_result.output = temp_output;

        int ret = SR_ERROR_INVALID_PARAMETER;

        switch (method) {
            case SR_METHOD_BICUBIC:
                ret = super_resolve_bicubic(current, scale, &temp_result);
                break;
            case SR_METHOD_LANCZOS:
                ret = super_resolve_lanczos(current, scale, 3, &temp_result);
                break;
            case SR_METHOD_EDGE_DIRECTED:
                ret = super_resolve_edge_directed(current, scale,
                                                  (EdgeDirectedParams*)params, &temp_result);
                break;
            default:
                ret = super_resolve_bicubic(current, scale, &temp_result);
                break;
        }

        image_destroy(current);

        if (ret != SR_SUCCESS) {
            image_destroy(temp_output);
            return ret;
        }

        current = temp_output;
        printf("Progressive step %d/%d complete (scale: %.2f)\n", 
               step + 1, num_steps, scale);
    }

    // 复制最终结果
    memcpy(result->output->data, current->data,
           current->width * current->height * current->channels * sizeof(float));

    image_destroy(current);

    clock_t end_time = clock();
    result->computation_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;

    return SR_SUCCESS;
}

// 图像金字塔超分辨率
int pyramid_super_resolve(const Image *input, float scale_factor,
                         int num_levels, SRMethod method, SRResult *result)
{
    if (!input || !result || scale_factor <= 1.0f || num_levels < 1) {
        return SR_ERROR_NULL_POINTER;
    }

    // 构建图像金字塔
    Image **pyramid = (Image**)malloc((num_levels + 1) * sizeof(Image*));
    if (!pyramid) {
        return SR_ERROR_MEMORY_ALLOCATION;
    }

    pyramid[0] = image_clone(input);
    if (!pyramid[0]) {
        free(pyramid);
        return SR_ERROR_MEMORY_ALLOCATION;
    }

    // 下采样构建金字塔
    for (int i = 1; i <= num_levels; i++) {
        int width = pyramid[i-1]->width / 2;
        int height = pyramid[i-1]->height / 2;
        
        if (width < 4 || height < 4) {
            num_levels = i - 1;
            break;
        }

        pyramid[i] = image_create(width, height, input->channels);
        if (!pyramid[i]) {
            for (int j = 0; j < i; j++) {
                image_destroy(pyramid[j]);
            }
            free(pyramid);
            return SR_ERROR_MEMORY_ALLOCATION;
        }

        // 简单下采样
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                for (int c = 0; c < input->channels; c++) {
                    int src_idx = ((y*2) * pyramid[i-1]->width + (x*2)) * input->channels + c;
                    int dst_idx = (y * width + x) * input->channels + c;
                    pyramid[i]->data[dst_idx] = pyramid[i-1]->data[src_idx];
                }
            }
        }
    }

    // 从最粗层开始向上超分辨率
    Image *current = pyramid[num_levels];
    
    for (int i = num_levels - 1; i >= 0; i--) {
        int target_width = pyramid[i]->width;
        int target_height = pyramid[i]->height;
        float level_scale = (float)target_width / current->width;

        Image *upsampled = image_create(target_width, target_height, input->channels);
        if (!upsampled) {
            for (int j = 0; j <= num_levels; j++) {
                image_destroy(pyramid[j]);
            }
            free(pyramid);
            return SR_ERROR_MEMORY_ALLOCATION;
        }

        SRResult temp_result;
        temp_result.output = upsampled;

        int ret = super_resolve_bicubic(current, level_scale, &temp_result);
        
        if (i < num_levels) {
            image_destroy(current);
        }

        if (ret != SR_SUCCESS) {
            image_destroy(upsampled);
            for (int j = 0; j <= num_levels; j++) {
                image_destroy(pyramid[j]);
            }
            free(pyramid);
            return ret;
        }

        // 融合细节
        for (int j = 0; j < target_width * target_height * input->channels; j++) {
            float detail = pyramid[i]->data[j] - upsampled->data[j];
            upsampled->data[j] += 0.5f * detail; // 50%细节融合
        }

        current = upsampled;
    }

    // 最终放大到目标尺寸
    int final_width = (int)(input->width * scale_factor + 0.5f);
    int final_height = (int)(input->height * scale_factor + 0.5f);
    float final_scale = (float)final_width / current->width;

    if (fabsf(final_scale - 1.0f) > 0.01f) {
        Image *final_output = image_create(final_width, final_height, input->channels);
        if (!final_output) {
            image_destroy(current);
            for (int j = 0; j <= num_levels; j++) {
                image_destroy(pyramid[j]);
            }
            free(pyramid);
            return SR_ERROR_MEMORY_ALLOCATION;
        }

        SRResult temp_result;
        temp_result.output = final_output;
        int ret = super_resolve_bicubic(current, final_scale, &temp_result);
        
        image_destroy(current);
        
        if (ret != SR_SUCCESS) {
            image_destroy(final_output);
            for (int j = 0; j <= num_levels; j++) {
                image_destroy(pyramid[j]);
            }
            free(pyramid);
            return ret;
        }

        memcpy(result->output->data, final_output->data,
               final_width * final_height * input->channels * sizeof(float));
        image_destroy(final_output);
    } else {
        memcpy(result->output->data, current->data,
               current->width * current->height * input->channels * sizeof(float));
        image_destroy(current);
    }

    // 清理金字塔
    for (int i = 0; i <= num_levels; i++) {
        image_destroy(pyramid[i]);
    }
    free(pyramid);

    return SR_SUCCESS;
}

// ============================================================================
// 错误处理和调试
// ============================================================================

const char* sr_error_string(int error_code)
{
    switch (error_code) {
        case SR_SUCCESS:
            return "Success";
        case SR_ERROR_NULL_POINTER:
            return "Null pointer error";
        case SR_ERROR_INVALID_PARAMETER:
            return "Invalid parameter";
        case SR_ERROR_MEMORY_ALLOCATION:
            return "Memory allocation failed";
        case SR_ERROR_FILE_IO:
            return "File I/O error";
        case SR_ERROR_INVALID_DIMENSIONS:
            return "Invalid image dimensions";
        case SR_ERROR_UNSUPPORTED_FORMAT:
            return "Unsupported format";
        case SR_ERROR_MODEL_NOT_TRAINED:
            return "Model not trained";
        case SR_ERROR_PROCESSING:
            return "Processing error";
        default:
            return "Unknown error";
    }
}

void sr_print_info(const SRResult *result)
{
    if (!result || !result->output) {
        printf("Invalid result\n");
        return;
    }

    printf("\n=== Super-Resolution Result ===\n");
    printf("Output size: %dx%d\n", result->output->width, result->output->height);
    printf("Channels: %d\n", result->output->channels);
    printf("Computation time: %.3f seconds\n", result->computation_time);
    printf("================================\n\n");
}

void sr_print_metrics(const QualityMetrics *metrics)
{
    if (!metrics) {
        printf("Invalid metrics\n");
        return;
    }

    printf("\n=== Quality Metrics ===\n");
    printf("PSNR: %.2f dB\n", metrics->psnr);
    printf("SSIM: %.4f\n", metrics->ssim);
    printf("MSE: %.6f\n", metrics->mse);
    printf("MAE: %.6f\n", metrics->mae);
    printf("=======================\n\n");
}

// ============================================================================
// 主函数示例
// ============================================================================

#ifdef SR_STANDALONE

void print_usage(const char *program_name)
{
    printf("Usage: %s [options] input_image output_image\n\n", program_name);
    printf("Options:\n");
    printf("  -s, --scale FACTOR      Scale factor (default: 2.0)\n");
    printf("  -m, --method METHOD     Method: bicubic, lanczos, edge, sparse, srcnn, espcn, edsr\n");
    printf("  -q, --quality LEVEL     Quality level: 0-1 (default: 0.8)\n");
    printf("  -p, --progressive       Use progressive upscaling\n");
    printf("  -e, --evaluate GT       Evaluate against ground truth image\n");
    printf("  -h, --help              Show this help message\n");
    printf("\nExamples:\n");
    printf("  %s -s 2 -m bicubic input.png output.png\n", program_name);
    printf("  %s -s 4 -m edge -q 0.9 input.jpg output.jpg\n", program_name);
    printf("  %s -s 2 -e ground_truth.png input.png output.png\n", program_name);
}

int main(int argc, char *argv[])
{
    if (argc < 3) {
        print_usage(argv[0]);
        return 1;
    }

    // 默认参数
    float scale_factor = 2.0f;
    SRMethod method = SR_METHOD_BICUBIC;
    float quality = 0.8f;
    bool progressive = false;
    const char *ground_truth_file = NULL;
    const char *input_file = NULL;
    const char *output_file = NULL;

    // 解析命令行参数
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-s") == 0 || strcmp(argv[i], "--scale") == 0) {
            if (i + 1 < argc) {
                scale_factor = atof(argv[++i]);
            }
        } else if (strcmp(argv[i], "-m") == 0 || strcmp(argv[i], "--method") == 0) {
            if (i + 1 < argc) {
                i++;
                if (strcmp(argv[i], "bicubic") == 0) {
                    method = SR_METHOD_BICUBIC;
                } else if (strcmp(argv[i], "lanczos") == 0) {
                    method = SR_METHOD_LANCZOS;
                } else if (strcmp(argv[i], "edge") == 0) {
                    method = SR_METHOD_EDGE_DIRECTED;
                } else if (strcmp(argv[i], "sparse") == 0) {
                    method = SR_METHOD_SPARSE_CODING;
                }
            }
        } else if (strcmp(argv[i], "-q") == 0 || strcmp(argv[i], "--quality") == 0) {
            if (i + 1 < argc) {
                quality = atof(argv[++i]);
            }
        } else if (strcmp(argv[i], "-p") == 0 || strcmp(argv[i], "--progressive") == 0) {
            progressive = true;
        } else if (strcmp(argv[i], "-e") == 0 || strcmp(argv[i], "--evaluate") == 0) {
            if (i + 1 < argc) {
                ground_truth_file = argv[++i];
            }
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        } else if (!input_file) {
            input_file = argv[i];
        } else if (!output_file) {
            output_file = argv[i];
        }
    }

    if (!input_file || !output_file) {
        fprintf(stderr, "Error: Input and output files required\n");
        print_usage(argv[0]);
        return 1;
    }

    printf("Super-Resolution Tool\n");
    printf("=====================\n");
    printf("Input: %s\n", input_file);
    printf("Output: %s\n", output_file);
    printf("Scale: %.1fx\n", scale_factor);
    printf("Method: %d\n", method);
    printf("\n");

    // 加载输入图像
    Image *input = image_load(input_file);
    if (!input) {
        fprintf(stderr, "Error: Failed to load input image\n");
        return 1;
    }

    printf("Input image: %dx%d, %d channels\n", 
           input->width, input->height, input->channels);

    // 创建输出图像
    int out_width = (int)(input->width * scale_factor + 0.5f);
    int out_height = (int)(input->height * scale_factor + 0.5f);
    Image *output = image_create(out_width, out_height, input->channels);

    if (!output) {
        fprintf(stderr, "Error: Failed to create output image\n");
        image_destroy(input);
        return 1;
    }

    // 

