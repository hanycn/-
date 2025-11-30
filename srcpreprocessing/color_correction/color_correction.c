/**
 * @file color_correction.c
 * @brief Color correction and color space conversion implementation
 */

#include "color_correction.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define EPSILON 1e-10
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define CLAMP(x, min, max) (MIN(MAX((x), (min)), (max)))

// ============================================================================
// Matrix Operations Implementation
// ============================================================================

void color_matrix_identity(ColorMatrix3x3 *matrix) {
    if (!matrix) return;
    
    memset(matrix->m, 0, sizeof(matrix->m));
    matrix->m[0][0] = 1.0;
    matrix->m[1][1] = 1.0;
    matrix->m[2][2] = 1.0;
}

void color_matrix_multiply(
    ColorMatrix3x3 *result,
    const ColorMatrix3x3 *a,
    const ColorMatrix3x3 *b)
{
    if (!result || !a || !b) return;
    
    ColorMatrix3x3 temp;
    
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            temp.m[i][j] = 0.0;
            for (int k = 0; k < 3; k++) {
                temp.m[i][j] += a->m[i][k] * b->m[k][j];
            }
        }
    }
    
    memcpy(result, &temp, sizeof(ColorMatrix3x3));
}

double color_matrix_determinant(const ColorMatrix3x3 *matrix) {
    if (!matrix) return 0.0;
    
    const double (*m)[3] = matrix->m;
    
    return m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
           m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
           m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
}

ColorError color_matrix_invert(
    ColorMatrix3x3 *result,
    const ColorMatrix3x3 *matrix)
{
    if (!result || !matrix) {
        return COLOR_ERROR_INVALID_PARAM;
    }
    
    double det = color_matrix_determinant(matrix);
    
    if (fabs(det) < EPSILON) {
        return COLOR_ERROR_INVALID_MATRIX;
    }
    
    const double (*m)[3] = matrix->m;
    double inv_det = 1.0 / det;
    
    ColorMatrix3x3 temp;
    
    // Calculate cofactor matrix and transpose
    temp.m[0][0] = (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * inv_det;
    temp.m[0][1] = (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * inv_det;
    temp.m[0][2] = (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * inv_det;
    
    temp.m[1][0] = (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * inv_det;
    temp.m[1][1] = (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * inv_det;
    temp.m[1][2] = (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * inv_det;
    
    temp.m[2][0] = (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * inv_det;
    temp.m[2][1] = (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * inv_det;
    temp.m[2][2] = (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * inv_det;
    
    memcpy(result, &temp, sizeof(ColorMatrix3x3));
    
    return COLOR_SUCCESS;
}

void color_matrix_apply(
    RGBColor *result,
    const ColorMatrix3x3 *matrix,
    const RGBColor *color)
{
    if (!result || !matrix || !color) return;
    
    double r = color->r;
    double g = color->g;
    double b = color->b;
    
    result->r = matrix->m[0][0] * r + matrix->m[0][1] * g + matrix->m[0][2] * b;
    result->g = matrix->m[1][0] * r + matrix->m[1][1] * g + matrix->m[1][2] * b;
    result->b = matrix->m[2][0] * r + matrix->m[2][1] * g + matrix->m[2][2] * b;
}

void color_matrix_transpose(
    ColorMatrix3x3 *result,
    const ColorMatrix3x3 *matrix)
{
    if (!result || !matrix) return;
    
    ColorMatrix3x3 temp;
    
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            temp.m[i][j] = matrix->m[j][i];
        }
    }
    
    memcpy(result, &temp, sizeof(ColorMatrix3x3));
}

// ============================================================================
// White Point Definitions
// ============================================================================

ColorError get_white_point(
    WhitePointCoords *coords,
    WhitePoint white_point)
{
    if (!coords) {
        return COLOR_ERROR_INVALID_PARAM;
    }
    
    switch (white_point) {
        case WHITE_POINT_D50:
            coords->x = 0.34567;
            coords->y = 0.35850;
            coords->Y = 1.0;
            break;
            
        case WHITE_POINT_D55:
            coords->x = 0.33242;
            coords->y = 0.34743;
            coords->Y = 1.0;
            break;
            
        case WHITE_POINT_D65:
            coords->x = 0.31271;
            coords->y = 0.32902;
            coords->Y = 1.0;
            break;
            
        case WHITE_POINT_D75:
            coords->x = 0.29902;
            coords->y = 0.31485;
            coords->Y = 1.0;
            break;
            
        case WHITE_POINT_A:
            coords->x = 0.44757;
            coords->y = 0.40745;
            coords->Y = 1.0;
            break;
            
        case WHITE_POINT_B:
            coords->x = 0.34842;
            coords->y = 0.35161;
            coords->Y = 1.0;
            break;
            
        case WHITE_POINT_C:
            coords->x = 0.31006;
            coords->y = 0.31616;
            coords->Y = 1.0;
            break;
            
        case WHITE_POINT_E:
            coords->x = 0.33333;
            coords->y = 0.33333;
            coords->Y = 1.0;
            break;
            
        case WHITE_POINT_F2:
            coords->x = 0.37208;
            coords->y = 0.37529;
            coords->Y = 1.0;
            break;
            
        case WHITE_POINT_F7:
            coords->x = 0.31292;
            coords->y = 0.32933;
            coords->Y = 1.0;
            break;
            
        case WHITE_POINT_F11:
            coords->x = 0.38052;
            coords->y = 0.37713;
            coords->Y = 1.0;
            break;
            
        default:
            return COLOR_ERROR_INVALID_PARAM;
    }
    
    return COLOR_SUCCESS;
}

// ============================================================================
// Gamma Correction Implementation
// ============================================================================

double srgb_gamma_encode(double value) {
    if (value <= 0.0031308) {
        return 12.92 * value;
    } else {
        return 1.055 * pow(value, 1.0 / 2.4) - 0.055;
    }
}

double srgb_gamma_decode(double value) {
    if (value <= 0.04045) {
        return value / 12.92;
    } else {
        return pow((value + 0.055) / 1.055, 2.4);
    }
}

double rec709_gamma_encode(double value) {
    if (value < 0.018) {
        return 4.5 * value;
    } else {
        return 1.099 * pow(value, 0.45) - 0.099;
    }
}

double rec709_gamma_decode(double value) {
    if (value < 0.081) {
        return value / 4.5;
    } else {
        return pow((value + 0.099) / 1.099, 1.0 / 0.45);
    }
}

ColorError apply_gamma(
    double *output,
    double input,
    GammaCurve gamma_type,
    double gamma_value)
{
    if (!output) {
        return COLOR_ERROR_INVALID_PARAM;
    }
    
    input = CLAMP(input, 0.0, 1.0);
    
    switch (gamma_type) {
        case GAMMA_LINEAR:
            *output = input;
            break;
            
        case GAMMA_SRGB:
            *output = srgb_gamma_encode(input);
            break;
            
        case GAMMA_REC709:
            *output = rec709_gamma_encode(input);
            break;
            
        case GAMMA_POWER:
            *output = pow(input, 1.0 / gamma_value);
            break;
            
        case GAMMA_L_STAR:
            if (input > 216.0 / 24389.0) {
                *output = 1.16 * pow(input, 1.0 / 3.0) - 0.16;
            } else {
                *output = input * 24389.0 / 2700.0;
            }
            break;
            
        default:
            return COLOR_ERROR_INVALID_PARAM;
    }
    
    return COLOR_SUCCESS;
}

ColorError remove_gamma(
    double *output,
    double input,
    GammaCurve gamma_type,
    double gamma_value)
{
    if (!output) {
        return COLOR_ERROR_INVALID_PARAM;
    }
    
    input = CLAMP(input, 0.0, 1.0);
    
    switch (gamma_type) {
        case GAMMA_LINEAR:
            *output = input;
            break;
            
        case GAMMA_SRGB:
            *output = srgb_gamma_decode(input);
            break;
            
        case GAMMA_REC709:
            *output = rec709_gamma_decode(input);
            break;
            
        case GAMMA_POWER:
            *output = pow(input, gamma_value);
            break;
            
        case GAMMA_L_STAR:
            if (input > 0.08) {
                *output = pow((input + 0.16) / 1.16, 3.0);
            } else {
                *output = input * 2700.0 / 24389.0;
            }
            break;
            
        default:
            return COLOR_ERROR_INVALID_PARAM;
    }
    
    return COLOR_SUCCESS;
}
// ============================================================================
// RGB to XYZ Conversion
// ============================================================================

ColorError rgb_to_xyz(
    XYZColor *xyz,
    const RGBColor *rgb,
    const ColorSpaceDefinition *colorspace)
{
    if (!xyz || !rgb || !colorspace) {
        return COLOR_ERROR_INVALID_PARAM;
    }
    
    // Remove gamma correction first
    double r_linear, g_linear, b_linear;
    
    ColorError err;
    err = remove_gamma(&r_linear, rgb->r, colorspace->gamma_curve, colorspace->gamma_value);
    if (err != COLOR_SUCCESS) return err;
    
    err = remove_gamma(&g_linear, rgb->g, colorspace->gamma_curve, colorspace->gamma_value);
    if (err != COLOR_SUCCESS) return err;
    
    err = remove_gamma(&b_linear, rgb->b, colorspace->gamma_curve, colorspace->gamma_value);
    if (err != COLOR_SUCCESS) return err;
    
    // Apply RGB to XYZ matrix
    const ColorMatrix3x3 *m = &colorspace->rgb_to_xyz;
    
    xyz->X = m->m[0][0] * r_linear + m->m[0][1] * g_linear + m->m[0][2] * b_linear;
    xyz->Y = m->m[1][0] * r_linear + m->m[1][1] * g_linear + m->m[1][2] * b_linear;
    xyz->Z = m->m[2][0] * r_linear + m->m[2][1] * g_linear + m->m[2][2] * b_linear;
    
    return COLOR_SUCCESS;
}

// ============================================================================
// XYZ to RGB Conversion
// ============================================================================

ColorError xyz_to_rgb(
    RGBColor *rgb,
    const XYZColor *xyz,
    const ColorSpaceDefinition *colorspace)
{
    if (!rgb || !xyz || !colorspace) {
        return COLOR_ERROR_INVALID_PARAM;
    }
    
    // Apply XYZ to RGB matrix
    const ColorMatrix3x3 *m = &colorspace->xyz_to_rgb;
    
    double r_linear = m->m[0][0] * xyz->X + m->m[0][1] * xyz->Y + m->m[0][2] * xyz->Z;
    double g_linear = m->m[1][0] * xyz->X + m->m[1][1] * xyz->Y + m->m[1][2] * xyz->Z;
    double b_linear = m->m[2][0] * xyz->X + m->m[2][1] * xyz->Y + m->m[2][2] * xyz->Z;
    
    // Apply gamma correction
    ColorError err;
    err = apply_gamma(&rgb->r, r_linear, colorspace->gamma_curve, colorspace->gamma_value);
    if (err != COLOR_SUCCESS) return err;
    
    err = apply_gamma(&rgb->g, g_linear, colorspace->gamma_curve, colorspace->gamma_value);
    if (err != COLOR_SUCCESS) return err;
    
    err = apply_gamma(&rgb->b, b_linear, colorspace->gamma_curve, colorspace->gamma_value);
    if (err != COLOR_SUCCESS) return err;
    
    return COLOR_SUCCESS;
}

// ============================================================================
// XYZ to Lab Conversion
// ============================================================================

static double lab_f(double t) {
    const double delta = 6.0 / 29.0;
    const double delta_cubed = delta * delta * delta;
    
    if (t > delta_cubed) {
        return pow(t, 1.0 / 3.0);
    } else {
        return t / (3.0 * delta * delta) + 4.0 / 29.0;
    }
}

ColorError xyz_to_lab(
    LabColor *lab,
    const XYZColor *xyz,
    const WhitePointCoords *white_point)
{
    if (!lab || !xyz || !white_point) {
        return COLOR_ERROR_INVALID_PARAM;
    }
    
    // Convert white point xy to XYZ
    double Xn = white_point->x * white_point->Y / white_point->y;
    double Yn = white_point->Y;
    double Zn = (1.0 - white_point->x - white_point->y) * white_point->Y / white_point->y;
    
    // Normalize
    double xr = xyz->X / Xn;
    double yr = xyz->Y / Yn;
    double zr = xyz->Z / Zn;
    
    // Apply Lab transformation
    double fx = lab_f(xr);
    double fy = lab_f(yr);
    double fz = lab_f(zr);
    
    lab->L = 116.0 * fy - 16.0;
    lab->a = 500.0 * (fx - fy);
    lab->b = 200.0 * (fy - fz);
    
    return COLOR_SUCCESS;
}

// ============================================================================
// Lab to XYZ Conversion
// ============================================================================

static double lab_f_inv(double t) {
    const double delta = 6.0 / 29.0;
    
    if (t > delta) {
        return t * t * t;
    } else {
        return 3.0 * delta * delta * (t - 4.0 / 29.0);
    }
}

ColorError lab_to_xyz(
    XYZColor *xyz,
    const LabColor *lab,
    const WhitePointCoords *white_point)
{
    if (!xyz || !lab || !white_point) {
        return COLOR_ERROR_INVALID_PARAM;
    }
    
    // Convert white point xy to XYZ
    double Xn = white_point->x * white_point->Y / white_point->y;
    double Yn = white_point->Y;
    double Zn = (1.0 - white_point->x - white_point->y) * white_point->Y / white_point->y;
    
    // Inverse Lab transformation
    double fy = (lab->L + 16.0) / 116.0;
    double fx = lab->a / 500.0 + fy;
    double fz = fy - lab->b / 200.0;
    
    xyz->X = Xn * lab_f_inv(fx);
    xyz->Y = Yn * lab_f_inv(fy);
    xyz->Z = Zn * lab_f_inv(fz);
    
    return COLOR_SUCCESS;
}

// ============================================================================
// RGB to HSV Conversion
// ============================================================================

void rgb_to_hsv(HSVColor *hsv, const RGBColor *rgb) {
    if (!hsv || !rgb) return;
    
    double r = rgb->r;
    double g = rgb->g;
    double b = rgb->b;
    
    double max_val = MAX(MAX(r, g), b);
    double min_val = MIN(MIN(r, g), b);
    double delta = max_val - min_val;
    
    // Value
    hsv->v = max_val;
    
    // Saturation
    if (max_val < EPSILON) {
        hsv->s = 0.0;
        hsv->h = 0.0;
        return;
    }
    
    hsv->s = delta / max_val;
    
    // Hue
    if (delta < EPSILON) {
        hsv->h = 0.0;
    } else if (fabs(max_val - r) < EPSILON) {
        hsv->h = 60.0 * fmod((g - b) / delta, 6.0);
    } else if (fabs(max_val - g) < EPSILON) {
        hsv->h = 60.0 * ((b - r) / delta + 2.0);
    } else {
        hsv->h = 60.0 * ((r - g) / delta + 4.0);
    }
    
    if (hsv->h < 0.0) {
        hsv->h += 360.0;
    }
}

// ============================================================================
// HSV to RGB Conversion
// ============================================================================

void hsv_to_rgb(RGBColor *rgb, const HSVColor *hsv) {
    if (!rgb || !hsv) return;
    
    double h = hsv->h;
    double s = hsv->s;
    double v = hsv->v;
    
    if (s < EPSILON) {
        rgb->r = rgb->g = rgb->b = v;
        return;
    }
    
    h = fmod(h, 360.0);
    if (h < 0.0) h += 360.0;
    
    double c = v * s;
    double x = c * (1.0 - fabs(fmod(h / 60.0, 2.0) - 1.0));
    double m = v - c;
    
    double r1, g1, b1;
    
    if (h < 60.0) {
        r1 = c; g1 = x; b1 = 0.0;
    } else if (h < 120.0) {
        r1 = x; g1 = c; b1 = 0.0;
    } else if (h < 180.0) {
        r1 = 0.0; g1 = c; b1 = x;
    } else if (h < 240.0) {
        r1 = 0.0; g1 = x; b1 = c;
    } else if (h < 300.0) {
        r1 = x; g1 = 0.0; b1 = c;
    } else {
        r1 = c; g1 = 0.0; b1 = x;
    }
    
    rgb->r = r1 + m;
    rgb->g = g1 + m;
    rgb->b = b1 + m;
}

// ============================================================================
// RGB to HSL Conversion
// ============================================================================

void rgb_to_hsl(HSLColor *hsl, const RGBColor *rgb) {
    if (!hsl || !rgb) return;
    
    double r = rgb->r;
    double g = rgb->g;
    double b = rgb->b;
    
    double max_val = MAX(MAX(r, g), b);
    double min_val = MIN(MIN(r, g), b);
    double delta = max_val - min_val;
    
    // Lightness
    hsl->l = (max_val + min_val) / 2.0;
    
    // Saturation
    if (delta < EPSILON) {
        hsl->s = 0.0;
        hsl->h = 0.0;
        return;
    }
    
    if (hsl->l < 0.5) {
        hsl->s = delta / (max_val + min_val);
    } else {
        hsl->s = delta / (2.0 - max_val - min_val);
    }
    
    // Hue
    if (fabs(max_val - r) < EPSILON) {
        hsl->h = 60.0 * fmod((g - b) / delta, 6.0);
    } else if (fabs(max_val - g) < EPSILON) {
        hsl->h = 60.0 * ((b - r) / delta + 2.0);
    } else {
        hsl->h = 60.0 * ((r - g) / delta + 4.0);
    }
    
    if (hsl->h < 0.0) {
        hsl->h += 360.0;
    }
}

// ============================================================================
// HSL to RGB Conversion
// ============================================================================

static double hsl_hue_to_rgb(double p, double q, double t) {
    if (t < 0.0) t += 1.0;
    if (t > 1.0) t -= 1.0;
    
    if (t < 1.0 / 6.0) return p + (q - p) * 6.0 * t;
    if (t < 1.0 / 2.0) return q;
    if (t < 2.0 / 3.0) return p + (q - p) * (2.0 / 3.0 - t) * 6.0;
    
    return p;
}

void hsl_to_rgb(RGBColor *rgb, const HSLColor *hsl) {
    if (!rgb || !hsl) return;
    
    double h = hsl->h / 360.0;
    double s = hsl->s;
    double l = hsl->l;
    
    if (s < EPSILON) {
        rgb->r = rgb->g = rgb->b = l;
        return;
    }
    
    double q = (l < 0.5) ? (l * (1.0 + s)) : (l + s - l * s);
    double p = 2.0 * l - q;
    
    rgb->r = hsl_hue_to_rgb(p, q, h + 1.0 / 3.0);
    rgb->g = hsl_hue_to_rgb(p, q, h);
    rgb->b = hsl_hue_to_rgb(p, q, h - 1.0 / 3.0);
}

// ============================================================================
// RGB to YCbCr Conversion
// ============================================================================

void rgb_to_ycbcr(
    YCbCrColor *ycbcr,
    const RGBColor *rgb,
    ColorSpace standard)
{
    if (!ycbcr || !rgb) return;
    
    double r = rgb->r;
    double g = rgb->g;
    double b = rgb->b;
    
    // BT.601 coefficients (also used for sRGB)
    double Kr = 0.299;
    double Kg = 0.587;
    double Kb = 0.114;
    
    if (standard == COLOR_SPACE_REC709) {
        Kr = 0.2126;
        Kg = 0.7152;
        Kb = 0.0722;
    } else if (standard == COLOR_SPACE_REC2020) {
        Kr = 0.2627;
        Kg = 0.6780;
        Kb = 0.0593;
    }
    
    ycbcr->Y = Kr * r + Kg * g + Kb * b;
    ycbcr->Cb = (b - ycbcr->Y) / (2.0 * (1.0 - Kb));
    ycbcr->Cr = (r - ycbcr->Y) / (2.0 * (1.0 - Kr));
}

// ============================================================================
// YCbCr to RGB Conversion
// ============================================================================

void ycbcr_to_rgb(
    RGBColor *rgb,
    const YCbCrColor *ycbcr,
    ColorSpace standard)
{
    if (!rgb || !ycbcr) return;
    
    double Y = ycbcr->Y;
    double Cb = ycbcr->Cb;
    double Cr = ycbcr->Cr;
    
    // BT.601 coefficients
    double Kr = 0.299;
    double Kb = 0.114;
    
    if (standard == COLOR_SPACE_REC709) {
        Kr = 0.2126;
        Kb = 0.0722;
    } else if (standard == COLOR_SPACE_REC2020) {
        Kr = 0.2627;
        Kb = 0.0593;
    }
    
    rgb->r = Y + Cr * 2.0 * (1.0 - Kr);
    rgb->b = Y + Cb * 2.0 * (1.0 - Kb);
    rgb->g = (Y - Kr * rgb->r - Kb * rgb->b) / (1.0 - Kr - Kb);
    
    // Clamp values
    rgb->r = CLAMP(rgb->r, 0.0, 1.0);
    rgb->g = CLAMP(rgb->g, 0.0, 1.0);
    rgb->b = CLAMP(rgb->b, 0.0, 1.0);
}
// ============================================================================
// Color Space Definitions
// ============================================================================

ColorError get_colorspace_definition(
    ColorSpaceDefinition *def,
    ColorSpace colorspace)
{
    if (!def) {
        return COLOR_ERROR_INVALID_PARAM;
    }
    
    memset(def, 0, sizeof(ColorSpaceDefinition));
    
    switch (colorspace) {
        case COLOR_SPACE_SRGB:
            def->colorspace = COLOR_SPACE_SRGB;
            def->white_point = WHITE_POINT_D65;
            def->gamma_curve = GAMMA_SRGB;
            def->gamma_value = 2.2;
            
            // sRGB to XYZ matrix (D65)
            def->rgb_to_xyz.m[0][0] = 0.4124564;
            def->rgb_to_xyz.m[0][1] = 0.3575761;
            def->rgb_to_xyz.m[0][2] = 0.1804375;
            def->rgb_to_xyz.m[1][0] = 0.2126729;
            def->rgb_to_xyz.m[1][1] = 0.7151522;
            def->rgb_to_xyz.m[1][2] = 0.0721750;
            def->rgb_to_xyz.m[2][0] = 0.0193339;
            def->rgb_to_xyz.m[2][1] = 0.1191920;
            def->rgb_to_xyz.m[2][2] = 0.9503041;
            
            // XYZ to sRGB matrix
            def->xyz_to_rgb.m[0][0] =  3.2404542;
            def->xyz_to_rgb.m[0][1] = -1.5371385;
            def->xyz_to_rgb.m[0][2] = -0.4985314;
            def->xyz_to_rgb.m[1][0] = -0.9692660;
            def->xyz_to_rgb.m[1][1] =  1.8760108;
            def->xyz_to_rgb.m[1][2] =  0.0415560;
            def->xyz_to_rgb.m[2][0] =  0.0556434;
            def->xyz_to_rgb.m[2][1] = -0.2040259;
            def->xyz_to_rgb.m[2][2] =  1.0572252;
            break;
            
        case COLOR_SPACE_ADOBE_RGB:
            def->colorspace = COLOR_SPACE_ADOBE_RGB;
            def->white_point = WHITE_POINT_D65;
            def->gamma_curve = GAMMA_POWER;
            def->gamma_value = 2.2;
            
            // Adobe RGB to XYZ matrix (D65)
            def->rgb_to_xyz.m[0][0] = 0.5767309;
            def->rgb_to_xyz.m[0][1] = 0.1855540;
            def->rgb_to_xyz.m[0][2] = 0.1881852;
            def->rgb_to_xyz.m[1][0] = 0.2973769;
            def->rgb_to_xyz.m[1][1] = 0.6273491;
            def->rgb_to_xyz.m[1][2] = 0.0752741;
            def->rgb_to_xyz.m[2][0] = 0.0270343;
            def->rgb_to_xyz.m[2][1] = 0.0706872;
            def->rgb_to_xyz.m[2][2] = 0.9911085;
            
            // XYZ to Adobe RGB matrix
            def->xyz_to_rgb.m[0][0] =  2.0413690;
            def->xyz_to_rgb.m[0][1] = -0.5649464;
            def->xyz_to_rgb.m[0][2] = -0.3446944;
            def->xyz_to_rgb.m[1][0] = -0.9692660;
            def->xyz_to_rgb.m[1][1] =  1.8760108;
            def->xyz_to_rgb.m[1][2] =  0.0415560;
            def->xyz_to_rgb.m[2][0] =  0.0134474;
            def->xyz_to_rgb.m[2][1] = -0.1183897;
            def->xyz_to_rgb.m[2][2] =  1.0154096;
            break;
            
        case COLOR_SPACE_PROPHOTO_RGB:
            def->colorspace = COLOR_SPACE_PROPHOTO_RGB;
            def->white_point = WHITE_POINT_D50;
            def->gamma_curve = GAMMA_POWER;
            def->gamma_value = 1.8;
            
            // ProPhoto RGB to XYZ matrix (D50)
            def->rgb_to_xyz.m[0][0] = 0.7976749;
            def->rgb_to_xyz.m[0][1] = 0.1351917;
            def->rgb_to_xyz.m[0][2] = 0.0313534;
            def->rgb_to_xyz.m[1][0] = 0.2880402;
            def->rgb_to_xyz.m[1][1] = 0.7118741;
            def->rgb_to_xyz.m[1][2] = 0.0000857;
            def->rgb_to_xyz.m[2][0] = 0.0000000;
            def->rgb_to_xyz.m[2][1] = 0.0000000;
            def->rgb_to_xyz.m[2][2] = 0.8252100;
            
            // XYZ to ProPhoto RGB matrix
            def->xyz_to_rgb.m[0][0] =  1.3459433;
            def->xyz_to_rgb.m[0][1] = -0.2556075;
            def->xyz_to_rgb.m[0][2] = -0.0511118;
            def->xyz_to_rgb.m[1][0] = -0.5445989;
            def->xyz_to_rgb.m[1][1] =  1.5081673;
            def->xyz_to_rgb.m[1][2] =  0.0205351;
            def->xyz_to_rgb.m[2][0] =  0.0000000;
            def->xyz_to_rgb.m[2][1] =  0.0000000;
            def->xyz_to_rgb.m[2][2] =  1.2118128;
            break;
            
        case COLOR_SPACE_REC709:
            def->colorspace = COLOR_SPACE_REC709;
            def->white_point = WHITE_POINT_D65;
            def->gamma_curve = GAMMA_REC709;
            def->gamma_value = 2.4;
            
            // Rec.709 to XYZ matrix (same as sRGB primaries)
            def->rgb_to_xyz.m[0][0] = 0.4124564;
            def->rgb_to_xyz.m[0][1] = 0.3575761;
            def->rgb_to_xyz.m[0][2] = 0.1804375;
            def->rgb_to_xyz.m[1][0] = 0.2126729;
            def->rgb_to_xyz.m[1][1] = 0.7151522;
            def->rgb_to_xyz.m[1][2] = 0.0721750;
            def->rgb_to_xyz.m[2][0] = 0.0193339;
            def->rgb_to_xyz.m[2][1] = 0.1191920;
            def->rgb_to_xyz.m[2][2] = 0.9503041;
            
            // XYZ to Rec.709 matrix
            def->xyz_to_rgb.m[0][0] =  3.2404542;
            def->xyz_to_rgb.m[0][1] = -1.5371385;
            def->xyz_to_rgb.m[0][2] = -0.4985314;
            def->xyz_to_rgb.m[1][0] = -0.9692660;
            def->xyz_to_rgb.m[1][1] =  1.8760108;
            def->xyz_to_rgb.m[1][2] =  0.0415560;
            def->xyz_to_rgb.m[2][0] =  0.0556434;
            def->xyz_to_rgb.m[2][1] = -0.2040259;
            def->xyz_to_rgb.m[2][2] =  1.0572252;
            break;
            
        case COLOR_SPACE_REC2020:
            def->colorspace = COLOR_SPACE_REC2020;
            def->white_point = WHITE_POINT_D65;
            def->gamma_curve = GAMMA_REC709;
            def->gamma_value = 2.4;
            
            // Rec.2020 to XYZ matrix (D65)
            def->rgb_to_xyz.m[0][0] = 0.6369580;
            def->rgb_to_xyz.m[0][1] = 0.1446169;
            def->rgb_to_xyz.m[0][2] = 0.1688810;
            def->rgb_to_xyz.m[1][0] = 0.2627002;
            def->rgb_to_xyz.m[1][1] = 0.6779981;
            def->rgb_to_xyz.m[1][2] = 0.0593017;
            def->rgb_to_xyz.m[2][0] = 0.0000000;
            def->rgb_to_xyz.m[2][1] = 0.0280727;
            def->rgb_to_xyz.m[2][2] = 1.0609851;
            
            // XYZ to Rec.2020 matrix
            def->xyz_to_rgb.m[0][0] =  1.7166512;
            def->xyz_to_rgb.m[0][1] = -0.3556708;
            def->xyz_to_rgb.m[0][2] = -0.2533663;
            def->xyz_to_rgb.m[1][0] = -0.6666844;
            def->xyz_to_rgb.m[1][1] =  1.6164812;
            def->xyz_to_rgb.m[1][2] =  0.0157685;
            def->xyz_to_rgb.m[2][0] =  0.0176399;
            def->xyz_to_rgb.m[2][1] = -0.0427706;
            def->xyz_to_rgb.m[2][2] =  0.9421031;
            break;
            
        case COLOR_SPACE_DCI_P3:
            def->colorspace = COLOR_SPACE_DCI_P3;
            def->white_point = WHITE_POINT_D65;
            def->gamma_curve = GAMMA_POWER;
            def->gamma_value = 2.6;
            
            // DCI-P3 to XYZ matrix (D65)
            def->rgb_to_xyz.m[0][0] = 0.4865709;
            def->rgb_to_xyz.m[0][1] = 0.2656677;
            def->rgb_to_xyz.m[0][2] = 0.1982173;
            def->rgb_to_xyz.m[1][0] = 0.2289746;
            def->rgb_to_xyz.m[1][1] = 0.6917385;
            def->rgb_to_xyz.m[1][2] = 0.0792869;
            def->rgb_to_xyz.m[2][0] = 0.0000000;
            def->rgb_to_xyz.m[2][1] = 0.0451134;
            def->rgb_to_xyz.m[2][2] = 1.0439444;
            
            // XYZ to DCI-P3 matrix
            def->xyz_to_rgb.m[0][0] =  2.4934969;
            def->xyz_to_rgb.m[0][1] = -0.9313836;
            def->xyz_to_rgb.m[0][2] = -0.4027108;
            def->xyz_to_rgb.m[1][0] = -0.8294890;
            def->xyz_to_rgb.m[1][1] =  1.7626641;
            def->xyz_to_rgb.m[1][2] =  0.0236247;
            def->xyz_to_rgb.m[2][0] =  0.0358458;
            def->xyz_to_rgb.m[2][1] = -0.0761724;
            def->xyz_to_rgb.m[2][2] =  0.9568845;
            break;
            
        case COLOR_SPACE_ACES:
            def->colorspace = COLOR_SPACE_ACES;
            def->white_point = WHITE_POINT_D60;
            def->gamma_curve = GAMMA_LINEAR;
            def->gamma_value = 1.0;
            
            // ACES AP0 to XYZ matrix (D60)
            def->rgb_to_xyz.m[0][0] = 0.9525523;
            def->rgb_to_xyz.m[0][1] = 0.0000000;
            def->rgb_to_xyz.m[0][2] = 0.0000936;
            def->rgb_to_xyz.m[1][0] = 0.3439664;
            def->rgb_to_xyz.m[1][1] = 0.7281660;
            def->rgb_to_xyz.m[1][2] = -0.0721325;
            def->rgb_to_xyz.m[2][0] = 0.0000000;
            def->rgb_to_xyz.m[2][1] = 0.0000000;
            def->rgb_to_xyz.m[2][2] = 1.0088251;
            
            // XYZ to ACES AP0 matrix
            def->xyz_to_rgb.m[0][0] =  1.0498110;
            def->xyz_to_rgb.m[0][1] =  0.0000000;
            def->xyz_to_rgb.m[0][2] = -0.0000974;
            def->xyz_to_rgb.m[1][0] = -0.4959030;
            def->xyz_to_rgb.m[1][1] =  1.3733130;
            def->xyz_to_rgb.m[1][2] =  0.0982400;
            def->xyz_to_rgb.m[2][0] =  0.0000000;
            def->xyz_to_rgb.m[2][1] =  0.0000000;
            def->xyz_to_rgb.m[2][2] =  0.9912520;
            break;
            
        default:
            return COLOR_ERROR_INVALID_PARAM;
    }
    
    return COLOR_SUCCESS;
}

// ============================================================================
// Chromatic Adaptation (Bradford Method)
// ============================================================================

ColorError chromatic_adaptation_matrix(
    ColorMatrix3x3 *result,
    const WhitePointCoords *source_wp,
    const WhitePointCoords *dest_wp)
{
    if (!result || !source_wp || !dest_wp) {
        return COLOR_ERROR_INVALID_PARAM;
    }
    
    // Bradford transformation matrix
    const double bradford[3][3] = {
        { 0.8951000,  0.2664000, -0.1614000},
        {-0.7502000,  1.7135000,  0.0367000},
        { 0.0389000, -0.0685000,  1.0296000}
    };
    
    // Inverse Bradford matrix
    const double bradford_inv[3][3] = {
        { 0.9869929, -0.1470543,  0.1599627},
        { 0.4323053,  0.5183603,  0.0492912},
        {-0.0085287,  0.0400428,  0.9684867}
    };
    
    // Convert white points to XYZ
    double Xs = source_wp->x * source_wp->Y / source_wp->y;
    double Ys = source_wp->Y;
    double Zs = (1.0 - source_wp->x - source_wp->y) * source_wp->Y / source_wp->y;
    
    double Xd = dest_wp->x * dest_wp->Y / dest_wp->y;
    double Yd = dest_wp->Y;
    double Zd = (1.0 - dest_wp->x - dest_wp->y) * dest_wp->Y / dest_wp->y;
    
    // Transform to cone response domain
    double rho_s = bradford[0][0] * Xs + bradford[0][1] * Ys + bradford[0][2] * Zs;
    double gamma_s = bradford[1][0] * Xs + bradford[1][1] * Ys + bradford[1][2] * Zs;
    double beta_s = bradford[2][0] * Xs + bradford[2][1] * Ys + bradford[2][2] * Zs;
    
    double rho_d = bradford[0][0] * Xd + bradford[0][1] * Yd + bradford[0][2] * Zd;
    double gamma_d = bradford[1][0] * Xd + bradford[1][1] * Yd + bradford[1][2] * Zd;
    double beta_d = bradford[2][0] * Xd + bradford[2][1] * Yd + bradford[2][2] * Zd;
    
    // Calculate scaling factors
    double scale_rho = rho_d / rho_s;
    double scale_gamma = gamma_d / gamma_s;
    double scale_beta = beta_d / beta_s;
    
    // Build adaptation matrix: M_inv * S * M
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            result->m[i][j] = 
                bradford_inv[i][0] * scale_rho * bradford[0][j] +
                bradford_inv[i][1] * scale_gamma * bradford[1][j] +
                bradford_inv[i][2] * scale_beta * bradford[2][j];
        }
    }
    
    return COLOR_SUCCESS;
}

// ============================================================================
// Color Space Conversion
// ============================================================================

ColorError convert_colorspace(
    RGBColor *output,
    const RGBColor *input,
    ColorSpace source_space,
    ColorSpace dest_space)
{
    if (!output || !input) {
        return COLOR_ERROR_INVALID_PARAM;
    }
    
    // If same color space, just copy
    if (source_space == dest_space) {
        *output = *input;
        return COLOR_SUCCESS;
    }
    
    ColorError err;
    
    // Get source and destination color space definitions
    ColorSpaceDefinition source_def, dest_def;
    
    err = get_colorspace_definition(&source_def, source_space);
    if (err != COLOR_SUCCESS) return err;
    
    err = get_colorspace_definition(&dest_def, dest_space);
    if (err != COLOR_SUCCESS) return err;
    
    // Convert source RGB to XYZ
    XYZColor xyz;
    err = rgb_to_xyz(&xyz, input, &source_def);
    if (err != COLOR_SUCCESS) return err;
    
    // Apply chromatic adaptation if white points differ
    if (source_def.white_point != dest_def.white_point) {
        WhitePointCoords source_wp, dest_wp;
        
        err = get_white_point(&source_wp, source_def.white_point);
        if (err != COLOR_SUCCESS) return err;
        
        err = get_white_point(&dest_wp, dest_def.white_point);
        if (err != COLOR_SUCCESS) return err;
        
        ColorMatrix3x3 adaptation;
        err = chromatic_adaptation_matrix(&adaptation, &source_wp, &dest_wp);
        if (err != COLOR_SUCCESS) return err;
        
        // Apply adaptation to XYZ
        double X = xyz.X;
        double Y = xyz.Y;
        double Z = xyz.Z;
        
        xyz.X = adaptation.m[0][0] * X + adaptation.m[0][1] * Y + adaptation.m[0][2] * Z;
        xyz.Y = adaptation.m[1][0] * X + adaptation.m[1][1] * Y + adaptation.m[1][2] * Z;
        xyz.Z = adaptation.m[2][0] * X + adaptation.m[2][1] * Y + adaptation.m[2][2] * Z;
    }
    
    // Convert XYZ to destination RGB
    err = xyz_to_rgb(output, &xyz, &dest_def);
    if (err != COLOR_SUCCESS) return err;
    
    // Clamp output values
    output->r = CLAMP(output->r, 0.0, 1.0);
    output->g = CLAMP(output->g, 0.0, 1.0);
    output->b = CLAMP(output->b, 0.0, 1.0);
    
    return COLOR_SUCCESS;
}
// ============================================================================
// White Balance Adjustment
// ============================================================================

ColorError white_balance_adjust(
    RGBColor *output,
    const RGBColor *input,
    double temperature,
    double tint)
{
    if (!output || !input) {
        return COLOR_ERROR_INVALID_PARAM;
    }
    
    // Temperature: 2000K - 10000K, Tint: -1.0 to 1.0
    temperature = CLAMP(temperature, 2000.0, 10000.0);
    tint = CLAMP(tint, -1.0, 1.0);
    
    // Calculate RGB multipliers based on temperature
    double r_mult, g_mult, b_mult;
    
    // Simplified temperature to RGB conversion
    if (temperature < 6600.0) {
        r_mult = 1.0;
        g_mult = 0.39 * log(temperature / 100.0) - 0.63;
        
        if (temperature <= 2000.0) {
            b_mult = 0.0;
        } else {
            b_mult = 0.54 * log(temperature / 100.0 - 10.0) - 1.19;
        }
    } else {
        r_mult = 1.29 * pow((temperature / 100.0 - 60.0), -0.13);
        g_mult = 1.13 * pow((temperature / 100.0 - 60.0), -0.08);
        b_mult = 1.0;
    }
    
    // Normalize multipliers
    double max_mult = fmax(r_mult, fmax(g_mult, b_mult));
    r_mult /= max_mult;
    g_mult /= max_mult;
    b_mult /= max_mult;
    
    // Apply tint (green-magenta shift)
    g_mult *= (1.0 + tint * 0.3);
    
    // Apply white balance
    output->r = CLAMP(input->r * r_mult, 0.0, 1.0);
    output->g = CLAMP(input->g * g_mult, 0.0, 1.0);
    output->b = CLAMP(input->b * b_mult, 0.0, 1.0);
    
    return COLOR_SUCCESS;
}

// ============================================================================
// Exposure Adjustment
// ============================================================================

ColorError exposure_adjust(
    RGBColor *output,
    const RGBColor *input,
    double exposure_stops)
{
    if (!output || !input) {
        return COLOR_ERROR_INVALID_PARAM;
    }
    
    // Exposure in stops: -5.0 to +5.0
    exposure_stops = CLAMP(exposure_stops, -5.0, 5.0);
    
    // Calculate multiplier: 2^stops
    double multiplier = pow(2.0, exposure_stops);
    
    output->r = CLAMP(input->r * multiplier, 0.0, 1.0);
    output->g = CLAMP(input->g * multiplier, 0.0, 1.0);
    output->b = CLAMP(input->b * multiplier, 0.0, 1.0);
    
    return COLOR_SUCCESS;
}

// ============================================================================
// Contrast Adjustment
// ============================================================================

ColorError contrast_adjust(
    RGBColor *output,
    const RGBColor *input,
    double contrast)
{
    if (!output || !input) {
        return COLOR_ERROR_INVALID_PARAM;
    }
    
    // Contrast: -1.0 to +1.0
    contrast = CLAMP(contrast, -1.0, 1.0);
    
    // Convert to multiplier: 0.0 to 2.0
    double factor = (contrast + 1.0);
    
    // Apply contrast around midpoint (0.5)
    output->r = CLAMP((input->r - 0.5) * factor + 0.5, 0.0, 1.0);
    output->g = CLAMP((input->g - 0.5) * factor + 0.5, 0.0, 1.0);
    output->b = CLAMP((input->b - 0.5) * factor + 0.5, 0.0, 1.0);
    
    return COLOR_SUCCESS;
}

// ============================================================================
// Saturation Adjustment
// ============================================================================

ColorError saturation_adjust(
    RGBColor *output,
    const RGBColor *input,
    double saturation)
{
    if (!output || !input) {
        return COLOR_ERROR_INVALID_PARAM;
    }
    
    // Saturation: -1.0 (grayscale) to +1.0 (double saturation)
    saturation = CLAMP(saturation, -1.0, 1.0);
    
    // Calculate luminance
    double luma = 0.299 * input->r + 0.587 * input->g + 0.114 * input->b;
    
    // Saturation factor: 0.0 to 2.0
    double factor = saturation + 1.0;
    
    // Interpolate between grayscale and original color
    output->r = CLAMP(luma + (input->r - luma) * factor, 0.0, 1.0);
    output->g = CLAMP(luma + (input->g - luma) * factor, 0.0, 1.0);
    output->b = CLAMP(luma + (input->b - luma) * factor, 0.0, 1.0);
    
    return COLOR_SUCCESS;
}

// ============================================================================
// Vibrance Adjustment (smart saturation)
// ============================================================================

ColorError vibrance_adjust(
    RGBColor *output,
    const RGBColor *input,
    double vibrance)
{
    if (!output || !input) {
        return COLOR_ERROR_INVALID_PARAM;
    }
    
    vibrance = CLAMP(vibrance, -1.0, 1.0);
    
    // Calculate current saturation
    double max_val = fmax(input->r, fmax(input->g, input->b));
    double min_val = fmin(input->r, fmin(input->g, input->b));
    double current_sat = (max_val > 0.0) ? (max_val - min_val) / max_val : 0.0;
    
    // Vibrance affects less saturated colors more
    double sat_factor = 1.0 - current_sat;
    double adjustment = vibrance * sat_factor;
    
    // Apply saturation adjustment
    return saturation_adjust(output, input, adjustment);
}

// ============================================================================
// Hue Shift
// ============================================================================

ColorError hue_shift(
    RGBColor *output,
    const RGBColor *input,
    double hue_degrees)
{
    if (!output || !input) {
        return COLOR_ERROR_INVALID_PARAM;
    }
    
    // Convert to HSV
    HSVColor hsv;
    ColorError err = rgb_to_hsv(&hsv, input);
    if (err != COLOR_SUCCESS) {
        return err;
    }
    
    // Shift hue
    hsv.h = fmod(hsv.h + hue_degrees, 360.0);
    if (hsv.h < 0.0) {
        hsv.h += 360.0;
    }
    
    // Convert back to RGB
    return hsv_to_rgb(output, &hsv);
}

// ============================================================================
// Selective Color Adjustment
// ============================================================================

ColorError selective_color_adjust(
    RGBColor *output,
    const RGBColor *input,
    ColorChannel channel,
    double cyan,
    double magenta,
    double yellow,
    double black)
{
    if (!output || !input) {
        return COLOR_ERROR_INVALID_PARAM;
    }
    
    // Clamp adjustments
    cyan = CLAMP(cyan, -1.0, 1.0);
    magenta = CLAMP(magenta, -1.0, 1.0);
    yellow = CLAMP(yellow, -1.0, 1.0);
    black = CLAMP(black, -1.0, 1.0);
    
    // Determine which channel to affect
    double weight = 0.0;
    
    switch (channel) {
        case COLOR_CHANNEL_RED:
            weight = input->r * (1.0 - input->g) * (1.0 - input->b);
            break;
        case COLOR_CHANNEL_GREEN:
            weight = input->g * (1.0 - input->r) * (1.0 - input->b);
            break;
        case COLOR_CHANNEL_BLUE:
            weight = input->b * (1.0 - input->r) * (1.0 - input->g);
            break;
        case COLOR_CHANNEL_CYAN:
            weight = (input->g + input->b) / 2.0 * (1.0 - input->r);
            break;
        case COLOR_CHANNEL_MAGENTA:
            weight = (input->r + input->b) / 2.0 * (1.0 - input->g);
            break;
        case COLOR_CHANNEL_YELLOW:
            weight = (input->r + input->g) / 2.0 * (1.0 - input->b);
            break;
        default:
            return COLOR_ERROR_INVALID_PARAM;
    }
    
    // Apply CMYK adjustments
    output->r = input->r - cyan * weight + magenta * weight;
    output->g = input->g - cyan * weight - yellow * weight;
    output->b = input->b + yellow * weight + magenta * weight;
    
    // Apply black adjustment
    double black_adj = black * weight;
    output->r = CLAMP(output->r - black_adj, 0.0, 1.0);
    output->g = CLAMP(output->g - black_adj, 0.0, 1.0);
    output->b = CLAMP(output->b - black_adj, 0.0, 1.0);
    
    return COLOR_SUCCESS;
}

// ============================================================================
// Shadow/Highlight Adjustment
// ============================================================================

ColorError shadow_highlight_adjust(
    RGBColor *output,
    const RGBColor *input,
    double shadows,
    double highlights)
{
    if (!output || !input) {
        return COLOR_ERROR_INVALID_PARAM;
    }
    
    shadows = CLAMP(shadows, -1.0, 1.0);
    highlights = CLAMP(highlights, -1.0, 1.0);
    
    // Calculate luminance
    double luma = 0.299 * input->r + 0.587 * input->g + 0.114 * input->b;
    
    // Shadow mask (affects darker areas)
    double shadow_mask = 1.0 - luma;
    shadow_mask = shadow_mask * shadow_mask;
    
    // Highlight mask (affects brighter areas)
    double highlight_mask = luma;
    highlight_mask = highlight_mask * highlight_mask;
    
    // Calculate adjustment
    double adjustment = shadows * shadow_mask - highlights * highlight_mask;
    
    output->r = CLAMP(input->r + adjustment, 0.0, 1.0);
    output->g = CLAMP(input->g + adjustment, 0.0, 1.0);
    output->b = CLAMP(input->b + adjustment, 0.0, 1.0);
    
    return COLOR_SUCCESS;
}
// ============================================================================
// Tone Curve Adjustment
// ============================================================================

typedef struct {
    double *points_x;  // Input values
    double *points_y;  // Output values
    int num_points;
} ToneCurve;

ColorError tone_curve_create(
    ToneCurve **curve,
    const double *points_x,
    const double *points_y,
    int num_points)
{
    if (!curve || !points_x || !points_y || num_points < 2) {
        return COLOR_ERROR_INVALID_PARAM;
    }
    
    *curve = (ToneCurve *)malloc(sizeof(ToneCurve));
    if (!*curve) {
        return COLOR_ERROR_OUT_OF_MEMORY;
    }
    
    (*curve)->num_points = num_points;
    (*curve)->points_x = (double *)malloc(num_points * sizeof(double));
    (*curve)->points_y = (double *)malloc(num_points * sizeof(double));
    
    if (!(*curve)->points_x || !(*curve)->points_y) {
        free((*curve)->points_x);
        free((*curve)->points_y);
        free(*curve);
        return COLOR_ERROR_OUT_OF_MEMORY;
    }
    
    memcpy((*curve)->points_x, points_x, num_points * sizeof(double));
    memcpy((*curve)->points_y, points_y, num_points * sizeof(double));
    
    return COLOR_SUCCESS;
}

void tone_curve_destroy(ToneCurve *curve)
{
    if (curve) {
        free(curve->points_x);
        free(curve->points_y);
        free(curve);
    }
}

// Linear interpolation for tone curve
static double tone_curve_interpolate(const ToneCurve *curve, double value)
{
    if (value <= curve->points_x[0]) {
        return curve->points_y[0];
    }
    
    if (value >= curve->points_x[curve->num_points - 1]) {
        return curve->points_y[curve->num_points - 1];
    }
    
    // Find the segment
    for (int i = 0; i < curve->num_points - 1; i++) {
        if (value >= curve->points_x[i] && value <= curve->points_x[i + 1]) {
            double t = (value - curve->points_x[i]) / 
                      (curve->points_x[i + 1] - curve->points_x[i]);
            return curve->points_y[i] + t * (curve->points_y[i + 1] - curve->points_y[i]);
        }
    }
    
    return value;
}

ColorError tone_curve_apply(
    RGBColor *output,
    const RGBColor *input,
    const ToneCurve *curve)
{
    if (!output || !input || !curve) {
        return COLOR_ERROR_INVALID_PARAM;
    }
    
    output->r = CLAMP(tone_curve_interpolate(curve, input->r), 0.0, 1.0);
    output->g = CLAMP(tone_curve_interpolate(curve, input->g), 0.0, 1.0);
    output->b = CLAMP(tone_curve_interpolate(curve, input->b), 0.0, 1.0);
    
    return COLOR_SUCCESS;
}

// ============================================================================
// Levels Adjustment
// ============================================================================

typedef struct {
    double input_black;
    double input_white;
    double input_gamma;
    double output_black;
    double output_white;
} LevelsAdjustment;

ColorError levels_adjust(
    RGBColor *output,
    const RGBColor *input,
    const LevelsAdjustment *levels)
{
    if (!output || !input || !levels) {
        return COLOR_ERROR_INVALID_PARAM;
    }
    
    // Clamp input levels
    double in_black = CLAMP(levels->input_black, 0.0, 1.0);
    double in_white = CLAMP(levels->input_white, 0.0, 1.0);
    double gamma = CLAMP(levels->input_gamma, 0.1, 10.0);
    double out_black = CLAMP(levels->output_black, 0.0, 1.0);
    double out_white = CLAMP(levels->output_white, 0.0, 1.0);
    
    // Ensure input white > input black
    if (in_white <= in_black) {
        in_white = in_black + 0.01;
    }
    
    // Process each channel
    double channels[3] = {input->r, input->g, input->b};
    double *out_channels[3] = {&output->r, &output->g, &output->b};
    
    for (int i = 0; i < 3; i++) {
        // Map input range to 0-1
        double normalized = (channels[i] - in_black) / (in_white - in_black);
        normalized = CLAMP(normalized, 0.0, 1.0);
        
        // Apply gamma
        double gamma_corrected = pow(normalized, 1.0 / gamma);
        
        // Map to output range
        *out_channels[i] = out_black + gamma_corrected * (out_white - out_black);
        *out_channels[i] = CLAMP(*out_channels[i], 0.0, 1.0);
    }
    
    return COLOR_SUCCESS;
}

// ============================================================================
// Color Balance Adjustment
// ============================================================================

typedef struct {
    double cyan_red;      // -1.0 (cyan) to +1.0 (red)
    double magenta_green; // -1.0 (magenta) to +1.0 (green)
    double yellow_blue;   // -1.0 (yellow) to +1.0 (blue)
} ColorBalance;

ColorError color_balance_adjust(
    RGBColor *output,
    const RGBColor *input,
    const ColorBalance *shadows,
    const ColorBalance *midtones,
    const ColorBalance *highlights,
    bool preserve_luminosity)
{
    if (!output || !input) {
        return COLOR_ERROR_INVALID_PARAM;
    }
    
    // Calculate luminance
    double luma = 0.299 * input->r + 0.587 * input->g + 0.114 * input->b;
    
    // Calculate tone masks
    double shadow_mask = 1.0 - luma;
    shadow_mask = shadow_mask * shadow_mask;
    
    double highlight_mask = luma;
    highlight_mask = highlight_mask * highlight_mask;
    
    double midtone_mask = 1.0 - shadow_mask - highlight_mask;
    
    // Initialize output
    output->r = input->r;
    output->g = input->g;
    output->b = input->b;
    
    // Apply shadow adjustments
    if (shadows) {
        output->r += shadows->cyan_red * shadow_mask * 0.3;
        output->g += shadows->magenta_green * shadow_mask * 0.3;
        output->b += shadows->yellow_blue * shadow_mask * 0.3;
    }
    
    // Apply midtone adjustments
    if (midtones) {
        output->r += midtones->cyan_red * midtone_mask * 0.3;
        output->g += midtones->magenta_green * midtone_mask * 0.3;
        output->b += midtones->yellow_blue * midtone_mask * 0.3;
    }
    
    // Apply highlight adjustments
    if (highlights) {
        output->r += highlights->cyan_red * highlight_mask * 0.3;
        output->g += highlights->magenta_green * highlight_mask * 0.3;
        output->b += highlights->yellow_blue * highlight_mask * 0.3;
    }
    
    // Clamp values
    output->r = CLAMP(output->r, 0.0, 1.0);
    output->g = CLAMP(output->g, 0.0, 1.0);
    output->b = CLAMP(output->b, 0.0, 1.0);
    
    // Preserve luminosity if requested
    if (preserve_luminosity) {
        double new_luma = 0.299 * output->r + 0.587 * output->g + 0.114 * output->b;
        if (new_luma > 0.0) {
            double ratio = luma / new_luma;
            output->r = CLAMP(output->r * ratio, 0.0, 1.0);
            output->g = CLAMP(output->g * ratio, 0.0, 1.0);
            output->b = CLAMP(output->b * ratio, 0.0, 1.0);
        }
    }
    
    return COLOR_SUCCESS;
}

// ============================================================================
// HDR Tone Mapping
// ============================================================================

// Reinhard tone mapping
ColorError tonemap_reinhard(
    RGBColor *output,
    const RGBColor *input,
    double white_point)
{
    if (!output || !input) {
        return COLOR_ERROR_INVALID_PARAM;
    }
    
    white_point = fmax(white_point, 1.0);
    double white_sq = white_point * white_point;
    
    // Apply Reinhard tone mapping to each channel
    output->r = (input->r * (1.0 + input->r / white_sq)) / (1.0 + input->r);
    output->g = (input->g * (1.0 + input->g / white_sq)) / (1.0 + input->g);
    output->b = (input->b * (1.0 + input->b / white_sq)) / (1.0 + input->b);
    
    output->r = CLAMP(output->r, 0.0, 1.0);
    output->g = CLAMP(output->g, 0.0, 1.0);
    output->b = CLAMP(output->b, 0.0, 1.0);
    
    return COLOR_SUCCESS;
}

// Filmic tone mapping (ACES approximation)
ColorError tonemap_filmic(
    RGBColor *output,
    const RGBColor *input)
{
    if (!output || !input) {
        return COLOR_ERROR_INVALID_PARAM;
    }
    
    // ACES filmic tone mapping curve
    const double a = 2.51;
    const double b = 0.03;
    const double c = 2.43;
    const double d = 0.59;
    const double e = 0.14;
    
    double channels[3] = {input->r, input->g, input->b};
    double *out_channels[3] = {&output->r, &output->g, &output->b};
    
    for (int i = 0; i < 3; i++) {
        double x = channels[i];
        *out_channels[i] = (x * (a * x + b)) / (x * (c * x + d) + e);
        *out_channels[i] = CLAMP(*out_channels[i], 0.0, 1.0);
    }
    
    return COLOR_SUCCESS;
}

// Uncharted 2 tone mapping
ColorError tonemap_uncharted2(
    RGBColor *output,
    const RGBColor *input,
    double exposure_bias)
{
    if (!output || !input) {
        return COLOR_ERROR_INVALID_PARAM;
    }
    
    const double A = 0.15;
    const double B = 0.50;
    const double C = 0.10;
    const double D = 0.20;
    const double E = 0.02;
    const double F = 0.30;
    
    auto uncharted2_tonemap = [A, B, C, D, E, F](double x) {
        return ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F;
    };
    
    double white_scale = 1.0 / uncharted2_tonemap(11.2);
    
    output->r = uncharted2_tonemap(input->r * exposure_bias) * white_scale;
    output->g = uncharted2_tonemap(input->g * exposure_bias) * white_scale;
    output->b = uncharted2_tonemap(input->b * exposure_bias) * white_scale;
    
    output->r = CLAMP(output->r, 0.0, 1.0);
    output->g = CLAMP(output->g, 0.0, 1.0);
    output->b = CLAMP(output->b, 0.0, 1.0);
    
    return COLOR_SUCCESS;
}

// ============================================================================
// Lift Gamma Gain (Color Grading)
// ============================================================================

typedef struct {
    RGBColor lift;   // Affects shadows
    RGBColor gamma;  // Affects midtones
    RGBColor gain;   // Affects highlights
} LiftGammaGain;

ColorError lift_gamma_gain_adjust(
    RGBColor *output,
    const RGBColor *input,
    const LiftGammaGain *lgg)
{
    if (!output || !input || !lgg) {
        return COLOR_ERROR_INVALID_PARAM;
    }
    
    // Apply lift (add to shadows)
    double r = input->r + lgg->lift.r * (1.0 - input->r);
    double g = input->g + lgg->lift.g * (1.0 - input->g);
    double b = input->b + lgg->lift.b * (1.0 - input->b);
    
    // Apply gamma (power function on midtones)
    if (lgg->gamma.r > 0.0) r = pow(r, 1.0 / lgg->gamma.r);
    if (lgg->gamma.g > 0.0) g = pow(g, 1.0 / lgg->gamma.g);
    if (lgg->gamma.b > 0.0) b = pow(b, 1.0 / lgg->gamma.b);
    
    // Apply gain (multiply highlights)
    r *= lgg->gain.r;
    g *= lgg->gain.g;
    b *= lgg->gain.b;
    
    output->r = CLAMP(r, 0.0, 1.0);
    output->g = CLAMP(g, 0.0, 1.0);
    output->b = CLAMP(b, 0.0, 1.0);
    
    return COLOR_SUCCESS;
}

// ============================================================================
// Color Grading with LUTs (Look-Up Tables)
// ============================================================================

typedef struct {
    int size;           // LUT dimension (e.g., 32 for 32x32x32)
    RGBColor *data;     // Flattened 3D array
} ColorLUT;

ColorError color_lut_create(ColorLUT **lut, int size)
{
    if (!lut || size < 2 || size > 256) {
        return COLOR_ERROR_INVALID_PARAM;
    }
    
    *lut = (ColorLUT *)malloc(sizeof(ColorLUT));
    if (!*lut) {
        return COLOR_ERROR_OUT_OF_MEMORY;
    }
    
    (*lut)->size = size;
    int total_entries = size * size * size;
    (*lut)->data = (RGBColor *)malloc(total_entries * sizeof(RGBColor));
    
    if (!(*lut)->data) {
        free(*lut);
        return COLOR_ERROR_OUT_OF_MEMORY;
    }
    
    // Initialize with identity LUT
    for (int r = 0; r < size; r++) {
        for (int g = 0; g < size; g++) {
            for (int b = 0; b < size; b++) {
                int index = r * size * size + g * size + b;
                (*lut)->data[index].r = (double)r / (size - 1);
                (*lut)->data[index].g = (double)g / (size - 1);
                (*lut)->data[index].b = (double)b / (size - 1);
            }
        }
    }
    
    return COLOR_SUCCESS;
}

void color_lut_destroy(ColorLUT *lut)
{
    if (lut) {
        free(lut->data);
        free(lut);
    }
}

ColorError color_lut_apply(
    RGBColor *output,
    const RGBColor *input,
    const ColorLUT *lut)
{
    if (!output || !input || !lut) {
        return COLOR_ERROR_INVALID_PARAM;
    }
    
    int size = lut->size;
    double scale = size - 1;
    
    // Convert input to LUT coordinates
    double r_coord = CLAMP(input->r * scale, 0.0, scale);
    double g_coord = CLAMP(input->g * scale, 0.0, scale);
    double b_coord = CLAMP(input->b * scale, 0.0, scale);
    
    // Get integer and fractional parts
    int r0 = (int)r_coord;
    int g0 = (int)g_coord;
    int b0 = (int)b_coord;
    
    int r1 = (r0 < size - 1) ? r0 + 1 : r0;
    int g1 = (g0 < size - 1) ? g0 + 1 : g0;
    int b1 = (b0 < size - 1) ? b0 + 1 : b0;
    
    double r_frac = r_coord - r0;
    double g_frac = g_coord - g0;
    double b_frac = b_coord - b0;
    
    // Trilinear interpolation
    RGBColor c000 = lut->data[r0 * size * size + g0 * size + b0];
    RGBColor c001 = lut->data[r0 * size * size + g0 * size + b1];
    RGBColor c010 = lut->data[r0 * size * size + g1 * size + b0];
    RGBColor c011 = lut->data[r0 * size * size + g1 * size + b1];
    RGBColor c100 = lut->data[r1 * size * size + g0 * size + b0];
    RGBColor c101 = lut->data[r1 * size * size + g0 * size + b1];
    RGBColor c110 = lut->data[r1 * size * size + g1 * size + b0];
    RGBColor c111 = lut->data[r1 * size * size + g1 * size + b1];
    
    // Interpolate along b axis
    RGBColor c00, c01, c10, c11;
    c00.r = c000.r + (c001.r - c000.r) * b_frac;
    c00.g = c000.g + (c001.g - c000.g) * b_frac;
    c00.b = c000.b + (c001.b - c000.b) * b_frac;
    
    c01.r = c010.r + (c011.r - c010.r) * b_frac;
    c01.g = c010.g + (c011.g - c010.g) * b_frac;
    c01.b = c010.b + (c011.b - c010.b) * b_frac;
    
    c10.r = c100.r + (c101.r - c100.r) * b_frac;
    c10.g = c100.g + (c101.g - c100.g) * b_frac;
    c10.b = c100.b + (c101.b - c100.b) * b_frac;
    
    c11.r = c110.r + (c111.r - c110.r) * b_frac;
    c11.g = c110.g + (c111.g - c110.g) * b_frac;
    c11.b = c110.b + (c111.b - c110.b) * b_frac;
    
    // Interpolate along g axis
    RGBColor c0, c1;
    c0.r = c00.r + (c01.r - c00.r) * g_frac;
    c0.g = c00.g + (c01.g - c00.g) * g_frac;
    c0.b = c00.b + (c01.b - c00.b) * g_frac;
    
    c1.r = c10.r + (c11.r - c10.r) * g_frac;
    c1.g = c10.g + (c11.g - c10.g) * g_frac;
    c1.b = c10.b + (c11.b - c10.b) * g_frac;
    
    // Interpolate along r axis
    output->r = c0.r + (c1.r - c0.r) * r_frac;
    output->g = c0.g + (c1.g - c0.g) * r_frac;
    output->b = c0.b + (c1.b - c0.b) * r_frac;
    
    return COLOR_SUCCESS;
}
// ============================================================================
// Color Temperature Conversion Utilities
// ============================================================================

// Convert color temperature (Kelvin) to RGB
ColorError temperature_to_rgb(
    RGBColor *output,
    double temperature)
{
    if (!output) {
        return COLOR_ERROR_INVALID_PARAM;
    }
    
    temperature = CLAMP(temperature, 1000.0, 40000.0);
    temperature /= 100.0;
    
    double r, g, b;
    
    // Calculate red
    if (temperature <= 66.0) {
        r = 255.0;
    } else {
        r = temperature - 60.0;
        r = 329.698727446 * pow(r, -0.1332047592);
        r = CLAMP(r, 0.0, 255.0);
    }
    
    // Calculate green
    if (temperature <= 66.0) {
        g = temperature;
        g = 99.4708025861 * log(g) - 161.1195681661;
        g = CLAMP(g, 0.0, 255.0);
    } else {
        g = temperature - 60.0;
        g = 288.1221695283 * pow(g, -0.0755148492);
        g = CLAMP(g, 0.0, 255.0);
    }
    
    // Calculate blue
    if (temperature >= 66.0) {
        b = 255.0;
    } else if (temperature <= 19.0) {
        b = 0.0;
    } else {
        b = temperature - 10.0;
        b = 138.5177312231 * log(b) - 305.0447927307;
        b = CLAMP(b, 0.0, 255.0);
    }
    
    output->r = r / 255.0;
    output->g = g / 255.0;
    output->b = b / 255.0;
    
    return COLOR_SUCCESS;
}

// ============================================================================
// Color Mixing and Blending
// ============================================================================

typedef enum {
    BLEND_NORMAL,
    BLEND_MULTIPLY,
    BLEND_SCREEN,
    BLEND_OVERLAY,
    BLEND_SOFT_LIGHT,
    BLEND_HARD_LIGHT,
    BLEND_COLOR_DODGE,
    BLEND_COLOR_BURN,
    BLEND_LINEAR_DODGE,
    BLEND_LINEAR_BURN,
    BLEND_LIGHTEN,
    BLEND_DARKEN,
    BLEND_DIFFERENCE,
    BLEND_EXCLUSION
} BlendMode;

static double blend_channel(double base, double blend, BlendMode mode)
{
    switch (mode) {
        case BLEND_NORMAL:
            return blend;
            
        case BLEND_MULTIPLY:
            return base * blend;
            
        case BLEND_SCREEN:
            return 1.0 - (1.0 - base) * (1.0 - blend);
            
        case BLEND_OVERLAY:
            if (base < 0.5) {
                return 2.0 * base * blend;
            } else {
                return 1.0 - 2.0 * (1.0 - base) * (1.0 - blend);
            }
            
        case BLEND_SOFT_LIGHT:
            if (blend < 0.5) {
                return 2.0 * base * blend + base * base * (1.0 - 2.0 * blend);
            } else {
                return 2.0 * base * (1.0 - blend) + sqrt(base) * (2.0 * blend - 1.0);
            }
            
        case BLEND_HARD_LIGHT:
            if (blend < 0.5) {
                return 2.0 * base * blend;
            } else {
                return 1.0 - 2.0 * (1.0 - base) * (1.0 - blend);
            }
            
        case BLEND_COLOR_DODGE:
            if (blend >= 1.0) return 1.0;
            return fmin(base / (1.0 - blend), 1.0);
            
        case BLEND_COLOR_BURN:
            if (blend <= 0.0) return 0.0;
            return 1.0 - fmin((1.0 - base) / blend, 1.0);
            
        case BLEND_LINEAR_DODGE:
            return fmin(base + blend, 1.0);
            
        case BLEND_LINEAR_BURN:
            return fmax(base + blend - 1.0, 0.0);
            
        case BLEND_LIGHTEN:
            return fmax(base, blend);
            
        case BLEND_DARKEN:
            return fmin(base, blend);
            
        case BLEND_DIFFERENCE:
            return fabs(base - blend);
            
        case BLEND_EXCLUSION:
            return base + blend - 2.0 * base * blend;
            
        default:
            return blend;
    }
}

ColorError color_blend(
    RGBColor *output,
    const RGBColor *base,
    const RGBColor *blend,
    BlendMode mode,
    double opacity)
{
    if (!output || !base || !blend) {
        return COLOR_ERROR_INVALID_PARAM;
    }
    
    opacity = CLAMP(opacity, 0.0, 1.0);
    
    // Apply blend mode
    double r = blend_channel(base->r, blend->r, mode);
    double g = blend_channel(base->g, blend->g, mode);
    double b = blend_channel(base->b, blend->b, mode);
    
    // Mix with base using opacity
    output->r = base->r + (r - base->r) * opacity;
    output->g = base->g + (g - base->g) * opacity;
    output->b = base->b + (b - base->b) * opacity;
    
    output->r = CLAMP(output->r, 0.0, 1.0);
    output->g = CLAMP(output->g, 0.0, 1.0);
    output->b = CLAMP(output->b, 0.0, 1.0);
    
    return COLOR_SUCCESS;
}

// ============================================================================
// Color Quantization and Posterization
// ============================================================================

ColorError color_posterize(
    RGBColor *output,
    const RGBColor *input,
    int levels)
{
    if (!output || !input || levels < 2) {
        return COLOR_ERROR_INVALID_PARAM;
    }
    
    levels = CLAMP(levels, 2, 256);
    double step = 1.0 / (levels - 1);
    
    output->r = round(input->r / step) * step;
    output->g = round(input->g / step) * step;
    output->b = round(input->b / step) * step;
    
    output->r = CLAMP(output->r, 0.0, 1.0);
    output->g = CLAMP(output->g, 0.0, 1.0);
    output->b = CLAMP(output->b, 0.0, 1.0);
    
    return COLOR_SUCCESS;
}

// ============================================================================
// Color Dithering (Floyd-Steinberg)
// ============================================================================

ColorError color_dither_floyd_steinberg(
    RGBColor *output_buffer,
    const RGBColor *input_buffer,
    int width,
    int height,
    int levels)
{
    if (!output_buffer || !input_buffer || width <= 0 || height <= 0) {
        return COLOR_ERROR_INVALID_PARAM;
    }
    
    levels = CLAMP(levels, 2, 256);
    double step = 1.0 / (levels - 1);
    
    // Create working buffer
    RGBColor *working = (RGBColor *)malloc(width * height * sizeof(RGBColor));
    if (!working) {
        return COLOR_ERROR_OUT_OF_MEMORY;
    }
    
    memcpy(working, input_buffer, width * height * sizeof(RGBColor));
    
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = y * width + x;
            
            // Get old pixel
            RGBColor old_pixel = working[idx];
            
            // Quantize pixel
            RGBColor new_pixel;
            new_pixel.r = round(old_pixel.r / step) * step;
            new_pixel.g = round(old_pixel.g / step) * step;
            new_pixel.b = round(old_pixel.b / step) * step;
            
            output_buffer[idx] = new_pixel;
            
            // Calculate error
            double err_r = old_pixel.r - new_pixel.r;
            double err_g = old_pixel.g - new_pixel.g;
            double err_b = old_pixel.b - new_pixel.b;
            
            // Distribute error to neighboring pixels
            if (x + 1 < width) {
                int idx_right = y * width + (x + 1);
                working[idx_right].r += err_r * 7.0 / 16.0;
                working[idx_right].g += err_g * 7.0 / 16.0;
                working[idx_right].b += err_b * 7.0 / 16.0;
            }
            
            if (y + 1 < height) {
                if (x > 0) {
                    int idx_bl = (y + 1) * width + (x - 1);
                    working[idx_bl].r += err_r * 3.0 / 16.0;
                    working[idx_bl].g += err_g * 3.0 / 16.0;
                    working[idx_bl].b += err_b * 3.0 / 16.0;
                }
                
                int idx_bottom = (y + 1) * width + x;
                working[idx_bottom].r += err_r * 5.0 / 16.0;
                working[idx_bottom].g += err_g * 5.0 / 16.0;
                working[idx_bottom].b += err_b * 5.0 / 16.0;
                
                if (x + 1 < width) {
                    int idx_br = (y + 1) * width + (x + 1);
                    working[idx_br].r += err_r * 1.0 / 16.0;
                    working[idx_br].g += err_g * 1.0 / 16.0;
                    working[idx_br].b += err_b * 1.0 / 16.0;
                }
            }
        }
    }
    
    free(working);
    return COLOR_SUCCESS;
}

// ============================================================================
// Color Histogram
// ============================================================================

typedef struct {
    int bins;
    int *red_histogram;
    int *green_histogram;
    int *blue_histogram;
    int *luminance_histogram;
} ColorHistogram;

ColorError color_histogram_create(
    ColorHistogram **histogram,
    int bins)
{
    if (!histogram || bins < 2 || bins > 1024) {
        return COLOR_ERROR_INVALID_PARAM;
    }
    
    *histogram = (ColorHistogram *)malloc(sizeof(ColorHistogram));
    if (!*histogram) {
        return COLOR_ERROR_OUT_OF_MEMORY;
    }
    
    (*histogram)->bins = bins;
    (*histogram)->red_histogram = (int *)calloc(bins, sizeof(int));
    (*histogram)->green_histogram = (int *)calloc(bins, sizeof(int));
    (*histogram)->blue_histogram = (int *)calloc(bins, sizeof(int));
    (*histogram)->luminance_histogram = (int *)calloc(bins, sizeof(int));
    
    if (!(*histogram)->red_histogram || !(*histogram)->green_histogram ||
        !(*histogram)->blue_histogram || !(*histogram)->luminance_histogram) {
        free((*histogram)->red_histogram);
        free((*histogram)->green_histogram);
        free((*histogram)->blue_histogram);
        free((*histogram)->luminance_histogram);
        free(*histogram);
        return COLOR_ERROR_OUT_OF_MEMORY;
    }
    
    return COLOR_SUCCESS;
}

void color_histogram_destroy(ColorHistogram *histogram)
{
    if (histogram) {
        free(histogram->red_histogram);
        free(histogram->green_histogram);
        free(histogram->blue_histogram);
        free(histogram->luminance_histogram);
        free(histogram);
    }
}

ColorError color_histogram_compute(
    ColorHistogram *histogram,
    const RGBColor *colors,
    int count)
{
    if (!histogram || !colors || count <= 0) {
        return COLOR_ERROR_INVALID_PARAM;
    }
    
    // Reset histograms
    memset(histogram->red_histogram, 0, histogram->bins * sizeof(int));
    memset(histogram->green_histogram, 0, histogram->bins * sizeof(int));
    memset(histogram->blue_histogram, 0, histogram->bins * sizeof(int));
    memset(histogram->luminance_histogram, 0, histogram->bins * sizeof(int));
    
    int bins = histogram->bins;
    
    for (int i = 0; i < count; i++) {
        // Compute bin indices
        int r_bin = (int)(colors[i].r * (bins - 1));
        int g_bin = (int)(colors[i].g * (bins - 1));
        int b_bin = (int)(colors[i].b * (bins - 1));
        
        r_bin = CLAMP(r_bin, 0, bins - 1);
        g_bin = CLAMP(g_bin, 0, bins - 1);
        b_bin = CLAMP(b_bin, 0, bins - 1);
        
        histogram->red_histogram[r_bin]++;
        histogram->green_histogram[g_bin]++;
        histogram->blue_histogram[b_bin]++;
        
        // Compute luminance
        double luma = 0.299 * colors[i].r + 0.587 * colors[i].g + 0.114 * colors[i].b;
        int luma_bin = (int)(luma * (bins - 1));
        luma_bin = CLAMP(luma_bin, 0, bins - 1);
        histogram->luminance_histogram[luma_bin]++;
    }
    
    return COLOR_SUCCESS;
}

// ============================================================================
// Auto Color Correction
// ============================================================================

ColorError auto_levels(
    RGBColor *output,
    const RGBColor *input,
    const ColorHistogram *histogram,
    double black_clip_percent,
    double white_clip_percent)
{
    if (!output || !input || !histogram) {
        return COLOR_ERROR_INVALID_PARAM;
    }
    
    black_clip_percent = CLAMP(black_clip_percent, 0.0, 10.0);
    white_clip_percent = CLAMP(white_clip_percent, 0.0, 10.0);
    
    // Calculate total pixels
    int total_pixels = 0;
    for (int i = 0; i < histogram->bins; i++) {
        total_pixels += histogram->luminance_histogram[i];
    }
    
    int black_threshold = (int)(total_pixels * black_clip_percent / 100.0);
    int white_threshold = (int)(total_pixels * white_clip_percent / 100.0);
    
    // Find black and white points
    int cumulative = 0;
    double black_point = 0.0;
    double white_point = 1.0;
    
    for (int i = 0; i < histogram->bins; i++) {
        cumulative += histogram->luminance_histogram[i];
        if (cumulative >= black_threshold && black_point == 0.0) {
            black_point = (double)i / (histogram->bins - 1);
        }
        if (cumulative >= total_pixels - white_threshold) {
            white_point = (double)i / (histogram->bins - 1);
            break;
        }
    }
    
    // Apply levels adjustment
    LevelsAdjustment levels = {
        .input_black = black_point,
        .input_white = white_point,
        .input_gamma = 1.0,
        .output_black = 0.0,
        .output_white = 1.0
    };
    
    return levels_adjust(output, input, &levels);
}

// ============================================================================
// Color Matching and Distance
// ============================================================================

// Calculate color difference using CIE76 (Euclidean distance in Lab)
double color_difference_cie76(const LABColor *color1, const LABColor *color2)
{
    if (!color1 || !color2) {
        return -1.0;
    }
    
    double dL = color1->L - color2->L;
    double da = color1->a - color2->a;
    double db = color1->b - color2->b;
    
    return sqrt(dL * dL + da * da + db * db);
}

// Calculate color difference using CIE94
double color_difference_cie94(const LABColor *color1, const LABColor *color2)
{
    if (!color1 || !color2) {
        return -1.0;
    }
    
    double dL = color1->L - color2->L;
    double da = color1->a - color2->a;
    double db = color1->b - color2->b;
    
    double C1 = sqrt(color1->a * color1->a + color1->b * color1->b);
    double C2 = sqrt(color2->a * color2->a + color2->b * color2->b);
    double dC = C1 - C2;
    
    double dH_sq = da * da + db * db - dC * dC;
    double dH = (dH_sq > 0) ? sqrt(dH_sq) : 0.0;
    
    // Weighting factors for graphic arts
    double kL = 1.0;
    double kC = 1.0;
    double kH = 1.0;
    double K1 = 0.045;
    double K2 = 0.015;
    
    double SL = 1.0;
    double SC = 1.0 + K1 * C1;
    double SH = 1.0 + K2 * C1;
    
    double dE = sqrt(
        pow(dL / (kL * SL), 2) +
        pow(dC / (kC * SC), 2) +
        pow(dH / (kH * SH), 2)
    );
    
    return dE;
}

// Calculate color difference using CIEDE2000
double color_difference_ciede2000(const LABColor *color1, const LABColor *color2)
{
    if (!color1 || !color2) {
        return -1.0;
    }
    
    // This is a simplified version. Full CIEDE2000 is quite complex.
    double dL = color2->L - color1->L;
    double da = color2->a - color1->a;
    double db = color2->b - color1->b;
    
    double C1 = sqrt(color1->a * color1->a + color1->b * color1->b);
    double C2 = sqrt(color2->a * color2->a + color2->b * color2->b);
    double C_avg = (C1 + C2) / 2.0;
    
    double G = 0.5 * (1.0 - sqrt(pow(C_avg, 7) / (pow(C_avg, 7) + pow(25.0, 7))));
    
    double a1_prime = color1->a * (1.0 + G);
    double a2_prime = color2->a * (1.0 + G);
    
    double C1_prime = sqrt(a1_prime * a1_prime + color1->b * color1->b);
    double C2_prime = sqrt(a2_prime * a2_prime + color2->b * color2->b);
    
    double dC_prime = C2_prime - C1_prime;
    
    double h1_prime = atan2(color1->b, a1_prime) * 180.0 / M_PI;
    if (h1_prime < 0) h1_prime += 360.0;
    
    double h2_prime = atan2(color2->b, a2_prime) * 180.0 / M_PI;
    if (h2_prime < 0) h2_prime += 360.0;
    
    double dh_prime = h2_prime - h1_prime;
    if (fabs(dh_prime) > 180.0) {
        if (h2_prime <= h1_prime) {
            dh_prime += 360.0;
        } else {
            dh_prime -= 360.0;
        }
    }
    
    double dH_prime = 2.0 * sqrt(C1_prime * C2_prime) * sin(dh_prime * M_PI / 360.0);
    
    // Simplified weighting (full CIEDE2000 has more complex weighting)
    double dE = sqrt(dL * dL + dC_prime * dC_prime + dH_prime * dH_prime);
    
    return dE;
}

// Find closest color from palette
int find_closest_color(
    const RGBColor *color,
    const RGBColor *palette,
    int palette_size)
{
    if (!color || !palette || palette_size <= 0) {
        return -1;
    }
    
    LABColor target_lab;
    rgb_to_lab(&target_lab, color);
    
    int closest_index = 0;
    double min_distance = INFINITY;
    
    for (int i = 0; i < palette_size; i++) {
        LABColor palette_lab;
        rgb_to_lab(&palette_lab, &palette[i]);
        
        double distance = color_difference_cie76(&target_lab, &palette_lab);
        
        if (distance < min_distance) {
            min_distance = distance;
            closest_index = i;
        }
    }
    
    return closest_index;
}

// ============================================================================
// Utility: Print Color Information
// ============================================================================

void print_color_info(const RGBColor *rgb)
{
    if (!rgb) return;
    
    printf("RGB: (%.3f, %.3f, %.3f)\n", rgb->r, rgb->g, rgb->b);
    
    HSVColor hsv;
    rgb_to_hsv(&hsv, rgb);
    printf("HSV: (%.1f°, %.1f%%, %.1f%%)\n", 
           hsv.h, hsv.s * 100.0, hsv.v * 100.0);
    
    HSLColor hsl;
    rgb_to_hsl(&hsl, rgb);
    printf("HSL: (%.1f°, %.1f%%, %.1f%%)\n", 
           hsl.h, hsl.s * 100.0, hsl.l * 100.0);
    
    LABColor lab;
    rgb_to_lab(&lab, rgb);
    printf("LAB: (%.2f, %.2f, %.2f)\n", lab.L, lab.a, lab.b);
    
    CMYKColor cmyk;
    rgb_to_cmyk(&cmyk, rgb);
    printf("CMYK: (%.1f%%, %.1f%%, %.1f%%, %.1f%%)\n",
           cmyk.c * 100.0, cmyk.m * 100.0, cmyk.y * 100.0, cmyk.k * 100.0);
}

// End of color_correction.c

