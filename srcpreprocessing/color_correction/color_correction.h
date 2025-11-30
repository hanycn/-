/**
 * @file color_correction.h
 * @brief Color correction and color space conversion header
 * @author hany
 * @version 1.0
 * @date 2025
 */

#ifndef COLOR_CORRECTION_H
#define COLOR_CORRECTION_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Error Codes
// ============================================================================

typedef enum {
    COLOR_SUCCESS = 0,
    COLOR_ERROR_INVALID_PARAM,
    COLOR_ERROR_OUT_OF_MEMORY,
    COLOR_ERROR_INVALID_MATRIX,
    COLOR_ERROR_INVALID_COLORSPACE,
    COLOR_ERROR_CONVERSION_FAILED,
    COLOR_ERROR_LUT_FAILED,
    COLOR_ERROR_PROFILE_FAILED
} ColorError;

// ============================================================================
// Enumerations
// ============================================================================

typedef enum {
    COLORSPACE_SRGB = 0,
    COLORSPACE_ADOBE_RGB,
    COLORSPACE_PROPHOTO_RGB,
    COLORSPACE_XYZ,
    COLORSPACE_LAB,
    COLORSPACE_HSV,
    COLORSPACE_HSL,
    COLORSPACE_YCBCR,
    COLORSPACE_YCBCR_709,
    COLORSPACE_YCBCR_2020,
    COLORSPACE_LINEAR_RGB,
    COLORSPACE_CUSTOM
} ColorSpace;

typedef enum {
    WHITE_POINT_D50 = 0,
    WHITE_POINT_D55,
    WHITE_POINT_D65,
    WHITE_POINT_D75,
    WHITE_POINT_A,
    WHITE_POINT_B,
    WHITE_POINT_C,
    WHITE_POINT_E,
    WHITE_POINT_F2,
    WHITE_POINT_F7,
    WHITE_POINT_F11,
    WHITE_POINT_CUSTOM
} WhitePoint;

typedef enum {
    GAMMA_LINEAR = 0,
    GAMMA_SRGB,
    GAMMA_REC709,
    GAMMA_POWER,
    GAMMA_L_STAR,
    GAMMA_CUSTOM
} GammaCurve;

typedef enum {
    INTENT_PERCEPTUAL = 0,
    INTENT_RELATIVE,
    INTENT_SATURATION,
    INTENT_ABSOLUTE
} RenderingIntent;

typedef enum {
    TONEMAP_LINEAR = 0,
    TONEMAP_REINHARD,
    TONEMAP_REINHARD_LOCAL,
    TONEMAP_DRAGO,
    TONEMAP_MANTIUK,
    TONEMAP_FILMIC,
    TONEMAP_UNCHARTED2,
    TONEMAP_ACES,
    TONEMAP_CUSTOM
} ToneMapOperator;

// ============================================================================
// Data Structures
// ============================================================================

typedef struct {
    double m[3][3];
} ColorMatrix3x3;

typedef struct {
    double m[3][4];
} ColorMatrix3x4;

typedef struct {
    double r;
    double g;
    double b;
} RGBColor;

typedef struct {
    double x;
    double y;
    double z;
} XYZColor;

typedef struct {
    double l;
    double a;
    double b;
} LabColor;

typedef struct {
    double h;
    double s;
    double v;
} HSVColor;

typedef struct {
    double h;
    double s;
    double l;
} HSLColor;

typedef struct {
    double y;
    double cb;
    double cr;
} YCbCrColor;

typedef struct {
    double x;
    double y;
    double Y;
} WhitePointCoords;

typedef struct {
    double rx, ry;
    double gx, gy;
    double bx, by;
} RGBPrimaries;

typedef struct {
    ColorSpace type;
    RGBPrimaries primaries;
    WhitePointCoords white;
    GammaCurve gamma_type;
    double gamma_value;
    ColorMatrix3x3 to_xyz;
    ColorMatrix3x3 from_xyz;
    char name[64];
} ColorSpaceDefinition;

typedef struct {
    int enabled;
    ColorMatrix3x3 matrix;
    double brightness;
    double contrast;
    double saturation;
    double hue_shift;
    double temperature;
    double tint;
    int use_lut;
    char lut_file[256];
} ColorCorrectionConfig;

typedef struct {
    ToneMapOperator operator;
    double exposure;
    double white_point;
    double key_value;
    double local_adaptation;
    double saturation;
    int auto_exposure;
} ToneMappingConfig;

typedef struct {
    char description[128];
    ColorSpace colorspace;
    WhitePoint white_point;
    GammaCurve gamma_curve;
    ColorMatrix3x3 matrix;
    void *lut_data;
    size_t lut_size;
} ColorProfile;

typedef struct {
    int size;
    double *data;
    double input_min;
    double input_max;
} LUT1D;

typedef struct {
    int size;
    double *data;
    double input_min[3];
    double input_max[3];
} LUT3D;

// ============================================================================
// Matrix Operations
// ============================================================================

void color_matrix_identity(ColorMatrix3x3 *matrix);

void color_matrix_multiply(
    ColorMatrix3x3 *result,
    const ColorMatrix3x3 *a,
    const ColorMatrix3x3 *b
);

ColorError color_matrix_invert(
    ColorMatrix3x3 *result,
    const ColorMatrix3x3 *matrix
);

void color_matrix_apply(
    RGBColor *result,
    const ColorMatrix3x3 *matrix,
    const RGBColor *color
);

void color_matrix_transpose(
    ColorMatrix3x3 *result,
    const ColorMatrix3x3 *matrix
);

double color_matrix_determinant(const ColorMatrix3x3 *matrix);
// ============================================================================
// Color Space Conversion Functions
// ============================================================================

ColorError rgb_to_xyz(
    XYZColor *xyz,
    const RGBColor *rgb,
    const ColorSpaceDefinition *colorspace
);

ColorError xyz_to_rgb(
    RGBColor *rgb,
    const XYZColor *xyz,
    const ColorSpaceDefinition *colorspace
);

ColorError xyz_to_lab(
    LabColor *lab,
    const XYZColor *xyz,
    const WhitePointCoords *white_point
);

ColorError lab_to_xyz(
    XYZColor *xyz,
    const LabColor *lab,
    const WhitePointCoords *white_point
);

void rgb_to_hsv(HSVColor *hsv, const RGBColor *rgb);

void hsv_to_rgb(RGBColor *rgb, const HSVColor *hsv);

void rgb_to_hsl(HSLColor *hsl, const RGBColor *rgb);

void hsl_to_rgb(RGBColor *rgb, const HSLColor *hsl);

void rgb_to_ycbcr(
    YCbCrColor *ycbcr,
    const RGBColor *rgb,
    ColorSpace standard
);

void ycbcr_to_rgb(
    RGBColor *rgb,
    const YCbCrColor *ycbcr,
    ColorSpace standard
);

ColorError convert_colorspace(
    RGBColor *output,
    const RGBColor *input,
    const ColorSpaceDefinition *from_space,
    const ColorSpaceDefinition *to_space,
    RenderingIntent intent
);

// ============================================================================
// White Point and Chromatic Adaptation
// ============================================================================

ColorError get_white_point(
    WhitePointCoords *coords,
    WhitePoint white_point
);

ColorError chromatic_adaptation_bradford(
    ColorMatrix3x3 *matrix,
    const WhitePointCoords *source_white,
    const WhitePointCoords *dest_white
);

ColorError chromatic_adaptation_von_kries(
    ColorMatrix3x3 *matrix,
    const WhitePointCoords *source_white,
    const WhitePointCoords *dest_white
);

ColorError chromatic_adaptation_xyz_scaling(
    ColorMatrix3x3 *matrix,
    const WhitePointCoords *source_white,
    const WhitePointCoords *dest_white
);

// ============================================================================
// Gamma Correction Functions
// ============================================================================

ColorError apply_gamma(
    double *output,
    double input,
    GammaCurve gamma_type,
    double gamma_value
);

ColorError remove_gamma(
    double *output,
    double input,
    GammaCurve gamma_type,
    double gamma_value
);

double srgb_gamma_encode(double value);

double srgb_gamma_decode(double value);

double rec709_gamma_encode(double value);

double rec709_gamma_decode(double value);

// ============================================================================
// Color Space Definitions
// ============================================================================

ColorError get_srgb_colorspace(ColorSpaceDefinition *colorspace);

ColorError get_adobe_rgb_colorspace(ColorSpaceDefinition *colorspace);

ColorError get_prophoto_rgb_colorspace(ColorSpaceDefinition *colorspace);

ColorError get_rec709_colorspace(ColorSpaceDefinition *colorspace);

ColorError get_rec2020_colorspace(ColorSpaceDefinition *colorspace);

ColorError create_custom_colorspace(
    ColorSpaceDefinition *colorspace,
    const RGBPrimaries *primaries,
    const WhitePointCoords *white_point,
    GammaCurve gamma_type,
    double gamma_value
);

// ============================================================================
// Color Correction Functions
// ============================================================================

ColorError apply_color_correction_matrix(
    double *image,
    int width,
    int height,
    const ColorMatrix3x3 *matrix
);

ColorError adjust_brightness(
    double *image,
    int width,
    int height,
    double brightness
);

ColorError adjust_contrast(
    double *image,
    int width,
    int height,
    double contrast
);

ColorError adjust_saturation(
    double *image,
    int width,
    int height,
    double saturation
);

ColorError adjust_hue(
    double *image,
    int width,
    int height,
    double hue_shift
);

ColorError adjust_temperature(
    double *image,
    int width,
    int height,
    double temperature
);

ColorError adjust_tint(
    double *image,
    int width,
    int height,
    double tint
);

ColorError apply_color_correction(
    double *image,
    int width,
    int height,
    const ColorCorrectionConfig *config
);
// ============================================================================
// Color Space Conversion Functions
// ============================================================================

ColorError rgb_to_xyz(
    XYZColor *xyz,
    const RGBColor *rgb,
    const ColorSpaceDefinition *colorspace
);

ColorError xyz_to_rgb(
    RGBColor *rgb,
    const XYZColor *xyz,
    const ColorSpaceDefinition *colorspace
);

ColorError xyz_to_lab(
    LabColor *lab,
    const XYZColor *xyz,
    const WhitePointCoords *white_point
);

ColorError lab_to_xyz(
    XYZColor *xyz,
    const LabColor *lab,
    const WhitePointCoords *white_point
);

void rgb_to_hsv(HSVColor *hsv, const RGBColor *rgb);

void hsv_to_rgb(RGBColor *rgb, const HSVColor *hsv);

void rgb_to_hsl(HSLColor *hsl, const RGBColor *rgb);

void hsl_to_rgb(RGBColor *rgb, const HSLColor *hsl);

void rgb_to_ycbcr(
    YCbCrColor *ycbcr,
    const RGBColor *rgb,
    ColorSpace standard
);

void ycbcr_to_rgb(
    RGBColor *rgb,
    const YCbCrColor *ycbcr,
    ColorSpace standard
);

ColorError convert_colorspace(
    RGBColor *output,
    const RGBColor *input,
    const ColorSpaceDefinition *from_space,
    const ColorSpaceDefinition *to_space,
    RenderingIntent intent
);

// ============================================================================
// White Point and Chromatic Adaptation
// ============================================================================

ColorError get_white_point(
    WhitePointCoords *coords,
    WhitePoint white_point
);

ColorError chromatic_adaptation_bradford(
    ColorMatrix3x3 *matrix,
    const WhitePointCoords *source_white,
    const WhitePointCoords *dest_white
);

ColorError chromatic_adaptation_von_kries(
    ColorMatrix3x3 *matrix,
    const WhitePointCoords *source_white,
    const WhitePointCoords *dest_white
);

ColorError chromatic_adaptation_xyz_scaling(
    ColorMatrix3x3 *matrix,
    const WhitePointCoords *source_white,
    const WhitePointCoords *dest_white
);

// ============================================================================
// Gamma Correction Functions
// ============================================================================

ColorError apply_gamma(
    double *output,
    double input,
    GammaCurve gamma_type,
    double gamma_value
);

ColorError remove_gamma(
    double *output,
    double input,
    GammaCurve gamma_type,
    double gamma_value
);

double srgb_gamma_encode(double value);

double srgb_gamma_decode(double value);

double rec709_gamma_encode(double value);

double rec709_gamma_decode(double value);

// ============================================================================
// Color Space Definitions
// ============================================================================

ColorError get_srgb_colorspace(ColorSpaceDefinition *colorspace);

ColorError get_adobe_rgb_colorspace(ColorSpaceDefinition *colorspace);

ColorError get_prophoto_rgb_colorspace(ColorSpaceDefinition *colorspace);

ColorError get_rec709_colorspace(ColorSpaceDefinition *colorspace);

ColorError get_rec2020_colorspace(ColorSpaceDefinition *colorspace);

ColorError create_custom_colorspace(
    ColorSpaceDefinition *colorspace,
    const RGBPrimaries *primaries,
    const WhitePointCoords *white_point,
    GammaCurve gamma_type,
    double gamma_value
);

// ============================================================================
// Color Correction Functions
// ============================================================================

ColorError apply_color_correction_matrix(
    double *image,
    int width,
    int height,
    const ColorMatrix3x3 *matrix
);

ColorError adjust_brightness(
    double *image,
    int width,
    int height,
    double brightness
);

ColorError adjust_contrast(
    double *image,
    int width,
    int height,
    double contrast
);

ColorError adjust_saturation(
    double *image,
    int width,
    int height,
    double saturation
);

ColorError adjust_hue(
    double *image,
    int width,
    int height,
    double hue_shift
);

ColorError adjust_temperature(
    double *image,
    int width,
    int height,
    double temperature
);

ColorError adjust_tint(
    double *image,
    int width,
    int height,
    double tint
);

ColorError apply_color_correction(
    double *image,
    int width,
    int height,
    const ColorCorrectionConfig *config
);

