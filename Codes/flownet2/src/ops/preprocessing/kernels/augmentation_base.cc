#include "augmentation_base.h"

#include <math.h>
#include <random>

namespace tensorflow {
/** TransMat Functions **/
void AugmentationLayerBase::TransMat::fromCoeff(AugmentationCoeff *coeff,
                                                int                out_width,
                                                int                out_height,
                                                int                src_width,
                                                int                src_height) {
  leftMultiply(1, 0, -0.5 * out_width,
               0, 1, -0.5 * out_height);

  if (coeff->angle) {
    leftMultiply(cos(coeff->angle()), -sin(coeff->angle()), 0,
                 sin(coeff->angle()), cos(coeff->angle()), 0);
  }

  if (coeff->dx || coeff->dy) {
    leftMultiply(1, 0, coeff->dx() * out_width,
                 0, 1, coeff->dy() * out_height);
  }

  if (coeff->zoom_x || coeff->zoom_y) {
    leftMultiply(1.0 / coeff->zoom_x(), 0, 0,
                 0, 1.0 / coeff->zoom_y(), 0);
  }

  leftMultiply(1, 0, 0.5 * src_width,
               0, 1, 0.5 * src_height);
}

void AugmentationLayerBase::TransMat::fromTensor(const float *tensor_data) {
  t0 = tensor_data[0];
  t1 = tensor_data[1];
  t2 = tensor_data[2];
  t3 = tensor_data[3];
  t4 = tensor_data[4];
  t5 = tensor_data[5];
}

AugmentationLayerBase::TransMat AugmentationLayerBase::TransMat::inverse() {
  float a = this->t0, b = this->t1, c = this->t2;
  float d = this->t3, e = this->t4, f = this->t5;

  float denom = a * e - b * d;

  TransMat result;

  result.t0 = e / denom;
  result.t1 = b / -denom;
  result.t2 = (c * e - b * f) / -denom;
  result.t3 = d / -denom;
  result.t4 = a / denom;
  result.t5 = (c * d - a * f) / denom;

  return result;
}

void AugmentationLayerBase::TransMat::leftMultiply(float u0,
                                                   float u1,
                                                   float u2,
                                                   float u3,
                                                   float u4,
                                                   float u5) {
  float t0 = this->t0, t1 = this->t1, t2 = this->t2;
  float t3 = this->t3, t4 = this->t4, t5 = this->t5;

  this->t0 = t0 * u0 + t3 * u1;
  this->t1 = t1 * u0 + t4 * u1;
  this->t2 = t2 * u0 + t5 * u1 + u2;
  this->t3 = t0 * u3 + t3 * u4;
  this->t4 = t1 * u3 + t4 * u4;
  this->t5 = t2 * u3 + t5 * u4 + u5;
}

void AugmentationLayerBase::TransMat::toIdentity() {
  t0 = 1; t1 = 0; t2 = 0;
  t3 = 0; t4 = 1; t5 = 0;
}

/** AugmentationCoeff Functions **/
void AugmentationCoeff::clear() {
  // Spatial variables
  dx.clear();
  dy.clear();
  angle.clear();
  zoom_x.clear();
  zoom_y.clear();

  // Chromatic variables
  gamma.clear();
  brightness.clear();
  contrast.clear();
  color1.clear();
  color2.clear();
  color3.clear();
}

void AugmentationCoeff::combine_with(const AugmentationCoeff& coeff) {
  // Spatial types
  if (coeff.dx) {
    dx = dx() * coeff.dx();
  }

  if (coeff.dy) {
    dy = dy() * coeff.dy();
  }

  if (coeff.angle) {
    angle = angle() * coeff.angle();
  }

  if (coeff.zoom_x) {
    zoom_x = zoom_x() * coeff.zoom_x();
  }

  if (coeff.zoom_y) {
    zoom_y = zoom_y() * coeff.zoom_y();
  }

  // Chromatic types
  if (coeff.gamma) {
    gamma = gamma() * coeff.gamma();
  }

  if (coeff.brightness) {
    brightness = brightness() * coeff.brightness();
  }

  if (coeff.contrast) {
    contrast = contrast() * coeff.contrast();
  }

  if (coeff.color1) {
    color1 = color1() * coeff.color1();
  }

  if (coeff.color2) {
    color2 = color2() * coeff.color2();
  }

  if (coeff.color3) {
    color3 = color3() * coeff.color3();
  }
}

void AugmentationCoeff::replace_with(const AugmentationCoeff& coeff) {
  // Spatial types
  if (coeff.dx) {
    dx = coeff.dx();
  }

  if (coeff.dy) {
    dy = coeff.dy();
  }

  if (coeff.angle) {
    angle = coeff.angle();
  }

  if (coeff.zoom_x) {
    zoom_x = coeff.zoom_x();
  }

  if (coeff.zoom_y) {
    zoom_y = coeff.zoom_y();
  }

  // Chromatic types
  if (coeff.gamma) {
    gamma = gamma() * coeff.gamma();
  }

  if (coeff.brightness) {
    brightness = coeff.brightness();
  }

  if (coeff.contrast) {
    contrast = coeff.contrast();
  }

  if (coeff.color1) {
    color1 = coeff.color1();
  }

  if (coeff.color2) {
    color2 = coeff.color2();
  }

  if (coeff.color3) {
    color3 = coeff.color3();
  }
}

/** AugmentationLayerBase Functions **/
float AugmentationLayerBase::rng_generate(const AugmentationParam& param,
                                          float                    discount_coeff,
                                          const float              default_value) {
  std::random_device rd;  // Will be used to obtain a seed for the random number
                          // engine
  std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()

  float spread = param.spread * discount_coeff;

  if (param.rand_type == "uniform_bernoulli") {
    float tmp1 = 0.0;
    bool  tmp2 = false;

    if (param.prob > 0.0) {
      std::bernoulli_distribution bernoulli(param.prob);
      tmp2 = bernoulli(gen);
    }

    if (!tmp2) {
      return default_value;
    }

    if (param.spread > 0.0) {
      std::uniform_real_distribution<> uniform(param.mean - spread,
                                               param.mean + spread);
      tmp1 = uniform(gen);
    } else {
      tmp1 = param.mean;
    }

    if (param.should_exp) {
      tmp1 = exp(tmp1);
    }

    return tmp1;
  } else if (param.rand_type == "gaussian_bernoulli") {
    float tmp1 = 0.0;
    bool  tmp2 = false;

    if (param.prob > 0.0) {
      std::bernoulli_distribution bernoulli(param.prob);
      tmp2 = bernoulli(gen);
    }

    if (!tmp2) {
      return default_value;
    }

    if (spread > 0.0) {
      std::normal_distribution<> normal(param.mean, spread);
      tmp1 = normal(gen);
    } else {
      tmp1 = param.mean;
    }

    if (param.should_exp) {
      tmp1 = exp(tmp1);
    }

    return tmp1;
  } else {
    throw "Unknown random type: " + param.rand_type;
  }
}

void AugmentationLayerBase::generate_chromatic_coeffs(float                     discount_coeff,
                                                      const AugmentationParams& aug,
                                                      AugmentationCoeff       & coeff) {
  if (aug.gamma) {
    coeff.gamma = rng_generate(aug.gamma(), discount_coeff, coeff.gamma.get_default());
  }

  if (aug.brightness) {
    coeff.brightness =
      rng_generate(aug.brightness(), discount_coeff, coeff.brightness.get_default());
  }

  if (aug.contrast) {
    coeff.contrast = rng_generate(aug.contrast(), discount_coeff, coeff.contrast.get_default());
  }

  if (aug.color) {
    coeff.color1 = rng_generate(aug.color(), discount_coeff, coeff.color1.get_default());
    coeff.color2 = rng_generate(aug.color(), discount_coeff, coeff.color2.get_default());
    coeff.color3 = rng_generate(aug.color(), discount_coeff, coeff.color3.get_default());
  }
}

void AugmentationLayerBase::generate_spatial_coeffs(float                     discount_coeff,
                                                    const AugmentationParams& aug,
                                                    AugmentationCoeff       & coeff) {
  if (aug.translate) {
    coeff.dx = rng_generate(aug.translate(), discount_coeff, coeff.dx.get_default());
    coeff.dy = rng_generate(aug.translate(), discount_coeff, coeff.dy.get_default());
  }

  if (aug.rotate) {
    coeff.angle = rng_generate(aug.rotate(), discount_coeff, coeff.angle.get_default());
  }

  if (aug.zoom) {
    coeff.zoom_x = rng_generate(aug.zoom(), discount_coeff, coeff.zoom_x.get_default());
    coeff.zoom_y = coeff.zoom_x();
  }

  if (aug.squeeze) {
    float squeeze_coeff = rng_generate(aug.squeeze(), discount_coeff, 1.0);
    coeff.zoom_x = coeff.zoom_x() * squeeze_coeff;
    coeff.zoom_y = coeff.zoom_y() * squeeze_coeff;
  }
}

void AugmentationLayerBase::generate_valid_spatial_coeffs(
  float                     discount_coeff,
  const AugmentationParams& aug,
  AugmentationCoeff       & coeff,
  int                       src_width,
  int                       src_height,
  int                       out_width,
  int                       out_height) {
  int   x, y;
  float x1, y1, x2, y2;
  int   counter     = 0;
  int   good_params = 0;
  AugmentationCoeff incoming_coeff(coeff);

  while (good_params < 4 && counter < 50) {
    coeff.clear();
    AugmentationLayerBase::generate_spatial_coeffs(discount_coeff, aug, coeff);
    coeff.combine_with(incoming_coeff);

    // Check if all 4 corners of the transformed image fit into the original
    // image
    good_params = 0;

    for (x = 0; x < out_width; x += out_width - 1) {
      for (y = 0; y < out_height; y += out_height - 1) {
        // move the origin
        x1 = x - 0.5 * out_width;
        y1 = y - 0.5 * out_height;

        // rotate
        x2 = cos(coeff.angle()) * x1 - sin(coeff.angle()) * y1;
        y2 = sin(coeff.angle()) * x1 + sin(coeff.angle()) * y1;

        // translate
        x2 = x2 + coeff.dx() * out_width;
        y2 = y2 + coeff.dy() * out_height;

        // zoom
        x2 = x2 / coeff.zoom_x();
        y2 = y2 / coeff.zoom_y();

        // move the origin back
        x2 = x2 + 0.5 * src_width;
        y2 = y2 + 0.5 * src_height;

        if (!((floor(x2) < 0) || (floor(x2) > src_width - 2.0) ||
              (floor(y2) < 0) || (floor(y2) > src_height - 2.0))) {
          good_params++;
        }
      }
    }
    counter++;
  }

  if (counter >= 50) {
    printf("Warning: No suitable spatial transformation after %d attempts.\n", counter);
    coeff.clear();
    coeff.replace_with(incoming_coeff);
  }
}

void AugmentationLayerBase::copy_chromatic_coeffs_to_tensor(
  const std::vector<AugmentationCoeff>& coeff_arr,
  typename TTypes<float, 2>::Tensor& out)
{
  float *out_ptr = out.data();
  int    counter = 0;

  for (AugmentationCoeff coeff : coeff_arr) {
    out_ptr[counter + 0] = coeff.gamma();
    out_ptr[counter + 1] = coeff.brightness();
    out_ptr[counter + 2] = coeff.contrast();
    out_ptr[counter + 3] = coeff.color1();
    out_ptr[counter + 4] = coeff.color2();
    out_ptr[counter + 5] = coeff.color3();
    counter             += 6;
  }
}

void AugmentationLayerBase::copy_spatial_coeffs_to_tensor(
  const std::vector<AugmentationCoeff>& coeff_arr,
  const int out_width,
  const int out_height,
  const int src_width,
  const int src_height,
  typename TTypes<float, 2>::Tensor& out,
  const bool invert)
{
  float   *out_ptr = out.data();
  int      counter = 0;
  TransMat t;

  for (AugmentationCoeff coeff : coeff_arr) {
    t.toIdentity();
    t.fromCoeff(&coeff, out_width, out_height, src_width, src_height);

    if (invert) {
      t = t.inverse();
    }

    out_ptr[counter + 0] = t.t0;
    out_ptr[counter + 1] = t.t1;
    out_ptr[counter + 2] = t.t2;
    out_ptr[counter + 3] = t.t3;
    out_ptr[counter + 4] = t.t4;
    out_ptr[counter + 5] = t.t5;
    counter             += 6;
  }
}
}
