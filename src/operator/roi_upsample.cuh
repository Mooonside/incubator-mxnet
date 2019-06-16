#ifndef MXNET_OPERATOR_ROI_UPSAMPLE_CUH_
#define MXNET_OPERATOR_ROI_UPSAMPLE_CUH_

#include "mxnet_op.h"
#include <mxnet/base.h>
#include <mxnet/operator.h>
#include "cuda.h"


namespace mxnet {
namespace op {

  using namespace mshadow::cuda;
// The maximum number of blocks to use in the default kernel call.
  constexpr int ROI_MAXIMUM_NUM_BLOCKS = 4096;
/**
 * @brief Compute the number of blocks needed to run N threads.
 */
  inline int ROI_GET_BLOCKS(const int N) {
    return std::max(
        std::min(
            (N + kMaxThreadsPerBlock - 1) / kMaxThreadsPerBlock,
            ROI_MAXIMUM_NUM_BLOCKS),
        // Use at least 1 block, since CUDA does not allow empty block
        1);
  }

  template <typename T>
  __device__ void inverse_bilinear_interpolate(
      const int height,
      const int width,
      T y,
      T x,
      T* w1,
      T* w2,
      T* w3,
      T* w4,
      int* x_low,
      int* x_high,
      int* y_low,
      int* y_high) {
    // deal with cases that inverse elements are out of feature map boundary
    if (y < -1.0 || y > height || x < -1.0 || x > width) {
      // empty
      *w1 = *w2 = *w3 = *w4 = 0.;
      *x_low = *x_high = *y_low = *y_high = -1;
      return;
    }

    if (y <= 0) {
      y = 0;
    }
    if (x <= 0) {
      x = 0;
    }

    *y_low = static_cast<int>(y);
    *x_low = static_cast<int>(x);

    if (*y_low >= height - 1) {
      *y_high = *y_low = height - 1;
      y = (T)*y_low;
    } else {
      *y_high = *y_low + 1;
    }

    if (*x_low >= width - 1) {
      *x_high = *x_low = width - 1;
      x = (T)*x_low;
    } else {
      *x_high = *x_low + 1;
    }

    T ly = y - *y_low;
    T lx = x - *x_low;
    T hy = 1. - ly, hx = 1. - lx;

    // normalize as introduced in paper
    T value_x = hx * hx + lx * lx;
    T value_y = hy * hy + ly * ly;
    T value = value_x * value_y;


    *w1 = hy * hx / value, *w2 = hy * lx / value, *w3 = ly * hx / value, *w4 = ly * lx / value;
    return;
  }


  template <typename T>
  __global__ void RoIUpsampleForwardKernel(
      const int nthreads,
      const T *roi_features,
      const T *bottom_rois,
      const int batch_size,
      const int channels,
      const int height,
      const int width,
      const int pooled_height,
      const int pooled_width,
      const int sampling_ratio,
      int rois_cols,
      T *top_data) {
    CUDA_KERNEL_LOOP(index, nthreads) {
      // (n, c, ph, pw) is an element in the pooled output
      int pw = index % pooled_width;
      int ph = (index / pooled_width) % pooled_height;
      int c = (index / pooled_width / pooled_height) % channels;
      int n = index / pooled_width / pooled_height / channels;

      int roi_batch_ind = 0;
      const T *offset_bottom_rois = bottom_rois + n * rois_cols;
      const T *offset_roi_features = roi_features + ((n * channels + c) * pooled_height + ph) * pooled_width + pw;

      if (rois_cols == 5) {
        roi_batch_ind = static_cast<int>(offset_bottom_rois[0]);
        offset_bottom_rois++;
      }

      if (roi_batch_ind >= batch_size or roi_batch_ind < 0) {
        continue;
      }
      T *offset_top_data = top_data + (roi_batch_ind * channels + c) * height * width;

      T roi_start_w = offset_bottom_rois[0];
      T roi_start_h = offset_bottom_rois[1];
      T roi_end_w = offset_bottom_rois[2];
      T roi_end_h = offset_bottom_rois[3];

      // Force malformed ROIs to be 1x1
      T roi_width = max(roi_end_w - roi_start_w, (T) 1.);
      T roi_height = max(roi_end_h - roi_start_h, (T) 1.);
      T bin_size_h = static_cast<T>(roi_height) / static_cast<T>(pooled_height);
      T bin_size_w = static_cast<T>(roi_width) / static_cast<T>(pooled_width);

      // We use roi_bin_grid to sample the grid and mimic integral
      int roi_bin_grid_h = (sampling_ratio > 0)
                           ? sampling_ratio
                           : ceil(roi_height / pooled_height);  // e.g., = 2
      int roi_bin_grid_w =
          (sampling_ratio > 0) ? sampling_ratio : ceil(roi_width / pooled_width);

      T rval = offset_roi_features[0];
      for (int iy = 0; iy < roi_bin_grid_h; iy++) {
        const T yy = roi_start_h + ph * bin_size_h +
                     static_cast<T>(iy + .5f) * bin_size_h /
                     static_cast<T>(roi_bin_grid_h);  // e.g., 0.5, 1.5
        for (int ix = 0; ix < roi_bin_grid_w; ix++) {
          const T xx = roi_start_w + pw * bin_size_w +
                       static_cast<T>(ix + .5f) * bin_size_w /
                       static_cast<T>(roi_bin_grid_w);

          // from [xx, yy] point
          T w1, w2, w3, w4;
          int x_low, x_high, y_low, y_high;

          inverse_bilinear_interpolate(
              height, width,
              yy, xx, &w1, &w2, &w3, &w4,
              &x_low, &x_high, &y_low, &y_high
          );

          T r1 = rval * w1;
          T r2 = rval * w2;
          T r3 = rval * w3;
          T r4 = rval * w4;
//                  VLOG(x) << xx << " " << yy  << " " << x_low  << " " << y_low  << " " << x_high  << " " << y_high ;

          if (x_low >= 0 && x_high >= 0 && y_low >= 0 && y_high >= 0) {
            // atomic add is not needed for 怿now since it is single threaded
            atomicAdd(offset_top_data + y_low * width + x_low, static_cast<T>(r1));
            atomicAdd(offset_top_data + y_low * width + x_high, static_cast<T>(r2));
            atomicAdd(offset_top_data + y_high * width + x_low, static_cast<T>(r3));
            atomicAdd(offset_top_data + y_high * width + x_high, static_cast<T>(r4));
          }  // if
        }
      }
    }
  }


  template<typename T>
  void RoIUpsampleForward(
      mshadow::Stream<gpu> *s,
      const T *roi_features,
      const T *bottom_rois,
      const int nrois,
      const int batch_size,
      const int channels,
      const int height,
      const int width,
      const int pooled_height,
      const int pooled_width,
      const int sampling_ratio,
      int rois_cols,
      T *top_data
  ) {
    int nthreads = nrois * channels * pooled_height * pooled_width;
    RoIUpsampleForwardKernel<T> <<< ROI_GET_BLOCKS(nthreads), kMaxThreadsPerBlock >>> (
      nthreads,
      roi_features,
      bottom_rois,
      batch_size,
      channels,
      height,
      width,
      pooled_height,
      pooled_width,
      sampling_ratio,
      rois_cols,
      top_data
    );
    return;
  }


  template<typename T>
  __device__ T inverse_bilinear_interpolate_gradient(
      const T *bottom_data,
      const int height,
      const int width,
      T y,
      T x) {
    // deal with cases that inverse elements are out of feature map boundary
    if (y < -1.0 || y > height || x < -1.0 || x > width) {
      // empty
      return 0;
    }

    if (y <= 0) {
      y = 0;
    }
    if (x <= 0) {
      x = 0;
    }

    int y_low = static_cast<int>(y);
    int x_low = static_cast<int>(x);
    int y_high;
    int x_high;

    if (y_low >= height - 1) {
      y_high = y_low = height - 1;
      y = (T) y_low;
    } else {
      y_high = y_low + 1;
    }

    if (x_low >= width - 1) {
      x_high = x_low = width - 1;
      x = (T) x_low;
    } else {
      x_high = x_low + 1;
    }

    T ly = y - y_low;
    T lx = x - x_low;
    T hy = 1. - ly, hx = 1. - lx;


    // do bilinear interpolation
    T v1 = bottom_data[y_low * width + x_low];
    T v2 = bottom_data[y_low * width + x_high];
    T v3 = bottom_data[y_high * width + x_low];
    T v4 = bottom_data[y_high * width + x_high];
    T w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

    T val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);

    // normalize as introduced in paper
    T value_x = hx * hx + lx * lx;
    T value_y = hy * hy + ly * ly;

    val = val / (value_x * value_y);
    return val;
  }


  template <typename T>
  __global__ void RoIUpsampleBackwardKernel(
      const int nthreads,
      const T *top_diff, // [batch_size, c, h, w]
      const T *bottom_rois,
      const int batch_size,
      const int channels,
      const int height,
      const int width,
      const int pooled_height,
      const int pooled_width,
      const int sampling_ratio,
      int rois_cols,
      T *bottom_diff //[nrois, c, pooled_height, pooled_width]
  ) {
    CUDA_KERNEL_LOOP(index, nthreads) {
      // (n, c, ph, pw) is an element in the pooled output
      int pw = index % pooled_width;
      int ph = (index / pooled_width) % pooled_height;
      int c = (index / pooled_width / pooled_height) % channels;
      int n = index / pooled_width / pooled_height / channels;

      int roi_batch_ind = 0;
      const T *offset_bottom_rois = bottom_rois + n * rois_cols;
      if (rois_cols == 5) {
        roi_batch_ind = static_cast<int>(offset_bottom_rois[0]);
        offset_bottom_rois++;
      }

      if (roi_batch_ind >= batch_size) {
        continue;
      }
      const T *offset_top_diff = top_diff + (roi_batch_ind * channels + c) * height * width;

      T roi_start_w = offset_bottom_rois[0];
      T roi_start_h = offset_bottom_rois[1];
      T roi_end_w = offset_bottom_rois[2];
      T roi_end_h = offset_bottom_rois[3];

      // Force malformed ROIs to be 1x1
      T roi_width = max(roi_end_w - roi_start_w, (T) 1.);
      T roi_height = max(roi_end_h - roi_start_h, (T) 1.);
      T bin_size_h = static_cast<T>(roi_height) / static_cast<T>(pooled_height);
      T bin_size_w = static_cast<T>(roi_width) / static_cast<T>(pooled_width);

      // We use roi_bin_grid to sample the grid and mimic integral
      int roi_bin_grid_h = (sampling_ratio > 0)
                           ? sampling_ratio
                           : ceil(roi_height / pooled_height);  // e.g., = 2
      int roi_bin_grid_w =
          (sampling_ratio > 0) ? sampling_ratio : ceil(roi_width / pooled_width);

      // We do average (integral) pooling inside a bin
      const T count = roi_bin_grid_h * roi_bin_grid_w;  // e.g. = 4

      T output_val = 0;
      for (int iy = 0; iy < roi_bin_grid_h; iy++) {
        const T yy = roi_start_h + ph * bin_size_h +
                     static_cast<T>(iy + .5f) * bin_size_h /
                     static_cast<T>(roi_bin_grid_h);  // e.g., 0.5, 1.5
        for (int ix = 0; ix < roi_bin_grid_w; ix++) {
          const T xx = roi_start_w + pw * bin_size_w +
                       static_cast<T>(ix + .5f) * bin_size_w /
                       static_cast<T>(roi_bin_grid_w);

          T val = inverse_bilinear_interpolate_gradient(
              offset_top_diff, height, width, yy, xx);
          output_val += val;
        }  // if
      }
      output_val /= count;
      bottom_diff[index] = output_val;
    }
  }


  template<typename T>
  void RoIUpsampleBackward(
      mshadow::Stream <gpu> *s,
      const T *top_diff, // [batch_size, c, h, w]
      const T *bottom_rois,
      const int nrois,
      const int batch_size,
      const int channels,
      const int height,
      const int width,
      const int pooled_height,
      const int pooled_width,
      const int sampling_ratio,
      int rois_cols,
      T *bottom_diff //[nrois, c, pooled_height, pooled_width]
  ) {
    int nthreads = nrois * channels * pooled_height * pooled_width;
    RoIUpsampleBackwardKernel<T> <<< ROI_GET_BLOCKS(nthreads), kMaxThreadsPerBlock >>> (
      nthreads,
      top_diff,
      bottom_rois,
      batch_size,
      channels,
      height,
      width,
      pooled_height,
      pooled_width,
      sampling_ratio,
      rois_cols,
      bottom_diff
    );
    return;
  }
} //op
} //mxnet


#endif