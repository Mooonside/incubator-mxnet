#ifndef MXNET_OPERATOR_ROI_UPSAMPLE_H_
#define MXNET_OPERATOR_ROI_UPSAMPLE_H_

#include <vector>
#include <sstream>


namespace mxnet {
  namespace op {
    template <typename T>
    void inverse_bilinear_interpolate_cpu(
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

    template <class T>
    inline void add(const T& val, T* address) {
      *address += val;
    }

    template<typename T>
    void RoIUpsampleForward(
      mshadow::Stream <cpu> *s,
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
      const T *offset_bottom_rois;
      const T *offset_roi_features;
      T *offset_top_data;

      for (int n = 0; n < nrois; ++n) {
        int roi_batch_ind = 0;
        offset_bottom_rois = bottom_rois + n * rois_cols;
        offset_roi_features = roi_features + n * channels * pooled_height * pooled_width;

        if (rois_cols == 5) {
          roi_batch_ind = static_cast<int>(offset_bottom_rois[0]);
          offset_bottom_rois++;
        }

        if (roi_batch_ind >= batch_size)
          continue;

        offset_top_data =  top_data + roi_batch_ind * channels * height * width;
        int index_n = n * channels * pooled_width * pooled_height;

        // Do not using rounding; this implementation detail is critical
        T roi_start_w = offset_bottom_rois[0];
        T roi_start_h = offset_bottom_rois[1];
        T roi_end_w = offset_bottom_rois[2];
        T roi_end_h = offset_bottom_rois[3];
//        VLOG(x) << roi_start_w << ", " << roi_start_h << ", "<< roi_end_w << ", " << roi_end_h ;
        // Force malformed ROIs to be 1x1
        T roi_width = std::max(roi_end_w - roi_start_w, (T)1.);
        T roi_height = std::max(roi_end_h - roi_start_h, (T)1.);
        T bin_size_h = static_cast<T>(roi_height) / static_cast<T>(pooled_height);
        T bin_size_w = static_cast<T>(roi_width) / static_cast<T>(pooled_width);

        // We use roi_bin_grid to sample the grid and mimic integral
        int roi_bin_grid_h = (sampling_ratio > 0)
                             ? sampling_ratio
                             : std::ceil(roi_height / pooled_height);  // e.g., = 2
        int roi_bin_grid_w =
            (sampling_ratio > 0) ? sampling_ratio : std::ceil(roi_width / pooled_width);

        // We do average (integral) pooling inside a bin
//        const T count = roi_bin_grid_h * roi_bin_grid_w;  // e.g. = 4
        for (int c = 0; c < channels; c++) {
          int index_n_c = index_n + c * pooled_height * pooled_width;
          int top_index_c = c * height * width;

          for (int ph = 0; ph < pooled_height; ph++) {
            for (int pw = 0; pw < pooled_width; pw++) {
              T rval = offset_roi_features[index_n_c + ph * pooled_width + pw];
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

                  inverse_bilinear_interpolate_cpu(
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
                    add(static_cast<T>(r1), offset_top_data + top_index_c + y_low * width + x_low);
                    add(static_cast<T>(r2), offset_top_data + top_index_c + y_low * width + x_high);
                    add(static_cast<T>(r3), offset_top_data + top_index_c + y_high * width + x_low);
                    add(static_cast<T>(r4), offset_top_data + top_index_c + y_high * width + x_high);
                  }  // if
                }
              }
            }
          }
        }
      }
    }

    template<typename T>
    T inverse_bilinear_interpolate_gradient_cpu(
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


    template<typename T>
    void RoIUpsampleBackward(
        mshadow::Stream <cpu> *s,
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
    ){
      const T *offset_bottom_rois;
      const T *offset_top_diff;

      for (int n = 0; n < nrois; ++n) {
        int roi_batch_ind = 0;
        offset_bottom_rois = bottom_rois + n * rois_cols;

        if (rois_cols == 5) {
          roi_batch_ind = static_cast<int>(offset_bottom_rois[0]);
          offset_bottom_rois++;
        }

        if (roi_batch_ind >= batch_size)
          continue;

        offset_top_diff = top_diff + roi_batch_ind * channels * height * width;
        int index_n = n * channels * pooled_width * pooled_height;

        // Do not using rounding; this implementation detail is critical
        T roi_start_w = offset_bottom_rois[0];
        T roi_start_h = offset_bottom_rois[1];
        T roi_end_w = offset_bottom_rois[2];
        T roi_end_h = offset_bottom_rois[3];
        // Force malformed ROIs to be 1x1
        T roi_width = std::max(roi_end_w - roi_start_w, (T) 1.);
        T roi_height = std::max(roi_end_h - roi_start_h, (T) 1.);
        T bin_size_h = static_cast<T>(roi_height) / static_cast<T>(pooled_height);
        T bin_size_w = static_cast<T>(roi_width) / static_cast<T>(pooled_width);

        // We use roi_bin_grid to sample the grid and mimic integral
        int roi_bin_grid_h = (sampling_ratio > 0)
                             ? sampling_ratio
                             : std::ceil(roi_height / pooled_height);  // e.g., = 2
        int roi_bin_grid_w =
            (sampling_ratio > 0) ? sampling_ratio : std::ceil(roi_width / pooled_width);

        // We do average (integral) pooling inside a bin
        const T count = roi_bin_grid_h * roi_bin_grid_w;  // e.g. = 4

        for (int c = 0; c < channels; c++) {
          int index_n_c = index_n + c * pooled_height * pooled_width;
          int top_index_c = c * height * width;

          for (int ph = 0; ph < pooled_height; ph++) {
            for (int pw = 0; pw < pooled_width; pw++) {
              T output_val = 0;
              for (int iy = 0; iy < roi_bin_grid_h; iy++) {
                const T yy = roi_start_h + ph * bin_size_h +
                             static_cast<T>(iy + .5f) * bin_size_h /
                             static_cast<T>(roi_bin_grid_h);  // e.g., 0.5, 1.5
                for (int ix = 0; ix < roi_bin_grid_w; ix++) {
                  const T xx = roi_start_w + pw * bin_size_w +
                               static_cast<T>(ix + .5f) * bin_size_w /
                               static_cast<T>(roi_bin_grid_w);

                  T val = inverse_bilinear_interpolate_gradient_cpu(
                      offset_top_diff + top_index_c, height, width, yy, xx);
                  output_val += val;
                }
              }
              output_val /= count;
              bottom_diff[index_n_c + ph * pooled_width + pw] = output_val;
            }
          }
        }
      }
    }
  } // end of op
} // end of mxnet

#ifdef __CUDACC__
#include "roi_upsample.cuh"
#endif

#endif