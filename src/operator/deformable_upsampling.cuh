#ifndef MXNET_OPERATOR_DEFORMABLE_UPSAMPLING_CUH_
#define MXNET_OPERATOR_DEFORMABLE_UPSAMPLING_CUH_

#include <mxnet/base.h>
#include <mxnet/operator.h>
#include <curand.h>
#include <curand_kernel.h>
#include "mxnet_op.h"


namespace mxnet{
  namespace op {
    template<typename T>
    __device__ T bilinear_interpolate(
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

      return val;
    }


    template <typename DType>
    __global__ void deformable_upsampling_forward_kernel(
        const DType *data,
        const DType *offset, // [grps * 2, H, W]
        const int nthreads,
        const int height,
        const int width,
        const int channels,
        const int resize_h,
        const int resize_w,
        const int channel_per_deformable_group,
        DType* outputs
    ){
      CUDA_KERNEL_LOOP(idx, nthreads){
        const int output_w = idx % resize_w;
        const int output_h = (idx / resize_w) % resize_h;
        const int output_c = idx / resize_w / resize_h;
        const int deformable_group_index = output_c / channel_per_deformable_group;

        const DType scale_height = resize_h > 1 ?
                                   static_cast<DType>(height - 1) / (resize_h - 1) : static_cast<DType>(0);
        const DType scale_width = resize_w > 1 ?
                                  static_cast<DType>(width - 1) / (resize_w - 1) : static_cast<DType>(0);

        const DType *offset_h_ptr = offset + deformable_group_index * 2 * resize_h * resize_w;
        const DType *offset_w_ptr = offset + (deformable_group_index * 2 + 1) * resize_h * resize_w;
        const DType* data_ptr = data + output_c * height * width;

        const DType im_h = output_h * scale_height;
        const DType im_w = output_w * scale_width;
        const DType offset_h = offset_h_ptr[output_h * resize_w + output_w];
        const DType offset_w = offset_w_ptr[output_h * resize_w + output_w];

        const DType im_y = im_h + offset_h;
        const DType im_x = im_w + offset_w;
        DType val = static_cast<DType>(0);
        if (im_y > -1 && im_x > -1 && im_y < height && im_x < width) {
          val = bilinear_interpolate<DType>(
              data_ptr,
              height,
              width,
              im_y,
              im_x
          );
        }
        outputs[idx] = val;
      }
    }

    template <typename DType>
    void deformable_upsampling_forward(
        mshadow::Stream<gpu> *s,
        const DType *data,
        const DType *offset, // [grps * 2, H, W]
        const int height,
        const int width,
        const int channels,
        const int resize_h,
        const int resize_w,
        const int channel_per_deformable_group,
        DType* outputs) {
      int num_threads = channels * resize_h * resize_w;
      using namespace mxnet_op;
      deformable_upsampling_forward_kernel<DType><<<cuda_get_num_blocks(num_threads), mshadow::cuda::kBaseThreadNum>>>(
        data,
        offset,
        num_threads,
        height, width,
        channels,
        resize_h, resize_w,
        channel_per_deformable_group,
        outputs
      );
    }



    /*/
     * Backward
     */
    template <typename T>
    __device__ void bilinear_interpolate_gradient(
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
        int* y_high)
    {
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

      *w1 = hy * hx, *w2 = hy * lx, *w3 = ly * hx, *w4 = ly * lx;

      return;
    }




    template <typename DType>
    __device__ DType get_coordinate_weight(DType argmax_h, DType argmax_w,
                                    const int height, const int width,
                                    const DType* im_data,
                                    const int bp_dir) {
      // modify here to accept (-1, 0] offsets
      DType h_tolerance = bp_dir == 0 ? 0 : -1;
      DType w_tolerance = bp_dir == 1 ? 0 : -1;

      if (argmax_h < h_tolerance || argmax_h > height -1 - h_tolerance ||
          argmax_w < w_tolerance || argmax_w > width - 1 - w_tolerance) {
        //empty
        return 0;
      }


      if (argmax_h < 0) argmax_h = 0;
      if (argmax_w < 0) argmax_w = 0;

      int argmax_h_low = (int)argmax_h;
      int argmax_w_low = (int)argmax_w;
      int argmax_h_high;
      int argmax_w_high;
      if (argmax_h_low >= height - 1) {
        argmax_h_high = argmax_h_low = height - 1;
        argmax_h = (DType)argmax_h_low;
      } else {
        argmax_h_high = argmax_h_low + 1;
      }
      if (argmax_w_low >= width - 1) {
        argmax_w_high = argmax_w_low = width - 1;
        argmax_w = (DType)argmax_w_low;
      } else {
        argmax_w_high = argmax_w_low + 1;
      }
      DType weight = 0;

      if (bp_dir == 0) {
        weight += -1 * (argmax_w_low + 1 - argmax_w) * im_data[argmax_h_low * width + argmax_w_low];
        weight += -1 * (argmax_w - argmax_w_low) * im_data[argmax_h_low * width + argmax_w_high];
        weight += (argmax_w_low + 1 - argmax_w) * im_data[argmax_h_high * width + argmax_w_low];
        weight += (argmax_w - argmax_w_low) * im_data[argmax_h_high * width + argmax_w_high];
      } else if (bp_dir == 1) {
        weight += -1 * (argmax_h_low + 1 - argmax_h) * im_data[argmax_h_low * width + argmax_w_low];
        weight += (argmax_h_low + 1 - argmax_h) * im_data[argmax_h_low * width + argmax_w_high];
        weight += -1 * (argmax_h - argmax_h_low) * im_data[argmax_h_high * width + argmax_w_low];
        weight += (argmax_h - argmax_h_low) * im_data[argmax_h_high * width + argmax_w_high];
      }

      return weight;
    }



    template <typename DType>
    __global__ void deformable_upsampling_backward_data_kernel(
        int num_threads,
        const DType *top_diff,
        const DType *offset,
        const int height,
        const int width,
        const int resize_h,
        const int resize_w,
        const int channel_per_deformable_group,
        DType *data_diff)
    {
      CUDA_KERNEL_LOOP(idx, num_threads){
        const int output_w = idx % resize_w;
        const int output_h = (idx / resize_w) % resize_h;
        const int output_c = idx / resize_w / resize_h;
        const int deformable_group_index = output_c / channel_per_deformable_group;

        const DType scale_height = resize_h > 1 ?
                                   static_cast<DType>(height - 1) / (resize_h - 1) : static_cast<DType>(0);
        const DType scale_width = resize_w > 1 ?
                                  static_cast<DType>(width - 1) / (resize_w - 1) : static_cast<DType>(0);

        const DType* offset_h_ptr = offset + deformable_group_index * 2 * resize_h * resize_w;
        const DType* offset_w_ptr = offset + (deformable_group_index * 2 + 1) * resize_h * resize_w;
        DType *data_diff_ptr = data_diff + output_c * height * width;

        const DType im_h = output_h * scale_height;
        const DType im_w = output_w * scale_width;

        const DType offset_h = offset_h_ptr[output_h * resize_w + output_w];
        const DType offset_w = offset_w_ptr[output_h * resize_w + output_w];

        const DType this_diff = top_diff[idx];
        const DType im_y = im_h + offset_h;
        const DType im_x = im_w + offset_w;

        DType w1, w2, w3, w4;
        int x_low, x_high, y_low, y_high;

        bilinear_interpolate_gradient<DType>(
            height,
            width,
            im_y, im_x,
            &w1, &w2, &w3, &w4,
            &x_low, &x_high, &y_low, &y_high
        );

        DType g1 = this_diff * w1;
        DType g2 = this_diff * w2;
        DType g3 = this_diff * w3;
        DType g4 = this_diff * w4;

        // update data diff
        if (x_low >= 0 && x_high >= 0 && y_low >= 0 && y_high >= 0) {
          // atomic add is not needed for now since it is single threaded
          atomicAdd(data_diff_ptr + y_low * width + x_low, g1);
          atomicAdd(data_diff_ptr + y_low * width + x_high, g2);
          atomicAdd(data_diff_ptr + y_high * height + x_low, g3);
          atomicAdd(data_diff_ptr + y_high * height + x_high, g4);
        }
      }
    };

    template <typename DType>
    __global__ void deformable_upsampling_backward_offset_kernel(
        int num_threads,
        const DType *top_diff,
        const DType *data,
        const DType *offset,
        const int height,
        const int width,
        const int resize_h,
        const int resize_w,
        const int channel_per_deformable_group,
        DType *offset_diff)
    {
      CUDA_KERNEL_LOOP(idx, num_threads){
        const int output_w = idx % resize_w;
        const int output_h = (idx / resize_w) % resize_h;
        const int offset_c = idx / resize_w / resize_h;

        const int deformable_group_index = offset_c / 2;
        const DType scale_height = resize_h > 1 ?
                                   static_cast<DType>(height - 1) / (resize_h - 1) : static_cast<DType>(0);
        const DType scale_width = resize_w > 1 ?
                                  static_cast<DType>(width - 1) / (resize_w - 1) : static_cast<DType>(0);

        const DType* offset_h_ptr = offset + deformable_group_index * 2 * resize_h * resize_w;
        const DType* offset_w_ptr = offset + (deformable_group_index * 2 + 1) * resize_h * resize_w;

        const DType im_h = output_h * scale_height;
        const DType im_w = output_w * scale_width;

        const DType offset_h = offset_h_ptr[output_h * resize_w + output_w];
        const DType offset_w = offset_w_ptr[output_h * resize_w + output_w];
        const DType im_y = im_h + offset_h;
        const DType im_x = im_w + offset_w;

        // this offset is shared among this deformable_group
        const int cstart = deformable_group_index * channel_per_deformable_group;
        const int cend = cstart + channel_per_deformable_group;
        const int bp_dir = offset_c % 2; // if ==0, then h offset

        const DType *offset_top_diff = top_diff + (cstart * resize_h + output_h) * resize_w + output_w;
        const DType *offset_data = data + cstart * height * width;

        for (int in_c = cstart; in_c < cend; ++in_c){
          DType this_diff = offset_top_diff[0];
          DType weight = get_coordinate_weight(im_y, im_x, height, width, offset_data, bp_dir);
          atomicAdd(offset_diff + idx, weight * this_diff);
          offset_top_diff += resize_h * resize_w;
          offset_data += height * width;
        }
      }
    }


    template <typename DType>
    void deformable_upsampling_backward_data(
        mshadow::Stream<gpu> *s,
        const DType *top_diff,
        const DType *offset,
        const int height,
        const int width,
        const int channels,
        const int resize_h,
        const int resize_w,
        const int channel_per_deformable_group,
        DType *data_diff

    ){
      int num_threads = channels * resize_h * resize_w;
      using namespace mxnet_op;
      deformable_upsampling_backward_data_kernel<DType> <<<cuda_get_num_blocks(num_threads), mshadow::cuda::kBaseThreadNum>>>(
        num_threads,
        top_diff,
        offset,
        height, width,
        resize_h, resize_w,
        channel_per_deformable_group,
        data_diff
      );
    }

    template <typename DType>
    void deformable_upsampling_backward_offset(
        mshadow::Stream<gpu> *s,
        const DType *top_diff,
        const DType *data,
        const DType *offset,
        const int height,
        const int width,
        const int resize_h,
        const int resize_w,
        const int deformable_group,
        const int channel_per_deformable_group,
        DType *offset_diff
    ){
      int num_threads = deformable_group * 2 * resize_h * resize_w;
      using namespace mxnet_op;
      deformable_upsampling_backward_offset_kernel<DType> <<<cuda_get_num_blocks(num_threads), mshadow::cuda::kBaseThreadNum>>>(
          num_threads,
          top_diff,
          data,
          offset,
          height,
          width,
          resize_h,
          resize_w,
          channel_per_deformable_group,
          offset_diff
      );
    }
  }
}




#endif