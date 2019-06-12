#ifndef MXNET_OPERATOR_SUPER_PIXEL_AVG_POOLING_CUH_
#define MXNET_OPERATOR_SUPER_PIXEL_AVG_POOLING_CUH_

#include "mxnet_op.h"
#include <mxnet/base.h>
#include <mxnet/operator.h>
#include "cuda.h"


namespace mxnet {
  namespace op {
    template<typename T>
    __global__ void count_kernel(
        const int nthreads,
        const T *data,
        const T *label,
        const int channels,
        const int height,
        const int width,
        const int max_spixels,
        T *buffer){
      CUDA_KERNEL_LOOP(index, nthreads) {
        int w = index % width;
        int h = (index / width) % height;
        int c = (index / width / height) % channels;

        auto sp_idx = static_cast<int>(label[h * width + w]);
        // check validness
        if (sp_idx < 0 or sp_idx >=max_spixels) continue;

        // add sp_idx item in buffer's c channel, and add 1 to channels + 1 , which stores count
        const int write_offset = sp_idx * (channels + 1);
        atomicAdd(buffer + write_offset + c, data[index]);
        if (c == 0) //only add once
          atomicAdd(buffer + write_offset + channels, static_cast<T>(1));
      }
    }

    template<typename T>
    __global__ void broadcast_kernel(
        const int nthreads,
        const T *label,
        const int channels,
        const int height,
        const int width,
        const int max_spixels,
        T *buffer,
        T *out
    ){
      CUDA_KERNEL_LOOP(index, nthreads) {
        int w = index % width;
        int h = (index / width) % height;
        int c = (index / width / height) % channels;
        const T *offset_label = label + h * width + w;
        auto sp_idx = static_cast<int>(offset_label[0]);
        // check validness
        if (sp_idx < 0 or sp_idx >=max_spixels) continue;
        T count = buffer[sp_idx*(channels + 1) + channels];
        if (count <  1) continue;

        // add sp_idx item in buffer's c channel, and add 1 to channels + 1 , which stores count
        out[index] = buffer[sp_idx*(channels + 1) + c] / count;
      }
    }

    template<typename T>
    void SuperPixelAvgPoolingForward(
        mshadow::Stream<gpu> *s,
        const T *data,
        const T *label,
        const int channels,
        const int height,
        const int width,
        const int max_spixels,
        T *buffer,
        T *out) {
      using namespace mxnet_op;
      // clean buffer, since it may be used again in the next batch
      cudaMemset(buffer, static_cast<T>(0), max_spixels * (channels+1) * sizeof(T));
      cudaMemset(out, static_cast<T>(0), channels * height * width * sizeof(T));

      int nthreads = channels * height * width;
      count_kernel<T><<<cuda_get_num_blocks(nthreads), mshadow::cuda::kBaseThreadNum>>>(
          nthreads,
          data,
          label,
          channels,
          height,
          width,
          max_spixels,
          buffer
      );

      broadcast_kernel<T><<<cuda_get_num_blocks(nthreads), mshadow::cuda::kBaseThreadNum>>>(
          nthreads,
          label,
          channels,
          height,
          width,
          max_spixels,
          buffer,
          out
      );
    }


    template <class T>
    void SuperPixelAvgPoolingBackward(
        mshadow::Stream<gpu> *s,
        const T *top_diff,
        const T *label,
        const int channels,
        const int height,
        const int width,
        const int max_spixels,
        T *buffer,
        T *bottom_diff
    ){
      using namespace mxnet_op;
      cudaMemset(buffer, static_cast<T>(0), max_spixels * (channels+1) * sizeof(T));
      int nthreads = channels * height * width;
      count_kernel<T><<<cuda_get_num_blocks(nthreads), mshadow::cuda::kBaseThreadNum>>>(
          nthreads,
          top_diff,
          label,
          channels,
          height,
          width,
          max_spixels,
          buffer
      );

      broadcast_kernel<T><<<cuda_get_num_blocks(nthreads), mshadow::cuda::kBaseThreadNum>>>(
          nthreads,
          label,
          channels,
          height,
          width,
          max_spixels,
          buffer,
          bottom_diff
      );
    }
  } // op
} //mxnet
#endif