#ifndef MXNET_OPERATOR_SUPER_PIXEL_AVG_POOLING_H_
#define MXNET_OPERATOR_SUPER_PIXEL_AVG_POOLING_H_
namespace mxnet {
  namespace op {
    template <class T>
    inline void add(T* address, const T& val) {
      *address += val;
    }

    template <class T>
    void SuperPixelAvgPoolingForward(
        mshadow::Stream<cpu> *s,
        const T *data,
        const T *label,
        const int channels,
        const int height,
        const int width,
        const int max_spixels,
        T *buffer,
        T *out
    ){
      int nthreads =  channels * height * width;
      // clean buffer, since it may be used again in the next batch
      for (int index=0; index<max_spixels * (channels+1); ++index){
        buffer[index] = 0;
      }
      for (int index=0; index<nthreads; ++index){
        out[index] = 0;
      }

      //add and count
      for (int index=0; index<nthreads; ++index){
        int w = index % width;
        int h = (index / width) % height;
        int c = (index / width / height) % channels;

        int sp_idx = static_cast<int>(label[h * width + w]);
        // check validness
        if (sp_idx < 0 or sp_idx >=max_spixels) continue;

        // add sp_idx item in buffer's c channel, and add 1 to channels + 1 , which stores count
        const int write_offset = sp_idx * (channels + 1);

        add(buffer + write_offset + c, data[index]);
        if (c == 0) //only add once
          add(buffer + write_offset + channels, static_cast<T>(1));
      }
//      VLOG(x) << buffer[channels];

      //calculate denormed feature
      for (int index=0; index<max_spixels; ++index){
        int offset = index * (channels+1);
        T count = buffer[offset + channels];
        if (count <= 1) continue;

        for (int c = 0; c < channels; c++){
          buffer[offset + c] /= count;
        }
      }

      for (int index=0; index<nthreads; ++index){
        int w = index % width;
        int h = (index / width) % height;
        int c = (index / width / height) % channels;

        const T *offset_label = label +  h * width + w;
        int sp_idx = static_cast<int>(offset_label[0]);
        // check validness
        if (sp_idx < 0 or sp_idx >=max_spixels) continue;

        // add sp_idx item in buffer's c channel, and add 1 to channels + 1 , which stores count
        out[index] = buffer[sp_idx*(channels + 1) + c];
      }
    }


    template <class T>
    void SuperPixelAvgPoolingBackward(
        mshadow::Stream<cpu> *s,
        const T *top_diff,
        const T *label,
        const int channels,
        const int height,
        const int width,
        const int max_spixels,
        T *buffer,
        T *bottom_diff
    ){
      int nthreads =  channels * height * width;
      // clean buffer, since it may be used again in the next batch
      for (int index=0; index<max_spixels * (channels+1); ++index){
        buffer[index] = 0;
      }
      // already set to zero outside...
//      for (int index=0; index<nthreads; ++index){
//        bottom_diff[index] = 0;
//      }

      //add and count
      for (int index=0; index<nthreads; ++index){
        int w = index % width;
        int h = (index / width) % height;
        int c = (index / width / height) % channels;

        int sp_idx = static_cast<int>(label[h * width + w]);
        // check validness
        if (sp_idx < 0 or sp_idx >=max_spixels) continue;

        // add sp_idx item in buffer's c channel, and add 1 to channels + 1 , which stores count
        const int write_offset = sp_idx * (channels + 1);

        add(buffer + write_offset + c, top_diff[index]);
        if (c == 0) //only add once
          add(buffer + write_offset + channels, static_cast<T>(1));
      }

      //calculate denormed feature
      for (int index=0; index<max_spixels; ++index){
        int offset = index * (channels+1);
        T count = buffer[offset + channels];
        if (count < 1) continue;

        for (int c = 0; c < channels; c++){
          buffer[offset + c] /= count;
        }
      }

      for (int index=0; index<nthreads; ++index){
        int w = index % width;
        int h = (index / width) % height;
        int c = (index / width / height) % channels;

        const T *offset_label = label +  h * width + w;
        int sp_idx = static_cast<int>(offset_label[0]);
        // check validness
        if (sp_idx < 0 or sp_idx >=max_spixels) continue;

        // add sp_idx item in buffer's c channel, and add 1 to channels + 1 , which stores count
        bottom_diff[index] = buffer[sp_idx*(channels + 1) + c];
      }
    }
  }
}
#ifdef __CUDACC__
#include "super_pixel_avg_pooling.cuh"
#endif
#endif