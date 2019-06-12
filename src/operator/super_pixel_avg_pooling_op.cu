#ifndef MXNET_OPERATOR_SUPER_PIXEL_AVG_POOLING_CU
#define MXNET_OPERATOR_SUPER_PIXEL_AVG_POOLING_CU

#include "super_pixel_avg_pooling.cuh"
#include "super_pixel_avg_pooling_op-inl.h"


namespace mxnet {
  namespace op {
    template<>
    Operator* CreateOp<gpu>(SuperPixelAvgPoolParam param, int dtype,
                            std::vector<TShape> *in_shape,
                            std::vector<TShape> *out_shape,
                            Context ctx) {
      Operator *op = NULL;
      MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
        op = new SuperPixelAvgPoolOp<gpu, DType>(param);
      })
      return op;
    }
  }
} //namespace mxnet

#endif