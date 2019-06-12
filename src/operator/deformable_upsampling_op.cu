#ifndef MXNET_OPERATOR_DEFORMABLE_UPSAMPLING_CU
#define MXNET_OPERATOR_DEFORMABLE_UPSAMPLING_CU

#include "deformable_upsampling.cuh"
#include "deformable_upsampling_op-inl.h"


namespace mxnet {
  namespace op {
    template<>
    Operator* CreateOp<gpu>(DeformableUpsamplingParam param, int dtype,
                            std::vector<TShape> *in_shape,
                            std::vector<TShape> *out_shape,
                            Context ctx) {
      Operator *op = NULL;
      MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
        op = new DeformableUpsamplingOp<gpu, DType>(param);
      })
      return op;
    }
  }
} //namespace mxnet

#endif