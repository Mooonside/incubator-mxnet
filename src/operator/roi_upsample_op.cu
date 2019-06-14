#ifndef MXNET_OPERATOR_ROI_UPSAMPLE_CU
#define MXNET_OPERATOR_ROI_UPSAMPLE_CU

#include "roi_upsample.cuh"
#include "roi_upsample_op-inl.h"


namespace mxnet {
  namespace op {
    template<>
      Operator* CreateOp<gpu>(RoIUpsampleParam param, int dtype,
                            std::vector<TShape> *in_shape,
                            std::vector<TShape> *out_shape,
                            Context ctx) {
      Operator *op = NULL;
      MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
        op = new RoIUpsampleOp<gpu, DType>(param);
      })
      return op;
    }
  }
} //namespace mxnet

#endif