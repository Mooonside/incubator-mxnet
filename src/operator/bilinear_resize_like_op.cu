#ifndef MXNET_OPERATOR_BILINEAR_RESIZE_LIKE_CU
#define MXNET_OPERATOR_BILINEAR_RESIZE_LIKE_CU

#include "bilinear_resize_like_op-inl.h"


namespace mxnet {
  namespace op {
    template<>
    Operator* CreateOp<gpu>(BilinearResizeLikeParam param, int dtype,
                            std::vector<TShape> *in_shape,
                            std::vector<TShape> *out_shape,
                            Context ctx) {
      Operator *op = NULL;
      MSHADOW_REAL_TYPE_SWITCH_EX(dtype, DType, AccReal, {
        op = new BilinearResizeLikeOp<gpu, DType, AccReal>(param);
      })
      return op;
    }
  }
} //namespace mxnet

#endif