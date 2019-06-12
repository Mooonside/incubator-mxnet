#include "bilinear_resize_like_op-inl.h"

namespace mxnet{
namespace op{
DMLC_REGISTER_PARAMETER(BilinearResizeLikeParam);

template<>
Operator* CreateOp<cpu>(BilinearResizeLikeParam param, int dtype,
    std::vector<TShape> *in_shape,
std::vector<TShape> *out_shape,
    Context ctx) {
  Operator *op = nullptr;
    MSHADOW_REAL_TYPE_SWITCH_EX(dtype, DType, AccReal, {
    op = new BilinearResizeLikeOp<cpu, DType, AccReal>(param);
  })
  return op;
}

// DO_BIND_DISPATCH comes from operator_common.h
Operator *BilinearResizeLikeProp::CreateOperatorEx(Context ctx,
                                     std::vector<TShape> *in_shape,
                                     std::vector<int> *in_type) const {
  std::vector<TShape> out_shape, aux_shape;
  std::vector<int> out_type, aux_type;
  CHECK(InferType(in_type, &out_type, &aux_type));
  CHECK(InferShape(in_shape, &out_shape, &aux_shape));
  DO_BIND_DISPATCH(CreateOp, param_, (*in_type)[0], in_shape, &out_shape, ctx);
}


MXNET_REGISTER_OP_PROPERTY(BilinearResizeLike, BilinearResizeLikeProp)
.describe(R"code(Bilinear Resize according to the spatial size of second input
)code" ADD_FILELINE)
.add_argument("data", "NDArray-or-Symbol", "Input feature, in [N, C, iH, iW]")
.add_argument("like", "NDArray-or-Symbol", "Input feature, in [N, C, oH, oW]")
.add_arguments(BilinearResizeLikeParam::__FIELDS__());

} //end op
} //end mxnet