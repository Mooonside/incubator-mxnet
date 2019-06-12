#include "super_pixel_avg_pooling_op-inl.h"

namespace mxnet{
namespace op{
DMLC_REGISTER_PARAMETER(SuperPixelAvgPoolParam);

template<>
Operator* CreateOp<cpu>(SuperPixelAvgPoolParam param, int dtype,
                        std::vector<TShape> *in_shape,
                        std::vector<TShape> *out_shape,
                        Context ctx) {
  Operator *op = nullptr;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new SuperPixelAvgPoolOp<cpu, DType>(param);
  })
  return op;
}

// DO_BIND_DISPATCH comes from operator_common.h
Operator *SuperPixelAvgPoolProp::CreateOperatorEx(Context ctx,
                                            std::vector<TShape> *in_shape,
                                            std::vector<int> *in_type) const {
  std::vector<TShape> out_shape, aux_shape;
  std::vector<int> out_type, aux_type;
  CHECK(InferType(in_type, &out_type, &aux_type));
  CHECK(InferShape(in_shape, &out_shape, &aux_shape));
  DO_BIND_DISPATCH(CreateOp, param_, (*in_type)[0], in_shape, &out_shape, ctx);
}


MXNET_REGISTER_OP_PROPERTY(SuperPixelAvgPool, SuperPixelAvgPoolProp)
    .describe(R"code(avg pool features according to superpixels and assign each pixel with super pixel feature.
)code" ADD_FILELINE)
    .add_argument("data", "NDArray-or-Symbol", "Input feature, in 4d [N, C, H, W]")
    .add_argument("label", "NDArray-or-Symbol", "Input label in 3d [N, H, W].")
    .add_arguments(SuperPixelAvgPoolParam::__FIELDS__());
}
}