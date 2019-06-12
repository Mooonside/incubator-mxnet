#include "deformable_upsampling_op-inl.h"

namespace mxnet{
  namespace op{
    DMLC_REGISTER_PARAMETER(DeformableUpsamplingParam);

    template<>
    Operator* CreateOp<cpu>(DeformableUpsamplingParam param, int dtype,
                            std::vector<TShape> *in_shape,
                            std::vector<TShape> *out_shape,
                            Context ctx) {
      Operator *op = nullptr;
      MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
        op = new DeformableUpsamplingOp<cpu, DType>(param);
      })
      return op;
    }

// DO_BIND_DISPATCH comes from operator_common.h
    Operator *DeformableUpsamplingProp::CreateOperatorEx(Context ctx,
                                                      std::vector<TShape> *in_shape,
                                                      std::vector<int> *in_type) const {
      std::vector<TShape> out_shape, aux_shape;
      std::vector<int> out_type, aux_type;
      CHECK(InferType(in_type, &out_type, &aux_type));
      CHECK(InferShape(in_shape, &out_shape, &aux_shape));
      DO_BIND_DISPATCH(CreateOp, param_, (*in_type)[0], in_shape, &out_shape, ctx);
    }


    MXNET_REGISTER_OP_PROPERTY(DeformableUpsampling, DeformableUpsamplingProp)
        .describe(R"code(dweformable upsampling
)code" ADD_FILELINE)
        .add_argument("data", "NDArray-or-Symbol", "Input feature, in 4d [N, C1, H, W]")
        .add_argument("offset", "NDArray-or-Symbol", "Input offset in 4d [N, C2, RH, RW].")
        .add_arguments(DeformableUpsamplingParam::__FIELDS__());
  }
}