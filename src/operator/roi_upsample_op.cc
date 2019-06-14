#include "roi_upsample_op-inl.h"

namespace mxnet{
  namespace op{
    DMLC_REGISTER_PARAMETER(RoIUpsampleParam);

    template<>
    Operator* CreateOp<cpu>(RoIUpsampleParam param, int dtype,
                            std::vector<TShape> *in_shape,
                            std::vector<TShape> *out_shape,
                            Context ctx) {
      Operator *op = nullptr;
      MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
        op = new RoIUpsampleOp<cpu, DType>(param);
      })
      return op;
    }

// DO_BIND_DISPATCH comes from operator_common.h
    Operator *RoIUpsampleProp::CreateOperatorEx(Context ctx,
                                                      std::vector<TShape> *in_shape,
                                                      std::vector<int> *in_type) const {
      std::vector<TShape> out_shape, aux_shape;
      std::vector<int> out_type, aux_type;
      CHECK(InferType(in_type, &out_type, &aux_type));
      CHECK(InferShape(in_shape, &out_shape, &aux_shape));
      DO_BIND_DISPATCH(CreateOp, param_, (*in_type)[0], in_shape, &out_shape, ctx);
    }


    MXNET_REGISTER_OP_PROPERTY(RoIUpsample, RoIUpsampleProp)
        .describe(R"code(RoIUpsample operation as described in http://arxiv.org/abs/1812.03904
roi_features : [N x C x pH x pW], roi_boxes : [N x {4, 5}]. If cols = 5, the first col stores its batch id.
returns : [output_batch x C x oH x oW]. Batch ids that exceeds batch_size is ignored.
)code" ADD_FILELINE)
        .add_argument("roi_features", "NDArray-or-Symbol", "Input feature, in 4d [N, C, H, W]")
        .add_argument("roi_boxes", "NDArray-or-Symbol", "Input feature, in 2d [N, {4, 5}]")
        .add_arguments(RoIUpsampleParam::__FIELDS__());
  }
}