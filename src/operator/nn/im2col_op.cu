#ifndef MXNET_OPERATOR_MY_IM2COL_INL_CU
#define MXNET_OPERATOR_MY_IM2COL_INL_CU

#include <mxnet/base.h>
#include <mxnet/operator.h>
#include <algorithm>
#include <cstring>
#include <vector>
#include "../../apache-mxnet-src-1.3.1.rc0-incubating/src/operator/mxnet_op.h"
#include "im2col.cuh"
#include "im2col_op-inl.h"


namespace mxnet {
  namespace op {
    template<>
    void Im2colCompute<gpu>(const nnvm::NodeAttrs& attrs,
                                 const OpContext& ctx,
                                 const std::vector<TBlob>& inputs,
                                 const std::vector<OpReqType>& req,
                                 const std::vector<TBlob>& outputs) {
      const Im2colParam & param = nnvm::get<Im2colParam>(attrs.parsed);
      int dtype = inputs[0].type_flag_;

      MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
        Im2colOp<gpu, DType> op;
        op.Init(param);
        op.Forward(ctx, inputs, req, outputs);
      })
    }

    template<>
    void Im2colGradCompute<gpu>(const nnvm::NodeAttrs& attrs,
                                     const OpContext& ctx,
                                     const std::vector<TBlob>& inputs,
                                     const std::vector<OpReqType>& req,
                                     const std::vector<TBlob>& outputs) {
      const Im2colParam& param = nnvm::get<Im2colParam>(attrs.parsed);
      std::vector<TBlob> in_data(inputs.begin() + 1, inputs.end());
      const TBlob &out_grad = inputs[0];
      const std::vector<TBlob> &in_grad = outputs;
      int dtype = out_grad.type_flag_;

      MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
        Im2colOp<gpu, DType> op;
        op.Init(param);
        op.Backward(ctx, std::vector<TBlob>{out_grad}, in_data, req, in_grad);
      })
    }

    NNVM_REGISTER_OP(Im2col)
    .set_attr<FCompute>("FCompute<gpu>", Im2colCompute<gpu>);

    NNVM_REGISTER_OP(_backward_Im2col)
    .set_attr<FCompute>("FCompute<gpu>", Im2colGradCompute<gpu>);
  }
}

#endif //MXNET_OPERATOR_MY_IM2COL_INL_CU