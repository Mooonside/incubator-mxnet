// register parameter in this file
#ifndef MXNET_OPERATOR_MY_IM2COL_INL_H_
#define MXNET_OPERATOR_MY_IM2COL_INL_H_

#include "../operator_common.h"
#include "im2col.h"
#include <mxnet/io.h>
#include <mxnet/base.h>
#include <mxnet/ndarray.h>
#include <mxnet/operator.h>
#include <mxnet/operator_util.h>
#include <mxnet/op_attr_types.h>
#include <dmlc/logging.h>
#include <dmlc/optional.h>
#include <algorithm>
#include <map>
#include <vector>
#include <string>
#include <utility>


namespace mxnet {
  namespace op {
    struct Im2colParam : public dmlc::Parameter<Im2colParam> {
      TShape kernel;
      TShape stride;
      TShape dilate;
      TShape pad;
      uint64_t workspace;
      dmlc::optional<int> cudnn_tune;
      bool cudnn_off;
      dmlc::optional<int> layout;

      DMLC_DECLARE_PARAMETER(Im2colParam) {
        DMLC_DECLARE_FIELD(kernel).describe("Convolution kernel size: (w,), (h, w) or (d, h, w)");
        DMLC_DECLARE_FIELD(stride).set_default(TShape())
            .describe("Convolution stride: (w,), (h, w) or (d, h, w). Defaults to 1 for each dimension.");
        DMLC_DECLARE_FIELD(dilate).set_default(TShape())
            .describe("Convolution dilate: (w,), (h, w) or (d, h, w). Defaults to 1 for each dimension.");
        DMLC_DECLARE_FIELD(pad).set_default(TShape())
            .describe("Zero pad for convolution: (w,), (h, w) or (d, h, w). Defaults to no padding.");
        DMLC_DECLARE_FIELD(workspace).set_default(1024).set_range(0, 8192)
            .describe("Maximum temporary workspace allowed (MB) in convolution."
                      "This parameter has two usages. When CUDNN is not used, it determines the "
                      "effective batch size of the convolution kernel. When CUDNN is used, it controls "
                      "the maximum temporary storage used for tuning the best CUDNN kernel when "
                      "`limited_workspace` strategy is used.");
        DMLC_DECLARE_FIELD(layout)
            .add_enum("NCHW", mshadow::kNCHW)
            .add_enum("NHWC", mshadow::kNHWC)
            .set_default(dmlc::optional<int>())
            .describe("Set layout for input. Only supported NCHW Now.");
      }
      // Adjusts kernel size for effects of dilation in the dimension `dim`.
      index_t DilatedKernelSize(int dim) const {
        return 1 + (kernel[dim] - 1) * dilate[dim];
      }

      bool operator==(const Im2colParam& other) const {
        return this->kernel == other.kernel &&
               this->stride == other.stride &&
               this->dilate == other.dilate &&
               this->pad == other.pad &&
               this->workspace == other.workspace &&
               this->cudnn_tune == other.cudnn_tune &&
               this->cudnn_off == other.cudnn_off &&
               this->layout == other.layout;
      }
    };
    void ConvolutionParamParser(nnvm::NodeAttrs* attrs);
  } // namespace op
} //namespace mxnet


namespace std {
  template<>
  struct hash<mxnet::op::Im2colParam> {
    size_t operator()(const mxnet::op::Im2colParam& val) {
      size_t ret = 0;
      ret = dmlc::HashCombine(ret, val.kernel);
      ret = dmlc::HashCombine(ret, val.stride);
      ret = dmlc::HashCombine(ret, val.dilate);
      ret = dmlc::HashCombine(ret, val.pad);
      ret = dmlc::HashCombine(ret, val.workspace);
      ret = dmlc::HashCombine(ret, val.cudnn_tune);
      ret = dmlc::HashCombine(ret, val.cudnn_off);
      ret = dmlc::HashCombine(ret, val.layout);
      return ret;
    }
  };
}  // namespace std

namespace mxnet {
  namespace op {
    template<typename xpu, typename DType>
    class Im2colOp {
    public:
      void Init(Im2colParam p) {
        this->param_ = p;
        // convert MBytes first to Bytes and then to elements.
        param_.workspace = (param_.workspace << 20) / sizeof(DType);
        CHECK(param_.layout.value() == mshadow::kNCHW) << "Only support NCHW layout";
      }

      void Forward(const OpContext &ctx,
                   const std::vector<TBlob>& in_data,
                   const std::vector<OpReqType>& req,
                   const std::vector<TBlob>& out_data) {
        using namespace mshadow;
        LayerSetUp(in_data[0].shape_, out_data[0].shape_);
        Stream<xpu>* s = ctx.get_stream<xpu>();
        // col shape to be c*k*k, h, w
        TShape col_shape = TShape(num_spatial_axes_ + 1);
        col_shape[0] = in_channels_ * param_.kernel.Size();
        for (index_t i = 1; i < col_shape.ndim(); ++i) {
          col_shape[i] = out_data[0].shape_[i+1];
        }

        for (index_t n = 0; n < num_; ++n) {
          im2col(s, in_data[0].dptr<DType>()+n*input_dim_, in_data[0].shape_,
                 col_shape, param_.kernel, param_.pad, param_.stride, param_.dilate,
                 out_data[0].dptr<DType>() + n*output_dim_);
        }
      }

      void Backward(const OpContext &ctx,
                    const std::vector<TBlob>& out_grad,
                    const std::vector<TBlob>& in_data,
                    const std::vector<OpReqType>& req,
                    const std::vector<TBlob>& in_grad) {
        using namespace mshadow;
        CHECK_EQ(out_grad.size(), 1U);
        CHECK_EQ(in_data.size(), 1);
        CHECK_EQ(in_grad.size(), 1);
        CHECK_EQ(req.size(), 1);

        LayerSetUp(in_grad[0].shape_, out_grad[0].shape_);
        Stream<xpu> *s = ctx.get_stream<xpu>();

        // calculate the shape of col_buffer
        TShape col_buffer_shape(num_spatial_axes_ + 1);
        col_buffer_shape[0] = in_channels_ * param_.kernel.Size();
        for (index_t i = 1; i < col_buffer_shape.ndim(); ++i) {
          col_buffer_shape[i] = out_grad[0].shape_[i+1];
        }

        // backward
        for (index_t n = 0; n < num_; ++n) {
          // gradient w.r.t. input data
          col2im(s, out_grad[0].dptr<DType>()+n*input_dim_, in_grad[0].shape_, col_buffer_shape,
                 param_.kernel, param_.pad, param_.stride, param_.dilate,
                 in_grad[0].dptr<DType>()+n*input_dim_, req[0]);
        }
      }

    private:
      void LayerSetUp(const TShape& ishape, const TShape& oshape) {
        channel_axis_ = 1;  // hard code channel axis
        const index_t first_spatial_axis = channel_axis_ + 1;
        const index_t num_axes = param_.kernel.ndim() + 2;
        num_spatial_axes_ = num_axes - first_spatial_axis;

        // batch size
        num_ = ishape[0];
        // number of input channels
        in_channels_ = ishape[1];
        input_dim_ = ishape.ProdShape(1, ishape.ndim());
        output_dim_ = oshape.ProdShape(1, oshape.ndim());

      }

      Im2colParam param_;
      index_t channel_axis_;  // channel axis of the input
      index_t in_channels_;  // number of channels of input image


      index_t num_spatial_axes_;  // number of spatial axes
      index_t num_;  // batch size

      index_t input_dim_;
      index_t output_dim_;
    };

    template<typename xpu>
    void Im2colCompute(const nnvm::NodeAttrs& attrs,
                       const OpContext& ctx,
                       const std::vector<TBlob>& inputs,
                       const std::vector<OpReqType>& req,
                       const std::vector<TBlob>& outputs) {
      const Im2colParam& param = nnvm::get<Im2colParam>(attrs.parsed);
      MSHADOW_REAL_TYPE_SWITCH(inputs[0].type_flag_, DType, {
        Im2colOp<xpu, DType> op;
        op.Init(param);
        op.Forward(ctx, inputs, req, outputs);
      });
    }

    template<typename xpu>
    void Im2colGradCompute(const nnvm::NodeAttrs& attrs,
                           const OpContext& ctx,
                           const std::vector<TBlob>& inputs,
                           const std::vector<OpReqType>& req,
                           const std::vector<TBlob>& outputs) {
      const Im2colParam& param = nnvm::get<Im2colParam>(attrs.parsed);
      std::vector<TBlob> in_data(inputs.begin() + 1, inputs.end());
      const TBlob &out_grad = inputs[0];
      const std::vector<TBlob> &in_grad = outputs;

      MSHADOW_REAL_TYPE_SWITCH(out_grad.type_flag_, DType, {
        Im2colOp<xpu, DType> op;
        op.Init(param);
        op.Backward(ctx, std::vector<TBlob>{out_grad}, in_data, req, in_grad);
      });
    }

  } // namespace op
} //namespace mxnet
#endif  // MXNET_OPERATOR_MY_IM2COL_INL_H_