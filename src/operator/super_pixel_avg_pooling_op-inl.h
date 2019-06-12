#ifndef MXNET_OPERATOR_SUPER_PIXEL_AVG_POOLING_INL_H_
#define MXNET_OPERATOR_SUPER_PIXEL_AVG_POOLING_INL_H_

#include "operator_common.h"
#include "super_pixel_avg_pooling.h"
#include <mxnet/io.h>
#include <mxnet/base.h>
#include <mxnet/ndarray.h>
#include <mxnet/operator.h>
#include <mxnet/operator_util.h>
#include <mxnet/op_attr_types.h>
#include <dmlc/optional.h>
#include <dmlc/logging.h>
#include "tensor/init_op.h"


namespace mxnet {
namespace op {

namespace spap {
  enum SpapOpInputs {kData, kLabel};
  enum SpapOpOutputs {kOut};
  enum SpapOpResource {kTempSpace};
}

  struct SuperPixelAvgPoolParam : public dmlc::Parameter<SuperPixelAvgPoolParam> {
    int max_spixels;
    DMLC_DECLARE_PARAMETER(SuperPixelAvgPoolParam) {
      DMLC_DECLARE_FIELD(max_spixels)
          .describe("Maximum super pixel id in input label");
    }
  };


  template<typename xpu, typename DType>
  class SuperPixelAvgPoolOp : public Operator {
  public:
    explicit SuperPixelAvgPoolOp(SuperPixelAvgPoolParam p) {
      this->param_ = p;
      max_spixels_ = param_.max_spixels;
    }

    virtual void Forward(const OpContext &ctx,
                         const std::vector<TBlob> &in_data,
                         const std::vector<OpReqType> &req,
                         const std::vector<TBlob> &out_data,
                         const std::vector<TBlob> &aux_args) {
      using namespace mshadow;
      Stream<xpu> *s = ctx.get_stream<xpu>();
      LayerSetUp(in_data[spap::kData].shape_);

      Tensor<xpu, 1, DType> workspace = ctx.requested[spap::kTempSpace].
          get_space_typed<xpu, 1, DType>(Shape1(buffer_size_), s);

      TBlob buffer(workspace.dptr_, buffer_size_, xpu::kDevMask, DataType<DType>::kFlag);

      for (int n = 0; n<batch_size_; ++n){
        SuperPixelAvgPoolingForward(
            s,
            in_data[spap::kData].dptr<DType>() + n * input_dim_,
            in_data[spap::kLabel].dptr<DType>() + n * label_dim_,
            channels_,
            height_,
            width_,
            max_spixels_,
            buffer.dptr<DType>(),
            out_data[spap::kOut].dptr<DType>() + n * output_dim_
        );
      }
    }

    virtual void Backward(const OpContext &ctx,
                          const std::vector<TBlob>& out_grad,
                          const std::vector<TBlob>& in_data,
                          const std::vector<TBlob>& out_data,
                          const std::vector<OpReqType>& req,
                          const std::vector<TBlob>& in_grad,
                          const std::vector<TBlob>& aux_args) {
      using namespace mshadow;
      Stream<xpu> *s = ctx.get_stream<xpu>();

      CHECK_EQ(out_grad.size(), 1);
      CHECK_EQ(in_grad.size(), 2);
      LayerSetUp(in_data[spap::kData].shape_);
      Tensor<xpu, 1, DType> workspace = ctx.requested[spap::kTempSpace].
          get_space_typed<xpu, 1, DType>(Shape1(buffer_size_), s);
      TBlob buffer(workspace.dptr_, buffer_size_, xpu::kDevMask, DataType<DType>::kFlag);

//      VLOG(x) << "Receiving argument " << batch_size_ << " " << channels_
//              << " " << height_ << " " << width_ << " " << max_spixels_;
//      VLOG(x) << "Setting buffer size as  " << buffer_size_;
//      VLOG(x) << "Setting input_dim as " << input_dim_;
//      VLOG(x) << "Setting label_dim as " << label_dim_;

      if (kAddTo == req[spap::kData] || kWriteTo == req[spap::kData]) {
        if (kWriteTo == req[spap::kData]) {
          Fill<false>(s, in_grad[spap::kData], kWriteTo, static_cast<DType>(0));
        }

        for (int n = 0; n < batch_size_; ++n) {
          SuperPixelAvgPoolingBackward(
              s,
              out_grad[spap::kOut].dptr<DType>() + n * output_dim_,
              in_data[spap::kLabel].dptr<DType>() + n * label_dim_,
              channels_,
              height_,
              width_,
              max_spixels_,
              buffer.dptr<DType>(),
              in_grad[spap::kData].dptr<DType>() + n * input_dim_
          );
        }
      }

      if (kWriteTo == req[spap::kLabel]) {
        Fill<false>(s, in_grad[spap::kLabel], kWriteTo, static_cast<DType>(0));
      }
    }

  private:

    void LayerSetUp(const TShape &in_shape) {
      batch_size_ = in_shape[0];
      channels_ = in_shape[1];
      height_ = in_shape[2];
      width_ = in_shape[3];
      input_dim_ = channels_ * height_ * width_;
      label_dim_ = height_ * width_;
      output_dim_ = input_dim_;
      buffer_size_ = max_spixels_ * (channels_ + 1);
    }


    SuperPixelAvgPoolParam param_;
    index_t max_spixels_;
    index_t batch_size_;
    index_t channels_;
    index_t height_;
    index_t width_;
    index_t buffer_size_;
    index_t input_dim_;
    index_t label_dim_;
    index_t output_dim_;
  };

  template<typename xpu>
  Operator* CreateOp(SuperPixelAvgPoolParam param, int dtype,
                     std::vector<TShape> *in_shape,
                     std::vector<TShape> *out_shape,
                     Context ctx);


#if DMLC_USE_CXX11
    class SuperPixelAvgPoolProp : public OperatorProperty {
    public:
      std::vector<std::string> ListArguments() const override {
        return {"data", "label"};
      }

      virtual std::vector<std::string> ListOutputs() const {
        return {"output"};
      }


      void Init(const std::vector<std::pair<std::string, std::string> > &kwargs) override {
        using namespace mshadow;
        param_.Init(kwargs);
      }

      std::map<std::string, std::string> GetParams() const override {
        return param_.__DICT__();
      }

      bool InferShape(std::vector<TShape> *in_shape,
                      std::vector<TShape> *out_shape,
                      std::vector<TShape> *aux_shape) const override {
        using namespace mshadow;
        CHECK_EQ(in_shape->size(), 2U)
          << "Input:[data, label]";
        out_shape->resize(1, TShape());

        const TShape &data_shape = (*in_shape)[spap::kData];
        const TShape &label_shape = (*in_shape)[spap::kLabel];

        CHECK_EQ(data_shape.ndim(), 4U) \
            << "Input data feature_shape should be 4D in NxCxHxW";
        CHECK_EQ(label_shape.ndim(), 3U) \
            << "Input label shape should be 3D in NxHxW";


        CHECK_EQ(data_shape[0], label_shape[0]) \
            << "Input data's batch dim mismatches with label";
        CHECK_EQ(data_shape[2], label_shape[1])\
            << "Input data's height dim mismatches with label";
        CHECK_EQ(data_shape[3], label_shape[2])\
            << "Input data's width dim mismatches with label";

        Shape<4> output_shape = Shape4(
            data_shape[0],
            data_shape[1],
            data_shape[2],
            data_shape[3]);
        SHAPE_ASSIGN_CHECK(*out_shape, 0, output_shape);
        return true;
      }

      bool InferType(std::vector<int> *in_type,
                     std::vector<int> *out_type,
                     std::vector<int> *aux_type) const override {
        CHECK_EQ(in_type->size(), 2U);
        int dtype = (*in_type)[spap::kData];
        CHECK_NE(dtype, -1) << "First input must have specified type";
        for (size_t i = 0; i < in_type->size(); ++i) {
          if ((*in_type)[i] == -1) {
            (*in_type)[i] = dtype;
          } else {
            UNIFORM_TYPE_CHECK((*in_type)[i], dtype, ListArguments()[i]);
          }
        }
        out_type->clear();
        out_type->push_back(dtype);
        return true;
      }

      OperatorProperty* Copy() const override {
        auto ptr = new SuperPixelAvgPoolProp();
        ptr->param_ = param_;
        return ptr;
      }

      std::string TypeString() const override {
        return "SuperPixelAvgPool";
      }

//      std::vector<int> DeclareBackwardDependency(
//          const std::vector<int> &out_grad,
//          const std::vector<int> &in_data,
//          const std::vector<int> &out_data) const override {
//        return{ out_grad[spap::kOut],
//                in_data[spap::kLabel] };
//      }

      std::vector<ResourceRequest> ForwardResource(
          const std::vector<TShape> &in_shape) const override {
        return{ ResourceRequest::kTempSpace };
      }

      std::vector<ResourceRequest> BackwardResource(
          const std::vector<TShape> &in_shape) const override {
        return{ ResourceRequest::kTempSpace };
      }

      Operator* CreateOperator(Context ctx) const override {
        LOG(FATAL) << "Not Implemented.";
        return NULL;
      }

      Operator* CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                                 std::vector<int> *in_type) const override;

    private:
      SuperPixelAvgPoolParam param_;
    }; // class prop
#endif // DMLC_USE_CXX11
}//end op
}//end mxnet

#endif