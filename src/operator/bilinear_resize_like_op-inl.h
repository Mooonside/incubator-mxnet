#ifndef MXNET_OPERATOR_BILINEAR_RESIZE_LIKE_INL_H_
#define MXNET_OPERATOR_BILINEAR_RESIZE_LIKE_INL_H_


#include "operator_common.h"
#include <mxnet/io.h>
#include <mxnet/base.h>
#include <mxnet/ndarray.h>
#include <mxnet/operator.h>
#include <mxnet/operator_util.h>
#include <mxnet/op_attr_types.h>
#include <dmlc/logging.h>
#include <dmlc/optional.h>
#include "tensor/init_op.h"


#ifndef MXNET_OPERATOR_CONTRIB_BILINEAR_RESIZE_INL_H_
#include "contrib/bilinear_resize-inl.h"
#endif

namespace mxnet {
namespace op {
  namespace brl{
    enum BilinearResizeLikeInputs {
      kData, kLike
    };
    enum BilinearResizeLikeOutputs {
      kOut
    };
  }

  struct BilinearResizeLikeParam : public dmlc::Parameter<BilinearResizeLikeParam> {
    DMLC_DECLARE_PARAMETER(BilinearResizeLikeParam) {

    }
  };

  template<typename xpu, typename DType, typename AccReal>
  class BilinearResizeLikeOp : public Operator {
  public:
    explicit BilinearResizeLikeOp(BilinearResizeLikeParam p) {
      this->param_ = p;
    }

    virtual void Forward(const OpContext &ctx,
                         const std::vector<TBlob> &in_data,
                         const std::vector<OpReqType> &req,
                         const std::vector<TBlob> &out_data,
                         const std::vector<TBlob> &aux_args) {
      using namespace mshadow;
      Stream<xpu> *s = ctx.get_stream<xpu>();
      CHECK_EQ(req[brl::kOut], kWriteTo);
      CHECK_EQ(in_data.size(), 2U);
      CHECK_EQ(out_data.size(), 1U);
//      LayerSetUp(in_data[brl::kData].shape_, out_data[brl::kOut].shape_);

      const std::vector<TBlob> real_in = {in_data[brl::kData]};
      SpatialUpSamplingBilinearUpdateOutput<xpu, DType, AccReal>(s, real_in, out_data);
    }

    virtual void Backward(const OpContext &ctx,
                          const std::vector<TBlob> &out_grad,
                          const std::vector<TBlob> &in_data,
                          const std::vector<TBlob> &out_data,
                          const std::vector<OpReqType> &req,
                          const std::vector<TBlob> &in_grad,
                          const std::vector<TBlob> &aux_args) {
      using namespace mshadow;
      Stream<xpu> *s = ctx.get_stream<xpu>();
      CHECK_EQ(out_grad.size(), 1);
      CHECK_EQ(in_grad.size(), 2);
      CHECK_EQ(req.size(), 2);
//        LayerSetUp(in_grad[brl::kData].shape_, out_grad[brl::kOut].shape_);
      // no grad to klike
      if (kWriteTo == req[brl::kLike]) {
        Fill<false>(s, in_grad[brl::kLike], kWriteTo, static_cast<DType>(0));
      }
      const std::vector<TBlob> real_in_grad = {in_grad[brl::kData]};
      SpatialUpSamplingBilinearUpdateGradInput<xpu, DType, AccReal>(s, out_grad, real_in_grad);
    }

  private:
//    void LayerSetUp(const TShape& ishape, const TShape& oshape) {
//      // batch size
//      num_ = ishape[0];
//      // input/output image size (#channels * height * width)
//      channels_ = ishape[1];
//      in_height_ = ishape[2];
//      in_width_ = ishape[3];
//      out_height_ = oshape[2];
//      out_width_ = oshape[3];
//      input_dim_ = ishape.ProdShape(1, ishape.ndim());
//      output_dim_ = oshape.ProdShape(1, oshape.ndim());
//    }
//    index_t num_;  // batch size
//    index_t channels_;
//    index_t in_height_;
//    index_t in_width_;
//    index_t out_height_;
//    index_t out_width_;
//    index_t input_dim_;
//    index_t output_dim_;


    BilinearResizeLikeParam param_;
  };

  template<typename xpu>
  Operator *CreateOp(BilinearResizeLikeParam param, int dtype,
                     std::vector<TShape> *in_shape,
                     std::vector<TShape> *out_shape,
                     Context ctx);

#if DMLC_USE_CXX11
    class BilinearResizeLikeProp : public OperatorProperty {
    public:
      std::vector<std::string> ListArguments() const override {
        return {"data", "like"};
      }

      std::vector<std::string> ListOutputs() const {
        return {"out"};
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
        CHECK_EQ(in_shape->size(), 2U) << "Input:[data, like]";

        out_shape->resize(1, TShape());
        const TShape &data_shape = (*in_shape)[brl::kData];
        CHECK_EQ(data_shape.ndim(), 4U) \
          << "Input data should be 4D in NxCxHxW";

        const TShape &like_shape = (*in_shape)[brl::kLike];
        CHECK_EQ(like_shape.ndim(), 4U) \
          << "Input like should be 4D in NxCxHxW";

        TShape oshape = TShape(4);
        oshape[0] = data_shape[0];
        oshape[1] = data_shape[1];
        oshape[2] = like_shape[2];
        oshape[3] = like_shape[3];
        SHAPE_ASSIGN_CHECK(*out_shape, brl::kOut, oshape);
        return true;
      }

      bool InferType(std::vector<int> *in_type,
                     std::vector<int> *out_type,
                     std::vector<int> *aux_type) const override {
        CHECK_GE(in_type->size(), 1U);
        int dtype = (*in_type)[0];
        CHECK_NE(dtype, -1) << "First input must have specified type";

        out_type->clear();
        out_type->push_back(dtype);
        return true;
      }

      OperatorProperty *Copy() const override {
        auto ptr = new BilinearResizeLikeProp();
        ptr->param_ = param_;
        return ptr;
      }

      std::string TypeString() const override {
        return "BilinearResizeLike";
      }

      std::vector<ResourceRequest> ForwardResource(
          const std::vector<TShape> &in_shape) const override {
        return {};
      }

      std::vector<ResourceRequest> BackwardResource(
          const std::vector<TShape> &in_shape) const override {
        return {};
      }

//      virtual std::vector<int> DeclareBackwardDependency(
//          const std::vector<int> &out_grad,
//          const std::vector<int> &in_data,
//          const std::vector<int> &out_data) const {
//        // By default requires to see all the things.
//        // remember to override this function to get a better performance.
//        return {};
//      }

      Operator *CreateOperator(Context ctx) const override {
        LOG(FATAL) << "Not Implemented.";
        return NULL;
      }

      Operator *CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                                 std::vector<int> *in_type) const override;

    private:
      BilinearResizeLikeParam param_;
    };// class BilinearResizeLike
#endif  // DMLC_USE_CXX11
} //op
} //mxnet

#endif