#ifndef MXNET_OPERATOR_ROI_UPSAMPLE_INL_H_
#define MXNET_OPERATOR_ROI_UPSAMPLE_INL_H_

#include "operator_common.h"
#include "roi_upsample.h"
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
namespace rus {
  enum RoIUpsampleOpInputs {kROIFeatures, kROIBoxes};
  enum RoIUpsampleOpOutputs {kOut};
}

struct RoIUpsampleParam : public dmlc::Parameter<RoIUpsampleParam> {
  int sample_ratio;
  int output_height;
  int output_width;
  int output_batch;

  DMLC_DECLARE_PARAMETER(RoIUpsampleParam) {
    DMLC_DECLARE_FIELD(sample_ratio)
        .set_default(-1)
        .describe("Sample ratio used in roi upsample");
    DMLC_DECLARE_FIELD(output_height)
        .describe("the output height of roi upsample");
    DMLC_DECLARE_FIELD(output_width)
        .describe("the output width of roi upsample");
    DMLC_DECLARE_FIELD(output_batch)
        .describe("the output batch of roi upsample, "
                  "if batch id in roi boxes exceeds it, the exceeded ones are discarded");
  }
};

template<typename xpu, typename DType>
class RoIUpsampleOp : public Operator{
public:
  explicit RoIUpsampleOp(RoIUpsampleParam p) {
    this->param_ = p;
    sample_ratio_ = p.sample_ratio;
    output_batch_ = p.output_batch;
    output_height_  = p.output_height;
    output_width_ = p.output_width;
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    Stream<xpu> *s = ctx.get_stream<xpu>();
    LayerSetUp(in_data[rus::kROIFeatures].shape_,
               in_data[rus::kROIBoxes].shape_,
               out_data[rus::kOut].shape_);
    Fill<false>(s, out_data[rus::kOut], kWriteTo, static_cast<DType>(0));

    RoIUpsampleForward(
        s,
        in_data[rus::kROIFeatures].dptr<DType>(),
        in_data[rus::kROIBoxes].dptr<DType>(),
        nrois_,
        output_batch_,
        channels_,
        output_height_,
        output_width_,
        pooled_height_,
        pooled_width_,
        sample_ratio_,
        rois_cols_,
        out_data[rus::kOut].dptr<DType>()
    );
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
    CHECK_EQ(req.size(), 2);

    LayerSetUp(in_grad[rus::kROIFeatures].shape_,
               in_grad[rus::kROIBoxes].shape_,
               out_grad[rus::kOut].shape_);

    const DType *bottom_rois = in_data[rus::kROIBoxes].dptr<DType>();
    const DType *top_diff = out_grad[rus::kOut].dptr<DType>();

    if (kWriteTo == req[rus::kROIBoxes]) {
      Fill<false>(s, in_grad[rus::kROIBoxes], kWriteTo, static_cast<DType>(0));
    }

    if (kAddTo == req[rus::kROIFeatures] || kWriteTo == req[rus::kROIFeatures]) {
      if (kWriteTo == req[rus::kROIFeatures]) {
        Fill<false>(s, in_grad[rus::kROIFeatures], kWriteTo, static_cast<DType>(0));
      }

      RoIUpsampleBackward(
        s,
        top_diff, // [batch_size, c, h, w]
        bottom_rois,
        nrois_,
        output_batch_,
        channels_,
        output_height_,
        output_width_,
        pooled_height_,
        pooled_width_,
        sample_ratio_,
        rois_cols_,
        in_grad[rus::kROIFeatures].dptr<DType>() //[nrois, c, pooled_height, pooled_width]
      );
    }
  }

private:
  void LayerSetUp(const TShape &roi_fshape, const TShape &roi_bshape, const TShape &oshape) {
    nrois_ = roi_fshape[0];
    channels_ = roi_fshape[1];
    pooled_height_ = roi_fshape[2];
    pooled_width_ = roi_fshape[3];
    rois_cols_ = roi_bshape[1];
  }

  RoIUpsampleParam param_;
  index_t nrois_;
  index_t channels_;
  index_t rois_cols_;
  index_t pooled_height_;
  index_t pooled_width_;
  index_t sample_ratio_;
  index_t output_height_;
  index_t output_width_;
  index_t output_batch_;
}; // class Op

template<typename xpu>
Operator* CreateOp(RoIUpsampleParam param, int dtype,
                   std::vector<TShape> *in_shape,
                   std::vector<TShape> *out_shape,
                   Context ctx);


#if DMLC_USE_CXX11
class RoIUpsampleProp : public OperatorProperty {
public:
  std::vector<std::string> ListArguments() const override {
    return {"roi_features", "roi_boxes"};
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
      << "Input:[roi_features, roi_boxes]";
    out_shape->resize(1, TShape());

    const TShape &feature_shape = (*in_shape)[rus::kROIFeatures];
    const TShape &box_shape = (*in_shape)[rus::kROIBoxes];

    CHECK_EQ(feature_shape.ndim(), 4U) \
        << "Input ROI feature_shape should be 4D in NxCxHxW";

    CHECK_EQ(box_shape.ndim(), 2U) \
        << "Input ROI roi boxes should be 2D \n";

    CHECK_GE(box_shape[1], 4U) \
        << "ROI boxes shape[2] can be only 4 or 5";
    CHECK_LE(box_shape[1], 5U) \
        << "ROI boxes shape[2] can be only 4 or 5";

    CHECK_EQ(feature_shape[0], box_shape[0])\
        << "ROI feature's batch dim mismatches with ROI boxes";
    //asign output shape
    Shape<4> output_shape = Shape4(
        param_.output_batch,
        feature_shape[1],
        param_.output_height,
        param_.output_width);
    SHAPE_ASSIGN_CHECK(*out_shape, 0, output_shape);
    return true;
  }

  bool InferType(std::vector<int> *in_type,
                 std::vector<int> *out_type,
                 std::vector<int> *aux_type) const override {
    CHECK_EQ(in_type->size(), 2U);
    int dtype = (*in_type)[rus::kROIFeatures];
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
    auto ptr = new RoIUpsampleProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "RoIUpsample";
  }


  Operator* CreateOperator(Context ctx) const override {
    LOG(FATAL) << "Not Implemented.";
    return NULL;
  }

  Operator* CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                             std::vector<int> *in_type) const override;

private:
  RoIUpsampleParam param_;
}; // class prop
#endif // DMLC_USE_CXX11
} // end op
} //end mxnet

#endif