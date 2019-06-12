#ifndef MXNET_OPERATOR_DEFORMABLE_UPSAMPLING_INL_H_
#define MXNET_OPERATOR_DEFORMABLE_UPSAMPLING_INL_H_

#include "operator_common.h"
#include "deformable_upsampling.h"
#include <mxnet/io.h>
#include <mxnet/base.h>
#include <mxnet/ndarray.h>
#include <mxnet/operator.h>
#include <mxnet/operator_util.h>
#include <mxnet/op_attr_types.h>
#include <dmlc/logging.h>
#include <dmlc/optional.h>
#include "tensor/init_op.h"


namespace mxnet {
  namespace op {
    namespace def_up {
      enum DeformableUpsamplingInputs {
        kData, kOffSets
      };
      enum DeformableUpsamplingOutputs {
        kOut
      };
    }

    //define param
    struct DeformableUpsamplingParam : public dmlc::Parameter<DeformableUpsamplingParam> {
      int deformable_group;
      int resize_h;
      int resize_w;

      DMLC_DECLARE_PARAMETER(DeformableUpsamplingParam) {
        DMLC_DECLARE_FIELD(deformable_group).set_default(1).describe("deformable groups");
        DMLC_DECLARE_FIELD(resize_h).describe("resize height");
        DMLC_DECLARE_FIELD(resize_w).describe("resize width");
      }
    };

    template<typename xpu, typename DType>
    class DeformableUpsamplingOp : public Operator {
    public:
      explicit DeformableUpsamplingOp(DeformableUpsamplingParam p) {
        this->deformable_group_ = p.deformable_group;
        this->resize_h_ = p.resize_h;
        this->resize_w_ = p.resize_w;
      }

      virtual void Forward(const OpContext &ctx,
                           const std::vector<TBlob> &in_data,
                           const std::vector<OpReqType> &req,
                           const std::vector<TBlob> &out_data,
                           const std::vector<TBlob> &aux_args) {
        using namespace mshadow;
        LayerSetUp(in_data[0].shape_, out_data[0].shape_);
        Stream<xpu> *s = ctx.get_stream<xpu>();
        for (index_t n = 0; n < batch_; ++n){
          deformable_upsampling_forward(
              s,
              in_data[def_up::kData].dptr<DType>() + n * input_data_dim_ ,
              in_data[def_up::kOffSets].dptr<DType>() + n * input_offset_dim_,
              data_h_,
              data_w_,
              channel_,
              resize_h_,
              resize_w_,
              channel_per_deformable_group_,
              out_data[def_up::kOut].dptr<DType>() + n * output_dim_
          );
        }
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
        LayerSetUp(in_data[0].shape_, out_grad[0].shape_);

        if (kAddTo == req[def_up::kData] || kWriteTo == req[def_up::kData]) {
          if (kWriteTo == req[def_up::kData]) {
            Fill<false>(s, in_grad[def_up::kData], kWriteTo, static_cast<DType>(0));
          }

          for (index_t n = 0; n < batch_; ++n){
            deformable_upsampling_backward_data(
                s,
                out_grad[def_up::kOut].dptr<DType>()+ n * output_dim_,
                in_data[def_up::kOffSets].dptr<DType>() + n * input_offset_dim_,
                data_h_,
                data_w_,
                channel_,
                resize_h_,
                resize_w_,
                channel_per_deformable_group_,
                in_grad[def_up::kData].dptr<DType>()+ n * input_data_dim_
            );
          }
        }

        if (kAddTo == req[def_up::kOffSets] || kWriteTo == req[def_up::kOffSets]){
          if (kWriteTo == req[def_up::kOffSets]) {
            Fill<false>(s, in_grad[def_up::kOffSets], kWriteTo, static_cast<DType>(0));
          }

          for (index_t n = 0; n < batch_ ; ++n){
            deformable_upsampling_backward_offset(
              s,
              out_grad[def_up::kOut].dptr<DType>()+ n * output_dim_,
              in_data[def_up::kData].dptr<DType>() + n * input_data_dim_,
              in_data[def_up::kOffSets].dptr<DType>() + n * input_offset_dim_,
              data_h_,
              data_w_,
              resize_h_,
              resize_w_,
              deformable_group_,
              channel_per_deformable_group_,
              in_grad[def_up::kOffSets].dptr<DType>()+ n * input_offset_dim_
            );
          }
        }
      }

    private:
      void LayerSetUp(const TShape &ishape, const TShape &oshape) {
        batch_ = ishape[0];
        channel_ = ishape[1];
        data_h_ = ishape[2];
        data_w_ = ishape[3];
        input_data_dim_ = channel_ * data_h_ * data_w_;
        input_offset_dim_ = 2 * deformable_group_ * resize_h_ * resize_w_;
        output_dim_ = channel_ * resize_h_ * resize_w_;
        channel_per_deformable_group_ = channel_ / deformable_group_;
      }

      index_t input_data_dim_;
      index_t input_offset_dim_;
      index_t output_dim_;
      index_t batch_;
      index_t channel_;
      index_t data_h_;
      index_t data_w_;
      index_t resize_h_;
      index_t resize_w_;
      index_t deformable_group_;
      index_t channel_per_deformable_group_;
    };

    template<typename xpu>
    Operator *CreateOp(DeformableUpsamplingParam param, int dtype,
                       std::vector<TShape> *in_shape,
                       std::vector<TShape> *out_shape,
                       Context ctx);


#if DMLC_USE_CXX11

    class DeformableUpsamplingProp : public OperatorProperty {
    public:
      std::vector<std::string> ListArguments() const override {
        return {"data", "offset"};
      }

      std::vector<std::string> ListOutputs() const {
        return {"outs"};
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
        CHECK_EQ(in_shape->size(), 2U) << "Input:[data, offset]";

        out_shape->resize(1, TShape());
        const TShape &data_shp = (*in_shape)[def_up::kData];
        CHECK_EQ(data_shp.ndim(), 4U) \
          << "Input data should be 4D in NxCxHxW";
        CHECK_EQ(data_shp[1] % param_.deformable_group, 0) \
          << "Input channels should be times of deformable_group";

        const TShape &offset_shp = (*in_shape)[def_up::kOffSets];
        CHECK_EQ(offset_shp.ndim(), 4U) \
          << "Offset data should be 4D in NxCxHxW";

        CHECK_EQ(data_shp[0], offset_shp[0]) \
          << "Input data and offset shape mismatchs in batch dim";
        CHECK_EQ(offset_shp[1], param_.deformable_group * 2) \
          << "Offset channels should equal to deformable_group * 2";
        CHECK_EQ(offset_shp[2], param_.resize_h) \
          << "Offset h mismatchs with resize_h";
        CHECK_EQ(offset_shp[3], param_.resize_w) \
          << "Offset w mismatchs with resize_w";


        TShape out_shp = TShape(4);
        out_shp[0] = data_shp[0];
        out_shp[1] = data_shp[1];
        out_shp[2] = offset_shp[2];
        out_shp[3] = offset_shp[3];
        SHAPE_ASSIGN_CHECK(*out_shape, def_up::kOut, out_shp);
        return true;
      }

      bool InferType(std::vector<int> *in_type,
                     std::vector<int> *out_type,
                     std::vector<int> *aux_type) const override {
        CHECK_GE(in_type->size(), 2U);
        int dtype = (*in_type)[0];
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

      OperatorProperty *Copy() const override {
        auto ptr = new DeformableUpsamplingProp();
        ptr->param_ = param_;
        return ptr;
      }

      std::string TypeString() const override {
        return "DeformableUpsampling";
      }

      std::vector<ResourceRequest> ForwardResource(
          const std::vector<TShape> &in_shape) const override {
        return {};
      }

      std::vector<ResourceRequest> BackwardResource(
          const std::vector<TShape> &in_shape) const override {
        return {};
      }


      Operator *CreateOperator(Context ctx) const override {
        LOG(FATAL) << "Not Implemented.";
        return NULL;
      }

      Operator *CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                                 std::vector<int> *in_type) const override;

    private:
      DeformableUpsamplingParam param_;
    };// class DeformableUpsampling
#endif  // DMLC_USE_CXX11
  } // end namespace op
} // end namespace mxnet
#endif  // MXNET_OPERATOR_DEFORMABLE_UPSAMPLING_INL_H_