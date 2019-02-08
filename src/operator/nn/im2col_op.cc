#include "im2col_op-inl.h"
 #include "../../apache-mxnet-src-1.3.1.rc0-incubating/src/operator/elemwise_op_common.h"
 #include "./mkldnn/mkldnn_ops-inl.h"
 #include "./mkldnn/mkldnn_base-inl.h"


namespace mxnet {
  namespace op {
    DMLC_REGISTER_PARAMETER(Im2colParam);

    static inline index_t AddPad(index_t dsize, index_t pad) {
      return dsize + 2 * pad;
    }

    static inline std::vector<std::string> ListArguments(const Im2colParam& param_) {
        return {"data"};
    }


    // define a shape inferer
    static bool Im2colShape(const nnvm::NodeAttrs& attrs,
                                 std::vector<TShape> *in_shape,
                                 std::vector<TShape> *out_shape) {
      using namespace mshadow;
      const Im2colParam& param_ = nnvm::get<Im2colParam>(attrs.parsed);
      CHECK_EQ(in_shape->size(), 1U) << "Input:[data]";

      // CHECK_EQ(out_shape->size(), 1) << "Output: [output]";
      out_shape->resize(1, TShape());

      const TShape &dshp = (*in_shape)[0];
      if (dshp.ndim() ==  0) return false;

      // 2d im2col
      CHECK_EQ(dshp.ndim(), 4U) \
    << "Input data should be 4D in batch-num_filter-y-x";

      Shape<4> dshape = ConvertLayout(dshp.get<4>(), param_.layout.value(), kNCHW);

      const index_t dilated_ksize_y = param_.DilatedKernelSize(0);
      const index_t dilated_ksize_x = param_.DilatedKernelSize(1);

      CHECK_GT(param_.kernel.Size(), 0U) \
    << "incorrect kernel size: " << param_.kernel;
      CHECK_GT(param_.stride.Size(), 0U) \
    << "incorrect stride size: " << param_.stride;
      CHECK_GT(param_.dilate.Size(), 0U) \
    << "incorrect dilate size: " << param_.dilate;
      Shape<4> oshape;

      oshape[0] = dshape[0];
      oshape[1] = dshape[1] * param_.kernel.Size();
      oshape[2] = dshape[2] ?
                  (AddPad(dshape[2], param_.pad[0]) - dilated_ksize_y) / param_.stride[0] + 1 : 0;
      oshape[3] = dshape[3] ?
                  (AddPad(dshape[3], param_.pad[1]) - dilated_ksize_x) / param_.stride[1] + 1 : 0;
      SHAPE_ASSIGN_CHECK(*out_shape, 0, ConvertLayout(oshape, kNCHW, param_.layout.value()));
      // Perform incomplete shape inference. Fill in the missing values in data shape.
      // 1) We can always fill in the batch_size.
      // 2) We can back-calculate the input height/width if the corresponding stride is 1.
      oshape = ConvertLayout((*out_shape)[0].get<4>(), param_.layout.value(), kNCHW);
      dshape[0] = oshape[0];
      if (oshape[2] && param_.stride[0] == 1) {
        dshape[2] = oshape[2] + dilated_ksize_y - 1 - 2 * param_.pad[0];
      }
      if (oshape[3] && param_.stride[1] == 1) {
        dshape[3] = oshape[3] + dilated_ksize_x - 1 - 2 * param_.pad[1];
      }
      SHAPE_ASSIGN_CHECK(*in_shape, 0, ConvertLayout(dshape, kNCHW, param_.layout.value()));
      // Check whether the kernel sizes are valid
      if (dshape[2] != 0) {
        CHECK_LE(dilated_ksize_y, AddPad(dshape[2], param_.pad[0])) << "kernel size exceed input";
      }
      if (dshape[3] != 0) {
        CHECK_LE(dilated_ksize_x, AddPad(dshape[3], param_.pad[1])) << "kernel size exceed input";
      }
      return true;
    }

    // define a dtype inference
    static bool Im2colType(const nnvm::NodeAttrs& attrs,
                                std::vector<int> *in_type, std::vector<int> *out_type) {
      const Im2colParam& param_ = nnvm::get<Im2colParam>(attrs.parsed);
      CHECK_GE(in_type->size(), 1U);
      int dtype = (*in_type)[0];
      CHECK_NE(dtype, -1) << "First input must have specified type";
      for (index_t i = 0; i < in_type->size(); ++i) {
        if ((*in_type)[i] == -1) {
          (*in_type)[i] = dtype;
        } else {
          UNIFORM_TYPE_CHECK((*in_type)[i], dtype, ListArguments(param_)[i]);
        }
      }
      out_type->clear();
      out_type->push_back(dtype);
      return true;
    }

    // define an arg parser
    void Im2colParamParser(nnvm::NodeAttrs* attrs) {
      using namespace mshadow;
      Im2colParam param_;
      try {
        param_.Init(attrs->dict);
      } catch (const dmlc::ParamError& e) {
        std::ostringstream os;
        os << e.what();
        os << ", in operator " << attrs->op->name << "("
           << "name=\"" << attrs->name << "\"";
        for (const auto& k : attrs->dict) {
          os << ", " << k.first << "=\"" << k.second << "\"";
        }
        os << ")";
        throw dmlc::ParamError(os.str());
      }

      CHECK_EQ(param_.kernel.ndim(), 2)
    << "Kernel must have size of 2.";

      // set default value
      param_.layout = param_.layout ? param_.layout.value() : mshadow::kNCHW;
      if (param_.stride.ndim() == 0) param_.stride = Shape2(1, 1);
      if (param_.dilate.ndim() == 0) param_.dilate = Shape2(1, 1);
      if (param_.pad.ndim() == 0) param_.pad = Shape2(0, 0);

      CHECK_EQ(param_.kernel.ndim(), param_.stride.ndim())
        << "Stride must have the same number of dimensions with kernel_size,"
        << "but kernel_size is set to " << param_.kernel << " while stride is "
        << param_.stride;
      CHECK_EQ(param_.kernel.ndim(), param_.dilate.ndim())
        << "Dilate must have the same number of dimensions with kernel_size,"
        << "but kernel_size is set to " << param_.kernel << " while dilate is "
        << param_.dilate;
      CHECK_EQ(param_.kernel.ndim(), param_.pad.ndim())
        << "Padding must have the same number of dimensions with kernel_size,"
        << "but kernel_size is set to " << param_.kernel << " while padding is "
        << param_.pad;
      attrs->parsed = std::move(param_);

    }

    inline static bool Im2colStorageType(const nnvm::NodeAttrs& attrs,
                                       const int dev_mask,
                                       DispatchMode* dispatch_mode,
                                       std::vector<int> *in_attrs,
                                       std::vector<int> *out_attrs) {
      const Im2colParam& param = nnvm::get<Im2colParam>(attrs.parsed);
      CHECK_EQ(in_attrs->size(), 1);
      CHECK_EQ(out_attrs->size(), 1);

      DispatchMode wanted_mode;
      wanted_mode = DispatchMode::kFCompute;
      return storage_type_assign(out_attrs, mxnet::kDefaultStorage,
                                 dispatch_mode, wanted_mode);
    }

    inline static bool BackwardIm2colStorageType(const nnvm::NodeAttrs& attrs,
                                               const int dev_mask,
                                               DispatchMode* dispatch_mode,
                                               std::vector<int> *in_attrs,
                                               std::vector<int> *out_attrs) {
      const Im2colParam& param = nnvm::get<Im2colParam>(attrs.parsed);
      // t
      uint32_t in_expected = 2;
      uint32_t out_expected = 1;
      CHECK_EQ(in_attrs->size(), in_expected);
      CHECK_EQ(out_attrs->size(), out_expected);

      DispatchMode wanted_mode;
      wanted_mode = DispatchMode::kFCompute;
      return storage_type_assign(out_attrs, mxnet::kDefaultStorage,
                                 dispatch_mode, wanted_mode);
    }

    struct Im2colGrad {
      const char *op_name;
      std::vector<nnvm::NodeEntry> operator()(const nnvm::NodePtr& n,
                                              const std::vector<nnvm::NodeEntry>& ograds) const {
        const Im2colParam& param = nnvm::get<Im2colParam>(n->attrs.parsed);
        std::vector<nnvm::NodeEntry> heads(ograds.begin(), ograds.end());
        heads.push_back(n->inputs[0]);
        return MakeGradNode(op_name, n, heads, n->attrs.dict);
      }
    };

    NNVM_REGISTER_OP(Im2col)
      .describe(R"code(
        im2col, only 2d supported... Orz
	input [N, C, H, W] return [N, C*kh*kw, fh, fw]
	can use reshape (0, -4, C, -1, -2) to extract it...
      )code" ADD_FILELINE)
      .set_num_inputs([](const NodeAttrs& attrs) {
        return 1;
      })
      .set_num_outputs(1)
      .set_attr_parser(Im2colParamParser)
      .set_attr<nnvm::FListInputNames>("FListInputNames",
                                       [](const NodeAttrs& attrs) {
                                         return std::vector<std::string>{"data"};
                                       })
      .set_attr<nnvm::FListOutputNames>("FListOutputNames",
                                        [](const NodeAttrs& attrs) {
                                          return std::vector<std::string>{"output"};
                                        })
      .set_attr<nnvm::FInferShape>("FInferShape", Im2colShape)
      .set_attr<nnvm::FInferType>("FInferType", Im2colType)
      .set_attr<FInferStorageType>("FInferStorageType", Im2colStorageType)
      .set_attr<FCompute>("FCompute<cpu>", Im2colCompute<cpu>)
      .set_attr<nnvm::FGradient>("FGradient", Im2colGrad{"_backward_Im2col"})
      .set_attr<FResourceRequest>("FResourceRequest", [](const NodeAttrs& n) {
        return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
      })
      .add_argument("data", "NDArray-or-Symbol", "Input data to the ConvolutionOp.")
      .add_arguments(Im2colParam::__FIELDS__());

  NNVM_REGISTER_OP(_backward_Im2col)
      .set_num_outputs([](const NodeAttrs& attrs) {
        return 1;
      })
      .set_attr<nnvm::TIsBackward>("TIsBackward", true)
      .set_attr<FInferStorageType>("FInferStorageType", BackwardIm2colStorageType)
      .set_attr<FResourceRequest>("FResourceRequest", [](const NodeAttrs& n) {
        return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
      })
      .set_attr_parser(Im2colParamParser)
      .set_attr<FCompute>("FCompute<cpu>", Im2colGradCompute<cpu>);
  }
}
