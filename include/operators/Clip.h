#pragma once
#include "core/common.h"
#include "core/exception.h"
#include "core/graph.h"
#include "core/op_type.h"
#include "core/operator.h"
#include "core/runtime.h"
#include "utils/utils.h"
#include <infiniop/handle.h>
#include <infiniop/ops/clip.h>
#include <infiniop/tensor_descriptor.h>

namespace infini {
class ClipObj : public OperatorObj {
  public:
    /**
     * @brief Construct a new Clip object
     *
     * @param type Operator type.
     * @param graph The computation graph that this operator belongs to.
     * @param input0 The first input tensor.
     * @param min The input min value.
     * @param max The input max value.
     * @param output The output tensor.
     */
    ClipObj(GraphObj *graph, Tensor input0, Tensor min, Tensor max,
            Tensor output);
    string toString() const override;
    ~ClipObj() override;

    void createOpDesc() override;
    optional<vector<ShapeExpr>> inferShape() override;
    vector<DataType> inferDataType() const override;
};
} // namespace infini
