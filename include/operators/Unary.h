#pragma once
#include "core/graph.h"
#include "core/operator.h"
#include <infiniop/ops/relu.h>
#include <infiniop/ops/sigmoid.h>
#include <infiniop/ops/silu.h>

#include <infiniop/ops/gelu.h>
#include <infiniop/ops/softplus.h>
#include <infiniop/ops/tanh.h>

namespace infini {
class UnaryObj : public OperatorObj {
  private:
    OpType type;

  public:
    /**
     * @brief Construct a new ElementWise object
     *
     * @param type Operator type.
     * @param graph The computation graph that this operator belongs to.
     * @param input0 The first input tensor.
     * @param input1 The second input tensor.
     * @param output The output tensor.
     */
    UnaryObj(GraphObj *graph, OpType type, Tensor input0, Tensor output);
    string toString() const override;
    ~UnaryObj() override;

    void createOpDesc() override;
    optional<vector<ShapeExpr>> inferShape() override;
    vector<DataType> inferDataType() const override;

    OpType getUnaryOpType() const;
};
} // namespace infini
