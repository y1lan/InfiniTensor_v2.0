#pragma once
#include <optional>
#ifndef GRAPH_BUILDER_H
#define GRAPH_BUILDER_H

#include "core/graph.h"
#include "core/op_type.h"
#include "operators/Clip.h"
#include "operators/ElementWise.h"
#include "operators/Gemm.h"
#include "operators/Unary.h"

namespace infini {

class GraphBuilderObj {
  private:
    Ref<GraphObj> g;

  public:
    GraphBuilderObj(Runtime runtime);

    Tensor tensor(ShapeExpr dims, DataType dtype,
                  std::optional<StrideExpr> stride = std::nullopt);
    Tensor constant(ShapeExpr dims, DataType dtype, double value);

    Tensor gemm(Tensor A, Tensor B, Tensor C, float alpha = 1.0,
                float beta = 1.0, bool transA = false, bool transB = false,
                std::optional<Tensor> Y = std::nullopt);
    // elementwise ops{
    //
    Tensor add(Tensor A, Tensor B, std::optional<Tensor> Y = std::nullopt);
    Tensor sub(Tensor A, Tensor B, std::optional<Tensor> Y = std::nullopt);
    Tensor mul(Tensor A, Tensor B, std::optional<Tensor> Y = std::nullopt);
    //
    // }

    Tensor clip(Tensor Input, std::optional<Tensor> MinVal = std::nullopt,
                std::optional<Tensor> MaxVal = std::nullopt,
                std::optional<Tensor> Output = std::nullopt);
    // unary ops{
    //
    Tensor relu(Tensor Input, std::optional<Tensor> Output);
    Tensor sigmoid(Tensor Input, std::optional<Tensor> Output);
    Tensor silu(Tensor Input, std::optional<Tensor> Output);
    Tensor gelu(Tensor Input, std::optional<Tensor> Output);
    Tensor softplus(Tensor Input, std::optional<Tensor> Output);
    Tensor tanh(Tensor Input, std::optional<Tensor> Output);
    //
    // }

    string printGraph() const;

    Graph getGraph() const;
};

} // namespace infini
#endif // GRAPH_BUILDER_H
