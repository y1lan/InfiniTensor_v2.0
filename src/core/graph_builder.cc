#include "core/graph_builder.h"
#include "core/runtime.h"
#include <algorithm>
#include <cstring>
#include <limits>
#include <utility>

namespace infini {

GraphBuilderObj::GraphBuilderObj(Runtime runtime)
    : g(make_ref<GraphObj>(std::move(runtime))) {}

Tensor GraphBuilderObj::tensor(ShapeExpr dims, DataType dtype,
                               std::optional<StrideExpr> stride) {
    if (stride.has_value()) {
        return g->addTensor(dims, stride.value(), dtype);
    } else {
        return g->addTensor(dims, dtype);
    }
}

namespace {
uint16_t fp32_to_bf16(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return static_cast<uint16_t>(bits >> 16);
}

template <typename T> void fill_buffer(void *buffer, size_t count, T value) {
    T *typed = static_cast<T *>(buffer);
    std::fill(typed, typed + count, value);
}
} // namespace

Tensor GraphBuilderObj::constant(ShapeExpr dims, DataType dtype, double value) {
    IT_ASSERT(dims->isConcrete(), "Constant tensor requires concrete shape");
    Tensor t = g->addTensor(dims, dtype);

    const size_t num_elements = static_cast<size_t>(t->getElement());
    const size_t total_bytes = static_cast<size_t>(t->getTotalBytes());
    std::vector<uint8_t> host(total_bytes);

    switch (dtype.getType()) {
    case INFINI_DTYPE_BOOL:
        fill_buffer<bool>(host.data(), num_elements, value != 0.0);
        break;
    case INFINI_DTYPE_I8:
        fill_buffer<int8_t>(host.data(), num_elements,
                            static_cast<int8_t>(value));
        break;
    case INFINI_DTYPE_I16:
        fill_buffer<int16_t>(host.data(), num_elements,
                             static_cast<int16_t>(value));
        break;
    case INFINI_DTYPE_I32:
        fill_buffer<int32_t>(host.data(), num_elements,
                             static_cast<int32_t>(value));
        break;
    case INFINI_DTYPE_I64:
        fill_buffer<int64_t>(host.data(), num_elements,
                             static_cast<int64_t>(value));
        break;
    case INFINI_DTYPE_U8:
        fill_buffer<uint8_t>(host.data(), num_elements,
                             static_cast<uint8_t>(value));
        break;
    case INFINI_DTYPE_U16:
        fill_buffer<uint16_t>(host.data(), num_elements,
                              static_cast<uint16_t>(value));
        break;
    case INFINI_DTYPE_U32:
        fill_buffer<uint32_t>(host.data(), num_elements,
                              static_cast<uint32_t>(value));
        break;
    case INFINI_DTYPE_U64:
        fill_buffer<uint64_t>(host.data(), num_elements,
                              static_cast<uint64_t>(value));
        break;
    case INFINI_DTYPE_F16: {
        uint16_t v = fp32_to_fp16(static_cast<float>(value));
        fill_buffer<uint16_t>(host.data(), num_elements, v);
        break;
    }
    case INFINI_DTYPE_BF16: {
        uint16_t v = fp32_to_bf16(static_cast<float>(value));
        fill_buffer<uint16_t>(host.data(), num_elements, v);
        break;
    }
    case INFINI_DTYPE_F32:
        fill_buffer<float>(host.data(), num_elements,
                           static_cast<float>(value));
        break;
    case INFINI_DTYPE_F64:
        fill_buffer<double>(host.data(), num_elements,
                            static_cast<double>(value));
        break;
    default:
        IT_TODO_HALT_MSG("Unsupported data type for constant tensor");
    }

    auto runtime = g->getRuntime();
    t->dataMalloc(runtime);
    auto kind = runtime->isCpu() ? INFINIRT_MEMCPY_H2H : INFINIRT_MEMCPY_H2D;
    runtime->memcpy(t->getRawDataPtr<void *>(), host.data(), total_bytes, kind);

    return t;
}

Tensor GraphBuilderObj::gemm(Tensor A, Tensor B, Tensor C, float alpha,
                             float beta, bool transA, bool transB,
                             std::optional<Tensor> Y) {
    if (Y.has_value()) {
        g->addOpWithOutputs<GemmObj>(std::move(A), std::move(B),
                                     std::move(Y.value()), std::move(C), alpha,
                                     beta, transA, transB);
        return Y.value();
    } else {
        return g
            ->addOp<GemmObj>(std::move(A), std::move(B), nullptr, std::move(C),
                             alpha, beta, transA, transB)
            ->getOutput(0);
    }
}

Tensor GraphBuilderObj::clip(Tensor Input, std::optional<Tensor> MinVal,
                             std::optional<Tensor> MaxVal,
                             std::optional<Tensor> Output) {
    cout << "running clip" << endl;
    Tensor local_min_val, local_max_val;
    if (MinVal.has_value()) {
        local_min_val = MinVal.value();
    } else {
        local_min_val = constant({}, Input->getDataType(),
                                 -std::numeric_limits<double>::infinity());
    }
    if (MaxVal.has_value()) {
        local_max_val = MaxVal.value();
    } else {
        local_max_val = constant({}, Input->getDataType(),
                                 std::numeric_limits<double>::infinity());
    }
    if (Output.has_value()) {
        g->addOpWithOutputs<ClipObj>(std::move(Input), std::move(local_min_val),
                                     std::move(local_max_val),
                                     std::move(Output.value()));
        return Output.value();
    } else {
        return g
            ->addOp<ClipObj>(std::move(Input), std::move(local_min_val),
                             std::move(local_max_val), nullptr)
            ->getOutput(0);
    }
}

#define DEFINE_UNARY_OP(OP, TYPE)                                              \
    Tensor GraphBuilderObj::OP(Tensor Input, std::optional<Tensor> Output) {   \
        if (Output.has_value()) {                                              \
            g->addOpWithOutputs<UnaryObj>(TYPE, std::move(Input),              \
                                          std::move(Output.value()));          \
            return Output.value();                                             \
        } else {                                                               \
            return g->addOp<UnaryObj>(TYPE, std::move(Input), nullptr)         \
                ->getOutput(0);                                                \
        }                                                                      \
    }

#define DEFINE_BINARY_OP(OP, TYPE)                                             \
    Tensor GraphBuilderObj::OP(Tensor A, Tensor B, std::optional<Tensor> Y) {  \
        if (Y.has_value()) {                                                   \
            g->addOpWithOutputs<ElementWiseObj>(                               \
                TYPE, std::move(A), std::move(B), std::move(Y.value()));       \
            return Y.value();                                                  \
        } else {                                                               \
            return g                                                           \
                ->addOp<ElementWiseObj>(TYPE, std::move(A), std::move(B),      \
                                        nullptr)                               \
                ->getOutput(0);                                                \
        }                                                                      \
    }

DEFINE_BINARY_OP(add, OpType::Add);
DEFINE_BINARY_OP(sub, OpType::Sub);
DEFINE_BINARY_OP(mul, OpType::Mul);

DEFINE_UNARY_OP(relu, OpType::Relu);
DEFINE_UNARY_OP(sigmoid, OpType::Sigmoid);
DEFINE_UNARY_OP(gelu, OpType::Gelu);
DEFINE_UNARY_OP(silu, OpType::Silu);
DEFINE_UNARY_OP(softplus, OpType::Softplus);
DEFINE_UNARY_OP(tanh, OpType::Tanh);

string GraphBuilderObj::printGraph() const { return g->toString(); }

Graph GraphBuilderObj::getGraph() const { return g; }
} // namespace infini
