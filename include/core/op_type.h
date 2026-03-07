#pragma once
#ifndef OP_TYPE_H
#define OP_TYPE_H

#include "core/common.h"

namespace infini {
struct OpType {
    using underlying_t = uint16_t;
    enum : underlying_t {
        Unknown,
        Add,
        Cast,
        Clip,
        Concat,
        Div,
        Gemm,
        Mul,
        MatMul,
        Relu,
        Sub,
        Transpose,
        Sigmoid,
        Silu,
        Gelu,
        Softplus,
        Tanh,

    } type;

    constexpr OpType(decltype(type) t) : type(t) {}
    constexpr explicit OpType(underlying_t val) : type((decltype(type))val) {}
    constexpr underlying_t underlying() const { return type; }

    bool operator==(OpType others) const { return type == others.type; }
    bool operator!=(OpType others) const { return type != others.type; }

    const char *toString() const {
#define CASE(NAME)                                                             \
    case OpType::NAME:                                                         \
        return #NAME

        switch (type) {
            CASE(Unknown);
            CASE(Add);
            CASE(Cast);
            CASE(Clip);
            CASE(Concat);
            CASE(Div);
            CASE(Gemm);
            CASE(Mul);
            CASE(MatMul);
            CASE(Relu);
            CASE(Sub);
            CASE(Transpose);
            CASE(Sigmoid);
            CASE(Silu);
            CASE(Gelu);
            CASE(Softplus);
            CASE(Tanh);

        default:
            return "Unknown";
        }

#undef CASE
    };
};

} // namespace infini

#endif // OP_TYPE_H
