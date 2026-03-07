
#include "operators/Unary.h"
#include "core/operator.h"
#include "core/ref.h"
#include "core/runtime.h"
#include "utils/utils.h"

#include <optional>
#include <sstream>
namespace infini {

// Possible op list:
// Relu
// Sigmoid
// Silu
// Gelu
// Softplus
// Tanh

UnaryObj::UnaryObj(GraphObj *graph, OpType type, Tensor input0, Tensor output)
    : OperatorObj(type, TensorVec{input0}, {output}), type(type) {

    IT_ASSERT(checkValid(graph));
}
string UnaryObj::toString() const {
    std::ostringstream os;
    os << type.toString();
    os << "(";
    os << "input0=" << inputs[0]->getGuid() << ",";
    os << "output=" << outputs[0]->getGuid() << ")";
    return os.str();
}
UnaryObj::~UnaryObj() {
    if (infiniOpDesc) {
        infiniStatus_t err = INFINI_STATUS_SUCCESS;
        if (type == OpType::Relu) {
            err = infiniopDestroyReluDescriptor(
                (infiniopReluDescriptor_t)infiniOpDesc);
        } else if (type == OpType::Sigmoid) {
            err = infiniopDestroySigmoidDescriptor(
                (infiniopSigmoidDescriptor_t)infiniOpDesc);
        } else if (type == OpType::Silu) {
            err = infiniopDestroySiluDescriptor(
                (infiniopSiluDescriptor_t)infiniOpDesc);
        } else if (type == OpType::Gelu) {
            err = infiniopDestroyGeluDescriptor(
                (infiniopGeluDescriptor_t)infiniOpDesc);
        } else if (type == OpType::Softplus) {
            err = infiniopDestroySoftplusDescriptor(
                (infiniopSoftplusDescriptor_t)infiniOpDesc);
        } else if (type == OpType::Tanh) {
            err = infiniopDestroyTanhDescriptor(
                (infiniopTanhDescriptor_t)infiniOpDesc);
        }
        if (err != INFINI_STATUS_SUCCESS) {
            std::cerr << "Warning: " << type.toString()
                      << " descriptor destroy failed with error code " << err
                      << std::endl;
        }
    }
}

void UnaryObj::createOpDesc() {
    auto outShape = outputs[0]->getShape();
    auto inputShape = inputs[0]->getShape();
    auto inputStride =
        broadcastStride(inputShape, inputs[0]->getStride(), outShape);

    auto outStride = outputs[0]->getStride();
    infiniopTensorDescriptor_t outputTensor, inputTensor;
    CHECK_INFINI_ERROR(infiniopCreateTensorDescriptor(
        &outputTensor, outShape->size(), outShape->getConstantValue().data(),
        outStride->getConstantValue().data(),
        outputs[0]->getDataType().getType()));
    CHECK_INFINI_ERROR(infiniopCreateTensorDescriptor(
        &inputTensor, outShape->size(), outShape->getConstantValue().data(),
        inputStride->getConstantValue().data(),
        inputs[0]->getDataType().getType()));
    infiniopHandle_t handle = nullptr;
    CHECK_INFINI_ERROR(infiniopCreateHandle(&handle));
    if (type == OpType::Relu) {
        CHECK_INFINI_ERROR(infiniopCreateReluDescriptor(
            handle, (infiniopReluDescriptor_t *)&infiniOpDesc, outputTensor,
            inputTensor));
    } else if (type == OpType::Sigmoid) {
        CHECK_INFINI_ERROR(infiniopCreateSigmoidDescriptor(
            handle, (infiniopSigmoidDescriptor_t *)&infiniOpDesc, outputTensor,
            inputTensor));
    } else if (type == OpType::Silu) {
        CHECK_INFINI_ERROR(infiniopCreateSiluDescriptor(
            handle, (infiniopSiluDescriptor_t *)&infiniOpDesc, outputTensor,
            inputTensor));
    } else if (type == OpType::Gelu) {
        CHECK_INFINI_ERROR(infiniopCreateGeluDescriptor(
            handle, (infiniopGeluDescriptor_t *)&infiniOpDesc, outputTensor,
            inputTensor));
    } else if (type == OpType::Softplus) {
        CHECK_INFINI_ERROR(infiniopCreateSoftplusDescriptor(
            handle, (infiniopSoftplusDescriptor_t *)&infiniOpDesc, outputTensor,
            inputTensor));
    } else if (type == OpType::Tanh) {
        CHECK_INFINI_ERROR(infiniopCreateTanhDescriptor(
            handle, (infiniopTanhDescriptor_t *)&infiniOpDesc, outputTensor,
            inputTensor));
    } else {
        IT_TODO_HALT_MSG("Unary operator not supported yet");
    }

    CHECK_INFINI_ERROR(infiniopDestroyTensorDescriptor(outputTensor));
    CHECK_INFINI_ERROR(infiniopDestroyTensorDescriptor(inputTensor));
}
optional<vector<ShapeExpr>> UnaryObj::inferShape() {
    auto shapeInput = inputs[0]->getShape();
    return {{shapeInput}};
}
vector<DataType> UnaryObj::inferDataType() const {
    return {inputs[0]->getDataType()};
}

OpType UnaryObj::getUnaryOpType() const { return type; }

} // namespace infini
