
#include "operators/Clip.h"
#include "core/common.h"
#include "core/exception.h"
#include "core/op_type.h"
#include "infiniop/ops/clip.h"
#include "infiniop/tensor_descriptor.h"
namespace infini {

ClipObj::ClipObj(GraphObj *graph, Tensor input, Tensor min, Tensor max,
                 Tensor output)
    : OperatorObj(OpType::Clip, {input, min, max}, {output}) {
    IT_ASSERT(checkValid(graph));
}
string ClipObj::toString() const {
    std::ostringstream os;
    os << type.toString();
    os << "(";
    os << "input=" << inputs[0]->getGuid() << ",";
    os << "min=" << inputs[1]->getGuid() << ",";
    os << "max=" << inputs[2]->getGuid() << ",";
    os << "output=" << outputs[0]->getGuid() << ")";

    return os.str();
}
ClipObj::~ClipObj() {}

void ClipObj::createOpDesc() {
    auto outShape = outputs[0]->getShape();
    auto inputShape = inputs[0]->getShape();
    auto minShape = inputs[1]->getShape();
    auto maxShape = inputs[2]->getShape();

    auto inputStride =
        broadcastStride(inputShape, inputs[0]->getStride(), outShape);
    auto minStride =
        broadcastStride(minShape, inputs[1]->getStride(), outShape);
    auto maxStride =
        broadcastStride(maxShape, inputs[2]->getStride(), outShape);

    auto outputStride = outputs[0]->getStride();

    infiniopTensorDescriptor_t outputTensor, inputTensor, minTensor, maxTensor;

    CHECK_INFINI_ERROR(infiniopCreateTensorDescriptor(
        &outputTensor, outShape->size(), outShape->getConstantValue().data(),
        outputStride->getConstantValue().data(),
        outputs[0]->getDataType().getType()));

    CHECK_INFINI_ERROR(infiniopCreateTensorDescriptor(
        &inputTensor, outShape->size(), outShape->getConstantValue().data(),
        inputStride->getConstantValue().data(),
        inputs[0]->getDataType().getType()));

    CHECK_INFINI_ERROR(infiniopCreateTensorDescriptor(
        &minTensor, outShape->size(), outShape->getConstantValue().data(),
        minStride->getConstantValue().data(),
        inputs[1]->getDataType().getType()));

    CHECK_INFINI_ERROR(infiniopCreateTensorDescriptor(
        &maxTensor, outShape->size(), outShape->getConstantValue().data(),
        maxStride->getConstantValue().data(),
        inputs[2]->getDataType().getType()));

    infiniopHandle_t handle = nullptr;
    CHECK_INFINI_ERROR(infiniopCreateHandle(&handle));

    CHECK_INFINI_ERROR(infiniopCreateClipDescriptor(
        handle, (infiniopClipDescriptor_t *)&infiniOpDesc, outputTensor,
        inputTensor, minTensor, maxTensor));

    CHECK_INFINI_ERROR(infiniopDestroyTensorDescriptor(outputTensor));
    CHECK_INFINI_ERROR(infiniopDestroyTensorDescriptor(inputTensor));
    CHECK_INFINI_ERROR(infiniopDestroyTensorDescriptor(minTensor));
    CHECK_INFINI_ERROR(infiniopDestroyTensorDescriptor(maxTensor));
}

optional<vector<ShapeExpr>> ClipObj::inferShape() {
    auto shape = inputs[0]->getShape();
    return {{shape}};
}

vector<DataType> ClipObj::inferDataType() const {
    IT_ASSERT(inputs[0]->getDataType() == inputs[1]->getDataType() &&
              inputs[1]->getDataType() == inputs[2]->getDataType());
    return {inputs[0]->getDataType()};
}

} // namespace infini