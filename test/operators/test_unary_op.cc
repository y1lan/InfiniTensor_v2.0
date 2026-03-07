#include "core/runtime.h"
#include "operators/Unary.h"
#include "utils/test_utils.h"
#include "gtest/gtest.h"

#include <cmath>
#include <vector>

namespace infini {

namespace {

float unaryRef(OpType type, float x) {
    if (type == OpType::Relu) {
        return x > 0.0f ? x : 0.0f;
    }
    if (type == OpType::Sigmoid) {
        return 1.0f / (1.0f + std::exp(-x));
    }
    if (type == OpType::Silu) {
        float sigmoid = 1.0f / (1.0f + std::exp(-x));
        return x * sigmoid;
    }
    if (type == OpType::Gelu) {
        return 0.5f * x * (1.0f + std::erf(x / std::sqrt(2.0f)));
    }
    if (type == OpType::Softplus) {
        return std::log1p(std::exp(x));
    }
    if (type == OpType::Tanh) {
        return std::tanh(x);
    }
    return x;
}

const char *opTypeName(OpType type) {
    if (type == OpType::Relu) {
        return "Relu";
    }
    if (type == OpType::Sigmoid) {
        return "Sigmoid";
    }
    if (type == OpType::Silu) {
        return "Silu";
    }
    if (type == OpType::Gelu) {
        return "Gelu";
    }
    if (type == OpType::Softplus) {
        return "Softplus";
    }
    if (type == OpType::Tanh) {
        return "Tanh";
    }
    return "Unknown";
}

} // namespace

class UnaryOpTest : public testing::TestWithParam<OpType> {
  protected:
    Runtime runtime;
    Graph graph;

    void SetUp() override {
        RuntimeObj::init();
        runtime = RuntimeObj::getInstance();
        runtime->initThreadContext(INFINI_DEVICE_CPU, 0);
        graph = make_ref<GraphObj>(runtime);
    }
};

TEST_P(UnaryOpTest, BasicConstruction) {
    auto input = graph->addTensor({2, 4}, DataType(INFINI_DTYPE_F32));
    auto unary = graph->addOp<UnaryObj>(GetParam(), input, nullptr);

    EXPECT_EQ(unary->getOpType(), GetParam());
    EXPECT_EQ(unary->getNumInputs(), 1);
    EXPECT_EQ(unary->getNumOutputs(), 1);
    EXPECT_EQ(unary->getUnaryOpType(), GetParam());
}

TEST_P(UnaryOpTest, ShapeAndTypeInference) {
    auto input = graph->addTensor({2, 4}, DataType(INFINI_DTYPE_F32));
    auto unary = graph->addOp<UnaryObj>(GetParam(), input, nullptr);

    auto inferredShapes = unary->inferShape();
    ASSERT_TRUE(inferredShapes.has_value());
    ASSERT_EQ(inferredShapes->size(), 1);

    auto outputShape = (*inferredShapes)[0];
    ASSERT_TRUE(outputShape->isConcrete());
    auto shapeValues = outputShape->getConstantValue();
    ASSERT_EQ(shapeValues.size(), 2);
    EXPECT_EQ(shapeValues[0], 2);
    EXPECT_EQ(shapeValues[1], 4);

    auto inferredTypes = unary->inferDataType();
    ASSERT_EQ(inferredTypes.size(), 1);
    EXPECT_EQ(inferredTypes[0], DataType(INFINI_DTYPE_F32));
}

TEST_P(UnaryOpTest, RunCpuAndCompareReference) {
    auto input = graph->addTensor({2, 4}, DataType(INFINI_DTYPE_F32));
    auto unary = graph->addOp<UnaryObj>(GetParam(), input, nullptr);

    std::vector<float> inputData = {-3.0f, -1.0f, -0.5f, 0.0f,
                                    0.5f,  1.0f,  2.0f,  3.0f};

    input->setData(inputData.data());
    runtime->dataMalloc(graph);
    runtime->run(graph);

    auto output = unary->getOutput(0);
    ASSERT_NE(output, nullptr);
    ASSERT_EQ(output->getElement(), inputData.size());

    std::vector<float> outputData(output->getElement());
    auto dataBlob = output->getData();
    ASSERT_NE(dataBlob, nullptr);

    void *hostPtr = runtime->allocHost(output->getTotalBytes());
    runtime->memcpy(hostPtr, dataBlob->getRawDataPtr(), output->getTotalBytes(),
                    INFINIRT_MEMCPY_D2H);
    copyAndConvertData(outputData, hostPtr, output->getElement(),
                       output->getDataType());
    runtime->deallocHost(hostPtr);

    float tolerance = (GetParam() == OpType::Gelu || GetParam() == OpType::Softplus)
                          ? 1e-3f
                          : 1e-4f;

    for (size_t i = 0; i < inputData.size(); ++i) {
        float expected = unaryRef(GetParam(), inputData[i]);
        EXPECT_NEAR(outputData[i], expected, tolerance)
            << "op=" << opTypeName(GetParam()) << ", idx=" << i
            << ", x=" << inputData[i] << ", got=" << outputData[i]
            << ", expected=" << expected;
    }
}

INSTANTIATE_TEST_SUITE_P(
    UnaryOps, UnaryOpTest,
    testing::Values(OpType::Relu, OpType::Sigmoid, OpType::Silu, OpType::Gelu,
                    OpType::Softplus, OpType::Tanh),
    [](const testing::TestParamInfo<OpType> &info) {
        return std::string(opTypeName(info.param));
    });

} // namespace infini
