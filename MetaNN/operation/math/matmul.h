//
//  Untitled.swift
//  MetaNNXC
//
//  Created by Vincent Predoehl on 3/8/26.
//

#pragma once

#include <cassert>
#include <type_traits>

#include "facilities/traits.h"
#include "facilities/cont_metafuns/helpers.h"
#include "evaluate/eval_group.h"
#include "evaluate/eval_handle.h"
#include "evaluate/eval_item.h"
#include "metal/metal_matmul.h"

namespace MetaNN::OpTags
{
    struct MatMul;
}

namespace MetaNN
{
    template <typename TOp1, typename TOp2>
    constexpr bool IsValidOper<OpTags::MatMul, TOp1, TOp2> =
        IsMatrix<RemConstRef<TOp1>> && IsMatrix<RemConstRef<TOp2>>;

    namespace OperMatMul
    {
        namespace NSCaseGen
        {
            template <typename TInputHandle1, typename TInputHandle2, typename TOutputHandle>
            class EvalItem : public BaseEvalItem
            {
                using CategoryTag = CategoryTagFromHandle<TOutputHandle>;
            public:
                EvalItem(TInputHandle1 oriHandle1,
                         TInputHandle2 oriHandle2,
                         TOutputHandle outputHandle,
                         Shape<CategoryTag::DimNum> outputShape)
                    : BaseEvalItem(TypeID<EvalItem>(),
                                   {oriHandle1.DataPtr(), oriHandle2.DataPtr()},
                                   outputHandle.DataPtr())
                    , m_inputHandle1(std::move(oriHandle1))
                    , m_inputHandle2(std::move(oriHandle2))
                    , m_outputHandle(std::move(outputHandle))
                    , m_outputShape(std::move(outputShape))
                {}

                const TInputHandle1 m_inputHandle1;
                const TInputHandle2 m_inputHandle2;
                TOutputHandle m_outputHandle;
                Shape<CategoryTag::DimNum> m_outputShape;
            };

            template <typename TInputHandle1, typename TInputHandle2, typename TOutputHandle>
            class EvalGroup : public TrivialEvalGroup<EvalItem<TInputHandle1, TInputHandle2, TOutputHandle>>
            {
                using EvalItemType = EvalItem<TInputHandle1, TInputHandle2, TOutputHandle>;
            protected:
                void EvalInternalLogic(EvalItemType& evalItem) final override
                {
                    const auto& in1 = evalItem.m_inputHandle1.Data();
                    const auto& in2 = evalItem.m_inputHandle2.Data();

                    // Matrix aliases are Tensor<...,2>, so shape[0]=rows, shape[1]=cols.
                    const size_t m = in1.Shape()[0];
                    const size_t k1 = in1.Shape()[1];
                    const size_t k2 = in2.Shape()[0];
                    const size_t n = in2.Shape()[1];

                    assert(k1 == k2);
                    assert(evalItem.m_outputShape[0] == m);
                    assert(evalItem.m_outputShape[1] == n);

                    using ResType = typename TOutputHandle::DataType;
                    using ElementType = typename ResType::ElementType;
                    using OutDevice = typename ResType::DeviceType;

                    static_assert(std::is_same_v<ElementType, float>,
                                  "Initial Metal MatMul implementation is float-only.");

                    ResType out(evalItem.m_outputShape);

                    if (m == 0 || k1 == 0 || n == 0)
                    {
                        evalItem.m_outputHandle.SetData(std::move(out));
                        return;
                    }

                    auto low_in1 = LowerAccess(in1);
                    auto low_in2 = LowerAccess(in2);
                    auto low_out = LowerAccess(out);

                    if constexpr (std::is_same_v<OutDevice, DeviceTags::Metal>)
                    {
                        const auto a_mem = low_in1.SharedMemory();
                        const auto b_mem = low_in2.SharedMemory();
                        auto c_mem = low_out.SharedMemory();

                        NSMetalMatMul::MatMul(a_mem,
                                              b_mem,
                                              c_mem,
                                              m, k1, n);
                    }
                    else if constexpr (std::is_same_v<OutDevice, DeviceTags::CPU>)
                    {
                        const ElementType* a = low_in1.RawMemory();
                        const ElementType* b = low_in2.RawMemory();
                        ElementType* c = low_out.MutableRawMemory();

                        for (size_t i = 0; i < m; ++i)
                        {
                            for (size_t j = 0; j < n; ++j)
                            {
                                ElementType sum = ElementType{};
                                for (size_t t = 0; t < k1; ++t)
                                {
                                    sum += a[i * k1 + t] * b[t * n + j];
                                }
                                c[i * n + j] = sum;
                            }
                        }
                    }
                    else
                    {
                        static_assert(std::is_same_v<OutDevice, DeviceTags::CPU> ||
                                      std::is_same_v<OutDevice, DeviceTags::Metal>,
                                      "Unsupported device type in MatMul EvalGroup");
                    }

                    evalItem.m_outputHandle.SetData(std::move(out));
                }
            };
        } // namespace NSCaseGen
    } // namespace OperMatMul

    template <>
    struct OperSeq_<OpTags::MatMul>
    {
        using type = OperCalAlgoChain<
            TailCalculator<
                OperMatMul::NSCaseGen::EvalItem,
                OperMatMul::NSCaseGen::EvalGroup,
                PolicyContainer<PPassShape>>>;
    };

    // interface
    template <typename TP1, typename TP2,
              std::enable_if_t<IsValidOper<OpTags::MatMul, TP1, TP2>>* = nullptr>
    auto MatMul(TP1&& p_m1, TP2&& p_m2)
    {
        using rawOp1 = RemConstRef<TP1>;
        using rawOp2 = RemConstRef<TP2>;
        using ResType = Operation<OpTags::MatMul,
                                  OperandContainer<rawOp1, rawOp2>,
                                  PolicyContainer<>>;
        return ResType(std::forward<TP1>(p_m1), std::forward<TP2>(p_m2));
    }
}

