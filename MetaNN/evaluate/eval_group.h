#pragma once

#include <MetaNN/evaluate/eval_item.h>
#include <list>
#include <memory>
#include <stdexcept>

namespace MetaNN
{
    class BaseEvalGroup
    {
    public:
        virtual ~BaseEvalGroup() = default;
        
        virtual void Add(std::unique_ptr<BaseEvalItem>) = 0;
        virtual void Eval() = 0;
        virtual std::list<const void*> ResultPointers() const = 0;
    };
    
    template <typename TEvalItem>
    class TrivialEvalGroup : public BaseEvalGroup
    {
    public:
        virtual void Add(std::unique_ptr<BaseEvalItem> item) override final
        {
            if (m_evalItem)
            {
                throw std::runtime_error("Cannot add to this group any more!");
            }
            BaseEvalItem* to = item.release();
            auto aim = dynamic_cast<TEvalItem*>(to);
            if (!aim)
            {
                delete to;
                throw std::runtime_error("Incorrect eval item type for this group.");
            }
            m_evalItem.reset(aim);
        }

        void Eval() override final
        {
            if (!m_evalItem)
                throw std::runtime_error("No eval item added now.");
            EvalInternalLogic(*m_evalItem);
        }
        
        virtual std::list<const void*> ResultPointers() const override final
        {
            if (!m_evalItem)
            {
                throw std::runtime_error("No eval item added now.");
            }
            const BaseEvalItem* base = dynamic_cast<const BaseEvalItem*>(m_evalItem.get());
            if (!base)
            {
                throw std::runtime_error("Stored eval item type is incompatible with BaseEvalItem.");
            }
            std::list<const void*> res;
            res.push_back(base->OutputPtr());
            return res;
        }

    protected:
        virtual void EvalInternalLogic(TEvalItem&) = 0;
    private:
        std::unique_ptr<TEvalItem> m_evalItem;
    };
}

