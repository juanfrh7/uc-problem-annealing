// Copyright 2022 D-Wave Systems Inc.
//
//    Licensed under the Apache License, Version 2.0 (the "License");
//    you may not use this file except in compliance with the License.
//    You may obtain a copy of the License at
//
//        http://www.apache.org/licenses/LICENSE-2.0
//
//    Unless required by applicable law or agreed to in writing, software
//    distributed under the License is distributed on an "AS IS" BASIS,
//    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//    See the License for the specific language governing permissions and
//    limitations under the License.

#pragma once

#include <algorithm>
#include <unordered_map>
#include <utility>
#include <vector>

#include "spdlog/spdlog.h"
#include "dimod/constrained_quadratic_model.h"

namespace dwave {
namespace presolve {

template <class Bias, class Index = int, class Assignment = double>
class Presolver;

template <class Bias, class Index, class Assignment>
class Postsolver {
 public:
    using bias_type = Bias;
    using index_type = Index;
    using size_type = std::size_t;
    using assignment_type = Assignment;

    /// Return a sample of the original CQM from a sample of the reduced CQM.
    template <class T>
    std::vector<T> apply(std::vector<T> reduced) const;

 private:
    // we want to track what changes were made
    enum TransformKind { FIX, SUBSTITUTE, ADD };

    // todo: we could get fancy with pointers and templates to save a bit of
    // space here
    struct Transform {
        TransformKind kind;
        index_type v;           // what variable it was applied to
        assignment_type value;  // if it was fixed what it was fixed to
        bias_type multiplier;
        bias_type offset;

        explicit Transform(TransformKind kind)
                : kind(kind), v(-1), value(NAN), multiplier(NAN), offset(NAN) {}
    };

    friend class Presolver<bias_type, index_type, assignment_type>;

    void add_variable(index_type v);

    void fix_variable(index_type v, assignment_type value);

    void substitute_variable(index_type v, bias_type multiplier, bias_type offset);

    std::vector<Transform> transforms_;
};

template <class bias_type, class index_type, class assignment_type>
void Postsolver<bias_type, index_type, assignment_type>::add_variable(index_type v) {
    transforms_.emplace_back(TransformKind::ADD);
    transforms_.back().v = v;
}

template <class bias_type, class index_type, class assignment_type>
template <class T>
std::vector<T> Postsolver<bias_type, index_type, assignment_type>::apply(
        std::vector<T> sample) const {
    // all that we have to do is undo the transforms back to front.
    for (auto it = transforms_.crbegin(); it != transforms_.crend(); ++it) {
        switch (it->kind) {
            case TransformKind::FIX:
                sample.insert(sample.begin() + it->v, it->value);
                break;
            case TransformKind::SUBSTITUTE:
                sample[it->v] *= it->multiplier;
                sample[it->v] += it->offset;
                break;
            case TransformKind::ADD:
                sample.erase(sample.begin() + it->v);
                break;
        }
    }
    return sample;
}

template <class bias_type, class index_type, class assignment_type>
void Postsolver<bias_type, index_type, assignment_type>::fix_variable(index_type v,
                                                                      assignment_type value) {
    transforms_.emplace_back(TransformKind::FIX);
    transforms_.back().v = v;
    transforms_.back().value = value;
}

template <class bias_type, class index_type, class assignment_type>
void Postsolver<bias_type, index_type, assignment_type>::substitute_variable(index_type v,
                                                                             bias_type multiplier,
                                                                             bias_type offset) {
    assert(multiplier);  // cannot undo when it's 0
    transforms_.emplace_back(TransformKind::SUBSTITUTE);
    transforms_.back().v = v;
    transforms_.back().multiplier = multiplier;
    transforms_.back().offset = offset;
}

template <class Bias, class Index, class Assignment>
class Presolver {
 public:
    using model_type = dimod::ConstrainedQuadraticModel<Bias, Index>;

    using bias_type = Bias;
    using index_type = Index;
    using size_type = typename model_type::size_type;

    using assignment_type = Assignment;

    /// Default constructor.
    Presolver();

    /// Construct a presolver from a constrained quadratic model.
    explicit Presolver(model_type model);

    /// Apply any loaded presolve techniques. Acts of the model() in-place.
    void apply();

    /// Detach the constrained quadratic model and return it.
    /// This clears the model from the presolver.
    model_type detach_model();

    /// Load the default presolve techniques.
    void load_default_presolvers();

    /// Return a const reference to the held constrained quadratic model.
    const model_type& model() const;

    /// Return a const reference to the held postsolver.
    const Postsolver<bias_type, index_type, assignment_type>& postsolver() const;

 private:
    model_type model_;
    Postsolver<bias_type, index_type, assignment_type> postsolver_;

    // todo: replace this with a vector of pointers or similar
    bool default_techniques_;

    bool detached_;

    void substitute_self_loops_expr(dimod::Expression<bias_type, index_type>& expression,
                                    std::unordered_map<index_type, index_type>& mapping) {
        size_type num_variables = expression.num_variables();
        for (size_type i = 0; i < num_variables; ++i) {
            index_type v = expression.variables()[i];

            if (!expression.has_interaction(v, v)) continue;  // no self loop

            auto out = mapping.emplace(v, model_.num_variables());

            if (out.second) {
                // we haven't seen this variable before
                model_.add_variable(model_.vartype(v), model_.lower_bound(v),
                                    model_.upper_bound(v));

                postsolver_.add_variable(out.first->second);
            }

            assert(static_cast<size_type>(out.first->second) < model_.num_variables());

            // now set the bias between v and the new variable
            expression.add_quadratic(v, out.first->second, expression.quadratic(v, v));
            expression.remove_interaction(v, v);
        }
    }

    //----- One-time Techniques -----//

    void technique_spin_to_binary() {
        for (size_type v = 0; v < model_.num_variables(); ++v) {
            if (model_.vartype(v) == dimod::Vartype::SPIN) {
                postsolver_.substitute_variable(v, 2, -1);
                model_.change_vartype(dimod::Vartype::BINARY, v);
            }
        }
    }
    void technique_remove_offsets() {
        for (size_type c = 0; c < model_.num_constraints(); ++c) {
            auto& constraint = model_.constraint_ref(c);
            if (constraint.offset()) {
                constraint.set_rhs(constraint.rhs() - constraint.offset());
                constraint.set_offset(0);
            }
        }
    }
    void technique_flip_constraints() {
        for (size_type c = 0; c < model_.num_constraints(); ++c) {
            auto& constraint = model_.constraint_ref(c);
            if (constraint.sense() == dimod::Sense::GE) {
                constraint.scale(-1);
            }
        }
    }
    void technique_remove_self_loops() {
        std::unordered_map<index_type, index_type> mapping;

        substitute_self_loops_expr(model_.objective, mapping);

        for (size_type c = 0; c < model_.num_constraints(); ++c) {
            substitute_self_loops_expr(model_.constraint_ref(c), mapping);
        }

        // now, we need to add equality constraints for all of the added variables
        for (const auto& uv : mapping) {
            // equality constraint
            model_.add_linear_constraint({uv.first, uv.second}, {1, -1}, dimod::Sense::EQ, 0);
        }
    }
    void technique_remove_invalid_markers() {
        std::vector<index_type> discrete;
        for (size_type c = 0; c < model_.num_constraints(); ++c) {
            auto& constraint = model_.constraint_ref(c);

            if (!constraint.marked_discrete()) continue;

            // we can check if it's well formed
            if (constraint.is_onehot()) {
                discrete.push_back(c);
            } else {
                constraint.mark_discrete(false);  // if it's not one-hot, it's not discrete
            }
        }
        // check if they overlap
        size_type i = 0;
        while (i < discrete.size()) {
            // check if ci overlaps with any other constraints
            auto& constraint = model_.constraint_ref(discrete[i]);

            bool overlap = false;
            for (size_type j = i + 1; j < discrete.size(); ++j) {
                if (model_.constraint_ref(discrete[j]).shares_variables(constraint)) {
                    // we have overlap!
                    overlap = true;
                    constraint.mark_discrete(false);
                    break;
                }
            }

            if (overlap) {
                discrete.erase(discrete.begin() + i);
                continue;
            }

            ++i;
        }
    }

    //----- Trivial Techniques -----//

    bool technique_check_for_nan() {
        // TODO: Implement
        return false;
    }
    bool technique_remove_single_variable_constraints() {
        bool ret = false;
        size_type c = 0;
        while (c < model_.num_constraints()) {
            auto& constraint = model_.constraint_ref(c);

            if (constraint.num_variables() == 0) {
                if (!constraint.is_soft()) {
                    // check feasibity
                    switch (constraint.sense()) {
                        case dimod::Sense::EQ:
                            if (constraint.offset() != constraint.rhs()) {
                                // need this exact message for Python
                                throw std::logic_error("infeasible");
                            }
                            break;
                        case dimod::Sense::LE:
                            if (constraint.offset() > constraint.rhs()) {
                                // need this exact message for Python
                                throw std::logic_error("infeasible");
                            }
                            break;
                        case dimod::Sense::GE:
                            if (constraint.offset() < constraint.rhs()) {
                                // need this exact message for Python
                                throw std::logic_error("infeasible");
                            }
                            break;
                    }
                }

                // we remove the constraint regardless of whether it's soft
                // or not. We could use the opportunity to update the objective
                // offset with the violation of the soft constraint, but
                // presolve does not preserve the energy in general, so it's
                // better to avoid side effects and just remove.
                model_.remove_constraint(c);
                ret = true;
                continue;
            } else if (constraint.num_variables() == 1 && !constraint.is_soft()) {
                index_type v = constraint.variables()[0];

                // ax ◯ c
                bias_type a = constraint.linear(v);
                assert(a);  // should have already been removed if 0

                // offset should have already been removed but may as well be safe
                bias_type rhs = (constraint.rhs() - constraint.offset()) / a;

                // todo: test if negative

                if (constraint.sense() == dimod::Sense::EQ) {
                    model_.set_lower_bound(v, std::max(rhs, model_.lower_bound(v)));
                    model_.set_upper_bound(v, std::min(rhs, model_.upper_bound(v)));
                } else if ((constraint.sense() == dimod::Sense::LE) != (a < 0)) {
                    model_.set_upper_bound(v, std::min(rhs, model_.upper_bound(v)));
                } else {
                    assert((constraint.sense() == dimod::Sense::GE) == (a >= 0));
                    model_.set_lower_bound(v, std::max(rhs, model_.lower_bound(v)));
                }

                model_.remove_constraint(c);
                ret = true;
                continue;
            }

            ++c;
        }
        return ret;
    }
    bool technique_remove_zero_biases() {
        bool ret = false;

        ret |= remove_zero_biases(model_.objective);
        for (size_t c = 0; c < model_.num_constraints(); ++c) {
            ret |= remove_zero_biases(model_.constraint_ref(c));
        }

        return ret;
    }
    bool technique_tighten_bounds() {
        bool ret = false;
        bias_type lb;
        bias_type ub;
        for (size_type v = 0; v < model_.num_variables(); ++v) {
            switch (model_.vartype(v)) {
                case dimod::Vartype::SPIN:
                case dimod::Vartype::BINARY:
                case dimod::Vartype::INTEGER:
                    ub = model_.upper_bound(v);
                    if (ub != std::floor(ub)) {
                        model_.set_upper_bound(v, std::floor(ub));
                        ret = true;
                    }
                    lb = model_.lower_bound(v);
                    if (lb != std::ceil(lb)) {
                        model_.set_lower_bound(v, std::ceil(lb));
                        ret = true;
                    }
                    break;
                case dimod::Vartype::REAL:
                    break;
            }
        }
        return ret;
    }
    bool technique_remove_fixed_variables() {
        bool ret = false; 
        size_type v = 0;
        while (v < model_.num_variables()) {
            if (model_.lower_bound(v) == model_.upper_bound(v)) {
                postsolver_.fix_variable(v, model_.lower_bound(v));
                model_.fix_variable(v, model_.lower_bound(v));
                ret = true;
            }
            ++v;
        }
        return ret;
    }

    static bool remove_zero_biases(dimod::Expression<bias_type, index_type>& expression) {
        // quadratic
        std::vector<std::pair<index_type, index_type>> empty_interactions;
        for (auto it = expression.cbegin_quadratic(); it != expression.cend_quadratic(); ++it) {
            if (!(it->bias)) {
                empty_interactions.emplace_back(it->u, it->v);
            }
        }
        for (auto& uv : empty_interactions) {
            expression.remove_interaction(uv.first, uv.second);
        }

        // linear
        std::vector<index_type> empty_variables;
        for (auto& v : expression.variables()) {
            if (expression.linear(v)) continue;
            if (expression.num_interactions(v)) continue;
            empty_variables.emplace_back(v);
        }
        for (auto& v : empty_variables) {
            expression.remove_variable(v);
        }

        return empty_interactions.size() || empty_variables.size();
    }
};

template <class bias_type, class index_type, class assignment_type>
Presolver<bias_type, index_type, assignment_type>::Presolver()
        : model_(), postsolver_(), default_techniques_(false), detached_(false) {}

template <class bias_type, class index_type, class assignment_type>
Presolver<bias_type, index_type, assignment_type>::Presolver(model_type model)
        : model_(std::move(model)), postsolver_(), default_techniques_(), detached_(false) {}

template <class bias_type, class index_type, class assignment_type>
void Presolver<bias_type, index_type, assignment_type>::apply() {
    if (detached_) throw std::logic_error("model has been detached, presolver is no longer valid");

    // If no techniques have been loaded, return early.
    if (!default_techniques_) return;

    // One time techniques ----------------------------------------------------

    // *-- spin-to-binary
    technique_spin_to_binary();
    // *-- remove offsets
    technique_remove_offsets();
    // *-- flip >= constraints
    technique_flip_constraints();
    // *-- remove self-loops
    technique_remove_self_loops();

    // Trivial techniques -----------------------------------------------------

    bool changes = true;
    const index_type max_num_rounds = 100;  // todo: make configurable
    for (index_type num_rounds = 0; num_rounds < max_num_rounds; ++num_rounds) {
        if (!changes) break;
        changes = false;

        // *-- clear out 0 variables/interactions in the constraints and objective
        changes |= technique_remove_zero_biases();
        // *-- todo: check for NAN
        changes |= technique_check_for_nan();
        // *-- remove single variable constraints
        changes |= technique_remove_single_variable_constraints();
        // *-- tighten bounds based on vartype
        changes |= technique_tighten_bounds();
        // *-- remove variables that are fixed by bounds
        changes |= technique_remove_fixed_variables();
   }

    // Cleanup

    // *-- remove any invalid discrete markers
    technique_remove_invalid_markers();
}

template <class bias_type, class index_type, class assignment_type>
dimod::ConstrainedQuadraticModel<bias_type, index_type>
Presolver<bias_type, index_type, assignment_type>::detach_model() {
    using std::swap;  // ADL, though doubt it makes a difference

    auto cqm = dimod::ConstrainedQuadraticModel<bias_type, index_type>();
    swap(model_, cqm);

    detached_ = true;

    return cqm;
}

template <class bias_type, class index_type, class assignment_type>
void Presolver<bias_type, index_type, assignment_type>::load_default_presolvers() {
    default_techniques_ = true;
}

template <class bias_type, class index_type, class assignment_type>
const dimod::ConstrainedQuadraticModel<bias_type, index_type>&
Presolver<bias_type, index_type, assignment_type>::model() const {
    return model_;
}

template <class bias_type, class index_type, class assignment_type>
const Postsolver<bias_type, index_type, assignment_type>&
Presolver<bias_type, index_type, assignment_type>::postsolver() const {
    return postsolver_;
}

}  // namespace presolve
}  // namespace dwave
