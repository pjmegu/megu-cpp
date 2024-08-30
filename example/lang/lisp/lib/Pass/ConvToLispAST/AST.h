#pragma once

#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace mokey::example::lisp {

struct AInt;
struct APlus;
struct AVariable;
struct AReturn;
struct ALet;

using AExpr = std::variant<AVariable, AInt, APlus>;
using AState = std::variant<AReturn, ALet>;
using ARange = std::pair<uint, uint>;
using AIdent = std::pair<std::string, ARange>;

struct AInt {
    int value;
    ARange range;
};

struct APlus {
    std::unique_ptr<AExpr> lhs;
    std::unique_ptr<AExpr> rhs;
    ARange range;

    inline APlus()
        : lhs(std::make_unique<AExpr>(AInt{0})),
          rhs(std::make_unique<AExpr>(AInt{0})) {}
};

struct AVariable {
    std::string name;
    ARange range;
};

struct AReturn {
    std::unique_ptr<AExpr> expr;
    ARange range;
};

struct ALet {
    AIdent name;
    std::unique_ptr<AExpr> expr;
    ARange range;
};

struct AFunc {
    AIdent name;
    std::vector<AState> states;
    ARange range;
};

struct APackage {
    AIdent name;
    std::vector<AFunc> funcs;
    ARange range;
};

} // namespace mokey::example::lisp