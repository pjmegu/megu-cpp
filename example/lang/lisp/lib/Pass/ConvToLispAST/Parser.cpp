#include "AST.h"
#include "mlir/Support/LogicalResult.h"
#include <algorithm>
#include <memory>
#include <optional>
#include <string>
#include <sys/types.h>
#include <tuple>
#include <utility>

namespace mokey::example::lisp {

namespace {

#define CHECK(r)                                                               \
    if (auto result = check(); mlir::failed(result)) {                         \
        return {result, r};                                                    \
    }

#define CHECK_CHAR(c, r)                                                       \
    if (source[index] != c)                                                    \
        return {mlir::failure(), r};                                           \
    index++;

#define CHECK_STR(s, r)                                                        \
    std::string __gen_##s = #s;                                                \
    if (source.substr(index, (__gen_##s).size()) != (__gen_##s))               \
        return {mlir::failure(), r};                                           \
    index += (__gen_##s).size();

#define CHECK_R(result, r)                                                     \
    if (mlir::failed(result)) {                                                \
        return {result, r};                                                    \
    }

class ParserImpl {
  public:
    ParserImpl(std::string source) : source(source) {}
    std::tuple<mlir::LogicalResult, APackage> parsePackage();

  private:
    std::string source;
    uint index = 0;

    void skip();
    mlir::LogicalResult check();
    std::tuple<mlir::LogicalResult, AIdent> parseIdent();
    std::tuple<mlir::LogicalResult, AFunc> parseFunc();
    std::tuple<mlir::LogicalResult, AInt> parseInt();
    std::tuple<mlir::LogicalResult, APlus> parsePlus();
    std::tuple<mlir::LogicalResult, AVariable> parseVariable();
    std::tuple<mlir::LogicalResult, AReturn> parseReturn();
    std::tuple<mlir::LogicalResult, ALet> parseLet();
    std::tuple<mlir::LogicalResult, AExpr> parseExpr();
    std::tuple<mlir::LogicalResult, AState> parseState();
};

std::tuple<bool, char> checkCharList(char c, std::initializer_list<char> list) {
    for (auto &i : list) {
        if (c == i) {
            return {true, i};
        }
    }

    return {false, 0};
}

std::tuple<bool, char> checkCharList(char c, std::string list) {
    for (auto &i : list) {
        if (c == i) {
            return {true, i};
        }
    }

    return {false, 0};
}

mlir::LogicalResult ParserImpl::check() {
    if (index >= source.size()) {
        return mlir::failure();
    } else {
        return mlir::success();
    }
}

void ParserImpl::skip() {
    while (true) {
        if (index >= source.size()) {
            return;
        } else if (auto [isSpace, _] =
                       checkCharList(source[index], {' ', '\n', '\t', '\r'});
                   isSpace) {
            index++;
        } else {
            return;
        }
    }
}

std::tuple<mlir::LogicalResult, AInt> ParserImpl::parseInt() {
    AInt aint;

    const std::string validChars = "0123456789";

    skip();

    auto start = index;

    while (true) {
        if (index >= source.size()) {
            return {mlir::failure(), aint};
        } else if (auto [isSpace, _] = checkCharList(source[index], {' ', ')'});
                   isSpace) {
            return {mlir::success(), aint};
        } else if (auto [isSpace, c] =
                       checkCharList(source[index], validChars.c_str());
                   isSpace) {
            aint.value = aint.value * 10 + (c - '0');
            index++;
        }
    }

    auto end = index;

    skip();

    aint.range = {start, end};
    return {mlir::success(), aint};
}

std::tuple<mlir::LogicalResult, APlus> ParserImpl::parsePlus() {
    APlus aplus;

    skip();

    auto start = index;
    CHECK_CHAR('(', std::move(aplus));

    skip();

    CHECK_CHAR('+', std::move(aplus));

    skip();

    auto [result, lhs] = parseExpr();
    CHECK_R(result, std::move(aplus));
    aplus.lhs = std::make_unique<AExpr>(std::move(lhs));

    skip();

    auto [result2, rhs] = parseExpr();
    CHECK_R(result2, std::move(aplus));
    aplus.rhs = std::make_unique<AExpr>(std::move(rhs));

    skip();

    auto end = index;
    CHECK_CHAR(')', std::move(aplus));

    aplus.range = {start, end};
    return {mlir::success(), std::move(aplus)};
}

std::tuple<mlir::LogicalResult, AVariable> ParserImpl::parseVariable() {
    AVariable avar;

    skip();

    auto [result, ident] = parseIdent();
    CHECK_R(result, avar);
    avar.name = ident.first;
    avar.range = ident.second;

    return {mlir::success(), avar};
}

std::tuple<mlir::LogicalResult, ALet> ParserImpl::parseLet() {
    ALet alet;

    skip();

    auto start = index;
    CHECK_CHAR('(', std::move(alet));

    skip();

    CHECK_STR(let, std::move(alet));

    skip();

    auto [result, name] = parseIdent();
    CHECK_R(result, std::move(alet));
    alet.name = name;

    skip();

    auto [result2, expr] = parseExpr();
    CHECK_R(result2, std::move(alet));
    alet.expr = std::make_unique<AExpr>(std::move(expr));

    skip();

    auto end = index;
    CHECK_CHAR(')', std::move(alet));

    alet.range = {start, end};
    return {mlir::success(), std::move(alet)};
}

std::tuple<mlir::LogicalResult, AExpr> ParserImpl::parseExpr() {
    AExpr aexpr;

    if (source[index] == '(') {
        if (source[index + 1] == '+') {
            auto [result, plus] = parsePlus();
            CHECK_R(result, std::move(aexpr));
            aexpr = std::move(plus);
            return {mlir::success(), std::move(aexpr)};
        } else {
            return {mlir::failure(), std::move(aexpr)};
        }
    } else if (source[index] == '\"') {
        auto [result, var] = parseVariable();
        CHECK_R(result, std::move(aexpr));
        aexpr = std::move(var);
        return {mlir::success(), std::move(aexpr)};
    } else if (auto [isSpace, _] =
                   checkCharList(source[index], {'0', '1', '2', '3', '4', '5',
                                                 '6', '7', '8', '9'});
               isSpace) {
        auto [result, aint] = parseInt();
        CHECK_R(result, std::move(aexpr));
        aexpr = std::move(aint);
        return {mlir::success(), std::move(aexpr)};
    } else {
        return {mlir::failure(), std::move(aexpr)};
    }
}

std::tuple<mlir::LogicalResult, AState> ParserImpl::parseState() {
    AState astate;

    if (source[index] == '(') {
        if (source[index + 1] == 'r') {
            auto [result, ret] = parseReturn();
            CHECK_R(result, std::move(astate));
            astate = std::move(ret);
            return {mlir::success(), std::move(astate)};
        } else if (source[index + 1] == 'l') {
            auto [result, let] = parseLet();
            CHECK_R(result, std::move(astate));
            astate = std::move(let);
            return {mlir::success(), std::move(astate)};
        } else {
            return {mlir::failure(), std::move(astate)};
        }
    } else {
        return {mlir::failure(), std::move(astate)};
    }
}

std::tuple<mlir::LogicalResult, AIdent> ParserImpl::parseIdent() {
    std::string ident;

    const std::string validChars =
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_";

    auto start = index;

    // CHECK_CHAR('\"', {ident, {0, 0}});
    if (source[index] != '\"')
        return {mlir ::failure(), {std ::string(""), {0, 0}}};
    index++;

    while (true) {
        if (index >= source.size()) {
            return {mlir::failure(), {ident, {0, 0}}};
        } else if (auto [isSpace, _] = checkCharList(source[index], {'\"'});
                   isSpace) {
            auto end = index;
            index++;
            return {mlir::success(), {ident, {start, end}}};
        } else if (auto [isSpace, c] =
                       checkCharList(source[index], validChars.c_str());
                   isSpace) {
            ident += c;
            index++;
        }
    }
}

std::tuple<mlir::LogicalResult, AFunc> ParserImpl::parseFunc() {
    AFunc func;
    skip();
    CHECK(func);

    auto start = index;
    CHECK_CHAR('(', func);

    skip();

    CHECK_STR(def, func);

    skip();

    auto [result, name] = parseIdent();
    CHECK_R(result, func);
    func.name = name;

    skip();

    CHECK_CHAR('(', func);

    skip();

    // parse exprs
    while (true) {
        skip();
        auto [result, state] = parseState();
        if (mlir::failed(result)) {
            break;
        } else {
            func.states.push_back(state);
        }
    }

    skip();

    CHECK_CHAR(')', func);

    skip();

    auto end = index;
    CHECK_CHAR(')', func);

    skip();

    func.range = {start, end};
    return {mlir::success(), func};
}

std::tuple<mlir::LogicalResult, APackage> ParserImpl::parsePackage() {
    APackage package;
    skip();
    CHECK(package);

    auto start = index;
    CHECK_CHAR('(', package);

    skip();

    CHECK_STR(package, package);

    skip();

    auto [result, name] = parseIdent();
    CHECK_R(result, package);
    package.name = name;

    skip();

    while (true) {
        skip();
        auto [result, func] = parseFunc();
        if (mlir::failed(result)) {
            break;
        } else {
            package.funcs.push_back(func);
        }
    }

    skip();

    auto end = index;
    CHECK_CHAR(')', package);

    package.range = {start, end};
    return {mlir::success(), package};
}

} // namespace

std::tuple<mlir::LogicalResult, APackage> parse(std::string source) {
    return ParserImpl(source).parsePackage();
}

} // namespace mokey::example::lisp