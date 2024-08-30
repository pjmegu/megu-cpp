#include "lisp/Pass/ConvToLispAST/.h"

#include "AST.h"
#include "lisp/LispAST/LispASTType.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "mokey/MokeyToken/MokeyTokenDialect.h"
#include "mokey/MokeyToken/MokeyTokenOps.h"

#include "lisp/LispAST/LispASTOps.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <cstdint>
#include <iostream>
#include <ostream>
#include <string>
#include <tuple>
#include <utility>
#include <variant>

namespace mokey::example::lisp {

// from ./Parser.cpp
std::tuple<mlir::LogicalResult, APackage> parse(std::string source);

#define GEN_PASS_DEF_CONVTOLISPAST
#include "lisp/Pass/ConvToLispAST/.h.inc"

namespace {

class ConvToLispASTRewriter
    : public mlir::OpRewritePattern<mokey::token::File> {
    using mlir::OpRewritePattern<mokey::token::File>::OpRewritePattern;

    mlir::FileLineColLoc getFileLineCol(mlir::PatternRewriter &rewriter,
                                        std::string source,
                                        uint32_t offset) const {
        uint32_t line = 1;
        uint32_t col = 1;
        for (uint32_t i = 0; i < offset; i++) {
            if (source[i] == '\n') {
                line++;
                col = 1;
            } else {
                col++;
            }
        }
        return mlir::FileLineColLoc::get(rewriter.getStringAttr(source), line,
                                         col);
    }

    mlir::TypedValue<ast::NumType> genExpr(mlir::PatternRewriter &rewriter,
                                           AExpr &expr, std::string file_path) const {
        if (std::holds_alternative<AVariable>(expr)) {
            auto v = std::get<AVariable>(expr);
            auto get_op = rewriter.create<ast::Get>(
                getFileLineCol(rewriter,file_path, v.range.first), v.name);

            return get_op.getResult();
        } else if (std::holds_alternative<APlus>(expr)) {
            auto p = std::move(std::get<APlus>(expr));
            auto lhs = genExpr(rewriter, *p.lhs, file_path);
            auto rhs = genExpr(rewriter, *p.rhs, file_path);
            auto plus_op = rewriter.create<ast::Math_Plus>(getFileLineCol(rewriter, file_path, p.range.first), lhs, rhs);

            return plus_op.getResult();
        } else if (std::holds_alternative<AInt>(expr)) {
            auto i = std::get<AInt>(expr);
            auto int_op = rewriter.create<ast::Num_Init>(getFileLineCol(rewriter, file_path, i.range.first), i.value);

            return int_op.getResult();
        }

        // returnなくて死んだらごめん☆
    }

    mlir::LogicalResult
    matchAndRewrite(mokey::token::File file,
                    mlir::PatternRewriter &rewriter) const final {
        auto file_path = file.getFilePath().str();
        auto source = file.getFileSource().str();

        auto [result, package] = parse(source);

        if (mlir::failed(result)) {
            return result;
        }

        rewriter.eraseOp(file);
        rewriter.setInsertionPointToStart(file->getBlock());

        auto package_op = rewriter.create<ast::Package>(
            getFileLineCol(rewriter, file_path, package.range.first),
            package.name.first);

        rewriter.setInsertionPointToStart(&package_op.getBody().emplaceBlock());

        for (auto &def : package.funcs) {
            auto def_op = rewriter.create<ast::Def>(
                getFileLineCol(rewriter,file_path, def.range.first), def.name.first);
            rewriter.setInsertionPointToStart(&def_op.getBody().emplaceBlock());
            for (auto &state : def.states) {
                if (std::holds_alternative<AReturn>(state)) {
                    auto value = std::move(std::get<AReturn>(state));
                    auto expr_op = genExpr(rewriter, *value.expr, file_path);
                    
                    rewriter.create<ast::Return>(getFileLineCol(rewriter, file_path, value.range.first), expr_op);
                    break;
                } else if (std::holds_alternative<ALet>(state)) {
                    auto let = std::move(std::get<ALet>(state));
                    auto expr_op = genExpr(rewriter, *let.expr, file_path);

                    rewriter.create<ast::Let>(getFileLineCol(rewriter, file_path, let.range.first), let.name.first, expr_op);
                }
            }
        }

        return mlir::success();
    }
};

class ConvToLispAST : public impl::ConvToLispASTBase<ConvToLispAST> {
  public:
    using impl::ConvToLispASTBase<ConvToLispAST>::ConvToLispASTBase;
    void runOnOperation() final {
        mlir::RewritePatternSet patterns(&getContext());
        patterns.add<ConvToLispASTRewriter>(&getContext());
        mlir::FrozenRewritePatternSet frozenPatterns(std::move(patterns));
        if (mlir::failed(mlir::applyPatternsAndFoldGreedily(
                getOperation(), std::move(frozenPatterns)))) {
            signalPassFailure();
        }
    }
};

} // namespace
} // namespace mokey::example::lisp