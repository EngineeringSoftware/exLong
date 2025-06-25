package org.etestgen.core;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import org.etestgen.util.AST;
import org.etestgen.util.AbstractConfig;
import org.etestgen.util.JavaParserASTExtractor;
import org.etestgen.util.Option;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.github.javaparser.StaticJavaParser;
import com.github.javaparser.ast.expr.Expression;
import com.github.javaparser.ast.expr.MethodCallExpr;
import com.github.javaparser.ast.stmt.BlockStmt;
import com.github.javaparser.ast.stmt.ExpressionStmt;
import com.github.javaparser.ast.visitor.ModifierVisitor;
import com.github.javaparser.ast.visitor.Visitable;

/**
 * Given the input of a list of test method body (does not include), remove all assertion statements, remove beginning and ending brackets, then tokenize.
 */
public class SrcTestMethodProcessor {
    public static class Config extends AbstractConfig {
        @Option
        public Path inPath;
        @Option
        public Path outPath;
        @Option
        public Path debugPath;
    }

    public static Config sConfig;

    public static void main(String... args) throws Exception {
        Path configPath = Paths.get(args[0]);
        sConfig = AbstractConfig.load(configPath, Config.class);
        action();
    }

    public static void action() throws Exception {
        ObjectMapper mapper = new ObjectMapper();
        List<String> inputs = null;
        inputs =
            mapper.readValue(sConfig.inPath.toFile(), new TypeReference<ArrayList<String>>() {});

        // RemoveAssertionModifier modifier = new RemoveAssertionModifier();
        JavaParserASTExtractor extractor = new JavaParserASTExtractor();
        List<List<String>> outputs = new LinkedList<>();
        for (String input : inputs) {
            BlockStmt block = StaticJavaParser.parseBlock(input);
            // block = (BlockStmt) block.accept(modifier, null);
            AST ast = extractor.extract(block);
            List<String> tokens = ast.getTokens();
            tokens.remove(0);
            tokens.remove(tokens.size() - 1);
            outputs.add(tokens);
        }

        // save records
        mapper.writeValue(sConfig.outPath.toFile(), outputs);
    }

    public static class RemoveAssertionModifier extends ModifierVisitor<Void> {
        @Override
        public Visitable visit(ExpressionStmt n, Void arg) {
            Expression expr = n.getExpression();
            if (expr.isMethodCallExpr()) {
                MethodCallExpr methodCallExpr = expr.asMethodCallExpr();
                String methodName = methodCallExpr.getNameAsString();

                // detect if this is an assertion method
                if (methodName.startsWith("assert")) {
                    if (methodCallExpr.getScope().isPresent()) {
                        String scope = methodCallExpr.getScope().get().toString();
                        if (scope.equals("org.junit.Assert") || scope.equals("Assert")
                            || scope.equals("org.junit.jupiter.api.Assertions")
                            || scope.equals("Assertions")) {
                            // is [org.junit.]Assert.assert* or [org.junit.jupiter.api.]Assertions.assert*, ok
                            return null;
                        } else {
                            // not an assertion method
                            return super.visit(n, arg);
                        }
                    }
                    return null;
                }

                // is an assertion method
                // the arguments in the assertion method may contain method call, but they should usually be free of side effects; revisit if we should keep them if problem
                // return null;
            }
            return super.visit(n, arg);
        }
    }
}
