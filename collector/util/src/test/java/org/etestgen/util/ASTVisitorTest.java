package org.etestgen.util;

import java.util.Arrays;
import org.junit.Assert;
import org.junit.Test;
import com.github.javaparser.StaticJavaParser;
import com.github.javaparser.ast.body.MethodDeclaration;
import com.github.javaparser.ast.stmt.BlockStmt;

public class ASTVisitorTest {

    @Test
    public void testTokenize() {
        MethodDeclaration method = (MethodDeclaration) StaticJavaParser
            .parseBodyDeclaration("public void foo() {System.out.println(\"Hello World\");}");
        ExtractASTVisitor visitor = new ExtractASTVisitor();
        AST ast = method.accept(visitor, new ExtractASTVisitor.Context());
        Assert.assertEquals(
            Arrays.asList(
                "public", "void", "foo", "(", ")", "{", "System", ".", "out", ".", "println", "(",
                "\"Hello World\"", ")", ";", "}"),
            ast.getTokens());
    }

    @Test
    public void testTokenizeModifiedAST() {
        MethodDeclaration method = (MethodDeclaration) StaticJavaParser
            .parseBodyDeclaration("public void foo() {System.out.println(\"Hello World\");}");
        method.setBody(new BlockStmt());
        ExtractASTVisitor visitor = new ExtractASTVisitor();
        AST ast = method.accept(visitor, new ExtractASTVisitor.Context());
        // using ASTVisitor to get tokens does not account for modifications made to AST
        Assert.assertNotEquals(
            Arrays.asList("public", "void", "foo", "(", ")", "{", "}"), ast.getTokens());
        Assert.assertEquals(
            Arrays.asList(
                "public", "void", "foo", "(", ")", "{", "System", ".", "out", ".", "println", "(",
                "\"Hello World\"", ")", ";", "}"),
            ast.getTokens());
    }
}
