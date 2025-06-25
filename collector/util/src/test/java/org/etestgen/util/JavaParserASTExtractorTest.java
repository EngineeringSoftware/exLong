package org.etestgen.util;

import java.util.Arrays;
import org.junit.Assert;
import org.junit.Test;
import com.github.javaparser.StaticJavaParser;
import com.github.javaparser.ast.body.MethodDeclaration;
import com.github.javaparser.ast.stmt.BlockStmt;

public class JavaParserASTExtractorTest {


    @Test
    public void testTokenize() {
        MethodDeclaration method = (MethodDeclaration) StaticJavaParser
            .parseBodyDeclaration("public void foo() {System.out.println(\"Hello World\");}");
        JavaParserASTExtractor extractor = new JavaParserASTExtractor();
        AST ast = extractor.extract(method);
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
        JavaParserASTExtractor extractor = new JavaParserASTExtractor();
        AST ast = extractor.extract(method);
        Assert.assertEquals(
            Arrays.asList("public", "void", "foo", "(", ")", "{", "}"), ast.getTokens());
    }

    @Test
    public void testTokenizeStmts() {
        BlockStmt block =
            StaticJavaParser.parseBlock("{" + "a(); System.out.println(\"Hello World\");" + "}");
        JavaParserASTExtractor extractor = new JavaParserASTExtractor();
        AST ast = extractor.extract(block);
        Assert.assertEquals(
            Arrays.asList(
                "{", "a", "(", ")", ";", "System", ".", "out", ".", "println", "(",
                "\"Hello World\"", ")", ";", "}"),
            ast.getTokens());
    }
}
