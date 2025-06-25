package org.etestgen.core;

import com.github.javaparser.StaticJavaParser;
import java.io.IOException;
import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.ast.body.ClassOrInterfaceDeclaration;
import com.github.javaparser.ast.body.MethodDeclaration;

import org.etestgen.util.AST;
import org.etestgen.util.ExtractASTVisitor;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.file.Path;
import java.util.LinkedList;
import java.util.List;
import com.fasterxml.jackson.databind.ObjectMapper;


class SrcTestASTParser {

    public static List<AST> asts = new LinkedList<>();

    public static void main(String... args) throws Exception {
        String methodFile = args[0];
        String outPath = args[1];
        Path filePath = Paths.get(methodFile);
        List<String> methodStr = Files.readAllLines(filePath);
        parseMethod(methodStr, outPath);

    }

    public static void parseMethod(List<String> method, String outPath) {

        StringBuilder sb = new StringBuilder();
        for (String s : method) {
            sb.append(s);
            sb.append("\n");
        }
        CompilationUnit cu = StaticJavaParser.parse("class DummyClass{" + sb.toString() + "}");

        ClassOrInterfaceDeclaration dummyClass = cu.getClassByName("DummyClass").get();

        for (MethodDeclaration node : dummyClass.getMethods()) {

            ExtractASTVisitor astVisitor = new ExtractASTVisitor();
            AST ast = node.accept(astVisitor, new ExtractASTVisitor.Context());
            asts.add(ast);
        }
        // save parsed asts
        ObjectMapper mapper = new ObjectMapper();
        try {
            mapper.writeValue(Paths.get(outPath).toFile(), asts.get(0));
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

}
