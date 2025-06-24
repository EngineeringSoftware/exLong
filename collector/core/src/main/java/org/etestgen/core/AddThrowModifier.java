package org.etestgen.core;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.Collections;
import java.util.LinkedList;
import java.util.List;
import org.etestgen.core.SrcTestScannerVisitor.Context;
import org.etestgen.util.AST;
import org.etestgen.util.AbstractConfig;
import org.etestgen.util.Option;
import org.etestgen.util.TypeResolver;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.github.javaparser.ParseResult;
import com.github.javaparser.StaticJavaParser;
import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.ast.body.MethodDeclaration;
import com.github.javaparser.ast.visitor.VoidVisitorAdapter;
import com.github.javaparser.utils.SourceRoot;
import org.etestgen.core.AddThrowsExceptionVisitor;

/**
 * Scan the given test source root, and collect all test methods. In addition, each test method is classified into an exceptional-behavior test or a non-exceptional-behavior test, and its test method context is collected.
 */
class AddThrowModifier {

    public static class Config extends AbstractConfig {
        @Option
        public String inFile;
        @Option
        public String outPath;
    }

    public static Config sConfig;

    public static void main(String... args) {
        Path configPath = Paths.get(args[0]);
        sConfig = AbstractConfig.load(configPath, Config.class);
        try {
            // Load the Java file
            FileInputStream in = new FileInputStream(sConfig.inFile);
            CompilationUnit cu = StaticJavaParser.parse(in);

            // Create a custom visitor to modify the code
            AddThrowsExceptionVisitor visitor = new AddThrowsExceptionVisitor();
            visitor.visit(cu, null);

            // Save the modified code
            FileOutputStream out = new FileOutputStream(sConfig.outPath);
            out.write(cu.toString().getBytes());
            out.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

}
