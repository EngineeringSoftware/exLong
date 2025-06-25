package org.etestgen.core;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.ArrayList;
import java.util.concurrent.Callable;

import org.etestgen.util.AST;
import org.etestgen.util.AbstractConfig;
import org.etestgen.util.JavaParserASTExtractor;
import org.etestgen.util.Option;
import org.etestgen.util.TypeResolver;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.github.javaparser.ParseProblemException;
import com.github.javaparser.ParseResult;
import com.github.javaparser.StaticJavaParser;
import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.ast.NodeList;
import com.github.javaparser.ast.body.CallableDeclaration;
import com.github.javaparser.ast.body.MethodDeclaration;
import com.github.javaparser.utils.SourceRoot;
import com.sun.org.apache.bcel.internal.classfile.Node;
import com.github.javaparser.ast.stmt.ThrowStmt;
import com.github.javaparser.ast.type.Type;
import com.github.javaparser.ast.expr.Expression;
import com.github.javaparser.ast.expr.ObjectCreationExpr;
import com.github.javaparser.ast.type.ClassOrInterfaceType;
import com.github.javaparser.ast.type.ReferenceType;
import com.github.javaparser.printer.lexicalpreservation.LexicalPreservingPrinter;

/**
 * Extract all methods with throws clauses in the given main src root. Output the method's key
 * (e.g., p1/class#method#(Lp2/Args;)Lp3/Ret;) and the tokens.
 */
public class SrcMainThrowCollector {
    public static class Config extends AbstractConfig {
        @Option
        public String methodString;
        @Option
        public String mainSrcRoot;
        @Option
        public String classpath;
        @Option
        public String testSrcRoot;
        @Option
        public Path outPath;
        @Option
        public Path debugPath;
    }

    public static Config sConfig;

    public static void main(String... args) {
        Path configPath = Paths.get(args[0]);
        sConfig = AbstractConfig.load(configPath, Config.class);
        action();
    }

    // public static LexicalPreservingPrinter lpp = new LexicalPreservingPrinter();

    public static void action() {
        TypeResolver.setup(sConfig.classpath, sConfig.mainSrcRoot, sConfig.testSrcRoot);

        // SourceRoot sourceRoot = new SourceRoot(Paths.get(sConfig.mainSrcRoot),
        // StaticJavaParser.getParserConfiguration());
        CompilationUnit cu;
        String tempJavaClass = "public class TempClass {\n" + sConfig.methodString + "\n}";
        try {
            cu = StaticJavaParser.parse(tempJavaClass);
        } catch (ParseProblemException e) {
            tempJavaClass = "interface TempInterface {\n" + sConfig.methodString + "\n}";
            cu = StaticJavaParser.parse(tempJavaClass);
        }
        Visitor visitor = new Visitor();
        SrcVisitorBase.Context context = new SrcVisitorBase.Context();
        visitor.visit(cu, context);
        // save records
        ObjectMapper mapper = new ObjectMapper();
        try {
            mapper.writeValue(sConfig.outPath.toFile(), records);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    public static class Record {
        public String method;
        public List<String> tokens;
        public String method_node;
        public List<String> exceptions;
        public String fqCName;
        public int startLine;
        public int endLine;

        public Record(String method, String fqCname, List<String> tokens, String method_node,
                List<String> exception, int startLine, int endLine) {
            this.method = method;
            this.fqCName = fqCname;

            this.tokens = tokens;
            this.method_node = method_node;
            this.exceptions = exception;
            this.startLine = startLine;
            this.endLine = endLine;
        }
    }

    public static JavaParserASTExtractor sASTExtractor = new JavaParserASTExtractor();
    public static List<HashMap> records = Collections.synchronizedList(new LinkedList<>());

    public static HashMap<String, List<Integer>> extractExceptions(CallableDeclaration<?> method) {
        List<ThrowStmt> throwStmts = method.findAll(ThrowStmt.class);
        HashMap<String, List<Integer>> exception2Lines = new HashMap<>();
        // Also consider throws keyword
        NodeList<ReferenceType> exceptionTypes = method.getThrownExceptions();
        for (ReferenceType ref : exceptionTypes) {
            try {
                String exception = TypeResolver.normalizeType(ref.resolve());
                if (exception2Lines.containsKey(exception)) {
                    exception2Lines.get(exception).add(-1);
                } else {
                    List<Integer> lines = new ArrayList<>();
                    lines.add(-1);
                    exception2Lines.put(exception, lines);
                }
            } catch (Throwable e) {
                if (exception2Lines.containsKey(ref.toString())) {
                    exception2Lines.get(ref.toString()).add(-1);
                } else {
                    List<Integer> lines = new ArrayList<>();
                    lines.add(-1);
                    exception2Lines.put(ref.toString(), lines);
                }
            }
        }
        for (ThrowStmt throwStmt : throwStmts) {
            Expression exceptionExpr = throwStmt.getExpression();
            try {
                String exception =
                        TypeResolver.normalizeType(exceptionExpr.calculateResolvedType());
                if (throwStmt.getRange().isPresent()) {
                    if (exception2Lines.containsKey(exception)) {
                        exception2Lines.get(exception)
                                .add(throwStmt.getRange().get().begin.line - 2);
                    } else {
                        List<Integer> lines = new ArrayList<>();
                        lines.add(throwStmt.getRange().get().begin.line - 2);
                        exception2Lines.put(exception, lines);
                    }
                }
            } catch (Throwable e) {
                debug("Failed to resolve type for: " + exceptionExpr.toString());
                if (exception2Lines.containsKey(exceptionExpr.toString())) {
                    exception2Lines.get(exceptionExpr.toString())
                            .add(throwStmt.getRange().get().begin.line - 2);
                } else {
                    List<Integer> lines = new ArrayList<>();
                    lines.add(throwStmt.getRange().get().begin.line - 2);
                    exception2Lines.put(exceptionExpr.toString(), lines);
                }
            }
        }
        return exception2Lines;
    }

    public static class Visitor extends SrcVisitorBase<SrcVisitorBase.Context> {

        @Override
        public void visitCallableDeclaration(CallableDeclaration<?> n, String name,
                List<String> extraParams, Context ctx) {
            HashMap<String, List<Integer>> exceptions2Line = extractExceptions(n);
            records.add(exceptions2Line);
        }
    }

    public static class Callback implements SourceRoot.Callback {
        public Visitor visitor = new Visitor();

        @Override
        public Result process(Path localPath, Path absolutePath,
                ParseResult<CompilationUnit> result) {
            if (result.isSuccessful()) {
                CompilationUnit cu = LexicalPreservingPrinter.setup(result.getResult().get());
                SrcVisitorBase.Context context = new SrcVisitorBase.Context();
                visitor.visit(cu, context);
            } else {
                debug("Parsing failed for: " + localPath + ", reason: " + result.getProblems());
            }
            return Result.DONT_SAVE;
        }

    }

    public static void debug(String message) {
        if (sConfig.debugPath != null) {
            message = "[" + SrcTestScanner.class.getSimpleName() + "] " + message;
            if (!message.endsWith("\n")) {
                message += "\n";
            }

            try {
                Files.write(sConfig.debugPath, message.getBytes(), StandardOpenOption.CREATE,
                        StandardOpenOption.APPEND);
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        }
    }
}
