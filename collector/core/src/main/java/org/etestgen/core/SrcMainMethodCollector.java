package org.etestgen.core;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedList;
import java.util.List;
import org.etestgen.util.AST;
import org.etestgen.util.AbstractConfig;
import org.etestgen.util.JavaParserASTExtractor;
import org.etestgen.util.Option;
import org.etestgen.util.TypeResolver;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.github.javaparser.ParseResult;
import com.github.javaparser.StaticJavaParser;
import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.ast.body.CallableDeclaration;
import com.github.javaparser.ast.expr.Expression;
import com.github.javaparser.ast.stmt.ThrowStmt;
import com.github.javaparser.ast.type.ReferenceType;
import com.github.javaparser.printer.lexicalpreservation.LexicalPreservingPrinter;
import com.github.javaparser.utils.SourceRoot;
import com.github.javaparser.ast.NodeList;

/**
 * Extract all methods with throws clauses in the given main src root. Output the method's key
 * (e.g., p1/class#method#(Lp2/Args;)Lp3/Ret;) and the tokens.
 */
public class SrcMainMethodCollector {
    public static class Config extends AbstractConfig {
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

        SourceRoot sourceRoot = new SourceRoot(Paths.get(sConfig.mainSrcRoot),
                StaticJavaParser.getParserConfiguration());
        Callback callback = new Callback();
        try {
            sourceRoot.parse("", callback);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

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
        public List<String> modifiers;

        public Record(String method, String fqCname, List<String> tokens, String method_node,
                List<String> exception, int startLine, int endLine, List<String> modifiers) {
            this.method = method;
            this.fqCName = fqCname;

            this.tokens = tokens;
            this.method_node = method_node;
            this.exceptions = exception;
            this.startLine = startLine;
            this.endLine = endLine;
            this.modifiers = modifiers;
        }
    }

    public static JavaParserASTExtractor sASTExtractor = new JavaParserASTExtractor();
    public static List<Record> records = Collections.synchronizedList(new LinkedList<>());

    public static List<String> extractExceptions(CallableDeclaration<?> method) {
        List<ThrowStmt> throwStmts = method.findAll(ThrowStmt.class);
        List<String> exceptions = new ArrayList<>();
        Integer line_number = -1;

        for (ThrowStmt throwStmt : throwStmts) {
            Expression exceptionExpr = throwStmt.getExpression();
            try {
                String exception =
                        TypeResolver.normalizeType(exceptionExpr.calculateResolvedType());
                // exceptions.add(exception);
                if (throwStmt.getRange().isPresent()) {
                    line_number = throwStmt.getRange().get().begin.line;
                }
                exceptions.add(exception + "@@" + line_number);
            } catch (Throwable e) {
                debug("Failed to resolve type for: " + exceptionExpr.toString());
                if (throwStmt.getRange().isPresent()) {
                    line_number = throwStmt.getRange().get().begin.line;
                }
                exceptions.add(exceptionExpr.toString() + "@@" + line_number);
            }
        }
        return exceptions;
    }

    public static class Visitor extends SrcVisitorBase<SrcVisitorBase.Context> {

        @Override
        public void visitCallableDeclaration(CallableDeclaration<?> n, String name,
                List<String> extraParams, Context ctx) {
            // only record non-abstract methods
            if (n.isAbstract()) {
                return;
            }

            String method_node = LexicalPreservingPrinter.print(n);
            String method = super.getMethodNameDesc(n, name, extraParams, ctx);
            AST ast = sASTExtractor.extract(n);
            List<String> exceptions = extractExceptions(n);
            List<String> modifiers = new ArrayList<>();
            n.getModifiers().forEach(m -> modifiers.add(m.toString().trim()));
            int startLine = n.getRange().get().begin.line;
            int endLine = n.getRange().get().end.line;
            records.add(new Record(method, ctx.fqCName, ast.getTokens(), method_node, exceptions,
                    startLine, endLine, modifiers));
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
                if (localPath.getParent() == null) {
                    context.pName = "";
                } else {
                    context.pName =
                            localPath.getParent().toString().replace(File.separatorChar, '.');
                }
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
