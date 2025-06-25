package org.etestgen.core;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.Collections;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import org.etestgen.util.AST;
import org.etestgen.util.AbstractConfig;
import org.etestgen.util.JavaParserASTExtractor;
import org.etestgen.util.Option;
import org.etestgen.util.TypeResolver;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.JsonMappingException.Reference;
import com.github.javaparser.ParseResult;
import com.github.javaparser.StaticJavaParser;
import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.ast.body.CallableDeclaration;
import com.github.javaparser.ast.body.MethodDeclaration;
import com.github.javaparser.ast.stmt.BlockStmt;
import com.github.javaparser.ast.stmt.IfStmt;
import com.github.javaparser.ast.stmt.Statement;
import com.github.javaparser.ast.stmt.ThrowStmt;
import com.github.javaparser.ast.type.ReferenceType;
import com.github.javaparser.utils.SourceRoot;
import com.github.javaparser.ast.expr.ObjectCreationExpr;
import com.github.javaparser.ast.Node;

public class SrcThrowMethodsCollector {
    public static class Config extends AbstractConfig {
        @Option
        public String mainSrcRoot;
        @Option
        public String classpath;
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

    public static void action() {
        TypeResolver.setup(sConfig.classpath, sConfig.mainSrcRoot);

        SourceRoot sourceRoot = new SourceRoot(
            Paths.get(sConfig.mainSrcRoot), StaticJavaParser.getParserConfiguration());
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
        public boolean hasConditions;
        public String exception;

        public Record(String method, boolean hasConditions, String exception) {
            this.method = method;
            this.hasConditions = hasConditions;
            this.exception = exception;
        }
    }

    public static class TContext extends SrcVisitorBase.Context {
        public String method;
        public boolean throwWithCondition;
        public List<String> exceptions = new LinkedList<>();
    }

    public static JavaParserASTExtractor sASTExtractor = new JavaParserASTExtractor();
    public static HashSet<Record> records = new HashSet<Record>();

    public static class Visitor extends SrcVisitorBase<TContext> {


        @Override
        public void visit(IfStmt ifStmt, TContext ctx) {
            super.visit(ifStmt, ctx);

            if (ifStmt.getThenStmt() instanceof BlockStmt) {
                BlockStmt thenBlock = ifStmt.getThenStmt().asBlockStmt();

                // visit each statement within the if block
                int size = thenBlock.getStatements().size();
                for (int i = 0; i < size; ++i) {
                    Statement statement = thenBlock.getStatement(i);
                    if (statement instanceof ThrowStmt) {
                        ThrowStmt ts = (ThrowStmt) statement;
                        Node exception = ts.getExpression();
                        if (exception instanceof ObjectCreationExpr) {
                            ObjectCreationExpr objectCreationExpr = (ObjectCreationExpr) exception;
                            String exceptionName = objectCreationExpr.getType().getNameAsString();
                            if (ctx.exceptions.contains(exceptionName)) {
                                ctx.throwWithCondition = true;
                                records.add(
                                    new Record(ctx.method, ctx.throwWithCondition, exceptionName));
                            }
                        }
                    }
                }
            } else {
                if (ifStmt.getThenStmt() instanceof ThrowStmt) {
                    ThrowStmt ts = (ThrowStmt) ifStmt.getThenStmt();
                    Node exception = ts.getExpression();
                    if (exception instanceof ObjectCreationExpr) {
                        ObjectCreationExpr objectCreationExpr = (ObjectCreationExpr) exception;
                        String exceptionName = objectCreationExpr.getType().getNameAsString();
                        if (ctx.exceptions.contains(exceptionName)) {
                            ctx.throwWithCondition = true;
                            records
                                .add(new Record(ctx.method, ctx.throwWithCondition, exceptionName));
                        }
                    }
                }
            }

            if (ifStmt.getElseStmt().isPresent()) {
                if (ifStmt.getElseStmt().get() instanceof IfStmt) {
                    // visit the else if statement
                    visit((IfStmt) ifStmt.getElseStmt().get(), ctx);
                    return;
                }
                if (ifStmt.getElseStmt().get() instanceof BlockStmt) {
                    BlockStmt elseBlock = ifStmt.getElseStmt().get().asBlockStmt();
                    // visit each statement within the else block
                    int size = elseBlock.getStatements().size();
                    for (int i = 0; i < size; ++i) {
                        Statement statement = elseBlock.getStatement(i);
                        if (statement instanceof ThrowStmt) {
                            ThrowStmt ts = (ThrowStmt) statement;
                            Node exception = ts.getExpression();
                            if (exception instanceof ObjectCreationExpr) {
                                ObjectCreationExpr objectCreationExpr =
                                    (ObjectCreationExpr) exception;
                                String exceptionName =
                                    objectCreationExpr.getType().getNameAsString();
                                if (ctx.exceptions.contains(exceptionName)) {
                                    ctx.throwWithCondition = true;
                                    records.add(
                                        new Record(
                                            ctx.method, ctx.throwWithCondition, exceptionName));
                                }
                            }
                        }
                    }
                } else {
                    if (ifStmt.getElseStmt().get() instanceof ThrowStmt) {
                        ThrowStmt ts = (ThrowStmt) ifStmt.getElseStmt().get();
                        Node exception = ts.getExpression();
                        if (exception instanceof ObjectCreationExpr) {
                            ObjectCreationExpr objectCreationExpr = (ObjectCreationExpr) exception;
                            String exceptionName = objectCreationExpr.getType().getNameAsString();
                            if (ctx.exceptions.contains(exceptionName)) {
                                ctx.throwWithCondition = true;
                                records.add(
                                    new Record(ctx.method, ctx.throwWithCondition, exceptionName));
                            }
                        }
                    }
                }
            }
        }

        @Override
        public void visitCallableDeclaration(CallableDeclaration<?> n, String name,
            List<String> extraParams, TContext ctx) {
            // only record methods with throws
            if (n.getThrownExceptions().isEmpty()) {
                return;
            }
            if (!(n instanceof MethodDeclaration)) {
                return;
            }
            // only record non-abstract methods
            if (n.isAbstract()) {
                return;
            }
            ctx.method = super.getMethodNameDesc(n, name, extraParams, ctx);
            ctx.throwWithCondition = false;
            for (ReferenceType ex : n.getThrownExceptions()) {
                ctx.exceptions.add(ex.toString());
            }

            super.visitCallableDeclaration(n, name, extraParams, ctx);
        }
    }

    public static class Callback implements SourceRoot.Callback {
        public Visitor visitor = new Visitor();

        @Override
        public Result process(Path localPath, Path absolutePath,
            ParseResult<CompilationUnit> result) {
            if (result.isSuccessful()) {
                CompilationUnit cu = result.getResult().get();
                TContext context = new TContext();
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
                Files.write(
                    sConfig.debugPath, message.getBytes(), StandardOpenOption.CREATE,
                    StandardOpenOption.APPEND);
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        }
    }
}
