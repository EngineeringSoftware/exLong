package org.etestgen.core;

import java.io.File;
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
import com.github.javaparser.printer.configuration.PrettyPrinterConfiguration;
import com.github.javaparser.utils.SourceRoot;

/**
 * Scan the given test source root, and collect all test methods. In addition, each test method is classified into an exceptional-behavior test or a non-exceptional-behavior test, and its test method context is collected.
 */
class SrcTestScanner {

    public static class Config extends AbstractConfig {
        @Option
        public String mainSrcRoot;
        @Option
        public String testSrcRoot;
        @Option
        public String classpath;
        @Option
        public Path outPath;
        @Option
        public Path debugPath;
        @Option
        public boolean noCContext = false;
    }

    public static Config sConfig;

    public static void main(String... args) {
        Path configPath = Paths.get(args[0]);
        sConfig = AbstractConfig.load(configPath, Config.class);
        debug(sConfig.outPath.toString());
        action();
    }

    public static void action() {
        TypeResolver.setup(sConfig.classpath, sConfig.mainSrcRoot, sConfig.testSrcRoot);

        SourceRoot sourceRoot = new SourceRoot(
            Paths.get(sConfig.testSrcRoot), StaticJavaParser.getParserConfiguration());
        Callback callback = new Callback();
        try {
            sourceRoot.parse("", callback);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        List<Record> records = callback.records;

        // save records
        ObjectMapper mapper = new ObjectMapper();
        try {
            mapper.writeValue(sConfig.outPath.toFile(), records);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    public static class Record {
        public String cname;
        public String mname;
        public String exception; // null if not exceptional behavior test
        public String pattern; // null if not exceptional behavior test
        public String comment;
        public String commentSummary;
        public AST ast; // TODO: cannot output AST because the transformations done in SrcTestScannerVisitor won't be reflected in ASTVisitor
        public String code;
        public String raw_code;
        public String ccontext;

        public Record(String cname, String mname, String exception, String pattern, AST ast,
            String comment, String commentSummary, String code, String raw_code) {
            this.cname = cname;
            this.mname = mname;
            this.exception = exception;
            this.pattern = pattern;
            this.ast = ast;
            this.raw_code = raw_code;
            this.code = code;
            this.comment = comment;
            this.commentSummary = commentSummary;
        }
    }

    public static class Callback implements SourceRoot.Callback {
        public List<Record> records = Collections.synchronizedList(new LinkedList<>());
        public SrcTestScannerVisitor visitor = new SrcTestScannerVisitor();
        public TestClassContextModifier ccontextModifier = new TestClassContextModifier();

        @Override
        public Result process(Path localPath, Path absolutePath,
            ParseResult<CompilationUnit> result) {
            if (result.isSuccessful()) {
                CompilationUnit cu = result.getResult().get();
                Context context = new Context();
                if (localPath.getParent() == null) {
                    context.pName = "";
                } else {
                    context.pName =
                        localPath.getParent().toString().replace(File.separatorChar, '.');
                }
                visitor.visit(cu, context);

                if (!context.records.isEmpty()) {
                    if (!sConfig.noCContext) {
                        cu = (CompilationUnit) cu.accept(
                            ccontextModifier, new TestClassContextModifier.Context(
                                localPath.getFileName().toString().replace(".java", "")));
                        for (Record record : context.records) {
                            record.ccontext = cu.toString(new PrettyPrinterConfiguration());
                            record.ccontext = cu.toString(new PrettyPrinterConfiguration());
                        }
                    }
                    records.addAll(context.records);
                }
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
