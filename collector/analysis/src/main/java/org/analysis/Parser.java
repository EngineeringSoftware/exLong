package org.analysis;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Arrays;

import javax.sound.midi.Instrument;

import org.analysis.collector.CollectMethodBody;
import org.analysis.collector.CollectMethodSig;
import org.analysis.modifier.CollectException;
import org.analysis.modifier.InstrumentMethod;
import org.analysis.util.Constant;
import org.analysis.util.Context;

import com.github.javaparser.StaticJavaParser;
import com.github.javaparser.ast.CompilationUnit;

class Parser {
    /**
     * Add method of collecting stack trace in
     * 
     * @param srcPath
     * @param outputFilePath
     * @throws IOException
     */
    public void collectException(String srcPath, String outputFilePath, String logFilePath)
            throws IOException {
        CompilationUnit cu = StaticJavaParser.parse(Paths.get(srcPath));
        Context ctx = new Context();
        ctx.srcPath = srcPath;
        ctx.outputFilePath = outputFilePath;
        ctx.logFilePath = logFilePath;
        ctx.task = Constant.TASK_EXCEPTION;
        CollectException visitor = new CollectException();
        cu.accept(visitor, ctx);
        // add import statements
        cu.addImport("java.nio.file.Files");
        cu.addImport("java.nio.file.Paths");
        cu.addImport("java.nio.file.StandardOpenOption");
        Files.write(new File(outputFilePath).toPath(), cu.toString().getBytes());
    }

    /**
     * Collect method given method Name.
     * 
     * @param srcPath
     * @param methodName
     * @param outputFilePath
     * @throws IOException
     */
    public static void collectMethodBody(String srcPath, String methodName, String outputFilePath) throws IOException {
        CompilationUnit cu = StaticJavaParser.parse(Paths.get(srcPath));
        Context ctx = new Context();
        ctx.srcPath = srcPath;
        ctx.methodName = methodName;
        ctx.outputFilePath = outputFilePath;
        ctx.task = Constant.TASK_METHOD;
        CollectMethodBody visitor = new CollectMethodBody();
        cu.accept(visitor, ctx);
    }

    /**
     * Instrument method given method Name.
     * 
     * @param srcPath
     * @param methodName
     * @param logFilePath
     * @throws IOException
     */
    public static void instrumentMethod(String srcPath, String methodName, String logFilePath) throws IOException {
        Context ctx = new Context();
        ctx.srcPath = srcPath;
        ctx.methodNames = Arrays.asList(methodName.split("#"));
        ctx.logFilePath = logFilePath;
        CompilationUnit cu = StaticJavaParser.parse(Paths.get(srcPath));
        InstrumentMethod visitor = new InstrumentMethod();
        cu.accept(visitor, ctx);
        Files.write(new File(srcPath).toPath(), cu.toString().getBytes());
    }

    /**
     * Collect method signature.
     * 
     * @param srcPath
     * @param outputFilePath
     * @throws IOException
     */
    public static void collectMethodSignature(String srcPath, String lineNum, String outputFilePath) throws IOException {
        CompilationUnit cu = StaticJavaParser.parse(Paths.get(srcPath));
        Context ctx = new Context();
        ctx.srcPath = srcPath;
        ctx.lineNum = Integer.parseInt(lineNum);
        ctx.outputFilePath = outputFilePath;
        ctx.task = Constant.TASK_METHOD_SIG;
        CollectMethodSig visitor = new CollectMethodSig();
        cu.accept(visitor, ctx);
    }
}