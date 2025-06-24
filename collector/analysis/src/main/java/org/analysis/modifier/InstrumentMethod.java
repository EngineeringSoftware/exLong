package org.analysis.modifier;

import org.analysis.util.Context;

import com.github.javaparser.StaticJavaParser;
import com.github.javaparser.ast.body.ClassOrInterfaceDeclaration;
import com.github.javaparser.ast.body.EnumDeclaration;
import com.github.javaparser.ast.body.MethodDeclaration;
import com.github.javaparser.ast.stmt.BlockStmt;
import com.github.javaparser.ast.stmt.Statement;
import com.github.javaparser.ast.visitor.ModifierVisitor;
import com.github.javaparser.ast.visitor.Visitable;

public class InstrumentMethod extends ModifierVisitor<Context> {
    @Override
    public Visitable visit(MethodDeclaration n, final Context ctx) {
        super.visit(n, ctx);
        if (!n.getBody().isPresent()) {
            return n; // skip methods without a body (e.g., abstract methods)
        }
        if (!ctx.methodNames.contains(n.getNameAsString())) {
            return n; // skip methods not in the list
        }
        BlockStmt body = n.getBody().get();
        // return type of Thread.currentThread().getStackTrace() is StackTraceElement[]
        String line = "formatStackTrace(Thread.currentThread().getStackTrace())";
        // java.nio.file.Files.write(path, content, StandardOpenOption.CREATE,
        // StandardOpenOption.APPEND);
        String printStmt = "try {" +
                "    java.nio.file.Files.write(java.nio.file.Paths.get(\"" + ctx.logFilePath + "\"), (" + line
                + ").getBytes(), java.nio.file.StandardOpenOption.CREATE, java.nio.file.StandardOpenOption.APPEND);" +
                "} catch (java.io.IOException jioe) {" +
                "    jioe.printStackTrace();" +
                "}";
        Statement tryStmt = StaticJavaParser.parseStatement(printStmt);
        body.addStatement(0, tryStmt);
        return n;
    }

    @Override
    public Visitable visit(final ClassOrInterfaceDeclaration n, final Context arg) {
        // add the formatStackTrace method into the outer class
        super.visit(n, arg);
        if (n.isTopLevelType()) {
            ModifierUtils.printStackTrace(n);
        }
        return n;
    }

    @Override
    public Visitable visit(final EnumDeclaration n, final Context arg) {
        super.visit(n, arg);
        if (n.isTopLevelType()) {
            ModifierUtils.printStackTrace(n);
        }
        return n;
    }
}
