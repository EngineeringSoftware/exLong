package org.analysis.collector;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;

import org.analysis.util.Context;

import com.github.javaparser.ast.body.ConstructorDeclaration;
import com.github.javaparser.ast.body.MethodDeclaration;
import com.github.javaparser.ast.visitor.VoidVisitorAdapter;

public class CollectMethodSig extends VoidVisitorAdapter<Context> {
    @Override
    public void visit(final MethodDeclaration n, final Context ctx) {
        int startLine = n.getRange().get().begin.line;
        int endLine = n.getRange().get().end.line;
        if (ctx.lineNum >= startLine && ctx.lineNum <= endLine) {
            String line = n.getDeclarationAsString(false, false);
            // save the method to the output file
            try {
                Files.write(Paths.get(ctx.outputFilePath), line.getBytes(), StandardOpenOption.CREATE);
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        }
    }

    @Override
    public void visit(final ConstructorDeclaration n, final Context ctx) {
        int startLine = n.getRange().get().begin.line;
        int endLine = n.getRange().get().end.line;
        if (ctx.lineNum >= startLine && ctx.lineNum <= endLine) {
            String line = n.getDeclarationAsString(false, false);
            // save the method to the output file
            try {
                Files.write(Paths.get(ctx.outputFilePath), line.getBytes(), StandardOpenOption.CREATE);
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        }
    }
}