package org.analysis.collector;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;

import org.analysis.util.Context;

import com.github.javaparser.ast.body.ConstructorDeclaration;
import com.github.javaparser.ast.body.MethodDeclaration;
import com.github.javaparser.ast.visitor.VoidVisitorAdapter;

public class CollectMethodBody extends VoidVisitorAdapter<Context>{
    @Override
    public void visit(final MethodDeclaration n, final Context ctx){
        if (n.getNameAsString().equals(ctx.methodName)) {
            String method = n.getDeclarationAsString() + n.getBody().get().toString();
            // save the method to the output file
            try {
                Files.write(Paths.get(ctx.outputFilePath), method.getBytes());
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        }
    }
    
    @Override
    public void visit(final ConstructorDeclaration n, final Context ctx) {
        if (n.getNameAsString().equals(ctx.methodName)) {
            String method = n.getDeclarationAsString() + n.getBody().toString();
            // save the method to the output file
            try {
                Files.write(Paths.get(ctx.outputFilePath), method.getBytes());
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        }
    }   
}
