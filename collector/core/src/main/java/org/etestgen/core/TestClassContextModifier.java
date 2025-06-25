package org.etestgen.core;

import java.util.Set;
import java.util.stream.Collectors;
import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.ast.body.ClassOrInterfaceDeclaration;
import com.github.javaparser.ast.body.ConstructorDeclaration;
import com.github.javaparser.ast.body.MethodDeclaration;
import com.github.javaparser.ast.comments.BlockComment;
import com.github.javaparser.ast.comments.Comment;
import com.github.javaparser.ast.comments.JavadocComment;
import com.github.javaparser.ast.comments.LineComment;
import com.github.javaparser.ast.expr.AnnotationExpr;
import com.github.javaparser.ast.type.ClassOrInterfaceType;
import com.github.javaparser.ast.visitor.ModifierVisitor;
import com.github.javaparser.ast.visitor.Visitable;

public class TestClassContextModifier extends ModifierVisitor<TestClassContextModifier.Context> {

    public static class Context {
        String className;

        public String getModifiedClassName() {
            return "adhoc_" + className;
        }

        public Context(String className) {
            this.className = className;
        }
    }

    protected static final String PLACEHOLDER = "TEST PLACEHOLDER";

    @Override
    public Visitable visit(CompilationUnit n, Context arg) {
        // remove all existing comments before moving on
        n.getAllContainedComments().forEach(Comment::remove);

        return super.visit(n, arg);
    }

    @Override
    public Visitable visit(ClassOrInterfaceDeclaration n, Context arg) {
        n = (ClassOrInterfaceDeclaration) super.visit(n, arg);
        
        if (arg.className.equals(n.getNameAsString())) {
            n.setName(arg.getModifiedClassName());
            // add a placeholder comment
            n.addOrphanComment(new BlockComment(PLACEHOLDER));
        }

        return n;
    }

    @Override
    public Visitable visit(ClassOrInterfaceType n, Context arg) {
        if (n.getNameAsString().equals(arg.className)) {
            n.setName(arg.getModifiedClassName());
        }
        return super.visit(n, arg);
    }

    @Override
    public Visitable visit(ConstructorDeclaration n, Context arg) {
        if (n.getNameAsString().equals(arg.className)) {
            n.setName(arg.getModifiedClassName());
        }
        return super.visit(n, arg);
    }

    @Override
    public Visitable visit(MethodDeclaration n, Context arg) {
        Set<String> annotations = n.getAnnotations().stream().map(AnnotationExpr::getNameAsString)
            .collect(Collectors.toSet());
        if (annotations.contains("Test")) {
            // remove test methods
            return null;
        } else {
            // keep all other methods (could be utility methods)
            return super.visit(n, arg);
        }
    }

    @Override
    public Visitable visit(JavadocComment n, Context arg) {
        // remove comment
        return null;
    }

    @Override
    public Visitable visit(BlockComment n, Context arg) {
        // remove comment
        return null;
    }

    @Override
    public Visitable visit(LineComment n, Context arg) {
        // remove comment
        return null;
    }
}
