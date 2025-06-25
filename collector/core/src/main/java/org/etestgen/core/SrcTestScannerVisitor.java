package org.etestgen.core;

import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import org.apache.commons.collections4.map.LazyMap;
import org.etestgen.core.SrcTestScanner.Record;
import org.etestgen.util.ExtractASTVisitor;
import org.etestgen.util.TypeResolver;
import org.etestgen.util.AST;
import com.github.javaparser.ast.Modifier;
import com.github.javaparser.ast.NodeList;
import com.github.javaparser.ast.body.AnnotationDeclaration;
import com.github.javaparser.javadoc.Javadoc;
import com.github.javaparser.ast.body.CallableDeclaration;
import com.github.javaparser.ast.body.ClassOrInterfaceDeclaration;
import com.github.javaparser.ast.body.ConstructorDeclaration;
import com.github.javaparser.ast.body.EnumDeclaration;
import com.github.javaparser.ast.body.MethodDeclaration;
import com.github.javaparser.ast.expr.AnnotationExpr;
import com.github.javaparser.ast.expr.ClassExpr;
import com.github.javaparser.ast.expr.Expression;
import com.github.javaparser.ast.expr.MemberValuePair;
import com.github.javaparser.ast.expr.MethodCallExpr;
import com.github.javaparser.ast.expr.NameExpr;
import com.github.javaparser.ast.comments.Comment;
import com.github.javaparser.ast.comments.LineComment;
import com.github.javaparser.ast.expr.NormalAnnotationExpr;
import com.github.javaparser.ast.expr.LambdaExpr;
import com.github.javaparser.ast.expr.ObjectCreationExpr;
import com.github.javaparser.ast.nodeTypes.NodeWithTypeParameters;
import com.github.javaparser.ast.stmt.CatchClause;
import com.github.javaparser.ast.stmt.ExpressionStmt;
import com.github.javaparser.ast.stmt.Statement;
import com.github.javaparser.ast.stmt.BlockStmt;
import com.github.javaparser.ast.stmt.TryStmt;
import com.github.javaparser.ast.type.ClassOrInterfaceType;
import com.github.javaparser.ast.type.TypeParameter;
import com.github.javaparser.ast.visitor.VoidVisitorAdapter;

// class MethodCallVisitor extends VoidVisitorAdapter<Void> {

// public String pattern = null;
// public String exception = null;
// public boolean isAssertThrows = false;

// @Override
// public void visit(MethodCallExpr n, Void arg) {
// // Found a method call
// String methodName = n.getNameAsString();
// if (methodName.equals("assertThrows")) {
// pattern = "assertThrows";
// exception = n.getArgument(0).toString();
// isAssertThrows = true;
// }
// super.visit(n, arg);
// }
// }

// TODO: rewrite to extend SrcVisitorBase
public class SrcTestScannerVisitor extends VoidVisitorAdapter<SrcTestScannerVisitor.Context> {

    public static class Context implements Cloneable {
        /** package name (could be empty) */
        String pName = "";

        /** class name */
        String cName = null;

        /** fully qualified class name = package name (if present) . class name */
        String fqCName = null;

        /**
         * anonymous class count tracker should be freshed for each new class context
         */
        int anonymousClassCount = 1;

        /**
         * local class count tracker not deep cloned: should be refreshed for each new
         * class context
         */
        LazyMap<String, Integer> localClassCount = LazyMap.lazyMap(new HashMap<>(), () -> 1);

        /**
         * additional constructor parameters tracker deep cloned
         */
        List<String> extraInitParams = new LinkedList<>();

        /**
         * type parameters mapping tracker deep cloned
         */
        Map<String, String> typeParams = new HashMap<>();

        /**
         * results shared
         */
        List<Record> records = new LinkedList<>();

        @Override
        public Context clone() {
            try {
                Context clone = (Context) super.clone();
                clone.typeParams = new HashMap<>(typeParams);
                clone.extraInitParams = new LinkedList<>(extraInitParams);
                return clone;
            } catch (CloneNotSupportedException e) {
                throw new RuntimeException(e);
            }
        }
    }

    @Override
    public void visit(ClassOrInterfaceDeclaration n, Context ctx) {
        String name = null;
        if (n.isLocalClassDeclaration()) {
            // local class is named as OuterClass$%dInnerClass
            int cnt = ctx.localClassCount.get(n.getNameAsString());
            name = cnt + n.getNameAsString();
            ctx.localClassCount.put(n.getNameAsString(), cnt + 1);
        } else {
            name = n.getNameAsString();
        }

        List<String> extraInitParams = null;
        if (ctx.cName != null && !n.isStatic()) {
            // non-static inner class has extra init parameter OuterClass
            extraInitParams = Arrays.asList(ctx.fqCName);
        }

        ctx = getNewContextForTypeDeclaration(name, extraInitParams, ctx);
        registerTypeParameters(n, ctx);
        // AllCollectors.warning("visit(ClassOrInterfaceDeclaration) " + ctx.fqCName);
        super.visit(n, ctx);
    }

    @Override
    public void visit(EnumDeclaration n, Context ctx) {
        // enum constructor has additional parameters String,int
        super.visit(
                n, getNewContextForTypeDeclaration(
                        n.getNameAsString(), Arrays.asList("java.lang.String", "int"), ctx));
    }

    @Override
    public void visit(AnnotationDeclaration n, Context ctx) {
        super.visit(n, getNewContextForTypeDeclaration(n.getNameAsString(), ctx));
    }

    @Override
    public void visit(ObjectCreationExpr n, Context ctx) {
        n.getAnonymousClassBody().ifPresent(l -> {
            Context newCtx = getNewContextForTypeDeclaration(String.valueOf(ctx.anonymousClassCount), ctx);
            ++ctx.anonymousClassCount;
            l.forEach(v -> v.accept(this, newCtx));
        });

        n.getArguments().forEach(p -> p.accept(this, ctx));
        n.getScope().ifPresent(l -> l.accept(this, ctx));
        n.getType().accept(this, ctx);
        n.getTypeArguments().ifPresent(l -> l.forEach(v -> v.accept(this, ctx)));
        n.getComment().ifPresent(l -> l.accept(this, ctx));
    }

    protected Context getNewContextForTypeDeclaration(String name, Context ctx) {
        return getNewContextForTypeDeclaration(name, null, ctx);
    }

    protected Context getNewContextForTypeDeclaration(String name, List<String> extraInitParams,
            Context ctx) {
        String newCName = "";
        if (ctx.cName != null) {
            // inner class
            newCName += ctx.cName + "$";
        }

        newCName += name;

        Context newCtx = ctx.clone();
        newCtx.cName = newCName;
        newCtx.anonymousClassCount = 1;
        newCtx.localClassCount = LazyMap.lazyMap(new HashMap<>(), () -> 1);
        if (extraInitParams != null) {
            newCtx.extraInitParams.addAll(extraInitParams);
        }

        if (newCtx.pName.isEmpty()) {
            newCtx.fqCName = newCName;
        } else {
            newCtx.fqCName = newCtx.pName + "." + newCName;
        }

        return newCtx;
    }

    protected void registerTypeParameters(NodeWithTypeParameters<?> n, Context ctx) {
        NodeList<TypeParameter> typeParams = n.getTypeParameters();
        if (typeParams != null && !typeParams.isEmpty()) {
            for (TypeParameter typeParameter : typeParams) {
                NodeList<ClassOrInterfaceType> typeBound = typeParameter.getTypeBound();
                String mapTo = "java.lang.Object";
                if (typeBound != null && !typeBound.isEmpty()) {
                    mapTo = TypeResolver.resolveType(typeBound.get(0), ctx.typeParams);
                }
                ctx.typeParams.put(typeParameter.getNameAsString(), mapTo);
            }
        }
    }

    @Override
    public void visit(MethodDeclaration n, Context ctx) {
        if (!n.getTypeParameters().isEmpty()) {
            ctx = ctx.clone();
            registerTypeParameters(n, ctx);
        }

        visitCallableDeclaration(n, n.getNameAsString(), ctx);
        super.visit(n, ctx);
    }

    @Override
    public void visit(ConstructorDeclaration n, Context ctx) {
        if (!n.getTypeParameters().isEmpty()) {
            ctx = ctx.clone();
            registerTypeParameters(n, ctx);
        }

        visitCallableDeclaration(n, "<init>", ctx.extraInitParams, ctx);
        super.visit(n, ctx);
    }

    private void visitCallableDeclaration(CallableDeclaration<?> n, String name, Context ctx) {
        visitCallableDeclaration(n, name, null, ctx);
    }

    public static ExtractASTVisitor astVisitor = new ExtractASTVisitor();

    private void visitCallableDeclaration(CallableDeclaration<?> n, String name,
            List<String> extraParams, Context ctx) {
        // find tests with exceptional exceptions
        if (!(n instanceof MethodDeclaration)) {
            return;
        }

        MethodDeclaration md = (MethodDeclaration) n;
        String method = ctx.fqCName + "." + name;
        String comment = "";
        String commentSummary = "";
        // String lineComments = "";

        // must have @Test annotation
        AnnotationExpr testAnnotation = md.getAnnotationByName("Test").orElse(null);
        if (testAnnotation == null) {
            // SrcTestScanner.debug(method + ": not a test because no @Test annotation");
            return;
        }
        // must have body
        if (md.getBody() == null) {
            SrcTestScanner.debug(method + ": not a test because no body");
            return;
        }
        // signature must be void()
        if (!md.getTypeAsString().equals("void") || !md.getParameters().isEmpty()) {
            SrcTestScanner.debug(method + ": not a test because not void()");
            return;
        }
        // must be public
        if (!md.getModifiers().contains(Modifier.publicModifier())) {
            SrcTestScanner.debug(method + ": not a test because not public");
            return;
        }

        // get javadoc
        if (md.getJavadoc().isPresent()) {
            Javadoc javadoc = md.getJavadoc().get();
            comment = javadoc.toText();

            // If javadoc is not English or no description part, the summary is empty string
            if (javadoc.getDescription().toText().trim().length() == 0) {
                commentSummary = "";
            } else {
                commentSummary = NLPUtils.getFirstSentence(javadoc.getDescription().toText()).orElse(null);
            }
        }

        // try to detect patterns for exception test
        String exception = null;
        String pattern = null;

        // @Test(expected)
        if (testAnnotation instanceof NormalAnnotationExpr) {
            for (MemberValuePair pair : ((NormalAnnotationExpr) testAnnotation).getPairs()) {
                if (pair.getNameAsString().equals("expected")) {
                    exception = TypeResolver.resolveType(pair.getValue().asClassExpr().getType());
                    pattern = "@Test(expected)";
                }
            }
        }

        // assertThrows. JUnit 5 Jupiter assertions API introduces the assertThrows
        // method for asserting exceptions.;
        if (pattern == null && md.getBody().toString().contains("assertThrows")) {
            pattern = "assertThrows";
            final BlockStmt[] lambdaBlockStmt = { null };
            final String[] exceptionHolder = { null };
            int nStatements = md.getBody().get().getStatements().size();
            n.findAll(MethodCallExpr.class).forEach(methodCallExpr -> {
                if (methodCallExpr.getNameAsString().equals("assertThrows")) {
                    Expression argument = methodCallExpr.getArguments().get(0);
                    if ((argument instanceof ClassExpr)) {
                        exceptionHolder[0] = TypeResolver.resolveType(((ClassExpr) argument).getType());
                    }
                    // extract code block
                    if (methodCallExpr.getArguments().get(1) instanceof LambdaExpr) {
                        if (methodCallExpr.getArguments().get(1).asLambdaExpr().getBody().isBlockStmt()) {
                            lambdaBlockStmt[0] = methodCallExpr.getArguments().get(1).asLambdaExpr().getBody()
                                    .asBlockStmt();
                        }
                    }
                }
            });
            exception = exceptionHolder[0];
            BlockStmt blockStmt = lambdaBlockStmt[0];
            SrcTestScanner.debug(
                    "method " + method + ": transforming code from: "
                            + md.getBody().get().toString());
            if (blockStmt != null) {
                int blockstmts = blockStmt.getStatements().size();
                // modify
                for (int i = 0; i < blockStmt.getStatements().size(); i++) {
                    md.getBody().get().getStatements().add(i, blockStmt.getStatement(i));
                }
                int last_stmts = md.getBody().get().getStatements().size();
                for (int i = last_stmts - 1; i >= blockstmts; --i) {
                    md.getBody().get().getStatements().remove(i);
                }
            }
            SrcTestScanner.debug(
                    "method " + method + ": transforming code from: "
                            + md.getBody().get().toString());
        }

        // rule expect(exception)
        if (pattern == null) {
            int found = -1;
            String exceptionRule = "";
            int size = md.getBody().get().getStatements().size();
            for (int i = 0; i < size; ++i) {
                Statement statement = md.getBody().get().getStatement(i);
                if (statement instanceof ExpressionStmt) {
                    ExpressionStmt expressionStmt = (ExpressionStmt) statement;
                    if (expressionStmt.getExpression() instanceof MethodCallExpr) {
                        MethodCallExpr methodCallExpr = (MethodCallExpr) expressionStmt.getExpression();
                        // check if the method expectedException.expect(
                        if (methodCallExpr.getNameAsString().equals("expect")) {
                            Expression argument = methodCallExpr.getArguments().get(0);
                            if (!(argument.isClassExpr())) {
                                continue;
                            }
                            if (methodCallExpr.getScope().isPresent()
                                    && methodCallExpr.getScope().get().isNameExpr()) {
                                exceptionRule = methodCallExpr.getScope().get().asNameExpr().getNameAsString();
                            }
                            exception = TypeResolver.resolveType(((ClassExpr) argument).getType());
                            pattern = "expectedException.expect()";
                            found = i;
                            break;
                        }
                    }
                }
            }
            // modify
            if (found != -1) {
                SrcTestScanner.debug(
                        "exception rule is: " + exceptionRule);
                SrcTestScanner.debug(
                        "method " + method + ": transforming code from: "
                                + md.getBody().get().toString()); // transform code to remove try-catch
                NodeList<Statement> newStatements = new NodeList<>();

                for (int i = 0; i < size; i++) {
                    if (md.getBody().get().getStatement(i).isExpressionStmt()) {
                        ExpressionStmt expressionStmt = md.getBody().get().getStatement(i).asExpressionStmt();
                        if (expressionStmt.getExpression() instanceof MethodCallExpr) {
                            MethodCallExpr methodCallExpr = (MethodCallExpr) expressionStmt.getExpression();
                            if (methodCallExpr.getScope().isPresent()
                                    && methodCallExpr.getScope().get().isNameExpr()) {
                                if (methodCallExpr.getScope().get().asNameExpr().getNameAsString()
                                        .equals(exceptionRule)) {
                                    // md.getBody().get().getStatements().remove(i);
                                    continue;
                                }
                            }
                        }
                    }
                    newStatements.add(md.getBody().get().getStatement(i));
                }
                BlockStmt newBody = new BlockStmt(newStatements);
                md.setBody(newBody);
                SrcTestScanner.debug("transforming code to: " + md.getBody().get().toString());
            }

        }

        // try{fail}catch
        if (pattern == null) {
            int found = -1;
            int size = md.getBody().get().getStatements().size();
            for (int i = 0; i < size; ++i) {
                Statement statement = md.getBody().get().getStatement(i);
                if (statement instanceof TryStmt) {
                    TryStmt tryStmt = (TryStmt) statement;
                    if (tryStmt.getCatchClauses().size() != 1) {
                        // only recognize single catch
                        SrcTestScanner
                                .debug(method + ": try stmt not single catch: " + tryStmt.toString());
                        continue;
                    }
                    CatchClause catchClause = tryStmt.getCatchClauses().get(0);
                    if (catchClause.getParameter().getType().isUnionType()) {
                        // only support single exception
                        SrcTestScanner.debug(
                                method + ": try stmt not single exception: " + tryStmt.toString());
                        continue;
                    }
                    if (tryStmt.getTryBlock().getStatements().size() < 2) {
                        // must have at least 2 statements in try
                        SrcTestScanner
                                .debug(method + ": try stmt less than 2 stmts: " + tryStmt.toString());
                        continue;
                    }
                    // check that last statement is fail(...) or Assert.fail(...)
                    Statement lastStmt = tryStmt.getTryBlock().getStatements()
                            .get(tryStmt.getTryBlock().getStatements().size() - 1);
                    if (!(lastStmt instanceof ExpressionStmt)) {
                        SrcTestScanner.debug(
                                method + ": try stmt last stmt not ExpressionStmt: "
                                        + tryStmt.toString());
                        continue;
                    }
                    ExpressionStmt expressionStmt = (ExpressionStmt) lastStmt;
                    if (!(expressionStmt.getExpression() instanceof MethodCallExpr)) {
                        SrcTestScanner.debug(
                                method + ": try stmt last stmt not MethodCallExpr: "
                                        + tryStmt.toString());
                        continue;
                    }
                    MethodCallExpr methodCallExpr = (MethodCallExpr) expressionStmt.getExpression();
                    if (!methodCallExpr.getNameAsString().equals("fail")) {
                        SrcTestScanner
                                .debug(method + ": try stmt last stmt not fail: " + tryStmt.toString());
                        continue;
                    }
                    if (methodCallExpr.hasScope()) {
                        String scope = methodCallExpr.getScope().get().toString();
                        if (!scope.equals("Assert") && !scope.equals("org.junit.Assert")) {
                            SrcTestScanner.debug(
                                    method + ": try stmt last stmt not [[org.junit.]Assert.]fail: "
                                            + tryStmt.toString());
                            continue;
                        }
                    }
                    exception = TypeResolver.resolveType(catchClause.getParameter().getType());
                    pattern = "try{fail}catch";
                    found = i;
                    break;
                }
            }
            if (found != -1) {
                SrcTestScanner.debug(
                        "method " + method + ": transforming code from: "
                                + md.getBody().get().toString()); // transform code to remove try-catch
                TryStmt tryStmt = (TryStmt) md.getBody().get().getStatement(found);

                for (int i = size - 1; i >= found; --i) {
                    md.getBody().get().getStatements().remove(i);
                }
                for (Expression resource : tryStmt.getResources()) {
                    md.getBody().get().getStatements().add(found, new ExpressionStmt(resource));
                }
                tryStmt.getTryBlock().getStatements().removeLast();
                for (Statement statement : tryStmt.getTryBlock().getStatements()) {
                    md.getBody().get().getStatements().add(statement);
                }
                SrcTestScanner.debug("transforming code to: " + md.getBody().get().toString());
            }
        }

        AST ast = n.accept(astVisitor, new ExtractASTVisitor.Context());

        ctx.records.add(
                new Record(
                        ctx.fqCName, name, exception, pattern, ast, comment, commentSummary,
                        md.getBody().get().toString(), md.toString()));
    }

}
