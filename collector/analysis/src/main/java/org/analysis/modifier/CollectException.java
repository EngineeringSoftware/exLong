package org.analysis.modifier;

import org.analysis.util.Context;

import com.github.javaparser.ast.body.ClassOrInterfaceDeclaration;
import com.github.javaparser.ast.body.Parameter;
import com.github.javaparser.ast.comments.Comment;
import com.github.javaparser.ast.stmt.BlockStmt;
import com.github.javaparser.ast.stmt.CatchClause;
import com.github.javaparser.ast.visitor.ModifierVisitor;
import com.github.javaparser.ast.visitor.Visitable;

public class CollectException extends ModifierVisitor<Context> {

    @Override
    public Visitable visit(final CatchClause n, final Context arg) {
        BlockStmt body = (BlockStmt) n.getBody().accept(this, arg);
        Parameter parameter = (Parameter) n.getParameter().accept(this, arg);

        // add the code of format stack trace into the catch clause
        String exceptionType = parameter.getName().toString();

        // add the code to the body of catch clause
        String stmtDeclare = "StackTraceElement[] stackTrace;";
        body.addStatement(stmtDeclare);

        String stmtIf = "if (" + exceptionType + ".getCause() != null) { "
                + "    stackTrace = " + exceptionType + ".getCause().getStackTrace(); "
                + "} else { "
                + "    stackTrace = " + exceptionType + ".getStackTrace(); "
                + "}";
        body.addStatement(stmtIf);

        String stmtClassName = "String curClassName = new Object(){}.getClass().getEnclosingClass().getName();";
        body.addStatement(stmtClassName);

        String stmtMethodName = "String curMethodName = new Object(){}.getClass().getEnclosingMethod().getName();";
        body.addStatement(stmtMethodName);

        String stmtexceptionType = "String exceptionType =" + exceptionType + ".getClass().getName();";
        body.addStatement(stmtexceptionType);

        String stmtInit = "String stackTraceString = curClassName + \"#\" + curMethodName + \"#\" + exceptionType + \"##\" + formatStackTrace(stackTrace);";
        body.addStatement(stmtInit);

        String stmtWrite = "try { "
                + "    Files.write(Paths.get(\"" + Context.logFilePath
                + "\"), stackTraceString.getBytes(), StandardOpenOption.CREATE, StandardOpenOption.APPEND);"
                + "} catch (Exception ioexception) {"
                + "}";
        body.addStatement(stmtWrite);

        Comment comment = n.getComment().map(s -> (Comment) s.accept(this, arg)).orElse(null);
        if (body == null || parameter == null)
            return null;
        n.setBody(body);
        n.setParameter(parameter);
        n.setComment(comment);
        return n;
    }

    @Override
    public Visitable visit(final ClassOrInterfaceDeclaration n, final Context arg) {
        super.visit(n, arg);
        ModifierUtils.printStackTrace(n);
        return n;
    }
}
