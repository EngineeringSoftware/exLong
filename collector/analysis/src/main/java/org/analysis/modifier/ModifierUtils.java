package org.analysis.modifier;

import com.github.javaparser.ast.Modifier;
import com.github.javaparser.ast.body.ClassOrInterfaceDeclaration;
import com.github.javaparser.ast.body.EnumDeclaration;
import com.github.javaparser.ast.body.MethodDeclaration;
import com.github.javaparser.ast.stmt.BlockStmt;

public class ModifierUtils {
    public static void printStackTrace(ClassOrInterfaceDeclaration n) {
        MethodDeclaration method = n.addMethod("formatStackTrace", Modifier.Keyword.PUBLIC, Modifier.Keyword.STATIC);
        printStackTraceHelper(method);
    }

    public static void printStackTrace(EnumDeclaration n) {
        MethodDeclaration method = n.addMethod("formatStackTrace", Modifier.Keyword.PUBLIC, Modifier.Keyword.STATIC);
        printStackTraceHelper(method);
    }

    public static void printStackTraceHelper(MethodDeclaration method) {
        method.setType(String.class);
        method.addParameter(StackTraceElement[].class, "stackTrace");
        BlockStmt body = new BlockStmt();
        body.addStatement("StringBuilder sb = new StringBuilder();");
        body.addStatement("for (StackTraceElement ste : stackTrace) {"
                + "    if (ste.getClassName().startsWith(\"org.junit\")) {"
                + "        break;"
                + "    }"
                + "    sb.append(ste.getClassName());"
                + "    sb.append(\"#\");"
                + "    sb.append(ste.getMethodName());"
                + "    sb.append(\"#\");"
                + "    sb.append(ste.getLineNumber());"
                + "    sb.append(\"##\");"
                + "}");
        body.addStatement("return sb.toString() + \"\\n\";");
        method.setBody(body);
    }
}
