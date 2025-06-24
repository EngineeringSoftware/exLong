package test;

import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;

public class A {

    public void foo() {
        try {
            System.out.println("Hello");
        } catch (Exception ex) {
            System.out.println("World");
            StackTraceElement[] stackTrace;
            if (ex.getCause() != null) {
                stackTrace = ex.getCause().getStackTrace();
            } else {
                stackTrace = ex.getStackTrace();
            }
            String curClassName = new Object() {
            }.getClass().getEnclosingClass().getName();
            String curMethodName = new Object() {
            }.getClass().getEnclosingMethod().getName();
            String exceptionType = ex.getClass().getName();
            String stackTraceString = curClassName + "#" + curMethodName + "#" + exceptionType + "##" + formatStackTrace(stackTrace);
            try {
                Files.write(Paths.get("null"), stackTraceString.getBytes(), StandardOpenOption.CREATE, StandardOpenOption.APPEND);
            } catch (Exception ioexception) {
            }
        }
    }

    public void bar() {
        System.out.println("Hello World");
    }

    public static String formatStackTrace(StackTraceElement[] stackTrace) {
        StringBuilder sb = new StringBuilder();
        for (StackTraceElement ste : stackTrace) {
            if (ste.getClassName().startsWith("org.junit")) {
                break;
            }
            sb.append(ste.getClassName());
            sb.append("#");
            sb.append(ste.getMethodName());
            sb.append("#");
            sb.append(ste.getLineNumber());
            sb.append("##");
        }
        return sb.toString() + "\n";
    }
}
