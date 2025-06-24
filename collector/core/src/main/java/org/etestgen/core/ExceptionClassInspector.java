package org.etestgen.core;

public class ExceptionClassInspector {

    private static String runtimeExceptionName = "java.lang.RuntimeException";
    private static String errorName = "java.lang.Error";
    public static Class<?> runtimeExceptionClass;
    public static Class<?> errorClass;

    public ExceptionClassInspector() {
        try {
            runtimeExceptionClass = Class.forName(runtimeExceptionName);
            errorClass = Class.forName(errorName);
        } catch (ClassNotFoundException e) {
            throw new RuntimeException(e);
        }
    }

    public static void main(String... args) {
        String exceptionName = "java.io.IOException";
        Class<?> classE = null;
        try {
            classE = Class.forName(exceptionName);
        } catch (ClassNotFoundException e) {
            throw new RuntimeException(e);
        }
        try {
            runtimeExceptionClass = Class.forName(runtimeExceptionName);
            errorClass = Class.forName(errorName);
        } catch (ClassNotFoundException e) {
            throw new RuntimeException(e);
        }
        if (runtimeExceptionClass.isAssignableFrom(classE)) {
            System.out.println("It is unchecked exception");
            // ClassA is a subclass of ClassB or ClassA is the same as ClassB
        } else {
            System.out.println("It is checked exception");
            // ClassA is not a subclass of ClassB
        }
    }
}
