package org.analysis.util;

import java.util.List;

/**
 * This class is used to store the context of the visitor.
 */
public class Context {
    public static String srcPath;
    public static String outputFilePath;
    public static String task;
    public static String logFilePath;

    public static String methodName;
    public static List<String> methodNames;

    public int lineNum;
}
