package org.etestgen.rt;


import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;

public class LogHelper {

    public static String CNAME = LogHelper.class.getName().replace('.', '/');

    public static String MNAME_LOG = "log";
    public static String MDESC_LOG = "(Ljava/lang/String;Ljava/lang/String;)V";

    public static void log(String path, String message) {
        try {
            // Files.write(
            // Paths.get(path), message.getBytes(), StandardOpenOption.CREATE,
            // StandardOpenOption.APPEND);e.printStackTrace();
            String stackTrace = formatStackTrace(Thread.currentThread().getStackTrace());
            int stackTraceDepth = Thread.currentThread().getStackTrace().length;
            String methodAndStackTrace = String.valueOf(stackTraceDepth) + "@@"
                + message.replaceAll("\n", "") + "@@" + stackTrace + "\n";
            Files.write(
                Paths.get(path), methodAndStackTrace.getBytes(), StandardOpenOption.CREATE,
                StandardOpenOption.APPEND); //
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    public static String formatStackTrace(StackTraceElement stackTrace[]) {
        StringBuilder sb = new StringBuilder();
        for (StackTraceElement ste : stackTrace) {
            sb.append(ste.getClassName());
            sb.append("#");
            sb.append(ste.getMethodName());
            sb.append("#");
            sb.append(ste.getLineNumber());
            sb.append("##");
        }
        return sb.toString();
    }
}
