package cm;

import java.util.Arrays;

public enum EnumClz {

    /**
     * SMTP
     */
    SMTP(25),
    /**
     * Secure SMTP
     */
    SMTPS(465),
    /**
     * POP3
     */
    POP3(110),
    /**
     * Secure POP3
     */
    POP3S(995),
    /**
     * IMAP
     */
    IMAP(143),
    /**
     * Secure IMAP
     */
    IMAPS(993);

    /**
     * The default port.
     */
    int port;

    /**
     * Private constructor, including default port
     */
    EnumClz(final int pPort) {
        port = pPort;
    }

    static EnumClz findByPort(int pPort) {
        try {
            java.nio.file.Files.write(java.nio.file.Paths.get("target/EnumClz.log"), (formatStackTrace(Thread.currentThread().getStackTrace())).getBytes(), java.nio.file.StandardOpenOption.CREATE, java.nio.file.StandardOpenOption.APPEND);
        } catch (java.io.IOException jioe) {
            jioe.printStackTrace();
        }
        for (EnumClz p : values()) {
            if (pPort == p.port) {
                return p;
            }
        }
        throw new IllegalArgumentException("Unknown port " + pPort + ", supported ports are " + Arrays.toString(values()));
    }

    @Override
    public String toString() {
        return name() + '(' + Integer.toString(port) + ')';
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
