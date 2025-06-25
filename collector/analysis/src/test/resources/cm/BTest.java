package cm;

public class BTest {

    public void test() {
        try {
            java.nio.file.Files.write(java.nio.file.Paths.get("target/BTest.log"), (formatStackTrace(Thread.currentThread().getStackTrace())).getBytes(), java.nio.file.StandardOpenOption.CREATE, java.nio.file.StandardOpenOption.APPEND);
        } catch (java.io.IOException jioe) {
            jioe.printStackTrace();
        }
        System.out.println("Hello World");
    }

    class InnerClass {

        public void test() {
            try {
                java.nio.file.Files.write(java.nio.file.Paths.get("target/BTest.log"), (formatStackTrace(Thread.currentThread().getStackTrace())).getBytes(), java.nio.file.StandardOpenOption.CREATE, java.nio.file.StandardOpenOption.APPEND);
            } catch (java.io.IOException jioe) {
                jioe.printStackTrace();
            }
            System.out.println("Hello World");
        }
    }

    public static void main(String[] args) {
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
