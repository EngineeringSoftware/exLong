package org.analysis;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;

import org.junit.jupiter.api.Test;

public class InstrumentMethodTest {
    @Test
    public void testNormalCase() throws IOException {
        // copy BTest to BTest.java
        File file = new File("src/test/resources/cm/BTest");
        String inputPath = "src/test/resources/cm/BTest.java";
        Files.copy(file.toPath(), (new File(inputPath)).toPath(), StandardCopyOption.REPLACE_EXISTING);
        Parser parser = new Parser();
        String logFilePath = "target/BTest.log";
        parser.instrumentMethod(inputPath, "test", logFilePath);
        assertTrue(java.nio.file.Files.exists(java.nio.file.Paths.get(inputPath)));
    }

    @Test
    public void testEnumClass() throws IOException {
        File file = new File("src/test/resources/cm/EnumClz");
        String inputPath = "src/test/resources/cm/EnumClz.java";
        Files.copy(file.toPath(), (new File(inputPath)).toPath(), StandardCopyOption.REPLACE_EXISTING);
        Parser parser = new Parser();
        String logFilePath = "target/EnumClz.log";
        parser.instrumentMethod(inputPath, "findByPort", logFilePath);
        assertTrue(java.nio.file.Files.exists(java.nio.file.Paths.get(inputPath)));
    }
}
