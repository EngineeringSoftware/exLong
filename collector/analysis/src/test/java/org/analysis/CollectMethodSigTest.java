package org.analysis;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;

import org.junit.jupiter.api.Test;

public class CollectMethodSigTest {
    @Test
    public void testNormalCase() throws IOException {
        Parser parser = new Parser();
        String inputPath = "src/test/resources/cm/ATest.java";
        String lineNum = "3";
        String logPath = "target/temp-output";
        parser.collectMethodSignature(inputPath, lineNum, logPath);
        assertTrue(java.nio.file.Files.exists(java.nio.file.Paths.get(logPath)));
        String output = TestUtil.getOutput(logPath);
        assertEquals("void m1()", output);
        java.nio.file.Files.delete(java.nio.file.Paths.get(logPath));
    }
}
