package org.analysis;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;

import org.junit.jupiter.api.Test;

public class CollectExceptionTest {
    @Test
    public void testNormalCase() throws IOException {
        Parser parser = new Parser();
        String inputPath = "src/test/resources/ce/A.java";
        String outputPath = "src/test/resources/ce/AUpdated.java";
        parser.collectException(inputPath, outputPath, null);
        String output = TestUtil.getOutput(outputPath);
        assertTrue(output.contains("formatStackTrace"));
        assertTrue(output.contains("stackTrace = ex.getStackTrace();"));
    }
}
