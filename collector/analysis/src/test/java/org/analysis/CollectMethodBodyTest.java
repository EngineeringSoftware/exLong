package org.analysis;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;

import org.junit.jupiter.api.Test;

public class CollectMethodBodyTest {
    @Test
    public void testNormalCase() throws IOException {
        Parser parser = new Parser();
        String inputPath = "src/test/resources/cm/ATest.java";
        String method = "m1";
        String outputPath = "target/ATestM.java";
        parser.collectMethodBody(inputPath, method, outputPath);
        assertTrue(java.nio.file.Files.exists(java.nio.file.Paths.get(outputPath)));
        String output = TestUtil.getOutput(outputPath);
        assertEquals("public void m1(){\n}", output);
        java.nio.file.Files.delete(java.nio.file.Paths.get(outputPath));
        
    }
}
