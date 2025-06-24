package org.analysis;

import java.io.IOException;

public class TestUtil {
    public static String getOutput(String outputPath) throws IOException {
        // read output file
        return java.nio.file.Files.lines(java.nio.file.Paths.get(outputPath))
                .collect(java.util.stream.Collectors.joining("\n"));
    }
}
