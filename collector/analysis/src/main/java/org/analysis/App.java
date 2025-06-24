package org.analysis;

import java.io.IOException;

import org.analysis.util.Constant;

public class App {
    public static void main(String[] args) throws IOException {
        String task = args[0];
        if (task.equals("collect-exception")) {
            String filePath = args[1];
            String outputFilePath = args[2];
            String logFilePath = args[3];
            new Parser().collectException(filePath, outputFilePath, logFilePath);
        } else if (task.equals(Constant.TASK_METHOD)) {
            String filePath = args[1];
            String methodName = args[2];
            String outputFilePath = args[3];
            new Parser().collectMethodBody(filePath, methodName, outputFilePath);
        } else if (task.equals(Constant.TASK_INS_METHOD)) {
            String filePath = args[1];
            String methodName = args[2];
            String logFilePath = args[3];
            new Parser().instrumentMethod(filePath, methodName, logFilePath);
        } else if (task.equals(Constant.TASK_METHOD_SIG)) {
            String filePath = args[1];
            String lineNum = args[2];
            String outputFilePath = args[3];
            new Parser().collectMethodSignature(filePath, lineNum, outputFilePath);
        } else {
            System.out.println("Invalid task: " + task);
        }
    }
}
