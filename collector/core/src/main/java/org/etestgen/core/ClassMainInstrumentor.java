package org.etestgen.core;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import org.etestgen.rt.LogHelper;
import org.etestgen.util.AbstractConfig;
import org.etestgen.util.ClassFileFinder;
import org.etestgen.util.Option;
import org.objectweb.asm.ClassReader;
import org.objectweb.asm.ClassWriter;
import org.objectweb.asm.Opcodes;
import org.objectweb.asm.tree.ClassNode;
import org.objectweb.asm.tree.InsnList;
import org.objectweb.asm.tree.LdcInsnNode;
import org.objectweb.asm.tree.MethodInsnNode;
import org.objectweb.asm.tree.MethodNode;
import org.objectweb.asm.tree.AbstractInsnNode;
import com.fasterxml.jackson.databind.ObjectMapper;

/**
 * Find all methods with throws clauses under the given class root (in class files). Can also modify (offline instrumentation) the methods to add a logging at the beginning of each method.
 */
class ClassMainInstrumentor {

    public static class Config extends AbstractConfig {
        @Option
        public String classroot;
        @Option
        public Path outPath;
        @Option
        public Path tcMethodsLogPath;
        @Option
        public Path exceptionLogPath;
        @Option
        public Path debugPath;
        @Option
        public boolean modify = false;
        @Option
        public boolean scanThrow = true;
    }

    public static Config sConfig;

    public static void main(String... args) {
        Path configPath = Paths.get(args[0]);
        sConfig = AbstractConfig.load(configPath, Config.class);
        debug(sConfig.outPath.toString());
        action();
    }

    public static void action() {
        List<Record> records = Collections.synchronizedList(new ArrayList<>());

        if (sConfig.scanThrow) {
            ClassFileFinder.findClassesParallel(
                sConfig.classroot, (className, classFile,
                    modifiable) -> scanClass(className, classFile, modifiable, records));
            // save records
            ObjectMapper mapper = new ObjectMapper();
            try {
                mapper.writeValue(sConfig.outPath.toFile(), records);
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        } else {
            ClassFileFinder.findClassesParallel(
                sConfig.classroot, (className, classFile,
                    modifiable) -> scanMethods(className, classFile, modifiable));
        }


    }

    public static class Record {
        public String method;
        public List<String> exceptions;

        public Record(String method, List<String> exceptions) {
            this.method = method;
            this.exceptions = exceptions;
        }
    }

    /**
     * Scans a class file to find all methods.
     */
    private static byte[] scanMethods(String className, byte[] classFile, boolean modifiable) {
        ClassReader cr = new ClassReader(classFile);
        ClassNode cn = new ClassNode();
        cr.accept(cn, 0);

        boolean modified = false;

        for (MethodNode mn : cn.methods) {

            if ((mn.access & Opcodes.ACC_ABSTRACT) != 0) {
                // skip abstract or non-public methods
                continue;
            }
            String methodLog = cn.name + "#" + mn.name + "#" + mn.desc;
            mn.instructions.insert(
                new MethodInsnNode(
                    Opcodes.INVOKESTATIC, LogHelper.CNAME, LogHelper.MNAME_LOG,
                    LogHelper.MDESC_LOG)); // LogHelper
            mn.instructions.insert(new LdcInsnNode(methodLog + "\n"));
            mn.instructions.insert(new LdcInsnNode(sConfig.tcMethodsLogPath.toString()));
            modified = true;
        }

        if (modified) {
            ClassWriter cw = new ClassWriter(cr, ClassWriter.COMPUTE_MAXS);
            cn.accept(cw);
            return cw.toByteArray();
        } else {
            return null;
        }
    }

    /**
     * Scans a class file to find all methods with throws clause.
     */
    private static byte[] scanClass(String className, byte[] classFile, boolean modifiable,
        List<Record> records) {
        ClassReader cr = new ClassReader(classFile);
        ClassNode cn = new ClassNode();
        cr.accept(cn, 0);


        boolean modified = false;

        for (MethodNode mn : cn.methods) {

            if ((mn.access & Opcodes.ACC_ABSTRACT) != 0) {
                // skip abstract or non-public methods
                continue;
            }
            String throwExceptionMethod = cn.name + "#" + mn.name + "#" + mn.desc;
            mn.instructions.insert(
                new MethodInsnNode(
                    Opcodes.INVOKESTATIC, LogHelper.CNAME, LogHelper.MNAME_LOG,
                    LogHelper.MDESC_LOG)); // LogHelper
            mn.instructions.insert(new LdcInsnNode(throwExceptionMethod + "\n"));
            mn.instructions.insert(new LdcInsnNode(sConfig.exceptionLogPath.toString()));
            modified = true;

            AbstractInsnNode temp = mn.instructions.getFirst();
            while (temp != null) {
                if (temp.getOpcode() == Opcodes.ATHROW) {
                    // add a log statement at the begging of the throw statement
                    InsnList logInsnList = new InsnList();
                    logInsnList.insert(
                        new MethodInsnNode(
                            Opcodes.INVOKESTATIC, LogHelper.CNAME, LogHelper.MNAME_LOG,
                            LogHelper.MDESC_LOG));
                    logInsnList.insert(new LdcInsnNode(throwExceptionMethod + "\n"));
                    logInsnList.insert(new LdcInsnNode(sConfig.exceptionLogPath.toString()));
                    mn.instructions.insertBefore(temp, logInsnList);
                    modified = true;
                }
                temp = temp.getNext();
            }

            // second: instrument public methods with throws clauses
            if (mn.exceptions != null && !mn.exceptions.isEmpty()) {
                if ((mn.access & Opcodes.ACC_ABSTRACT) != 0) {
                    // skip abstract or non-public methods
                    continue;
                }
                // skip non-public methods
                if ((mn.access & Opcodes.ACC_PUBLIC) == 0) {
                    continue;
                }
                // record this method and throws information
                String throwsClauseMethod = cn.name + "#" + mn.name + "#" + mn.desc;
                if (mn.exceptions != null && !mn.exceptions.isEmpty())
                    records.add(new Record(throwsClauseMethod, mn.exceptions));

                if (sConfig.modify) {
                    // add a log statement at the beginning of the method
                    // LDC logPath
                    // LDC method
                    // INVOKESTATIC LogHelper.log(String path, String message)
                    mn.instructions.insert(
                        new MethodInsnNode(
                            Opcodes.INVOKESTATIC, LogHelper.CNAME, LogHelper.MNAME_LOG,
                            LogHelper.MDESC_LOG)); // LogHelper
                    mn.instructions.insert(new LdcInsnNode(throwsClauseMethod + "\n"));
                    mn.instructions.insert(new LdcInsnNode(sConfig.tcMethodsLogPath.toString()));

                    modified = true;
                }
            }
        }

        if (modified) {
            ClassWriter cw = new ClassWriter(cr, ClassWriter.COMPUTE_MAXS);
            cn.accept(cw);
            return cw.toByteArray();
        } else {
            return null;
        }
    }

    public static void debug(String message) {
        if (sConfig.debugPath != null) {
            message = "[" + ClassMainInstrumentor.class.getSimpleName() + "] " + message;
            if (!message.endsWith("\n")) {
                message += "\n";
            }

            try {
                Files.write(
                    sConfig.debugPath, message.getBytes(), StandardOpenOption.CREATE,
                    StandardOpenOption.APPEND);
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        }
    }
}
