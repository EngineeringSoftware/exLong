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
 * Find all methods with throw statement under the given class root (in class
 * files). Can also modify (offline instrumentation) the methods to add a
 * logging at the beginning of each method.
 */
class ThrowStatementInstrumentor {

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
     * Scans a class file to find all methods with throw statement and instrument
     * this method at the beginning.
     */
    private static byte[] scanClass(String className, byte[] classFile, boolean modifiable,
            List<Record> records) {
        ClassReader cr = new ClassReader(classFile);
        ClassNode cn = new ClassNode();
        cr.accept(cn, 0);

        boolean modified = true;
        boolean found = false;

        for (MethodNode mn : cn.methods) {
            found = false;
            if ((mn.access & Opcodes.ACC_ABSTRACT) != 0) {
                // skip abstract methods
                continue;
            }
            AbstractInsnNode temp = mn.instructions.getFirst();
            while (temp != null) {
                if (temp.getOpcode() == Opcodes.ATHROW) {
                    found = true;
                    break;
                }
                temp = temp.getNext();
            }
            if (found) {
                String methodLog = cn.name + "#" + mn.name + "#" + mn.desc;
                mn.instructions.insert(
                        new MethodInsnNode(
                                Opcodes.INVOKESTATIC, LogHelper.CNAME, LogHelper.MNAME_LOG,
                                LogHelper.MDESC_LOG)); // LogHelper
                mn.instructions.insert(new LdcInsnNode(methodLog + "\n"));
                mn.instructions.insert(new LdcInsnNode(sConfig.tcMethodsLogPath.toString()));
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
