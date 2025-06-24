package org.smethods;

import org.objectweb.asm.ClassVisitor;
import org.objectweb.asm.MethodVisitor;
import org.objectweb.asm.Opcodes;

import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;

import static org.smethods.Macros.*;

public class MethodCallCollectorCV extends ClassVisitor {

    // Name of the class being visited.
    private String mClassName;

    Map<String, Set<String>> methodName2InvokedMethodNames;
    Map<String, Set<String>> hierarchy_parents;
    Map<String, Set<String>> hierarchy_children;
    Map<String, Set<String>> class2ContainedMethodNames;
    Set<String> classesInConstantPool;

    public MethodCallCollectorCV(Map<String, Set<String>> methodName2MethodNames,
            Map<String, Set<String>> hierarchy_parents, Map<String, Set<String>> hierarchy_children,
            Map<String, Set<String>> class2ContainedMethodNames,
            Set<String> classesInConstantPool) {
        super(ASM_VERSION);
        this.methodName2InvokedMethodNames = methodName2MethodNames;
        this.hierarchy_parents = hierarchy_parents;
        this.hierarchy_children = hierarchy_children;
        this.class2ContainedMethodNames = class2ContainedMethodNames;
        this.classesInConstantPool = classesInConstantPool;
    }

    public MethodCallCollectorCV(Map<String, Set<String>> methodName2MethodNames,
            Map<String, Set<String>> hierarchy_parents, Map<String, Set<String>> hierarchy_children,
            Map<String, Set<String>> class2ContainedMethodNames) {
        super(ASM_VERSION);
        this.methodName2InvokedMethodNames = methodName2MethodNames;
        this.hierarchy_parents = hierarchy_parents;
        this.hierarchy_children = hierarchy_children;
        this.class2ContainedMethodNames = class2ContainedMethodNames;
    }

    @Override
    public void visit(int version, int access, String name, String signature, String superName,
            String[] interfaces) {
        super.visit(version, access, name, signature, superName, interfaces);
        mClassName = name;
    }

    @Override
    public MethodVisitor visitMethod(int access, final String outerName, final String outerDesc,
            String signature, String[] exceptions) {
        // append arguments to key, remove what after ) of desc
        if (outerName.equals(PROJECT_PACKAGE)) {
            return null;
        }
        String outerMethodSig = outerName + "#" + outerDesc;
        String key = mClassName.replace('/', '.') + "#" + outerMethodSig;
        Set<String> mInvokedMethods =
                methodName2InvokedMethodNames.computeIfAbsent(key, k -> new TreeSet<>());
        return new MethodVisitor(ASM_VERSION) {
            @Override
            public void visitMethodInsn(int opcode, String owner, String name, String desc,
                    boolean itf) {
                if (!owner.startsWith("java/") && !owner.startsWith("org/junit/")
                        && !owner.startsWith(PROJECT_PACKAGE)) {
                    String methodSig = name + "#" + desc;
                    if (class2ContainedMethodNames.getOrDefault(owner, new HashSet<>())
                            .contains(methodSig)) {
                        String invokedKey = owner.replace('/', '.') + "#" + methodSig;
                        mInvokedMethods.add(invokedKey);
                    } else {
                        // find the first parent that implements the method
                        String firstParent = findFirstParent(owner, methodSig);
                        if (!firstParent.equals("")) {
                            String invokedKey = firstParent.replace('/', '.') + "#" + methodSig;
                            mInvokedMethods.add(invokedKey);
                        }
                    }
                    if (!methodSig.startsWith("<init>") && !methodSig.startsWith("<clinit>")) {
                        HashSet<String> visitedClass = new HashSet<>();
                        HashSet<String> subMethods = new HashSet<>();
                        getAllSubMethods(owner, methodSig, visitedClass, subMethods);
                        mInvokedMethods.addAll(subMethods);
                    }
                }
                super.visitMethodInsn(opcode, owner, name, desc, itf);
            }

        };
    }

    public void getAllSubMethods(String currentClass, String methodSig, Set<String> visitedClass, Set<String> subMethods) {
        if (!visitedClass.contains(currentClass)) {
            visitedClass.add(currentClass);
            if (class2ContainedMethodNames.getOrDefault(currentClass, new HashSet<>())
                        .contains(methodSig)) {
                subMethods.add(currentClass.replace('/', '.') + "#" + methodSig);
            }
            for (String subClass : hierarchy_children.getOrDefault(currentClass, new HashSet<>())) {
                getAllSubMethods(subClass, methodSig, visitedClass, subMethods);
            }
        }
    }

    public String findFirstParent(String currentClass, String methodSig) {
        for (String parent : hierarchy_parents.getOrDefault(currentClass, new HashSet<>())) {
            if (class2ContainedMethodNames.getOrDefault(parent, new HashSet<>())
                    .contains(methodSig)) {
                return parent;
            } else {
                String firstParent = findFirstParent(parent, methodSig);
                if (!firstParent.equals("")) {
                    return firstParent;
                }
            }
        }
        return "";
    }
}
