package org.teco.joint;

import java.util.HashSet;
import java.util.LinkedList;
import java.util.Queue;
import java.util.Set;
import org.apache.commons.lang3.tuple.Pair;
import org.objectweb.asm.Opcodes;
import org.objectweb.asm.tree.AbstractInsnNode;
import org.objectweb.asm.tree.InvokeDynamicInsnNode;
import org.objectweb.asm.tree.MethodInsnNode;
import org.objectweb.asm.tree.MethodNode;
import org.teco.util.BytecodeUtils;
import org.teco.util.ParallelUtils;

import org.teco.AllCollectors;

public class CGAnalyzer {
    public CallGraph cg = new CallGraph();

    public void analyze() {
        ParallelUtils.parallelForEach(JointCollector.methodNodes.entrySet(), e -> {
            int mid = e.getKey();
            MethodNode mn = e.getValue();

            // collect OVERRIDE edges
            findOverrideEdges(mid, mn);

            // collect CALL edges
            findCallEdges(mid, mn);
        });
    }

    private void findOverrideEdges(int mid, MethodNode mn) {
        // TODO: may not strictly follow the JVM spec, e.g., in terms of checking private/static methods
        int overriddenMid = -1;

        MethodStructure ms = JointCollector.methods.get(mid);

        // private/static method never overrides
        if ((ms.access & Opcodes.ACC_STATIC) != 0 || (ms.access & Opcodes.ACC_PRIVATE) != 0) {
            return;
        }

        ClassStructure cs = JointCollector.classes.get(ms.clz);

        // first, try to find in super class
        Queue<Integer> implCids = new LinkedList<>();
        implCids.addAll(cs.impl);

        int superCid = cs.ext;

        while (superCid >= 0) {
            ClassStructure superCS = JointCollector.classes.get(superCid);
            implCids.addAll(superCS.impl);
            for (int superMid : superCS.methods) {
                MethodStructure superMS = JointCollector.methods.get(superMid);
                if (superMS.name.equals(ms.name) && superMS.ptypes.equals(ms.ptypes)) {
                    overriddenMid = superMid;
                    // add the override edge if found
                    cg.addEdge(mid, overriddenMid, CallGraph.EDGE_OVERRIDE);
                }
            }
            superCid = superCS.ext;
        }

        // then, try to find in interfaces
        Set<Integer> visitedImplCids = new HashSet<>();
        // if (overriddenMid < 0) {
        if (true) {
            out: while (!implCids.isEmpty()) {
                int implCid = implCids.poll();

                if (implCid < 0 || visitedImplCids.contains(implCid)) {
                    continue;
                }
                ClassStructure implCS = JointCollector.classes.get(implCid);
                for (int implMid : implCS.methods) {
                    MethodStructure implMS = JointCollector.methods.get(implMid);
                    if (implMS.name.equals(ms.name) && implMS.ptypes.equals(ms.ptypes)) {
                        overriddenMid = implMid;
                        // add the override edge if found
                        cg.addEdge(mid, overriddenMid, CallGraph.EDGE_OVERRIDE);
                    }
                }
                visitedImplCids.add(implCid);
                for (int superImplCid : implCS.impl) {
                    if (superImplCid < 0 || visitedImplCids.contains(superImplCid)) {
                        continue;
                    }
                    implCids.add(superImplCid);
                }
            }
        }
    }

    protected void findCallEdges(int callerMid, MethodNode mn) {
        // use a set here to collapse duplicate method invocations
        Set<Pair<String, String>> calls = new HashSet<>();
        // go over all instructions
        for (AbstractInsnNode insn : mn.instructions) {
            if (insn instanceof MethodInsnNode) {
                // record the method invocation
                MethodInsnNode mInsn = (MethodInsnNode) insn;
                calls.add(
                    Pair.of(
                        BytecodeUtils.i2qName(mInsn.owner),
                        mInsn.name + BytecodeUtils.i2qMethodDesc(mInsn.desc)));
            } else if (insn instanceof InvokeDynamicInsnNode) {
                // TODO: handle invokedynamic in the future? it probably won't affect focal method identification
            }
        }

        // For each method invocation, add one edge
        for (Pair<String, String> call : calls) {
            String cname = call.getLeft();
            int cid = JointCollector.name2cid.getOrDefault(cname, -2);
            if (cid < 0 || !JointCollector.cid2sign2mid.containsKey(cid)) {
                continue;
            }

            String sign = call.getRight();
            int calleeMid = JointCollector.cid2sign2mid.get(cid).getOrDefault(sign, -2);

            if (calleeMid < 0) {
                Queue<Integer> implCids = new LinkedList<>();
                // look for method in parent classes
                MethodStructure ms = JointCollector.methods.get(callerMid);
                ClassStructure cs = JointCollector.classes.get(ms.clz);
                int superCid = cs.ext;
                while (superCid >= 0) {
                    ClassStructure superCS = JointCollector.classes.get(superCid);
                    calleeMid = JointCollector.cid2sign2mid.get(superCid).getOrDefault(sign, -2);
                    if (calleeMid >= 0) {
                        break;
                    }
                    superCid = superCS.ext;
                }

                // then, try to find in interfaces
                Set<Integer> visitedImplCids = new HashSet<>();
                if (calleeMid < 0) {
                    out: while (!implCids.isEmpty()) {
                        int implCid = implCids.poll();
                        if (implCid < 0 || visitedImplCids.contains(implCid)) {
                            continue;
                        }
                        ClassStructure implCS = JointCollector.classes.get(implCid);
                        calleeMid = JointCollector.cid2sign2mid.get(implCid).getOrDefault(sign, -2);
                        if (calleeMid >= 0) {
                            break out;
                        }
                        visitedImplCids.add(implCid);
                        for (int superImplCid : implCS.impl) {
                            if (superImplCid < 0 || visitedImplCids.contains(superImplCid)) {
                                continue;
                            }
                            implCids.add(superImplCid);
                        }
                    }
                }
            }

            if (calleeMid < 0) {
                AllCollectors.debug("Cannot find callee for " + call.toString());
                continue;
            }

            // add the edge
            cg.addEdge(callerMid, calleeMid, CallGraph.EDGE_CALL);
        }
    }
}
