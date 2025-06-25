package org.etestgen.core;

import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import org.apache.commons.collections4.map.LazyMap;
import org.etestgen.util.BytecodeUtils;
import org.etestgen.util.TypeResolver;
import com.github.javaparser.ast.NodeList;
import com.github.javaparser.ast.body.AnnotationDeclaration;
import com.github.javaparser.ast.body.CallableDeclaration;
import com.github.javaparser.ast.body.ClassOrInterfaceDeclaration;
import com.github.javaparser.ast.body.ConstructorDeclaration;
import com.github.javaparser.ast.body.EnumDeclaration;
import com.github.javaparser.ast.body.MethodDeclaration;
import com.github.javaparser.ast.body.Parameter;
import com.github.javaparser.ast.expr.ObjectCreationExpr;
import com.github.javaparser.ast.nodeTypes.NodeWithTypeParameters;
import com.github.javaparser.ast.type.ClassOrInterfaceType;
import com.github.javaparser.ast.type.TypeParameter;
import com.github.javaparser.ast.visitor.VoidVisitorAdapter;

/**
 * A basic source code visitor that keeps track of types information:
 * the fully qualified name of the current class (including inner and anonymous
 * classes);
 * type parameters;
 * extra constructor parameters (caused by inner class).
 */
public class SrcVisitorBase<A extends SrcVisitorBase.Context> extends VoidVisitorAdapter<A> {

    /**
     * The context used by the visitor that stores types information.
     *
     * Subclass of this visitor should usually extend this context
     * class, e.g., to add some fields to store results.
     */
    public static class Context implements Cloneable {
        public Context() {
            this("");
        }

        public Context(String pName) {
            this.pName = pName;
        }

        /** package name (empty means no package) */
        public String pName = "";

        /** class name */
        public String cName = null;

        /** fully qualified class name = package name (if present) . class name */
        public String fqCName = null;

        /**
         * anonymous class count tracker
         * should be refreshed for each new class context
         */
        public int anonymousClassCount = 1;

        /**
         * local class count tracker
         * should be refreshed for each new class context
         */
        public LazyMap<String, Integer> localClassCount = LazyMap.lazyMap(new HashMap<>(), () -> 1);

        /**
         * additional constructor parameters tracker
         * deep cloned
         */
        public List<String> extraInitParams = new LinkedList<>();

        /**
         * type parameters mapping tracker
         * deep cloned
         */
        public Map<String, String> typeParams = new HashMap<>();

        /**
         * Clones the context. Cloning happens when the visitor enters
         * a type declaration (could be inner or anonymous class) or
         * enters a callable declaration with type parameters.
         * 
         * In this class, {@link #extraInitParams} and
         * {@link #typeParams} are deep cloned, and other fields are
         * shallow cloned (default). Subclass should override this
         * method if some of the newly defined fields should not be
         * shallow cloned.
         */
        @Override
        public Context clone() {
            try {
                Context clone = (Context) super.clone();
                clone.typeParams = new HashMap<>(typeParams);
                clone.extraInitParams = new LinkedList<>(extraInitParams);
                return clone;
            } catch (CloneNotSupportedException e) {
                throw new RuntimeException(e);
            }
        }

    }

    protected A cloneForTypeDecl(A ctx, String name) {
        return cloneForTypeDecl(ctx, name, null);
    }

    protected A cloneForTypeDecl(A ctx, String name, List<String> extraInitParams) {
        String newCName = "";
        if (ctx.cName != null) {
            // inner class
            newCName += ctx.cName + "$";
        }

        newCName += name;

        A newCtx = (A) ctx.clone();
        newCtx.cName = newCName;
        newCtx.anonymousClassCount = 1;
        newCtx.localClassCount = LazyMap.lazyMap(new HashMap<>(), () -> 1);
        if (extraInitParams != null) {
            newCtx.extraInitParams.addAll(extraInitParams);
        }

        if (newCtx.pName.isEmpty()) {
            newCtx.fqCName = newCName;
        } else {
            newCtx.fqCName = newCtx.pName + "." + newCName;
        }

        return newCtx;
    }

    protected void registerTypeParameters(Context ctx, NodeWithTypeParameters<?> n) {
        NodeList<TypeParameter> typeParams = n.getTypeParameters();
        if (typeParams != null && !typeParams.isEmpty()) {
            for (TypeParameter typeParameter : typeParams) {
                NodeList<ClassOrInterfaceType> typeBound = typeParameter.getTypeBound();
                String mapTo = "java.lang.Object";
                if (typeBound != null && !typeBound.isEmpty()) {
                    mapTo = TypeResolver.resolveType(typeBound.get(0), ctx.typeParams);
                }
                ctx.typeParams.put(typeParameter.getNameAsString(), mapTo);
            }
        }
    }

    @Override
    public void visit(ClassOrInterfaceDeclaration n, A ctx) {
        String name = null;
        if (n.isLocalClassDeclaration()) {
            // local class is named as OuterClass$%dInnerClass
            int cnt = ctx.localClassCount.get(n.getNameAsString());
            name = cnt + n.getNameAsString();
            ctx.localClassCount.put(n.getNameAsString(), cnt + 1);
        } else {
            name = n.getNameAsString();
        }

        List<String> extraInitParams = null;
        if (ctx.cName != null && !n.isStatic()) {
            // non-static inner class has extra init parameter OuterClass
            extraInitParams = Arrays.asList(ctx.fqCName);
        }

        ctx = cloneForTypeDecl(ctx, name, extraInitParams);
        registerTypeParameters(ctx, n);
        // AllCollectors.warning("visit(ClassOrInterfaceDeclaration) " + ctx.fqCName);
        super.visit(n, ctx);
    }

    @Override
    public void visit(EnumDeclaration n, A ctx) {
        // enum constructor has additional parameters String,int
        super.visit(
                n,
                cloneForTypeDecl(ctx, n.getNameAsString(), Arrays.asList("java.lang.String", "int")));
    }

    @Override
    public void visit(AnnotationDeclaration n, A ctx) {
        super.visit(n, cloneForTypeDecl(ctx, n.getNameAsString()));
    }

    @Override
    public void visit(ObjectCreationExpr n, A ctx) {
        n.getAnonymousClassBody().ifPresent(l -> {
            A newCtx = cloneForTypeDecl(ctx, String.valueOf(ctx.anonymousClassCount));
            ++ctx.anonymousClassCount;
            l.forEach(v -> v.accept(this, newCtx));
        });

        n.getArguments().forEach(p -> p.accept(this, ctx));
        n.getScope().ifPresent(l -> l.accept(this, ctx));
        n.getType().accept(this, ctx);
        n.getTypeArguments().ifPresent(l -> l.forEach(v -> v.accept(this, ctx)));
        n.getComment().ifPresent(l -> l.accept(this, ctx));
    }

    @Override
    public void visit(MethodDeclaration n, A ctx) {
        if (!n.getTypeParameters().isEmpty()) {
            ctx = (A) ctx.clone();
            registerTypeParameters(ctx, n);
        }

        visitCallableDeclaration(n, n.getNameAsString(), null, ctx);
        super.visit(n, ctx);
    }

    @Override
    public void visit(ConstructorDeclaration n, A ctx) {
        if (!n.getTypeParameters().isEmpty()) {
            ctx = (A) ctx.clone();
            registerTypeParameters(ctx, n);
        }

        visitCallableDeclaration(n, "<init>", ctx.extraInitParams, ctx);
        super.visit(n, ctx);
    }

    /**
     * Processes a method/constructor declaration. Subclass can
     * override this method to do whatever processing needed, and call
     * {@link #getMethodReturnType}, {@link #getMethodParameterTypes},
     * {@link #getMethodDesc}, and {@link #getMethodNameDesc} to get
     * the information of the method.
     *
     * This method is called by {@link #visit(MethodDeclaration, A)}
     * and {@link #visit(ConstructorDeclaration, A)}.
     *
     * @param n               the method/constructor declaration
     * @param name            the name of the method, or {@code "<init>"} for
     *                        constructor
     * @param extraInitParams the extra parameters to be added to the beginning of
     *                        the list; currently it only applies for the inner
     *                        class's constructor having the outer class's instance
     *                        as the first parameter
     * @param ctx             the context
     */
    public void visitCallableDeclaration(CallableDeclaration<?> n, String name,
            List<String> extraParams, A ctx) {
    }

    /**
     * Gets the return type of the current method/constructor. The
     * return type for constructor is always void.
     * 
     * @param n   the method/constructor declaration
     * @param ctx the context
     * @return the return type, in fully qualified name (e.g., int,
     *         java.lang.String)
     */
    public String getMethodReturnType(CallableDeclaration<?> n, A ctx) {
        if (n instanceof MethodDeclaration) {
            return TypeResolver.resolveType(((MethodDeclaration) n).getType(), ctx.typeParams);
        } else {
            return "void";
        }
    }

    /**
     * Gets the list of parameter types of the current method/constructor.
     * 
     * @param n           the method/constructor declaration
     * @param extraParams the extra parameters to be added to the beginning of the
     *                    list; currently it only applies for the inner class's
     *                    constructor having the outer class's instance as the first
     *                    parameter
     * @param ctx         the context
     * @return the parameter types, in fully qualified name (e.g., int,
     *         java.lang.String)
     */
    public List<String> getMethodParameterTypes(CallableDeclaration<?> n, List<String> extraParams,
            A ctx) {
        List<String> resolvedPtypes = new LinkedList<>();
        if (extraParams != null) {
            resolvedPtypes.addAll(extraParams);
        }
        for (Parameter param : n.getParameters()) {
            String ptype = TypeResolver.resolveType(param.getType(), ctx.typeParams);
            if (param.isVarArgs()) {
                ptype = ptype + "[]";
            }
            resolvedPtypes.add(ptype);
        }
        return resolvedPtypes;
    }

    /**
     * Gets the descriptor of the current method/constructor (same as
     * the format in bytecode).
     * 
     * @param n           the method/constructor declaration
     * @param extraParams the extra parameters to be added to the beginning of the
     *                    list; currently it only applies for the inner class's
     *                    constructor having the outer class's instance as the first
     *                    parameter
     * @param ctx         the context
     * @return the descriptor as appeared in bytecode (e.g., (Ljava/lang/String;I)V)
     */
    public String getMethodDesc(CallableDeclaration<?> n, List<String> extraParams, A ctx) {
        String desc = "(";
        for (String ptype : getMethodParameterTypes(n, extraParams, ctx)) {
            desc += BytecodeUtils.q2iClassDesc(ptype);
        }
        desc += ")";
        desc += BytecodeUtils.q2iClassDesc(getMethodReturnType(n, ctx));
        return desc;
    }

    /**
     * Gets the full class name + method name + descriptor of the
     * method, to uniquely identify it.
     * 
     * @param n           the method/constructor declaration
     * @param name        the method name
     * @param extraParams the extra parameters to be added to the beginning of the
     *                    list; currently it only applies for the inner class's
     *                    constructor having the outer class's instance as the first
     *                    parameter
     * @param ctx         the context
     * @return the full class name + method name + descriptor connected by '#'
     *         (e.g., java.lang.String#indexOf#(I)I)
     */
    public String getMethodNameDesc(CallableDeclaration<?> n, String name, List<String> extraParams,
            A ctx) {
        return ctx.fqCName + "#" + name + "#"
                + getMethodDesc(n, extraParams, ctx);
    }
}
