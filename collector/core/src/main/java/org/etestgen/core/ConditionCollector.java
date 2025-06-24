package org.etestgen.core;

import com.github.javaparser.StaticJavaParser;
import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.ast.body.ConstructorDeclaration;
import com.github.javaparser.ast.body.MethodDeclaration;
import com.github.javaparser.ast.visitor.VoidVisitorAdapter;
import com.alibaba.fastjson.JSON;
import com.github.javaparser.ast.body.Parameter;
import com.github.javaparser.ast.stmt.SwitchEntry;
import com.github.javaparser.ast.stmt.SwitchStmt;
import com.github.javaparser.ast.expr.AssignExpr;
import com.github.javaparser.ast.expr.BinaryExpr;
import com.github.javaparser.ast.expr.NameExpr;
import com.github.javaparser.ast.expr.VariableDeclarationExpr;
import com.github.javaparser.ast.stmt.BlockStmt;
import com.github.javaparser.ast.body.VariableDeclarator;
import com.github.javaparser.ast.expr.UnaryExpr;
import com.github.javaparser.ast.expr.UnaryExpr.Operator;
import com.github.javaparser.ast.stmt.ThrowStmt;
import com.github.javaparser.ast.stmt.WhileStmt;
import com.github.javaparser.ast.stmt.IfStmt;
import com.github.javaparser.ast.stmt.ForStmt;
import com.github.javaparser.ast.stmt.ExpressionStmt;
import com.github.javaparser.ast.stmt.ForEachStmt;
import java.util.stream.Collectors;
import javax.management.RuntimeErrorException;
import com.github.javaparser.ast.Node;
import com.github.javaparser.ast.NodeList;
import com.github.javaparser.ast.stmt.Statement;
import com.github.javaparser.ast.expr.Expression;
import com.github.javaparser.ast.expr.MethodCallExpr;
import java.util.*;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.io.BufferedWriter;
import java.io.FileWriter;

/**
 * 1. starting from the position that throwing condition, collect all the conditional expressions
 * along the trace 2. walk through the stack trace, whenever nameExpr can be substituted, do that
 */

class ConditionCollector {

    public static ArrayList<Node> nodesStack = new ArrayList<>();
    static List<Node> conditionNodes = new ArrayList<>();

    public static String readFileAsString(String filePath) throws IOException {
        Path path = Paths.get(filePath);
        return new String(Files.readAllBytes(path));
    }

    public static void writeListToFile(List<String> stringList, String filePath) {
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(filePath))) {
            for (String line : stringList) {
                writer.write(line);
                writer.newLine(); // Add a newline after each line
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static void main(String... args) {
        String inPath = args[0];
        String outPath = args[1];
        List<String> conditions = new ArrayList<>();
        // read data from file
        try {
            String jsonText = readFileAsString(inPath);
            List<HashMap> jsonList = JSON.parseArray(jsonText, HashMap.class);
            for (HashMap stackTrace : jsonList) {
                nodesStack.clear();
                conditionNodes.clear();
                List<String> methodList = (List<String>) stackTrace.get("methodStrings");
                List<Integer> lineList = (List<Integer>) stackTrace.get("lineNumbers");
                // Need to reverse here to traverse from the root cause
                Collections.reverse(methodList);
                Collections.reverse(lineList);
                String condition = getConditionsFromTrace(methodList, lineList);
                String jsonString = JSON.toJSONString(condition);
                conditions.add(jsonString);
            }
            writeListToFile(conditions, outPath);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static String getConditionsFromTrace(List<String> methodList,
            List<Integer> lineNumbers) {
        int stackTraceLength = methodList.size();
        for (int i = 0; i < stackTraceLength; i++) {
            int lineNumber = lineNumbers.get(i);
            String methodString = methodList.get(i);
            String tempJavaClass = "public class TempClass {\n" + methodString + "\n}";
            try {
                CompilationUnit compilationUnit = StaticJavaParser.parse(tempJavaClass);
                // Create a visitor to traverse method declaration nodes
                ConditionVisitor methodVisitor = new ConditionVisitor();
                methodVisitor.visit(compilationUnit, lineNumber);
            } catch (Exception e) {

                // TODO: write to log
                if (e.getMessage().contains("'default' is not allowed here")) {
                    tempJavaClass = "interface TempClass {\n" + methodString + "\n}";
                    CompilationUnit compilationUnit = StaticJavaParser.parse(tempJavaClass);
                    // Create a visitor to traverse method declaration nodes
                    ConditionVisitor methodVisitor = new ConditionVisitor();
                    methodVisitor.visit(compilationUnit, lineNumber);
                } else {
                    // StackTraceElement[] stackTrace = e.getStackTrace();
                    // for (StackTraceElement ex : stackTrace) {
                    // System.out.println(ex);
                    // }
                    System.out.println(e.getMessage());
                }
            }
        }
        // merge condition path
        extractPathCondition(nodesStack);
        List<String> conditionsStrings =
                conditionNodes.stream().map(Object::toString).collect(Collectors.toList());
        Collections.reverse(conditionsStrings);
        String conditionString = String.join(" && ", conditionsStrings);
        return conditionString;
    }

    // Custom visitor class to traverse method declaration nodes
    private static class ConditionVisitor extends VoidVisitorAdapter<Integer> {
        private boolean foundMethod = false;

        @Override
        public void visit(ConstructorDeclaration constructorDeclaration, Integer lineNumber) {
            boolean negate = false;
            if (foundMethod)
                return;
            Node beginStmt = findNodeByLine(constructorDeclaration, lineNumber);
            if (beginStmt == null) {
                System.out.println("Cannot find the line number in the constructor declaration "
                        + lineNumber + " " + constructorDeclaration);
                return;
            }

            List<Node> nodesToCheck = new ArrayList<>();
            List<Node> parsedNodes = new ArrayList<>();
            nodesToCheck.add(beginStmt);
            parsedNodes.add(beginStmt);
            // Walk back through the method declaration to find nodes that might have
            // condition
            Node parent = beginStmt.getParentNode().orElse(null);
            Node current = beginStmt;
            while (parent != null) {
                if (!(parent instanceof IfStmt))
                    negate = false;
                parsedNodes.add(parent);
                if (parent instanceof ForStmt) {
                    nodesToCheck.add(((ForStmt) parent).getCompare().get());
                }
                if (parent instanceof ForEachStmt) {
                    AssignExpr assignExpr = new AssignExpr(
                            new NameExpr(
                                    ((ForEachStmt) parent).getVariable().getVariable(0).getName()),
                            ((ForEachStmt) parent).getIterable(), AssignExpr.Operator.ASSIGN);
                    nodesToCheck.add(assignExpr);
                }
                if (parent instanceof WhileStmt) {
                    WhileStmt whileStmt = (WhileStmt) parent;
                    nodesToCheck.add(whileStmt.getCondition());
                }
                if (parent instanceof BlockStmt) {
                    BlockStmt blockStmt = (BlockStmt) parent;
                    NodeList<Statement> statements = blockStmt.getStatements();
                    for (Statement statement : statements) {
                        if (parsedNodes.contains(statement))
                            break;
                        List<Node> expressionStmtList = new ArrayList<>();
                        processStatement(statement, expressionStmtList);
                        nodesToCheck.addAll(expressionStmtList);
                    }
                }
                if (parent instanceof IfStmt) {
                    IfStmt ifStmt = (IfStmt) parent;
                    if (ifStmt.getElseStmt().isPresent() == true
                            && ifStmt.getElseStmt().get() == current
                            && !ifStmt.getElseStmt().get().isIfStmt()) {
                        negate = true;
                        Expression ifCondition = ifStmt.getCondition();
                        Expression newCondition =
                                new UnaryExpr(ifCondition, Operator.LOGICAL_COMPLEMENT);
                        nodesToCheck.add(newCondition);
                    } else if (negate) {
                        Expression ifCondition = ifStmt.getCondition();
                        Expression newCondition =
                                new UnaryExpr(ifCondition, Operator.LOGICAL_COMPLEMENT);
                        nodesToCheck.add(newCondition);
                    } else if (ifStmt.getElseStmt().isPresent() == false
                            || ifStmt.getThenStmt() == current) {
                        nodesToCheck.add(ifStmt.getCondition());
                    }
                }
                if (parent instanceof SwitchStmt) {
                    SwitchStmt switchStmt = (SwitchStmt) parent;
                    if (current instanceof SwitchEntry) {
                        SwitchEntry switchEntry = (SwitchEntry) current;
                        if (switchEntry.getLabels().size() > 0) {
                            for (Expression expr : switchEntry.getLabels()) {
                                BinaryExpr binaryExpr = new BinaryExpr(switchStmt.getSelector(),
                                        expr, BinaryExpr.Operator.EQUALS);
                                nodesToCheck.add(binaryExpr);
                            }
                        } else {
                            // negate every switch selector
                            for (SwitchEntry entry : switchStmt.getEntries()) {
                                if (entry == switchEntry)
                                    continue;
                                BinaryExpr binaryExpr = new BinaryExpr(switchStmt.getSelector(),
                                        entry.getLabels().get(0), BinaryExpr.Operator.NOT_EQUALS);
                                nodesToCheck.add(binaryExpr);
                            }
                        }
                    }

                }
                // nodesToCheck.add(parent);
                current = parent;
                parent = parent.getParentNode().orElse(null);
                if (parent instanceof ConstructorDeclaration) {
                    nodesToCheck.add(parent);
                    break;
                }
            }
            nodesStack.addAll(nodesToCheck);
            foundMethod = true;
            super.visit(constructorDeclaration, lineNumber);
        }

        @Override
        public void visit(MethodDeclaration methodDeclaration, Integer lineNumber) {
            boolean negate = false;
            if (foundMethod)
                return;
            Node beginStmt = findNodeByLine(methodDeclaration, lineNumber);
            if (beginStmt == null) {
                System.out.println("Cannot find the line number in the method declaration "
                        + lineNumber + " " + methodDeclaration);
                return;
            }
            List<Node> nodesToCheck = new ArrayList<>();
            List<Node> parsedNodes = new ArrayList<>();
            nodesToCheck.add(beginStmt);
            parsedNodes.add(beginStmt);
            // Walk back through the method declaration to find nodes that might have
            // condition
            Node parent = beginStmt.getParentNode().orElse(null);
            Node current = beginStmt;
            while (parent != null) {
                if (!(parent instanceof IfStmt))
                    negate = false;
                parsedNodes.add(parent);
                if (parent instanceof ForStmt) {
                    if (((ForStmt) parent).getCompare().isPresent())
                        nodesToCheck.add(((ForStmt) parent).getCompare().get());
                }
                if (parent instanceof ForEachStmt) {
                    AssignExpr assignExpr = new AssignExpr(
                            new NameExpr(
                                    ((ForEachStmt) parent).getVariable().getVariable(0).getName()),
                            ((ForEachStmt) parent).getIterable(), AssignExpr.Operator.ASSIGN);
                    nodesToCheck.add(assignExpr);
                }
                if (parent instanceof WhileStmt) {
                    WhileStmt whileStmt = (WhileStmt) parent;
                    nodesToCheck.add(whileStmt.getCondition());
                }
                if (parent instanceof BlockStmt) {
                    BlockStmt blockStmt = (BlockStmt) parent;
                    NodeList<Statement> statements = blockStmt.getStatements();
                    NodeList<Statement> stmtBeforeStop = new NodeList<>();
                    // Collections.reverse(bstatements);
                    for (Statement statement : statements) {
                        if (parsedNodes.contains(statement))
                            break;
                        stmtBeforeStop.add(statement);
                    }
                    for (int i = stmtBeforeStop.size() - 1; i >= 0; i--) {
                        Statement statement = stmtBeforeStop.get(i);
                        List<Node> expressionStmtList = new ArrayList<>();
                        processStatement(statement, expressionStmtList);
                        nodesToCheck.addAll(expressionStmtList);
                    }
                }
                if (parent instanceof IfStmt) {
                    IfStmt ifStmt = (IfStmt) parent;
                    if (ifStmt.getElseStmt().isPresent() == true
                            && ifStmt.getElseStmt().get() == current
                            && !ifStmt.getElseStmt().get().isIfStmt()) {
                        negate = true;
                        Expression ifCondition = ifStmt.getCondition();
                        Expression newCondition =
                                new UnaryExpr(ifCondition, Operator.LOGICAL_COMPLEMENT);
                        nodesToCheck.add(newCondition);
                    } else if (negate) {
                        Expression ifCondition = ifStmt.getCondition();
                        Expression newCondition =
                                new UnaryExpr(ifCondition, Operator.LOGICAL_COMPLEMENT);
                        nodesToCheck.add(newCondition);
                    } else if (ifStmt.getElseStmt().isPresent() == false
                            || ifStmt.getThenStmt() == current) {
                        nodesToCheck.add(ifStmt.getCondition());
                    }
                }
                if (parent instanceof SwitchStmt) {
                    SwitchStmt switchStmt = (SwitchStmt) parent;
                    if (current instanceof SwitchEntry) {
                        SwitchEntry switchEntry = (SwitchEntry) current;
                        if (switchEntry.getLabels().size() > 0) {
                            for (Expression expr : switchEntry.getLabels()) {
                                BinaryExpr binaryExpr = new BinaryExpr(switchStmt.getSelector(),
                                        expr, BinaryExpr.Operator.EQUALS);
                                nodesToCheck.add(binaryExpr);
                            }
                        } else {
                            // negate every switch selector
                            for (SwitchEntry entry : switchStmt.getEntries()) {
                                if (entry == switchEntry)
                                    continue;
                                BinaryExpr binaryExpr = new BinaryExpr(switchStmt.getSelector(),
                                        entry.getLabels().get(0), BinaryExpr.Operator.NOT_EQUALS);
                                nodesToCheck.add(binaryExpr);
                            }
                        }
                    }

                }
                // nodesToCheck.add(parent);
                current = parent;
                parent = parent.getParentNode().orElse(null);
                if (parent instanceof MethodDeclaration) {
                    nodesToCheck.add(parent);
                    break;
                }
            }
            nodesStack.addAll(nodesToCheck);
            foundMethod = true;
            super.visit(methodDeclaration, lineNumber);
        }
    }

    private static List<Expression> ifConditions = new ArrayList<>();

    private static boolean isExitMethodStmt(Statement statement) {
        if (statement.isExpressionStmt()) {
            ExpressionStmt expressionStmt = statement.asExpressionStmt();
            if (expressionStmt.getExpression().isMethodCallExpr()) {
                return true;
            }
        } else if (statement.isThrowStmt()) {
            return true;
        } else if (statement.isReturnStmt()) {
            return true;
        }
        return false;
    }

    private static void traverseIfStmt(IfStmt ifStmt, List<Node> nodesToCheck) {

        // If condition
        ifConditions
                .add(new UnaryExpr(ifStmt.getCondition(), UnaryExpr.Operator.LOGICAL_COMPLEMENT));
        if (isExitMethodStmt(ifStmt.getThenStmt())) {
            nodesToCheck.addAll(ifConditions);
        } else if (ifStmt.getThenStmt().isBlockStmt()) {
            BlockStmt thenBlockStmt = ifStmt.getThenStmt().asBlockStmt();
            for (Statement statement : thenBlockStmt.getStatements()) {
                if (statement.isIfStmt()) {
                    traverseIfStmt((IfStmt) statement, nodesToCheck);
                } else if (isExitMethodStmt(statement)) {
                    nodesToCheck.addAll(ifConditions);
                }
            }
        }
        if (ifConditions.size() > 0)
            ifConditions.remove(ifConditions.size() - 1);
        if (ifStmt.getElseStmt().isPresent()) {
            if (!ifStmt.getElseStmt().get().isIfStmt())
                ifConditions.add(ifStmt.getCondition());
            if (ifStmt.getElseStmt().get().isIfStmt()) {
                traverseIfStmt((IfStmt) ifStmt.getElseStmt().get(), nodesToCheck);
            } else if (isExitMethodStmt(ifStmt.getElseStmt().get())) {
                nodesToCheck.addAll(ifConditions);
            } else if (ifStmt.getElseStmt().get().isBlockStmt()) {
                BlockStmt elseBlockStmt = ifStmt.getElseStmt().get().asBlockStmt();
                for (Statement statement : elseBlockStmt.getStatements()) {
                    if (statement.isIfStmt()) {
                        traverseIfStmt((IfStmt) statement, nodesToCheck);
                    } else if (isExitMethodStmt(statement)) {
                        nodesToCheck.addAll(ifConditions);
                    }
                }
            }
        }
        if (ifConditions.size() > 0)
            ifConditions.remove(ifConditions.size() - 1);
    }

    // extract all expression statements from a statement
    private static void processStatement(Statement statement, List<Node> expressionStmtList) {
        // Check if the statement is an ExpressionStmt
        if (statement instanceof ExpressionStmt) {
            ExpressionStmt expressionStmt = (ExpressionStmt) statement;
            // Add the ExpressionStmt to the list
            expressionStmtList.add(expressionStmt);
        }

        // Recursively check nested statements for ExpressionStmt
        if (statement.isBlockStmt()) {
            BlockStmt blockStmt = statement.asBlockStmt();
            blockStmt.getStatements().forEach(
                    nestedStatement -> processStatement(nestedStatement, expressionStmtList));
        } else if (statement.isIfStmt()) {
            IfStmt ifStmt = statement.asIfStmt();
            ifConditions.clear();
            traverseIfStmt(ifStmt, expressionStmtList);
        } else if (statement.isForStmt()) {
            ForStmt forStmt = statement.asForStmt();
            processStatement(forStmt.getBody(), expressionStmtList);
        } // Add more cases for other statement types as needed
    }

    private static Node removeNodeComment(Node node) {
        if (node.getComment().isPresent()) {
            node.getComment().get().remove();
        }
        return node;
    }

    private static void extractPathCondition(ArrayList<Node> nodeList) {
        int nodeStackSize = nodeList.size();
        if (nodeStackSize > 0 && !(nodeList.get(0) instanceof ThrowStmt)) { // add pesudo throw
            Node firstNode = nodeList.get(0);
            if (firstNode instanceof ExpressionStmt) {
                Expression expr = ((ExpressionStmt) firstNode).getExpression();
                // if it is an VariableDeclarationExpr, extract the expression
                conditionNodes.add(removeNodeComment(expr));
            } else {
                // conditionNodes.add(removeNodeComment(firstNode));
            }

        }
        // traverse the reamining nodes
        for (int i = 1; i < nodeStackSize; i++) {
            Node cur_node = nodesStack.get(i);
            if (cur_node instanceof AssignExpr) {
                extractAssign((AssignExpr) cur_node);
            } else if (cur_node instanceof MethodDeclaration) {
                Map<String, Expression> expressionMap = new HashMap<>();
                String methodName = ((MethodDeclaration) cur_node).getNameAsString();
                if (i + 1 < nodeStackSize) {
                    Node next_node = nodesStack.get(i + 1);
                    // find all methodCallExpressions in the next_node that has the same method node
                    // as cur_node
                    List<MethodCallExpr> methodCallExprs = next_node.findAll(MethodCallExpr.class);
                    for (MethodCallExpr methodCallExpr : methodCallExprs) {
                        if (methodCallExpr.getNameAsString().equals(methodName) && methodCallExpr
                                .getArguments()
                                .size() == ((MethodDeclaration) cur_node).getParameters().size()) {
                            // find all the arguments in the method call expression
                            NodeList<Expression> arguments = methodCallExpr.getArguments();
                            // find the params list in the method declaration
                            NodeList<Parameter> params =
                                    ((MethodDeclaration) cur_node).getParameters();
                            List<String> paramNames = new ArrayList<>();
                            for (Parameter param : params) {
                                paramNames.add(param.getNameAsString());
                            }
                            // add to the expression map
                            for (int j = 0; j < arguments.size(); j++) {
                                expressionMap.put(paramNames.get(j), arguments.get(j));
                            }
                            break;
                        }
                    }
                    findAndSubstituteExpr(expressionMap);
                }
            } else if (cur_node instanceof ConstructorDeclaration) {
                Map<String, Expression> expressionMap = new HashMap<>();
                String methodName = ((ConstructorDeclaration) cur_node).getNameAsString();
                if (i + 1 < nodeStackSize) {
                    Node next_node = nodesStack.get(i + 1);
                    // find all methodCallExpressions in the next_node that has the same method node
                    // as cur_node
                    List<MethodCallExpr> methodCallExprs = next_node.findAll(MethodCallExpr.class);
                    for (MethodCallExpr methodCallExpr : methodCallExprs) {
                        if (methodCallExpr.getNameAsString().equals(methodName)) {
                            // find all the arguments in the method call expression
                            NodeList<Expression> arguments = (methodCallExpr).getArguments();
                            // find the params list in the method declaration
                            NodeList<Parameter> params =
                                    ((ConstructorDeclaration) cur_node).getParameters();
                            List<String> paramNames = new ArrayList<>();
                            for (Parameter param : params) {
                                paramNames.add(param.getNameAsString());
                            }
                            // add to the expression map
                            for (int j = 0; j < arguments.size(); j++) {
                                expressionMap.put(paramNames.get(j), arguments.get(j));
                            }
                            break;
                        }
                    }
                    findAndSubstituteExpr(expressionMap);
                }
            } else if (cur_node instanceof ExpressionStmt) {
                Expression expr = null;
                if (cur_node instanceof ExpressionStmt) {
                    expr = ((ExpressionStmt) cur_node).getExpression();
                }
                if (expr instanceof VariableDeclarationExpr) { // keep this
                    extractVarDeclarExpr((VariableDeclarationExpr) expr);
                } else if (expr instanceof AssignExpr) {
                    extractAssign((AssignExpr) expr);
                }
            } else if (cur_node instanceof Expression) {
                conditionNodes.add(removeNodeComment(cur_node));
            }
        }

        return;

    }

    private static void extractVarDeclarExpr(VariableDeclarationExpr varDeclarExpr) {
        // int x = a;
        Map<String, Expression> expressionMap = new HashMap<>();
        NodeList<VariableDeclarator> varDeclarators = varDeclarExpr.getVariables();
        for (VariableDeclarator varDeclarator : varDeclarators) {
            if (varDeclarator.getInitializer().isPresent()) {
                expressionMap.put(varDeclarator.getNameAsString(),
                        varDeclarator.getInitializer().get());
            }
        }
        findAndSubstituteExpr(expressionMap);

    }

    private static void extractAssign(AssignExpr assignment) {
        // c = a + b;
        Map<String, Expression> expressionMap = new HashMap<>();
        Expression leftSide = assignment.getTarget();
        Expression rightSide = assignment.getValue();
        if (leftSide.isNameExpr()) {
            expressionMap.put(leftSide.asNameExpr().getNameAsString(), rightSide);
        }
        findAndSubstituteExpr(expressionMap);
    }

    private static void findAndSubstituteExpr(Map<String, Expression> exprMap) {
        // given the map, substitue all possible places
        for (Node n : conditionNodes) {
            if (n instanceof Expression) {
                Expression expression = (Expression) n;
                expression.findAll(NameExpr.class).forEach(nameExpr -> {
                    if (exprMap.containsKey(nameExpr.getNameAsString())) {
                        // a, b; a = c+d, b = c + e, c = c+1
                        nameExpr.replace(exprMap.get(nameExpr.getNameAsString()).clone());
                    }
                });
            }
        }

        return;
    }

    private static Node findNodeByLine(Node node, int lineNumber) {
        if (node.getRange().isPresent() && node.getRange().get().begin.line == lineNumber)
            return node;
        if (node.getRange().isPresent() && node.getRange().get().begin.line <= lineNumber
                && node.getRange().get().end.line >= lineNumber) {
            for (Node child : node.getChildNodes()) {
                Node result = findNodeByLine(child, lineNumber);
                if (result != null) {
                    return result;
                }
            }
            // return node;
        }
        return null;
    }

}
