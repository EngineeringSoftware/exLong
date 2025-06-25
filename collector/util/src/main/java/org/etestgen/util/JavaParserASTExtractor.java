package org.etestgen.util;

import java.util.Collection;
import java.util.LinkedList;
import java.util.List;
import com.github.javaparser.JavaToken;
import com.github.javaparser.TokenTypes;
import com.github.javaparser.ast.Node;
import com.github.javaparser.ast.NodeList;
import com.github.javaparser.ast.comments.BlockComment;
import com.github.javaparser.ast.comments.Comment;
import com.github.javaparser.ast.comments.JavadocComment;
import com.github.javaparser.ast.comments.LineComment;
import com.github.javaparser.ast.observer.ObservableProperty;
import com.github.javaparser.printer.ConcreteSyntaxModel;
import com.github.javaparser.printer.Stringable;
import com.github.javaparser.printer.concretesyntaxmodel.CsmAttribute;
import com.github.javaparser.printer.concretesyntaxmodel.CsmChar;
import com.github.javaparser.printer.concretesyntaxmodel.CsmComment;
import com.github.javaparser.printer.concretesyntaxmodel.CsmConditional;
import com.github.javaparser.printer.concretesyntaxmodel.CsmElement;
import com.github.javaparser.printer.concretesyntaxmodel.CsmIndent;
import com.github.javaparser.printer.concretesyntaxmodel.CsmList;
import com.github.javaparser.printer.concretesyntaxmodel.CsmMix;
import com.github.javaparser.printer.concretesyntaxmodel.CsmNone;
import com.github.javaparser.printer.concretesyntaxmodel.CsmOrphanCommentsEnding;
import com.github.javaparser.printer.concretesyntaxmodel.CsmSequence;
import com.github.javaparser.printer.concretesyntaxmodel.CsmSingleReference;
import com.github.javaparser.printer.concretesyntaxmodel.CsmString;
import com.github.javaparser.printer.concretesyntaxmodel.CsmToken;
import com.github.javaparser.printer.concretesyntaxmodel.CsmUnindent;
import com.github.javaparser.utils.PositionUtils;

public class JavaParserASTExtractor {

    public static final String TERMINAL = "Terminal";
    public static final String NODE_LIST = "NodeList";

    public boolean includeComment = false;

    public JavaParserASTExtractor() {
        this(false);
    }

    public JavaParserASTExtractor(boolean includeComment) {
        this.includeComment = includeComment;
    }

    public AST extract(Node n) {
        return extract(n, ConcreteSyntaxModel.forClass(n.getClass()), null);
    }

    /* 
     * when parent is not null, n could be the node corresponding to
     * the parent of current csm
     *
     * @return a new AST node created, or null if no new AST node should be created (when this node is empty OR the children are already added to the parent node)
     */
    private AST extract(Node n, CsmElement csm, AST parent) {
        // filter out ignored tokens
        if (csm instanceof CsmNone || csm instanceof CsmIndent || csm instanceof CsmUnindent) {
            return null;
        } else if ((csm instanceof CsmComment || csm instanceof CsmOrphanCommentsEnding)
            && !includeComment) {
            return null;
        }

        AST ast = null; // used only if creating a new AST node

        // processing this CSM
        if (csm instanceof CsmToken) {
            // terminal node
            ast = new AST();
            CsmToken csmToken = (CsmToken) csm;
            JavaToken.Category category = TokenTypes.getCategory(csmToken.getTokenType());
            // ignore whitespace
            if (category.isWhitespace()) {
                return null;
            }
            ast.astType = TERMINAL;
            ast.tokKind = category.toString();
            ast.tok = csmToken.getContent(n);
        } else if (csm instanceof CsmComment) {
            // we already checked includeComment=true
            ast = new AST();
            Comment comment = (Comment) n;
            ast.astType = comment.getClass().getSimpleName();
            ast.tokKind = JavaToken.Category.COMMENT.toString();
            ast.tok = processComment(comment);

        } else if (csm instanceof CsmOrphanCommentsEnding) {
            List<Node> everything = new LinkedList<>();
            everything.addAll(n.getChildNodes());
            PositionUtils.sortByBeginPosition(everything);
            if (everything.isEmpty()) {
                return null;
            }

            int commentsAtEnd = 0;
            boolean findingComments = true;
            while (findingComments && commentsAtEnd < everything.size()) {
                Node last = everything.get(everything.size() - 1 - commentsAtEnd);
                findingComments = (last instanceof Comment);
                if (findingComments) {
                    commentsAtEnd++;
                }
            }
            for (int i = 0; i < commentsAtEnd; i++) {
                Comment c = (Comment) everything.get(everything.size() - commentsAtEnd + i);
                AST childAst = new AST();
                childAst.astType = c.getClass().getSimpleName();
                childAst.tokKind = JavaToken.Category.COMMENT.toString();
                childAst.tok = processComment(c);
                addAstChild(parent, childAst);
            }
        } else if (csm instanceof CsmSequence) {
            // non-terminal node
            ast = new AST();
            CsmSequence csmSequence = (CsmSequence) csm;
            ast.astType = n.getClass().getSimpleName();
            ast.children = new LinkedList<>();
            for (CsmElement child : csmSequence.getElements()) {
                addAstChild(ast, extract(n, child, ast));
            }
        } else if (csm instanceof CsmList) {
            // multiple children from the same field (n should be parent of current csm)
            CsmList csmList = (CsmList) csm;
            ObservableProperty property = csmList.getProperty();
            CsmElement preceeding = csmList.getPreceeding();
            CsmElement following = csmList.getFollowing();
            CsmElement separatorPre = csmList.getSeparatorPre();
            CsmElement separatorPost = csmList.getSeparatorPost();

            if (property.isAboutNodes()) {
                NodeList<? extends Node> nodeList = property.getValueAsMultipleReference(n);
                if (nodeList == null || nodeList.isEmpty()) {
                    return null;
                }
                if (preceeding != null) {
                    addAstChild(parent, extract(n, preceeding, parent));
                }
                for (int i = 0; i < nodeList.size(); ++i) {
                    if (separatorPre != null && i != 0) {
                        addAstChild(parent, extract(n, separatorPre, parent));
                    }
                    addAstChild(parent, extract(nodeList.get(i)));
                    if (separatorPost != null && i != (nodeList.size() - 1)) {
                        addAstChild(parent, extract(n, separatorPost, parent));
                    }
                }
                if (following != null) {
                    addAstChild(parent, extract(n, following, parent));
                }
            } else {
                Collection<?> values = property.getValueAsCollection(n);
                // TODO: not sure what to do in this case
                throw new RuntimeException(
                    "not sure what to do with CsmList with values; parent ast type: "
                        + n.getClass().getSimpleName() + "; property: " + property + "; values: "
                        + values);
                // if (values == null || values.isEmpty()) {
                //     return null;
                // }
                // if (preceeding != null) {
                //     parent.children.add(extract(n, preceeding));
                // }
                // for (Object value : values) {
                //     if (separatorPre != null) {
                //         parent.children.add(extract(n, separatorPre));
                //     }
                //     // TODO
                //     if (separatorPost != null) {
                //         parent.children.add(extract(n, separatorPost));
                //     }
                // }
                // if (following != null) {
                //     parent.children.add(extract(n, following));
                // }
            }
        } else if (csm instanceof CsmMix) {
            // add children to parent
            CsmMix csmMix = (CsmMix) csm;
            for (CsmElement child : csmMix.getElements()) {
                addAstChild(parent, extract(n, child, parent));
            }
        } else if (csm instanceof CsmSingleReference) {
            // add child to parent
            CsmSingleReference csmSingleReference = (CsmSingleReference) csm;
            ObservableProperty property = csmSingleReference.getProperty();
            Node childNode = property.getValueAsSingleReference(n);
            if (childNode == null) {
                return null;
            }
            addAstChild(parent, extract(childNode));
        } else if (csm instanceof CsmConditional) {
            // evaluate the condition then add the appropriate child to parent
            CsmConditional csmConditional = (CsmConditional) csm;
            CsmConditional.Condition condition = csmConditional.getCondition();
            boolean test = false;
            for (ObservableProperty prop : csmConditional.getProperties()) {
                test = test || evaluateCondition(condition, n, prop);
            }
            if (test) {
                addAstChild(parent, extract(n, csmConditional.getThenElement(), parent));
            } else {
                addAstChild(parent, extract(n, csmConditional.getElseElement(), parent));
            }
        } else if (csm instanceof CsmAttribute) {
            // add child to parent
            ast = new AST();
            CsmAttribute csmAttribute = (CsmAttribute) csm;
            ObservableProperty property = csmAttribute.getProperty();
            switch (property) {
                case IDENTIFIER:
                case NAME:
                    ast.tokKind = JavaToken.Category.IDENTIFIER.toString();
                    break;
                case KEYWORD:
                case TYPE:
                    ast.tokKind = JavaToken.Category.KEYWORD.toString();
                    break;
                case OPERATOR:
                    ast.tokKind = JavaToken.Category.OPERATOR.toString();
                    break;
                case VALUE:
                    ast.tokKind = JavaToken.Category.LITERAL.toString();
                    break;
                default:
                    throw new RuntimeException("unknown property as CsmAttribute: " + property);
            }
            ast.astType = TERMINAL;
            ast.tok = valueToString(property.getRawValue(n));
        } else if (csm instanceof CsmString) {
            CsmString csmString = (CsmString) csm;
            ast = new AST();
            ast.astType = TERMINAL;
            ast.tokKind = JavaToken.Category.LITERAL.toString();
            ast.tok = "\"" + csmString.getProperty().getValueAsStringAttribute(n) + "\"";
        } else if (csm instanceof CsmChar) {
            CsmChar csmChar = (CsmChar) csm;
            ast = new AST();
            ast.astType = TERMINAL;
            ast.tokKind = JavaToken.Category.LITERAL.toString();
            ast.tok = "'" + csmChar.getProperty().getValueAsStringAttribute(n) + "'";
        } else {
            throw new RuntimeException(
                "Unsupported CsmElement: " + csm.getClass().getSimpleName() + " in node: "
                    + n.getClass().getSimpleName());
        }

        if (ast != null) {
            // set lineno if available in node
            if (n.getRange().isPresent()) {
                ast.setLineno(n.getRange().get().begin.line, n.getRange().get().end.line);
            }
        }
        return ast;
    }

    static private void addAstChild(AST parent, AST child) {
        if (child != null) {
            parent.children.add(child);
        }
    }

    static private boolean evaluateCondition(CsmConditional.Condition condition, Node node,
        ObservableProperty property) {
        switch (condition) {
            case IS_PRESENT:
                return !property.isNullOrNotPresent(node);
            case FLAG:
                return property.getValueAsBooleanAttribute(node);
            case IS_EMPTY:
                NodeList<? extends Node> v1 = property.getValueAsMultipleReference(node);
                return v1 == null || v1.isEmpty();
            case IS_NOT_EMPTY:
                NodeList<? extends Node> v2 = property.getValueAsMultipleReference(node);
                return v2 != null && !v2.isEmpty();
        }
        throw new UnsupportedOperationException(condition.name());
    }

    static private String processComment(Comment comment) {
        if (comment instanceof BlockComment) {
            return "/*" + comment.getContent() + "*/";
        } else if (comment instanceof JavadocComment) {
            return "/**" + comment.getContent() + "*/";
        } else if (comment instanceof LineComment) {
            return "//" + comment.getContent();
        } else {
            throw new UnsupportedOperationException(comment.getClass().getSimpleName());
        }
    }

    static private String valueToString(Object value) {
        if (value instanceof Stringable) {
            return ((Stringable) value).asString();
        }
        if (value instanceof Enum) {
            return ((Enum) value).name().toLowerCase();
        } else {
            if (value != null) {
                return value.toString();
            }
        }
        return "";
    }
}
