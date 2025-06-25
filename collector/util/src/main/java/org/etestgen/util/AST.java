package org.etestgen.util;

import java.util.LinkedList;
import java.util.List;
import com.fasterxml.jackson.databind.annotation.JsonSerialize;

@JsonSerialize(using = ASTSerializer.class)
public class AST {
    // Source Code
    public String rawCode = null;
    // the AST type
    public String astType = null;
    // (terminal only) the token kind
    public String tokKind = null;

    // (terminal only) the literal token
    public String tok = null;
    // (non-terminal only) the children
    public List<AST> children = null;

    // lineno range of this node
    public String lineno = null;

    public void setLineno(int linenoBeg, int linenoEnd) {
        if (linenoBeg == linenoEnd) {
            lineno = String.valueOf(linenoBeg);
        } else {
            lineno = String.valueOf(linenoBeg) + "-" + String.valueOf(linenoEnd);
        }
    }

    public List<String> getTokens() {
        List<String> tokens = new LinkedList<>();
        if (tok != null) {
            tokens.add(tok);
        } else {
            for (AST child : children) {
                tokens.addAll(child.getTokens());
            }
        }
        return tokens;
    }
}
