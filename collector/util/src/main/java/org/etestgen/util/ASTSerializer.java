package org.etestgen.util;

import java.io.IOException;
import com.fasterxml.jackson.core.JsonGenerator;
import com.fasterxml.jackson.databind.JsonSerializer;
import com.fasterxml.jackson.databind.SerializerProvider;

public class ASTSerializer extends JsonSerializer<AST> {

    @Override
    public void serialize(AST value, JsonGenerator gen, SerializerProvider serializers)
        throws IOException {
        gen.writeStartArray();

        // target format:
        // non-terminal: [astType, lineno, children...]
        // terminal: [astType:tokKind, lineno, tok]

        if (value.tokKind != null) {
            // terminal
            gen.writeString(value.astType + ":" + value.tokKind);
        } else {
            // non-terminal
            gen.writeString(value.astType);

        }

        gen.writeString(value.lineno);

        if (value.tokKind != null) {
            // terminal
            gen.writeString(value.tok);
        } else {
            // non-terminal
            for (AST child : value.children) {
                gen.writeObject(child);
            }
        }

        gen.writeString(value.rawCode);
        gen.writeEndArray();
    }
}
