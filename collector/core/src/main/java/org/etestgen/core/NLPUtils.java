package org.etestgen.core;
import edu.stanford.nlp.simple.Document;

import java.nio.charset.Charset;
import java.util.Optional;

public class NLPUtils {
    
    public static Optional<String> getFirstSentence(String str) {
        try {
            Document document = new Document(str);
            return Optional.of(document.sentence(0).toString());
        } catch(Exception e) {
            System.err.println("Cannot get first sentence of: " + str);
            return Optional.empty();
        }
    }
    
    public static boolean isValidISOLatin(String s) {
        return Charset.forName("US-ASCII").newEncoder().canEncode(s);
    }
}

