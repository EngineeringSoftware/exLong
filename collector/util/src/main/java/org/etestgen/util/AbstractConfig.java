package org.etestgen.util;

import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.Reader;
import java.lang.reflect.Field;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.function.Function;
import org.apache.commons.lang3.tuple.Pair;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;


public abstract class AbstractConfig {

    public boolean repOk() {
        return true;
    }

    public void autoInfer() {}

    private static final Map<Class<?>, Function<JsonNode, Object>> SUPPORTED_CLASSES_2_READERS;
    static {
        Map<Class<?>, Function<JsonNode, Object>> supportedClasses2Readers = new HashMap<>();
        supportedClasses2Readers.put(Boolean.TYPE, JsonNode::asBoolean);
        supportedClasses2Readers.put(Boolean.class, JsonNode::asBoolean);
        supportedClasses2Readers.put(Integer.TYPE, JsonNode::asInt);
        supportedClasses2Readers.put(Integer.class, JsonNode::asInt);
        supportedClasses2Readers.put(Double.TYPE, JsonNode::asDouble);
        supportedClasses2Readers.put(Double.class, JsonNode::asDouble);
        supportedClasses2Readers.put(String.class, JsonNode::asText);
        supportedClasses2Readers.put(Path.class, e -> Paths.get(e.asText()));
        SUPPORTED_CLASSES_2_READERS = Collections.unmodifiableMap(supportedClasses2Readers);
    }

    public static <T extends AbstractConfig> T load(Path configPath, Class<T> clz) {
        try {
            return load(new FileReader(configPath.toFile()), clz);
        } catch (FileNotFoundException e) {
            throw new RuntimeException("Could not load config " + configPath, e);
        }
    }

    /**
     * Loads the config of type {@code T} from the reader, which should provide a json dict.
     * @param reader  the reader that provides the config
     * @param clz  T.class
     * @param <T>  the type of the config
     * @return  the loaded config
     */
    public static <T extends AbstractConfig> T load(Reader reader, Class<T> clz) {
        try {
            T config = clz.newInstance();

            Map<String, Pair<Class<?>, Field>> options = new HashMap<>();
            for (Field f : clz.getDeclaredFields()) {
                if (f.getAnnotation(Option.class) != null) {
                    Class<?> fType = f.getType();
                    if (!SUPPORTED_CLASSES_2_READERS.containsKey(fType)) {
                        throw new RuntimeException(
                            "Unsupported option type " + fType + ", for field " + f.getName()
                                + " in class " + clz);
                    }
                    options.put(f.getName(), Pair.of(fType, f));
                }
            }

            ObjectMapper mapper = new ObjectMapper();
            ObjectNode rootNode = (ObjectNode) mapper.readTree(reader);
            for (Map.Entry<String, JsonNode> entry : (Iterable<Map.Entry<String, JsonNode>>) () -> rootNode
                .fields()) {
                Pair<Class<?>, Field> p = options.get(entry.getKey());
                if (p != null) {
                    p.getRight().setAccessible(true);
                    p.getRight().set(
                        config,
                        SUPPORTED_CLASSES_2_READERS.get(p.getLeft()).apply(entry.getValue()));
                } else {
                    throw new RuntimeException("Unknown config key " + entry.getKey());
                }
            }

            config.autoInfer();
            if (!config.repOk()) {
                throw new RuntimeException("Config invalid: " + config.toString());
            }

            return config;
        } catch (IOException e) {
            throw new RuntimeException("Cannot read config", e);
        } catch (RuntimeException e) {
            throw new RuntimeException("Malformed config", e);
        } catch (IllegalAccessException | InstantiationException e) {
            throw new RuntimeException("The config class " + clz + " is not correctly set up", e);
        }
    }
}
