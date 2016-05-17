package org.maochen.nlp.ml.classifier;

import com.google.common.collect.BiMap;
import com.google.common.collect.HashBiMap;
import org.maochen.nlp.ml.Tuple;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * Created by Maochen on 12/4/14.
 */
public class LabelIndexer {
    private static final Logger LOG = LoggerFactory.getLogger(LabelIndexer.class);

    // They are pairs
    public BiMap<String, Integer> labelIndexer = HashBiMap.create();

    public int getIndex(String label) {
        return labelIndexer.get(label);
    }

    public String getLabel(int index) {
        return labelIndexer.inverse().get(index);
    }

    public void putByLabels(List<String> labels) {
        Collections.sort(labels);
        int maxIndex = labelIndexer.values().stream().max(Integer::compareTo).orElse(-1);
        IntStream.range(0, labels.size())
                .forEachOrdered(i -> labelIndexer.put(labels.get(i), maxIndex + 1 + i));
    }

    public boolean hasLabel(String label) {
        return labelIndexer.containsKey(label);
    }

    public Set<Integer> getIndexSet() {
        return labelIndexer.inverse().keySet();
    }

    public int getLabelSize() {
        return labelIndexer.size();
    }

    // Convert Index to actual string.
    public Map<String, Double> convertMapKey(Map<Integer, Double> probs) {
        Map<String, Double> stringKeyProb = new HashMap<>();
        probs.entrySet().stream().forEach(e -> stringKeyProb.put(getLabel(e.getKey()), e.getValue()));
        return stringKeyProb;
    }

    public String serializeToString() {
        return labelIndexer.entrySet().stream()
                .map(e -> e.getKey() + System.lineSeparator() + e.getValue())
                .collect(Collectors.joining(System.lineSeparator()));
    }

    public void readFromSerializedString(String str) {
        if (str == null || str.isEmpty()) {
            throw new IllegalArgumentException("serialized string is invalid.");
        }

        String[] fields = str.split(System.lineSeparator());
        for (int i = 0; i < fields.length; i += 2) {
            labelIndexer.put(fields[i], Integer.parseInt(fields[i + 1]));
        }

        LOG.info("Successfully loaded " + labelIndexer.size() + " indices.");
    }

    public LabelIndexer(final List<Tuple> trainingData) {
        List<String> labels = trainingData.parallelStream().map(tuple -> tuple.label).distinct().collect(Collectors.toList());
        putByLabels(labels);
    }
}
