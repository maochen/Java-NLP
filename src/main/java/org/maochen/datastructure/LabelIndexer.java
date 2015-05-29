package org.maochen.datastructure;

import com.google.common.collect.BiMap;
import com.google.common.collect.HashBiMap;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * Created by Maochen on 12/4/14.
 */
public class LabelIndexer {
    // They are pairs
    private BiMap<String, Integer> labelIndexer = HashBiMap.create();

    public int getIndex(String label) {
        return labelIndexer.get(label);
    }

    public String getLabel(int index) {
        return labelIndexer.inverse().get(index);
    }

    public void putByLabels(List<String> labels) {
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

    public LabelIndexer(final List<Tuple> trainingData) {
        List<String> labels = trainingData.parallelStream().map(tuple -> tuple.label).distinct().collect(Collectors.toList());
        putByLabels(labels);
    }
}
