package org.maochen.datastructure;

import java.util.HashMap;
import java.util.Map;
import java.util.Set;

/**
 * Created by Maochen on 12/4/14.
 */
public class LabelIndexer {
    // They are pairs
    private Map<String, Integer> labelIndexer = new HashMap<>();

    int nextId = 0;

    public int getIndex(String str) {
        return labelIndexer.get(str);
    }

    public void put(String label) {
        labelIndexer.put(label, nextId++);
    }

    public boolean hasLabel(String label) {
        return labelIndexer.containsKey(label);
    }

    public Set<String> getAllLabels() {
        return labelIndexer.keySet();
    }

    public String getLabel(int index) {
        String label = "";

        for (String labelKey : labelIndexer.keySet()) {
            if (labelIndexer.get(labelKey) == index) {
                label = labelKey;
                break;
            }
        }
        return label;
    }
}
