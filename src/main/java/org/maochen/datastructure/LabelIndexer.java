package org.maochen.datastructure;

import com.google.common.collect.BiMap;
import com.google.common.collect.HashBiMap;

import java.util.Set;

/**
 * Created by Maochen on 12/4/14.
 */
public class LabelIndexer {
    // They are pairs
    private BiMap<String, Integer> labelIndexer = HashBiMap.create();
    int nextId = 0;

    public int getIndex(String label) {
        return labelIndexer.get(label);
    }

    public void putByLabel(String label) {
        labelIndexer.put(label, nextId++);
    }

    public boolean hasLabel(String label) {
        return labelIndexer.containsKey(label);
    }

    public Set<Integer> getIndexSet() {
        return labelIndexer.inverse().keySet();
    }

    public String getLabel(int index) {
        return labelIndexer.inverse().get(index);
    }
}
