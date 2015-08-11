package org.maochen.nlp.parser;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

@SuppressWarnings("serial")
/**
 * Row, Column, Value
 */
public class DoubleKeyMap<K1, K2, V> implements Serializable{

    private Map<K1, Map<K2, V>> data;
    private Map<K2, Set<K1>> k2Cache;

    public DoubleKeyMap() {
        data = new HashMap<>();
        k2Cache = new HashMap<>();
    }

    public void clear() {
        data = new HashMap<>();
        k2Cache = new HashMap<>();
    }

    public boolean containsKey1(K1 k1) {
        return data.containsKey(k1);
    }

    public boolean containsKey2(K2 k2) {
        return k2Cache.containsKey(k2);
    }

    public boolean containsValue(V value) {
        for (K1 key1 : data.keySet()) {
            for (K2 key2 : data.get(key1).keySet()) {
                V fetchedValue = data.get(key1).get(key2);
                if (fetchedValue.equals(value)) {
                    return true;
                }
            }
        }
        return false;
    }

    public Map<K2, V> row(K1 key1) {
        if (!data.containsKey(key1)) return null;
        return data.get(key1);
    }

    public Map<K1, V> column(K2 key2) {
        if (!k2Cache.containsKey(key2)) return null;

        Map<K1, V> result = new HashMap<K1, V>();
        Set<K1> k1Set = k2Cache.get(key2);
        for (K1 key1 : k1Set) {
            result.put(key1, data.get(key1).get(key2));
        }
        return result;
    }

    public V get(K1 key1, K2 key2) {
        if (key1 == null || key2 == null) {
            throw new RuntimeException("[DoubleKeyMap.key] Please fill all keys.");
        }
        return data.get(key1).get(key2);
    }

    public boolean isEmpty() {
        return data.isEmpty();
    }

    public V put(K1 key1, K2 key2, V value) {
        if (key1 == null || key2 == null || value == null) {
            throw new RuntimeException("[DoubleKeyMap.put] Please fill all keys and value.");
        }

        Map<K2, V> entry = data.containsKey(key1) ? data.get(key1) : new HashMap<K2, V>();
        entry.put(key2, value);
        data.put(key1, entry);

        Set<K1> k2CacheK1Set = k2Cache.containsKey(key2) ? k2Cache.get(key2) : new HashSet<K1>();
        k2CacheK1Set.add(key1);
        k2Cache.put(key2, k2CacheK1Set);

        return value;
    }

    public V remove(K1 key1, K2 key2) {
        if (key1 == null || key2 == null) {
            throw new RuntimeException("[DoubleKeyMap.remove] Please fill all keys.");
        }

        V removedObject = null;
        boolean isValueExisted = false;

        if (data.containsKey(key1)) {
            Map<K2, V> entry = data.get(key1);
            if (entry.containsKey(key2)) {
                isValueExisted = true;
            }
        }

        if (isValueExisted) {
            Map<K2, V> entry = data.get(key1);
            removedObject = entry.remove(key2);
            k2Cache.get(key2).remove(key1);
        }

        return removedObject;

    }

    public Map<K2, V> removeK1(K1 key1) {
        Set<K2> key2List = data.get(key1).keySet();
        for (K2 key2 : key2List) {
            k2Cache.get(key2).remove(key1);
        }

        return data.remove(key1);

    }

    public Map<K1, V> removeK2(K2 key2) {
        Map<K1, V> result = new HashMap<K1, V>();

        Set<K1> key1List = k2Cache.get(key2);
        for (K1 key1 : key1List) {
            result.put(key1, data.get(key1).remove(key2));
        }

        k2Cache.remove(key2);
        return result;
    }

    public Set<K1> rowKeySet() {
        return data.keySet();
    }

    public Set<K2> key2Set() {
        return k2Cache.keySet();
    }

    public int size() {
        int size = 0;
        for (K1 key1 : data.keySet()) {
            size += data.get(key1).keySet().size();
        }
        return size;
    }

    public String toString() {
        List<String> str = new ArrayList<String>();
        for (K1 key1 : data.keySet()) {
            for (K2 key2 : data.get(key1).keySet()) {
                String formattedStr = "[" + key1 + ", " + key2 + "]=" + data.get(key1).get(key2) + "\n";
                str.add(formattedStr);
            }

        }

        return str.toString();

    }
}
