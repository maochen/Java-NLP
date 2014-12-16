package org.maochen.utils;

import org.maochen.datastructure.LabelIndexer;

import java.util.HashMap;
import java.util.Map;

/**
 * Created by Maochen on 12/4/14.
 */
public class TupleUtils {
    //    public static Element inputConverter(int id, String[] entry, boolean isPredict) {
    //        int newLength = (isPredict) ? entry.length : entry.length - 1;
    //        double[] vector = new double[newLength];
    //        String label = null;
    //
    //        for (int i = 0; i < entry.length; i++) {
    //            if (!isPredict && i == (entry.length - 1)) {
    //                label = entry[i];
    //                break;
    //            }
    //
    //            vector[i] = Double.parseDouble(entry[i]);
    //        }
    //
    //        Element element = new Element(id, vector, label);
    //        return element;
    //    }

    // Convert Index to actual string.
    public static Map<String, Double> convertMap(Map<Integer, Double> probs, LabelIndexer labelIndexer) {
        Map<String, Double> stringKeyProb = new HashMap<>();
        probs.entrySet().stream().forEach(e -> stringKeyProb.put(labelIndexer.getLabel(e.getKey()), e.getValue()));
        return stringKeyProb;
    }


}
