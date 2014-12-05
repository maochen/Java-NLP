package org.maochen.utils;

import org.maochen.datastructure.Element;
import org.maochen.datastructure.LabelIndexer;

import java.util.Map;

/**
 * Created by Maochen on 12/4/14.
 */
public class ElementUtils {
    public static Element inputConverter(int id, String[] entry, boolean isPredict) {
        int newLength = (isPredict) ? entry.length : entry.length - 1;
        double[] vector = new double[newLength];
        String label = null;

        for (int i = 0; i < entry.length; i++) {
            if (!isPredict && i == (entry.length - 1)) {
                label = entry[i];
                break;
            }

            vector[i] = Double.parseDouble(entry[i]);
        }

        Element element = new Element(id, vector, label);
        return element;
    }

    public static void print(Map<Integer, Double> probs, LabelIndexer labelIndexer) {
        int maxLabelIndex = 0;
        for (Integer index : probs.keySet()) {
            String label = labelIndexer.getLabel(index);
            System.out.println(label + "\t:\t" + probs.get(index));

            if (probs.get(index) > probs.get(maxLabelIndex)) {
                maxLabelIndex = index;
            }

        }

        System.out.println("Result\t:\t" + labelIndexer.getLabel(maxLabelIndex));
    }


}
