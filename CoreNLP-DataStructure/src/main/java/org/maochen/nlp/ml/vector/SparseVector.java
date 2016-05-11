package org.maochen.nlp.ml.vector;

import java.util.HashMap;
import java.util.Map;

/**
 * Created by mguan on 5/11/16.
 */
public class SparseVector implements IVector {

    private Map<Integer, Double> sparseMap = new HashMap<>();

    @Override
    public void setVector(double[] val) {
        for (int i = 0; i < val.length; i++) {
            if (val[i] != 0) {
                sparseMap.put(i, val[i]);
            }
        }
    }

    @Override
    public double[] getVector() {
        int maxSize = sparseMap.keySet().stream().mapToInt(x -> x).max().orElse(-1);
        double[] result = new double[maxSize + 1];

        for (Integer i : sparseMap.keySet()) {
            result[i] = sparseMap.get(i);
        }

        return result;
    }
}
