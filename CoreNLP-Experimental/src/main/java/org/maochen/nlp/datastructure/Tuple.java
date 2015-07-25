package org.maochen.nlp.datastructure;

import org.apache.commons.lang3.StringUtils;

/**
 * Created by Maochen on 12/3/14.
 */
public class Tuple {
    public int id;
    public String label;
    public double[] featureVector;
    public double distance;

    // This is for predict
    public Tuple(double[] featureVector) {
        this.featureVector = featureVector;
        label = null;
    }

    // This is for training data
    public Tuple(int id, double[] featureVector, String label) {
        this.id = id;
        this.featureVector = featureVector;
        this.label = label;
    }

    @Override
    public String toString() {
        StringBuilder stringBuilder = new StringBuilder();
        stringBuilder.append(label).append(StringUtils.SPACE);
        for (double v : featureVector) {
            stringBuilder.append(v).append(StringUtils.SPACE);
        }

        return stringBuilder.toString().trim();
    }
}
