package org.maochen.datastructure;

import java.util.Arrays;

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
    }

    // This is for training data
    public Tuple(int id, double[] featureVector, String label) {
        this.id = id;
        this.featureVector = featureVector;
        this.label = label;
    }

    @Override
    public String toString() {
        return "id:" + id + " " + Arrays.toString(featureVector) + " -> " + label;
    }
}
