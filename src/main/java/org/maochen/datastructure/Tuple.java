package org.maochen.datastructure;

import java.util.Arrays;

/**
 * Created by Maochen on 12/3/14.
 */
public class Tuple {
    public int id;
    public String label;
    public double[] featureVector;
    public int[] featureVectorIndex;
    public double distance;


    private void getFeatureVectorIndex() {
        featureVectorIndex = new int[featureVector.length];
        for (int i = 0; i < featureVector.length; i++) {
            featureVectorIndex[i] = i;
        }
    }

    // This is for predict
    public Tuple(double[] featureVector) {
        this.featureVector = featureVector;
        getFeatureVectorIndex();
    }

    // This is for training data
    public Tuple(int id, double[] featureVector, String label) {
        this.id = id;
        this.featureVector = featureVector;
        this.label = label;
        getFeatureVectorIndex();
    }

    @Override
    public String toString() {
        return "id:" + id + " " + Arrays.toString(featureVector) + " -> " + label;
    }
}
