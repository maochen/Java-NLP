package org.maochen.datastructure;

import org.apache.commons.lang3.StringUtils;

import java.util.Arrays;
import java.util.stream.IntStream;

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
        // Generate a[index]=index
        featureVectorIndex = IntStream.range(0, featureVector.length).parallel().toArray();
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
        return "id:" + id + StringUtils.SPACE + Arrays.toString(featureVector) + " -> " + label;
    }
}
