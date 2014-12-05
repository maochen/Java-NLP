package org.maochen.datastructure;

/**
 * Created by Maochen on 12/3/14.
 */
public class Element {
    public int id;
    public String label;
    public double[] featureVector;
    public double distance;


    // This is for predict
    public Element(double[] featureVector) {
        this.featureVector = featureVector;
    }

    // This is for training data
    public Element(int id, double[] featureVector, String label) {
        this.id = id;
        this.featureVector = featureVector;
        this.label = label;
    }
}
