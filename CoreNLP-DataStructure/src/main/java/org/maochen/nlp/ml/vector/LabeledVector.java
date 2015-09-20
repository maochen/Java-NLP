package org.maochen.nlp.ml.vector;

/**
 * Created by Maochen on 9/19/15.
 */
public class LabeledVector extends DenseVector {
    public String[] featsName = null;

    public LabeledVector(double[] vector) {
        super(vector);
    }
}
