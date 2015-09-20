package org.maochen.nlp.ml.vector;

/**
 * Created by Maochen on 9/18/15.
 */
public class DenseVector implements IVector {

    private double[] vector;

    @Override
    public void setVector(double[] vector) {
        this.vector = vector;
    }

    @Override
    public double[] getVector() {
        return this.vector;
    }

    public DenseVector(double[] vector) {
        this.vector = vector;
    }
}
