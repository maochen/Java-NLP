package org.maochen.nlp.ml.vector;

import java.util.Arrays;

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

    @Override
    public boolean equals(Object o) {
        if (!(o instanceof DenseVector)) {
            return false;
        }

        double[] oVector = ((DenseVector) o).getVector();
        if (this.vector.length != oVector.length) {
            return false;
        }

        for (int i = 0; i < this.vector.length; i++) {
            if (this.vector[i] != oVector[i]) {
                return false;
            }
        }

        return true;
    }

    @Override
    public int hashCode() {
        return Arrays.hashCode(vector);
    }

    public DenseVector(double[] vector) {
        this.vector = vector;
    }
}
