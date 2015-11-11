package org.maochen.nlp.ml.vector;

import java.util.stream.IntStream;

/**
 * Created by Maochen on 9/19/15.
 */
public class LabeledVector extends DenseVector {
    public String[] featsName = null;

    public LabeledVector(double[] vector) {
        super(vector);
    }

    public LabeledVector(String[] feats) {
        super(IntStream.range(0, feats.length).mapToDouble(x -> 1.0D).toArray());
        this.featsName = feats;
    }
}
