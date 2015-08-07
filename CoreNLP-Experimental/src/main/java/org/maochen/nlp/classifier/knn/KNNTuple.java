package org.maochen.nlp.classifier.knn;

import org.maochen.nlp.datastructure.Tuple;

/**
 * Created by Maochen on 8/6/15.
 */
public class KNNTuple extends Tuple {
    public double distance;

    public KNNTuple(int id, double[] featureVector, String label) {
        super(id, featureVector, label);
    }

    public KNNTuple(double[] featureVector) {
        super(featureVector);
    }

    public KNNTuple(Tuple t) {
        super(t.id, t.featureVector, t.label);
    }
}
