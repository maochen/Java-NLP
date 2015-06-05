package org.maochen.classifier.perceptron;

import org.maochen.classifier.IClassifier;
import org.maochen.datastructure.Tuple;

import java.util.List;
import java.util.Map;

/**
 * Created by Maochen on 6/5/15.
 */
public class Perceptron implements IClassifier {

    PerceptronModel model;

    double threshold=0.5;

    @Override
    public IClassifier train(List<Tuple> trainingData) {
        model = new PerceptronModel();



        return null;
    }

    @Override
    public Map<String, Double> predict(Tuple predict) {
        return null;
    }

    @Override
    public void setParameter(Map<String, String> paraMap) {

    }
}
