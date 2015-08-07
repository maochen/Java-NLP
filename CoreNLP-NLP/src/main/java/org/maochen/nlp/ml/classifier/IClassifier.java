package org.maochen.nlp.ml.classifier;

import org.maochen.nlp.ml.datastructure.Tuple;

import java.util.List;
import java.util.Map;

public interface IClassifier {
    IClassifier train(List<Tuple> trainingData);

    Map<String, Double> predict(Tuple predict);

    void setParameter(Map<String, String> paraMap);
}
