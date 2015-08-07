package org.maochen.nlp.ml.classifier;

import org.maochen.nlp.ml.datastructure.Tuple;

import java.io.IOException;
import java.util.List;
import java.util.Map;

public interface IClassifier {
    IClassifier train(List<Tuple> trainingData);

    Map<String, Double> predict(Tuple predict);

    void setParameter(Map<String, String> paraMap);

    void persistModel(String modelFile) throws IOException;

    void loadModel(String modelFile) throws IOException;
}
