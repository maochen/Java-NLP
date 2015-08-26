package org.maochen.nlp.ml.classifier;

import org.maochen.nlp.ml.Tuple;

import java.io.IOException;
import java.io.InputStream;
import java.util.List;
import java.util.Map;

public interface IClassifier {
    IClassifier train(List<Tuple> trainingData);

    Map<String, Double> predict(Tuple predict);

    void setParameter(Map<String, String> paraMap);

    void persistModel(String modelFile) throws IOException;

    void loadModel(InputStream modelFile);
}
