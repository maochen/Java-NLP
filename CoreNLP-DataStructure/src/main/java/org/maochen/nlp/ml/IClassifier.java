package org.maochen.nlp.ml;

import java.io.IOException;
import java.io.InputStream;
import java.util.List;
import java.util.Map;
import java.util.Properties;

public interface IClassifier {
    IClassifier train(List<Tuple> trainingData);

    Map<String, Double> predict(Tuple predict);

    void setParameter(Properties props);

    void persistModel(String modelFile) throws IOException;

    void loadModel(InputStream modelFile);
}
