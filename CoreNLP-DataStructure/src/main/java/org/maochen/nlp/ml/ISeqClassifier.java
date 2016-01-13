package org.maochen.nlp.ml;

import org.apache.commons.lang3.tuple.Pair;

import java.io.IOException;
import java.io.InputStream;
import java.util.List;
import java.util.Properties;

/**
 * Created by Maochen on 1/12/16.
 */
public interface ISeqClassifier {
    ISeqClassifier train(List<SequenceTuple> trainingData);

    // For each tuple, generate a map with (label, confidence score)
    List<Pair<String, Double>> predict(SequenceTuple predict);

    void setParameter(Properties props);

    void persistModel(String modelFile) throws IOException;

    void loadModel(InputStream modelFile);
}
