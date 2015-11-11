package org.maochen.nlp.app;

import org.maochen.nlp.ml.IClassifier;
import org.maochen.nlp.parser.DTree;

import java.util.Map;

/**
 * Created by Maochen on 11/10/15.
 */
public interface ITagger extends IClassifier {
    void train(String trainFilePath);

    Map<String, Double> predict(String sentence);

    Map<String, Double> predict(DTree tree);
}
