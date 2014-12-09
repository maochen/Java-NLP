package org.maochen.classifier;


import org.maochen.datastructure.Tuple;

import java.util.List;
import java.util.Map;

public interface IClassifier {
    public IClassifier train(List<Tuple> trainingData);

    public Map<String, Double> predict(Tuple predict);

    public void setParameter(Map<String, String> paraMap);

}
