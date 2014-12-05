package org.maochen.classifier;


import org.maochen.datastructure.Element;

import java.util.List;
import java.util.Map;

public interface IClassifier {
    public IClassifier train(List<Element> trainingData);

    public Map<String, Double> predict(Element predict);

    public void setParameter(Map<String, String> paraMap);

}
