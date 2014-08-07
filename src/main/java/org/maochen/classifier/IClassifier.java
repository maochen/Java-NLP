package org.maochen.classifier;


import java.util.List;
import java.util.Map;

public interface IClassifier {
    public IClassifier train(List<String[]> trainingData);

    public Map<String, Double> predict(String[] featureVector);

    public String getResult();
    
    public void setParameter(Map<String, String> paraMap);

}
