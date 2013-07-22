package classifier;


import java.util.List;
import java.util.Map;

public interface IClassifier {
    public IClassifier train(List<String[]> trainingdata);

    public Map<String, Double> predict(String[] featureVector);

    public String getResult();
    
    public void setParameter(Map<String, String> paraMap);

}
