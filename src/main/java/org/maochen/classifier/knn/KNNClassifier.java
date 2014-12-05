package org.maochen.classifier.knn;

import org.maochen.classifier.IClassifier;
import org.maochen.datastructure.Element;
import org.maochen.utils.ElementUtils;

import java.util.*;

/**
 * Simple Wrapper, Id is based on the input sequential.
 *
 * @author Maochen
 */
public class KNNClassifier implements IClassifier {

    private List<Element> trainingData;
    private Element predict;

    private int idCounter = 1;
    private int k;
    private int mode;

    private KNNEngine engine;


    /**
     * k: k nearest neighbors.
     * mode: 0 - EuclideanDistance, 1 - ChebyshevDistance, 2 - ManhattanDistance
     *
     * @param paraMap Parameters Map.
     */
    @Override
    public void setParameter(Map<String, String> paraMap) {
        if (paraMap.containsKey("k")) {
            this.k = Integer.parseInt(paraMap.get("k"));
        }
        if (paraMap.containsKey("mode")) {
            this.mode = Integer.parseInt(paraMap.get("mode"));
        }
    }

    public KNNClassifier() {
        this.idCounter = 0;
        this.k = 1;
        this.mode = 0;
        engine = new KNNEngine();
    }

    /**
     * train() method for knn is just used for loading trainingdata!!
     */
    @Override
    public IClassifier train(List<String[]> trainingData) {
        this.trainingData = new ArrayList<>();

        for (String[] entry : trainingData) {
            this.trainingData.add(ElementUtils.inputConverter(idCounter, entry, false));
            idCounter++;
        }

        return this;
    }

    /**
     * Return the predict to every other train vector's distance.
     *
     * @return return by Id which is ordered by input sequential.
     */
    @Override
    public Map<String, Double> predict(String[] predict) {
        this.predict = ElementUtils.inputConverter(idCounter, predict, true);
        idCounter++;
        engine.initialize(this.predict, trainingData, k);
        if (mode == 1) {
            engine.ChebyshevDistance();
        } else if (mode == 2) {
            engine.ManhattanDistance();
        } else {
            engine.EuclideanDistance();
        }

        engine.getResult();

        Map<String, Double> outputMap = new HashMap<String, Double>();

        for (Element dtos : trainingData) {
            outputMap.put(String.valueOf(dtos.id), dtos.distance);
        }
        return outputMap;
    }

    @Override
    public String getResult() {
        return predict.label;
    }


    public static void main(String[] args) {
        int k = 3;

        String[] vectorA = new String[]{"9", "32", "65.1", "A"};
        String[] vectorB = new String[]{"12", "65", "86.1", "C"};
        String[] vectorC = new String[]{"19", "54", "45.1", "C"};

        List<String[]> trainList = new ArrayList<String[]>();
        trainList.add(vectorA);
        trainList.add(vectorB);
        trainList.add(vectorC);

        String[] predict = new String[]{"74", "55", "22"};

        IClassifier knn = new KNNClassifier().train(trainList);
        Map<String, String> paraMap = new HashMap<String, String>();
        paraMap.put("k", String.valueOf(k));

        System.out.println("Euclidean Distance:");
        paraMap.put("mode", "0");
        knn.setParameter(paraMap);
        Map<String, Double> details = knn.predict(predict);
        System.out.println("Prediction data: " + Arrays.toString(predict) + " -> " + knn.getResult());
        System.out.println(details);

        System.out.println("Chebyshev Distance:");
        paraMap.put("mode", "1");
        knn.setParameter(paraMap);
        details = knn.predict(predict);
        System.out.println("Prediction data: " + Arrays.toString(predict) + " -> " + knn.getResult());
        System.out.println(details);

        System.out.println("Manhattan Distance:");
        paraMap.put("mode", "2");
        knn.setParameter(paraMap);
        details = knn.predict(predict);
        System.out.println("Prediction data: " + Arrays.toString(predict) + " -> " + knn.getResult());
        System.out.println(details);
    }

}
