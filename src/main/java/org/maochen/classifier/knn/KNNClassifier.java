package org.maochen.classifier.knn;

import org.maochen.classifier.IClassifier;
import org.maochen.datastructure.Tuple;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Simple Wrapper, Id is based on the input sequence.
 *
 * @author Maochen
 */
public class KNNClassifier implements IClassifier{

    private List<Tuple> trainingData;

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
        this.k = 1;
        this.mode = 0;
        engine = new KNNEngine();
    }

    /**
     * train() method for knn is just used for loading trainingdata!!
     */
    @Override
    public IClassifier train(List<Tuple> trainingData) {
        this.trainingData = new ArrayList<>();

        for (Tuple t : trainingData) {
            this.trainingData.add(t);
        }
        return this;
    }

    /**
     * Return the predict to every other train vector's distance.
     *
     * @return return by Id which is ordered by input sequential.
     */
    @Override
    public Map<String, Double> predict(Tuple predict) {
        engine.initialize(predict, trainingData, k);
        if (mode == 1) {
            engine.ChebyshevDistance();
        } else if (mode == 2) {
            engine.ManhattanDistance();
        } else {
            engine.EuclideanDistance();
        }

        String result = engine.getResult();
        predict.label = result;

        Map<String, Double> outputMap = new HashMap<String, Double>();

        for (Tuple dtos : trainingData) {
            outputMap.put(String.valueOf(dtos.id), dtos.distance);
        }
        return outputMap;
    }

    public static void main(String[] args) {
        int k = 3;

        Tuple vectorA = new Tuple(1, new double[]{9, 32, 65.1}, "A");
        Tuple vectorB = new Tuple(2, new double[]{12, 65, 86.1}, "C");
        Tuple vectorC = new Tuple(3, new double[]{19, 54, 45.1}, "C");

        List<Tuple> trainList = new ArrayList<>();
        trainList.add(vectorA);
        trainList.add(vectorB);
        trainList.add(vectorC);

        Tuple predict = new Tuple(new double[]{74, 55, 22});

        IClassifier knn = new KNNClassifier();
        knn.train(trainList);
        Map<String, String> paraMap = new HashMap<String, String>();
        paraMap.put("k", String.valueOf(k));

        System.out.println("Euclidean Distance:");
        paraMap.put("mode", "0");
        knn.setParameter(paraMap);
        Map<String, Double> details = knn.predict(predict);
        System.out.println("Prediction data: " + predict);
        System.out.println(details);
        System.out.println();

        System.out.println("Chebyshev Distance:");
        paraMap.put("mode", "1");
        knn.setParameter(paraMap);
        details = knn.predict(predict);
        System.out.println("Prediction data: " + predict);
        System.out.println(details);
        System.out.println();

        System.out.println("Manhattan Distance:");
        paraMap.put("mode", "2");
        knn.setParameter(paraMap);
        details = knn.predict(predict);
        System.out.println("Prediction data: " + predict);
        System.out.println(details);
    }

}
