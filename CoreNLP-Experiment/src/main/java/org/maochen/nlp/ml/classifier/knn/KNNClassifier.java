package org.maochen.nlp.ml.classifier.knn;

import org.maochen.nlp.ml.classifier.IClassifier;
import org.maochen.nlp.ml.datastructure.Tuple;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Simple Wrapper, Id is based on the input sequence.
 *
 * @author Maochen
 */
public class KNNClassifier implements IClassifier {
    public static final String DISTANCE = "distance";

    private List<Tuple> trainingData;

    private int k = 1;
    private int mode = 0;

    public KNNClassifier() {
    }

    /**
     * k: k nearest neighbors. mode: 0 - EuclideanDistance, 1 - ChebyshevDistance, 2 -
     * ManhattanDistance
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

    @Override
    public void persistModel(String modelFile) throws IOException {
        throw new IllegalArgumentException();
    }

    @Override
    public void loadModel(String modelFile) throws IOException {
        throw new IllegalArgumentException();
    }

    /**
     * train() method for knn is just used for loading trainingdata!!
     */
    @Override
    public IClassifier train(List<Tuple> trainingData) {
        this.trainingData = trainingData;
        return this;
    }

    /**
     * Return the predict to every other train vector's distance.
     *
     * @return return by Id which is ordered by input sequential.
     */
    @Override
    public Map<String, Double> predict(Tuple predict) {
        KNNEngine engine = new KNNEngine(predict, trainingData, k);

        if (mode == 1) {
            engine.getDistance(engine.chebyshevDistance);
        } else if (mode == 2) {
            engine.getDistance(engine.manhattanDistance);
        } else {
            engine.getDistance(engine.euclideanDistance);
        }

        predict.label = engine.getResult();

        Map<String, Double> outputMap = new ConcurrentHashMap<>();
        trainingData.parallelStream().forEach(x -> outputMap.put(String.valueOf(x.id), (Double) x.getExtra().get(DISTANCE)));

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
        Map<String, String> paraMap = new HashMap<>();
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

    /**
     Euclidean Distance:
     Prediction data: id:0 [74.0, 55.0, 22.0] -> C
     {1=81.31180726069246, 2=89.73745037608323, 3=59.66246726376642}

     Chebyshev Distance:
     Prediction data: id:0 [74.0, 55.0, 22.0] -> C
     {1=8.06225774829855, 2=8.006247560499238, 3=7.416198487095663}

     Manhattan Distance:
     Prediction data: id:0 [74.0, 55.0, 22.0] -> C
     {1=11.449890829173874, 2=11.666190466471907, 3=8.893818077743664}
     */
}
