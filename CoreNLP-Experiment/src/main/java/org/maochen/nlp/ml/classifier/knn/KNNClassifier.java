package org.maochen.nlp.ml.classifier.knn;

import org.maochen.nlp.ml.IClassifier;
import org.maochen.nlp.ml.Tuple;

import java.io.IOException;
import java.io.InputStream;
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
     * k: k nearest neighbors. mode: 0 - EuclideanDistance, 1 - ChebyshevDistance, 2 - ManhattanDistance
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
    public void loadModel(InputStream inputStream) {
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
}
