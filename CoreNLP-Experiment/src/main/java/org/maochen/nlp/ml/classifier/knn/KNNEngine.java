package org.maochen.nlp.ml.classifier.knn;

import org.apache.commons.lang3.StringUtils;
import org.maochen.nlp.ml.Tuple;
import org.maochen.nlp.utils.VectorUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.BiFunction;

/**
 * Should not exposed.
 *
 * @author Maochen
 */
final class KNNEngine {
    private static final Logger LOG = LoggerFactory.getLogger(KNNEngine.class);

    private Tuple predict;
    private List<Tuple> trainingData;
    private int k;

    public KNNEngine(Tuple predict, List<Tuple> trainingData, int k) {
        this.predict = predict;
        this.trainingData = trainingData;
        this.k = k;
    }

    public BiFunction<double[], double[], Double> euclideanDistance = (v1, v2) -> Math.sqrt(Arrays.stream(VectorUtils.zip(v1, v2, (x, y) -> Math.pow(x - y, 2))).parallel().sum());
    public BiFunction<double[], double[], Double> chebyshevDistance = (v1, v2) -> Math.sqrt(Arrays.stream(VectorUtils.zip(v1, v2, (x, y) -> Math.abs(x - y))).max().getAsDouble());
    public BiFunction<double[], double[], Double> manhattanDistance = (v1, v2) -> Math.sqrt(Arrays.stream(VectorUtils.zip(v1, v2, (x, y) -> Math.abs(x - y))).parallel().sum());

    public void getDistance(BiFunction<double[], double[], Double> distanceFunction) {
        for (Tuple tuple : trainingData) {
            if (predict.featureVector.length != tuple.featureVector.length) {
                LOG.error("2 Vectors must has same dimension.");
                return;
            }
            tuple.addExtra(KNNClassifier.DISTANCE, distanceFunction.apply(predict.featureVector, tuple.featureVector));
        }
    }

    public String getResult() {
        Map<String, Integer> resultMap = new HashMap<>();
        trainingData.parallelStream().sorted((tuple1, tuple2) -> {
            double diff = (Double) tuple1.getExtra().get(KNNClassifier.DISTANCE) - (Double) tuple2.getExtra().get(KNNClassifier.DISTANCE);
            if (Math.abs(diff) < Double.MIN_VALUE) {
                return 0;
            } else {
                return ((Double) tuple1.getExtra().get(KNNClassifier.DISTANCE)).compareTo((Double) tuple2.getExtra().get(KNNClassifier.DISTANCE));
            }
        });

        for (int i = 0; i < k; i++) {
            Tuple tuple = trainingData.get(i);
            int count = resultMap.containsKey(tuple.label) ? resultMap.get(tuple.label) : 0;
            count++;
            resultMap.put(tuple.label, count);
        }

        String maxVote = StringUtils.EMPTY;
        int maxCount = 0;
        int maxCountEntryNumber = 0; // equal vote

        for (String label : resultMap.keySet()) {
            int currentCount = resultMap.get(label);

            if (currentCount == maxCount) {
                maxCountEntryNumber++;
            } else if (currentCount > maxCount) {
                maxCount = currentCount;
                maxVote = label;
                maxCountEntryNumber = 1;
            }

        }

        if (maxCountEntryNumber != 1) {
            LOG.info("Equal Max Vote, take the first max!");
        }
        predict.label = maxVote;
        return maxVote;
    }
}
