package org.maochen.classifier.knn;

import org.apache.commons.lang3.StringUtils;
import org.maochen.datastructure.Tuple;
import org.maochen.utils.VectorUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;

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

    public void initialize(Tuple predict, List<Tuple> trainingData, int k) {
        this.predict = predict;
        this.trainingData = trainingData;
        this.k = k;
    }



    public void EuclideanDistance() {
        for (Tuple tuple : trainingData) {
            if (predict.featureVector.length != tuple.featureVector.length) {
                LOG.error("2 Vectors must has same dimension.");
                return;
            }

            double result = Arrays.stream(VectorUtils.operate(predict.featureVector, tuple.featureVector, (x, y) -> Math.pow(x - y, 2)))
                    .parallel().sum();
            result = Math.sqrt(result);
            tuple.distance = result;
        }
    }

    public void ChebyshevDistance() {
        for (Tuple tuple : trainingData) {
            if (predict.featureVector.length != tuple.featureVector.length) {
                LOG.error("2 Vectors must has same dimension.");
                return;
            }

            double result = Arrays.stream(VectorUtils.operate(predict.featureVector, tuple.featureVector, (x, y) -> Math.abs(x - y)))
                    .max().getAsDouble();

            result = Math.sqrt(result);
            tuple.distance = result;
        }
    }

    public void ManhattanDistance() {
        for (Tuple tuple : trainingData) {
            if (predict.featureVector.length != tuple.featureVector.length) {
                LOG.error("2 Vectors must has same dimension.");
                return;
            }

            double result = Arrays.stream(VectorUtils.operate(predict.featureVector, tuple.featureVector, (x, y) -> Math.abs(x - y)))
                    .parallel().sum();
            result = Math.sqrt(result);
            tuple.distance = result;
        }
    }

    public String getResult() {
        Map<String, Integer> resultMap = new HashMap<String, Integer>();
        Collections.sort(trainingData, (tuple1, tuple2) -> {
            double diff = tuple1.distance - tuple2.distance;
            if (Math.abs(diff) < Double.MIN_VALUE) {
                return 0;
            } else {
                return Double.compare(tuple1.distance, tuple2.distance);
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
