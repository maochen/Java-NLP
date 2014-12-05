package org.maochen.classifier.knn;

import org.maochen.datastructure.Element;
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

    protected class DistanceComparator implements Comparator<Element> {
        public int compare(Element e1, Element e2) {
            double diff = e1.distance - e2.distance;

            if (Math.abs(diff) < Double.MIN_VALUE)
                return 0;
            else if (diff > 0)
                return 1;
            else return -1;
        }

    }

    private Element predict;
    private List<Element> trainingData;
    private int k;

    public void initialize(Element predict, List<Element> trainingData, int k) {
        this.predict = predict;
        this.trainingData = trainingData;
        this.k = k;
    }

    public void EuclideanDistance() {
        for (Element t : trainingData) {
            if (predict.featureVector.length != t.featureVector.length) {
                LOG.error("2 Vectors must has same dimension.");
                return;
            }

            double result = 0.0;
            for (int i = 0; i < predict.featureVector.length; i++) {
                result += Math.pow(predict.featureVector[i] - t.featureVector[i], 2);
            }

            result = Math.sqrt(result);
            t.distance = result;
        }
    }

    public void ChebyshevDistance() {
        for (Element element : trainingData) {
            if (predict.featureVector.length != element.featureVector.length) {
                LOG.error("2 Vectors must has same dimension.");
                return;
            }

            double result = 0.0;
            for (int i = 0; i < predict.featureVector.length; i++) {
                double diff = Math.abs(predict.featureVector[i] - element.featureVector[i]);
                if (diff > result) result = diff;
            }

            result = Math.sqrt(result);
            element.distance = result;
        }
    }

    public void ManhattanDistance() {
        for (Element element : trainingData) {
            if (predict.featureVector.length != element.featureVector.length) {
                LOG.error("2 Vectors must has same dimension.");
                return;
            }

            double result = 0.0;
            for (int i = 0; i < predict.featureVector.length; i++) {
                double diff = Math.abs(predict.featureVector[i] - element.featureVector[i]);
                result += diff;
            }

            result = Math.sqrt(result);
            element.distance = result;
        }
    }

    public String getResult() {
        Map<String, Integer> resultMap = new HashMap<String, Integer>();
        Collections.sort(trainingData, new DistanceComparator());

        for (int i = 0; i < k; i++) {
            Element element = trainingData.get(i);
            int count = resultMap.containsKey(element.label) ? resultMap.get(element.label) : 0;
            count++;
            resultMap.put(element.label, count);

        }

        String maxVote = "";
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
