package knn;

import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class KNNEngine {

    protected class DistanceComparator implements Comparator<KNNDTO> {
        public int compare(KNNDTO turple1, KNNDTO turple2) {
            double diff = turple1.getDistance() - turple2.getDistance();

            if (Math.abs(diff) < Double.MIN_VALUE)
                return 0;
            else if (diff > 0)
                return 1;
            else
                return -1;
        }

    }

    private KNNDTO predict;
    private List<KNNDTO> trainRecordList;
    private int k;

    public void initialize(KNNDTO predict, List<KNNDTO> trainRecordList, int k) {
        if (k % 2 == 0) {
            throw new RuntimeException("K should be odd for voting!");
        }

        this.predict = predict;
        this.trainRecordList = trainRecordList;
        this.k = k;
    }

    public void EuclideanDistance() {
        for (KNNDTO record : trainRecordList) {
            if (predict.getVector().length != record.getVector().length) {
                throw new RuntimeException("2 Vectors must has same dimension.");
            }

            double result = 0.0;
            for (int i = 0; i < predict.getVector().length; i++) {
                result += Math.pow(predict.getVector()[i] - record.getVector()[i], 2);
            }

            result = Math.sqrt(result);
            record.setDistance(result);

        }

    }

    public void ChebyshevDistance() {
        for (KNNDTO record : trainRecordList) {
            if (predict.getVector().length != record.getVector().length) {
                throw new RuntimeException("2 Vectors must has same dimension.");
            }

            double result = 0.0;
            for (int i = 0; i < predict.getVector().length; i++) {
                double diff = Math.abs(predict.getVector()[i] - record.getVector()[i]);
                if (diff > result) result = diff;
            }

            result = Math.sqrt(result);
            record.setDistance(result);
        }
    }

    public void ManhattanDistance() {
        for (KNNDTO record : trainRecordList) {
            if (predict.getVector().length != record.getVector().length) {
                throw new RuntimeException("2 Vectors must has same dimension.");
            }

            double result = 0.0;
            for (int i = 0; i < predict.getVector().length; i++) {
                double diff = Math.abs(predict.getVector()[i] - record.getVector()[i]);
                result += diff;
            }

            result = Math.sqrt(result);
            record.setDistance(result);
        }
    }

    public String getResult() {
        Map<String, Integer> resultMap = new HashMap<String, Integer>();
        Collections.sort(trainRecordList, new DistanceComparator());

        for (int i = 0; i < k; i++) {
            KNNDTO entry = trainRecordList.get(i);
            String key = entry.getKey();
            if (!resultMap.containsKey(key)) {
                resultMap.put(key, 1);
            }
            else {
                int count = resultMap.get(key) + 1;
                resultMap.put(key, count);
            }
        }

        String maxVote = "";
        int maxCount = 0;
        for (String key : resultMap.keySet()) {
            int currentCount = resultMap.get(key);
            if (currentCount > maxCount) {
                maxCount = currentCount;
                maxVote = key;
            }
        }

        return maxVote;

    }
}
