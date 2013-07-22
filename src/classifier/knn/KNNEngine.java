package classifier.knn;

import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Should not exposed.
 * 
 * @author MaochenG
 */
public final class KNNEngine {

    protected class DistanceComparator implements Comparator<KNNDTO<String>> {
        public int compare(KNNDTO<String> turple1, KNNDTO<String> turple2) {
            double diff = turple1.getDistance() - turple2.getDistance();

            if (Math.abs(diff) < Double.MIN_VALUE)
                return 0;
            else if (diff > 0)
                return 1;
            else return -1;
        }

    }

    private KNNDTO<String> predict;
    private List<KNNDTO<String>> trainRecordList;
    private int k;

    public void initialize(KNNDTO<String> predict, List<KNNDTO<String>> trainRecordList, int k) {
        this.predict = predict;
        this.trainRecordList = trainRecordList;
        this.k = k;
    }

    public void EuclideanDistance() {
        for (KNNDTO<String> record : trainRecordList) {
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
        for (KNNDTO<String> record : trainRecordList) {
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
        for (KNNDTO<String> record : trainRecordList) {
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
            KNNDTO<String> entry = trainRecordList.get(i);
            String label = entry.getLabel();
            if (!resultMap.containsKey(label)) {
                resultMap.put(label, 1);
            }
            else {
                int count = resultMap.get(label) + 1;
                resultMap.put(label, count);
            }
        }

        String maxVote = "";
        int maxCount = 0;
        int maxCountEntryNumber = 0; // equal vote

        for (String label : resultMap.keySet()) {
            int currentCount = resultMap.get(label);

            if (currentCount == maxCount) {
                maxCountEntryNumber++;
            }
            else if (currentCount > maxCount) {
                maxCount = currentCount;
                maxVote = label;
                maxCountEntryNumber = 1;
            }

        }

        if (maxCountEntryNumber != 1) System.out.println("Equal Max Vote, just grab the first max!");
        predict.setLabel(maxVote);
        return maxVote;

    }
}
