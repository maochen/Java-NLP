package org.maochen.classifier.knn;

import org.maochen.classifier.IClassifier;
import org.maochen.classifier.KNNClassifier;

import java.util.*;

public class KNNSample {

    public static void main(String[] args) {
        int k = 3;

        String[] vectorA = new String[] { "9", "32", "65.1", "A" };
        String[] vectorB = new String[] { "12", "65", "86.1", "C" };
        String[] vectorC = new String[] { "19", "54", "45.1", "C" };

        List<String[]> trainList = new ArrayList<String[]>();
        trainList.add(vectorA);
        trainList.add(vectorB);
        trainList.add(vectorC);

        String[] predict = new String[] { "74", "55", "22" };

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
