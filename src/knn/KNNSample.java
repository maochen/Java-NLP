package knn;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

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

        String[] newVector = new String[] { "74", "55", "22" };

        KNN knn = new KNN(trainList);

        System.out.println("Euclidean Distance:");
        String result = knn.predict(newVector, k, 0);
        System.out.println("Prediction data: " + Arrays.toString(newVector) + " -> " + result);
        System.out.println(knn.getDetails());
        
        System.out.println("Chebyshev Distance:");
        result = knn.predict(newVector, k, 1);
        System.out.println("Prediction data: " + Arrays.toString(newVector) + " -> " + result);
        System.out.println(knn.getDetails());
        
        System.out.println("Manhattan Distance:");
        result = knn.predict(newVector, k, 2);
        System.out.println("Prediction data: " + Arrays.toString(newVector) + " -> " + result);
        System.out.println(knn.getDetails());
    }

}
