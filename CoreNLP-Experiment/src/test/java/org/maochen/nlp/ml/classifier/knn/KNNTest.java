package org.maochen.nlp.ml.classifier.knn;

import org.junit.Test;
import org.maochen.nlp.ml.IClassifier;
import org.maochen.nlp.ml.Tuple;
import org.maochen.nlp.ml.vector.DenseVector;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertEquals;

/**
 * Created by Maochen on 9/19/15.
 */
public class KNNTest {


    /**
     * Euclidean Distance: Prediction data: id:0 [74.0, 55.0, 22.0] -> C
     *
     * {1=81.31180726069246, 2=89.73745037608323, 3=59.66246726376642}
     *
     * Chebyshev Distance: Prediction data: id:0 [74.0, 55.0, 22.0] -> C
     *
     * {1=8.06225774829855, 2=8.006247560499238, 3=7.416198487095663}
     *
     * Manhattan Distance: Prediction data: id:0 [74.0, 55.0, 22.0] -> C
     *
     * {1=11.449890829173874, 2=11.666190466471907, 3=8.893818077743664}
     */
    @Test
    public void test() {
        int k = 3;

        Tuple vectorA = new Tuple(1, new DenseVector(new double[]{9, 32, 65.1}), "A");
        Tuple vectorB = new Tuple(2, new DenseVector(new double[]{12, 65, 86.1}), "C");
        Tuple vectorC = new Tuple(3, new DenseVector(new double[]{19, 54, 45.1}), "C");

        List<Tuple> trainList = new ArrayList<>();
        trainList.add(vectorA);
        trainList.add(vectorB);
        trainList.add(vectorC);

        Tuple predict = new Tuple(new double[]{74, 55, 22});

        IClassifier knn = new KNNClassifier();
        knn.train(trainList);
        Map<String, String> paraMap = new HashMap<>();
        paraMap.put("k", String.valueOf(k));

//        System.out.println("Euclidean Distance:");
        paraMap.put("mode", "0");
        knn.setParameter(paraMap);
        Map<String, Double> details = knn.predict(predict);
        assertEquals(details.get("1"), 81.31180726069246, Double.MIN_NORMAL);
        assertEquals(details.get("2"), 89.73745037608323, Double.MIN_NORMAL);
        assertEquals(details.get("3"), 59.66246726376642, Double.MIN_NORMAL);

//        System.out.println("Chebyshev Distance:");
        paraMap.put("mode", "1");
        knn.setParameter(paraMap);
        details = knn.predict(predict);
        assertEquals(details.get("1"), 8.06225774829855, Double.MIN_NORMAL);
        assertEquals(details.get("2"), 8.006247560499238, Double.MIN_NORMAL);
        assertEquals(details.get("3"), 7.416198487095663, Double.MIN_NORMAL);

//        System.out.println("Manhattan Distance:");
        paraMap.put("mode", "2");
        knn.setParameter(paraMap);
        details = knn.predict(predict);
        assertEquals(details.get("1"), 11.449890829173874, Double.MIN_NORMAL);
        assertEquals(details.get("2"), 11.666190466471907, Double.MIN_NORMAL);
        assertEquals(details.get("3"), 8.893818077743664, Double.MIN_NORMAL);
    }
}
