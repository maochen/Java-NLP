package org.maochen.nlp.ml.classifier.naivebayes;

import org.junit.Test;
import org.maochen.nlp.ml.IClassifier;
import org.maochen.nlp.ml.Tuple;
import org.maochen.nlp.ml.vector.DenseVector;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Map;

/**
 * Created by Maochen on 9/19/15.
 */
public class NaiveBayesTest {

    private IClassifier nbc = new NaiveBayesClassifier();

    @Test
    public void test() {
        List<Tuple> trainingData = new ArrayList<>();
        trainingData.add(new Tuple(1, new DenseVector(new double[]{6, 180, 12}), "male"));
        trainingData.add(new Tuple(2, new DenseVector(new double[]{5.92, 190, 11}), "male"));
        trainingData.add(new Tuple(3, new DenseVector(new double[]{5.58, 170, 12}), "male"));
        trainingData.add(new Tuple(4, new DenseVector(new double[]{5.92, 165, 10}), "male"));
        trainingData.add(new Tuple(5, new DenseVector(new double[]{5, 100, 6}), "female"));
        trainingData.add(new Tuple(6, new DenseVector(new double[]{5.5, 150, 8}), "female"));
        trainingData.add(new Tuple(7, new DenseVector(new double[]{5.42, 130, 7}), "female"));
        trainingData.add(new Tuple(8, new DenseVector(new double[]{5.75, 150, 9}), "female"));

        Tuple predict = new Tuple(new double[]{6, 130, 8});

        nbc.train(trainingData);
        Map<String, Double> probs = nbc.predict(predict);

        // male: 6.197071843878083E-9;
        // female: 5.377909183630024E-4;

        List<Map.Entry<String, Double>> result = new ArrayList<>(); // Just for ordered display.
        Comparator<Map.Entry<String, Double>> reverseCmp = Collections.reverseOrder(Comparator.comparing(Map.Entry::getValue));
        probs.entrySet().stream().sorted(reverseCmp).forEach(result::add);

        System.out.println("Result: " + predict);
        result.forEach(e -> System.out.println(e.getKey() + "\t:\t" + e.getValue()));
    }
}
