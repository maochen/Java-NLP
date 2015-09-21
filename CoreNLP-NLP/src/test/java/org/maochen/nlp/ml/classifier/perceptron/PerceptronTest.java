package org.maochen.nlp.ml.classifier.perceptron;

import org.junit.Before;
import org.junit.Test;
import org.maochen.nlp.ml.Tuple;
import org.maochen.nlp.ml.vector.DenseVector;

import java.io.FileInputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertEquals;

/**
 * Created by Maochen on 8/9/15.
 */
public class PerceptronTest {

    private PerceptronClassifier perceptronClassifier;

    @Before
    public void setUp() {
        perceptronClassifier = new PerceptronClassifier();
    }

    @Test
    public void test() throws IOException {
        List<Tuple> data = new ArrayList<>();
        data.add(new Tuple(1, new DenseVector(new double[]{1, 0, 0}), String.valueOf(1)));
        data.add(new Tuple(2, new DenseVector(new double[]{1, 0, 1}), String.valueOf(1)));
        data.add(new Tuple(3, new DenseVector(new double[]{1, 1, 0}), String.valueOf(1)));
        data.add(new Tuple(4, new DenseVector(new double[]{1, 1, 1}), String.valueOf(0)));
        perceptronClassifier.train(data);

        String modelPath = PerceptronClassifier.class.getResource("/").getPath() + "/temp_perceptron_model.dat";
        perceptronClassifier.persistModel(modelPath);
        perceptronClassifier = new PerceptronClassifier();
        perceptronClassifier.loadModel(new FileInputStream(modelPath));

        Tuple test = new Tuple(5, new DenseVector(new double[]{1, 1, 1}), null);
        Map<String, Double> actualMap = perceptronClassifier.predict(test);

        String actualLabel = actualMap.entrySet().stream()
                .max((e1, e2) -> e1.getValue().compareTo(e2.getValue()))
                .map(Map.Entry::getKey)
                .orElse(null);
        String expectedLabel = "0";
        assertEquals(expectedLabel, actualLabel);
    }

    @Test
    public void testModel() {
        PerceptronModel model = new PerceptronModel();
        model.threshold = 0.5;
        model.bias = new double[]{1.3};
        model.learningRate = 0.03;
        model.weights = new double[3][9];
        PerceptronModel cloneModel = new PerceptronModel(model);
        assertEquals(0.5, cloneModel.threshold, Double.MIN_NORMAL);
        assertEquals(0.03, cloneModel.learningRate, Double.MIN_NORMAL);
        assertEquals(1.3, cloneModel.bias[0], Double.MIN_NORMAL);
    }
}
