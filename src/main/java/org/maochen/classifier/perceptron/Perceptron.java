package org.maochen.classifier.perceptron;

import org.maochen.classifier.IClassifier;
import org.maochen.datastructure.Tuple;
import org.maochen.utils.VectorUtils;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Created by Maochen on 6/5/15.
 */
public class Perceptron implements IClassifier {

    private PerceptronModel model = null;

    @Override
    public IClassifier train(List<Tuple> trainingData) {
        PerceptronModel perceptronModel = PerceptronTrainingEngine.train(trainingData);
        this.model = perceptronModel;
        return this;
    }

    @Override
    public Map<String, Double> predict(Tuple predict) {
        if (this.model.weights == null) {
            throw new RuntimeException("Get the model first.");
        }

        double sum = VectorUtils.dotProduct(predict.featureVector, model.weights);
        int network = sum > model.threshold ? 1 : 0;

        Map<String, Double> results = new HashMap<>();
        results.put(String.valueOf(network), 1.0);
        results.put(String.valueOf(network == 1 ? 0 : 1), 0.0);

        return results;
    }

    @Override
    public void setParameter(Map<String, String> paraMap) {

    }

    public Perceptron() {
        this.model = new PerceptronModel();
    }


    public static void main(String[] args) {
        Perceptron perceptron = new Perceptron();

        List<Tuple> data = new ArrayList<>();
        data.add(new Tuple(1, new double[]{1, 0, 0}, String.valueOf(1)));
        data.add(new Tuple(2, new double[]{1, 0, 1}, String.valueOf(1)));
        data.add(new Tuple(3, new double[]{1, 1, 0}, String.valueOf(1)));
        data.add(new Tuple(4, new double[]{1, 1, 1}, String.valueOf(0)));
        perceptron.train(data);
        perceptron.model.persist("/Users/Maochen/Desktop/perceptron_model.dat");

        perceptron = new Perceptron();
        perceptron.model.load("/Users/Maochen/Desktop/perceptron_model.dat");
        Tuple test = new Tuple(5, new double[]{0, 0, 1}, null);
        System.out.println(perceptron.train(data).predict(test));
    }
}
