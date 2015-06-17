package org.maochen.classifier.perceptron;

import org.maochen.classifier.IClassifier;
import org.maochen.datastructure.Tuple;
import org.maochen.utils.VectorUtils;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Created by Maochen on 6/5/15.
 */
public class PerceptronClassifier implements IClassifier {

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

    public PerceptronClassifier() {
        this.model = new PerceptronModel();
    }

    public PerceptronClassifier(InputStream modelIs) {
        this();
        this.model.load(modelIs);
    }


    public static void main(String[] args) throws FileNotFoundException {
        PerceptronClassifier perceptronClassifier = new PerceptronClassifier();

        List<Tuple> data = new ArrayList<>();
        data.add(new Tuple(1, new double[]{1, 0, 0}, String.valueOf(1)));
        data.add(new Tuple(2, new double[]{1, 0, 1}, String.valueOf(1)));
        data.add(new Tuple(3, new double[]{1, 1, 0}, String.valueOf(1)));
        data.add(new Tuple(4, new double[]{1, 1, 1}, String.valueOf(0)));
        perceptronClassifier.train(data);
        perceptronClassifier.model.persist("/Users/Maochen/Desktop/perceptron_model.dat");

        perceptronClassifier = new PerceptronClassifier();
        perceptronClassifier.model.load(new FileInputStream("/Users/Maochen/Desktop/perceptron_model.dat"));
        Tuple test = new Tuple(5, new double[]{0, 0, 1}, null);
        System.out.println(perceptronClassifier.train(data).predict(test));
    }
}
