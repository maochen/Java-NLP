package org.maochen.classifier.perceptron;

import org.maochen.classifier.IClassifier;
import org.maochen.datastructure.Tuple;
import org.maochen.utils.VectorUtils;

import java.io.FileNotFoundException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Function;

/**
 * Created by Maochen on 6/5/15.
 */
public class PerceptronClassifier implements IClassifier {

    protected PerceptronModel model = null;

    protected boolean trainBias = true;

    @Override
    public IClassifier train(List<Tuple> trainingData) {
        PerceptronModel perceptronModel = PerceptronTrainingEngine.train(trainingData, this);
        this.model = perceptronModel;
        return this;
    }

    @Override
    public Map<String, Double> predict(Tuple predict) {
        // stochastic binary assue prob add up to 1.
        return predict(predict, VectorUtils.stochasticBinary);
    }

    public Map<String, Double> predict(Tuple predict, Function<Double, Double> outputLayerFunction) {
        if (this.model.weights == null) {
            throw new RuntimeException("Get the model first.");
        }

        double sum = VectorUtils.dotProduct(predict.featureVector, model.weights);
        sum += model.bias;
        sum = outputLayerFunction.apply(sum);

        Map<String, Double> results = new HashMap<>();

        int network = sum > model.threshold ? 1 : 0;
        results.put(String.valueOf(network), sum);
        results.put(String.valueOf(1 - network), 1 - sum);

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
        String modelPath = "/Users/Maochen/Desktop/perceptron_model.dat";
        PerceptronClassifier perceptronClassifier = new PerceptronClassifier();
        // For reproduce wiki's example for the following, plz disable trainBias.
        perceptronClassifier.trainBias = true;

        List<Tuple> data = new ArrayList<>();
        data.add(new Tuple(1, new double[]{1, 0, 0}, String.valueOf(1)));
        data.add(new Tuple(2, new double[]{1, 0, 1}, String.valueOf(1)));
        data.add(new Tuple(3, new double[]{1, 1, 0}, String.valueOf(1)));
        data.add(new Tuple(4, new double[]{1, 1, 1}, String.valueOf(0)));
        perceptronClassifier.train(data);

        //        perceptronClassifier.model.persist(modelPath);
        //        perceptronClassifier = new PerceptronClassifier();
        //        perceptronClassifier.model.load(new FileInputStream(modelPath));

        Tuple test = new Tuple(5, new double[]{1, 1, 1}, null);
        System.out.println(perceptronClassifier.predict(test));
    }
}
