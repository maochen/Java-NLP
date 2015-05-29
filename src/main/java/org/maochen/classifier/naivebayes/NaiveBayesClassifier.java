package org.maochen.classifier.naivebayes;

import org.apache.commons.lang3.NotImplementedException;
import org.apache.commons.lang3.StringUtils;
import org.maochen.classifier.IClassifier;
import org.maochen.datastructure.Tuple;
import org.maochen.utils.VectorUtils;

import java.util.*;
import java.util.Map.Entry;

/**
 * Created by Maochen on 12/3/14.
 */
public class NaiveBayesClassifier implements IClassifier {

    private NaiveBayesModel model;

    @Override
    public void setParameter(Map<String, String> paraMap) {
        throw new NotImplementedException("not implemented");
    }

    // Integer is label's Index from labelIndexer
    @Override
    public Map<String, Double> predict(Tuple predict) {
        Map<Integer, Double> labelProbs = new HashMap<>();

        // P(male) -- P(C)
        model.labelIndexer.getIndexSet().stream().forEach(index -> labelProbs.put(index, 0.5));

        for (Integer labelIndex : model.labelIndexer.getIndexSet()) {
            double posteriorLabel = labelProbs.get(labelIndex);

            for (int i = 0; i < predict.featureVector.length; i++) {
                double fi = predict.featureVector[i];
                posteriorLabel = posteriorLabel * VectorUtils.gaussianPDF(model.meanVectors[labelIndex][i], model.varianceVectors[labelIndex][i], fi);
            }

            labelProbs.put(labelIndex, posteriorLabel);
        }

        Map<String, Double> result = model.labelIndexer.convertMapKey(labelProbs);
        predict.label = result.entrySet().stream().max((e1, e2) -> e1.getValue().compareTo(e2.getValue())).map(Entry::getKey).orElse(StringUtils.EMPTY);
        return result;
    }

    @Override
    public IClassifier train(List<Tuple> trainingData) {
        model = new TrainingEngine(trainingData).train();
        return this;
    }

    public static void main(String[] args) {
        IClassifier nbc = new NaiveBayesClassifier();

        List<Tuple> trainingData = new ArrayList<>();
        trainingData.add(new Tuple(1, new double[]{6, 180, 12}, "male"));
        trainingData.add(new Tuple(2, new double[]{5.92, 190, 11}, "male"));
        trainingData.add(new Tuple(3, new double[]{5.58, 170, 12}, "male"));
        trainingData.add(new Tuple(4, new double[]{5.92, 165, 10}, "male"));
        trainingData.add(new Tuple(5, new double[]{5, 100, 6}, "female"));
        trainingData.add(new Tuple(6, new double[]{5.5, 150, 8}, "female"));
        trainingData.add(new Tuple(7, new double[]{5.42, 130, 7}, "female"));
        trainingData.add(new Tuple(8, new double[]{5.75, 150, 9}, "female"));

        Tuple predict = new Tuple(new double[]{6, 130, 8});

        nbc.train(trainingData);
        Map<String, Double> probs = nbc.predict(predict);


        // male: 6.197071843878083E-9;
        // female: 5.377909183630024E-4;

        List<Entry<String, Double>> result = new ArrayList<>(); // Just for ordered display.
        Comparator<Entry<String, Double>> reverseCmp = Collections.reverseOrder(Comparator.comparing(Entry::getValue));
        probs.entrySet().stream().sorted(reverseCmp).forEach(result::add);

        System.out.println("Result: " + predict);
        result.forEach(e -> System.out.println(e.getKey() + "\t:\t" + e.getValue()));
    }


}
