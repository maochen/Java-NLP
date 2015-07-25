package org.maochen.nlp.classifier.naivebayes;

import org.apache.commons.lang3.NotImplementedException;
import org.apache.commons.lang3.StringUtils;
import org.maochen.nlp.classifier.IClassifier;
import org.maochen.nlp.datastructure.Tuple;
import org.maochen.nlp.utils.VectorUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.util.*;
import java.util.Map.Entry;
import java.util.stream.Collectors;

/**
 * Created by Maochen on 12/3/14.
 */
public class NaiveBayesClassifier implements IClassifier {

    private static final Logger LOG = LoggerFactory.getLogger(NaiveBayesClassifier.class);

    private NaiveBayesModel model;

    public NaiveBayesClassifier(InputStream modelInputStream) {
        model = new NaiveBayesModel();
        model.load(modelInputStream);
    }

    public NaiveBayesClassifier() {

    }

    @Override
    public void setParameter(Map<String, String> paraMap) {
        throw new NotImplementedException("not implemented");
    }

    // Integer is label's Index from labelIndexer
    @Override
    public Map<String, Double> predict(Tuple predict) {
        Map<Integer, Double> labelProb = new HashMap<>();
        for (Integer labelIndex : model.labelIndexer.getIndexSet()) {
            double likelihood = 1.0D;

            for (int i = 0; i < predict.featureVector.length; i++) {
                double fi = predict.featureVector[i];
                likelihood = likelihood * VectorUtils.gaussianPDF(model.meanVectors[labelIndex][i], model.varianceVectors[labelIndex][i], fi);
            }

            double posterior = model.labelPrior.get(labelIndex) * likelihood; // prior*likelihood, This is numerator of posterior
            labelProb.put(labelIndex, posterior);
        }

        double evidence = labelProb.values().stream().reduce((e1, e2) -> e1 + e2).orElse(-1D);
        if (evidence == -1) {
            LOG.error("Evidence is Empty!");
            return new HashMap<>();
        }

        labelProb.entrySet().forEach(entry -> {
            double prob = entry.getValue() / evidence;
            if (prob > 0.999) {
                prob = 1D;
            } else if (prob < 0.001) {
                prob = 0D;
            }
            labelProb.put(entry.getKey(), prob);
        }); // This is denominator of posterior

        Map<String, Double> result = model.labelIndexer.convertMapKey(labelProb);

        if (predict.label == null || predict.label.isEmpty()) { // Just for write to predict tuple.
            predict.label = result.entrySet().stream().max((e1, e2) -> e1.getValue().compareTo(e2.getValue())).map(Entry::getKey).orElse(StringUtils.EMPTY);
        }
        return result;
    }

    public String predictLabel(Tuple predict) {
        Map<String, Double> prob = predict(predict);
        return prob.entrySet().stream().max((x1, x2) -> x1.getValue().compareTo(x2.getValue())).map(Entry::getKey).orElse(null);
    }

    @Override
    public IClassifier train(List<Tuple> trainingData) {
        model = new NBTrainingEngine(trainingData).train();
        return this;
    }

    public void persistModel(String filename) {
        if (model != null) {
            model.persist(filename);
        }
    }

    public void loadModel(String filename) {
        model = new NaiveBayesModel();
        try {
            model.load(new FileInputStream(filename));
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
    }

    public static List<Tuple> readTrainingData(String filename, String delimiter) {
        List<Tuple> data = new ArrayList<>();
        List<String> trainingDataString = new ArrayList<>();

        try (BufferedReader br = new BufferedReader(new FileReader(filename))) {
            String line = br.readLine();

            while (line != null) {
                trainingDataString.add(line.trim());
                line = br.readLine();
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        trainingDataString = trainingDataString.parallelStream().distinct().collect(Collectors.toList());

        for (String line : trainingDataString) {
            String[] tokens = line.trim().split(delimiter);
            String label = tokens[0];
            double[] values = new double[tokens.length - 1];
            for (int i = 1; i < tokens.length; i++) {
                String tokenString = tokens[i].contains(":") ? tokens[i].split(":")[1] : tokens[i];
                values[i - 1] = Double.parseDouble(tokenString);
            }
            data.add(new Tuple(0, values, label));
        }

        return data;
    }

    public static void writeToFile(List<Tuple> dataset, String filename) {
        try (BufferedWriter output = new BufferedWriter(new FileWriter(new File(filename)))) {
            for (Tuple t : dataset) {
                output.write(t.toString() + System.lineSeparator());
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

    }

    // Split the data between trainable and wrong.
    public static void splitData(final String originalTrainingDataFile) {
        List<Tuple> trainingData = NaiveBayesClassifier.readTrainingData(originalTrainingDataFile, "\\s");
        List<Tuple> wrongData = new ArrayList<>();

        int lastTrainingDataSize;
        int iterCount = 0;
        do {
            System.out.println("Iteration:\t" + (++iterCount));
            lastTrainingDataSize = trainingData.size();

            NaiveBayesClassifier nbc = new NaiveBayesClassifier();
            nbc.train(trainingData);

            Iterator<Tuple> trainingDataIter = trainingData.iterator();
            while (trainingDataIter.hasNext()) {
                Tuple t = trainingDataIter.next();
                String actual = nbc.predictLabel(t);
                if (!t.label.equals(actual) && !t.label.equals("1")) { // preserve 1 since too few.
                    wrongData.add(t);
                    trainingDataIter.remove();
                }
            }

            Iterator<Tuple> wrongDataIter = wrongData.iterator();
            while (wrongDataIter.hasNext()) {
                Tuple t = wrongDataIter.next();
                String actual = nbc.predictLabel(t);
                if (t.label.equals(actual)) {
                    trainingData.add(t);
                    wrongDataIter.remove();
                }
            }
        } while (trainingData.size() != lastTrainingDataSize);

        writeToFile(trainingData, originalTrainingDataFile + ".aligned");
        writeToFile(wrongData, originalTrainingDataFile + ".wrong");
    }

    public static void main(String[] args) {
        String folder = "/Users/Maochen/Desktop/w2v_weight_training/";
        String outputModelFolder = "/Users/Maochen/workspace/amelia/eliza-ir/src/main/resources/";
        //        splitData(folder + "training.all.txt");


        NaiveBayesClassifier nbc = new NaiveBayesClassifier();
        List<Tuple> trainingData = readTrainingData(folder + "/training.all.txt.aligned", "\\s");
        nbc.train(trainingData);
        nbc.persistModel(outputModelFolder + "/nb_model.dat");

        nbc.loadModel(outputModelFolder + "/nb_model.dat");
        Scanner scan = new Scanner(System.in);
        String input = StringUtils.EMPTY;

        String quitRegex = "q|quit|exit";
        while (!input.matches(quitRegex)) {
            System.out.println("Please enter feats:");
            input = scan.nextLine();
            if (!input.trim().isEmpty() && !input.matches(quitRegex)) {
                double[] feats = Arrays.stream(input.split("\\s")).mapToDouble(Double::parseDouble).toArray();
                Map<String, Double> results = nbc.predict(new Tuple(feats));
                System.out.println(results);
            }
        }


    }

    public static void main1(String[] args) {
        IClassifier nbc = new NaiveBayesClassifier();

        List<Tuple> trainingData;

        trainingData = new ArrayList<>();
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
