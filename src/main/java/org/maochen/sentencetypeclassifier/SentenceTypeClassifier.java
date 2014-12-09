package org.maochen.sentencetypeclassifier;

import opennlp.maxent.*;
import opennlp.maxent.io.GISModelWriter;
import opennlp.maxent.io.SuffixSensitiveGISModelReader;
import opennlp.maxent.io.SuffixSensitiveGISModelWriter;
import opennlp.model.EventStream;
import opennlp.model.RealValueFileEventStream;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.util.HashSet;
import java.util.Scanner;
import java.util.Set;

public class SentenceTypeClassifier {

    private static final Logger LOG = LoggerFactory.getLogger(SentenceTypeClassifier.class);

    public static final String DELIMITER = " ";
    private static final boolean USE_SMOOTHING = true;
    private static final int ITERATIONS = 100;
    private static final int CUTOFF = 0;
    private String filepathPrefix;
    private GISModel model;
    private boolean isRealFeature = false;

    public void train(String trainFilePath, TrainingFeatureExtractor trainingFeatureExtractor) throws IOException {
        // Preparing training data
        Set<String> trainingData = new HashSet<>();
        try (BufferedReader br = new BufferedReader(new FileReader(trainFilePath))) {
            String line = br.readLine();
            while (line != null) {
                trainingData.add(line);
                line = br.readLine();
            }
        }
        // -----------

        this.isRealFeature = trainingFeatureExtractor.isRealFeature;
        trainingFeatureExtractor.extractFeature(trainingData);

        EventStream es;
        if (isRealFeature) {
            es = new RealBasicEventStream(new PlainTextByLineDataStream(new FileReader(filepathPrefix + "/featureVector.txt")));
        } else {
            es = new BasicEventStream(new PlainTextByLineDataStream(new FileReader(filepathPrefix + "/featureVector.txt")));
        }

        model = GIS.trainModel(es, ITERATIONS, CUTOFF, USE_SMOOTHING, true);
    }

    public void persist(String modelPath) throws IOException {
        File outputFile = new File(modelPath);
        GISModelWriter writer = new SuffixSensitiveGISModelWriter(model, outputFile);
        writer.persist();
    }

    public void loadModel(String modelPath) throws IOException {
        long start = System.currentTimeMillis();
        LOG.info("Loading MaxEnt model ... ");
        model = (GISModel) new SuffixSensitiveGISModelReader(new File(modelPath)).getModel();
        long end = System.currentTimeMillis();
        long duration = (end - start) / 1000;
        LOG.info("completed ... " + duration + " secs.");

    }

    private double[] predictOCS(String sentence, FeatureExtractor featureExtractor) {
        sentence = sentence.replaceAll("\\s+", "_");
        String vector = featureExtractor.getFeats(sentence + DELIMITER + "?");
        String[] contexts = vector.split(DELIMITER);

        double[] ocs;

        if (isRealFeature) {
            float[] values = RealValueFileEventStream.parseContexts(contexts);
            ocs = model.eval(contexts, values);
        } else {
            ocs = model.eval(contexts);
        }
        return ocs;
    }

    public String predict(String sentence, FeatureExtractor featureExtractor) {
        return model.getBestOutcome(predictOCS(sentence, featureExtractor));
    }

    public String predictAllOutcome(String sentence, FeatureExtractor featureExtractor) {
        return model.getAllOutcomes(predictOCS(sentence, featureExtractor));
    }

    public SentenceTypeClassifier(String filepathPrefix) {
        this.filepathPrefix = filepathPrefix;
    }

    public static void annotate(String testFilePath, SentenceTypeClassifier sentenceTypeClassifier, FeatureExtractor featureExtractor) {

        try (BufferedReader br = new BufferedReader(new FileReader(testFilePath))) {
            BufferedWriter bw = new BufferedWriter(new FileWriter(testFilePath + ".result"));

            String line = br.readLine();
            while (line != null) {
                String result = sentenceTypeClassifier.predict(line, featureExtractor);
                bw.write(line.replaceAll("\\s", "_") + " " + result + System.getProperty("line.separator"));

                line = br.readLine();
            }

            bw.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static void main(String[] args) throws IOException {
        String prefix = "/Users/Maochen/Desktop/temp";
        String trainFilePath = SentenceTypeClassifier.class.getResource("/annotatedTrainingData.txt").getPath();
        String modelPath = prefix + "/model.dat";

        SentenceTypeClassifier sentenceTypeClassifier = new SentenceTypeClassifier(prefix);
        TrainingFeatureExtractor trainingFeatureExtractor = new TrainingFeatureExtractor(prefix, SentenceTypeClassifier.DELIMITER);
        sentenceTypeClassifier.train(trainFilePath, trainingFeatureExtractor);
        sentenceTypeClassifier.persist(modelPath);


        FeatureExtractor featureExtractor = trainingFeatureExtractor;
        sentenceTypeClassifier.loadModel(modelPath);

        Scanner scanner = new Scanner(System.in);
        System.out.println("Input Sentence:");
        while (true) {
            String sentence = scanner.nextLine();
            if (sentence.equalsIgnoreCase("exit")) {
                break;
            }
            sentence = sentence.replaceAll("\\s", "_");
            String vector = featureExtractor.getFeats(sentence + DELIMITER + "?");
            System.out.println(vector);
            String result = sentenceTypeClassifier.predictAllOutcome(sentence, featureExtractor);
            System.out.println(result + " ||| " + sentenceTypeClassifier.predict(sentence, featureExtractor));
        }
        scanner.close();
        System.exit(0);
    }
}
