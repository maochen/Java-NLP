package org.maochen.nlp.sentencetypeclassifier;

import org.apache.commons.lang3.StringUtils;
import org.maochen.nlp.datastructure.DTree;
import org.maochen.nlp.ml.classifier.maxent.MaxEntClassifier;
import org.maochen.nlp.ml.datastructure.Tuple;
import org.maochen.nlp.parser.IParser;
import org.maochen.nlp.parser.stanford.pcfg.StanfordPCFGParser;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Scanner;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;

public class SentenceTypeClassifier {

    private static final Logger LOG = LoggerFactory.getLogger(SentenceTypeClassifier.class);

    private MaxEntClassifier maxEntClassifier = new MaxEntClassifier();
    private IParser parser = new StanfordPCFGParser();

    public void train(String trainFilePath, FeatureExtractor featureExtractor) throws IOException {
        final Map<String, DTree> depTreeCache = new ConcurrentHashMap<>();

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
        LOG.info("Loaded Training data.");

        LOG.info("Generating parse tree.");
        trainingData.parallelStream().map(x -> {
            String sentence = x.split("\\s")[0].replaceAll("_", StringUtils.SPACE);
            depTreeCache.put(sentence, parser.parse(sentence));
            return null;
        }).collect(Collectors.toSet());

        LOG.info("Generating feats");
        List<Tuple> trainingTuples = trainingData.stream().map(line -> {
            String sentence = line.split("\\s")[0].replaceAll("_", StringUtils.SPACE);
            String label = line.split("\\s")[1];
            DTree parseTree = depTreeCache.get(sentence);

            List<String> feats = featureExtractor.generateFeats(sentence, parseTree);
            String[] featsName = feats.stream().toArray(String[]::new);
            double[] feat = feats.stream().mapToDouble(x -> 1.0).toArray();
            return new Tuple(1, featsName, feat, label);
        }).collect(Collectors.toList());
        LOG.info("Extracted Feats.");
        maxEntClassifier.train(trainingTuples);

    }

    public void persist(String modelPath) throws IOException {
        maxEntClassifier.persistModel(modelPath);
    }

    public void loadModel(String modelPath) throws IOException {
        maxEntClassifier.loadModel(modelPath);
    }

    public Map<String, Double> predict(String sentence, FeatureExtractor featureExtractor) {
        List<String> feats = featureExtractor.generateFeats(sentence, parser.parse(sentence));
        String[] featsName = feats.stream().toArray(String[]::new);
        double[] feat = feats.stream().mapToDouble(x -> 1.0).toArray();
        Tuple predict = new Tuple(-1, featsName, feat, null);
        return maxEntClassifier.predict(predict);
    }

    public static void main(String[] args) throws IOException {
        String prefix = "/Users/Maochen/Desktop/";
        String trainFilePath = "/Users/Maochen/workspace/nlp-service_training-data/sentence_classifier_annotated_training_data.txt";
        String modelPath = prefix + "/sent_type_model.dat";

        SentenceTypeClassifier sentenceTypeClassifier = new SentenceTypeClassifier();
        FeatureExtractor featureExtractor = new FeatureExtractor();

        sentenceTypeClassifier.train(trainFilePath, featureExtractor);
        sentenceTypeClassifier.persist(modelPath);

        sentenceTypeClassifier.loadModel(modelPath);

        Scanner scanner = new Scanner(System.in);
        System.out.println("Input Sentence:");
        while (true) {
            String sentence = scanner.nextLine();
            if (sentence.equalsIgnoreCase("exit")) {
                break;
            }

            Map<String, Double> result = sentenceTypeClassifier.predict(sentence, featureExtractor);
            System.out.println(result);
        }
        scanner.close();
        System.exit(0);
    }
}
