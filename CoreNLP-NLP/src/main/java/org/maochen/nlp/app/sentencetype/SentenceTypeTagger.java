package org.maochen.nlp.app.sentencetype;

import org.apache.commons.lang3.NotImplementedException;
import org.apache.commons.lang3.StringUtils;
import org.maochen.nlp.app.ITagger;
import org.maochen.nlp.ml.Tuple;
import org.maochen.nlp.ml.classifier.maxent.MaxEntClassifier;
import org.maochen.nlp.ml.vector.FeatNamedVector;
import org.maochen.nlp.parser.DTree;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileReader;
import java.io.IOException;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.Scanner;
import java.util.Set;
import java.util.stream.Collectors;

public class SentenceTypeTagger extends MaxEntClassifier implements ITagger {

    private static final Logger LOG = LoggerFactory.getLogger(SentenceTypeTagger.class);

    private SentenceTypeFeatureExtractor featureExtractor = new SentenceTypeFeatureExtractor();

    @Override
    public void train(String trainFilePath) {
        Properties props = new Properties();
        props.setProperty("iter", "120");
        super.setParameter(props);

        // Preparing training data
        Set<String> trainingData = new HashSet<>();
        try (BufferedReader br = new BufferedReader(new FileReader(trainFilePath))) {
            String line = br.readLine();
            while (line != null) {
                trainingData.add(line);
                line = br.readLine();
            }
        } catch (IOException e) {
            LOG.error("load data err.", e);
        }

        // -----------
        LOG.info("Loaded Training data.");

        LOG.info("Generating feats");
        List<Tuple> trainingTuples = trainingData.stream().map(line -> {
            String sentence = line.split("\\t")[1];
            String label = line.split("\\t")[0];
            sentence = sentence.replaceAll("(\\p{Punct}+$)", " $1");
            List<String> feats = featureExtractor.generateFeats(sentence.split("\\s"));

            String[] featsName = feats.stream().toArray(String[]::new);
            FeatNamedVector featNamedVector = new FeatNamedVector(featsName);
            Tuple t = new Tuple(1, featNamedVector, label);
            t.addExtra("sentence", sentence);
            return t;
        }).collect(Collectors.toList());
        LOG.info("Extracted Feats.");
        super.train(trainingTuples);
        double err = simpleValidator(trainingTuples);
        LOG.info("Err rate: " + err * 100 + "%");
    }

    private double simpleValidator(List<Tuple> trainingData) {
        int wrongCount = 0;
        for (Tuple tuple : trainingData) {
            String actualLabel = super.predict(tuple).entrySet().stream().max((t1, t2) -> t1.getValue().compareTo(t2.getValue())).map(Map.Entry::getKey).orElse(null);
            if (!tuple.label.equals(actualLabel)) {
                LOG.info("Wrong Predicted sample: Expected[" + tuple.label + "]\tActual[" + actualLabel + "] -> " + tuple.getExtra().get("sentence"));
                wrongCount++;
            }
        }

        return ((double) wrongCount) / trainingData.size();
    }

    @Override
    public Map<String, Double> predict(DTree tree) {
        throw new NotImplementedException("Sentence type classifier doesn't require parse tree.");
    }

    @Override
    public Map<String, Double> predict(String sentence) {
        sentence = sentence.replaceAll("(\\p{Punct}+$)", " $1");
        List<String> feats = featureExtractor.generateFeats(sentence.split("\\s"));
        String[] featsName = feats.stream().toArray(String[]::new);
        double[] feat = feats.stream().mapToDouble(x -> 1.0).toArray();

        FeatNamedVector vector = new FeatNamedVector(feat);
        vector.featsName = featsName;
        Tuple predict = new Tuple(vector);
        return super.predict(predict);
    }

    public SentenceTypeTagger() {

    }

    public static void main(String[] args) throws IOException {
        String prefix = "/Users/mguan/Desktop";
        String trainFilePath = "/Users/mguan/workspace/ameliang/ameliang/amelia-nlp/src/main/resources/classifierData/utteranceClassifierData/training/sentencetype/sentencetype.train";
        String modelPath = prefix + "/sent_type_model.dat";

        ITagger sentenceTypeTagger = new SentenceTypeTagger();

        sentenceTypeTagger.train(trainFilePath);
        sentenceTypeTagger.persistModel(modelPath);

        sentenceTypeTagger.loadModel(new FileInputStream(modelPath));

        Scanner scanner = new Scanner(System.in);
        System.out.println("Input Sentence:");
        while (true) {
            String sentence = scanner.nextLine();
            if (sentence.equalsIgnoreCase("exit")) {
                break;
            }

            Map<String, Double> result = sentenceTypeTagger.predict(sentence);
            System.out.println(result);
            String type = result.entrySet().stream().max((e1, e2) -> e1.getValue().compareTo(e2.getValue())).map(Map.Entry::getKey).orElse(null);
            System.out.println(StringUtils.capitalize(type));
        }
        scanner.close();
        System.exit(0);
    }

}