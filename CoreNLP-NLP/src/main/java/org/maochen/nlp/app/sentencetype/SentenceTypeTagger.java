package org.maochen.nlp.app.sentencetype;

import org.apache.commons.lang3.StringUtils;
import org.maochen.nlp.app.ITagger;
import org.maochen.nlp.ml.Tuple;
import org.maochen.nlp.ml.classifier.maxent.MaxEntClassifier;
import org.maochen.nlp.ml.vector.LabeledVector;
import org.maochen.nlp.parser.DTree;
import org.maochen.nlp.parser.IParser;
import org.maochen.nlp.parser.stanford.nn.StanfordNNDepParser;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileReader;
import java.io.IOException;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Scanner;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;

public class SentenceTypeTagger extends MaxEntClassifier implements ITagger {

    private static final Logger LOG = LoggerFactory.getLogger(SentenceTypeTagger.class);

    private SentenceTypeFeatureExtractor featureExtractor = new SentenceTypeFeatureExtractor();
    private IParser parser;

    @Override
    public void train(String trainFilePath) {
        Map<String, String> para = new HashMap<String, String>() {{
            put("iterations", "120");
        }};
        super.setParameter(para);

        parser.parse("."); // For loading POS Tagger.
        final Map<String, DTree> depTreeCache = new ConcurrentHashMap<>();

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

        LOG.info("Generating parse tree.");
        trainingData.parallelStream().map(x -> {
            String sentence = x.split("\\t")[1];
            depTreeCache.put(sentence, parser.parse(sentence));
            return null;
        }).collect(Collectors.toSet());

        LOG.info("Generating feats");
        List<Tuple> trainingTuples = trainingData.stream().map(line -> {
            String sentence = line.split("\\t")[1];
            String label = line.split("\\t")[0];
            DTree parseTree = depTreeCache.get(sentence);

            List<String> feats = featureExtractor.generateFeats(parseTree);

            String[] featsName = feats.stream().toArray(String[]::new);
            double[] feat = feats.stream().mapToDouble(x -> 1.0).toArray();
            LabeledVector labeledVector = new LabeledVector(feat);
            labeledVector.featsName = featsName;
            Tuple t = new Tuple(1, labeledVector, label);
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
        List<String> feats = featureExtractor.generateFeats(tree);
        String[] featsName = feats.stream().toArray(String[]::new);
        double[] feat = feats.stream().mapToDouble(x -> 1.0).toArray();

        LabeledVector vector = new LabeledVector(feat);
        vector.featsName = featsName;
        Tuple predict = new Tuple(vector);
        return super.predict(predict);
    }

    @Override
    public Map<String, Double> predict(String sentence) {
        return predict(parser.parse(sentence));
    }

    public SentenceTypeTagger(final IParser parser) {
        this.parser = parser;
    }

    public static void main(String[] args) throws IOException {
        String prefix = "/Users/mguan/Desktop";
        String trainFilePath = "/Users/mguan/workspace/nlp-service_training-data/sentence_type_corpus.txt";
        String modelPath = prefix + "/sent_type_model.dat";

        ITagger sentenceTypeTagger = new SentenceTypeTagger(new StanfordNNDepParser());

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