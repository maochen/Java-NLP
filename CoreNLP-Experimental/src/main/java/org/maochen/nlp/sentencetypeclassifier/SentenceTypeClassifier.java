//package org.maochen.nlp.sentencetypeclassifier;
//
//import org.apache.commons.lang3.StringUtils;
//import org.maochen.nlp.classifier.maxent.MaxEntClassifier;
//import org.maochen.nlp.datastructure.DTree;
//import org.maochen.nlp.datastructure.Tuple;
//import org.maochen.nlp.parser.IParser;
//import org.maochen.nlp.parser.stanford.pcfg.StanfordPCFGParser;
//import org.slf4j.Logger;
//import org.slf4j.LoggerFactory;
//
//import java.io.BufferedReader;
//import java.io.BufferedWriter;
//import java.io.FileReader;
//import java.io.FileWriter;
//import java.io.IOException;
//import java.util.HashMap;
//import java.util.HashSet;
//import java.util.List;
//import java.util.Map;
//import java.util.Scanner;
//import java.util.Set;
//import java.util.UUID;
//import java.util.concurrent.ConcurrentHashMap;
//import java.util.stream.Collectors;
//
//public class SentenceTypeClassifier {
//
//    private static final Logger LOG = LoggerFactory.getLogger(SentenceTypeClassifier.class);
//
//    public static final String DELIMITER = StringUtils.SPACE;
//    private String filepathPrefix;
//    private MaxEntClassifier maxEntClassifier = new MaxEntClassifier(true);
//    private IParser parser = new StanfordPCFGParser();
//
//    public void train(String trainFilePath, TrainingFeatureExtractor trainingFeatureExtractor) throws IOException {
//        final Map<String, DTree> depTreeCache = new ConcurrentHashMap<>();
//
//        // Preparing training data
//        Set<String> trainingData = new HashSet<>();
//        try (BufferedReader br = new BufferedReader(new FileReader(trainFilePath))) {
//            String line = br.readLine();
//            while (line != null) {
//                trainingData.add(line);
//                line = br.readLine();
//            }
//        }
//        // -----------
//
//        List<Tuple> trainingTuples = trainingData.stream().map(line -> {
//            String sentence = line.split("\\s")[0];
//            String label = line.split("\\s")[1];
//            DTree parseTree;
//            if (depTreeCache.containsKey(sentence)) {
//                parseTree = depTreeCache.get(sentence);
//            } else {
//                parseTree = parser.parse(sentence);
//                depTreeCache.put(sentence, parseTree);
//            }
//
//            List<String> feats = trainingFeatureExtractor.extractFeature(sentence, parseTree);
//            String[] featsName = ;
//            double[] feat = ;
//            return new Tuple(UUID.randomUUID().clockSequence(), featsName, feat, label);
//        }).collect(Collectors.toList());
//
//        maxEntClassifier.train(trainingTuples);
//
//    }
//
//    public void persist(String modelPath) throws IOException {
//        maxEntClassifier.persist(modelPath);
//    }
//
//    public void loadModel(String modelPath) throws IOException {
//        maxEntClassifier.loadModel(modelPath);
//    }
//
//    public Map<String, Double> predict(String sentence, FeatureExtractor featureExtractor) {
//        String feats = featureExtractor.generateFeats(sentence, parser.parse(sentence));
//        Tuple predict=new Tuple();
//        return maxEntClassifier.predict(predict);
//    }
//
//    public SentenceTypeClassifier(String filepathPrefix) {
//        this.filepathPrefix = filepathPrefix;
//    }
//
//    public static void main(String[] args) throws IOException {
//        String prefix = "/Users/Maochen/Desktop/temp";
//        String trainFilePath = SentenceTypeClassifier.class.getResource("/sentence_type_corpus.txt").getPath();
//        String modelPath = prefix + "/model.dat";
//
//        SentenceTypeClassifier sentenceTypeClassifier = new SentenceTypeClassifier(prefix);
//
//        TrainingFeatureExtractor trainingFeatureExtractor = new TrainingFeatureExtractor(prefix, SentenceTypeClassifier.DELIMITER);
//        sentenceTypeClassifier.train(trainFilePath, trainingFeatureExtractor);
//        sentenceTypeClassifier.persist(modelPath);
//
//
//        FeatureExtractor featureExtractor = trainingFeatureExtractor;
//        sentenceTypeClassifier.loadModel(modelPath);
//
//        Scanner scanner = new Scanner(System.in);
//        System.out.println("Input Sentence:");
//        while (true) {
//            String sentence = scanner.nextLine();
//            if (sentence.equalsIgnoreCase("exit")) {
//                break;
//            }
//
//            Map<String, Double> result = sentenceTypeClassifier.predict(sentence, featureExtractor);
//            System.out.println(result);
//        }
//        scanner.close();
//        System.exit(0);
//    }
//}
