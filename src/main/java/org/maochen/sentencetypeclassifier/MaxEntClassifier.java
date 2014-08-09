package org.maochen.sentencetypeclassifier;

import opennlp.maxent.*;
import opennlp.maxent.io.GISModelWriter;
import opennlp.maxent.io.SuffixSensitiveGISModelReader;
import opennlp.maxent.io.SuffixSensitiveGISModelWriter;
import opennlp.model.EventStream;
import opennlp.model.RealValueFileEventStream;

import java.io.*;
import java.util.HashSet;
import java.util.Set;

public class MaxEntClassifier {
    // Annotated Training Data Delimiter
    private static final String DELIMITER = " ";
    private static final boolean USE_SMOOTHING = true;
    private static final int ITERATIONS = 100;
    private static final int CUTOFF = 0;
    private FeatureExtractor featureExtractor;
    private String outputDirPath;
    private GISModel model;
    private boolean isRealFeature = false;

    private void extractFeatureVector(Set<String> trainingData) {
        // Set isRealFeatureFlag
        this.isRealFeature = featureExtractor.getIsRealFeature();

        File featureVectorFile = new File(outputDirPath + "/featureVector.txt");
        if (!featureVectorFile.exists()) {
            try {
                featureVectorFile.createNewFile();

                FileWriter fw = new FileWriter(featureVectorFile.getAbsoluteFile());
                BufferedWriter bw = new BufferedWriter(fw);

                Set<String> featureVector = featureExtractor.getFeats(trainingData);
                for (String s : featureVector) {
                    bw.write(s + System.getProperty("line.separator"));
                }

                bw.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    public void train(String trainFilePath) throws IOException {
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

        extractFeatureVector(trainingData);

        EventStream es;
        if (isRealFeature) {
            es = new RealBasicEventStream(new PlainTextByLineDataStream(new FileReader(outputDirPath + "/featureVector.txt")));
        } else {
            es = new BasicEventStream(new PlainTextByLineDataStream(new FileReader(outputDirPath + "/featureVector.txt")));
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
        System.out.println("Loading MaxEnt model ... ");
        model = (GISModel) new SuffixSensitiveGISModelReader(new File(modelPath)).getModel();
        long end = System.currentTimeMillis();
        long duration = (end - start) / 1000;
        System.out.println("completed ... " + duration + " secs.");

    }

    private double[] predictOCS(String sentence) {
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

    public String predict(String sentence) {
        return model.getBestOutcome(predictOCS(sentence));
    }

    public String predictAllOutcome(String sentence) {
        return model.getAllOutcomes(predictOCS(sentence));
    }

    public MaxEntClassifier(String outputDirPath) {
        this.outputDirPath = outputDirPath;
        featureExtractor = new FeatureExtractor(outputDirPath, DELIMITER);
    }

    // Annotate a whole file.
    public static void annotate(String testFilePath, MaxEntClassifier maxEntClassifier) {
        try (BufferedReader br = new BufferedReader(new FileReader(testFilePath))) {
            BufferedWriter bw = new BufferedWriter(new FileWriter(testFilePath + ".result"));

            String line = br.readLine();
            while (line != null) {
                String result = maxEntClassifier.predict(line);
                bw.write(line.replaceAll("\\s", "_") + " " + result + System.getProperty("line.separator"));

                line = br.readLine();
            }

            bw.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static void main(String[] args) throws IOException {
        String prefix = "/Users/Maochen/workspace/Test/fixture/";
        String trainFilePath = prefix + "annotatedTrainingData.txt";
        String modelPath = prefix + "model.dat";

        MaxEntClassifier maxEntClassifier = new MaxEntClassifier(prefix);
        maxEntClassifier.train(trainFilePath);
        maxEntClassifier.persist(modelPath);

        maxEntClassifier.loadModel(modelPath);
        annotate("/Users/Maochen/Desktop/test.txt", maxEntClassifier);

        //        Scanner scanner = new Scanner(System.in);
        //        String sentence = "";
        //        System.out.println("Input Sentence:");
        //        while (!sentence.equalsIgnoreCase("exit")) {
        //            sentence = scanner.nextLine();
        //            sentence = sentence.replaceAll("\\s", "_");
        //            // String vector = maxEnt.featureExtractor.getFeats(sentence + DELIMITER + "?");
        //            // System.out.println(vector);
        //            String result = maxEnt.predictAllOutcome(sentence);
        //            System.out.println(result + " ||| " + maxEnt.predict(sentence));
        //        }
    }
}
