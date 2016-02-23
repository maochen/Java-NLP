package org.maochen.nlp.app.ner;

import org.apache.commons.lang3.StringUtils;
import org.apache.commons.lang3.tuple.Pair;
import org.maochen.nlp.app.featextractor.IFeatureExtractor;
import org.maochen.nlp.ml.SequenceTuple;
import org.maochen.nlp.ml.Tuple;
import org.maochen.nlp.ml.classifier.crfsuite.CRFClassifier;
import org.maochen.nlp.ml.util.TrainingDataUtils;
import org.maochen.nlp.ml.vector.IVector;
import org.maochen.nlp.ml.vector.LabeledVector;
import org.maochen.nlp.util.ValidationUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Properties;
import java.util.Scanner;
import java.util.stream.Collectors;

/**
 * Created by Maochen on 2/22/16.
 */
public class CRFNER extends CRFClassifier {

    private static final Logger LOG = LoggerFactory.getLogger(CRFNER.class);

    public static String TRAIN_FILE_DELIMITER = "\t";
    public static final int TRAIN_FILE_TAG_COL = 1;

    public IFeatureExtractor featureExtractor;

    public void train(final String trainFilePath) throws FileNotFoundException {

        // Preparing training data
        List<SequenceTuple> trainingData = TrainingDataUtils.readSeqFile(new FileInputStream(new File(trainFilePath)), TRAIN_FILE_DELIMITER, TRAIN_FILE_TAG_COL);
        // -----------
        LOG.info("Loaded Training data.");

        LOG.info("Generating feats");
        trainingData.stream().forEach(seq -> seq.entries = featureExtractor.extractFeat(seq));
        LOG.info("Extracted Feats.");

        super.train(trainingData);
    }

    public SequenceTuple predict(final String[] words) {
        SequenceTuple st = new SequenceTuple();
        st.entries = new ArrayList<>();
        for (int i = 0; i < words.length; i++) {
            IVector v = new LabeledVector(new String[]{words[i]});
            st.entries.add(new Tuple(v));
        }

        st.entries = featureExtractor.extractFeat(st);
        List<Pair<String, Double>> result = super.predict(st);
        List<String> tags = result.stream().map(Pair::getLeft).collect(Collectors.toList());

        for (int i = 0; i < tags.size(); i++) {
            st.entries.get(i).label = tags.get(i);
        }

        return st;
    }

    public void validate(String testFile) throws FileNotFoundException {
        List<SequenceTuple> testData = TrainingDataUtils.readSeqFile(new FileInputStream(new File(testFile)), TRAIN_FILE_DELIMITER, TRAIN_FILE_TAG_COL);
        int errCount = 0;
        int total = 0;

        for (SequenceTuple st : testData) {
            total += st.entries.size();
            List<String> expectedTags = new ArrayList<>(st.getLabel());
            String[] words = st.entries.stream().map(x -> ((LabeledVector) x.vector).featsName[0]).toArray(String[]::new);

            st = predict(words);

            boolean isThisSeqPrinted = false;
            for (int i = 0; i < expectedTags.size(); i++) {
                if (!expectedTags.get(i).equals(st.entries.get(i).label)) {
                    if (!isThisSeqPrinted) {
                        ValidationUtils.printSequenceTuple(st, expectedTags);
                        System.out.println("");
                        isThisSeqPrinted = true;
                    }
                    errCount++;
                }
            }
        }

        System.out.println("Err/Total:\t" + errCount + "/" + total);
        System.out.println("Accuracy:\t" + (1 - (errCount / (double) total)) * 100 + "%");
    }

    public static void main(String[] args) throws IOException {
        CRFNER ner = new CRFNER();
        ner.featureExtractor = new NERFeatureExtractor();
        String modelPath = "/Users/mguan/Desktop/ner.crf.model";

        Properties para = new Properties();
        para.setProperty("model", modelPath);
        para.setProperty("algorithm", "l2sgd");
        para.setProperty("feature.possible_transitions", "1");
        para.setProperty("feature.possible_states", "1");

        ner.setParameter(para);

        String trainFile = "/Users/mguan/Desktop/npaper.collx.txt";

        ner.train(trainFile);
        para.setProperty("model", modelPath);
        ner.setParameter(para);
        ner.loadModel(null);

        ner.validate("/Users/mguan/Desktop/npaper.collx.txt");

        Scanner scan = new Scanner(System.in);
        String input = StringUtils.EMPTY;
        String quitRegex = "q|quit|exit";
        while (!input.matches(quitRegex)) {
            System.out.println("Please enter sentence:");
            input = scan.nextLine();
            if (!input.trim().isEmpty() && !input.matches(quitRegex)) {
                String[] words = input.split("\\s");

                SequenceTuple st = ner.predict(words);
                ValidationUtils.printSequenceTuple(st, null);
            }
        }
    }
}
