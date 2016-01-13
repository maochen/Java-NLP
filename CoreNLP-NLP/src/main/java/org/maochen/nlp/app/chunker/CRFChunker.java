package org.maochen.nlp.app.chunker;

import org.apache.commons.lang3.StringUtils;
import org.apache.commons.lang3.tuple.Pair;
import org.maochen.nlp.ml.SequenceTuple;
import org.maochen.nlp.ml.Tuple;
import org.maochen.nlp.ml.classifier.crfsuite.CRFClassifier;
import org.maochen.nlp.ml.util.TrainingDataUtils;
import org.maochen.nlp.ml.vector.IVector;
import org.maochen.nlp.ml.vector.LabeledVector;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Properties;
import java.util.Scanner;
import java.util.stream.Collectors;

import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.TaggedWord;
import edu.stanford.nlp.tagger.maxent.MaxentTagger;

/**
 * Created by Maochen on 11/10/15.
 */
public class CRFChunker extends CRFClassifier {

    private static final Logger LOG = LoggerFactory.getLogger(CRFChunker.class);

    public static String TRAIN_FILE_DELIMITER = "\t";

    public void train(final String trainFilePath) throws FileNotFoundException {
        // Preparing training data
        List<SequenceTuple> trainingData = TrainingDataUtils.readSeqFile(new FileInputStream(new File(trainFilePath)), TRAIN_FILE_DELIMITER, 2);
        // -----------
        LOG.info("Loaded Training data.");

        LOG.info("Generating feats");
        trainingData.stream().forEach(seq -> seq.entries = ChunkerFeatureExtractor.extractFeat(seq));
        LOG.info("Extracted Feats.");

        super.train(trainingData);
    }

    public SequenceTuple predict(final String[] words, final String[] pos) {
        SequenceTuple st = new SequenceTuple();
        st.entries = new ArrayList<>();
        for (int i = 0; i < words.length; i++) {
            IVector v = new LabeledVector(new String[]{words[i], pos[i]});
            st.entries.add(new Tuple(v));
        }

        st.entries = ChunkerFeatureExtractor.extractFeat(st);
        List<Pair<String, Double>> result = super.predict(st);
        List<String> tags = result.stream().map(Pair::getLeft).collect(Collectors.toList());

        for (int i = 0; i < tags.size(); i++) {
            st.entries.get(i).label = tags.get(i);
        }

        return st;
    }

    public void validate(String testFile) throws FileNotFoundException {
        List<SequenceTuple> testData = TrainingDataUtils.readSeqFile(new FileInputStream(new File(testFile)), TRAIN_FILE_DELIMITER, 2);
        int errCount = 0;
        int total = 0;

        for (SequenceTuple st : testData) {
            total += st.entries.size();
            List<String> expectedTags = new ArrayList<>(st.getLabel());
            String[] words = st.entries.stream().map(x -> ((LabeledVector) x.vector).featsName[ChunkerFeatureExtractor.WORD_INDEX]).toArray(String[]::new);
            String[] pos = st.entries.stream().map(x -> ((LabeledVector) x.vector).featsName[ChunkerFeatureExtractor.POS_INDEX]).toArray(String[]::new);

            st = predict(words, pos);

            boolean isThisSeqPrinted = false;
            for (int i = 0; i < expectedTags.size(); i++) {
                if (!expectedTags.get(i).equals(st.entries.get(i).label)) {
                    if (!isThisSeqPrinted) {
                        printSequenceTuple(st, expectedTags);
                        System.out.println("");
                        isThisSeqPrinted = true;
                    }
                    errCount++;
                }
            }
        }

        System.out.println("Err/Total:\t" + errCount + "/" + total);
        System.out.println("Accurancy:\t" + (1 - (errCount / (double) total)) * 100 + "%");
    }


    public static void printSequenceTuple(final SequenceTuple st, List<String> correctTag) {
        String[] tokens = st.entries.stream()
                .map(tuple -> Arrays.stream(((LabeledVector) tuple.vector).featsName)
                        .filter(x -> x.startsWith("w0="))
                        .map(t -> t.split("=")[1])
                        .findFirst().orElse(StringUtils.EMPTY))
                .toArray(String[]::new);

        for (int i = 0; i < tokens.length; i++) {
            String out = tokens[i] + "\t" + st.entries.get(i).label;
            if (correctTag != null && !st.entries.get(i).label.equals(correctTag.get(i))) {
                out += "\t" + "Expected:\t" + correctTag.get(i);
            }
            System.out.println(out);
        }
    }

    public static void main(String[] args) throws IOException {
        CRFChunker chunker = new CRFChunker();
        CRFChunker.TRAIN_FILE_DELIMITER = StringUtils.SPACE;
        String modelPath = "/Users/mguan/Desktop/chunker.crf.model";

        Properties para = new Properties();
        para.setProperty("model", modelPath);
        para.setProperty("algorithm", "l2sgd");
        para.setProperty("feature.possible_transitions", "1");
        para.setProperty("feature.possible_states", "1");

        chunker.setParameter(para);

        String trainFile = "/Users/mguan/workspace/nlp-service_training-data/corpora/CoNLL_Shared_Task/CoNLL_2000_Chunking/train.txt";

//        chunker.train(trainFile);
//        chunker.persistModel(modelPath + ".dup");
//        para.setProperty("model", modelPath + ".dup");
//        chunker.setParameter(para);
//        chunker.loadModel(null);

        chunker.validate("/Users/mguan/workspace/nlp-service_training-data/corpora/CoNLL_Shared_Task/CoNLL_2000_Chunking/test.txt");

        MaxentTagger posTagger = new MaxentTagger("edu/stanford/nlp/models/pos-tagger/english-left3words/english-left3words-distsim.tagger");
        Scanner scan = new Scanner(System.in);
        String input = StringUtils.EMPTY;
        String quitRegex = "q|quit|exit";
        while (!input.matches(quitRegex)) {
            System.out.println("Please enter sentence:");
            input = scan.nextLine();
            if (!input.trim().isEmpty() && !input.matches(quitRegex)) {
                String[] words = input.split("\\s");

                List<CoreLabel> tokens = Arrays.stream(words).map(word -> {
                    CoreLabel coreLabel = new CoreLabel();
                    coreLabel.setWord(word);
                    coreLabel.setOriginalText(word);
                    coreLabel.setValue(word);
                    return coreLabel;
                }).collect(Collectors.toList());

                List<TaggedWord> posList = posTagger.tagSentence(tokens);
                for (int i = 0; i < tokens.size(); i++) {
                    String pos = posList.get(i).tag();
                    tokens.get(i).setTag(pos);
                }

                String[] pos = tokens.stream().map(CoreLabel::tag).toArray(String[]::new);

                SequenceTuple st = chunker.predict(words, pos);
                printSequenceTuple(st, null);
            }
        }


    }
}
