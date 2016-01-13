package org.maochen.nlp.app.chunker;

import org.apache.commons.lang3.StringUtils;
import org.maochen.nlp.ml.SequenceTuple;
import org.maochen.nlp.ml.Tuple;
import org.maochen.nlp.ml.classifier.maxent.MaxEntClassifier;
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
import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * Not as good as CRFs.
 *
 * Accuracy: 82.95375393123246%
 *
 * <p>Created by Maochen on 11/10/15.
 */
public class MaxEntChunker extends MaxEntClassifier {

    private static final Logger LOG = LoggerFactory.getLogger(MaxEntChunker.class);

    public static String TRAIN_FILE_DELIMITER = "\t";

    public void train(String trainFilePath) throws FileNotFoundException {
        // Preparing training data
        List<SequenceTuple> trainingData = TrainingDataUtils.readSeqFile(new FileInputStream(new File(trainFilePath)), TRAIN_FILE_DELIMITER, 2);
        // -----------
        LOG.info("Loaded Training data.");

        LOG.info("Generating feats");
        List<Tuple> trainingTuples = trainingData.parallelStream().map(x -> ChunkerFeatureExtractor.extractFeat(x, false)).flatMap(Collection::stream).collect(Collectors.toList());

        LOG.info("Extracted Feats.");
        super.train(trainingTuples);

    }

    public SequenceTuple predict(final String[] words, final String[] pos) {
        SequenceTuple st = new SequenceTuple();
        st.entries = new ArrayList<>();
        for (int i = 0; i < words.length; i++) {
            IVector v = new LabeledVector(new String[]{words[i], pos[i]});
            st.entries.add(new Tuple(v));
        }

        for (int i = 0; i < st.entries.size(); i++) {
            Tuple t = st.entries.get(i);
            String[] currentFeats = ChunkerFeatureExtractor.extractFeatSingle(i, words, pos, st.getLabel().stream().toArray(String[]::new)).stream().toArray(String[]::new);
            ((LabeledVector) t.vector).featsName = currentFeats;
            t.vector.setVector(IntStream.range(0, currentFeats.length).mapToDouble(x -> 1.0D).toArray());
            t.label = super.predict(t).entrySet().stream()
                    .max((t1, t2) -> t1.getValue().compareTo(t2.getValue())).map(Map.Entry::getKey).get();
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
                        CRFChunker.printSequenceTuple(st, expectedTags);
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
        MaxEntChunker chunker = new MaxEntChunker();
        MaxEntChunker.TRAIN_FILE_DELIMITER = StringUtils.SPACE;
        String modelPath = "/Users/mguan/Desktop/chunker.maxent.model";

        Properties para = new Properties();
//        para.put("iter", "250");
        chunker.setParameter(para);

        String trainFile = "/Users/mguan/workspace/nlp-service_training-data/corpora/CoNLL_Shared_Task/CoNLL_2000_Chunking/train.txt";

        chunker.train(trainFile);
        chunker.persistModel(modelPath);

        chunker.loadModel(new FileInputStream(new File(modelPath)));
        chunker.validate("/Users/mguan/workspace/nlp-service_training-data/corpora/CoNLL_Shared_Task/CoNLL_2000_Chunking/test.txt");
    }
}
