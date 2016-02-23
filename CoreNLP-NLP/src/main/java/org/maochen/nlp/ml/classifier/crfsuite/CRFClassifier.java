package org.maochen.nlp.ml.classifier.crfsuite;

import com.github.jcrfsuite.util.CrfSuiteLoader;

import org.apache.commons.lang3.tuple.ImmutablePair;
import org.apache.commons.lang3.tuple.Pair;
import org.maochen.nlp.ml.ISeqClassifier;
import org.maochen.nlp.ml.SequenceTuple;
import org.maochen.nlp.ml.Tuple;
import org.maochen.nlp.ml.vector.LabeledVector;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.List;
import java.util.Properties;
import java.util.stream.Collectors;

import third_party.org.chokkan.crfsuite.Attribute;
import third_party.org.chokkan.crfsuite.Item;
import third_party.org.chokkan.crfsuite.ItemSequence;
import third_party.org.chokkan.crfsuite.StringList;
import third_party.org.chokkan.crfsuite.Tagger;
import third_party.org.chokkan.crfsuite.Trainer;

/**
 * Created by Maochen on 1/11/16.
 */
public class CRFClassifier implements ISeqClassifier {

    private static final Logger LOG = LoggerFactory.getLogger(CRFClassifier.class);

    private Properties props = new Properties();

    private String modelPath = null;
    private Tagger tagger = null;

    // Algorithm type: lbfgs, l2sgd, averaged-perceptron, passive-aggressive, arow
    private static final String DEFAULT_ALGORITHM = "lbfgs";
    private static final String DEFAULT_GRAPHICAL_MODEL_TYPE = "crf1d";

    static {
        try {
            CrfSuiteLoader.load();
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    private static Pair<List<ItemSequence>, List<StringList>> loadTrainingData(List<SequenceTuple> trainingData) {
        List<ItemSequence> xseqs = new ArrayList<>();
        List<StringList> yseqs = new ArrayList<>();

        for (SequenceTuple sequenceTuple : trainingData) {
            xseqs.add(getXseqForOneSeqTuple(sequenceTuple));
            StringList yseq = new StringList();
            sequenceTuple.getLabel().stream().forEach(yseq::add);
            yseqs.add(yseq);
        }

        return new ImmutablePair<>(xseqs, yseqs);
    }

    private static ItemSequence getXseqForOneSeqTuple(final SequenceTuple sequenceTuple) {
        ItemSequence xseq = new ItemSequence();

        for (Tuple t : sequenceTuple.entries) {
            // Add item which is a list of attributes
            Item item = new Item();
            for (int i = 0; i < t.vector.getVector().length; i++) {
                Attribute attr;

                if (t.vector instanceof LabeledVector) {
                    attr = new Attribute(((LabeledVector) t.vector).featsName[i]);
                } else {
                    attr = new Attribute(String.valueOf(i), t.vector.getVector()[i]);
                }

                item.add(attr);
            }
            xseq.add(item);
        }

        return xseq;
    }

    /**
     * Train CRF Suite with annotated item sequences.
     */
    @Override
    public ISeqClassifier train(List<SequenceTuple> trainingData) {
        if (trainingData == null || trainingData.size() == 0) {
            LOG.warn("Training data is empty.");
            return this;
        }

        if (modelPath == null) {
            try {
                modelPath = Files.createTempDirectory("crfsuite").toAbsolutePath().toString();
            } catch (IOException e) {
                LOG.error("Create temp directory failed.", e);
                e.printStackTrace();
            }
        }

        Pair<List<ItemSequence>, List<StringList>> crfCompatibleTrainingData = loadTrainingData(trainingData);
        Trainer trainer = new Trainer();

        String algorithm = (String) props.getOrDefault("algorithm", DEFAULT_ALGORITHM);
        props.remove("algorithm");
        String graphicalModelType = (String) props.getOrDefault("graphicalModelType", DEFAULT_GRAPHICAL_MODEL_TYPE);
        props.remove("graphicalModelType");

        trainer.select(algorithm, graphicalModelType);

        // Set parameters
        props.entrySet().forEach(pair -> trainer.set((String) pair.getKey(), (String) pair.getValue()));

        // Add training data into the trainer
        for (int i = 0; i < trainingData.size(); i++) {
            // Use group id = 0 but the API doesn't say what it is used for :(
            trainer.append(crfCompatibleTrainingData.getLeft().get(i), crfCompatibleTrainingData.getRight().get(i), 0);
        }

        // Start training without hold-outs. trainer.message()
        // will be called to report the training process
        trainer.train(modelPath, -1);
        return this;
    }

//    public static void writeTmpFile(String filename, Pair<List<ItemSequence>, List<StringList>> data) {
//        List<ItemSequence> feats = data.getKey();
//        List<StringList> labels = data.getValue();
//        try (BufferedWriter output = new BufferedWriter(new FileWriter(new File(filename)))) {
//            for (int i = 0; i < feats.size(); i++) {
//                ItemSequence feat = feats.get(i);
//                StringList tags = labels.get(i);
//                StringBuilder stringBuilder = new StringBuilder();
//                for (int j = 0; j < feat.size(); j++) {
//                    String tag = tags.get(j);
//                    stringBuilder.append(tag).append("\t");
//
//                    Item item = feat.get(j);
//                    for (int k = 0; k < item.size(); k++) {
//                        String key = item.get(k).getAttr();
//                        stringBuilder.append(key);
//                        stringBuilder.append("\t");
//                    }
//                    stringBuilder.deleteCharAt(stringBuilder.length() - 1);
//                    stringBuilder.append(System.lineSeparator());
//                }
//                stringBuilder.append(System.lineSeparator());
//                output.write(stringBuilder.toString());
//            }
//        } catch (IOException e) {
//            e.printStackTrace();
//        }
//    }

    @Override
    public synchronized List<Pair<String, Double>> predict(SequenceTuple sequenceTuple) {
        if (tagger == null) {
            loadModel(null);
        }

        List<Pair<String, Double>> taggedSentences = new ArrayList<>();

        tagger.set(getXseqForOneSeqTuple(sequenceTuple));
        StringList labels = tagger.viterbi();

        for (int i = 0; i < labels.size(); i++) {
            String label = labels.get(i);
            taggedSentences.add(new ImmutablePair<>(label, tagger.marginal(label, i)));
        }

        return taggedSentences;
    }

    @Override
    public void setParameter(Properties props) {
//        trainer.set("c1", "0.25");
//        trainer.set("c2", "0.1");
//        trainer.set("epsilon", "0.0000001");
//        trainer.set("delta", "0.0000001");
//        trainer.set("num_memories", "6");

        modelPath = (String) props.getOrDefault("model", null);
        props.remove("model");
        this.props = props;
    }

    @Override
    public void persistModel(String modelFile) throws IOException {
        if (modelPath.equals(modelFile)) {
            throw new IOException("same as original model path.");
        }

        File sourceFile = new File(modelPath);
        File destFile = new File(modelFile);
        Files.copy(sourceFile.toPath(), destFile.toPath());
    }


    // Err, Total
    public Pair<Integer, Integer> validate(List<SequenceTuple> testingData) {
        int total = testingData.stream().mapToInt(st -> st.entries.size()).sum();
        int err = 0;
        for (SequenceTuple st : testingData) {
            List<String> actual = predict(st).stream().map(Pair::getLeft).collect(Collectors.toList());
            List<String> expected = st.getLabel();
            if (actual.size() != expected.size()) {
                throw new RuntimeException("Actual size: " + actual.size() + "\tExpected size: " + expected.size());
            }

            for (int i = 0; i < actual.size(); i++) {
                if (!actual.get(i).equals(expected.get(i))) {
                    err++;
                }
            }
        }

        System.out.println("Err/Total: " + err + "/" + total);
        System.out.println("Accuracy: " + (1 - (err / (double) total)) * 100 + "%");
        return new ImmutablePair<>(err, total);
    }

    @Override
    public void loadModel(InputStream modelFile) {
        if (modelPath == null) {
            throw new IllegalArgumentException("Please set model path parameter to load model");
        } else {
            tagger = new Tagger();
            boolean ret = tagger.open(modelPath);
            if (!ret) {
                LOG.error("Unable load model: " + modelPath);
            }
        }
    }

    public CRFClassifier() {

    }

    public CRFClassifier(Properties props) {
        setParameter(props);
    }
}
