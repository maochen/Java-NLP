package org.maochen.nlp.app.chunker;

import org.maochen.nlp.app.ISeqTagger;
import org.maochen.nlp.ml.SequenceTuple;
import org.maochen.nlp.ml.Tuple;
import org.maochen.nlp.ml.classifier.maxent.MaxEntClassifier;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * Created by Maochen on 11/10/15.
 */
public class Chunker extends MaxEntClassifier implements ISeqTagger {

    private static final Logger LOG = LoggerFactory.getLogger(Chunker.class);

    public static final int WORD_INDEX = 0;
    public static final int POS_INDEX = 1;

    @Override
    public void train(String trainFilePath) {
        Map<String, String> para = new HashMap<String, String>() {{
            put("iterations", "120");
        }};
        super.setParameter(para);

        // Preparing training data
        Set<SequenceTuple> trainingData = new HashSet<>();


        Map<Integer, List<String>> feats = new HashMap<>();
        List<String> words = new ArrayList<>();
        List<String> pos = new ArrayList<>();
        List<String> tag = new ArrayList<>();
        feats.put(WORD_INDEX, words);
        feats.put(POS_INDEX, pos);

        try (BufferedReader br = new BufferedReader(new FileReader(trainFilePath))) {
            String line = br.readLine();
            while (line != null) {
                if (!line.trim().isEmpty()) {
                    String[] fields = line.split("\t");
                    words.add(fields[0]);
                    pos.add(fields[1]);
                    tag.add(fields[2]); // chunker label.
                } else {
                    SequenceTuple current = new SequenceTuple(feats, tag);
                    trainingData.add(current);
                    words = new ArrayList<>();
                    pos = new ArrayList<>();
                    tag = new ArrayList<>();
                }

                line = br.readLine();
            }
        } catch (IOException e) {
            LOG.error("load data err.", e);
        }

        // -----------
        LOG.info("Loaded Training data.");

        LOG.info("Generating feats");
        List<Tuple> trainingTuples = ChunkerFeatureExtractor.extract(trainingData);
        LOG.info("Extracted Feats.");
        super.train(trainingTuples);

    }

    @Override
    public SequenceTuple predict(String sentence) {
        throw new IllegalArgumentException("Please use SequenceTuple");
    }

    @Override
    public void predict(final SequenceTuple sequenceTuple) {
        if (sequenceTuple == null) {
            return;
        }

        List<Tuple> predicts = ChunkerFeatureExtractor.extractFeat(sequenceTuple);

        sequenceTuple.tag = IntStream.range(0, predicts.size())
                .mapToObj(i ->
                        super.predict(predicts.get(i))
                                .entrySet().stream()
                                .max((t1, t2) -> t1.getValue().compareTo(t2.getValue())).map(Map.Entry::getKey).get()
                ).collect(Collectors.toList());

    }
}
