package org.maochen.nlp.ml.classifier.crfsuite;

import org.apache.commons.lang3.tuple.Pair;
import org.junit.Test;
import org.maochen.nlp.ml.SequenceTuple;
import org.maochen.nlp.ml.util.TrainingDataUtils;

import java.io.IOException;
import java.nio.file.Files;
import java.util.List;
import java.util.Properties;

import static org.junit.Assert.assertEquals;

/**
 * Created by Maochen on 1/12/16.
 */
public class CRFClassifierTest {

    private CRFClassifier crfClassifier = new CRFClassifier();

    @Test
    public void test() throws IOException {
        String modelPath = Files.createTempDirectory("crfsuite").toAbsolutePath().toString();
        Properties properties = new Properties();
        properties.setProperty("model", modelPath + "/crf.model");
        crfClassifier.setParameter(properties);

        List<SequenceTuple> trainingData = TrainingDataUtils.readSeqFile(CRFClassifierTest.class.getResourceAsStream("/tweet-pos/train-oct27.txt"), "\t", 0);
        crfClassifier.train(trainingData);

        List<SequenceTuple> testingData = TrainingDataUtils.readSeqFile(CRFClassifierTest.class.getResourceAsStream("/tweet-pos/test-daily547.txt"), "\t", 0);
        Pair<Integer, Integer> errTotal = crfClassifier.validate(testingData);
        assertEquals((Integer) 542, errTotal.getLeft());
        assertEquals((Integer) 7707, errTotal.getRight());
    }
}
