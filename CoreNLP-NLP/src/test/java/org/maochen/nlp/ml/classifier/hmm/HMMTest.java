package org.maochen.nlp.ml.classifier.hmm;

import com.google.common.collect.Lists;
import org.junit.Test;
import org.maochen.nlp.ml.SequenceTuple;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.*;

/**
 * Created by Maochen on 8/5/15.
 */
public class HMMTest {

    @Test
    public void testWriteReadModel() {
        HMMModel model = new HMMModel();
        model.emission.put("fish", "NN", 8D);
        model.emission.put("sleep", "NN", 2D);

        model.emission.put("fish", "VB", 5D);
        model.emission.put("sleep", "VB", 5D);

        try {
            Path tempDir = Files.createTempDirectory("HMMModelTest");
            String path = tempDir.toAbsolutePath().toString() + "/hmm_model.dat";
            HMM.saveModel(path, model);
            HMMModel newModel = HMM.loadModel(path);
            assertNotNull(newModel);
            assertEquals(5D, newModel.emission.get("fish", "VB"), Double.MIN_NORMAL);
        } catch (IOException e) {
            fail(e.getMessage());
        }

    }

    @Test
    public void testNormalize() {
        HMMModel model = new HMMModel();
        model.emission.put("fish", "NN", 8D);
        model.emission.put("sleep", "NN", 2D);

        model.emission.put("fish", "VB", 5D);
        model.emission.put("sleep", "VB", 5D);

        HMM.normalizeEmission(model);

        assertEquals(0.8, model.emission.get("fish", "NN"), Double.MIN_NORMAL);
        assertEquals(0.2, model.emission.get("sleep", "NN"), Double.MIN_NORMAL);

        assertEquals(0.5, model.emission.get("fish", "VB"), Double.MIN_NORMAL);
        assertEquals(0.5, model.emission.get("sleep", "VB"), Double.MIN_NORMAL);


        model.transition.put("NN", "VB", 4D);
        model.transition.put("NN", "JJ", 6D);

        model.transition.put("DT", "NN", 6D);
        HMM.normalizeTrans(model);

        assertEquals(0.4, model.transition.get("NN", "VB"), Double.MIN_NORMAL);
        assertEquals(0.6, model.transition.get("NN", "JJ"), Double.MIN_NORMAL);

        assertEquals(1.0, model.transition.get("DT", "NN"), Double.MIN_NORMAL);
    }

    @Test
    public void testViterbi() {
        HMMModel model = new HMMModel();
        model.emission.put("fish", "NN", 0.8);
        model.emission.put("sleep", "NN", 0.2);

        model.emission.put("fish", "VB", 0.5);
        model.emission.put("sleep", "VB", 0.5);

        model.emission.put(HMM.START, HMM.START, 1.0);
        model.emission.put(HMM.END, HMM.END, 1.0);


        model.transition.put(HMM.START, "NN", 0.8);
        model.transition.put(HMM.START, "VB", 0.2);
        model.transition.put("NN", "NN", 0.1);
        model.transition.put("NN", "VB", 0.8);
        model.transition.put("NN", HMM.END, 0.1);
        model.transition.put("VB", HMM.END, 0.7);
        model.transition.put("VB", "VB", 0.1);
        model.transition.put("VB", "NN", 0.2);

        List<String> result = Viterbi.resolve(model, new String[]{"fish", "sleep"});
        List<String> expected = Lists.newArrayList("NN", "VB");
        assertEquals(result, expected);
    }

    @Test
    public void testEnd2end() {
        String trainingFile = HMMTest.class.getResource("/brown_masc_pos/training.pos").getPath();
        String devFile = HMMTest.class.getResource("/brown_masc_pos/development.pos").getPath();

        List<SequenceTuple> trainData = HMM.readTrainFile(trainingFile, "\t", 0, 1);

        HMMModel model = HMM.train(trainData);
        Map<String, Double> result = HMM.eval(model, devFile, "\t", 0, 1, false);
        assertEquals(0.8778, result.get("accuracy"), 0.001);

        // Please add dot in the end. All training data ends with dot, so the transition from anything other than dot to <END> is 0
        String str = "The quick brown fox jumped over the lazy dog .";
        String[] predictions = HMM.viterbi(model, str.split("\\s")).stream().toArray(String[]::new);
        String[] expected = new String[]{"DT", "JJ", "NN", "NN", "VBD", "IN", "DT", "JJ", "NN", "."};
        assertEquals(Arrays.toString(expected), Arrays.toString(predictions));
    }

}
