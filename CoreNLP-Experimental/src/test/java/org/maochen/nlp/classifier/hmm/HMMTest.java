package org.maochen.nlp.classifier.hmm;

import com.google.common.collect.Lists;

import org.junit.Test;

import java.util.List;

import static org.junit.Assert.assertEquals;

/**
 * Created by Maochen on 8/5/15.
 */
public class HMMTest {

    @Test
    public void testNormalize() {
        HMMModel model = new HMMModel();
        model.emission.put("fish", "NN", 8D);
        model.emission.put("sleep", "NN", 2D);

        model.emission.put("fish", "VB", 5D);
        model.emission.put("sleep", "VB", 5D);

        HMM.normalizeEmission(model);

        assertEquals(model.emission.get("fish", "NN"), 0.8, Double.MIN_NORMAL);
        assertEquals(model.emission.get("sleep", "NN"), 0.2, Double.MIN_NORMAL);

        assertEquals(model.emission.get("fish", "VB"), 0.5, Double.MIN_NORMAL);
        assertEquals(model.emission.get("sleep", "VB"), 0.5, Double.MIN_NORMAL);
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

        List<String> result = Viterbi.resolve(model, Lists.newArrayList("fish", "sleep"));
        List<String> expected = Lists.newArrayList("NN", "VB");
        assertEquals(result, expected);
    }
}
