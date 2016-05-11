package org.maochen.nlp.ml;

import org.junit.Test;
import org.maochen.nlp.ml.vector.FeatNamedVector;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static junit.framework.Assert.assertEquals;

/**
 * Created by Maochen on 11/11/15.
 */
public class SequenceTupleTest {

    @Test
    public void test() {
        Map<Integer, List<String>> feats = new HashMap<>();

        List<String> token = new ArrayList<>();
        token.add("Energy");

        List<String> pos = new ArrayList<>();
        pos.add("NNP");
        feats.put(0, token);
        feats.put(1, pos);

        List<String> tag = new ArrayList<>();
        tag.add("B-NP");
        SequenceTuple st = new SequenceTuple(feats, tag);

        assertEquals(1, st.entries.size());
        Tuple entry = st.entries.get(0);
        assertEquals("B-NP", entry.label);

        String expected = Arrays.toString(new String[]{"Energy", "NNP"});
        String actual = Arrays.toString(((FeatNamedVector) entry.vector).featsName);
        assertEquals(expected, actual);

    }
}
