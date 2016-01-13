package org.maochen.nlp.app.chunker;

import org.junit.Test;
import org.maochen.nlp.ml.SequenceTuple;
import org.maochen.nlp.ml.Tuple;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * Created by Maochen on 11/10/15.
 */
public class ChunkerFeatureExtractorTest {

    private static String[] MOCK_TOKENS = new String[]{
            "Forecasts", "for", "the", "trade", "figures", "range", "widely"
    };

    private static String[] MOCK_POS = new String[]{
            "NNS", "IN", "DT", "NN", "NNS", "VBP", "RB"
    };

    private static String[] MOCK_TAGS = new String[]{
            "B-NP", "B-PP", "B-NP", "I-NP", "I-NP", "B-VP", "B-ADVP"
    };

    @Test
    public void testExtractFeatFromString() {
        List<String> feats = ChunkerFeatureExtractor.extractFeatSingle(3, MOCK_TOKENS, MOCK_POS, MOCK_TAGS);

        assertEquals(31, feats.size());
        assertTrue(feats.contains("tag-3=B-NP"));
        assertTrue(feats.contains("pos-10+1=DT_NN_NNS"));
        assertTrue(feats.contains("pos-3=NNS"));
        assertTrue(feats.contains("w0=trade"));
        assertTrue(feats.contains("pos-2-1=IN_DT"));
        assertTrue(feats.contains("tag-2-1pos0=B-PP_B-NP_NN"));
        assertTrue(feats.contains("pos-3-2=NNS_IN"));
        assertTrue(feats.contains("pos0=NN"));
        assertTrue(feats.contains("pos-2=IN"));
        assertTrue(feats.contains("pos0+1=NN_NNS"));
        assertTrue(feats.contains("w-1=the"));
        assertTrue(feats.contains("pos-1=DT"));
        assertTrue(feats.contains("pos0+1+2=NN_NNS_VBP"));
        assertTrue(feats.contains("tag-3-2-1pos0=B-NP_B-PP_B-NP_NN"));
        assertTrue(feats.contains("tag-1=B-NP"));
        assertTrue(feats.contains("pos-30=NNS_NN"));
        assertTrue(feats.contains("pos-1tag-1pos0=DT_B-NP_NN"));
        assertTrue(feats.contains("tag-2=B-PP"));
        assertTrue(feats.contains("pos-20=IN_NN"));
        assertTrue(feats.contains("pos-2-10+1=IN_DT_NN_NNS"));
        assertTrue(feats.contains("pos0+2=NN_VBP"));
        assertTrue(feats.contains("w+1=figures"));
        assertTrue(feats.contains("pos+1=NNS"));
        assertTrue(feats.contains("pos+2=VBP"));

        assertTrue(feats.contains("w-10=the_trade"));
        assertTrue(feats.contains("pos-10=DT_NN"));
        assertTrue(feats.contains("w-2=for"));
        assertTrue(feats.contains("pos-2-10=IN_DT_NN"));
        assertTrue(feats.contains("w0+1=trade_figures"));
        assertTrue(feats.contains("w+2=range"));
        assertTrue(feats.contains("pos+1+2=NNS_VBP"));

        feats.remove("tag-3=B-NP");
        feats.remove("pos-10+1=DT_NN_NNS");
        feats.remove("pos-3=NNS");
        feats.remove("w0=trade");
        feats.remove("pos-2-1=IN_DT");
        feats.remove("tag-2-1pos0=B-PP_B-NP_NN");
        feats.remove("pos-3-2=NNS_IN");
        feats.remove("pos0=NN");
        feats.remove("pos-2=IN");
        feats.remove("pos0+1=NN_NNS");
        feats.remove("w-1=the");
        feats.remove("pos-1=DT");
        feats.remove("pos0+1+2=NN_NNS_VBP");
        feats.remove("tag-3-2-1pos0=B-NP_B-PP_B-NP_NN");
        feats.remove("tag-1=B-NP");
        feats.remove("pos-30=NNS_NN");
        feats.remove("pos-1tag-1pos0=DT_B-NP_NN");
        feats.remove("tag-2=B-PP");
        feats.remove("pos-20=IN_NN");
        feats.remove("pos-2-10+1=IN_DT_NN_NNS");
        feats.remove("pos0+2=NN_VBP");
        feats.remove("w+1=figures");
        feats.remove("pos+1=NNS");
        feats.remove("pos+2=VBP");
        feats.remove("w-10=the_trade");
        feats.remove("pos-10=DT_NN");
        feats.remove("w-2=for");
        feats.remove("pos-2-10=IN_DT_NN");
        feats.remove("w0+1=trade_figures");
        feats.remove("w+2=range");
        feats.remove("pos+1+2=NNS_VBP");

        assertTrue(feats.isEmpty());
    }

    @Test
    public void testExtractFeatSentenceTuple() {
        Map<Integer, List<String>> feats = new HashMap<>();
        feats.put(ChunkerFeatureExtractor.WORD_INDEX, Arrays.asList(MOCK_TOKENS));
        feats.put(ChunkerFeatureExtractor.POS_INDEX, Arrays.asList(MOCK_POS));
        SequenceTuple st = new SequenceTuple(feats, Arrays.asList(MOCK_TAGS));

        List<Tuple> tuples = ChunkerFeatureExtractor.extractFeat(st, true);

        assertEquals(7, tuples.size());

        for (int i = 0; i < tuples.size(); i++) {
            Tuple tuple = tuples.get(i);
            assertEquals(MOCK_TAGS[i], tuple.label);
        }
    }
}
