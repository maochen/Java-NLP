package org.maochen.nlp.coref;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Lists;

import org.junit.Test;
import org.maochen.nlp.parser.stanford.coref.StanfordCoref;
import org.maochen.nlp.parser.stanford.pcfg.StanfordPCFGParser;

import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.assertEquals;


/**
 * Created by Maochen on 7/20/15.
 */
public class CorefTest {

    private static StanfordPCFGParser parser = new StanfordPCFGParser(null, null, new ArrayList<>());
    private static StanfordCoref coref = new StanfordCoref(parser);

    @Test
    public void test() {
        List<String> texts = Lists.newArrayList("Google invests in companies since they have more choices.", "They understand it is hard.");
        List<String> actualList = coref.getCoref(texts);
        List<String> expectedList = ImmutableList.of("Google invests in companies since companies have more choices .", "companies understand Google is hard .");

        for (int i = 0; i < actualList.size(); i++) {
            assertEquals(expectedList.get(i), actualList.get(i));
        }
    }

    @Test
    public void testStanfordCoref2() {
        String input1 = "Other important legislation involved economic matters, including the first income tax and higher tariffs.";
        String input2 = "The Morrill Land-Grant Colleges Act, also signed in 1862, provided government grants for agricultural universities in each state.";
        List<String> corefed = coref.getCoref(Lists.newArrayList(input1, input2));

        List<String> expected = Lists.newArrayList("Other important legislation involved economic matters , including the first income tax and higher tariffs .",
                "The Morrill Land-Grant Colleges Act , also signed in 1862 , provided government grants for agricultural universities in each state .");

        for (int i = 0; i < expected.size(); i++) {
            assertEquals(expected.get(i), corefed.get(i));
        }
    }
}
