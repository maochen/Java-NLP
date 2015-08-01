package org.maochen.nlp.coref;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Lists;
import org.junit.Assert;
import org.junit.Test;
import org.maochen.nlp.parser.stanford.coref.StanfordCoref;
import org.maochen.nlp.parser.stanford.pcfg.StanfordPCFGParser;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by Maochen on 7/20/15.
 */
public class CorefTest {

    @Test
    public void test() {
        StanfordPCFGParser parser = new StanfordPCFGParser(null, null, new ArrayList<>());
        StanfordCoref coref = new StanfordCoref(parser);

        List<String> texts = Lists.newArrayList("Google invests in companies since they have more choices.", "They understand it is hard.");
        List<String> actualList = coref.getCoref(texts);
        List<String> expectedList = ImmutableList.of("Google invests in companies since companies have more choices .", "companies understand Google is hard .");

        for (int i = 0; i < actualList.size(); i++) {
            Assert.assertEquals(expectedList.get(i), actualList.get(i));
        }
    }
}
