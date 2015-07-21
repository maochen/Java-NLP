package org.maochen.stanford.coref;

import com.google.common.collect.Lists;
import org.apache.commons.lang3.StringUtils;
import org.junit.Test;
import org.maochen.parser.stanford.coref.StanfordCoref;
import org.maochen.parser.stanford.pcfg.StanfordPCFGParser;

import java.util.List;

import static org.junit.Assert.assertEquals;

/**
 * Created by Maochen on 7/20/15.
 */
public class CorefTest {

    @Test
    public void test() {
        StanfordPCFGParser parser = new StanfordPCFGParser(null, null, true);
        StanfordCoref coref = new StanfordCoref(parser);

        List<String> texts = Lists.newArrayList("Google invests in companies since they have more choices");
        String actual = coref.getCoref(texts).stream().findFirst().orElse(StringUtils.EMPTY);
        String expected = "Google invests in companies since companies have more choices";
        assertEquals(expected, actual);
    }
}
