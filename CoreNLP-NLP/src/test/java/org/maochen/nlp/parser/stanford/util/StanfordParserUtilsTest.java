package org.maochen.nlp.parser.stanford.util;

import org.junit.Test;

import java.util.List;

import static org.junit.Assert.assertEquals;

/**
 * Created by Maochen on 2/18/16.
 */
public class StanfordParserUtilsTest {

    @Test
    public void testTokenize() {
        List<String> result = StanfordParserUtils.tokenize("I have a car.");
        assertEquals(5, result.size());
        assertEquals("I", result.get(0));
        assertEquals("have", result.get(1));
        assertEquals("a", result.get(2));
        assertEquals("car", result.get(3));
        assertEquals(".", result.get(4));
    }

    @Test
    public void testSegmenter() {
        String blob = "Caterpillar Inc., is an American corporation. " +
                "Caterpillar was ranked number one in its industry and number 44 overall in the 2009 Fortune 500. " +
                "Caterpillar stock is a component of the Dow Jones Industrial Average. " +
                "Caterpillar Inc. traces its origins to the 1925.";

        List<String> actual = StanfordParserUtils.segmenter(blob);

        assertEquals(4, actual.size());
        assertEquals("Caterpillar Inc. , is an American corporation .", actual.get(0));
        assertEquals("Caterpillar was ranked number one in its industry and number 44 overall in the 2009 Fortune 500 .", actual.get(1));
        assertEquals("Caterpillar stock is a component of the Dow Jones Industrial Average .", actual.get(2));
        assertEquals("Caterpillar Inc. traces its origins to the 1925 .", actual.get(3));
    }
}
