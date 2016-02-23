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
    public void testTokenizer2() {
        String sentence = "<ENAMEX TYPE=\"PERSON\">Danasia</ENAMEX> was traveling.";
        List<String> result = StanfordParserUtils.tokenize(sentence);
        assertEquals(6, result.size());
        assertEquals("<ENAMEX TYPE=\"PERSON\">", result.get(0));
    }

    @Test
    public void testSegmenter() {
        String blob = "I have a \"car\". T.J. Watson is located in (NYC).";

        List<String> actual = StanfordParserUtils.segmenter(blob);

        assertEquals(2, actual.size());
        assertEquals("I have a \"car\".", actual.get(0));
        assertEquals("T.J. Watson is located in (NYC).", actual.get(1));
    }
}
