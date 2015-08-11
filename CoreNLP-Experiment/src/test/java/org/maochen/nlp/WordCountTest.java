package org.maochen.nlp;

import org.junit.Test;

import static org.junit.Assert.assertEquals;

/**
 * Created by Maochen on 8/10/15.
 */
public class WordCountTest {
    @Test
    public void test() {
        WordCount wordCount = new WordCount();
        wordCount.put("c");
        wordCount.put("a");
        wordCount.put("b");
        wordCount.put("b");
        wordCount.put("b");
        wordCount.put("a");
        for (int i = 0; i < 199; i++) {
            wordCount.put("ZZ");
        }
        wordCount.normalize();
        wordCount.normalize();
        wordCount.normalize();

        assertEquals(0.9707317073170731, wordCount.getAllWords().get("ZZ"), Double.MIN_NORMAL);
        assertEquals(0.00975609756097561, wordCount.getAllWords().get("a"), Double.MIN_NORMAL);
        assertEquals(0.014634146341463415, wordCount.getAllWords().get("b"), Double.MIN_NORMAL);
        assertEquals(0.004878048780487805, wordCount.getAllWords().get("c"), Double.MIN_NORMAL);

        wordCount.remove("ZZ");
        assertEquals(0.3333333333333333, wordCount.getAllWords().get("a"), Double.MIN_NORMAL);
        assertEquals(0.5, wordCount.getAllWords().get("b"), Double.MIN_NORMAL);
        assertEquals(0.16666666666666666, wordCount.getAllWords().get("c"), Double.MIN_NORMAL);
    }
}
