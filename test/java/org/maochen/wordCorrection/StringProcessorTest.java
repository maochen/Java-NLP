package org.maochen.wordcorrection;

import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.assertEquals;

public class StringProcessorTest {
    StringProcessor sp;

    @Before
    public void setUp() throws Exception {
        sp = new StringProcessor();
    }

    private void tokenizeEval(String str, String expected) {
        String[] token = sp.tokenize(str);
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < token.length; i++) {
            if (i != 0) sb.append(" ");
            sb.append(token[i]);
        }

        String actual = sb.toString();
        if (!expected.equals(actual)) {
            System.out.println(actual);
        }
        assertEquals(expected, actual);
    }

    @Test
    public void tokenizeTest() {
        String str = " http://wWw.maoChen....org!   ";
        tokenizeEval(str, "http www maochen org");
    }

    @Test
    public void tokenizeTestAppS() {
        String str = "Tom's";
        tokenizeEval(str, "tom's");
        str = "'remember";
        tokenizeEval(str, "remember");
        str = "Pink 'un' protruding";
        tokenizeEval(str, "pink un protruding");

    }

    @Test
    public void tokenizeTestAppS2() {
        String str = "file('the_adventures_of_sherlock_holmes.txt').read()";
        tokenizeEval(str, "file the adventures of sherlock holmes txt read");
    }
}
