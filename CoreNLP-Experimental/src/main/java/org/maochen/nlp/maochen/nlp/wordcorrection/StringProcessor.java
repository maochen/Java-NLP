package org.maochen.nlp.maochen.nlp.wordcorrection;

import org.apache.commons.lang3.StringUtils;

public class StringProcessor {
    public String[] tokenize(String str) {
        str = str.toLowerCase();

        str = str.replaceAll("[^a-zA-Z\\s']", StringUtils.SPACE);
        str = str.replaceAll("\\s'|'\\s|^'|'$", StringUtils.SPACE);

        str = str.trim();
        str = str.replaceAll("\\s+", StringUtils.SPACE);
        str = str.replaceAll("'+", "'");

        String[] token = str.split("\\s");

        return token;
    }
}
