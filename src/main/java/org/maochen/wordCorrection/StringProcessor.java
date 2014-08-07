package org.maochen.wordCorrection;

public class StringProcessor {
    public String[] tokenize(String str) {
        str = str.toLowerCase();

        str = str.replaceAll("[^a-zA-Z\\s']", " ");
        str = str.replaceAll("\\s'|'\\s|^'|'$", " ");

        str = str.trim();
        str = str.replaceAll("\\s+", " ");
        str = str.replaceAll("'+", "'");

        String[] token = str.split("\\s");

        return token;
    }
}
