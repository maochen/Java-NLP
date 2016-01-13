package org.maochen.nlp.app.sentencetype;

import com.google.common.collect.ImmutableSet;

import org.apache.commons.lang3.StringUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.Set;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

/**
 * Created by Maochen on 8/5/14.
 */
public class SentenceTypeFeatureExtractor {

    private static final Logger LOG = LoggerFactory.getLogger(SentenceTypeFeatureExtractor.class);

    private static final Set<String> IMPERATIVE_KEYWORDS = ImmutableSet.of("verify", "ask", "say", "solve", "run", "execute");
    private static final Set<String> QUESTION_PREFIX = ImmutableSet.of("let me know", "clarify for me", "name");

    // Bossssssss.... currently all binary features.
    public List<String> generateFeats(String[] tokens) {
        tokens = Arrays.stream(tokens).filter(Objects::nonNull).filter(x -> !x.trim().isEmpty())
                .map(String::toLowerCase).toArray(String[]::new);

        if (tokens.length == 0) {
            return new ArrayList<>();
        }

        List<String> feats = new ArrayList<>();
        feats.add("word_count_" + String.valueOf(tokens.length));

        if (tokens[0].toLowerCase().startsWith("wh")) {
            feats.add("wh_start");
        }

        // Verify, Ask, Say - imperative
        boolean isImperativeStart = IMPERATIVE_KEYWORDS.contains(tokens[0]);
        String lastToken = tokens[tokens.length - 1];
        if (isImperativeStart) {
            feats.add("imperative_start");
            if (!Pattern.matches("\\p{Punct}+", lastToken)) {
                String[] newTokens = new String[tokens.length + 1];
                System.arraycopy(tokens, 0, newTokens, 0, tokens.length);
                tokens = newTokens;
            }
            tokens[tokens.length - 1] = "!";
        }

        // Start with question words.
        String sentence = Arrays.stream(tokens).collect(Collectors.joining(StringUtils.SPACE));
        long questionPrefixCount = QUESTION_PREFIX.stream().filter(sentence::startsWith).count();
        if (questionPrefixCount > 0) {
            feats.add("question_prefix");
            if (!Pattern.matches("\\p{Punct}+", lastToken)) {
                String[] newTokens = new String[tokens.length + 1];
                System.arraycopy(tokens, 0, newTokens, 0, tokens.length);
                tokens = newTokens;
            }
            tokens[tokens.length - 1] = "?";
        }

        feats.add("first_word_" + tokens[0]);

        if (tokens.length > 1) {
            feats.add("sec_word_" + tokens[1]);
        }

        lastToken = tokens[tokens.length - 1];
        if (Pattern.matches("\\p{Punct}+", lastToken)) {
            feats.add("punct_" + lastToken);
        }

        // whether
        long whetherKeyWord = Arrays.stream(tokens).filter("whether"::equals).count();
        if (whetherKeyWord > 0) {
            feats.add("has_whether");
        }

        List<String> biWord = new ArrayList<>();
        List<String> triWord = new ArrayList<>();

        for (int i = 0; i < tokens.length && i < 6; i++) { // 6 words maximum
            // Bigram
            if (i + 1 < tokens.length) {
                biWord.add(tokens[i] + "_" + tokens[i + 1]);
            }

            // Trigram
            if (i + 2 < tokens.length) {
                triWord.add(tokens[i] + "_" + tokens[i + 1] + "_" + tokens[i + 2]);
            }
        }

        feats.addAll(biWord);
        feats.addAll(triWord);

//        LOG.debug("feats: " + feats);
        return feats;
    }

}
