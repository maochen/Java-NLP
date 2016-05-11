package org.maochen.nlp.app.ner;

import org.maochen.nlp.app.featextractor.BrownFeatExtractor;
import org.maochen.nlp.app.featextractor.IFeatureExtractor;
import org.maochen.nlp.ml.SequenceTuple;
import org.maochen.nlp.ml.Tuple;
import org.maochen.nlp.ml.vector.FeatNamedVector;

import java.util.ArrayList;
import java.util.List;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * Created by Maochen on 2/22/16.
 */
public class NERFeatureExtractor implements IFeatureExtractor {

    private static String getWordShape(final String str) {
        StringBuilder stringBuilder = new StringBuilder();
        for (int i = 0; i < str.length(); i++) {
            if (Character.isUpperCase(str.charAt(i))) {
                stringBuilder.append("X");
            } else {
                stringBuilder.append("x");
            }
        }
        return stringBuilder.toString().trim();
    }

    public List<String> extractFeatSingle(int i, final String[] tokens) {

        List<String> currentFeats = new ArrayList<>();

        for (int index = Math.max(0, i - 2); index < Math.min(i + 3, tokens.length); index++) { // [-2,2]
            IFeatureExtractor.addFeat(currentFeats, "w" + (index - i), tokens[index]);
            IFeatureExtractor.addFeat(currentFeats, "word_length", String.valueOf(tokens[index].length()));
            IFeatureExtractor.addFeat(currentFeats, "word_shape", getWordShape(tokens[index]));

            boolean containsDigit = Pattern.compile("\\d+").matcher(tokens[index]).find();
            boolean containsTwoDigit = Pattern.compile("\\d{2}").matcher(tokens[index]).find();
            boolean containsFourDigit = Pattern.compile("\\d{4}").matcher(tokens[index]).find();

            boolean containsChar = Pattern.compile("[%|,|.|/|-]").matcher(tokens[index]).find();
            boolean containsDigitCharacter = containsChar && containsDigit;

            if (containsChar) {
                IFeatureExtractor.addFeat(currentFeats, "contains_char");
            }

            if (containsDigit) {
                IFeatureExtractor.addFeat(currentFeats, "contains_digit");
            }
            if (containsTwoDigit) {
                IFeatureExtractor.addFeat(currentFeats, "contains_two_digit");
            }

            if (containsFourDigit) {
                IFeatureExtractor.addFeat(currentFeats, "contains_four_digit");
            }
            if (containsDigitCharacter) {
                IFeatureExtractor.addFeat(currentFeats, "contains_digit_char");
            }

            if (index == i - 1) {
                IFeatureExtractor.addFeat(currentFeats, "w-10", tokens[i - 1], tokens[i]);
            } else if (index == i + 1) {
                IFeatureExtractor.addFeat(currentFeats, "w0+1", tokens[i], tokens[i + 1]);
            }
        }

        currentFeats.addAll(BrownFeatExtractor.extractBrownFeat(i, -2, 2, tokens));
        return currentFeats;
    }


    @Override
    public List<Tuple> extractFeat(final SequenceTuple entry) {
        String[] tokens = entry.entries.stream().map(tuple -> ((FeatNamedVector) tuple.vector).featsName[0]).toArray(String[]::new);

        List<List<String>> feats = IntStream.range(0, tokens.length)
                .mapToObj(i -> extractFeatSingle(i, tokens))
                .collect(Collectors.toList());

        List<Tuple> tuples = new ArrayList<>();

        for (int i = 0; i < feats.size(); i++) {
            List<String> singleTokenFeat = feats.get(i);
            FeatNamedVector v = new FeatNamedVector(singleTokenFeat.stream().toArray(String[]::new));

            Tuple t = new Tuple(v);
            t.label = entry.entries.get(i).label;

            tuples.add(t);
        }

        return tuples;
    }
}
