package org.maochen.nlp.ml.classifier.hmm;

import org.apache.commons.lang3.StringUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Objects;
import java.util.Set;
import java.util.stream.Collectors;

/**
 * Created by Maochen on 8/5/15.
 */
public class Viterbi {
    private static final Logger LOG = LoggerFactory.getLogger(Viterbi.class);

    public static List<String> resolve(HMMModel model, List<String> words) {
        Set<String> tagSet = words.stream().map(x -> model.emission.row(x).keySet()).reduce((s1, s2) -> {
            Set<String> s = new HashSet<>();
            s.addAll(s1);
            s.addAll(s2);
            return s;
        }).orElse(null);

        List<String> tags = new ArrayList<>(tagSet);
        tags.add(0, HMM.START);
        tags.add(HMM.END);

        List<String> rowString = tags;
        List<String> colString = new ArrayList<>(words);
        colString.add(0, HMM.START);
        colString.add(HMM.END);

        double[][] matrix = new double[rowString.size()][colString.size()];
        int[] path = new int[colString.size()]; // store row index, position is the column index. no need to restore both.

        matrix[0][0] = 1;
        for (int col = 1; col < matrix[0].length; col++) {
            for (int row = 1; row < matrix.length; row++) {

                String currentTag = rowString.get(row);
                String word = colString.get(col);

                for (int prevRow = 0; prevRow < matrix.length; prevRow++) {
                    if (matrix[prevRow][col - 1] == 0) {
                        continue;
                    }

                    String prevTag = rowString.get(prevRow);

                    Double trans = model.transition.get(prevTag, currentTag);
                    trans = trans == null ? 0 : trans;

                    Double emission = model.emission.get(word, currentTag);
                    if (emission == null) {
                        if (!model.emission.rowKeySet().contains(word)) {
                            LOG.debug("Missing word: " + word);
                            emission = model.emissionMin.get(currentTag); //OOV
                        }

                        if (emission == null) { // Has word, just word wont get that tag.
                            emission = 0D;
                        }
                    }

                    double prevViterbi = matrix[prevRow][col - 1];
                    double newViterbi = prevViterbi * trans * emission;

                    if (newViterbi > matrix[row][col]) {
                        matrix[row][col] = newViterbi;
                        path[col - 1] = prevRow;
                    }
                }
            }
        }

        if (LOG.isDebugEnabled()) {
            LOG.debug("Path: " + Arrays.stream(path).filter(Objects::nonNull).mapToObj(String::valueOf).reduce((x1, x2) -> x1 + StringUtils.SPACE + x2).orElse(null));
            StringBuilder stringBuilder = new StringBuilder();
            stringBuilder.append("\t\t").append(colString.stream().reduce((x1, x2) -> x1 + "\t" + x2).orElse(null)).append(System.lineSeparator());
            for (int i = 0; i < matrix.length; i++) {
                for (int j = 0; j < matrix[i].length; j++) {
                    if (j == 0) {
                        stringBuilder.append(rowString.get(i)).append("\t");
                    }
                    stringBuilder.append(String.format("%.4f", matrix[i][j])).append("\t");
                }
                stringBuilder.append(System.lineSeparator());
            }

            Arrays.stream(stringBuilder.toString().split(System.lineSeparator())).forEach(LOG::debug);
        }

        List<String> result = Arrays.stream(path)
                .filter(Objects::nonNull)
                .filter(rowIndex -> rowIndex != 0)
                .mapToObj(rowString::get)
                .collect(Collectors.toList());
        LOG.debug(result.toString());
        return result;
    }
}
