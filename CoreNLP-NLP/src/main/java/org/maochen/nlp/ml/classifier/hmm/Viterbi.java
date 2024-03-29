package org.maochen.nlp.ml.classifier.hmm;

import org.apache.commons.lang3.StringUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;
import java.util.function.Predicate;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

/**
 * Created by Maochen on 8/5/15.
 */
public class Viterbi {
    private static final Logger LOG = LoggerFactory.getLogger(Viterbi.class);

    private static Predicate<String> isPunct = str -> Pattern.compile("\\p{Punct}+").matcher(str).find();

    public static List<String> resolve(HMMModel model, String[] words) {
        Set<String> tagSet = new HashSet<>();

        // Get all possible tags. Don't worry about OOV for this step.
        for (int i = 0; i < words.length; i++) {
            String word = words[i];
            tagSet.addAll(model.emission.row(word).keySet());
        }

        List<String> tags = new ArrayList<>(tagSet);
        tags.add(0, HMM.START);
        tags.add(HMM.END);
        // ----------

        // row - tags | col - words
        //           <START> fish sleep <END>
        // <START>
        // VB
        // NN
        // <END>
        List<String> rowString = tags;
        List<String> colString = Arrays.stream(words).collect(Collectors.toList());
        colString.add(0, HMM.START);
        colString.add(HMM.END);

        double[][] matrix = new double[rowString.size()][colString.size()];
        int[] path = new int[colString.size()]; // store row index, position is the column index. no need to restore both.

        matrix[0][0] = 1;
        for (int col = 1; col < matrix[0].length; col++) {
            String word = colString.get(col);

            for (int row = 1; row < matrix.length; row++) {
                String currentTag = rowString.get(row);

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
                            if (isPunct.test(currentTag) && !isPunct.test(word)) {
                                emission = 0D; // eliminate punct tag with non-punct word.
                            } else {
                                emission = model.emissionMin.get(currentTag); // OOV
                            }
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
                    stringBuilder.append(matrix[i][j]).append("\t");
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
